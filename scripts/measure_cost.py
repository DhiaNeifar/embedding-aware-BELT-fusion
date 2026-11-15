#!/usr/bin/env python3
"""Cost of each AlignFormer stage (T7's cost paragraph): runtime on ONE CPU core and bytes added to a message.

On the first ``--frames`` frames of the test split at sigma = 1, seed 0 (the cached noisy boxes, so the inputs are
the paper's), single-threaded torch, best of ``--repeats`` runs:

* Propose: building the pair batch + the network's forward (Sinkhorn, weighted Kabsch, Wald statistic), per pair;
* Verify + Abstain: the one-frame method on the cached proposals (C++ exact fits, RANSAC, Wald, agreement);
* Wald test alone (part of the above);
* pooling over time: the full method minus the one-frame method on the same pairs;
* FreeAlign on the same pairs, for reference.

Bytes: what the plug-in needs from a CAV that a network's message may not carry: its detections (7 box values and a
score, float32 = 32 B each) and its reported pose (6 float64 = 48 B, sent by every cooperative system anyway).
In late fusion all of it is already in the message.

    python scripts/measure_cost.py --dataset v2xreal --output outputs/paper/cost_v2xreal.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_verify_core  # noqa: E402
import reproduce_opv2v_sweep as base  # noqa: E402

from embedding_aware_belt_fusion.alignformer_ext import odometry, propose_cache, variants, verify_cpp  # noqa: E402
from embedding_aware_belt_fusion.alignformer_pva.inference import build_pairwise_batch  # noqa: E402

DATASETS = {
    "opv2v": dict(table="outputs/opv2v_frame_table.pt", cache="outputs/propose_cache_opv2v",
                  checkpoint="models/opv2v/alignformer_boxes_only.pth", fa="models/opv2v/freealign_calibration.json",
                  odometry="outputs/odometry/opv2v_test.npz", geometry="outputs/r140_geometry_test_cache"),
    "v2xreal": dict(table="outputs/v2xreal_test_frame_table.pt", cache="outputs/propose_cache_v2xreal",
                    checkpoint="models/v2xreal/alignformer_boxes_only.pth",
                    fa="models/v2xreal/freealign_calibration.json",
                    odometry="outputs/odometry/v2xreal_test.npz", geometry="outputs/v2xreal_geometry_test_cache"),
}
BOX_BYTES = 8 * 4      # 7 box values + score, float32
POSE_BYTES = 6 * 8     # reported pose, float64


def prefix(cache: dict, frames: int) -> dict:
    """The cache's pairs of the first ``frames`` frames (pairs are stored in frame order)."""
    k = int(np.sum(cache["frame"] < frames))
    out = dict(cache)
    for key in ("frame", "cav_rank", "attempted", "propose", "statistic", "dof", "plan_shape"):
        out[key] = cache[key][:k]
    out["cav_off"] = cache["cav_off"][: k + 1]
    rows = int(cache["cav_off"][k])
    out["cav_boxes"] = cache["cav_boxes"][:rows]
    if "freealign_boxes" in cache:
        out["freealign_boxes"] = cache["freealign_boxes"][:rows]
    out["plan"] = cache["plan"][: int(np.sum(np.prod(cache["plan_shape"][:k], axis=1)))]
    return out


def best_of(repeats: int, fn) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = DATASETS[args.dataset]

    records = check_verify_core.load_frame_table(Path(cfg["table"]))
    odometry.ACTIVE = odometry.OdometryTable.for_records(Path(cfg["odometry"]), Path(cfg["geometry"]), records)
    model = base.load_alignformer_pva(Path(cfg["checkpoint"]), torch.device("cpu"))[0]
    robust = base.RobustSolveConfig(mode="huber", iterations=2)
    freealign = base.freealign_config_from_calibration(Path(cfg["fa"]))
    with np.load(propose_cache.condition_path(Path(cfg["cache"]), 1.0, 0)) as z:
        cache = prefix({k: z[k] for k in z.files}, args.frames)
    records = records[: args.frames]
    inputs = variants.PairInputs(records, cache, model, robust, base=base)
    pairs = [int(p) for p in inputs.attempted]
    frames_with_pairs = len({int(cache["frame"][p]) for p in pairs})

    def cav_of(p):
        record = records[int(cache["frame"][p])]
        return record, sorted(record.cav_boxes)[int(cache["cav_rank"][p])]

    def propose():
        with torch.no_grad():
            for p in tqdm(pairs, desc="Propose (network, 1 core)", leave=False):
                record, cav_id = cav_of(p)
                boxes = torch.from_numpy(cache["cav_boxes"][cache["cav_off"][p]:cache["cav_off"][p + 1]])
                batch = build_pairwise_batch({"boxes": record.ego_boxes, "scores": record.ego_scores},
                                             {"boxes": boxes, "scores": record.cav_scores[cav_id]})
                model(batch, robust=robust, statistic=True)

    def freealign_all():
        for p in tqdm(pairs, desc="FreeAlign (1 core)", leave=False):
            record, cav_id = cav_of(p)
            boxes = torch.from_numpy(cache["cav_boxes"][cache["cav_off"][p]:cache["cav_off"][p + 1]])
            simple = {"ego": {"boxes": record.ego_boxes, "scores": record.ego_scores},
                      cav_id: {"boxes": boxes, "scores": record.cav_scores[cav_id]}}
            base.freealign_correct(simple, ["ego", cav_id], freealign)

    a = inputs.attempted
    seconds = {}
    for name, fn in tqdm([
        ("propose", propose),
        ("verify_abstain", lambda: variants.alignformer_c(inputs)),
        ("wald", lambda: verify_cpp.wald_answered(cache["statistic"][a], cache["dof"][a], 0.2)),
        ("full_method", lambda: variants.alignformer_tk(inputs)),
        ("freealign", freealign_all),
    ], desc=f"cost {args.dataset}: stages x {args.repeats} repeats"):
        seconds[name] = best_of(args.repeats, fn)
    seconds["pooling"] = max(seconds["full_method"] - seconds["verify_abstain"], 0.0)

    detections = [len(r.cav_boxes[c]) for r in records for c in r.cav_boxes]
    summary = {
        "dataset": args.dataset, "frames": args.frames, "frames_with_pairs": frames_with_pairs, "pairs": len(pairs),
        "threads": torch.get_num_threads(),
        "ms_per_frame": {k: 1000 * v / max(frames_with_pairs, 1) for k, v in seconds.items()},
        "ms_per_pair": {k: 1000 * v / max(len(pairs), 1) for k, v in seconds.items()},
        "detections_per_cav_message": float(np.mean(detections)) if detections else 0.0,
        "bytes_per_cav_message": float(np.mean(detections)) * BOX_BYTES + POSE_BYTES if detections else POSE_BYTES,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=1))
    print(f"Wrote {args.output}")
    print("  ms per frame (1 core): " + "  ".join(f"{k} {v:.1f}" for k, v in summary["ms_per_frame"].items()))
    print(f"  plug-in input per CAV message: {summary['detections_per_cav_message']:.1f} detections, "
          f"{summary['bytes_per_cav_message']:.0f} bytes")


if __name__ == "__main__":
    main()
