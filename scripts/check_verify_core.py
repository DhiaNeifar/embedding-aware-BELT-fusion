#!/usr/bin/env python3
"""Regression gate: C++ Verify (verify_core) must reproduce the deployed AlignFormer.

For one (sigma, seed) condition over the OPV2V frame table:
  1. Propose is run once per pair (alignformer_ext.propose_cache) and fed to the C++ Verify + Abstain + agreement.
  2. The original Python path (alignformer_pva.inference.estimate_correction, FULL_* configs) runs on the same pairs.
  3. Per-pair decided corrections are compared, then AP of both against each other and, on the full frame set,
     against the saved per-seed AP in --reference-json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reproduce_opv2v_sweep as base  # noqa: E402

from embedding_aware_belt_fusion.alignformer_ext import evaluate, propose_cache, verify_cpp  # noqa: E402
from embedding_aware_belt_fusion.alignformer_pva.inference import estimate_correction  # noqa: E402


def load_frame_table(path: Path):
    main_module = sys.modules["__main__"]
    if not hasattr(main_module, "FrameRecord"):
        main_module.FrameRecord = base.FrameRecord
    payload = torch.load(path, weights_only=False)
    return payload if isinstance(payload, list) else payload["records"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame-table", type=Path, default=Path("outputs/opv2v_frame_table.pt"))
    parser.add_argument("--opencood-config", type=Path, default=Path("configs/pointpillar_late_fusion_r140.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/opv2v/alignformer_boxes_only.pth"))
    parser.add_argument("--reference-json", type=Path, default=Path("outputs/sweep_full_20seeds_v2.json"))
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--min-cavs", type=int, default=None,
        help="Keep frames with at least this many CAVs, then index frames within that subset, exactly as "
             "reproduce_opv2v_sweep.py --min-cavs did (the sweep_cavs_ge3*.json runs).",
    )
    args = parser.parse_args()

    records = load_frame_table(args.frame_table)
    if args.min_cavs is not None:
        records = [r for r in records if len(r.cav_boxes) >= args.min_cavs]
    records = records[: args.max_frames]
    nms_thresh = float(base.load_yaml(str(args.opencood_config), None)["postprocess"]["nms_thresh"])
    model, _ = base.load_alignformer_pva(args.checkpoint, torch.device("cpu"))
    robust = base.RobustSolveConfig(mode="huber", iterations=2)

    cache = propose_cache.build_condition(base, records, model, robust, args.sigma, args.seed)
    attempted = np.flatnonzero(cache["attempted"])
    cav_ids_by_frame = [sorted(r.cav_boxes) for r in records]
    ego_rows, cav_rows = [], []
    for p in attempted:
        record = records[cache["frame"][p]]
        cav_id = cav_ids_by_frame[cache["frame"][p]][cache["cav_rank"][p]]
        ego_rows.append(verify_cpp.box_rows(record.ego_boxes, record.ego_scores, model.variance_model))
        cav = torch.from_numpy(cache["cav_boxes"][cache["cav_off"][p]:cache["cav_off"][p + 1]])
        cav_rows.append(verify_cpp.box_rows(cav, record.cav_scores[cav_id], model.variance_model))
    settings = verify_cpp.VerifySettings.from_configs(
        base.FULL_REFINE_CONFIG, base.FULL_ABSTAIN_CONFIG, base.FULL_AGREEMENT_CONFIG
    )
    decided_cpp = np.zeros((len(cache["frame"]), 3), np.float32)
    out = verify_cpp.verify(
        ego_rows, cav_rows, cache["propose"][attempted], cache["statistic"][attempted], cache["dof"][attempted],
        model.variance_model, model.heading_lambda, settings,
    )
    decided_cpp[attempted] = out[0]

    decided_py = np.zeros_like(decided_cpp)
    with torch.no_grad():
        for p in attempted:
            record = records[cache["frame"][p]]
            cav_id = cav_ids_by_frame[cache["frame"][p]][cache["cav_rank"][p]]
            cav = torch.from_numpy(cache["cav_boxes"][cache["cav_off"][p]:cache["cav_off"][p + 1]])
            correction, stats = estimate_correction(
                model, {"boxes": record.ego_boxes, "scores": record.ego_scores},
                {"boxes": cav, "scores": record.cav_scores[cav_id]},
                refine_config=base.FULL_REFINE_CONFIG, abstain_config=base.FULL_ABSTAIN_CONFIG,
                robust_config=robust, agreement_config=base.FULL_AGREEMENT_CONFIG,
            )
            if correction is not None and stats["accepted"]:
                decided_py[p] = (float(correction["yaw_correction"]), *map(float, correction["translation"]))

    accept_cpp, accept_py = np.any(decided_cpp != 0, 1), np.any(decided_py != 0, 1)
    gap = np.abs(decided_cpp - decided_py)
    off = (gap[:, 0] > 1e-4) | (gap[:, 1:].max(1) > 1e-3)
    print(f"sigma={args.sigma:g} seed={args.seed}: {len(records)} frames, {len(attempted)} attempted pairs")
    print(f"  accept decision differs: {int((accept_cpp != accept_py).sum())} pairs")
    print(f"  pose differs (>1e-4 rad or >1e-3 m): {int(off.sum())} pairs; max |dpsi|={gap[:, 0].max():.2e} rad, "
          f"max |dt|={gap[:, 1:].max():.2e} m")
    for p in np.flatnonzero(off)[:5]:
        print(f"    pair {p} frame {cache['frame'][p]}: cpp={decided_cpp[p]} py={decided_py[p]}")

    ap_cpp = evaluate.ap_table(base, evaluate.score_condition(base, records, cache, decided_cpp, nms_thresh))
    ap_py = evaluate.ap_table(base, evaluate.score_condition(base, records, cache, decided_py, nms_thresh))
    for bucket in evaluate.BUCKETS:
        print(f"  AP@0.7 {bucket:>4}: cpp={ap_cpp[bucket]['ap_70']:.6f} py={ap_py[bucket]['ap_70']:.6f}")
    if args.max_frames is None and args.reference_json.exists():
        reference = json.loads(args.reference_json.read_text())["sigma"][f"sigma_{args.sigma:g}"]["per_seed"]
        ref = next(s["ap"]["alignformer"] for s in reference if s["seed"] == args.seed)
        print("  saved sweep alignformer: " + " ".join(f"{k}={v:.6f}" for k, v in ref.items()))
        print("  cpp all               : " + " ".join(f"{k}={v:.6f}" for k, v in ap_cpp["all"].items()))


if __name__ == "__main__":
    main()
