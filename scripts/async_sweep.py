#!/usr/bin/env python3
"""Score asynchronous late fusion (every CAV message ``delay`` frames old) from scripts/build_async_cache.py.

Rows per (delay, compensated, sigma), mean / std over seeds, buckets all / multi:

* ``uncorrected``: stale boxes at the stale reported pose; ``oracle``: stale boxes at their TRUE pose (no motion);
* ``freealign`` / ``alignformer``: per-frame alignment of the stale boxes against the ego's CURRENT boxes;
* ``past_freealign`` / ``past_alignformer`` / ``past_alignformer_tg``: the same methods registering IN THE PAST,
  against the ego's own boxes of the message's timestamp (the synchronous pair of that frame), moved to now by the
  ego's own motion (async_cache.from_the_past).

With ``_comp`` caches every stale box was first moved forward by the sender's velocity (async_cache.cav_velocities).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_verify_core  # noqa: E402
import reproduce_opv2v_sweep as base  # noqa: E402

from embedding_aware_belt_fusion.alignformer_ext import async_cache, evaluate, propose_cache, temporal, variants  # noqa: E402

_W = {}
ROWS = ["uncorrected", "oracle", "freealign", "alignformer", "past_freealign", "past_alignformer", "past_alignformer_tg",
        "past_tg_check"]
AGREE_M = 1.0   # Verify's own agreement tolerance


def check_against_now(inputs, past: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Multi-step correction: start from the decision made in the past (moved to now) and CHECK it against the
    receiver's current boxes. Verify (the deployed C++ RANSAC with shrinking gates) is run from that estimate; its
    refined fit replaces it only when Verify engaged and the two agree within AGREE_M on every CAV box (the residual
    is then the prediction error common to the CAV's boxes); otherwise the past estimate stands. Where nothing was
    decided in the past (abstained there), the present per-frame decision is used."""
    from embedding_aware_belt_fusion.alignformer_ext import evaluate as ev, verify_cpp

    a = inputs.attempted
    start = past[a].astype(np.float32)
    _, refined, engaged = verify_cpp.verify(
        inputs.ego_rows, inputs.cav_rows, start, np.full(len(a), 1e6, np.float32), np.full(len(a), 3.0, np.float32),
        inputs.model.variance_model, inputs.model.heading_lambda,
    )[:3]
    out = present.copy()
    cache = inputs.cache
    for i, p in enumerate(a):
        if not np.any(past[p] != 0):
            continue
        out[p] = past[p]
        if not engaged[i]:
            continue
        boxes = torch.from_numpy(cache["cav_boxes"][cache["cav_off"][p]:cache["cav_off"][p + 1]])
        moved_past = ev.apply_correction(boxes, *map(float, past[p]))[:, :2]
        moved_ref = ev.apply_correction(boxes, *map(float, refined[i]))[:, :2]
        if float((moved_past - moved_ref).norm(dim=1).max()) <= AGREE_M:
            out[p] = refined[i]
    return out


def run(group):
    """All async conditions of one (sigma, seed): the synchronous decisions are computed once and shared."""
    torch.set_num_threads(1)
    sigma, seed, paths = group
    records, nms, model, robust = _W["records"], _W["nms"], _W["model"], _W["robust"]
    with np.load(propose_cache.condition_path(_W["sync_dir"], sigma, seed)) as data:
        sync = {k: data[k] for k in data.files}
    sync_inputs = variants.PairInputs(records, sync, model, robust, base=base)
    past = {
        "past_alignformer": variants.alignformer(sync_inputs),
        "past_alignformer_tg": variants.alignformer_tg(sync_inputs),
        "past_freealign": temporal._freealign_measurements(sync_inputs),
    }
    sync_index = {(int(sync["frame"][q]), int(sync["cav_rank"][q])): q for q in range(len(sync["frame"]))}
    out = []
    for path in paths:
        with np.load(path) as data:
            cache = {k: data[k] for k in data.files}
        ap = {
            "uncorrected": evaluate.ap_table(base, evaluate.score_condition(base, records, cache, None, nms)),
            "oracle": evaluate.ap_table(base, evaluate.score_condition(base, records, cache, None, nms, oracle=True)),
            "freealign": evaluate.ap_table(base, evaluate.score_condition(base, records, cache, None, nms, freealign=True)),
        }
        inputs = variants.PairInputs(records, cache, model, robust, base=base)
        present = variants.alignformer(inputs)
        ap["alignformer"] = evaluate.ap_table(base, evaluate.score_condition(base, records, cache, present, nms))
        moved = {}
        for name, decided in past.items():
            moved[name] = async_cache.from_the_past(decided, sync_index, cache)
            ap[name] = evaluate.ap_table(base, evaluate.score_condition(base, records, cache, moved[name], nms))
        checked = check_against_now(inputs, moved["past_alignformer_tg"], present)
        ap["past_tg_check"] = evaluate.ap_table(base, evaluate.score_condition(base, records, cache, checked, nms))
        key = f"d{int(cache['delay'])}{'_comp' if bool(cache['compensated']) else ''}"
        out.append((key, float(sigma), int(seed), ap))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--async-dir", type=Path, default=Path("outputs/async_opv2v"))
    parser.add_argument("--sync-dir", type=Path, default=Path("outputs/propose_cache_opv2v"))
    parser.add_argument("--frame-table", type=Path, default=Path("outputs/opv2v_frame_table.pt"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/opv2v/alignformer_boxes_only.pth"))
    parser.add_argument("--opencood-config", type=Path, default=Path("configs/pointpillar_late_fusion_r140.yaml"))
    parser.add_argument("--eval-range", type=float, nargs=2, default=[140.0, 40.0], metavar=("X", "Y"))
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base.fusion_core.set_filters(args.eval_range[0], args.eval_range[1], -3.0, 1.0, 6.0)

    groups = {}
    for path in sorted(args.async_dir.glob("async_d*_sigma_*_seed_*.npz")):
        parts = path.stem.split("_")
        sigma, seed = float(parts[parts.index("sigma") + 1]), int(parts[parts.index("seed") + 1])
        if args.max_seeds is not None and seed >= args.max_seeds:
            continue
        groups.setdefault((sigma, seed), []).append(path)
    if not groups:
        raise FileNotFoundError(f"no async conditions in {args.async_dir}; run scripts/build_async_cache.py")
    _W.update(
        records=check_verify_core.load_frame_table(args.frame_table),
        model=base.load_alignformer_pva(args.checkpoint, torch.device("cpu"))[0],
        robust=base.RobustSolveConfig(mode="huber", iterations=2),
        nms=float(base.load_yaml(str(args.opencood_config), None)["postprocess"]["nms_thresh"]),
        sync_dir=args.sync_dir,
    )
    results = {}
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
        futures = [pool.submit(run, (sigma, seed, paths)) for (sigma, seed), paths in groups.items()]
        for future in tqdm(cf.as_completed(futures), total=len(futures), desc="async sweep (sigma, seed)"):
            for key, sigma, seed, ap in future.result():
                results.setdefault(key, {}).setdefault(f"{sigma:g}", []).append({"seed": seed, "ap": ap})
    summary = {}
    for key, by_sigma in results.items():
        for sigma, per_seed in by_sigma.items():
            per_seed.sort(key=lambda e: e["seed"])
            for bucket in evaluate.BUCKETS:
                for row in ROWS:
                    for metric in ("ap_30", "ap_50", "ap_70"):
                        values = np.array([e["ap"][row][bucket][metric] for e in per_seed])
                        summary.setdefault(key, {}).setdefault(sigma, {}).setdefault(bucket, {}).setdefault(row, {})[metric] = {
                            "mean": float(values.mean()), "std": float(values.std()), "n": len(values)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "per_seed": results}, indent=1))
    print(f"Wrote {args.output}")
    for key in sorted(summary):
        print(f"\n{key}: AP@0.7 (all frames)")
        print("sigma " + "".join(f"{r[:18]:>20}" for r in ROWS))
        for sigma in sorted(summary[key], key=float):
            s = summary[key][sigma]["all"]
            print(f"{sigma:<6}" + "".join(f"{s[r]['ap_70']['mean']:>20.4f}" for r in ROWS))


if __name__ == "__main__":
    main()
