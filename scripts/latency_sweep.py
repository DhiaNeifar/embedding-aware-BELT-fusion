#!/usr/bin/env python3
"""Late fusion under unknown delay (Table T6): pose and latency from boxes and sender velocities.

Per (delay, sigma, seed), from scripts/build_async_cache.py --compensate caches (raw and compensated). The plug-in
is never told the delay. Rows (``*_frame`` = the one-frame method, alignformer_c; the others = the full method over
time, alignformer_tk; which one to report under delay is decided on V2X-Real validation):

* ``bare`` (reported pose, stale boxes), ``oracle`` (true pose, stale boxes);
* ``freealign``: the baseline, latency ignored; ``freealign_clock``: with its authors' clock search;
* ``ignored``: AlignFormer with the latency ignored;
* ``latency``: AlignFormer on the boxes moved by the estimated latency (latency.refine_latency: continuous joint pose
  and latency fit, accepted only if significant against "no delay", pooled per CAV);
* ``latency_scores``: plus staleness-aware scores (fusion.staleness_confidence);
* ``told``: AlignFormer with the boxes moved by the TRUE delay, plus staleness scores (upper bound, not a method).

Also reports the latency estimate's error.
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

from embedding_aware_belt_fusion.alignformer_ext import async_cache, evaluate, freealign_clock, fusion, latency, variants  # noqa: E402

_W = {}
METHODS = ["ignored", "latency", "latency_scores", "told"]
ROWS = ["bare", "oracle", "freealign", "freealign_clock"] + [m + suffix for m in METHODS for suffix in ("_frame", "")]


def run(condition):
    torch.set_num_threads(1)
    delay, sigma, seed = condition
    records, nms, model, robust = _W["records"], _W["nms"], _W["model"], _W["robust"]
    load = lambda comp: dict(np.load(async_cache.condition_path(_W["async_dir"], sigma, seed, delay, comp)))
    raw, known = load(False), load(True)
    score = lambda cache, decided=None, **kw: evaluate.ap_table(base, evaluate.score_condition(base, records, cache, decided, nms, **kw))
    one_frame, over_time = variants.alignformer_c, variants.alignformer_tk
    ap = {"bare": score(raw), "oracle": score(raw, oracle=True), "freealign": score(raw, freealign=True)}
    clock, _ = freealign_clock.freealign_with_clock_search(base, records, raw, _W["freealign"], _W["scenarios"])
    ap["freealign_clock"] = score(raw, clock)

    raw_inputs = variants.PairInputs(records, raw, model, robust, base=base)
    known_inputs = variants.PairInputs(records, known, model, robust, base=base)
    known_scale = fusion.staleness_confidence(known, _W["table"], np.full(len(known["frame"]), delay))
    grid = latency.latency_grid(records, raw, model, robust, base, _W["taus"])
    _, tau, _ = latency.refine_latency(grid, raw, _W["scenarios"])
    moved, inputs = latency.decide_at(records, raw, tau, model, robust, base)
    scale = fusion.staleness_confidence(moved, _W["table"], tau)
    for suffix, method in (("_frame", one_frame), ("", over_time)):
        decided = method(inputs)
        ap["ignored" + suffix] = score(raw, method(raw_inputs))
        ap["latency" + suffix] = score(moved, decided)
        ap["latency_scores" + suffix] = score(moved, decided, score_scale=scale)
        ap["told" + suffix] = score(known, method(known_inputs), score_scale=known_scale)
    c = tau[grid[_W["taus"][0]]["inputs"].attempted]
    accuracy = {"exact": float(np.mean(np.abs(c - delay) < 0.5)), "within_1": float(np.mean(np.abs(c - delay) <= 1)),
                "mae_ms": float(np.mean(np.abs(c - delay)) * 100.0)}
    return delay, sigma, seed, ap, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--async-dir", type=Path, default=Path("outputs/async_opv2v"))
    parser.add_argument("--frame-table", type=Path, default=Path("outputs/opv2v_frame_table.pt"))
    parser.add_argument("--geometry-cache", type=Path, default=Path("outputs/r140_geometry_test_cache"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/opv2v/alignformer_boxes_only.pth"))
    parser.add_argument("--freealign-calibration", type=Path, default=Path("models/opv2v/freealign_calibration.json"))
    parser.add_argument("--opencood-config", type=Path, default=Path("configs/pointpillar_late_fusion_r140.yaml"))
    parser.add_argument("--eval-range", type=float, nargs=2, default=[140.0, 40.0], metavar=("X", "Y"))
    parser.add_argument("--taus", type=int, nargs="+", default=list(range(7)), help="candidate latencies (frames)")
    parser.add_argument("--odometry", type=Path, default=None,
                        help="real LiDAR odometry table (scripts/build_odometry_table.py) for the pooled rows")
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base.fusion_core.set_filters(args.eval_range[0], args.eval_range[1], -3.0, 1.0, 6.0)

    conditions = []
    for path in sorted(args.async_dir.glob("async_d*_sigma_*_seed_*.npz")):
        if "_comp_" in path.name:
            continue
        parts = path.stem.split("_")
        delay, sigma, seed = int(parts[1][1:]), float(parts[parts.index("sigma") + 1]), int(parts[parts.index("seed") + 1])
        if args.max_seeds is not None and seed >= args.max_seeds:
            continue
        if async_cache.condition_path(args.async_dir, sigma, seed, delay, True).exists():
            conditions.append((delay, sigma, seed))
    if not conditions:
        raise FileNotFoundError(f"no raw + compensated condition pairs in {args.async_dir}")
    records = check_verify_core.load_frame_table(args.frame_table)
    scenarios = async_cache.scenario_ids(args.geometry_cache)
    _W.update(
        records=records, scenarios=scenarios, async_dir=args.async_dir, taus=args.taus,
        table=fusion.prediction_error_table(base, records, scenarios, range(max(args.taus) + 1)),
        model=base.load_alignformer_pva(args.checkpoint, torch.device("cpu"))[0],
        freealign=base.freealign_config_from_calibration(args.freealign_calibration),
        robust=base.RobustSolveConfig(mode="huber", iterations=2),
        nms=float(base.load_yaml(str(args.opencood_config), None)["postprocess"]["nms_thresh"]),
    )
    if args.odometry is not None:
        from embedding_aware_belt_fusion.alignformer_ext import odometry

        odometry.ACTIVE = odometry.OdometryTable.for_records(args.odometry, args.geometry_cache, records)
    results = {}
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
        futures = [pool.submit(run, c) for c in conditions]
        for future in tqdm(cf.as_completed(futures), total=len(futures), desc="latency sweep (conditions)"):
            delay, sigma, seed, ap, accuracy = future.result()
            results.setdefault(f"d{delay}", {}).setdefault(f"{sigma:g}", []).append({"seed": seed, "ap": ap, "latency": accuracy})
    summary = {}
    for key, by_sigma in results.items():
        for sigma, per_seed in by_sigma.items():
            per_seed.sort(key=lambda e: e["seed"])
            for bucket in evaluate.BUCKETS:
                for row in ROWS:
                    for metric in ("ap_30", "ap_50", "ap_70"):
                        v = np.array([e["ap"][row][bucket][metric] for e in per_seed])
                        summary.setdefault(key, {}).setdefault(sigma, {}).setdefault(bucket, {}).setdefault(row, {})[metric] = {
                            "mean": float(v.mean()), "std": float(v.std()), "n": len(v)}
            for k in ("exact", "within_1", "mae_ms"):
                summary[key][sigma].setdefault("latency", {})[k] = float(np.mean([e["latency"][k] for e in per_seed]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "per_seed": results}, indent=1))
    print(f"Wrote {args.output}")
    for key in sorted(summary, key=lambda k: int(k[1:])):
        print(f"\n{key} ({int(key[1:]) * 100} ms): AP@0.7, all frames")
        print("sigma " + "".join(f"{r[:15]:>16}" for r in ROWS) + "   latency error (ms)")
        for sigma in sorted(summary[key], key=float):
            s = summary[key][sigma]
            lat = s["latency"]
            print(f"{sigma:<6}" + "".join(f"{s['all'][r]['ap_70']['mean']:>16.4f}" for r in ROWS)
                  + f"   {lat['mae_ms']:.0f}")


if __name__ == "__main__":
    main()
