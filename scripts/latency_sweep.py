#!/usr/bin/env python3
"""Idea A (alignformer_ext/latency.py): pose AND unknown latency from boxes + sender velocities, under delay.

Per (delay, sigma, seed), from scripts/build_async_cache.py --compensate caches (raw and compensated):

* ``uncorrected``, ``oracle`` (true pose, stale boxes), ``oracle_known`` (true pose, boxes moved by the TRUE delay);
* ``freealign``: the FreeAlign baseline exactly as in the CVPR draft (stale boxes, latency ignored);
* ``alignformer``: per-frame AlignFormer on the stale boxes, latency ignored (as in the draft);
* ``alignformer_known``: the same after moving the boxes by the TRUE delay (told the latency; reference);
* ``freealign_clock``: FreeAlign with its authors' clock-deviation search (freealign_clock.py);
* ``latency_frame`` / ``latency_pooled``: AlignFormer with the latency ESTIMATED per frame / pooled per CAV over time
  (grid search, idea A); ``*_conf``: plus staleness-aware scores (idea D);
* ``alignformer_tg``: the full method (temporal layer + GPS confidence), latency ignored; ``a2_full`` / ``a2_full_conf``:
  the full method on top of A2 (the temporal layer reads each message's own pose and odometry), without / with D;
* ``a2`` / ``a2_conf``: A2 = continuous joint pose + latency fit, accepted only if significant against "no delay",
  pooled per CAV weighted by how observable the latency was (latency.refine_latency), then AlignFormer on the boxes
  moved by it; plus D.

Also reports how often the estimated latency equals the true one.
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
# FreeAlign appears only as the baseline, exactly as used in the CVPR draft (no component of ours added to it)
ROWS = ["uncorrected", "oracle", "oracle_known", "freealign", "freealign_clock", "alignformer", "alignformer_known",
        "latency_frame", "latency_pooled", "alignformer_known_conf", "latency_pooled_conf", "a2", "a2_conf",
        "alignformer_tg", "a2_full", "a2_full_conf", "a2_full_fixed", "ours"]


def run(condition):
    torch.set_num_threads(1)
    delay, sigma, seed = condition
    records, nms, model, robust = _W["records"], _W["nms"], _W["model"], _W["robust"]
    load = lambda comp: dict(np.load(async_cache.condition_path(_W["async_dir"], sigma, seed, delay, comp)))
    raw, known = load(False), load(True)
    score = lambda cache, decided=None, **kw: evaluate.ap_table(base, evaluate.score_condition(base, records, cache, decided, nms, **kw))
    known_decided = variants.alignformer(variants.PairInputs(records, known, model, robust, base=base))
    ap = {
        "uncorrected": score(raw), "oracle": score(raw, oracle=True), "oracle_known": score(known, oracle=True),
        "freealign": score(raw, freealign=True),
        "alignformer": score(raw, variants.alignformer(variants.PairInputs(records, raw, model, robust, base=base))),
        "alignformer_known": score(known, known_decided),
        # idea D: each CAV box scored by P(its prediction is still within the IoU-0.7 tolerance)
        "alignformer_known_conf": score(known, known_decided, score_scale=fusion.staleness_confidence(
            known, _W["table"], np.full(len(known["frame"]), delay))),
    }
    clock, _ = freealign_clock.freealign_with_clock_search(base, records, raw, _W["freealign"], _W["scenarios"])
    ap["freealign_clock"] = score(raw, clock)
    grid = latency.latency_grid(records, raw, model, robust, base, _W["taus"])
    per_frame, pooled = latency.choose_latency(grid, raw, _W["scenarios"])
    accuracy = {}
    attempted = grid[_W["taus"][0]]["inputs"].attempted
    for name, choice in (("latency_frame", per_frame), ("latency_pooled", pooled)):
        cache, decided = latency.assemble(grid, raw, choice)
        ap[name] = score(cache, decided)
        if name == "latency_pooled":
            ap["latency_pooled_conf"] = score(cache, decided, score_scale=fusion.staleness_confidence(cache, _W["table"], choice))
        c = choice[attempted]
        accuracy[name] = {"exact": float(np.mean(c == delay)), "within_1": float(np.mean(np.abs(c - delay) <= 1)),
                          "mae_ms": float(np.mean(np.abs(c - delay)) * 100.0)}
    _, a2_tau, a2_error = latency.refine_latency(grid, raw, _W["scenarios"])
    a2_cache, a2_decided, a2_inputs = latency.decide_at(records, raw, a2_tau, model, robust, base)
    a2_scale = fusion.staleness_confidence(a2_cache, _W["table"], a2_tau)
    ap["a2"] = score(a2_cache, a2_decided)
    ap["a2_conf"] = score(a2_cache, a2_decided, score_scale=a2_scale)
    # the full method (temporal layer + GPS confidence): latency ignored, and on top of A2 (+ D)
    ap["alignformer_tg"] = score(raw, variants.alignformer_tg(variants.PairInputs(records, raw, model, robust, base=base)))
    full = variants.alignformer_tg(a2_inputs)
    ap["a2_full"] = score(a2_cache, full)
    ap["a2_full_conf"] = score(a2_cache, full, score_scale=a2_scale)
    # the fix: GPS confidence governs the pose; a detected delay's residual motion error is still absorbed
    fixed = latency.absorb_residual_motion(a2_inputs, full, a2_tau, a2_error, _W["table"])
    ap["a2_full_fixed"] = score(a2_cache, fixed, score_scale=a2_scale)
    # ours: whole-frame latency (grid, pooled per CAV) where the pooled test says a delay exists, else 0;
    # GPS confidence only for fresh messages (a delayed one carries motion error GPS cannot fix); D on top
    ours_tau = np.where(a2_tau > 0.0, pooled, 0).astype(np.float64)
    ours_cache, _, ours_inputs = latency.decide_at(records, raw, ours_tau, model, robust, base)
    fresh, delayed = variants.alignformer_tg(ours_inputs), variants.alignformer_tk(ours_inputs)
    ours = np.where((ours_tau > 0.0)[:, None], delayed, fresh)
    ap["ours"] = score(ours_cache, ours, score_scale=fusion.staleness_confidence(ours_cache, _W["table"], ours_tau))
    c = ours_tau[attempted]
    accuracy["ours"] = {"exact": float(np.mean(np.abs(c - delay) < 0.5)), "within_1": float(np.mean(np.abs(c - delay) <= 1)),
                        "mae_ms": float(np.mean(np.abs(c - delay)) * 100.0)}
    c = a2_tau[attempted]
    accuracy["a2"] = {"exact": float(np.mean(np.abs(c - delay) < 0.5)), "within_1": float(np.mean(np.abs(c - delay) <= 1)),
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
            for name in ("latency_frame", "latency_pooled", "a2", "ours"):
                for k in ("exact", "within_1", "mae_ms"):
                    summary[key][sigma].setdefault("latency", {}).setdefault(name, {})[k] = float(
                        np.mean([e["latency"][name][k] for e in per_seed]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "per_seed": results}, indent=1))
    print(f"Wrote {args.output}")
    for key in sorted(summary, key=lambda k: int(k[1:])):
        print(f"\n{key} ({int(key[1:]) * 100} ms): AP@0.7, all frames")
        print("sigma " + "".join(f"{r[:15]:>16}" for r in ROWS) + "   latency error ms (pooled grid | A2)")
        for sigma in sorted(summary[key], key=float):
            s = summary[key][sigma]
            lat = s["latency"]
            print(f"{sigma:<6}" + "".join(f"{s['all'][r]['ap_70']['mean']:>16.4f}" for r in ROWS)
                  + f"   {lat['latency_pooled']['mae_ms']:.0f} | {lat['a2']['mae_ms']:.0f}")


if __name__ == "__main__":
    main()
