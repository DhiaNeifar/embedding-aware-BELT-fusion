#!/usr/bin/env python3
"""Score AlignFormer variants in late fusion from the Propose cache (C++ Verify, NMS and AP; no network calls).

Rows: bare (the reported pose), oracle (the true pose), freealign (the baseline, cached on the same noise
draws), and every requested variant from ``alignformer_ext.variants`` (``alignformer`` = the deployed one-frame
method, always included as the reference). AP per CAV bucket
(all / lt3 / ge3), mean and std over seeds. Build the cache first with
``scripts/build_propose_cache.py``.
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
    os.environ.setdefault(_var, "1")   # one BLAS thread per worker process (set before numpy loads)

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_verify_core  # noqa: E402
import reproduce_opv2v_sweep as base  # noqa: E402

from embedding_aware_belt_fusion.alignformer_ext import evaluate, variants  # noqa: E402

_W = {}


def run(path: Path):
    torch.set_num_threads(1)
    with np.load(path) as data:
        cache = {k: data[k] for k in data.files}
    records, nms = _W["records"], _W["nms"]
    ap = {
        "bare": evaluate.ap_table(base, evaluate.score_condition(base, records, cache, None, nms)),
        "freealign": evaluate.ap_table(base, evaluate.score_condition(base, records, cache, None, nms, freealign=True)),
    }
    inputs = variants.PairInputs(records, cache, _W["model"], _W["robust"], base=base)
    for name in _W["variants"]:
        result = variants.VARIANTS[name](inputs)
        decided, scale = result if isinstance(result, tuple) else (result, None)   # (decided, CAV-box score scale)
        ap[name] = evaluate.ap_table(base, evaluate.score_condition(base, records, cache, decided, nms, score_scale=scale))
        if _W.get("pose_error"):
            ap.setdefault("pose_error", {})[name] = pose_error(records, cache, decided)
    if float(cache["sigma"]) == 0.0:
        ap["oracle"] = evaluate.ap_table(base, evaluate.score_condition(base, records, cache, None, nms, oracle=True))
    return float(cache["sigma"]), int(cache["seed"]), ap


def pose_error(records, cache, decided) -> dict:
    """Translation and yaw error of every CAV's corrected CAV-to-ego transform against the recorded poses (evaluation
    only: the truth is read here, after the estimator has decided). Same export as scripts/pose_accuracy.py reads."""
    from embedding_aware_belt_fusion.alignformer_ext import odometry, poses

    arrays = poses.build(base, records, odometry.ACTIVE.frames, cache, decided)
    t, yaw = [], []
    for f, cav, transform in zip(arrays["frame_index"], arrays["cav_id"], arrays["transform"]):
        record = records[int(f)]
        residual = np.linalg.inv(transform) @ base.x1_to_x2(record.cav_poses[str(cav)], record.ego_pose)
        t.append(float(np.hypot(residual[0, 3], residual[1, 3])))
        yaw.append(abs(float(np.degrees(np.arctan2(residual[1, 0], residual[0, 0])))))
    t, yaw = np.array(t), np.array(yaw)
    return {"pairs": int(len(t)), "median_m": float(np.median(t)), "p90_m": float(np.percentile(t, 90)),
            "median_deg": float(np.median(yaw)), "p90_deg": float(np.percentile(yaw, 90))}


def delta_label(variant: str) -> str:
    """'alignformer' -> 'alignformer-FA', 'alignformer_c' -> 'alignformerC-FA'."""
    head, _, suffix = variant.partition("_")
    return f"{head}{suffix.upper()}-FA"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/propose_cache_opv2v"))
    parser.add_argument("--frame-table", type=Path, default=Path("outputs/opv2v_frame_table.pt"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/opv2v/alignformer_boxes_only.pth"))
    parser.add_argument("--opencood-config", type=Path, default=Path("configs/pointpillar_late_fusion_r140.yaml"))
    parser.add_argument("--variants", nargs="+", default=["alignformer"], choices=sorted(variants.VARIANTS))
    parser.add_argument("--sigmas", type=float, nargs="*", default=None, help="subset of cached sigmas")
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument(
        "--eval-range", type=float, nargs=2, default=[140.0, 40.0], metavar=("X", "Y"),
        help="fusion_core's evaluation range |x|, |y| in metres: OPV2V 140 40 (default), V2X-Real 102.4 51.2.",
    )
    parser.add_argument(
        "--odometry", type=Path, default=None,
        help="real LiDAR odometry table (scripts/build_odometry_table.py) for the temporal layer; needs --geometry-cache",
    )
    parser.add_argument("--geometry-cache", type=Path, default=Path("outputs/r140_geometry_test_cache"))
    parser.add_argument("--odometry-error", choices=["none", "white", "ar1", "scale", "yawdrift"], default="none",
                        help="audit E3: inject this odometry error into the odometry the estimator is fed "
                             "(alignformer_ext.odometry.PerturbedOdometry); the estimator still assumes LOAM accuracy")
    parser.add_argument("--odometry-error-seed", type=int, default=0)
    parser.add_argument("--pose-error", action="store_true",
                        help="also report each variant's pose error against the recorded poses (needs --odometry)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base.fusion_core.set_filters(args.eval_range[0], args.eval_range[1], -3.0, 1.0, 6.0)

    names = ["alignformer"] + [v for v in args.variants if v != "alignformer"]
    paths = []
    for path in sorted(args.cache_dir.glob("sigma_*_seed_*.npz")):
        sigma, seed = float(path.stem.split("_")[1]), int(path.stem.split("_")[3])
        if args.sigmas is not None and sigma not in args.sigmas:
            continue
        if args.max_seeds is not None and seed >= args.max_seeds:
            continue
        paths.append(path)
    if not paths:
        raise FileNotFoundError(f"no cached conditions in {args.cache_dir}; run scripts/build_propose_cache.py")

    _W.update(
        records=check_verify_core.load_frame_table(args.frame_table),
        model=base.load_alignformer_pva(args.checkpoint, torch.device("cpu"))[0],
        robust=base.RobustSolveConfig(mode="huber", iterations=2),
        nms=float(base.load_yaml(str(args.opencood_config), None)["postprocess"]["nms_thresh"]),
        variants=names,
    )
    if args.odometry is not None:
        from embedding_aware_belt_fusion.alignformer_ext import odometry

        odometry.ACTIVE = odometry.OdometryTable.for_records(args.odometry, args.geometry_cache, _W["records"])
        if args.odometry_error != "none":
            odometry.ACTIVE = odometry.PerturbedOdometry(odometry.ACTIVE, args.odometry_error, args.odometry_error_seed)
    elif args.odometry_error != "none" or args.pose_error:
        raise SystemExit("--odometry-error and --pose-error need --odometry")
    _W["pose_error"] = args.pose_error
    results = {}
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
        futures = [pool.submit(run, path) for path in paths]
        # as each condition finishes (not in submission order), so the bar moves with the first finished condition
        for future in tqdm(cf.as_completed(futures), total=len(futures), desc="variants sweep (conditions)"):
            sigma, seed, ap = future.result()
            results.setdefault(f"{sigma:g}", []).append({"seed": seed, "ap": ap})
    for per_seed in results.values():
        per_seed.sort(key=lambda entry: entry["seed"])

    oracle = next(s["ap"]["oracle"] for s in results.get("0", [])) if "0" in results else None
    rows = ["bare", "freealign"] + names
    summary = {}
    for sigma, per_seed in results.items():
        for bucket in evaluate.BUCKETS:
            for row in rows:
                for key in ("ap_30", "ap_50", "ap_70"):
                    values = np.array([s["ap"][row][bucket][key] for s in per_seed])
                    summary.setdefault(sigma, {}).setdefault(bucket, {}).setdefault(row, {})[key] = {
                        "mean": float(values.mean()), "std": float(values.std()), "n": len(values)
                    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.pose_error:
        for sigma, per_seed in results.items():
            for row in names:
                for key in ("median_m", "p90_m", "median_deg", "p90_deg"):
                    values = np.array([s["ap"]["pose_error"][row][key] for s in per_seed])
                    summary[sigma].setdefault("pose_error", {}).setdefault(row, {})[key] = float(values.mean())
    setting = {"odometry": str(args.odometry), "odometry_error": args.odometry_error,
               "odometry_error_seed": args.odometry_error_seed}
    args.output.write_text(json.dumps({"variants": names, "oracle": oracle, "setting": setting, "summary": summary,
                                       "per_seed": results}, indent=1))
    print(f"Wrote {args.output}")

    for bucket in evaluate.BUCKETS:
        oracle_txt = f", oracle {oracle[bucket]['ap_70']:.4f}" if oracle else ""
        print(f"\nAP@0.7, CAVs {bucket}{oracle_txt}")
        print("sigma " + "".join(f"{r[:16]:>17}" for r in rows) + "".join(f" {delta_label(v):>17}" for v in names))
        for sigma in sorted(summary, key=float):
            s = summary[sigma][bucket]
            cells = "".join(f"{s[r]['ap_70']['mean']:>17.4f}" for r in rows)
            deltas = "".join(f" {s[v]['ap_70']['mean'] - s['freealign']['ap_70']['mean']:>+17.4f}" for v in names)
            print(f"{sigma:<6}{cells}{deltas}")


if __name__ == "__main__":
    main()
