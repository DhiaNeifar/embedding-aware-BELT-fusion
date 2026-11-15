#!/usr/bin/env python3
"""Cache the asynchronous-fusion conditions (alignformer_ext.async_cache): every CAV message ``delay`` frames old.

One ``.npz`` per (delay, compensated, sigma, seed) under ``--output-dir``; existing files are skipped, so an
interrupted run resumes. ``--compensate`` also builds the variant where each stale box is moved forward by the
sender's own velocity estimate. Score with ``scripts/async_sweep.py``.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
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

from embedding_aware_belt_fusion.alignformer_ext import async_cache  # noqa: E402

_W = {}


def run(condition):
    torch.set_num_threads(1)
    sigma, seed, delay, compensated = condition
    path = async_cache.condition_path(_W["output_dir"], sigma, seed, delay, compensated)
    if path.exists():
        return condition
    cache = async_cache.build_async_condition(
        base, _W["records"], _W["model"], _W["robust"], sigma, seed, delay, _W["scenarios"],
        velocities=_W["velocities"], freealign_config=_W["freealign"], compensate=compensated,
    )
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **cache)
    tmp.rename(path)
    return condition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame-table", type=Path, default=Path("outputs/opv2v_frame_table.pt"))
    parser.add_argument("--geometry-cache", type=Path, default=Path("outputs/r140_geometry_test_cache"),
                        help="the geometry cache the frame table was built from (scenario ids)")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/opv2v/alignformer_boxes_only.pth"))
    parser.add_argument("--freealign-calibration", type=Path, default=Path("models/opv2v/freealign_calibration.json"))
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.4, 1.0, 2.0])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--delays", type=int, nargs="+", default=[1, 2, 4],
                        help="frames (0.1 s each); 1 / 2 / 4 = the CVPR draft's 100 / 200 / 400 ms")
    parser.add_argument("--compensate", action="store_true", help="also build the velocity-compensated conditions")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/async_opv2v"))
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--odometry", type=Path, required=True,
                        help="each sender's own odometry (scripts/build_odometry_table.py), for its velocity tracking")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    conditions = [
        (sigma, seed, delay, comp)
        for delay in args.delays for comp in ([False, True] if args.compensate else [False])
        for sigma in args.sigmas for seed in ([0] if sigma == 0.0 else range(args.seeds))
        if not async_cache.condition_path(args.output_dir, sigma, seed, delay, comp).exists()
    ]
    print(f"{len(conditions)} conditions to build in {args.output_dir}")
    if not conditions:
        return
    records = check_verify_core.load_frame_table(args.frame_table)
    scenarios = async_cache.scenario_ids(args.geometry_cache)
    if len(scenarios) != len(records):
        raise ValueError(f"{args.geometry_cache}: {len(scenarios)} frames, frame table has {len(records)}")
    from embedding_aware_belt_fusion.alignformer_ext import odometry

    odometry.ACTIVE = odometry.OdometryTable.for_records(args.odometry, args.geometry_cache, records)
    _W.update(
        records=records, scenarios=scenarios,
        velocities=async_cache.cav_velocities(base, records, scenarios),
        model=base.load_alignformer_pva(args.checkpoint, torch.device("cpu"))[0],
        robust=base.RobustSolveConfig(mode="huber", iterations=2),
        freealign=base.freealign_config_from_calibration(args.freealign_calibration),
        output_dir=args.output_dir,
    )
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
        futures = [pool.submit(run, c) for c in conditions]
        for future in tqdm(cf.as_completed(futures), total=len(futures), desc="async cache (conditions)"):
            future.result()


if __name__ == "__main__":
    main()
