#!/usr/bin/env python3
"""Write each plug-in's corrected CAV-to-ego transforms to files, for the intermediate-fusion hosts to read.

One file per (dataset, plug-in, sigma, seed) under ``outputs/poses/`` (format: alignformer_ext/poses.py). Reads the
same Propose caches as late fusion, so every host sees the same noise draws and the same corrections. Plug-ins here:
``alignformer`` (the full method, alignformer_tk, with the dataset's odometry) and ``freealign`` (the baseline).
Every file is checked: its transforms must reproduce the corrected boxes late fusion scored, to 1 mm.

    python scripts/export_poses.py --dataset opv2v
    python scripts/export_poses.py --dataset v2xreal
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

from embedding_aware_belt_fusion.alignformer_ext import odometry, poses, propose_cache, variants  # noqa: E402

DATASETS = {
    "opv2v": dict(cache="outputs/propose_cache_opv2v", table="outputs/opv2v_frame_table.pt",
                  geometry="outputs/r140_geometry_test_cache", odometry="outputs/odometry/opv2v_test.npz",
                  checkpoint="models/opv2v/alignformer_boxes_only.pth"),
    "v2xreal": dict(cache="outputs/propose_cache_v2xreal", table="outputs/v2xreal_test_frame_table.pt",
                    geometry="outputs/v2xreal_geometry_test_cache", odometry="outputs/odometry/v2xreal_test.npz",
                    checkpoint="models/v2xreal/alignformer_boxes_only.pth"),
}
TOLERANCE_M = 1e-3
_W = {}


def run(job):
    torch.set_num_threads(1)
    plugin, sigma, seed = job
    path = poses.condition_path(_W["output"], _W["dataset"], plugin, sigma, seed)
    if path.exists() and not _W["overwrite"]:
        return job, None
    with np.load(propose_cache.condition_path(_W["cache"], sigma, seed)) as data:
        cache = {k: data[k] for k in data.files}
    records = _W["records"]
    if plugin == "alignformer":
        decided = variants.alignformer_tk(variants.PairInputs(records, cache, _W["model"], _W["robust"], base=base))
    else:
        decided = poses.freealign_corrections(cache)
    arrays = poses.build(base, records, _W["frames"], cache, decided)
    worst = poses.check_against_cache(base, records, cache, decided, arrays)
    if worst > TOLERANCE_M:
        raise RuntimeError(f"{plugin} sigma {sigma:g} seed {seed}: exported transforms miss the scored boxes by {worst:.4f} m")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.rename(path)
    return job, worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--plugins", nargs="+", default=["alignformer", "freealign"], choices=["alignformer", "freealign"])
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("outputs/poses"))
    parser.add_argument("--overwrite", action="store_true", help="rewrite files that already exist")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    records = check_verify_core.load_frame_table(Path(cfg["table"]))
    odometry.ACTIVE = odometry.OdometryTable.for_records(Path(cfg["odometry"]), Path(cfg["geometry"]), records)
    _W.update(
        dataset=args.dataset, output=args.output, overwrite=args.overwrite, cache=Path(cfg["cache"]), records=records,
        frames=odometry.ACTIVE.frames,
        model=base.load_alignformer_pva(Path(cfg["checkpoint"]), torch.device("cpu"))[0],
        robust=base.RobustSolveConfig(mode="huber", iterations=2),
    )
    jobs = [(plugin, sigma, seed) for plugin in args.plugins for sigma in args.sigmas
            for seed in ([0] if sigma == 0.0 else range(args.seeds))]
    worst = 0.0
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in tqdm(cf.as_completed(futures), total=len(futures), desc=f"export poses {args.dataset}"):
            _, w = future.result()
            worst = max(worst, w or 0.0)
    print(f"{len(jobs)} files in {args.output / args.dataset}; largest mismatch with late fusion's boxes {worst * 1000:.3f} mm")


if __name__ == "__main__":
    main()
