#!/usr/bin/env python3
"""One-time cache of AlignFormer's Propose output (and FreeAlign's corrected boxes) for the OPV2V sweep.

One ``.npz`` per (sigma, seed) under ``--output-dir`` (see
``alignformer_ext.propose_cache``). Conditions already on disk are skipped,
so an interrupted run resumes. Afterwards every Verify/Abstain variant is
scored from this cache by ``scripts/alignformer_variants_sweep.py`` with no
network or FreeAlign calls.

Noise is ``reproduce_opv2v_sweep.py``'s current (independent per-CAV) draws.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_verify_core  # noqa: E402  (frame-table loader)
import reproduce_opv2v_sweep as base  # noqa: E402

from embedding_aware_belt_fusion.alignformer_ext import propose_cache  # noqa: E402

_W = {}


def run(condition):
    torch.set_num_threads(1)
    sigma, seed = condition
    path = propose_cache.condition_path(_W["output_dir"], sigma, seed)
    if not path.exists():
        cache = propose_cache.build_condition(
            base, _W["records"], _W["model"], _W["robust"], sigma, seed, freealign_config=_W["freealign"]
        )
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **cache)
        tmp.rename(path)
    return condition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame-table", type=Path, default=Path("outputs/opv2v_frame_table.pt"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/opv2v/alignformer_boxes_only.pth"))
    parser.add_argument("--freealign-calibration", type=Path, default=Path("models/opv2v/freealign_calibration.json"))
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/propose_cache_opv2v"))
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    conditions = [
        (sigma, seed) for sigma in args.sigmas for seed in ([0] if sigma == 0.0 else range(args.seeds))
        if not propose_cache.condition_path(args.output_dir, sigma, seed).exists()
    ]
    print(f"{len(conditions)} conditions to build in {args.output_dir}")
    if not conditions:
        return
    _W.update(
        records=check_verify_core.load_frame_table(args.frame_table),
        model=base.load_alignformer_pva(args.checkpoint, torch.device("cpu"))[0],
        robust=base.RobustSolveConfig(mode="huber", iterations=2),
        freealign=base.freealign_config_from_calibration(args.freealign_calibration),
        output_dir=args.output_dir,
    )
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
        futures = [pool.submit(run, c) for c in conditions]
        for future in tqdm(cf.as_completed(futures), total=len(futures), desc="propose cache (conditions)"):
            future.result()


if __name__ == "__main__":
    main()
