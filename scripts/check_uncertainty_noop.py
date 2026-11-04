#!/usr/bin/env python3
"""Idea 1 sanity gate: re-solving Propose from the cached Sinkhorn plan with the spread term OFF must
reproduce the cached Propose output (psi, t, Wald statistic, dof) -- i.e. the re-solve path is the
original Propose, and only the spread term differs in alignformer_u."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_verify_core  # noqa: E402
import reproduce_opv2v_sweep as base  # noqa: E402

from embedding_aware_belt_fusion.alignformer_ext import variants  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cache", type=Path, default=Path("outputs/propose_cache_opv2v/sigma_1_seed_0.npz"))
args = parser.parse_args()
records = check_verify_core.load_frame_table(Path("outputs/opv2v_frame_table.pt"))
model = base.load_alignformer_pva(Path("models/opv2v/alignformer_boxes_only.pth"), torch.device("cpu"))[0]
with np.load(args.cache) as data:
    cache = {k: data[k] for k in data.files}
inputs = variants.PairInputs(records, cache, model, base.RobustSolveConfig(mode="huber", iterations=2))
a = inputs.attempted
off = inputs.repropose(spread=False)
on = inputs.repropose(spread=True)
ref = np.column_stack([cache["propose"][a], cache["statistic"][a], cache["dof"][a]])
names = ["psi", "tx", "ty", "statistic", "dof"]
print(f"{len(a)} pairs; spread OFF vs cached Propose, max |diff|: "
      + ", ".join(f"{n}={np.abs(off[:, i] - ref[:, i]).max():.2e}" for i, n in enumerate(names)))
print("spread ON vs cached, median |diff|: "
      + ", ".join(f"{n}={np.median(np.abs(on[:, i] - ref[:, i])):.3g}" for i, n in enumerate(names)))
