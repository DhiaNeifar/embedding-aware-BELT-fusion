#!/usr/bin/env python3
"""CoAlign's and RoCo's corrected poses, in the same files as the other plug-ins (alignformer_ext/poses.py format).

Each method runs its own code, from its own repository and environment, exactly as its intermediate-fusion dataset
calls it: ``box_alignment_relative_sample_np`` (agent-object pose graph; RoCo's version replaces CoAlign's matching)
on the method's own stage-1 boxes and uncertainties, with the ``box_align.args`` of its own config. The ego (first
agent) is fixed. Input poses are the noisy reported poses read from the AlignFormer pose files, so every plug-in
corrects the same noise draws. Output: the corrected CAV-to-ego transform ``x1_to_x2(refined CAV, refined ego)``.

Run in the method's own environment (it needs g2o):

    /mnt/external/backup/dhia/conda_envs/coalign-blackwell/bin/python scripts/export_poses_pgo.py --method coalign
    /mnt/external/backup/dhia/conda_envs/roco-blackwell/bin/python scripts/export_poses_pgo.py --method roco
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import yaml
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
METHODS = {
    "coalign": dict(repo=REPO / "external" / "CoAlign", config=REPO / "models/opv2v/intermediate_fusion/coalign_boxalign/config.yaml"),
    "roco": dict(repo=REPO / "external" / "RoCo_merged", config=REPO / "models/opv2v/intermediate_fusion/roco/config.yaml"),
}
# Stage-1 boxes and uncertainty per dataset. OPV2V: the CoAlign bundle's file. V2X-Real: our stage-1 detector trained
# on V2X-Real in CoAlign's code (scripts/coalign_v2xreal.py). The box_align args read from METHODS' configs are the
# same for every dataset in CoAlign's own configs (opv2v, dairv2x, v2xsim), so they are shared.
# OPV2V: ours since Nov 6 (CoAlign's recipe, last epoch); the labmate's bundle boxes, used before, are still at
# models/opv2v/intermediate_fusion/coalign_boxalign/stage1_boxes.json (pass it with --stage1 to reproduce those runs).
STAGE1 = {"opv2v": REPO / "models/opv2v/intermediate_fusion/coalign_stage1/stage1_boxes_test.json",
          "v2xreal": REPO / "models/v2xreal/intermediate_fusion/coalign_boxalign/stage1_boxes_test.json"}
_W = {}


def run(job):
    sigma, seed = job
    from opencood.models.sub_modules.box_align_v2 import box_alignment_relative_sample_np
    from opencood.utils.transformation_utils import x1_to_x2

    source = _W["poses"] / _W["dataset"] / "alignformer" / f"sigma_{sigma:g}_seed_{seed}.npz"
    target = _W["poses"] / _W["dataset"] / _W["method"] / source.name
    if target.exists() and not _W["overwrite"]:
        return job, None
    with np.load(source) as z:
        arrays = {k: z[k] for k in z.files}
    n = len(arrays["cav_id"])
    transform = np.empty((n, 4, 4))
    correction = np.zeros((n, 3))
    corrected = 0
    for f in np.unique(arrays["frame_index"]):
        rows = np.flatnonzero(arrays["frame_index"] == f)
        stage1 = _W["stage1"].get(str(int(f)))      # absent only when no agent detected anything (CoAlign skips it too)
        ego_id = str(arrays["ego_id"][rows[0]])
        ego_pose = arrays["ego_pose"][rows[0]].astype(np.float64)
        order = [ego_id] + [str(arrays["cav_id"][r]) for r in rows]
        noisy = np.stack([ego_pose] + [arrays["reported_pose"][r].astype(np.float64) for r in rows])
        refined = noisy.copy()
        corners = []
        if stage1 is not None:
            ids = [str(i) for i in stage1["cav_id_list"]]
            corners = [np.array(stage1["pred_corner3d_np_list"][ids.index(a)], np.float64) for a in order]
            uncertainty = [np.array(stage1["uncertainty_np_list"][ids.index(a)], np.float64) for a in order]
        if sum(len(c) for c in corners):
            refined[:, [0, 1, 4]] = box_alignment_relative_sample_np(corners, noisy.copy(), uncertainty_list=uncertainty,
                                                                     **_W["args"])
        for k, r in enumerate(rows, start=1):
            transform[r] = x1_to_x2(refined[k].tolist(), refined[0].tolist())
            reported = x1_to_x2(noisy[k].tolist(), noisy[0].tolist())
            c = transform[r] @ np.linalg.inv(reported)          # the correction in the ego frame
            correction[r] = (np.arctan2(c[1, 0], c[0, 0]), c[0, 3], c[1, 3])
            corrected += int(np.abs(correction[r]).max() > 1e-9)
    arrays.update(transform=transform, correction=correction)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.rename(target)
    return job, corrected / max(n, 1)


def box_align_block(config: Path) -> str:
    """The ``box_align`` section of a run config (the rest carries python-specific yaml tags safe_load rejects)."""
    lines, out = open(config).read().splitlines(), []
    for line in lines:
        if out and line and not line.startswith((" ", "\t")):
            break
        if out or line.startswith("box_align:"):
            out.append(line)
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--dataset", choices=sorted(STAGE1), default="opv2v")
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--poses", type=Path, default=REPO / "outputs" / "poses")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stage1", type=Path, help="stage-1 boxes file (default: STAGE1[dataset])")
    args = parser.parse_args()

    method = METHODS[args.method]
    sys.path.insert(0, str(method["repo"]))
    box_align = yaml.safe_load(box_align_block(method["config"]))["box_align"]
    stage1_path = args.stage1 or STAGE1[args.dataset]
    if not stage1_path.exists():
        raise SystemExit(f"{stage1_path} missing: train stage 1 and write its boxes first "
                         f"(scripts/run_paper.sh {args.dataset}-stage1, then {args.dataset}-stage1-boxes)")
    print(f"stage-1 boxes: {stage1_path}")
    _W.update(method=args.method, dataset=args.dataset, poses=args.poses, overwrite=args.overwrite,
              args=box_align["args"], stage1=json.load(open(stage1_path)))
    jobs = [(sigma, seed) for sigma in args.sigmas for seed in ([0] if sigma == 0.0 else range(args.seeds))]
    missing = [j for j in jobs if not (args.poses / args.dataset / "alignformer" / f"sigma_{j[0]:g}_seed_{j[1]}.npz").exists()]
    if missing:
        raise SystemExit(f"run scripts/export_poses.py first: {len(missing)} AlignFormer pose files missing, e.g. {missing[0]}")
    fractions = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"export poses {args.dataset} {args.method}"):
            _, fraction = future.result()
            if fraction is not None:
                fractions.append(fraction)
    share = f"; CAVs it changed: {np.mean(fractions) * 100:.0f}%" if fractions else ""
    print(f"{len(jobs)} files in {args.poses / args.dataset / args.method}{share}")


if __name__ == "__main__":
    main()
