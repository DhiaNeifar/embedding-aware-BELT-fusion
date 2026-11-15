#!/usr/bin/env python3
"""T4: pose accuracy of every plug-in, independent of any detector-fusion network. No AP, no run of any model.

Reads the exported poses (outputs/poses/<dataset>/<plugin>/sigma_<s>_seed_<k>.npz, the files behind T1 and T3) and
compares each corrected CAV-to-ego transform with the one from the dataset's recorded poses:

* error: translation (m) and yaw (deg) of T_est^-1 T_recorded, per (frame, CAV) pair;
* answered: the plug-in changed the reported pose (FreeAlign and AlignFormer may decline; CoAlign and RoCo almost
  always move it);
* sparse: pairs where fewer than 3 of the ego's detections have one of the CAV's detections within 2 m once both are
  placed with the recorded poses (few shared objects, where graph matching declines).

On V2X-Real the recorded poses carry their own error (AlignFormer's median correction at sigma = 0 is 29 cm), so
errors there are against the recorded pose, not the truth.

    python scripts/pose_accuracy.py --dataset v2xreal --output outputs/paper/pose_accuracy_v2xreal.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_verify_core  # noqa: E402
import reproduce_opv2v_sweep as base  # noqa: E402

TABLES = {"opv2v": "outputs/opv2v_frame_table.pt", "v2xreal": "outputs/v2xreal_test_frame_table.pt"}
PLUGINS = ["freealign", "coalign", "roco", "alignformer"]
SHARED_RADIUS_M = 2.0
SPARSE_BELOW = 3


def yaw_of(matrix: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))


def shared_objects(record, cav_id: str, true_transform: np.ndarray) -> int:
    ego = record.ego_boxes[:, :2].numpy()
    cav = base.transform_boxes_to_ego(record.cav_boxes[cav_id], torch.as_tensor(true_transform, dtype=torch.float32))
    cav = cav[:, :2].numpy()
    if len(ego) == 0 or len(cav) == 0:
        return 0
    distance = np.linalg.norm(ego[:, None, :] - cav[None, :, :], axis=-1)
    return int(np.sum(distance.min(axis=1) < SHARED_RADIUS_M))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(TABLES), required=True)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--poses", type=Path, default=Path("outputs/poses"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = check_verify_core.load_frame_table(Path(TABLES[args.dataset]))
    truth, sparse = {}, {}
    rows = {"bare": {"t": [], "yaw": [], "answered": [], "sparse": []}}
    rows.update({p: {"t": [], "yaw": [], "answered": [], "sparse": []} for p in PLUGINS})
    jobs = [(plugin, seed) for plugin in PLUGINS for seed in range(args.seeds)]
    for plugin, seed in tqdm(jobs, desc=f"pose accuracy {args.dataset} sigma {args.sigma:g}"):
        path = args.poses / args.dataset / plugin / f"sigma_{args.sigma:g}_seed_{seed}.npz"
        with np.load(path) as z:
            for f, cav, reported, ego, correction, transform in zip(z["frame_index"], z["cav_id"], z["reported_pose"],
                                                                     z["ego_pose"], z["correction"], z["transform"]):
                key = (int(f), str(cav))
                record = records[key[0]]
                if key not in truth:
                    truth[key] = base.x1_to_x2(record.cav_poses[key[1]], record.ego_pose)
                    sparse[key] = shared_objects(record, key[1], truth[key]) < SPARSE_BELOW
                residual = np.linalg.inv(transform) @ truth[key]
                rows[plugin]["t"].append(float(np.linalg.norm(residual[:2, 3])))
                rows[plugin]["yaw"].append(abs(yaw_of(residual)))
                rows[plugin]["answered"].append(bool(np.any(np.abs(correction) > 1e-9)))
                rows[plugin]["sparse"].append(sparse[key])
                if plugin == PLUGINS[0]:      # no correction: the reported pose, once per pair and seed
                    bare = np.linalg.inv(base.x1_to_x2(list(reported), list(ego))) @ truth[key]
                    rows["bare"]["t"].append(float(np.linalg.norm(bare[:2, 3])))
                    rows["bare"]["yaw"].append(abs(yaw_of(bare)))
                    rows["bare"]["answered"].append(False)
                    rows["bare"]["sparse"].append(sparse[key])

    summary = {}
    for name, row in rows.items():
        t, yaw = np.array(row["t"]), np.array(row["yaw"])
        answered, is_sparse = np.array(row["answered"]), np.array(row["sparse"])
        summary[name] = {
            "pairs": int(len(t)), "median_m": float(np.median(t)), "p90_m": float(np.percentile(t, 90)),
            "median_deg": float(np.median(yaw)), "answered": float(answered.mean()),
            "sparse_share": float(is_sparse.mean()),
            "sparse_answered": float(answered[is_sparse].mean()) if is_sparse.any() else float("nan"),
            "sparse_median_m": float(np.median(t[is_sparse])) if is_sparse.any() else float("nan"),
            # error distribution for the figure: the 0th..100th percentiles
            "percentiles_m": [float(v) for v in np.percentile(t, np.arange(101))],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"dataset": args.dataset, "sigma": args.sigma, "seeds": args.seeds,
                                       "shared_radius_m": SHARED_RADIUS_M, "sparse_below": SPARSE_BELOW,
                                       "summary": summary}, indent=1))
    print(f"Wrote {args.output}")
    for name, s in summary.items():
        print(f"  {name:12s} median {s['median_m']:.3f} m  p90 {s['p90_m']:.3f} m  yaw {s['median_deg']:.2f} deg  "
              f"answered {100 * s['answered']:.0f}%  sparse answered {100 * s['sparse_answered']:.0f}%")


if __name__ == "__main__":
    main()
