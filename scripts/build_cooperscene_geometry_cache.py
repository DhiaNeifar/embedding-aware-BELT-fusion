#!/usr/bin/env python3
"""CooperScene geometry cache from the cached GenCPM PointPillars detections.

No detector inference: ``gencpm infer`` already ran the CooperScene-trained
PointPillars (``AdverCPM_hf/models/detectors/cooperscene_pointpillars``,
net_epoch15) on every agent of validate/ and test/, writing per agent-frame
``<ts>_pred.npy`` (post-processed (N, 8, 3) corners in that agent's own LiDAR
frame) and ``<ts>_score.npy``. This converts them to the shard schema
``scripts/reproduce_opv2v_sweep.py`` reads from ``outputs/r140_geometry_test_cache``
(``scenario_id``/``frame_id``/``agents[{agent_id, boxes (N, 7) x y z h w l yaw,
scores, transformation_matrix}]``), with ``transformation_matrix`` the true
agent->ego transform for the chosen ego (identity for the ego itself).

Corners -> centres goes through OpenCOOD's own ``corner_to_center`` (hwl), the
inverse of the ``boxes_to_corners_3d`` GenCPM used to write them; heading is
recovered from the corner order, not just mod pi (round trip checked in
``tests/test_multidataset_sources.py``).

Usage:
    python scripts/build_cooperscene_geometry_cache.py \\
        --gencpm-root /home/dhia/cooperscene_preds/pp/TownCooperScene \\
        --split test --output-dir outputs/cooperscene_geometry_test_cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "external" / "OpenCOOD"))

from opencood.utils import box_utils  # noqa: E402
from opencood.utils.transformation_utils import x1_to_x2  # noqa: E402

from embedding_aware_belt_fusion.multidataset.sources import COOPERSCENE_DEFAULT_EGO  # noqa: E402

_Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gencpm-root", type=Path, required=True)
    parser.add_argument("--split", choices=("validate", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ego-agent", default=COOPERSCENE_DEFAULT_EGO)
    parser.add_argument("--shard-frames", type=int, default=200)
    return parser.parse_args()


def local_boxes(pred_path: Path, score_path: Path):
    if not pred_path.exists():
        return torch.zeros((0, 7)), torch.zeros((0,))
    corners = np.load(pred_path).astype(np.float32)
    scores = np.load(score_path).astype(np.float32)
    if corners.shape[0] != scores.shape[0]:
        raise ValueError(f"{pred_path}: {corners.shape[0]} boxes but {scores.shape[0]} scores")
    if corners.shape[0] == 0:
        return torch.zeros((0, 7)), torch.zeros((0,))
    centres = box_utils.corner_to_center(corners, order="hwl").astype(np.float32)
    return torch.from_numpy(centres), torch.from_numpy(scores)


def lidar_pose(yaml_path: Path):
    with open(yaml_path) as stream:
        return list(map(float, yaml.load(stream, Loader=_Loader)["lidar_pose"]))


def main() -> None:
    args = parse_args()
    episodes = sorted(p for p in args.gencpm_root.iterdir() if p.is_dir() and p.name.startswith(f"{args.split}_"))
    if not episodes:
        raise FileNotFoundError(f"no {args.split}_* episodes under {args.gencpm_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shard, shard_index, frame_total, box_total = [], 0, 0, 0

    def flush():
        nonlocal shard, shard_index
        if shard:
            torch.save(shard, args.output_dir / f"{args.split}_{shard_index:05d}.pt")
            shard_index += 1
            shard = []

    for episode in episodes:
        agents = sorted(p.name for p in episode.iterdir() if p.is_dir())
        if args.ego_agent not in agents:
            raise KeyError(f"{episode.name} has no ego agent {args.ego_agent!r} (agents: {agents})")
        timestamps = sorted(
            p.name[:-5] for p in (episode / args.ego_agent).iterdir()
            if p.name.endswith(".yaml") and not p.name.endswith("_preds.yaml")
        )
        for ts in tqdm(timestamps, desc=episode.name, leave=False):
            ego_pose = lidar_pose(episode / args.ego_agent / f"{ts}.yaml")
            frame = {"scenario_id": episode.name, "frame_id": ts, "agents": []}
            for agent in agents:
                yaml_path = episode / agent / f"{ts}.yaml"
                if not yaml_path.exists():
                    continue
                boxes, scores = local_boxes(episode / agent / f"{ts}_pred.npy", episode / agent / f"{ts}_score.npy")
                transform = (
                    torch.eye(4) if agent == args.ego_agent
                    else torch.as_tensor(x1_to_x2(lidar_pose(yaml_path), ego_pose), dtype=torch.float32)
                )
                frame["agents"].append(
                    {"agent_id": agent, "boxes": boxes, "scores": scores, "transformation_matrix": transform}
                )
                box_total += int(boxes.shape[0])
            shard.append(frame)
            frame_total += 1
            if len(shard) >= args.shard_frames:
                flush()
    flush()
    print(f"Wrote {frame_total} frames ({box_total} boxes, {len(episodes)} episodes) to {args.output_dir}")


if __name__ == "__main__":
    main()
