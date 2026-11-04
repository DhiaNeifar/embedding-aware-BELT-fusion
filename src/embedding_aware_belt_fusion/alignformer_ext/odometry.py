"""Each vehicle's own LiDAR odometry for the temporal layer: KISS-ICP trajectories from scans alone.

``scripts/build_odometry_table.py`` runs KISS-ICP on every agent of a split without reading any recorded pose. This
module maps those trajectories onto a frame table. ``between`` returns the same quantity as
``base.x1_to_x2(pose at source, pose at target)``, reduced to (x, y, yaw), so the temporal layer's carry is unchanged.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

EGO = "ego"
ACTIVE = None   # set by the sweep's --odometry (inherited by forked workers); None = simulated odometry


def active():
    return ACTIVE


def _frame_index(geometry_dir: Path, records: List) -> List[Tuple[str, str, str]]:
    """(scenario, frame id, ego agent) of every frame-table row; the ego is the agent whose cached boxes are the
    record's ego boxes, decided per scenario because an agent with no boxes matches any empty frame."""
    frames = []
    for shard in sorted(glob.glob(str(Path(geometry_dir) / "*.pt"))):
        frames += torch.load(shard, weights_only=False)
    if len(frames) != len(records):
        raise ValueError(f"{geometry_dir} has {len(frames)} frames, the frame table has {len(records)}")
    votes: Dict[str, Dict[str, int]] = {}
    for frame, record in zip(frames, records):
        if len(record.ego_boxes) == 0:
            continue
        for agent in frame["agents"]:
            if torch.equal(torch.as_tensor(agent["boxes"]), record.ego_boxes):
                count = votes.setdefault(frame["scenario_id"], {})
                count[agent["agent_id"]] = count.get(agent["agent_id"], 0) + 1
    ego = {scenario: max(count, key=count.get) for scenario, count in votes.items()}
    return [(f["scenario_id"], f["frame_id"], ego.get(f["scenario_id"], f["agents"][0]["agent_id"])) for f in frames]


class OdometryTable:
    def __init__(self, table_path: Path, frames: List[Tuple[str, str, str]]):
        table = np.load(table_path)
        self.frames = frames
        self.pose = table["pose"]
        self.row = {key: i for i, key in enumerate(zip(table["scenario"].tolist(), table["agent"].tolist(),
                                                       table["frame"].tolist()))}

    @classmethod
    def for_records(cls, table_path: Path, geometry_dir: Path, records: List) -> "OdometryTable":
        return cls(table_path, _frame_index(geometry_dir, records))

    def between(self, source: int, target: int, agent: str) -> Optional[np.ndarray]:
        """(x, y, yaw) of frame ``source`` of ``agent`` expressed in its frame ``target``; None without both scans."""
        scenario, source_id, ego = self.frames[source]
        target_scenario, target_id, _ = self.frames[target]
        if scenario != target_scenario:
            return None
        name = ego if agent == EGO else agent
        a, b = self.row.get((scenario, name, source_id)), self.row.get((scenario, name, target_id))
        if a is None or b is None:
            return None
        m = np.linalg.inv(self.pose[b]) @ self.pose[a]
        return np.array([m[0, 3], m[1, 3], np.arctan2(m[1, 0], m[0, 0])])
