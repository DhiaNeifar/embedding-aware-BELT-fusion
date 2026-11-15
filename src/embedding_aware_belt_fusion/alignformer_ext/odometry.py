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

    def pose2d(self, frame: int, agent: str) -> Optional[np.ndarray]:
        """(x, y, yaw) of ``agent`` at frame-table row ``frame`` in its own odometry frame; None without a scan."""
        scenario, frame_id, ego = self.frames[frame]
        r = self.row.get((scenario, ego if agent == EGO else agent, frame_id))
        if r is None:
            return None
        m = self.pose[r]
        return np.array([m[0, 3], m[1, 3], np.arctan2(m[1, 0], m[0, 0])])

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


# ---------------------------------------------------------------------------------------------------------------------
# Injected odometry error (audit E3). Simulator side only: it corrupts the odometry the estimator is FED, while the
# estimator keeps assuming LOAM accuracy (temporal.TemporalSettings is unchanged), so nothing here leaks to it.
# ---------------------------------------------------------------------------------------------------------------------

ERROR_KINDS = ("white", "ar1", "scale", "yawdrift")
LOAM_REL = (0.0055, 0.0013)       # LOAM on KITTI: 0.55 % of distance, 0.0013 deg per metre (= temporal.LOAM_KITTI)
AR1_COEFFICIENT = 0.95
SCALE_BIAS = 0.01                 # translation of every step 1 % too long
YAW_DRIFT_DEG_PER_M = 0.01        # a constant yaw drift, deg per metre travelled


class PerturbedOdometry:
    """An odometry table whose consecutive steps carry an injected error of one kind.

    The error is added to the step ``X = between(earlier, later)`` of each agent as ``X + e`` on (x, y, yaw), the same
    additive model the estimator assumes; the reverse step is its inverse. ``white`` and ``ar1`` draw ``e`` at LOAM's
    level for the distance moved (``ar1``: coefficient 0.95 per step, same stationary level); ``scale`` lengthens the
    translation by 1 %; ``yawdrift`` adds 0.01 deg per metre. One realisation per ``seed``, deterministic per
    (scenario, agent, step) whatever order the steps are asked in.
    """

    def __init__(self, table: OdometryTable, kind: str, seed: int = 0):
        if kind not in ERROR_KINDS:
            raise ValueError(f"unknown odometry error {kind!r}, expected one of {ERROR_KINDS}")
        self.table, self.kind, self.seed = table, kind, int(seed)
        self.frames = table.frames
        self._unit: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}

    def pose2d(self, frame: int, agent: str) -> Optional[np.ndarray]:
        return self.table.pose2d(frame, agent)

    def _units(self, scenario: str, name: str) -> Dict[str, np.ndarray]:
        """Unit-variance error per step of one agent, keyed by the later frame id of the step."""
        key = (scenario, name)
        if key not in self._unit:
            import zlib

            ids = sorted(f for (s, a, f) in self.table.row if s == scenario and a == name)
            rng = np.random.default_rng([self.seed, zlib.crc32(scenario.encode()), zlib.crc32(name.encode())])
            draws = rng.normal(0.0, 1.0, (len(ids), 3))
            if self.kind == "ar1":
                a = AR1_COEFFICIENT
                for k in range(1, len(ids)):
                    draws[k] = a * draws[k - 1] + np.sqrt(1.0 - a * a) * draws[k]
            self._unit[key] = dict(zip(ids, draws))
        return self._unit[key]

    def between(self, source: int, target: int, agent: str) -> Optional[np.ndarray]:
        step = self.table.between(source, target, agent)
        if step is None:
            return None
        from embedding_aware_belt_fusion.alignformer_ext.se2 import inverse

        scenario, source_id, ego = self.frames[source]
        _, target_id, _ = self.frames[target]
        forward = source_id < target_id              # frame ids are zero-padded, so string order is time order
        x = step if forward else inverse(step)       # the step from the earlier frame to the later one
        moved = float(np.hypot(x[0], x[1]))
        if self.kind == "scale":
            x = np.array([x[0] * (1.0 + SCALE_BIAS), x[1] * (1.0 + SCALE_BIAS), x[2]])
        elif self.kind == "yawdrift":
            x = np.array([x[0], x[1], x[2] + np.radians(YAW_DRIFT_DEG_PER_M * moved)])
        else:
            name = ego if agent == EGO else agent
            unit = self._units(scenario, name)[max(source_id, target_id)]
            sd = np.array([LOAM_REL[0] * moved, LOAM_REL[0] * moved, np.radians(LOAM_REL[1] * moved)])
            x = x + unit * sd
        return x if forward else inverse(x)
