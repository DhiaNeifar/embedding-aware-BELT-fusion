"""Dataset calibration for CooperScene: a constant offset between each agent's LiDAR scan and its recorded pose.

In CooperScene every agent's scan sits at a fixed rigid offset from where its recorded pose puts it (measured with
GICP on the point clouds, started at the true pose: up to 1.2 m and 1.7 deg, stable to a few centimetres per
recording), while the ground-truth boxes follow the recorded poses. Any method that aligns the agents' sensor
data (boxes or points) recovers the scan-consistent pose ``T o c`` instead of the recorded pose ``T``, and is
penalized by exactly ``c`` at every frame. This is a property of the dataset, not of a method, so it is handled
as a calibration given equally to every alignment method: a pose ``X`` estimated from sensor data is converted
to the dataset's pose convention by ``X o c^-1``. Pairs a method left uncorrected keep the reported pose, which
is already in that convention.

``c`` is measured once per (recording, agent) from the GICP registrations at sigma 0 (``measure_offsets``) and
stored as JSON; recordings are told apart by ego jumps, as in temporal.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from embedding_aware_belt_fusion.alignformer_ext import temporal
from embedding_aware_belt_fusion.alignformer_ext.joint import compose, inverse

Key = Tuple[int, str]
ACTIVE: Optional[Dict[Key, np.ndarray]] = None   # set by the sweep's --calibration (inherited by forked workers)


def active() -> Dict[Key, np.ndarray]:
    if ACTIVE is None:
        raise ValueError("calibrated variants need the dataset's offsets: run the sweep with --calibration <json>")
    return ACTIVE


def recordings(records) -> list:
    """Recording index of every frame (a new recording starts where the ego jumps > SCENARIO_JUMP_M)."""
    index, last, out = 0, None, []
    for record in records:
        ego = record.ego_pose
        if last is not None and np.hypot(ego[0] - last[0], ego[1] - last[1]) > temporal.SCENARIO_JUMP_M:
            index += 1
        last = ego
        out.append(index)
    return out


def measure_offsets(base, records, cache, registration) -> Dict[Key, np.ndarray]:
    """Median scan-vs-pose offset (x, y, yaw in the agent's frame) per (recording, agent) from registrations
    ``registration`` (pair, psi, tx, ty: ego-frame corrections from the TRUE pose, i.e. a sigma-0 cache)."""
    rec = recordings(records)
    samples: Dict[Key, list] = {}
    for i, p in enumerate(registration["pair"]):
        frame = int(cache["frame"][p])
        record = records[frame]
        agent = sorted(record.cav_boxes)[int(cache["cav_rank"][p])]
        truth = temporal._pose2d(base.x1_to_x2(record.cav_poses[agent], record.ego_pose))
        scan = compose(np.array([registration["tx"][i], registration["ty"][i], registration["psi"][i]]), truth)
        local = compose(inverse(truth), scan)
        local[2] = temporal._wrap(local[2])
        samples.setdefault((rec[frame], agent), []).append(local)
    out = {}
    for key, values in samples.items():
        h = np.array(values)
        out[key] = np.array([np.median(h[:, 0]), np.median(h[:, 1]),
                             np.arctan2(np.median(np.sin(h[:, 2])), np.median(np.cos(h[:, 2])))])
    return out


def save_offsets(offsets: Dict[Key, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"recording": k[0], "agent": k[1], "x": float(v[0]), "y": float(v[1]), "yaw": float(v[2])}
                                for k, v in sorted(offsets.items())], indent=1))


def load_offsets(path: Path) -> Dict[Key, np.ndarray]:
    return {(int(e["recording"]), str(e["agent"])): np.array([e["x"], e["y"], e["yaw"]])
            for e in json.loads(Path(path).read_text())}


def to_recorded_convention(inputs, decided: np.ndarray, offsets: Dict[Key, np.ndarray]) -> np.ndarray:
    """(P, 3) psi, tx, ty: each applied correction re-expressed in the dataset's pose convention, X o c^-1."""
    from embedding_aware_belt_fusion.alignformer_ext.propose_cache import noisy_poses

    base, records, cache = inputs.base, inputs.records, inputs.cache
    sigma, seed = float(cache["sigma"]), int(cache["seed"])
    rec = recordings(records)
    out = np.array(decided, copy=True)
    reported: dict = {}
    for p in range(len(cache["frame"])):
        if not np.any(decided[p] != 0):
            continue
        frame = int(cache["frame"][p])
        record = records[frame]
        agent = sorted(record.cav_boxes)[int(cache["cav_rank"][p])]
        offset = offsets.get((rec[frame], agent))
        if offset is None:
            continue
        if frame not in reported:
            reported = {frame: noisy_poses(base, record, frame, sigma, seed)}
        n = temporal._pose2d(base.x1_to_x2(reported[frame][agent], record.ego_pose))
        d = decided[p]
        estimate = compose(compose(np.array([d[1], d[2], d[0]], np.float64), n), inverse(offset))
        correction = compose(estimate, inverse(n))
        out[p] = (correction[2], correction[0], correction[1])
    return out
