"""Pose-noise kinds the benchmark lacks (T5). Simulator side only: nothing here is visible to any plug-in.

Every kind perturbs the CAV's reported pose (x, y in m, yaw in deg; yaw std = sigma deg per sigma m, as the
benchmark) and is drawn once per (seed, kind, scenario, CAV), so the same draw is shared by every method:

* ``drift``: Gauss-Markov per CAV and scenario, time constant 10 s at 10 Hz, stationary std ``sigma``: slow,
  correlated error like real GNSS. Errors no longer average out over a window.
* ``bias``: one constant offset per CAV and scenario, std ``sigma``.
* ``mixed``: each CAV and scenario draws its own std, uniform on [0, sqrt(3) sigma] (so the RMS over CAVs is
  ``sigma``), then independent per frame as the benchmark. Tests the online error level, which is pooled across CAVs.

The benchmark kind (independent per frame and CAV) stays in ``propose_cache.noisy_cav_boxes``'s own code path.
"""

from __future__ import annotations

import zlib
from typing import Dict, List, Tuple

import numpy as np

KINDS = ("drift", "bias", "mixed")
TAU_S = 10.0
DT_S = 0.1
_KIND_ID = {"drift": 1, "bias": 2, "mixed": 3}


def _crc(text: str) -> int:
    return zlib.crc32(str(text).encode())


class PoseNoise:
    """``offset(frame_index, cav_id)`` -> (dx m, dy m, dyaw deg) for one (kind, sigma, seed) condition.

    ``frames`` is the frame table's (scenario, frame_id, ego_id) per row (odometry.OdometryTable.frames); a frame's
    step within its scenario is its rank among that scenario's rows (consecutive rows are 0.1 s apart).
    """

    def __init__(self, kind: str, sigma: float, seed: int, frames: List[Tuple[str, str, str]], yaw_per_m: float = 1.0):
        if kind not in KINDS:
            raise ValueError(f"unknown noise kind {kind!r}, expected one of {KINDS}")
        self.kind, self.sigma, self.seed, self.frames = kind, float(sigma), int(seed), frames
        self.scale = np.array([1.0, 1.0, yaw_per_m])
        self.step: List[int] = []
        self.length: Dict[str, int] = {}
        for scenario, _, _ in frames:
            self.step.append(self.length.get(scenario, 0))
            self.length[scenario] = self.length.get(scenario, 0) + 1
        self._series: Dict[Tuple[str, str], np.ndarray] = {}

    def _rng(self, scenario: str, cav_id: str) -> np.random.Generator:
        return np.random.default_rng([self.seed, _KIND_ID[self.kind], int(round(self.sigma * 1000)),
                                      _crc(scenario), _crc(cav_id)])

    def _make(self, scenario: str, cav_id: str) -> np.ndarray:
        """(n_steps, 3) unit-free offsets, std ``sigma`` per axis before the yaw scale."""
        n, rng, s = self.length[scenario], self._rng(scenario, cav_id), self.sigma
        if self.kind == "bias":
            return np.repeat(rng.normal(0.0, s, (1, 3)), n, axis=0)
        if self.kind == "mixed":
            own = rng.uniform(0.0, np.sqrt(3.0) * s)
            return rng.normal(0.0, own, (n, 3))
        a = np.exp(-DT_S / TAU_S)                       # drift: AR(1), stationary from the first step
        out = np.empty((n, 3))
        out[0] = rng.normal(0.0, s, 3)
        innovation = rng.normal(0.0, s * np.sqrt(1.0 - a * a), (n, 3))
        for k in range(1, n):
            out[k] = a * out[k - 1] + innovation[k]
        return out

    def offset(self, frame_index: int, cav_id: str) -> np.ndarray:
        scenario = self.frames[frame_index][0]
        key = (scenario, str(cav_id))
        if key not in self._series:
            self._series[key] = self._make(*key)
        return self._series[key][self.step[frame_index]] * self.scale

    def perturb(self, pose, frame_index: int, cav_id: str, pose_x: int = 0, pose_y: int = 1, pose_yaw: int = 4):
        noisy = list(map(float, pose))
        dx, dy, dyaw = self.offset(frame_index, cav_id)
        noisy[pose_x] += float(dx)
        noisy[pose_y] += float(dy)
        noisy[pose_yaw] += float(dyaw)
        return noisy


__all__ = ["KINDS", "PoseNoise", "TAU_S", "DT_S"]
