"""Planar rigid transforms as (tx, ty, psi) with ``X(p) = R(psi) p + t``."""

from __future__ import annotations

import numpy as np


def _wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def compose(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``X o Y``."""
    c, s = np.cos(x[2]), np.sin(x[2])
    return np.array([c * y[0] - s * y[1] + x[0], s * y[0] + c * y[1] + x[1], _wrap(x[2] + y[2])])


def inverse(x: np.ndarray) -> np.ndarray:
    c, s = np.cos(x[2]), np.sin(x[2])
    return np.array([-(c * x[0] + s * x[1]), -(-s * x[0] + c * x[1]), -x[2]])
