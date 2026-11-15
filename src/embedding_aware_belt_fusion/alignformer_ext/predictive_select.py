"""Cross-fitted predictive selection between a correction and the reported pose.

The transport plan is treated as a distribution over a real partner or the
dustbin.  For every ego object, a pose is fitted without that object and its
predictive likelihood is compared with the identity on the held-out object.
The selector never changes an accepted correction; it only decides whether the
full AlignFormer estimate is safer than retaining the reported pose.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp


def _plans(cache):
    pointer, output = 0, []
    for rows, columns in cache["plan_shape"]:
        size = int(rows * columns)
        output.append(cache["plan"][pointer:pointer + size].reshape(int(rows), int(columns)) if size else None)
        pointer += size
    if pointer != len(cache["plan"]):
        raise RuntimeError("transport-plan cache is inconsistent")
    return output


def _solve(target: np.ndarray, source: np.ndarray, weights: np.ndarray):
    """Weighted SE(2) fit over all soft pair edges."""
    mass = float(weights.sum())
    row_mass = weights.sum(1)
    if mass <= 1e-9 or np.count_nonzero(row_mass > 1e-5 * mass) < 2:
        return None
    pbar = (row_mass[:, None] * target).sum(0) / mass
    col_mass = weights.sum(0)
    qbar = (col_mass[:, None] * source).sum(0) / mass
    covariance = (source - qbar).T @ weights.T @ (target - pbar)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = pbar - rotation @ qbar
    return np.array([math.atan2(rotation[1, 0], rotation[0, 0]), *translation])


def _move(points: np.ndarray, theta: np.ndarray):
    yaw, tx, ty = map(float, theta)
    c, s = math.cos(yaw), math.sin(yaw)
    return points @ np.array([[c, s], [-s, c]]) + np.array([tx, ty])


def _row_gain(log_plan_row, target, source, candidate, scale_m):
    """log p(point|candidate) - log p(point|identity), dustbin included."""
    logits = np.asarray(log_plan_row, np.float64)
    logits = logits - logsumexp(logits)

    def score(theta):
        residual2 = np.sum((_move(source, theta) - target) ** 2, axis=1)
        real = logits[:-1] - 0.5 * residual2 / (scale_m * scale_m)
        # Uniform clutter likelihood over a disc.  Expressed relative to the
        # Gaussian normalizer, so it is a fixed, interpretable 3-sigma cost.
        dustbin = logits[-1] - 4.5
        return float(logsumexp(np.r_[real, dustbin]))

    return score(candidate) - score(np.zeros(3))


def evidence(inputs, *, scale_m: float = 1.0, mode: str = "crossfit"):
    """Per-pair predictive gain, standard error, and number of held-out rows."""
    if mode not in {"crossfit", "insample"}:
        raise ValueError(mode)
    plans = _plans(inputs.cache)
    gains = np.full(len(inputs.cache["frame"]), np.nan, np.float64)
    errors = np.full_like(gains, np.inf)
    counts = np.zeros(len(gains), np.int32)
    for local, pair_index in enumerate(inputs.attempted):
        plan = plans[pair_index]
        if plan is None or min(plan.shape) <= 1:
            continue
        batch = inputs.batches[local]
        ego = batch["ego_boxes"][0, :, :2].detach().cpu().numpy().astype(np.float64)
        cav = batch["cav_boxes"][0, :, :2].detach().cpu().numpy().astype(np.float64)
        if len(ego) < 3 or len(cav) < 2:
            continue
        log_rows = plan[:-1]
        weights = np.exp(plan[:-1, :-1].astype(np.float64))
        observations = []
        full = _solve(ego, cav, weights) if mode == "insample" else None
        for held in range(len(ego)):
            if mode == "crossfit":
                keep = np.arange(len(ego)) != held
                candidate = _solve(ego[keep], cav, weights[keep])
            else:
                candidate = full
            if candidate is not None:
                observations.append(_row_gain(log_rows[held], ego[held], cav, candidate, scale_m))
        if len(observations) < 2:
            continue
        values = np.asarray(observations)
        gains[pair_index] = float(values.mean())
        errors[pair_index] = float(values.std(ddof=1) / math.sqrt(len(values)))
        counts[pair_index] = len(values)
    return gains, errors, counts


def decide(inputs, decided: np.ndarray, *, mode: str = "crossfit", confidence_z: float = 0.0,
           scale_m: float = 1.0):
    """Abstain unless predictive gain's lower bound is positive.

    Pairs without enough objects retain the existing AlignFormer decision: a
    new selector must not turn missing evidence into systematic rejection.
    """
    gain, error, count = evidence(inputs, scale_m=scale_m, mode=mode)
    output = decided.copy()
    tested = count >= 2
    reject = tested & ((gain - confidence_z * error) <= 0.0)
    output[reject] = 0.0
    return output, {"gain": gain, "error": error, "count": count, "rejected": reject}


__all__ = ["decide", "evidence"]
