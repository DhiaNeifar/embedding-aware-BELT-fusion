"""Audit E2: is the learned Propose stage needed? Per-frame baselines without it, and their pooled versions.

Every baseline maps one cached condition to per-pair decided corrections (psi, tx, ty), like the variants in
``variants.py``, and reads only what AlignFormer reads: both agents' boxes in the ego frame under the REPORTED pose
(``PairInputs``) and, for ``propose_hard``, Propose's own cached soft assignment. None reads sigma, the noise kind or
any true pose.

* ``verify_from_reported``: Verify started from the reported pose (proposal zero) with its gates and decision rule; the
  Wald statistic is zero, so a proposal can never be accepted and only Verify can answer.
* ``hungarian_ransac``: Hungarian assignment of box centres under the reported pose (cost capped at 6 m), then RANSAC
  over two-pair samples (inliers within 1 m) and a least-squares refit on the consensus centres.
* ``propose_hard``: Propose's row-wise argmax assignment (dustbin excluded), then a least-squares fit on centres.
* ``*_tk``: the same per-frame decisions pooled over time by ``temporal.temporal_decisions`` with alignformer_tk's
  settings, in place of alignformer_c's.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import numpy as np

from embedding_aware_belt_fusion.alignformer_ext import variants, verify_cpp

COST_CAP_M = 6.0
INLIER_M = 1.0
MIN_PAIRS = 3
RANSAC_SAMPLES = 512


def _fit(p: np.ndarray, q: np.ndarray):
    """Least-squares (psi, t) taking centres q onto p (closed-form Kabsch in the plane)."""
    pc, qc = p.mean(0), q.mean(0)
    p0, q0 = p - pc, q - qc
    psi = np.arctan2(np.sum(q0[:, 0] * p0[:, 1] - q0[:, 1] * p0[:, 0]), np.sum(q0[:, 0] * p0[:, 0] + q0[:, 1] * p0[:, 1]))
    c, s = np.cos(psi), np.sin(psi)
    t = pc - np.array([c * qc[0] - s * qc[1], s * qc[0] + c * qc[1]])
    return psi, t


def _residual(p, q, psi, t):
    c, s = np.cos(psi), np.sin(psi)
    moved = np.stack([c * q[:, 0] - s * q[:, 1] + t[0], s * q[:, 0] + c * q[:, 1] + t[1]], 1)
    return np.linalg.norm(moved - p, axis=1)


def _ransac(p: np.ndarray, q: np.ndarray, rng: np.random.Generator):
    """(psi, tx, ty) or None: RANSAC over two-pair samples, then the refit on the largest consensus."""
    k = len(p)
    if k < MIN_PAIRS:
        return None
    samples = list(combinations(range(k), 2))
    if len(samples) > RANSAC_SAMPLES:
        samples = [tuple(rng.choice(k, 2, replace=False)) for _ in range(RANSAC_SAMPLES)]
    best, best_key = None, None
    for i, j in samples:
        if np.linalg.norm(q[i] - q[j]) < 1e-3:
            continue
        psi, t = _fit(p[[i, j]], q[[i, j]])
        r = _residual(p, q, psi, t)
        inliers = r < INLIER_M
        key = (int(inliers.sum()), -float(r[inliers].sum()))
        if best_key is None or key > best_key:
            best, best_key = inliers, key
    if best is None or best.sum() < MIN_PAIRS:
        return None
    psi, t = _fit(p[best], q[best])
    return psi, t[0], t[1]


def verify_from_reported(inputs: variants.PairInputs) -> np.ndarray:
    c, a = inputs.cache, inputs.attempted
    zeros = np.zeros((len(a), 3), np.float32)
    decided = verify_cpp.verify(
        inputs.ego_rows, inputs.cav_rows, zeros, np.zeros(len(a), np.float32), c["dof"][a],
        inputs.model.variance_model, inputs.model.heading_lambda,
        replace(verify_cpp.VerifySettings(), heading_weight=0.0),
    )[0]
    return inputs.scatter(decided)


def hungarian_ransac(inputs: variants.PairInputs) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment

    rng = np.random.default_rng(0)
    out = np.zeros((len(inputs.attempted), 3), np.float32)
    for i, (ego, cav) in enumerate(zip(inputs.ego_rows, inputs.cav_rows)):
        if len(ego) == 0 or len(cav) == 0:
            continue
        cost = np.linalg.norm(ego[:, None, :2] - cav[None, :, :2], axis=2)
        rows, cols = linear_sum_assignment(np.minimum(cost, COST_CAP_M + 1.0))
        keep = cost[rows, cols] <= COST_CAP_M
        fit = _ransac(ego[rows[keep], :2].astype(np.float64), cav[cols[keep], :2].astype(np.float64), rng)
        if fit is not None:
            out[i] = fit
    return inputs.scatter(out)


def propose_hard(inputs: variants.PairInputs) -> np.ndarray:
    c, a = inputs.cache, inputs.attempted
    shapes = c["plan_shape"]
    offsets = np.concatenate([[0], np.cumsum(shapes[:, 0] * shapes[:, 1])])
    out = np.zeros((len(a), 3), np.float32)
    for i, p in enumerate(a):
        rows_n, cols_n = int(shapes[p, 0]), int(shapes[p, 1])
        if rows_n < 2 or cols_n < 2:
            continue
        plan = c["plan"][offsets[p]:offsets[p + 1]].reshape(rows_n, cols_n)
        best = plan[:-1].argmax(axis=1)                       # each ego box's most likely partner, dustbin included
        matched = np.flatnonzero(best < cols_n - 1)            # the last column is the dustbin
        if len(matched) < MIN_PAIRS:
            continue
        ego, cav = inputs.ego_rows[i], inputs.cav_rows[i]
        psi, t = _fit(ego[matched, :2].astype(np.float64), cav[best[matched], :2].astype(np.float64))
        out[i] = (psi, t[0], t[1])
    return inputs.scatter(out)


def _pooled(per_frame):
    def run(inputs: variants.PairInputs) -> np.ndarray:
        from embedding_aware_belt_fusion.alignformer_ext import temporal

        settings = temporal.TemporalSettings(odometry_sd=(0.0, 0.0), odometry_rel=variants._LOAM, consistency="adaptive")
        return temporal.temporal_decisions(inputs, settings, measurements=per_frame(inputs))

    run.__doc__ = f"{per_frame.__name__}'s decisions pooled over time with alignformer_tk's settings."
    return run


BASELINES = {
    "verify_from_reported": verify_from_reported,
    "verify_from_reported_tk": _pooled(verify_from_reported),
    "hungarian_ransac": hungarian_ransac,
    "hungarian_ransac_tk": _pooled(hungarian_ransac),
    "propose_hard": propose_hard,
}
