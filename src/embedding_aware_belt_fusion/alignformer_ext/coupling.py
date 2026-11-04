"""Idea 2: one Propose-Verify-Abstain decision instead of three ad-hoc rules.

The deployed rule applies Verify's re-solve when it agrees with Propose
within a fixed 1 m, OR whenever it rests on >= 4 RANSAC inliers (which then
overrides the Wald test entirely); the Wald test itself only ever looks at
Propose's soft fit. This replaces all three with statistics of the fit that is
actually applied, and the same test level the Abstain stage already uses:

* **Verify's own uncertainty.** The weighted least-squares covariance of
  Verify's final (centres-only) fit over its inlier pairs, with the scale
  estimated from those pairs' residuals, and the Wald test of its correction
  against zero at ``p * F_{1-level}(p, 2K - 3)`` -- exactly Abstain's
  reference (``abstain.abstention_threshold``), now applied to the fit that
  is applied. A fit whose objects sit in one cluster, or whose residuals are
  large (noisy real detections), is uncertain in rotation and is not applied
  unless its correction clearly exceeds that uncertainty.
* **Propose's match confidence.** Of Verify's inlier pairs, how many are also
  Propose's own decided matches (the CAV box holding >= half of that ego row's
  Sinkhorn mass). Verify must rest on at least ``min_pairs`` of them: a
  re-solve that Propose's association does not back is not applied.
* **Agreement, optional.** The Mahalanobis distance between Verify's and
  Propose's corrections under the sum of their covariances (Propose's is its
  own ``offset_precision``), against the same level -- the principled form
  of the fixed 1 m tolerance.

Where Verify did not engage, the deployed fallback is kept (Propose's
correction if its Wald test accepts, else none). No constant is fitted: the
level is the deployed 0.2, ``min_pairs`` the deployed 3, the 0.5 mass is a
majority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from embedding_aware_belt_fusion.alignformer_pva import abstain

SHARP_MASS = 0.5
RESIDUAL_SD_FLOOR_M = abstain.RESIDUAL_SD_FLOOR_M
KNOWN_SCALE_DOF = 4000.0  # top of abstain's F table: p * F(p, inf) ~ chi-square_p


def _threshold(dof: float, level: float) -> float:
    config = abstain.AbstentionConfig(mode=abstain.ABSTAIN, level=level)
    return float(abstain.abstention_threshold(torch.tensor([dof], dtype=torch.float32), config)[0])


@dataclass
class FitStats:
    wald: float
    dof: float
    covariance: np.ndarray  # (3, 3) over (tx, ty, psi)
    support: int
    inliers: int


def verify_fit_stats(ego_rows, cav_rows, pose, inlier_pairs, weight_ref: float, plan: Optional[np.ndarray]) -> Optional[FitStats]:
    """Statistics of Verify's final centres-only fit ``pose`` = (psi, tx, ty) over ``inlier_pairs`` (K, 2)."""
    k = len(inlier_pairs)
    if k < 2:
        return None
    e, c = ego_rows[inlier_pairs[:, 0]], cav_rows[inlier_pairs[:, 1]]
    psi, tx, ty = (float(v) for v in pose)
    cos, sin = np.cos(psi), np.sin(psi)
    rq = np.stack([cos * c[:, 0] - sin * c[:, 1], sin * c[:, 0] + cos * c[:, 1]], 1)
    residual = rq + np.array([tx, ty]) - e[:, :2]
    w = weight_ref / (e[:, 3] + c[:, 3])
    jac = np.zeros((k, 2, 3))
    jac[:, 0, 0] = jac[:, 1, 1] = 1.0
    jac[:, 0, 2], jac[:, 1, 2] = -rq[:, 1], rq[:, 0]
    normal = np.einsum("k,kai,kaj->ij", w, jac, jac)
    dof = 2.0 * k - 3.0
    scale = (w * (residual ** 2).sum(1)).sum() / dof
    scale = max(scale, RESIDUAL_SD_FLOOR_M ** 2 * w.mean())
    covariance = scale * np.linalg.pinv(normal)
    theta = np.array([tx, ty, psi])
    wald = float(theta @ (normal / scale) @ theta)
    support = 0
    if plan is not None:
        mass = np.exp(plan[:-1, :-1].astype(np.float64))
        best = mass.argmax(1)
        sharp = mass[np.arange(len(best)), best] >= SHARP_MASS
        support = int(sum(sharp[i] and best[i] == j for i, j in inlier_pairs))
    return FitStats(wald, dof, covariance, support, k)


def decide(stats: Optional[FitStats], verify_pose, propose_pose, propose_answered: bool, engaged: bool, *,
           level: float = 0.2, min_support: int = 3, propose_precision: Optional[np.ndarray] = None, verify_wald: bool = True) -> np.ndarray:
    """The coupled decision for one pair: Verify's pose, Propose's fallback, or no correction."""
    fallback = np.asarray(propose_pose, np.float32) if propose_answered else np.zeros(3, np.float32)
    if not engaged or stats is None:
        return fallback
    if verify_wald and stats.wald <= _threshold(max(stats.dof, abstain.DOF_FLOOR), level):
        return np.zeros(3, np.float32)
    if stats.support < min_support:
        return np.zeros(3, np.float32)
    if propose_precision is not None:
        dp = np.array([verify_pose[1] - propose_pose[1], verify_pose[2] - propose_pose[2],
                       np.arctan2(np.sin(verify_pose[0] - propose_pose[0]), np.cos(verify_pose[0] - propose_pose[0]))])
        combined = stats.covariance + np.linalg.pinv(propose_precision)
        if float(dp @ np.linalg.pinv(combined) @ dp) > _threshold(KNOWN_SCALE_DOF, level):
            return np.zeros(3, np.float32)
    return np.asarray(verify_pose, np.float32)
