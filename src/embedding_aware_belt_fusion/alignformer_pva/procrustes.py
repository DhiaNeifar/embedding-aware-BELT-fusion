"""Differentiable weighted SE(2) Procrustes, and the heading augmentation.

Given soft correspondences, this recovers the SE(2) transform in closed
form -- yaw is solved analytically here rather than learned, since a
regressed yaw tends to collapse to the conditional mean on this kind of data.

Augmenting each object with a second virtual point at
``centre + lam * heading`` folds box headings into the same least-squares
problem as centres. A useful consequence is that a *single* matched object
then determines the full SE(2), removing the collinearity degeneracy of
centre-only Procrustes.

That augmentation is only sound once the heading's pi ambiguity is resolved.
A vehicle bounding box is symmetric under a 180 degree rotation, and a
detector with no direction head fixes the box's *axis*, not which way it
points -- so two agents detecting the same physical object can report
headings ~180 degrees apart. Left unresolved, each such pair displaces a
heading virtual point by ``2 * lam``. :func:`heading_orientation` is the fold
that prevents this.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

# Total soft-match mass below which the correction is suppressed: less than
# one effective matched object carries no usable registration evidence.
MIN_MATCH_MASS = 1.0


def augment_with_heading(centres: Tensor, yaws: Tensor, lam: float) -> Tensor:
    """Append a heading virtual point per object.

    Parameters
    ----------
    centres: ``(B, N, 2)`` box centres.
    yaws: ``(B, N)`` box headings in radians.
    lam: offset in metres; about half a vehicle length makes headings and
        centres contribute comparably to the fit.

    Returns
    -------
    Tensor
        ``(B, 2N, 2)``, centres first then the corresponding heading tips, so
        a weight vector is extended by ``torch.cat([w, w], dim=1)``.
    """
    if centres.shape[:2] != yaws.shape:
        raise ValueError(
            f"centres {tuple(centres.shape)} and yaws {tuple(yaws.shape)} disagree"
        )
    direction = torch.stack([torch.cos(yaws), torch.sin(yaws)], dim=-1)
    return torch.cat([centres, centres + lam * direction], dim=1)


def heading_orientation(reference_yaws: Tensor, candidate_yaws: Tensor) -> Tensor:
    """``(B, M, N)`` of +1/-1 folding each candidate heading onto its reference's half-plane.

    ``-1`` marks a candidate pointing the other way along the same axis as the
    reference it is being compared against. Multiplying a candidate's unit
    heading vector by this sign before it is averaged or turned into a
    virtual point makes the heading channel invariant to the 180-degree
    ambiguity.

    The threshold is exactly perpendicular, so nothing within 90 degrees of
    the reference is ever moved. The sign is piecewise constant and carries no
    gradient of its own; gradients still flow through the heading vector it
    multiplies.

    Parameters
    ----------
    reference_yaws: ``(B, M)`` headings to fold onto, in radians.
    candidate_yaws: ``(B, N)`` headings to fold, in radians.
    """
    if reference_yaws.dim() != 2 or candidate_yaws.dim() != 2:
        raise ValueError(
            f"expected (B, M) and (B, N) yaws, got {tuple(reference_yaws.shape)} "
            f"and {tuple(candidate_yaws.shape)}"
        )
    alignment = torch.cos(candidate_yaws.unsqueeze(1) - reference_yaws.unsqueeze(2))
    return torch.where(
        alignment < 0, -torch.ones_like(alignment), torch.ones_like(alignment)
    )


def weighted_se2_kabsch(
    p: Tensor, q: Tensor, w: Tensor, *, eps: float = 1e-6
) -> Tuple[Tensor, Tensor]:
    """Closed-form weighted SE(2) fit taking ``q`` onto ``p``.

    Minimizes ``sum_n w_n |R(psi) q_n + t - p_n|^2``.

    Parameters
    ----------
    p: ``(B, N, 2)`` target points (the ego's own objects).
    q: ``(B, N, 2)`` source points (CAV objects projected with the noisy pose).
    w: ``(B, N)`` non-negative correspondence weights.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``psi`` of shape ``(B,)`` in radians, and ``t`` of shape ``(B, 2)``.
        Degenerate inputs (too little evidence, or evidence that pins
        translation but not rotation -- e.g. a single matched point) yield
        the identity transform with finite gradients, rather than guessing a
        rotation an under-determined fit can't actually support.
    """
    if p.shape != q.shape:
        raise ValueError(f"p {tuple(p.shape)} and q {tuple(q.shape)} must match")
    if w.shape != p.shape[:2]:
        raise ValueError(f"w {tuple(w.shape)} must be {tuple(p.shape[:2])}")

    w = w.clamp_min(0.0)
    mass = w.sum(dim=1, keepdim=True)
    safe_mass = mass.clamp_min(eps)

    weights = w.unsqueeze(-1)
    p_bar = (weights * p).sum(dim=1) / safe_mass
    q_bar = (weights * q).sum(dim=1) / safe_mass

    dp = p - p_bar.unsqueeze(1)
    dq = q - q_bar.unsqueeze(1)
    cross = (w * (dq[..., 0] * dp[..., 1] - dq[..., 1] * dp[..., 0])).sum(dim=1)
    dot = (w * (dq[..., 0] * dp[..., 0] + dq[..., 1] * dp[..., 1])).sum(dim=1)

    # atan2(0, 0) is finite but its gradient is not. Route degenerate entries
    # through a constant branch so no NaN can reach the optimizer.
    magnitude = torch.sqrt(cross * cross + dot * dot)
    resolvable = magnitude > eps
    cross_safe = torch.where(resolvable, cross, torch.zeros_like(cross))
    dot_safe = torch.where(resolvable, dot, torch.ones_like(dot))
    psi = torch.atan2(cross_safe, dot_safe)

    cos, sin = torch.cos(psi), torch.sin(psi)
    rotated_q_bar = torch.stack(
        [cos * q_bar[:, 0] - sin * q_bar[:, 1], sin * q_bar[:, 0] + cos * q_bar[:, 1]],
        dim=-1,
    )
    t = p_bar - rotated_q_bar

    # Suppress the correction entirely when there is too little match evidence.
    usable = (mass.squeeze(1) >= MIN_MATCH_MASS) & resolvable
    psi = torch.where(usable, psi, torch.zeros_like(psi))
    t = torch.where(usable.unsqueeze(-1), t, torch.zeros_like(t))
    return psi, t
