"""Per-pair abstention from the estimator's OWN covariance.

At low localization error the correct action is to not correct, and head B
corrects anyway. Two guards already exist and neither has the right shape.
``procrustes.MIN_MATCH_MASS`` suppresses on **evidence volume** -- fewer than
one effective matched object -- and never asks whether the *estimated
correction* is meaningful. Positive-part shrinkage (:mod:`alignformer.shrinkage`)
does ask, but through a single global ``tau`` fitted once at sigma = 0: it can
only scale the whole population by one number, and a dense twelve-correspondence
pair and a sparse two-correspondence pair do not have the same estimator
variance. Task 23 measured exactly that dead end -- refitting the global tau
bought the low-noise end and gave back the clean case, because a smaller tau
shrinks less *everywhere*.

This module is the per-pair replacement. Nothing is learned and nothing new is
fitted: every quantity it uses is one the weighted solve and the fitted
disagreement-variance model (:mod:`alignformer.variance`) already compute.

The statistic
-------------

Head B minimizes ``sum_n w_n |R(psi) q_n + t - p_n|^2`` over
``theta = (t_x, t_y, psi)``. Writing ``u_n = R(psi) q_n``, the residual's
Jacobian is ``J_n = [I_2 | perp(u_n)]`` with ``perp(a, b) = (-b, a)``, so the
weighted normal matrix is ``A = sum_n w_n J_n^T J_n``.

``sigma^2 A^-1`` is **not** this estimator's covariance, and using it was a real
error in an earlier draft of this module. The points are heading-augmented:
:func:`procrustes.augment_with_heading` puts object ``m``'s second point at
``centre_m + lam * u(psi_m)``, whose residual is the object's centre error
**plus** ``lam`` times its heading error. The two points of one object share
their entire centre error -- at this project's own fitted constants the
correlation is about 0.84 -- so treating ``2M`` points as ``2M`` independent
observations overstates the information, understates the covariance, and makes
the test fire far more often than any nominal level would suggest. It does so
*worst where the evidence is thinnest*, which inverts the very behaviour this
module exists to produce.

So the covariance is the **sandwich**, which is what the task brief allows for
exactly this case::

    Cov(theta) = A^-1 B A^-1,
    B = sum_n sum_n' w_n w_n' J_n^T Cov(eps_n, eps_n') J_n'

and the noise covariance is not guessed -- it is the fitted model. For object
``m`` with per-axis centre variance ``a_m``, angular variance ``g_m`` and unit
heading direction ``d_m``::

    Cov(eps_c, eps_c) = a_m I          Cov(eps_c, eps_h) = a_m I
    Cov(eps_h, eps_h) = a_m I + lam^2 g_m perp(d_m) perp(d_m)^T

Substituting collapses ``B`` to two rank-limited terms per object, with no
cross-object coupling::

    B = sum_m [ a_m G_m^T G_m  +  lam^2 g_m w_h,m^2 c_m c_m^T ]
    G_m = w_c,m J_c,m + w_h,m J_h,m = [ (w_c + w_h) I | w_c perp(u_c) + w_h perp(u_h) ]
    c_m = J_h,m^T perp(d_m) = [ perp(d_m) ; perp(u_h,m) . perp(d_m) ]

``d_m`` is recovered from the points themselves -- the heading point sits at
``centre + lam * d``, so ``d = (u_h - u_c) / lam`` -- and ``a_m``, ``g_m`` come
from :meth:`variance.CorrespondenceVarianceModel.correspondence_variances`,
which is the same model the solve's own weights are built from.

The Wald statistic of the emitted correction against the null "no correction is
needed" is then, with no matrix inverse of ``A`` ever formed::

    T^2 = theta^T Cov^-1 theta = (A theta)^T B^-1 (A theta) / tau^2

``tau^2`` is the one scale the model does not supply: the unitless factor by
which the real residuals exceed the model's predicted ones, estimated per pair
as ``sum_n w_n |r_n|^2 / v_n`` over the model's own expectation for that sum
(:func:`expected_residual_sum`) rather than over a count of points. The count
is wrong by a factor of two at two matched objects.

The reference
-------------

``tau^2`` is estimated, not known, so ``T^2 / p`` is referred to ``F(p, D)``
rather than to ``chi-square(p) / p``. Its denominator degrees of freedom are
``D = 3 n_eff_objects - 3``: the residual of one object spans **three**
independent noise dimensions -- two centre, one heading -- not four, however
many augmented points it contributes. ``n_eff`` is **Kish's** effective sample
size throughout, so a row whose soft-match mass sits almost entirely on one
candidate counts as roughly one observation and not as the eight it has entries
for.

Together these corrections are what make the level mean the same thing at two
matched objects as at twenty. Measured by simulation on the deployed geometry
with the fitted r140 constants, at a nominal 5%, the realized false-fire rate
is::

    matched objects      2      3      4      6     12     24
    naive covariance  .637      -      -      -      -   .266
    this module       .075   .070   .068   .061   .057   .045

That is not a detail: sigma = 0 on the sparse slice is the single cell this
whole intervention exists to fix, and a statistic that over-fires by 13x
exactly where the evidence is thinnest inverts the behaviour the module is
justified by. The simulation is a test
(``test_the_level_holds_at_every_matched_object_count``), not a footnote.

**Scale invariance.** ``weighted_se2_kabsch`` is scale-invariant in its weights,
and both :func:`variance.augmented_weights` and :func:`robust._renormalized`
rescale them. ``T^2`` inherits that: ``A`` is linear in the weights and ``B``
quadratic, so the quadratic form is invariant, and the weights are normalized to
``sum w = n_eff`` only to give the residual floor a meaning in metres.

The three decisions
-------------------

``abstain``
    Zero the correction at or below the level's threshold, apply it unchanged
    above.
``per_pair``
    The James-Stein positive-part factor ``max(0, 1 - p / T^2)`` -- the SAME
    rule :mod:`alignformer.shrinkage` applies, with the pair's own standardized
    statistic in place of the global-tau one. :func:`shrinkage.shrinkage_factor`
    is reused rather than re-derived, so the two cannot drift apart.
``both``
    A hard threshold, then the positive-part factor above it.

``both`` at level 1.0 -- a threshold of zero -- IS ``per_pair``. That
equivalence is asserted in the tests so the three arms cannot silently become
two.

An abstained pair emits ``(psi, t) = (0, 0)``, which is what
:func:`stage2.is_fallback` reads, so **coverage falls out of the existing pose
statistics** and can be reported beside every AP number. It has to be: an
abstaining method answers less often and therefore looks better on what it does
answer, which is the reporting trap this project criticized in the published
competitor's own numbers and which now applies to us.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from embedding_aware_belt_fusion.alignformer_pva.shrinkage import (
    POSE_DIMENSIONS,
    shrinkage_factor,
)

# ``none`` is the deployed behaviour: no per-pair decision at all, so an arm
# configured with it reproduces what it wraps bit for bit.
NONE = "none"
ABSTAIN = "abstain"
PER_PAIR = "per_pair"
BOTH = "both"
# The per-pair rule's per-axis generalization. ``per_pair`` applies one scalar
# to all of SE(2); when two directions are well determined and a third is not
# -- the far-pair geometry, where all the matched objects lie in one narrow
# angular wedge -- that scalar has to either keep the unidentified direction's
# runaway or discard the two good directions with it. This mode shrinks in the
# estimator's own principal axes instead, at ``p = 1`` per axis.
DIRECTIONAL = "directional"
ABSTENTION_MODES = (NONE, ABSTAIN, PER_PAIR, BOTH, DIRECTIONAL)

# Modes whose factor is not one number, so ``decision_factor`` cannot express
# them and says so rather than returning something plausible.
_SCALAR_MODES = (NONE, ABSTAIN, PER_PAIR, BOTH)

# Lower bound on the fitted per-axis residual sd, in metres. Same number and
# same reason as ``robust.SCALE_FLOOR_M``: the detector's own fitted
# per-detection RMS centre error is 0.2516 m (``variance.py``), so nothing real
# is resolved at 5 cm, and a fit that happens to be exact must not divide by
# zero.
RESIDUAL_SD_FLOOR_M = 0.05
# Floor on the residual degrees of freedom. Below about two matched objects the
# SE(2) is barely identified; such a pair gets a large variance, a small
# statistic and a large threshold, and therefore abstains -- which is the
# intended answer, not a guard against one.
DOF_FLOOR = 1.0
# Relative ridge on the sandwich's middle matrix, as a fraction of its mean
# eigenvalue. One matched object cannot determine three parameters, so B is
# rank-deficient there; the ridge keeps the solve finite and, being additive on
# the variance, can only make the pair abstain more.
SANDWICH_RIDGE = 1e-6


@dataclass(frozen=True)
class AbstentionConfig:
    """One per-pair decision rule, and the name the sweep reports it under.

    Parameters
    ----------
    mode: one of :data:`ABSTENTION_MODES`.
    level: the false-fire rate the hard threshold targets when there is nothing
        to correct -- ``0.05`` abstains on a pair whose statistic is below the
        95th percentile of its own null. It is a *level*, not a raw statistic,
        because the threshold depends on the pair's own degrees of freedom
        (:func:`abstention_threshold`) and a single number could not mean the
        same thing at two matched objects as at twenty. Meaningful for
        ``abstain`` and ``both`` only; ``per_pair`` refuses one rather than
        ignoring it, because a constant silently dropped is a constant a reader
        would believe was in force.
    dimensions: the pooled problem's dimension, in the reference distribution
        and in the positive-part rule.
    """

    mode: str = NONE
    level: float = 0.0
    dimensions: int = POSE_DIMENSIONS

    def __post_init__(self) -> None:
        if self.mode not in ABSTENTION_MODES:
            raise ValueError(
                f"mode must be one of {ABSTENTION_MODES}, got {self.mode!r}"
            )
        if not (math.isfinite(self.level) and 0.0 <= self.level <= 1.0):
            raise ValueError(f"level must be in [0, 1], got {self.level}")
        if self.mode in (NONE, PER_PAIR, DIRECTIONAL) and self.level != 0.0:
            raise ValueError(
                f"mode {self.mode!r} has no use for a level, got {self.level}"
            )
        if self.mode in (ABSTAIN, BOTH) and self.level == 0.0:
            raise ValueError(
                f"mode {self.mode!r} needs a level in (0, 1]; a level of 0 would "
                "put the threshold at infinity and abstain on everything"
            )
        if self.dimensions < 1:
            raise ValueError(f"dimensions must be at least 1, got {self.dimensions}")
    @property
    def enabled(self) -> bool:
        return self.mode != NONE

    @property
    def name(self) -> str:
        """The sweep condition key: ``alignformer_<mode>[_<level>]``."""
        if self.mode in (PER_PAIR, DIRECTIONAL):
            return f"alignformer_{self.mode}"
        return f"alignformer_{self.mode}_{self.level:g}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "level": self.level,
            "dimensions": self.dimensions,
            "name": self.name,
            "enabled": self.enabled,
        }

    @classmethod
    def parse(cls, spec: str) -> "AbstentionConfig":
        """``"abstain:0.05"``, ``"both:0.01"`` or ``"per_pair"``."""
        mode, separator, level = str(spec).partition(":")
        if separator == "":
            return cls(mode=mode)
        try:
            value = float(level)
        except ValueError as error:
            raise ValueError(
                f"{spec!r} is not a <mode>[:<level>] abstention spec"
            ) from error
        return cls(mode=mode, level=value)


def effective_sample_size(weights: Tensor) -> Tensor:
    """``(B,)`` Kish effective sample size ``(sum w)^2 / sum w^2``.

    Equal to the count when every weight is equal, and collapsing towards one
    when a single entry carries the mass. A row with no weight at all returns
    zero, which floors the degrees of freedom downstream instead of dividing by
    nothing.
    """
    weights = weights.clamp_min(0.0)
    total = weights.sum(dim=1)
    squared = (weights * weights).sum(dim=1)
    tiny = torch.finfo(weights.dtype).tiny
    return torch.where(
        squared > 0, total * total / squared.clamp_min(tiny), torch.zeros_like(total)
    )


def _normalized(weights: Tensor) -> Tensor:
    """``weights`` rescaled to total ``n_eff``, so the mean weight is one."""
    weights = weights.clamp_min(0.0)
    total = weights.sum(dim=1, keepdim=True)
    tiny = torch.finfo(weights.dtype).tiny
    scale = torch.where(
        total > 0,
        effective_sample_size(weights).unsqueeze(-1) / total.clamp_min(tiny),
        torch.zeros_like(total),
    )
    return weights * scale


def residual_scale(
    residual: Tensor, weights: Tensor, point_variance: Tensor, divisor: Tensor
) -> Tensor:
    """``(B,)`` ``tau^2``: how far the real residuals exceed the model's.

    ``point_variance`` is ``E|r_n|^2`` under the fitted disagreement model, so a
    perfectly-specified model gives ``tau^2 = 1``.

    ``divisor`` is :func:`expected_residual_sum`: the model's own expectation
    for the numerator, computed from the fit rather than counted off the point
    list. A count cannot be right here -- an object's two augmented rows carry
    three independent noise coordinates, not four, and the weighted hat
    operator removes a different amount of them at two matched objects than at
    twenty.

    Floored so that the implied per-axis residual sd cannot fall below
    :data:`RESIDUAL_SD_FLOOR_M`, which is where the floor keeps its meaning in
    metres.
    """
    normalized = _normalized(weights)
    standardized = (residual * residual).sum(dim=-1) / point_variance.clamp_min(
        torch.finfo(residual.dtype).tiny
    )
    total = (normalized * standardized).sum(dim=1)

    mass = normalized.sum(dim=1).clamp_min(torch.finfo(residual.dtype).tiny)
    mean_variance = (normalized * point_variance).sum(dim=1) / mass
    floor = 2.0 * RESIDUAL_SD_FLOOR_M ** 2 / mean_variance.clamp_min(
        torch.finfo(residual.dtype).tiny
    )
    return (total / divisor.clamp_min(DOF_FLOOR)).clamp_min(floor)


@lru_cache(maxsize=None)
def _f_quantile_table(level: float, dimensions: int) -> Tuple[Any, Any]:
    """``(dof grid, p * F_{1-level}(p, dof))``, built once per (level, p).

    A table rather than a call per pair because the quantile is needed inside
    the sweep's inner loop and depends only on a scalar; interpolating a
    120-point grid is exact to well under a per cent over the range of effective
    object counts this dataset produces.
    """
    from scipy.stats import f as f_distribution

    grid = np.concatenate(
        [
            np.linspace(DOF_FLOOR, 12.0, 60),
            np.geomspace(12.0, 4000.0, 60)[1:],
        ]
    )
    values = dimensions * f_distribution.ppf(1.0 - level, dimensions, grid)
    return grid, values


def abstention_threshold(dof: Tensor, config: "AbstentionConfig") -> Tensor:
    """``(B,)`` the statistic at or below which this rule abstains.

    ``p * F_{1-level}(p, dof)`` -- the reference for a Wald statistic whose
    scale was estimated rather than known. At large ``dof`` it converges to the
    chi-square quantile; at the two or three matched objects where this dataset
    is thinnest it is several times larger, which is the whole point.
    """
    grid, values = _f_quantile_table(float(config.level), int(config.dimensions))
    reference = torch.as_tensor(grid, dtype=dof.dtype, device=dof.device)
    quantiles = torch.as_tensor(values, dtype=dof.dtype, device=dof.device)
    clamped = dof.clamp(float(grid[0]), float(grid[-1]))
    upper = torch.searchsorted(reference, clamped.contiguous()).clamp(1, len(grid) - 1)
    lower = upper - 1
    span = (reference[upper] - reference[lower]).clamp_min(
        torch.finfo(dof.dtype).tiny
    )
    weight = (clamped - reference[lower]) / span
    return quantiles[lower] + weight * (quantiles[upper] - quantiles[lower])


def _perp(vectors: Tensor) -> Tensor:
    """``(-y, x)`` on the last dimension."""
    return torch.stack([-vectors[..., 1], vectors[..., 0]], dim=-1)


def _normal_matrix(rotated: Tensor, weights: Tensor) -> Tensor:
    """``(B, 3, 3)`` ``A = sum_n w_n J_n^T J_n`` for ``J_n = [I | perp(u_n)]``."""
    perpendicular = _perp(rotated)
    total = weights.sum(dim=1)
    cross = (weights.unsqueeze(-1) * perpendicular).sum(dim=1)
    quadratic = (weights * (rotated * rotated).sum(dim=-1)).sum(dim=1)

    matrix = rotated.new_zeros(rotated.shape[0], 3, 3)
    matrix[:, 0, 0] = total
    matrix[:, 1, 1] = total
    matrix[:, :2, 2] = cross
    matrix[:, 2, :2] = cross
    matrix[:, 2, 2] = quadratic
    return matrix


def _middle(
    rotated: Tensor,
    weights_left: Tensor,
    weights_right: Tensor,
    variance_centre: Tensor,
    variance_heading: Tensor,
    heading_lambda: float,
) -> Tensor:
    """``(B, 3, 3)`` ``sum_n sum_n' wl_n wr_n' J_n^T Cov(eps_n, eps_n') J_n'``.

    Two terms per object and no cross-object coupling; see the module docstring
    for the collapse. ``variance_centre`` is the 2-D total, so the per-axis
    centre variance is half of it, and ``variance_heading - variance_centre`` is
    the lever-armed angular term ``lam^2 g_m``.

    Two weight vectors rather than one because the residual-scale correction
    needs the same quantity between the solve's weights and the standardizing
    ones; with ``weights_left is weights_right`` this is the symmetric sandwich
    filling ``B``.
    """
    count = rotated.shape[1] // 2
    centre, heading = rotated[:, :count], rotated[:, count:]
    perpendicular_centre, perpendicular_heading = _perp(centre), _perp(heading)

    def _blocks(weights):
        weight_centre, weight_heading = weights[:, :count], weights[:, count:]
        scale = weight_centre + weight_heading
        lever = (
            weight_centre.unsqueeze(-1) * perpendicular_centre
            + weight_heading.unsqueeze(-1) * perpendicular_heading
        )
        return scale, lever, weight_heading

    scale_left, lever_left, heading_left = _blocks(weights_left)
    scale_right, lever_right, heading_right = _blocks(weights_right)

    # G_l^T G_r for G = [s I | v]: [[s_l s_r I, s_l v_r], [s_r v_l^T, v_l . v_r]]
    per_axis = 0.5 * variance_centre
    shared = rotated.new_zeros(rotated.shape[0], 3, 3)
    shared[:, 0, 0] = (per_axis * scale_left * scale_right).sum(dim=1)
    shared[:, 1, 1] = shared[:, 0, 0]
    shared[:, :2, 2] = ((per_axis * scale_left).unsqueeze(-1) * lever_right).sum(dim=1)
    shared[:, 2, :2] = ((per_axis * scale_right).unsqueeze(-1) * lever_left).sum(dim=1)
    shared[:, 2, 2] = (per_axis * (lever_left * lever_right).sum(dim=-1)).sum(dim=1)

    # The heading point's own angular term, rank one per object:
    # c_m = [ perp(d_m) ; perp(u_h,m) . perp(d_m) ].
    direction = (heading - centre) / max(heading_lambda, torch.finfo(rotated.dtype).eps)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(rotated.dtype).tiny
    )
    normal = _perp(direction)
    projection = (perpendicular_heading * normal).sum(dim=-1)
    vector = torch.cat([normal, projection.unsqueeze(-1)], dim=-1)
    angular = (variance_heading - variance_centre).clamp_min(0.0) * (
        heading_left * heading_right
    )
    return shared + torch.einsum("bm,bmi,bmj->bij", angular, vector, vector)


def sandwich_middle(
    rotated: Tensor,
    weights: Tensor,
    variance_centre: Tensor,
    variance_heading: Tensor,
    heading_lambda: float,
) -> Tensor:
    """``(B, 3, 3)`` ``B = sum_n sum_n' w w' J^T Cov(eps, eps') J``."""
    return _middle(
        rotated, weights, weights, variance_centre, variance_heading, heading_lambda
    )


def expected_residual_sum(
    rotated: Tensor,
    weights: Tensor,
    variance_centre: Tensor,
    variance_heading: Tensor,
    heading_lambda: float,
    normal_matrix: Tensor,
    middle: Tensor,
) -> Tensor:
    """``(B,)`` ``E[sum_n w_n |r_n|^2 / v_n]`` under the model, at ``tau^2 = 1``.

    The divisor a mean-unbiased ``tau^2`` needs, computed rather than counted.
    With ``H = J A^-1 J^T W`` the weighted hat operator, ``r = (I - H) eps`` and
    ``D = diag(w_n / v_n)``::

        E = tr(D Sigma) - 2 tr(A^-1 J^T W Sigma D J) + tr(A^-1 B A^-1 J^T D J)

    ``tr(D Sigma)`` is exactly ``sum_n w_n``, because ``v_n`` is by definition
    ``E|eps_n|^2``. The other two terms are 3x3 traces over quantities this
    module already forms.

    Counting instead of computing is what a naive ``2 n_eff - 3`` does, and it
    is wrong in the direction that matters: measured on the deployed geometry
    the true expectation at two matched objects is **half** the counted one, so
    the counted version halves ``tau^2``, doubles the statistic, and over-fires
    exactly where the evidence is thinnest.
    """
    point_variance = torch.cat([variance_centre, variance_heading], dim=1)
    standardizing = weights / point_variance.clamp_min(
        torch.finfo(weights.dtype).tiny
    )
    cross = _middle(
        rotated, weights, standardizing, variance_centre, variance_heading,
        heading_lambda,
    )
    curvature = _normal_matrix(rotated, standardizing)

    inverse = torch.linalg.pinv(normal_matrix)
    first = torch.matmul(inverse, cross).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    second = (
        torch.matmul(torch.matmul(torch.matmul(inverse, middle), inverse), curvature)
        .diagonal(dim1=-2, dim2=-1)
        .sum(dim=-1)
    )
    return weights.sum(dim=1) - 2.0 * first + second


def solve_or_zero(matrix: Tensor, rhs: Tensor) -> Tensor:
    """``matrix^-1 rhs`` per batch element, zeros where there is no such thing.

    A pair whose sandwich middle is exactly zero -- every correspondence
    weight zero, so no evidence at all -- gets a ridge of ``SANDWICH_RIDGE *
    tiny``, a denormal that the GPU flushes to zero, and ``linalg.solve`` then
    raises on the singular matrix (the V2X-Real test sweep, 2026-09-29). A
    non-finite middle is the same situation by another route. Both mean the
    pair has nothing to say, and ``T^2 = 0`` is what nothing-to-say scores;
    every invertible element goes through the same LU as ``linalg.solve``.
    """
    finite = torch.isfinite(matrix).flatten(1).all(dim=1)
    identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    safe = torch.where(finite.reshape(-1, 1, 1), matrix, identity.expand_as(matrix))
    solution, info = torch.linalg.solve_ex(safe, rhs)
    usable = (finite & (info == 0)).reshape(-1, 1, 1)
    return torch.where(usable, solution, torch.zeros_like(solution))


class WaldParts(NamedTuple):
    """The Wald test, plus the axes its statistic is a sum over.

    ``statistic`` and ``dof`` are exactly what :func:`wald_statistic` has
    always returned. ``precision`` is the estimator's inverse covariance
    ``M = A B^-1 A / tau^2``, of which the statistic is the quadratic form
    ``theta^T M theta`` -- so a rule that reads ``M`` is a refinement of the
    deployed one and not a second, uncalibrated statistic.

    The eigenbasis is deliberately NOT computed here. ``M`` mixes metres and
    radians, so its axes only mean something after whitening by a prior with
    those units, and the prior belongs to the decision rule
    (:class:`AbstentionConfig`) rather than to the fit.
    """

    statistic: Tensor
    dof: Tensor
    precision: Tensor


def wald_parts(
    p: Tensor,
    q: Tensor,
    weights: Tensor,
    psi: Tensor,
    t: Tensor,
    *,
    variance_centre: Tensor,
    variance_heading: Tensor,
    heading_lambda: float,
) -> WaldParts:
    """The Wald statistic of the emitted correction, resolved onto its axes.

    Parameters
    ----------
    p, q: ``(B, 2M, 2)`` target and source points, exactly as
        :func:`procrustes.weighted_se2_kabsch` took them -- centres first, then
        the heading virtual points.
    weights: ``(B, 2M)`` the weights the FINAL solve used. For the IRLS arm that
        is the re-weighted vector (:attr:`robust.RobustSolution.weights`), not
        the one it started from: the covariance belongs to the fit that was
        actually reported.
    variance_centre, variance_heading: ``(B, M)`` from
        :meth:`variance.CorrespondenceVarianceModel.correspondence_variances`.
    psi, t: the emitted correction, ``(B,)`` and ``(B, 2)``.

    ``dof`` is ``3 n_eff_objects - 3``, the reference distribution's denominator.
    An identity correction scores exactly zero, which abstains under every rule.
    """
    if p.shape != q.shape:
        raise ValueError(f"p {tuple(p.shape)} and q {tuple(q.shape)} must match")
    if weights.shape != p.shape[:2]:
        raise ValueError(f"weights {tuple(weights.shape)} must be {tuple(p.shape[:2])}")
    if p.shape[1] % 2 != 0:
        raise ValueError(
            f"expected heading-augmented points (an even count), got {p.shape[1]}"
        )
    count = p.shape[1] // 2
    if variance_centre.shape != (p.shape[0], count):
        raise ValueError(
            f"variance_centre {tuple(variance_centre.shape)} must be "
            f"{(p.shape[0], count)}"
        )
    if variance_heading.shape != variance_centre.shape:
        raise ValueError(
            f"variance_heading {tuple(variance_heading.shape)} must match "
            f"variance_centre {tuple(variance_centre.shape)}"
        )

    normalized = _normalized(weights)
    cos, sin = torch.cos(psi).unsqueeze(-1), torch.sin(psi).unsqueeze(-1)
    x, y = q[..., 0], q[..., 1]
    rotated = torch.stack([cos * x - sin * y, sin * x + cos * y], dim=-1)
    residual = rotated + t.unsqueeze(1) - p

    # Objects, not points: the residual of ONE object spans three independent
    # noise dimensions however many augmented points it contributes.
    object_weights = normalized[:, :count] + normalized[:, count:]
    dof = (
        POSE_DIMENSIONS * effective_sample_size(object_weights) - POSE_DIMENSIONS
    ).clamp_min(DOF_FLOOR)

    theta = torch.cat([t, psi.reshape(-1, 1)], dim=-1).unsqueeze(-1)
    curvature = _normal_matrix(rotated, normalized)
    score = torch.matmul(curvature, theta)
    middle = sandwich_middle(
        rotated, normalized, variance_centre, variance_heading, heading_lambda
    )

    point_variance = torch.cat([variance_centre, variance_heading], dim=1)
    tau_squared = residual_scale(
        residual,
        normalized,
        point_variance,
        expected_residual_sum(
            rotated, normalized, variance_centre, variance_heading,
            heading_lambda, curvature, middle,
        ),
    )
    trace = middle.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / POSE_DIMENSIONS
    ridge = torch.eye(3, dtype=middle.dtype, device=middle.device) * (
        SANDWICH_RIDGE * trace.clamp_min(torch.finfo(middle.dtype).tiny)
    ).reshape(-1, 1, 1)
    quadratic = torch.matmul(
        score.transpose(-1, -2), solve_or_zero(middle + ridge, score)
    ).reshape(-1)
    statistic = (quadratic.clamp_min(0.0) / tau_squared).nan_to_num(
        nan=0.0, posinf=0.0, neginf=0.0
    )

    # The same quadratic form as a matrix: `statistic == theta^T M theta`, so
    # a rule that reads `M` inherits every calibration argument made for the
    # scalar one. The symmetrization is not cosmetic -- `linalg.solve` leaves
    # an asymmetry of order the condition number, and a large condition number
    # is precisely the case this exists for.
    precision = torch.matmul(curvature, solve_or_zero(middle + ridge, curvature))
    precision = 0.5 * (precision + precision.transpose(-1, -2))
    precision = (precision / tau_squared.reshape(-1, 1, 1)).nan_to_num(
        nan=0.0, posinf=0.0, neginf=0.0
    )
    return WaldParts(statistic=statistic, dof=dof, precision=precision)


def wald_statistic(*args, **kwargs) -> Tuple[Tensor, Tensor]:
    """``(T^2, dof)`` -- :func:`wald_parts` without the per-axis decomposition.

    Kept as the module's original entry point, with its original signature and
    its original values, so the shipped ``per_pair`` arm reproduces bit for bit
    in the very run that measures the directional one.
    """
    parts = wald_parts(*args, **kwargs)
    return parts.statistic, parts.dof


def directional_theta(theta: Tensor, precision: Tensor) -> Tensor:
    """``(B, 3)`` the correction shrunk per PRINCIPAL AXIS, not by one factor.

    Parameters
    ----------
    theta: ``(B, 3)`` the emitted correction as ``[t_x, t_y, psi]``, the
        ordering :func:`wald_parts` forms its quadratic in.
    precision: ``(B, 3, 3)`` ``M`` from :class:`WaldParts`.

    The rule is one ridge factor per eigenvalue of ``M``::

        theta' = M (M + c I)^-1 theta,     c = p det(M)^(1/3) / T^2

    and it introduces **no constant**: ``p`` is the three the deployed rule
    already uses and ``c`` is built from the fit's own invariants. An axis the
    pair cannot identify has ``mu << det(M)^(1/3)`` and is crushed; the axes
    the same pair determines well have ``mu >> c`` and pass through. On an
    isotropic fit every axis takes the single factor ``T^2 / (T^2 + p)``, so
    with nothing anisotropic to see the rule does not invent a distinction.

    Why the factor may depend on ``mu`` and never on the components
    --------------------------------------------------------------

    Because this is a **spectral matrix function**, ``F(M) = M (M + cI)^-1``,
    two properties hold that nothing else tried here had.

    *Rotation invariance.* Equal eigenvalues take equal factors, so a
    degenerate block acts as a scalar. That is not academic: on the
    well-conditioned pairs that are the majority of the population the two
    translation axes sit at ``mu_2 / mu_1 = 1.09`` to ``1.15``, near enough to
    degenerate that the eigenbasis is arbitrary. A per-COMPONENT rule --
    ``max(0, 1 - 1/z_i^2)`` on each axis, which was the formulation tried
    before this one -- is not invariant there, and measurably so: on an
    isotropic ``M`` it returned ``(0.565, -1.568, 0.000)`` in one basis and
    ``(0.549, -1.340, 0.794)`` in a rotated one, for the same fit. An answer
    that depends on which eigenvectors LAPACK happened to return is not an
    answer.

    *Continuity.* ``mu / (mu + c)`` is smooth and bounded, so ``F`` is Lipschitz
    in ``M`` where a positive part is not.

    Two other forms were measured and rejected. Per-axis ``max(0, 1 - p/z^2)``
    alone throws away the pooling that makes James-Stein work and emitted MORE
    spurious correction than the scalar rule at sigma = 0 (1.35x on a
    ten-object wedge, 1.69x on an open scatter). The posterior mean under
    ``shrinkage``'s fitted tau is the best rule at sigma = 0 by a wide margin
    and unusable anyway: that tau was calibrated at sigma = 0, and against a
    true two-metre correction a prior saying "about 0.15 m" turned a 0.116 m
    error into 0.967 m. That is task 23's dead end from the other side, and it
    is why the scale here is taken from the fit rather than from a constant.

    **Known limitation, measured.** On the ill-conditioned pairs this exists
    for, ``cond(M)`` runs to ``1e6``, and there the output moves under
    perturbations of ``M`` that the well-conditioned pairs do not feel
    (``1e-1`` m against ``1e-4`` m for a relative nudge of ``1e-6``). That
    sensitivity is the SOLVE's, not this rule's -- a fit whose normal matrix
    has a condition number of a million does not determine its own weak
    direction either -- but it does mean the gain here is a population
    average and not a per-pair guarantee. Recorded rather than papered over.
    """
    if theta.shape[-1] != POSE_DIMENSIONS:
        raise ValueError(
            f"theta {tuple(theta.shape)} must end in {POSE_DIMENSIONS}"
        )
    expected = theta.shape[:-1] + (POSE_DIMENSIONS, POSE_DIMENSIONS)
    if precision.shape != expected:
        raise ValueError(f"precision {tuple(precision.shape)} must be {expected}")

    tiny = torch.finfo(theta.dtype).tiny
    symmetric = 0.5 * (precision + precision.transpose(-1, -2))
    evidence, basis = torch.linalg.eigh(symmetric)
    evidence = evidence.clamp_min(0.0)

    statistic = torch.einsum("bi,bij,bj->b", theta, symmetric, theta).clamp_min(0.0)
    # The geometric mean is det(M)^(1/3) computed in the log domain: the
    # eigenvalues here span six orders of magnitude and their product
    # underflows single precision long before their sum of logs does.
    typical = torch.exp(torch.log(evidence.clamp_min(tiny)).mean(dim=-1))
    ridge = (
        POSE_DIMENSIONS * typical / statistic.clamp_min(tiny)
    ).unsqueeze(-1)

    factor = evidence / (evidence + ridge).clamp_min(tiny)
    components = torch.matmul(
        basis.transpose(-1, -2), theta.unsqueeze(-1)
    ).squeeze(-1)
    return torch.matmul(
        basis, (factor * components).unsqueeze(-1)
    ).squeeze(-1).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)


def decision_factor(
    statistic: Tensor, dof: Optional[Tensor], config: AbstentionConfig
) -> Tensor:
    """``(B,)`` multiplier in ``[0, 1]`` this rule applies to the correction.

    One factor for the whole SE(2), exactly as :func:`shrinkage.shrink` does, so
    the result always lies on the path between the identity and the correction
    the solver proposed.
    """
    if not config.enabled:
        return torch.ones_like(statistic)
    if config.mode not in _SCALAR_MODES:
        raise ValueError(
            f"mode {config.mode!r} does not act by one factor on the whole "
            "SE(2) -- that is the point of it -- so it has no decision factor; "
            "use abstain.decide, which rewrites the correction itself"
        )

    if config.mode == PER_PAIR:
        return shrinkage_factor(statistic, config.dimensions)

    if dof is None:
        raise ValueError(
            f"mode {config.mode!r} needs the fit's degrees of freedom to place "
            "its threshold; solve with statistic=True"
        )
    # Inclusive at the threshold: an exact hit abstains. A rule stated as "at or
    # below" has to be implemented as one, or the constant a report quotes is
    # not the constant that ran.
    answered = (statistic > abstention_threshold(dof, config)).to(statistic.dtype)
    if config.mode == ABSTAIN:
        return answered
    return shrinkage_factor(statistic, config.dimensions) * answered


def decide(estimate, config: AbstentionConfig):
    """Return a **new** ``PoseEstimate`` with this pair's decision applied.

    A disabled configuration returns the input object itself, so an arm meant to
    reproduce what it wraps does so by identity rather than by arithmetic that
    happens to be a no-op.
    """
    if not config.enabled:
        return estimate
    statistic: Optional[Tensor] = getattr(estimate, "offset_statistic", None)
    if statistic is None:
        raise ValueError(
            "this estimate carries no offset statistic; solve it with "
            "solve_pose(..., statistic=True) before asking for a decision"
        )

    if config.mode == DIRECTIONAL:
        precision: Optional[Tensor] = getattr(estimate, "offset_precision", None)
        if precision is None:
            raise ValueError(
                "the directional rule needs the fit's precision matrix; this "
                "estimate carries none. Solve it with "
                "solve_pose(..., statistic=True)"
            )
        theta = torch.cat([estimate.t, estimate.psi.reshape(-1, 1)], dim=-1)
        shrunk = directional_theta(theta, precision)
        return replace(estimate, psi=shrunk[..., 2], t=shrunk[..., :2])

    factor = decision_factor(
        statistic, getattr(estimate, "offset_dof", None), config
    )
    return replace(
        estimate, psi=estimate.psi * factor, t=estimate.t * factor.unsqueeze(-1)
    )


__all__ = [
    "ABSTAIN",
    "ABSTENTION_MODES",
    "BOTH",
    "DIRECTIONAL",
    "DOF_FLOOR",
    "NONE",
    "PER_PAIR",
    "RESIDUAL_SD_FLOOR_M",
    "SANDWICH_RIDGE",
    "solve_or_zero",
    "AbstentionConfig",
    "abstention_threshold",
    "decide",
    "decision_factor",
    "directional_theta",
    "effective_sample_size",
    "expected_residual_sum",
    "residual_scale",
    "sandwich_middle",
    "wald_parts",
    "wald_statistic",
    "WaldParts",
]
