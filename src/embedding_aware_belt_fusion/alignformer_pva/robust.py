"""An IRLS loop wrapped around head B's closed-form solve, not replacing it.

Head B solves the SE(2) **once**, by weighted least squares over the Sinkhorn
soft-match mass times the inverse-variance precision
(:func:`model.solve_pose`). Least squares has no redescending influence
function: a wrong correspondence with small-but-nonzero mass biases the fit in
proportion to its residual and nothing ever removes it. The published
competitor this project measures itself against (:mod:`alignformer.freealign`,
Lei et al., ICRA 2024) instead fits **hard** -- LMedS over a matched subgraph,
where an outlier correspondence is *discarded* -- and its measured advantage on
this data is precisely a dense-regime, typical-case one: on the 90.3% of test
pairs sharing three or more objects its answered translation median is
0.0984 m against AlignFormer's 0.1024 m, while AlignFormer owns the mean
(0.2203 m against 1.2984 m) and the tail (0.93% against 2.55% beyond 3 m).

This module is the intervention that difference suggests: re-weight by the
residuals and re-solve, a bounded number of times.

Three properties are load-bearing and each is pinned by a test.

**It wraps, it does not replace.** ``mode="none"`` (and ``iterations=0``)
return :func:`procrustes.weighted_se2_kabsch` bit for bit, so every
measurement taken before this module still reproduces and the new arm is the
only thing that moved.

**The guard.** Trimming needs evidence. AlignFormer's structural advantage over
a distance-graph method lives in the pairs sharing one or two objects -- on the
test split it gains +0.184 AP@0.7 over doing nothing there, against FreeAlign's
+0.011 -- and an IRLS loop handed three effective points will happily trim its
way to nonsense. Below :attr:`RobustSolveConfig.min_evidence` effective
correspondences the loop is skipped and the single solve is returned untouched,
per sample rather than per batch. The threshold is chosen on the
scenario-disjoint validation slice (``scripts/calibrate_robust_solve.py``),
never on test, and defaults to the same 3 below which a relative-distance graph
is degenerate.

**The gradient path is unchanged.** The residual scale and the robust factor
are :meth:`~torch.Tensor.detach`\\ ed every iteration, so the graph the
optimizer sees is still one weighted Kabsch -- with different constants in the
weight vector. Nothing here was in the training graph of the deployed
checkpoint and this is an inference-time change; the detach keeps it usable in
training without turning a closed-form solve into an unrolled optimization.

The scale is fitted **per sample** from the mass-weighted residuals rather than
being a fixed number of metres, because a fixed metre threshold cannot transfer
across a sweep whose pose error spans 0 to 2 m. It is a weighted median --
the same statistic LMedS selects on -- converted to a per-axis sigma through
the Rayleigh median/sigma ratio :data:`RAYLEIGH_MEDIAN_OVER_SIGMA` so that the
Huber constant keeps its textbook meaning, and floored at
:data:`SCALE_FLOOR_M` so an almost-exact fit cannot divide by zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from embedding_aware_belt_fusion.alignformer_pva.procrustes import weighted_se2_kabsch

# ``none`` is the deployed single solve; the other two are the psi functions
# the brief asks to be compared. Huber is monotone and bounded-influence (an
# outlier keeps a constant, small pull); Geman-McClure is redescending (an
# outlier's pull goes to zero), so it trims harder and risks more.
ROBUST_NONE = "none"
HUBER = "huber"
GEMAN_MCCLURE = "geman_mcclure"
ROBUST_MODES = (ROBUST_NONE, HUBER, GEMAN_MCCLURE)

# Huber's classical 95%-efficiency constant, in units of the fitted sigma.
DEFAULT_HUBER_CONSTANT = 1.345
# Residual scale below which the fit is already exact for this detector: the
# fitted per-detection RMS centre error at the reference confidence is 0.2516 m
# (variance.py), so nothing real is resolved at 5 cm and flooring there cannot
# trim signal. It exists to keep ``residual / scale`` finite on a perfect fit.
SCALE_FLOOR_M = 0.05
# median / sigma for a 2-D isotropic Gaussian's magnitude (Rayleigh), exactly
# sqrt(2 ln 2). Dividing the weighted median residual by it turns a robust
# location statistic into the sigma the Huber constant is quoted against.
RAYLEIGH_MEDIAN_OVER_SIGMA = math.sqrt(2.0 * math.log(2.0))
# Effective matched correspondences below which the loop is skipped. Three is
# where a relative-distance graph stops being degenerate and is the boundary
# the sweep's sparse slice (``shared_1_2``) is drawn at, so a regression caused
# by trimming on thin evidence shows up in a slice rather than in an average.
DEFAULT_MIN_EVIDENCE = 3.0
DEFAULT_ITERATIONS = 1


@dataclass(frozen=True)
class RobustSolveConfig:
    """Every tunable of the loop, in one immutable place.

    Parameters
    ----------
    mode: one of :data:`ROBUST_MODES`. ``none`` disables the loop entirely.
    iterations: how many re-weight/re-solve rounds. ``0`` also disables it.
    min_evidence: effective matched correspondences below which a sample keeps
        its single solve untouched. See the module docstring.
    huber_constant: the Huber cut, in fitted sigmas. Unused by Geman-McClure.
    scale_floor_m: lower bound on the fitted residual scale, in metres.
    """

    mode: str = ROBUST_NONE
    iterations: int = DEFAULT_ITERATIONS
    min_evidence: float = DEFAULT_MIN_EVIDENCE
    huber_constant: float = DEFAULT_HUBER_CONSTANT
    scale_floor_m: float = SCALE_FLOOR_M

    def __post_init__(self) -> None:
        if self.mode not in ROBUST_MODES:
            raise ValueError(f"mode must be one of {ROBUST_MODES}, got {self.mode!r}")
        if self.iterations < 0:
            raise ValueError(f"iterations must be >= 0, got {self.iterations}")
        if self.min_evidence < 0.0:
            raise ValueError(f"min_evidence must be >= 0, got {self.min_evidence}")
        if not (self.huber_constant > 0.0 and math.isfinite(self.huber_constant)):
            raise ValueError(
                f"huber_constant must be finite and positive, got {self.huber_constant}"
            )
        if not (self.scale_floor_m > 0.0 and math.isfinite(self.scale_floor_m)):
            raise ValueError(
                f"scale_floor_m must be finite and positive, got {self.scale_floor_m}"
            )

    @property
    def enabled(self) -> bool:
        return self.mode != ROBUST_NONE and self.iterations > 0

    def to_dict(self) -> dict:
        """JSON-safe record of the configuration, for the result file."""
        return {
            "mode": self.mode,
            "iterations": self.iterations,
            "min_evidence": self.min_evidence,
            "huber_constant": self.huber_constant,
            "scale_floor_m": self.scale_floor_m,
            "enabled": self.enabled,
        }


# The configuration every prior measurement was taken under.
DISABLED = RobustSolveConfig()


@dataclass(frozen=True)
class RobustSolution:
    """What the loop produced, and the evidence that it ran.

    ``weights`` is the weight vector the final solve actually used -- equal to
    the input for a sample the guard declined -- and ``engaged`` is the guard's
    per-sample verdict. Both are returned rather than kept private because the
    honest check on this experiment is whether the loop fired at all, and that
    has to be measurable from outside.
    """

    psi: Tensor
    t: Tensor
    weights: Tensor
    engaged: Tensor


def weighted_median(values: Tensor, weights: Tensor) -> Tensor:
    """``(B,)`` weighted median of ``(B, N)`` values under ``(B, N)`` weights.

    The smallest value whose cumulative weight reaches half the total, which is
    the lower weighted median. Order-invariant, and defined (as the minimum)
    for a row whose weights are all zero -- such a row is suppressed by
    ``MIN_MATCH_MASS`` downstream anyway, and returning a finite number keeps
    the division that follows away from NaN.
    """
    if values.shape != weights.shape:
        raise ValueError(
            f"values {tuple(values.shape)} and weights {tuple(weights.shape)} disagree"
        )
    order = values.argsort(dim=1)
    ordered_values = values.gather(1, order)
    cumulative = weights.clamp_min(0.0).gather(1, order).cumsum(dim=1)
    total = cumulative[:, -1:].clamp_min(torch.finfo(values.dtype).tiny)
    index = (cumulative >= 0.5 * total).to(values.dtype).argmax(dim=1)
    return ordered_values.gather(1, index.unsqueeze(1)).squeeze(1)


def residual_scale(
    residual: Tensor, weights: Tensor, config: RobustSolveConfig
) -> Tensor:
    """``(B, 1)`` per-sample robust residual scale, detached and floored.

    Fitted from the *original* weights at every iteration rather than from the
    current ones: re-estimating it from already-trimmed weights makes each
    round trim harder than the last, which turns a bounded re-weighting into an
    unbounded one.
    """
    median = weighted_median(residual.detach(), weights.detach())
    scale = (median / RAYLEIGH_MEDIAN_OVER_SIGMA).clamp_min(config.scale_floor_m)
    return scale.unsqueeze(-1)


def robust_factor(
    residual: Tensor, scale: Tensor, config: RobustSolveConfig
) -> Tensor:
    """The multiplicative weight ``psi(r / s)``, always detached.

    Detaching here is what keeps the gradient path the single-solve one: the
    factor enters the next solve as a constant, so the optimizer still sees one
    weighted Kabsch rather than an unrolled fixed-point iteration.
    """
    standardized = (residual.detach() / scale.detach()).clamp_min(0.0)
    if config.mode == HUBER:
        tiny = torch.finfo(standardized.dtype).tiny
        return (config.huber_constant / standardized.clamp_min(tiny)).clamp_max(1.0)
    if config.mode == GEMAN_MCCLURE:
        return 1.0 / (1.0 + standardized * standardized) ** 2
    raise ValueError(f"no robust factor for mode {config.mode!r}")


def _residuals(p: Tensor, q: Tensor, psi: Tensor, t: Tensor) -> Tensor:
    """``(B, N)`` of ``|p_n - (R(psi) q_n + t)|`` for one transform per sample."""
    cos, sin = torch.cos(psi).unsqueeze(-1), torch.sin(psi).unsqueeze(-1)
    x, y = q[..., 0], q[..., 1]
    moved = torch.stack([cos * x - sin * y, sin * x + cos * y], dim=-1)
    return (moved + t.unsqueeze(1) - p).norm(dim=-1)


def _renormalized(candidate: Tensor, reference: Tensor) -> Tensor:
    """``candidate`` rescaled to carry the total ``reference`` carries.

    ``weighted_se2_kabsch`` is scale-invariant in its weights but its
    ``MIN_MATCH_MASS`` suppression is not -- it reads the total -- so without
    this the loop would change *which pairs are answered at all* and the
    comparison would no longer be of the solve alone. Same pattern, and same
    reason, as ``variance.augmented_weights``.
    """
    target = reference.sum(dim=1, keepdim=True)
    total = candidate.sum(dim=1, keepdim=True)
    scale = torch.where(
        total > 0,
        target / total.clamp_min(torch.finfo(total.dtype).tiny),
        torch.ones_like(total),
    )
    return candidate * scale


def robust_se2_kabsch(
    p: Tensor,
    q: Tensor,
    w: Tensor,
    *,
    evidence: Tensor,
    config: RobustSolveConfig,
) -> RobustSolution:
    """Solve once as deployed, then re-weight by the residuals and re-solve.

    Parameters
    ----------
    p: ``(B, N, 2)`` target points, ``q``: ``(B, N, 2)`` source points and
        ``w``: ``(B, N)`` weights, exactly as :func:`weighted_se2_kabsch`
        takes them -- here the heading-augmented ``N = 2M`` vectors.
    evidence: ``(B,)`` effective matched correspondences, which is
        ``PoseEstimate.confidence``. Compared against
        ``config.min_evidence`` per sample; below it the sample keeps the
        single solve untouched.

    A disabled ``config`` returns :func:`weighted_se2_kabsch` bit for bit.
    """
    psi, t = weighted_se2_kabsch(p, q, w)
    engaged = torch.as_tensor(evidence, device=p.device) >= config.min_evidence
    # An empty point set has no residual to re-weight by, and a weighted median
    # over zero values is not defined. `weighted_se2_kabsch` answers the
    # identity for it rather than raising, and so must this -- an agent that
    # detected nothing is a real situation here, not a synthetic edge case
    # (model._is_empty). It is guarded upstream today; guarding it here too
    # means a future caller cannot reintroduce the crash.
    if not config.enabled or p.shape[1] == 0:
        return RobustSolution(psi, t, w, torch.zeros_like(engaged))

    weights = w
    for _ in range(config.iterations):
        residual = _residuals(p, q, psi, t)
        factor = robust_factor(residual, residual_scale(residual, w, config), config)
        weights = torch.where(engaged.unsqueeze(-1), _renormalized(w * factor, w), w)
        candidate_psi, candidate_t = weighted_se2_kabsch(p, q, weights)
        psi = torch.where(engaged, candidate_psi, psi)
        t = torch.where(engaged.unsqueeze(-1), candidate_t, t)
    return RobustSolution(psi=psi, t=t, weights=weights, engaged=engaged)


__all__ = [
    "DEFAULT_HUBER_CONSTANT",
    "DEFAULT_ITERATIONS",
    "DEFAULT_MIN_EVIDENCE",
    "DISABLED",
    "GEMAN_MCCLURE",
    "HUBER",
    "RAYLEIGH_MEDIAN_OVER_SIGMA",
    "ROBUST_MODES",
    "ROBUST_NONE",
    "SCALE_FLOOR_M",
    "RobustSolution",
    "RobustSolveConfig",
    "residual_scale",
    "robust_factor",
    "robust_se2_kabsch",
    "weighted_median",
]
