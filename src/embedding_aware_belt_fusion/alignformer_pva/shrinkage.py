"""Calibrated shrinkage of the estimated SE(2) correction.

AlignFormer's pose estimator has a noise floor: with a perfect solver and a
near-perfect correspondence it still disagrees with the truth by whatever the
detector's own box error leaves behind after averaging. Measured at sigma = 0
on the scenario-disjoint validation split, that residual is **noise, not
bias** -- its signed mean is a few per cent of its spread (task-15 report) --
which means the estimator is unbiased and merely imprecise.

An unbiased-but-imprecise correction is harmful exactly when there is little
error to correct: at sigma = 0 the truth is the identity and every metre the
estimator emits is a metre of damage. The statistically correct treatment is
not a noise-conditioned switch but shrinkage. Write the observed correction as

    x = e + n,     e ~ N(0, sigma^2 I)  (the true pose error, unknown scale)
                   n ~ N(0, tau^2 I)    (this estimator's noise, calibrated)

The posterior mean of ``e`` is ``sigma^2 / (sigma^2 + tau^2) * x``. ``sigma``
is not known at inference, so it is estimated from the only evidence there is,
the observed correction itself. Standardize each component by its own
calibrated noise scale and pool them into

    z^2 = 2 |t|^2 / tau_t^2 + psi^2 / tau_psi^2

which has expectation ``p (1 + sigma^2 / tau^2)`` over ``p = 3`` dimensions
(dx, dy, dpsi). Inverting that gives ``sigma^2_hat`` and the factor collapses
to a single number for the whole correction:

    k(x) = max(0, 1 - p / z^2)

-- the positive-part empirical-Bayes rule, with James-Stein's dominance result
applying because the pooled problem has three dimensions rather than the two a
translation alone would have. One factor, applied to the whole SE(2), so the
shrunk correction is always somewhere on the path between the identity and the
one the solver proposed.

**Pooling is not a convenience.** Estimating ``sigma`` from the yaw alone is
one observation of a heavy-tailed quantity and shrinks far too hard: measured,
a per-component rule pushed yaw MAE at sigma = 0.2 m back onto the predict-zero
baseline, while the pooled rule -- same tau, same calibration -- keeps it
clearly below. The translation carries most of the evidence about how large the
localization error is, and the yaw is entitled to use it.

Nothing here is fitted per sigma. ``tau`` is measured once, on the validation
split only, at sigma = 0, where the true correction is exactly the identity and
the emitted correction therefore IS the residual.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Dict

import torch
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover - imported for annotations only
    # Deferred: alignformer.model imports alignformer.abstain, which imports
    # the positive-part rule from here. `from __future__ import annotations`
    # above makes every annotation a string, so nothing below needs the class
    # at runtime and the cycle does not have to exist.
    from embedding_aware_belt_fusion.alignformer_pva.model import PoseEstimate


@dataclass(frozen=True)
class ShrinkageCalibration:
    """The estimator's own noise scale, measured on held-out validation data.

    ``tau_translation_m`` is the RMS of the residual's **norm** (both axes
    together), which is the ``E|n|^2`` the module docstring's formula wants;
    ``tau_yaw_rad`` is the RMS of the scalar yaw residual. ``pairs``, ``split``
    and ``sigma_m`` are provenance: a calibration is only meaningful next to
    what it was measured on.
    """

    tau_translation_m: float
    tau_yaw_rad: float
    pairs: int
    split: str
    sigma_m: float

    def __post_init__(self) -> None:
        for name in ("tau_translation_m", "tau_yaw_rad"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                # Strictly positive, not merely non-negative: z^2 divides by
                # tau^2, so a zero tau gives z^2 = inf and a shrinkage factor
                # of 1 -- no shrinkage at all, from a calibration that found
                # no signal. Matches CorrespondenceVarianceModel's own check.
                raise ValueError(f"{name} must be finite and positive, got {value}")

    @property
    def tau_yaw_deg(self) -> float:
        return math.degrees(self.tau_yaw_rad)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tau_translation_m": self.tau_translation_m,
            "tau_yaw_rad": self.tau_yaw_rad,
            "tau_yaw_deg": self.tau_yaw_deg,
            "pairs": self.pairs,
            "split": self.split,
            "sigma_m": self.sigma_m,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ShrinkageCalibration":
        return cls(
            tau_translation_m=float(payload["tau_translation_m"]),
            tau_yaw_rad=float(payload["tau_yaw_rad"]),
            pairs=int(payload["pairs"]),
            split=str(payload["split"]),
            sigma_m=float(payload["sigma_m"]),
        )


# The pooled problem's dimension: (dx, dy, dpsi).
POSE_DIMENSIONS = 3


def standardized_magnitude_squared(
    psi: Tensor, t: Tensor, calibration: "ShrinkageCalibration"
) -> Tensor:
    """``z^2`` for each sample: the correction measured in units of its own noise.

    Translation contributes ``2 |t|^2 / tau_t^2`` because ``tau_translation_m``
    is the RMS of the residual's *norm* over both axes, so the per-axis
    variance it implies is ``tau_t^2 / 2``.
    """
    translation = 2.0 * (t * t).sum(dim=-1) / (calibration.tau_translation_m ** 2)
    yaw = (psi * psi) / (calibration.tau_yaw_rad ** 2)
    return translation + yaw


def shrinkage_factor(
    standardized_squared: Tensor, dimensions: int = POSE_DIMENSIONS
) -> Tensor:
    """``max(0, 1 - p / z^2)``, clamped to ``[0, 1]``.

    A ``z^2`` of exactly zero (the suppressed identity correction) returns zero
    rather than dividing by it.
    """
    if dimensions < 1:
        raise ValueError(f"dimensions must be at least 1, got {dimensions}")
    ratio = torch.where(
        standardized_squared > 0,
        dimensions / standardized_squared.clamp_min(
            torch.finfo(standardized_squared.dtype).tiny
        ),
        torch.ones_like(standardized_squared),
    )
    return (1.0 - ratio).clamp_min(0.0)


def shrink(
    estimate: "PoseEstimate", calibration: "ShrinkageCalibration"
) -> "PoseEstimate":
    """Return a **new** :class:`PoseEstimate` with the correction shrunk.

    One factor for the whole SE(2): the translation's direction and the yaw's
    sign are untouched, so the result always lies between the identity and the
    correction the solver proposed. The input estimate is never mutated --
    ``PoseEstimate`` is frozen and this rebuilds it with
    :func:`dataclasses.replace`.
    """
    factor = shrinkage_factor(
        standardized_magnitude_squared(estimate.psi, estimate.t, calibration)
    )
    return replace(
        estimate, psi=estimate.psi * factor, t=estimate.t * factor.unsqueeze(-1)
    )


def calibrate_shrinkage(
    residual_t: Tensor,
    residual_psi: Tensor,
    *,
    pairs: int,
    split: str,
    sigma_m: float,
) -> ShrinkageCalibration:
    """Measure ``tau`` from residuals the estimator produced on held-out data.

    Parameters
    ----------
    residual_t: ``(N, 2)`` signed translation residuals, in metres.
    residual_psi: ``(N,)`` signed yaw residuals, in radians.

    ``tau`` is the plain RMS, not the standard deviation about the observed
    mean: the formula in the module docstring wants ``E|n|^2``, and using the
    centred spread instead would quietly subtract off any bias rather than
    charging the estimator for it.
    """
    if residual_t.dim() != 2 or residual_t.shape[-1] != 2:
        raise ValueError(f"residual_t must be (N, 2), got {tuple(residual_t.shape)}")
    if residual_psi.dim() != 1 or residual_psi.shape[0] != residual_t.shape[0]:
        raise ValueError(
            f"residual_psi must be ({residual_t.shape[0]},), got "
            f"{tuple(residual_psi.shape)}"
        )
    return ShrinkageCalibration(
        tau_translation_m=float(
            torch.sqrt((residual_t.double() ** 2).sum(dim=-1).mean())
        ),
        tau_yaw_rad=float(torch.sqrt((residual_psi.double() ** 2).mean())),
        pairs=int(pairs),
        split=split,
        sigma_m=float(sigma_m),
    )
