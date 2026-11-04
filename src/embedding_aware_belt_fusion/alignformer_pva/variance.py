"""Inverse-variance weighting of the correspondences entering the Procrustes fit.

Head B solves the SE(2) by weighted least squares over soft correspondences,
and least squares is only the maximum-likelihood estimator when every
observation carries the same variance. It does not here. Measured on the
scenario-disjoint validation slice under the **true** pose projection -- so
what is measured is detector disagreement, never pose error -- the
per-correspondence cross-agent disagreement is strongly heteroscedastic, and
the variable it depends on is **detection confidence**:

====================  ======  ==================  ================
min(score) decile     n       RMS centre (m)      RMS yaw (deg)
====================  ======  ==================  ================
0.200 - 0.284         6398    0.532               13.56
0.415 - 0.445         6396    0.281                4.23
0.560 - 0.773         6400    0.214                3.15
====================  ======  ==================  ================

Range matters too, but mostly because far objects are detected less
confidently: once confidence is in the model, a range term adds nothing (the
fitted range scale runs off to infinity and the validation NLL does not move
past the fifth decimal). Both statements come out of
``scripts/fit_correspondence_variance.py``, which is where the parameters
below are fitted and recorded.

The model is one line. Each *detection* carries a variance that falls as a
power of its confidence, and a correspondence's disagreement variance is the
sum of the two independent detections' variances:

    v(s) = sigma_1^2 * (s / s_ref)^(-2 b)
    Var_centre  = v_t(s_ego) + v_t(s_cav)
    Var_heading = Var_centre + lam^2 * (v_psi(s_ego) + v_psi(s_cav))

``Var_heading`` is not a free choice: the heading virtual point sits at
``centre + lam * u(psi)``, so it carries the centre error *and* ``lam`` times
the heading error. That is why one scalar weight per correspondence is the
wrong shape -- the fitted confidence exponents differ (1.13 for the centre,
1.93 for the heading), so the heading-to-centre weight ratio itself falls with
confidence and no single number can express it. :data:`VARIANCE_MODES` keeps
the scalar variant available so the difference is measurable rather than
asserted.

The inverse-variance factor **multiplies** the Sinkhorn soft-match mass rather
than replacing it: a correspondence that is uncertain in identity *and*
imprecise in position should be discounted twice.

One subtlety. ``procrustes.weighted_se2_kabsch`` is scale-invariant in its
weights, but its ``MIN_MATCH_MASS`` suppression is not -- it reads the total.
So the re-weighted vector is renormalized to the total the unweighted one had,
per sample. That is a no-op for the fit and keeps the fallback firing on
exactly the pairs it fired on before.

PORTING NOTE (AlignFormer task, 2026-10-16): this module was ported verbatim
from the labmate's validated ``embedding-aware-BELT-fusion`` repo
(``origin/alignformer:src/embedding_aware_belt_fusion/alignformer/variance.py``)
into this project's ``alignformer_pva`` backend. The numeric decile table
above and the default fitted constants anywhere a config supplies
``sigma_translation_m``, ``translation_exponent``, ``sigma_yaw_rad``/
``sigma_yaw_deg``, ``yaw_exponent``, and the ``score_reference`` default of
0.4 were all calibrated on the labmate's own detector and validation split.
They are placeholders here and are NOT valid for our detector/dataset until
re-fit against our own scenario-disjoint validation data
(``scripts/fit_correspondence_variance.py`` in the source repo is the fitting
procedure referenced above, not yet ported). This recalibration is tracked as
an explicit future step; see individual docstrings/comments below for which
values are structural math vs. fitted calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import torch
from torch import Tensor

# ``none`` is the unweighted estimator this project deployed before; ``scalar``
# gives each correspondence one weight; ``split`` weights its centre and its
# heading virtual point separately.
VARIANCE_MODES = ("none", "scalar", "split")

# Confidences below this are treated as this. Collated batches pad absent
# objects with score 0, and a raw power of 0 is infinite; the detector's own
# score threshold is 0.2, so nothing real is ever clamped.
#
# STRUCTURAL: this is a numerical-stability floor (prevents 0^-exponent),
# not a fitted calibration value. The 0.05 value is safe to keep as-is
# regardless of detector, as long as the detector's own score threshold stays
# above it (so "nothing real is ever clamped" remains true for our detector
# too -- worth double-checking when we port/recalibrate for our own data).
SCORE_FLOOR = 0.05


@dataclass(frozen=True)
class CorrespondenceVarianceModel:
    """Per-detection disagreement variance as a power law in detection confidence.

    Parameters
    ----------
    mode: one of :data:`VARIANCE_MODES`.
    sigma_translation_m: per-detection RMS centre error, as a 2-D norm, at
        ``score_reference``.
    translation_exponent: ``b_t`` in ``sigma_t(s) = sigma_t1 * (s/s_ref)^-b_t``.
    sigma_yaw_rad: per-detection RMS heading error, in radians, at
        ``score_reference``.
    yaw_exponent: ``b_psi``, the same power for the heading channel.
    score_reference: the confidence the two sigmas are quoted at. Purely a
        parameterization choice; it also fixes the normalization, so a
        correspondence of two reference-confidence detections has centre
        precision exactly 1.
    """

    mode: str
    sigma_translation_m: float
    translation_exponent: float
    sigma_yaw_rad: float
    yaw_exponent: float
    score_reference: float = 0.4

    def __post_init__(self) -> None:
        if self.mode not in VARIANCE_MODES:
            raise ValueError(f"mode must be one of {VARIANCE_MODES}, got {self.mode!r}")
        for name in ("sigma_translation_m", "sigma_yaw_rad", "score_reference"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value}")
        for name in ("translation_exponent", "yaw_exponent"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "sigma_translation_m": self.sigma_translation_m,
            "translation_exponent": self.translation_exponent,
            "sigma_yaw_rad": self.sigma_yaw_rad,
            "sigma_yaw_deg": math.degrees(self.sigma_yaw_rad),
            "yaw_exponent": self.yaw_exponent,
            "score_reference": self.score_reference,
        }

    def detection_variances(self, scores: Tensor) -> Tuple[Tensor, Tensor]:
        """``(centre variance, heading variance)`` of one detection, elementwise.

        Centre variance is a 2-D total (both axes); heading variance is the
        scalar angular variance in rad^2.
        """
        normalized = scores.clamp_min(SCORE_FLOOR) / self.score_reference
        centre = (self.sigma_translation_m * normalized.pow(-self.translation_exponent)) ** 2
        heading = (self.sigma_yaw_rad * normalized.pow(-self.yaw_exponent)) ** 2
        return centre, heading

    def precisions(
        self,
        ego_scores: Tensor,
        cav_variance_centre: Tensor,
        cav_variance_heading: Tensor,
        heading_lambda: float,
    ) -> Tuple[Tensor, Tensor]:
        """``(rho_centre, rho_heading)``, each ``(B, M)``, normalized at the reference.

        ``cav_variance_*`` are the *soft-matched* CAV-side variances: each ego
        row's Sinkhorn-weighted average of its candidates' per-detection
        variances, which is the plug-in estimate for a row whose CAV counterpart
        is a mixture rather than a single box.
        """
        variance_centre, variance_heading = self.correspondence_variances(
            ego_scores, cav_variance_centre, cav_variance_heading, heading_lambda
        )
        reference = 2.0 * self.sigma_translation_m ** 2
        return reference / variance_centre, reference / variance_heading

    def correspondence_variances(
        self,
        ego_scores: Tensor,
        cav_variance_centre: Tensor,
        cav_variance_heading: Tensor,
        heading_lambda: float,
    ) -> Tuple[Tensor, Tensor]:
        """``(Var_centre, Var_heading)``, each ``(B, M)``, in metres squared.

        The same two quantities :meth:`precisions` inverts, exposed unnormalized
        because :mod:`alignformer_pva.abstain` needs the disagreement covariance
        itself and not a weight derived from it. ``Var_centre`` is the 2-D total
        over both axes; ``Var_heading`` is that plus ``lam^2`` times the two
        angular variances, because the heading virtual point carries the centre
        error AND the lever-armed heading error -- which is exactly the
        correlation between an object's two augmented points.
        """
        ego_centre, ego_heading = self.detection_variances(ego_scores)
        variance_centre = ego_centre + cav_variance_centre
        variance_heading = variance_centre + (heading_lambda ** 2) * (
            ego_heading + cav_variance_heading
        )
        return variance_centre, variance_heading

    def augmented_weights(
        self,
        mass: Tensor,
        ego_scores: Tensor,
        cav_variance_centre: Tensor,
        cav_variance_heading: Tensor,
        heading_lambda: float,
    ) -> Tensor:
        """``(B, 2M)`` Kabsch weights: centres first, then heading virtual points.

        ``mode="none"`` returns ``cat([mass, mass])`` bit for bit, which is what
        every measurement before this module was taken under.
        """
        if self.mode == "none":
            return torch.cat([mass, mass], dim=1)

        rho_centre, rho_heading = self.precisions(
            ego_scores, cav_variance_centre, cav_variance_heading, heading_lambda
        )
        if self.mode == "scalar":
            rho_heading = rho_centre

        weights = torch.cat([mass * rho_centre, mass * rho_heading], dim=1)
        # Preserve the total the MIN_MATCH_MASS gate reads; see the module
        # docstring. The fit itself is invariant to this rescaling.
        target = 2.0 * mass.sum(dim=1, keepdim=True)
        total = weights.sum(dim=1, keepdim=True)
        scale = torch.where(
            total > 0, target / total.clamp_min(torch.finfo(total.dtype).tiny),
            torch.ones_like(total),
        )
        return weights * scale


# The model a config without a `correspondence_variance` block implies: the
# unweighted estimator, so every checkpoint trained before this module reloads
# and reproduces its own numbers.
#
# STRUCTURAL: mode="none" and all-neutral values (sigma=1, exponent=0) make
# this a pure identity/no-op fallback -- it is not a calibrated model and
# needs no recalibration; it exists only so an absent config block degrades
# to the pre-existing unweighted estimator.
UNWEIGHTED = CorrespondenceVarianceModel(
    mode="none",
    sigma_translation_m=1.0,
    translation_exponent=0.0,
    sigma_yaw_rad=1.0,
    yaw_exponent=0.0,
)


def variance_model_from_config(
    model_config: Mapping[str, Any], mode: Optional[str] = None
) -> CorrespondenceVarianceModel:
    """Build the model from a config's ``model`` block, optionally forcing ``mode``.

    The fitted parameters are *data* and live in the config; ``mode`` is the
    experimental switch and may be overridden from the command line. Asking for
    a weighted mode with no fitted parameters present is an error rather than a
    silent fallback to made-up constants.

    FITTED CALIBRATION (placeholder, uncalibrated for our detector -- see the
    module-level porting note above, AlignFormer variance.py port task,
    2026-10-16): any ``model_config["correspondence_variance"]`` block a
    caller supplies -- ``sigma_translation_m``, ``translation_exponent``,
    ``sigma_yaw_rad``/``sigma_yaw_deg``, ``yaw_exponent``, and the
    ``score_reference`` default of 0.4 used when a block omits it -- are
    regression constants fit on the labmate's own detector/validation split
    (via the as-yet-unported ``scripts/fit_correspondence_variance.py``).
    They must be re-fit against our own detector/dataset before this mode is
    used for anything beyond functional testing.
    """
    block = model_config.get("correspondence_variance")
    if block is None:
        if mode is not None and mode != "none":
            raise ValueError(
                f"--variance-weighting {mode} needs a model.correspondence_variance "
                "block with fitted parameters; this config has none"
            )
        return UNWEIGHTED

    if "sigma_yaw_rad" in block:
        sigma_yaw_rad = float(block["sigma_yaw_rad"])
    else:
        sigma_yaw_rad = math.radians(float(block["sigma_yaw_deg"]))
    return CorrespondenceVarianceModel(
        mode=str(mode if mode is not None else block.get("mode", "none")),
        sigma_translation_m=float(block["sigma_translation_m"]),
        translation_exponent=float(block["translation_exponent"]),
        sigma_yaw_rad=sigma_yaw_rad,
        yaw_exponent=float(block["yaw_exponent"]),
        score_reference=float(block.get("score_reference", 0.4)),
    )
