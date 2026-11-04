"""The two interchangeable pose heads, over a shared trunk.

Head A regresses the pose from a mean-pooled descriptor. Head B builds a soft
correspondence and solves for the pose in closed form. They share tokenization
and trunk exactly, so a comparison between them isolates the head.

Head A deliberately mean-pools rather than reading out a learned query token:
a content-free token entering the residual stream makes the prediction
nearly input-independent and collapses it to the conditional mean.

Head B never regresses yaw: a regressed yaw never leaves the conditional mean
on this kind of data, so B solves the pose analytically from a Sinkhorn
correspondence instead (see procrustes.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from embedding_aware_belt_fusion.alignformer_pva.abstain import wald_parts
from embedding_aware_belt_fusion.alignformer_pva.head_match import (
    DEFAULT_SINKHORN_ITERATIONS,
    log_sinkhorn,
)
from embedding_aware_belt_fusion.alignformer_pva.procrustes import (
    augment_with_heading,
    heading_orientation,
    weighted_se2_kabsch,
)
from embedding_aware_belt_fusion.alignformer_pva.robust import (
    RobustSolveConfig,
    robust_se2_kabsch,
)
from embedding_aware_belt_fusion.alignformer_pva.trunk import AlignFormerTrunk, tokenize
from embedding_aware_belt_fusion.alignformer_pva.variance import (
    UNWEIGHTED,
    CorrespondenceVarianceModel,
)

# Index of the yaw field in our own [x, y, z, h, w, l, yaw] box layout,
# confirmed to match the source repo's convention (its boxes.py, BOX_YAW = 6).
BOX_YAW = 6

# Heading virtual-point offset in metres, about half a vehicle length so that
# headings and centres contribute comparably to the Kabsch fit.
DEFAULT_HEADING_LAMBDA = 2.0
# Score assigned to padded positions so they can never win a match.
_MASKED_SCORE = -1e4
# Below this, a soft-match row's total weight is treated as zero (no matched
# evidence): guards the /safe_mass divisions from a literal 0/0 without
# perturbing any row that has real, non-negligible weight.
_MIN_MASS = 1e-6


@dataclass(frozen=True)
class PoseEstimate:
    """A CAV's estimated SE(2) correction, with the evidence behind it.

    ``offset_statistic`` is the Wald statistic of ``(psi, t)`` against the
    weighted least-squares fit's OWN covariance, and ``offset_dof`` the degrees
    of freedom its reference distribution is taken at
    (:mod:`alignformer_pva.abstain`). Both are ``None`` unless the solve was
    asked for them, so every estimate produced by a caller that does not want
    a per-pair decision is unchanged in shape and in value.
    """

    psi: Tensor
    t: Tensor
    confidence: Tensor
    log_assignment: Optional[Tensor] = None
    offset_statistic: Optional[Tensor] = None
    offset_dof: Optional[Tensor] = None
    # The fit's inverse covariance, for the directional arm
    # (:mod:`alignformer_pva.abstain`). ``None`` on every estimate that did
    # not ask for a statistic, so nothing older changes shape; the statistic
    # is its quadratic form, ``theta^T offset_precision theta``.
    offset_precision: Optional[Tensor] = None
    # ``(B,)`` bool: which samples the exact re-solve (``alignformer_pva.refine``)
    # actually re-solved. ``None`` on every estimate the refinement never saw,
    # so nothing older changes shape; the agreement rule reads it to know
    # whether a second estimate exists at all for a sample.
    refined: Optional[Tensor] = None
    # ``(B,)`` long: the hard pairs behind the re-solve's final fit (its
    # consensus in RANSAC mode), zero where it did not engage. Set by
    # ``alignformer_pva.refine``; the consensus-floor agreement rule reads it.
    consensus: Optional[Tensor] = None


@dataclass(frozen=True)
class Correspondence:
    """One correspondence, reduced to one virtual CAV point per ego object.

    ``log_assignment`` is ``None`` for a correspondence that did not come from
    Sinkhorn -- the ground-truth assignment ``alignformer_pva.oracle``
    substitutes in, for instance, has no log-domain form to report.
    """

    virtual_centres: Tensor
    virtual_yaws: Tensor
    mass: Tensor
    cav_variance_centre: Tensor
    cav_variance_heading: Tensor
    log_assignment: Optional[Tensor] = None


def _tokens(batch: Mapping[str, Tensor], prefix: str) -> Tensor:
    return tokenize(
        batch[f"{prefix}_boxes"], batch[f"{prefix}_scores"], batch[f"{prefix}_embeddings"]
    )


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    weights = mask.unsqueeze(-1).to(x.dtype)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _is_empty(batch: Mapping[str, Tensor]) -> bool:
    """True when either object set's token dimension is zero.

    ``AlignFormerTrunk`` raises inside ``nn.MultiheadAttention`` when handed a
    ``(B, 0, D)`` token tensor - a real situation (an agent that detected
    nothing), not a synthetic edge case. This MUST be checked before the trunk
    is invoked, not after: there is no exception to catch usefully once
    ``nn.MultiheadAttention`` has already failed its internal reshape.
    """
    return batch["ego_boxes"].shape[1] == 0 or batch["cav_boxes"].shape[1] == 0


def _zero_estimate(
    batch: Mapping[str, Tensor], statistic: bool = False
) -> PoseEstimate:
    """The identity correction with zero confidence, for an empty object set.

    ``statistic`` fills ``offset_statistic`` with zeros rather than leaving it
    ``None``: a pair with no objects has emitted no correction, so its Wald
    statistic IS zero, and a caller applying a per-pair decision should get the
    abstention that implies rather than an error about a missing field.
    """
    reference = batch["ego_boxes"]
    batch_size, device, dtype = reference.shape[0], reference.device, reference.dtype
    psi = torch.zeros(batch_size, device=device, dtype=dtype)
    t = torch.zeros(batch_size, 2, device=device, dtype=dtype)
    confidence = torch.zeros(batch_size, device=device, dtype=dtype)
    return PoseEstimate(
        psi=psi,
        t=t,
        confidence=confidence,
        offset_statistic=torch.zeros_like(psi) if statistic else None,
        offset_dof=torch.zeros_like(psi) if statistic else None,
    )


def reduce_correspondence(
    weights: Tensor,
    batch: Mapping[str, Tensor],
    variance_model: CorrespondenceVarianceModel,
    *,
    log_assignment: Optional[Tensor] = None,
) -> Correspondence:
    """Reduce a ``(B, M, N)`` correspondence matrix to one virtual CAV point per ego row.

    Split out so the ground-truth assignment can be substituted for the
    Sinkhorn one and go through *literally* this code rather than through a
    second implementation of it: an oracle-correspondence ceiling is only a
    ceiling on **matching** if nothing else about the estimator changed with it.

    ``weights`` must already be masked to valid (ego, CAV) positions. Rows with
    no weight -- a padded ego position, or an ego object the assignment left
    unmatched -- come back with zero mass and are therefore dropped by the
    Kabsch fit downstream.
    """
    mass = weights.sum(dim=2)
    safe_mass = mass.clamp_min(_MIN_MASS).unsqueeze(-1)
    cav_centres = batch["cav_boxes"][..., :2]
    cav_yaws = batch["cav_boxes"][..., BOX_YAW]

    virtual_centres = weights @ cav_centres / safe_mass
    # Fold each CAV heading onto the half-plane of the ego object it is
    # being matched to BEFORE averaging. The detector reports no direction
    # (procrustes.heading_orientation), so a meaningful fraction of
    # cross-agent detections of the same object point the opposite way
    # along the same axis; averaging those raw would cancel real heading
    # evidence and push the heading virtual point up to 2 * heading_lambda
    # off. The fold has to happen per (ego row, CAV object) rather than once
    # per CAV object, because a single row can soft-match CAV boxes that are
    # flipped differently from one another.
    orientation = heading_orientation(batch["ego_boxes"][..., BOX_YAW], cav_yaws)
    cav_direction = torch.stack([torch.cos(cav_yaws), torch.sin(cav_yaws)], dim=-1)
    virtual_direction = (weights * orientation) @ cav_direction / safe_mass
    # atan2's gradient is undefined at exactly (0, 0) -- a mathematical
    # singularity -- and any row with zero soft-match weight (an ordinary
    # padded ego position in a collated batch, or a wholly empty sample)
    # lands exactly there, since `weights` is identically 0 for it. Route
    # those rows through a placeholder direction with a well-defined
    # gradient before atan2, mirroring weighted_se2_kabsch's own atan2 guard.
    #
    # The placeholder's value is never read: for the dominant case (one
    # padded row inside an otherwise ordinary, non-empty sample) that row's
    # own Kabsch WEIGHT (`augmented_mass` below) is 0, so it drops out of
    # weighted_se2_kabsch's sums regardless of its yaw. For a wholly empty
    # sample specifically, weighted_se2_kabsch's own MIN_MATCH_MASS zeroes
    # the entire correction instead. Threshold at `_MIN_MASS`, not 0, to
    # agree with the `safe_mass` clamp above.
    has_weight = mass > _MIN_MASS
    placeholder_direction = virtual_direction.new_tensor([1.0, 0.0]).expand_as(
        virtual_direction
    )
    safe_direction = torch.where(
        has_weight.unsqueeze(-1), virtual_direction, placeholder_direction
    )
    virtual_yaws = torch.atan2(safe_direction[..., 1], safe_direction[..., 0])

    # The CAV side of each correspondence is a mixture, so its variance is
    # the soft-match-weighted average of its candidates' variances. Same
    # `weights @ x / safe_mass` reduction as the centre and the heading.
    cav_centre_variance, cav_heading_variance = (
        variance_model.detection_variances(batch["cav_scores"])
    )
    return Correspondence(
        virtual_centres=virtual_centres,
        virtual_yaws=virtual_yaws,
        mass=mass,
        cav_variance_centre=(
            (weights @ cav_centre_variance.unsqueeze(-1)).squeeze(-1)
            / safe_mass.squeeze(-1)
        ),
        cav_variance_heading=(
            (weights @ cav_heading_variance.unsqueeze(-1)).squeeze(-1)
            / safe_mass.squeeze(-1)
        ),
        log_assignment=log_assignment,
    )


def solve_pose(
    correspondence: Correspondence,
    batch: Mapping[str, Tensor],
    heading_lambda: float,
    variance_model: CorrespondenceVarianceModel,
    robust: Optional[RobustSolveConfig] = None,
    *,
    statistic: bool = False,
) -> PoseEstimate:
    """Head B's closed-form SE(2) solve over an already-reduced correspondence.

    Split out for the same reason :func:`reduce_correspondence` was:
    whatever produced the correspondence, the pose that follows from it is
    solved by exactly this code.

    ``robust``, when given, wraps that solve in the IRLS loop of
    :mod:`alignformer_pva.robust`: solve, re-weight by the residuals,
    re-solve. ``None`` -- and a disabled configuration -- run the single
    weighted Kabsch. The loop's evidence gate reads ``mass.sum(dim=1)``,
    which is the same quantity reported as ``PoseEstimate.confidence``.

    ``statistic`` additionally reports the Wald statistic of the emitted
    correction against this fit's own covariance, and the degrees of freedom
    its reference is taken at (:func:`alignformer_pva.abstain.wald_statistic`),
    computed from the points and the weights the FINAL solve used -- the IRLS
    loop's re-weighted vector when it ran, the inverse-variance one when it did
    not. It is read off the solve and changes nothing about it: ``psi`` and
    ``t`` are bit-identical either way.
    """
    mass = correspondence.mass

    source = augment_with_heading(
        correspondence.virtual_centres, correspondence.virtual_yaws, heading_lambda
    )
    target = augment_with_heading(
        batch["ego_boxes"][..., :2],
        batch["ego_boxes"][..., BOX_YAW],
        heading_lambda,
    )
    # Inverse-variance weighting MULTIPLIES the soft-match mass rather than
    # replacing it: identity uncertainty and positional imprecision are two
    # separate reasons to discount a correspondence. `mode="none"` returns
    # cat([mass, mass]) bit for bit (variance.py).
    augmented_mass = variance_model.augmented_weights(
        mass,
        batch["ego_scores"],
        correspondence.cav_variance_centre,
        correspondence.cav_variance_heading,
        heading_lambda,
    )

    confidence = mass.sum(dim=1)
    if robust is None:
        psi, t = weighted_se2_kabsch(target, source, augmented_mass)
        solved_weights = augmented_mass
    else:
        solution = robust_se2_kabsch(
            target, source, augmented_mass, evidence=confidence, config=robust
        )
        psi, t, solved_weights = solution.psi, solution.t, solution.weights

    offset_statistic = offset_dof = offset_precision = None
    if statistic:
        # The DISAGREEMENT covariance, from the same fitted model the weights
        # are built from -- not a second uncertainty model. It is needed in
        # full because an object's two augmented points share their centre
        # error, which is exactly what the sandwich accounts for.
        variance_centre, variance_heading = variance_model.correspondence_variances(
            batch["ego_scores"],
            correspondence.cav_variance_centre,
            correspondence.cav_variance_heading,
            heading_lambda,
        )
        parts = wald_parts(
            target,
            source,
            solved_weights,
            psi,
            t,
            variance_centre=variance_centre,
            variance_heading=variance_heading,
            heading_lambda=heading_lambda,
        )
        offset_statistic, offset_dof = parts.statistic, parts.dof
        offset_precision = parts.precision
    return PoseEstimate(
        psi=psi,
        t=t,
        confidence=confidence,
        log_assignment=correspondence.log_assignment,
        offset_statistic=offset_statistic,
        offset_dof=offset_dof,
        offset_precision=offset_precision,
    )


class _Base(nn.Module):
    def __init__(self, embed_dim: int, **trunk_kwargs) -> None:
        super().__init__()
        self.trunk = AlignFormerTrunk(embed_dim=embed_dim, **trunk_kwargs)

    def _encode(self, batch: Mapping[str, Tensor]) -> Tuple[Tensor, Tensor]:
        return self.trunk(
            _tokens(batch, "ego"),
            _tokens(batch, "cav"),
            batch["ego_mask"],
            batch["cav_mask"],
        )


class AlignFormerA(_Base):
    """Direct regression from a mean-pooled descriptor of both sets."""

    def __init__(self, embed_dim: int, **trunk_kwargs) -> None:
        super().__init__(embed_dim, **trunk_kwargs)
        model_dim = self.trunk.model_dim
        self.head = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 3),
        )

    def forward(self, batch: Mapping[str, Tensor]) -> PoseEstimate:
        # Guard BEFORE the trunk call: see _is_empty's docstring.
        if _is_empty(batch):
            return _zero_estimate(batch)

        ego, cav = self._encode(batch)
        pooled = torch.cat(
            [_masked_mean(ego, batch["ego_mask"]), _masked_mean(cav, batch["cav_mask"])],
            dim=-1,
        )
        output = self.head(pooled)

        present = batch["ego_mask"].any(dim=1) & batch["cav_mask"].any(dim=1)
        psi = torch.where(present, output[:, 2], torch.zeros_like(output[:, 2]))
        t = torch.where(
            present.unsqueeze(-1), output[:, :2], torch.zeros_like(output[:, :2])
        )
        return PoseEstimate(psi=psi, t=t, confidence=present.to(psi.dtype))


class AlignFormerB(_Base):
    """Soft correspondence, then a closed-form weighted SE(2) Kabsch solve."""

    def __init__(
        self,
        embed_dim: int,
        heading_lambda: float = DEFAULT_HEADING_LAMBDA,
        sinkhorn_iterations: int = DEFAULT_SINKHORN_ITERATIONS,
        variance_model: CorrespondenceVarianceModel = UNWEIGHTED,
        **trunk_kwargs,
    ) -> None:
        super().__init__(embed_dim, **trunk_kwargs)
        model_dim = self.trunk.model_dim
        self.match_projection = nn.Linear(model_dim, model_dim)
        self.dustbin = nn.Parameter(torch.tensor(1.0))
        self.log_temperature = nn.Parameter(torch.tensor(0.1).log())
        self.heading_lambda = heading_lambda
        self.sinkhorn_iterations = sinkhorn_iterations
        # Carries no parameters and no buffers, so it stays out of state_dict
        # and an old checkpoint still loads strictly.
        self.variance_model = variance_model
        # Test hook: score on the raw embeddings, bypassing an untrained trunk.
        self.use_raw_embedding_scores = False

    def _soft_correspondence(
        self,
        batch: Mapping[str, Tensor],
        ego: Tensor,
        cav: Tensor,
        ego_mask: Tensor,
        cav_mask: Tensor,
    ) -> Correspondence:
        """Score, Sinkhorn-normalize, and reduce to one virtual CAV point per ego object.

        Returns a :class:`Correspondence`: the per-ego-object soft-matched
        CAV centre/heading (``(B, M, 2)`` / ``(B, M)``), each row's total
        soft-match weight (``(B, M)``), the Sinkhorn-weighted average of its
        candidates' per-detection disagreement variances (which is what the
        inverse-variance weighting consumes on the CAV side), and
        ``log_assignment`` passed through unchanged for :class:`PoseEstimate`.
        """
        if self.use_raw_embedding_scores:
            ego_features = batch["ego_embeddings"]
            cav_features = batch["cav_embeddings"]
        else:
            ego_features = F.normalize(self.match_projection(ego), dim=-1)
            cav_features = F.normalize(self.match_projection(cav), dim=-1)

        scores = ego_features @ cav_features.transpose(1, 2) / self.log_temperature.exp()
        valid = ego_mask.unsqueeze(2) & cav_mask.unsqueeze(1)
        scores = scores.masked_fill(~valid, _MASKED_SCORE)

        log_assignment = log_sinkhorn(scores, self.dustbin, self.sinkhorn_iterations)
        weights = log_assignment[:, :-1, :-1].exp() * valid
        return reduce_correspondence(
            weights, batch, self.variance_model, log_assignment=log_assignment
        )

    def forward(
        self,
        batch: Mapping[str, Tensor],
        robust: Optional[RobustSolveConfig] = None,
        statistic: bool = False,
    ) -> PoseEstimate:
        # Guard BEFORE the trunk call: see _is_empty's docstring. With an empty
        # object set there is no correspondence to build anyway, so the zero
        # correction below is not just a crash-avoidance shortcut - it is the
        # correct answer.
        if _is_empty(batch):
            return _zero_estimate(batch, statistic)

        ego_mask, cav_mask = batch["ego_mask"], batch["cav_mask"]
        ego, cav = self._encode(batch)

        correspondence = self._soft_correspondence(batch, ego, cav, ego_mask, cav_mask)
        return solve_pose(
            correspondence,
            batch,
            self.heading_lambda,
            self.variance_model,
            robust,
            statistic=statistic,
        )
