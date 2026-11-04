"""The exact re-solve: hard correspondences under the soft estimate, then an
unweighted closed-form SE(2) fit, iterated with a shrinking gate.

What it answers: the soft solve's input is one virtual CAV point per ego
object, a Sinkhorn-mass mixture of the candidates; as the tokens get noisier
the row spreads over neighbours and the mixture lands between them, so the
least-squares answer is blurred rather than wrong. A correspondence-deciding
baseline's advantage in that regime is that it *decides* the correspondence
and then fits exactly. This module gives the soft estimate the same second
step.

The step is plain ICP, initialised by the soft estimate: move the CAV boxes
by the current (psi, t), take the mutually-nearest ego/CAV centre pairs
within a gate, solve SE(2) exactly over them with the heading augmentation
and the heading fold the soft solve already uses, and repeat with a tighter
gate. Everything here is inference-time and parameter-free apart from the
gate schedule and the evidence floor, both chosen on validation.

Two things are deliberate. The refinement **keeps the soft fit's decision
inputs** (the Wald statistic, its degrees of freedom and the precision) and
changes only ``(psi, t)``, so the deployed decision rules apply to the
refined correction unchanged. And it **engages only above an evidence
floor** of hard pairs, because a nearest-neighbour step with two points will
happily snap to the wrong neighbour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import torch
from torch import Tensor

from embedding_aware_belt_fusion.alignformer_pva.procrustes import (
    augment_with_heading,
    heading_orientation,
    weighted_se2_kabsch,
)

# Index of the yaw field in our own [x, y, z, h, w, l, yaw] box layout,
# confirmed to match the source repo's convention (its boxes.py, BOX_YAW = 6).
BOX_YAW = 6

REFINE_NONE = "none"
ICP = "icp"
# The same gates and pairs, but every fit is RANSAC over the hard pairs
# (minimal two-pair samples, a consensus gate, the heading-augmented exact
# fit over the consensus).
ICP_RANSAC = "icp_ransac"
REFINE_MODES = (REFINE_NONE, ICP, ICP_RANSAC)
# The gate schedule: the first gate must admit the soft fit's own residual
# at moderate noise, the last should be near the detector's per-detection
# centre error so a neighbour a lane away can never be taken for the object
# itself.
DEFAULT_GATES_M = (2.0, 1.0, 0.5)
# Hard correspondences below which the soft estimate is kept: three is where
# a relative-distance graph stops being degenerate.
DEFAULT_MIN_PAIRS = 3
DEFAULT_INLIER_M = 1.0
# Above this many hard pairs the minimal samples are drawn, not enumerated.
EXHAUSTIVE_PAIRS = 16
RANSAC_SAMPLES = 512
# The search step (``search_m`` > 0): when the first gate finds fewer than
# ``min_pairs`` mutual pairs, every ego/CAV pair within ``search_m`` of each
# other under the soft estimate is a candidate, two-candidate samples are
# drawn, and the consensus (one-to-one nearest within ``inlier_m``) seeds the
# gates.
SEARCH_OFF = 0.0
SEARCH_SAMPLES = 1024
REFINED_SUFFIX = "_icp"
RANSAC_SUFFIX = "_icpr"
WEIGHTED_SUFFIX = "w"
_SUFFIXES = {ICP: REFINED_SUFFIX, ICP_RANSAC: RANSAC_SUFFIX}


@dataclass(frozen=True)
class RefineConfig:
    """Every tunable of the re-solve, in one immutable place."""

    mode: str = REFINE_NONE
    gates_m: Tuple[float, ...] = DEFAULT_GATES_M
    min_pairs: int = DEFAULT_MIN_PAIRS
    inlier_m: float = DEFAULT_INLIER_M
    search_m: float = SEARCH_OFF
    # The final exact fit over the hard pairs weighted by the same
    # confidence-based precisions the soft solve uses (``pair_weights``),
    # instead of unweighted. Where detections are precise and the
    # association already sharp, the unweighted hard fit can lose to the
    # weighted soft one; this keeps the decided correspondence and the
    # right weights.
    weighted: bool = False

    def __post_init__(self) -> None:
        if self.mode not in REFINE_MODES:
            raise ValueError(f"mode must be one of {REFINE_MODES}, got {self.mode!r}")
        if not (self.inlier_m > 0.0 and math.isfinite(self.inlier_m)):
            raise ValueError(f"inlier_m must be finite and positive, got {self.inlier_m}")
        if not (self.search_m >= 0.0 and math.isfinite(self.search_m)):
            raise ValueError(f"search_m must be finite and non-negative (0 = off), got {self.search_m}")
        if len(self.gates_m) == 0:
            raise ValueError("gates_m needs at least one gate")
        if any(not (g > 0.0 and math.isfinite(g)) for g in self.gates_m):
            raise ValueError(f"every gate must be finite and positive, got {self.gates_m}")
        if self.min_pairs < 1:
            raise ValueError(f"min_pairs must be >= 1, got {self.min_pairs}")

    @property
    def enabled(self) -> bool:
        return self.mode != REFINE_NONE

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "gates_m": list(self.gates_m),
            "min_pairs": self.min_pairs,
            "inlier_m": self.inlier_m,
            "search_m": self.search_m,
            "weighted": self.weighted,
            "enabled": self.enabled,
        }


def refined_name(condition: str, mode: str = ICP, weighted: bool = False) -> str:
    """The refined arm's condition name, e.g. ``alignformer`` -> ``alignformer_icp``
    (``_icpr`` in RANSAC mode, a trailing ``w`` when the final fit is weighted)."""
    suffix = _SUFFIXES[mode] + (WEIGHTED_SUFFIX if weighted else "")
    return condition + suffix


def _moved(centres: Tensor, psi: Tensor, t: Tensor) -> Tensor:
    """``R(psi) centres + t`` for one sample: ``(N, 2)``."""
    cos, sin = torch.cos(psi), torch.sin(psi)
    x, y = centres[:, 0], centres[:, 1]
    return torch.stack([cos * x - sin * y, sin * x + cos * y], dim=-1) + t


def mutual_nearest(ego: Tensor, cav: Tensor, gate_m: float) -> Tuple[Tensor, Tensor]:
    """Indices of the mutually-nearest ``(ego, cav)`` centre pairs closer than ``gate_m``.

    One-to-one by construction: a CAV object is taken only by the ego object
    it is nearest to, and only if that ego object is nearest to it in turn.
    """
    if ego.shape[0] == 0 or cav.shape[0] == 0:
        empty = torch.zeros(0, dtype=torch.long, device=ego.device)
        return empty, empty
    distance = torch.cdist(ego, cav)
    nearest_cav = distance.argmin(dim=1)
    nearest_ego = distance.argmin(dim=0)
    ego_idx = torch.arange(ego.shape[0], device=ego.device)
    mutual = nearest_ego[nearest_cav] == ego_idx
    within = distance[ego_idx, nearest_cav] < gate_m
    keep = mutual & within
    return ego_idx[keep], nearest_cav[keep]


def pair_weights(
    variance_model, ego_scores: Tensor, cav_scores: Tensor, ego_idx: Tensor, cav_idx: Tensor, heading_lambda: float
) -> Tensor:
    """``(1, 2K)`` Kabsch weights for the hard pairs: centres first, then heading
    points, each the inverse of the pair's disagreement variance under the
    fitted model, exactly as the soft solve weights its correspondences;
    ``mode="none"`` gives ones."""
    count = int(ego_idx.numel())
    if getattr(variance_model, "mode", "none") == "none":
        return torch.ones(1, 2 * count, dtype=ego_scores.dtype, device=ego_scores.device)
    cav_centre, cav_heading = variance_model.detection_variances(cav_scores[cav_idx].unsqueeze(0))
    rho_centre, rho_heading = variance_model.precisions(
        ego_scores[ego_idx].unsqueeze(0), cav_centre, cav_heading, heading_lambda
    )
    if variance_model.mode == "scalar":
        rho_heading = rho_centre
    return torch.cat([rho_centre, rho_heading], dim=1)


def _exact_solve(
    ego_boxes: Tensor, cav_boxes: Tensor, ego_idx: Tensor, cav_idx: Tensor, heading_lambda: float,
    weights: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """SE(2) fit over the hard pairs, heading-augmented and folded; unweighted
    unless ``weights`` (``(1, 2K)``, from :func:`pair_weights`) is given."""
    ego_yaw = ego_boxes[ego_idx, BOX_YAW]
    cav_yaw = cav_boxes[cav_idx, BOX_YAW]
    # Fold each CAV heading onto its ego partner's half-plane (the detector
    # reports an axis, not a direction).
    sign = heading_orientation(ego_yaw.unsqueeze(0), cav_yaw.unsqueeze(0))[0].diagonal()
    folded_yaw = torch.where(sign < 0, cav_yaw + math.pi, cav_yaw)
    target = augment_with_heading(ego_boxes[ego_idx, :2].unsqueeze(0), ego_yaw.unsqueeze(0), heading_lambda)
    source = augment_with_heading(cav_boxes[cav_idx, :2].unsqueeze(0), folded_yaw.unsqueeze(0), heading_lambda)
    if weights is None:
        weights = torch.ones(1, target.shape[1], dtype=target.dtype, device=target.device)
    psi, t = weighted_se2_kabsch(target, source, weights)
    return psi[0], t[0]


def se2_from_two_pairs(target: Tensor, source: Tensor) -> Tuple[Tensor, Tensor]:
    """The SE(2) mapping two ``source`` centres onto two ``target`` centres: ``(psi, t)``.

    The rotation aligns the chord between the two points; the translation
    moves the rotated midpoint onto the target midpoint. Closed form, no
    SVD, so hundreds of minimal samples cost nothing.
    """
    chord_t, chord_s = target[1] - target[0], source[1] - source[0]
    psi = torch.atan2(chord_t[1], chord_t[0]) - torch.atan2(chord_s[1], chord_s[0])
    cos, sin = torch.cos(psi), torch.sin(psi)
    mid_s, mid_t = source.mean(dim=0), target.mean(dim=0)
    rotated = torch.stack([cos * mid_s[0] - sin * mid_s[1], sin * mid_s[0] + cos * mid_s[1]])
    return psi, mid_t - rotated


def se2_from_pair_samples(target: Tensor, source: Tensor) -> Tuple[Tensor, Tensor]:
    """:func:`se2_from_two_pairs` over ``(S, 2, 2)`` samples at once: ``(psi (S,), t (S, 2))``."""
    chord_t, chord_s = target[:, 1] - target[:, 0], source[:, 1] - source[:, 0]
    psi = torch.atan2(chord_t[:, 1], chord_t[:, 0]) - torch.atan2(chord_s[:, 1], chord_s[:, 0])
    cos, sin = torch.cos(psi), torch.sin(psi)
    mid_s, mid_t = source.mean(dim=1), target.mean(dim=1)
    rotated = torch.stack([cos * mid_s[:, 0] - sin * mid_s[:, 1], sin * mid_s[:, 0] + cos * mid_s[:, 1]], dim=-1)
    return psi, mid_t - rotated


def _best_consensus(target: Tensor, source: Tensor, samples: Tensor, inlier_m: float) -> Tensor:
    """``(K,)`` bool: the inliers of the sample with the most pairs within
    ``inlier_m``, ties to the smallest inlier residual sum, then the first
    sample. Every sample is scored in one batch."""
    psi, t = se2_from_pair_samples(target[samples], source[samples])
    cos, sin = torch.cos(psi)[:, None], torch.sin(psi)[:, None]
    x, y = source[None, :, 0], source[None, :, 1]
    moved = torch.stack([cos * x - sin * y, sin * x + cos * y], dim=-1) + t[:, None, :]
    residual = (moved - target[None]).norm(dim=-1)
    inliers = residual < inlier_m
    count = inliers.sum(dim=1)
    total = (residual * inliers).sum(dim=1)
    candidates = count == count.max()
    total = torch.where(candidates, total, torch.full_like(total, float("inf")))
    return inliers[int(torch.argmin(total))]


def _minimal_samples(count: int, device) -> Tensor:
    """``(S, 2)`` index pairs: every pair when there are few, a fixed draw when many."""
    if count <= EXHAUSTIVE_PAIRS:
        return torch.combinations(torch.arange(count, device=device), r=2)
    generator = torch.Generator(device="cpu").manual_seed(count)
    draws = torch.randint(0, count, (RANSAC_SAMPLES, 2), generator=generator)
    draws = draws[draws[:, 0] != draws[:, 1]]
    return draws.to(device)


def robust_exact_solve(
    ego_boxes: Tensor, cav_boxes: Tensor, ego_idx: Tensor, cav_idx: Tensor, heading_lambda: float, inlier_m: float,
    weights: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """RANSAC over the hard pairs, then the exact heading-augmented fit over the consensus.

    Returns ``(psi, t, inliers)``. Minimal samples are two pairs; the
    consensus is the sample with the most pairs within ``inlier_m`` of their
    partner, ties broken by the inlier residual sum. Fewer than two inliers
    (never, with two-pair samples) falls back to every pair.
    """
    target, source = ego_boxes[ego_idx, :2], cav_boxes[cav_idx, :2]
    count = target.shape[0]
    if count < 3:
        psi, t = _exact_solve(ego_boxes, cav_boxes, ego_idx, cav_idx, heading_lambda, weights)
        return psi, t, torch.ones(count, dtype=torch.bool, device=target.device)
    samples = _minimal_samples(count, target.device)
    best_inliers = _best_consensus(target, source, samples, inlier_m)
    if int(best_inliers.sum()) < 2:
        best_inliers = torch.ones(count, dtype=torch.bool, device=target.device)
    kept = None if weights is None else torch.cat([weights[:, :count][:, best_inliers], weights[:, count:][:, best_inliers]], dim=1)
    psi, t = _exact_solve(ego_boxes, cav_boxes, ego_idx[best_inliers], cav_idx[best_inliers], heading_lambda, kept)
    return psi, t, best_inliers


def _one_to_one_inliers(target: Tensor, moved: Tensor, inlier_m: float) -> Tuple[Tensor, Tensor]:
    """Mutually-nearest pairs within ``inlier_m`` under one candidate pose."""
    return mutual_nearest(target, moved, inlier_m)


def candidate_search(
    psi: Tensor, t: Tensor, ego_boxes: Tensor, cav_boxes: Tensor, *, radius_m: float, inlier_m: float,
    heading_lambda: float, min_pairs: int,
) -> Tuple[Tensor, Tensor, int]:
    """RANSAC over every ego/CAV candidate pair within ``radius_m`` of the soft estimate.

    Returns ``(psi, t, consensus size)``; a consensus below ``min_pairs``
    returns the input pose with size 0. Samples are two candidate pairs with
    distinct ego and distinct CAV boxes, drawn by a fixed generator so the
    answer is a function of the inputs alone.
    """
    target, source = ego_boxes[:, :2], cav_boxes[:, :2]
    if target.shape[0] < min_pairs or source.shape[0] < min_pairs:
        return psi, t, 0
    distance = torch.cdist(target, _moved(source, psi, t))
    ego_c, cav_c = torch.nonzero(distance < radius_m, as_tuple=True)
    if ego_c.numel() < 2:
        return psi, t, 0
    generator = torch.Generator(device="cpu").manual_seed(int(ego_c.numel()))
    draws = torch.randint(0, ego_c.numel(), (SEARCH_SAMPLES, 2), generator=generator).to(target.device)
    distinct = (ego_c[draws[:, 0]] != ego_c[draws[:, 1]]) & (cav_c[draws[:, 0]] != cav_c[draws[:, 1]])
    best_n, best_sum, best = 0, float("inf"), None
    ego_list, cav_list = ego_c.tolist(), cav_c.tolist()
    for a, b in draws[distinct].tolist():
        rows, cols = [ego_list[a], ego_list[b]], [cav_list[a], cav_list[b]]
        s_psi, s_t = se2_from_two_pairs(target[rows], source[cols])
        e_idx, c_idx = _one_to_one_inliers(target, _moved(source, s_psi, s_t), inlier_m)
        n = int(e_idx.numel())
        if n < best_n:
            continue
        total = float((_moved(source[c_idx], s_psi, s_t) - target[e_idx]).norm(dim=-1).sum())
        if n > best_n or total < best_sum:
            best_n, best_sum, best = n, total, (e_idx, c_idx)
    if best is None or best_n < min_pairs:
        return psi, t, 0
    s_psi, s_t = _exact_solve(ego_boxes, cav_boxes, best[0], best[1], heading_lambda)
    return s_psi, s_t, best_n


def _refine_one(
    psi: Tensor, t: Tensor, ego_boxes: Tensor, cav_boxes: Tensor, config: RefineConfig, heading_lambda: float,
    weigh=None,
) -> Tuple[Tensor, Tensor, bool, int]:
    """ICP over one sample's valid boxes; ``(psi, t, engaged, consensus)``.

    ``consensus`` is the number of hard pairs behind the final fit (the
    RANSAC inliers in that mode), zero when nothing engaged. ``weigh`` maps
    ``(ego_idx, cav_idx)`` to the stage's Kabsch weights, or is ``None``."""
    ego_centres, cav_centres = ego_boxes[:, :2], cav_boxes[:, :2]
    engaged, consensus = False, 0
    if config.search_m > SEARCH_OFF:
        first = mutual_nearest(ego_centres, _moved(cav_centres, psi, t), config.gates_m[0])[0]
        if first.numel() < config.min_pairs:
            psi, t, found = candidate_search(psi, t, ego_boxes, cav_boxes, radius_m=config.search_m,
                                             inlier_m=config.inlier_m, heading_lambda=heading_lambda, min_pairs=config.min_pairs)
            engaged, consensus = found > 0, found
    for gate in config.gates_m:
        ego_idx, cav_idx = mutual_nearest(ego_centres, _moved(cav_centres, psi, t), gate)
        if ego_idx.numel() < config.min_pairs:
            break
        weights = None if weigh is None else weigh(ego_idx, cav_idx)
        if config.mode == ICP_RANSAC:
            psi, t, inliers = robust_exact_solve(
                ego_boxes, cav_boxes, ego_idx, cav_idx, heading_lambda, config.inlier_m, weights
            )
            consensus = int(inliers.sum())
        else:
            psi, t = _exact_solve(ego_boxes, cav_boxes, ego_idx, cav_idx, heading_lambda, weights)
            consensus = int(ego_idx.numel())
        engaged = True
    return psi, t, engaged, consensus


def _weigher(batch, b: int, ego_mask, cav_mask, variance_model, heading_lambda: float):
    """The per-stage weight function for sample ``b`` (:func:`pair_weights` over its valid scores)."""
    ego_scores, cav_scores = batch["ego_scores"][b], batch["cav_scores"][b]
    if ego_mask is not None:
        ego_scores = ego_scores[ego_mask[b]]
    if cav_mask is not None:
        cav_scores = cav_scores[cav_mask[b]]
    return lambda ego_idx, cav_idx: pair_weights(variance_model, ego_scores, cav_scores, ego_idx, cav_idx, heading_lambda)


def icp_refine(estimate, batch, config: RefineConfig, heading_lambda: float, variance_model=None):
    """A **new** estimate with ``(psi, t)`` re-solved exactly; the decision inputs unchanged.

    A disabled ``config`` returns the input object itself. A sample the
    evidence floor declines keeps its ``(psi, t)`` bit for bit. A
    ``weighted`` config needs the ``variance_model`` the soft solve used and
    the batch's ``ego_scores`` / ``cav_scores``.
    """
    if not config.enabled:
        return estimate
    if config.weighted and variance_model is None:
        raise ValueError("a weighted re-solve needs the variance model the soft solve used (variance_model is None)")
    ego_mask = batch.get("ego_mask")
    cav_mask = batch.get("cav_mask")
    psis, ts, engaged, consensus = [], [], [], []
    for b in range(estimate.psi.shape[0]):
        ego_boxes = batch["ego_boxes"][b]
        cav_boxes = batch["cav_boxes"][b]
        if ego_mask is not None:
            ego_boxes = ego_boxes[ego_mask[b]]
        if cav_mask is not None:
            cav_boxes = cav_boxes[cav_mask[b]]
        weigh = _weigher(batch, b, ego_mask, cav_mask, variance_model, heading_lambda) if config.weighted else None
        psi, t, did, count = _refine_one(
            estimate.psi[b], estimate.t[b], ego_boxes, cav_boxes, config, heading_lambda, weigh
        )
        psis.append(psi)
        ts.append(t)
        engaged.append(did)
        consensus.append(count)
    return replace(
        estimate,
        psi=torch.stack(psis),
        t=torch.stack(ts),
        refined=torch.tensor(engaged, dtype=torch.bool, device=estimate.psi.device),
        consensus=torch.tensor(consensus, dtype=torch.long, device=estimate.psi.device),
    )


# ---------------------------------------------------------------------------
# The agreement rule
# ---------------------------------------------------------------------------
#
# With the re-solve in hand every pair has two estimates of its correction:
# the soft weighted fit and the exact hard fit. They are built from the same
# detections but reach the answer by different routes (a Sinkhorn mixture
# against a decided correspondence). Two routes landing in the same place
# within detection noise is evidence the correction is real; landing apart is
# grounds to abstain. On a sample the re-solve did not engage on there is no
# second estimate, and the rule hands back the decision arm it wraps.

AGREEMENT_SUFFIX = "_agree_"


@dataclass(frozen=True)
class AgreementConfig:
    """``tolerance_m``: the soft/exact disagreement, in metres, above which the pair abstains,
    unless ``consensus_floor`` > 0 and the exact fit rests on at least that many hard pairs.

    The floor is the exact fit's own evidence: a soft estimate can be metres
    off on a pair the re-solve lands within fractions of a metre on with a
    large consensus, and the plain rule would otherwise abstain on exactly
    those."""

    tolerance_m: float
    consensus_floor: int = 0
    # Gated: the rule only ever chooses BETWEEN the exact fit and abstention
    # on a pair the decision arm corrected; a pair it abstained on stays
    # uncorrected. Ungated (the original), the rule answers every engaged
    # pair on which the two fits agree, which can over-engage on already-
    # clean data (no pose error to correct in the first place).
    gated: bool = False

    def __post_init__(self) -> None:
        if not (self.tolerance_m > 0.0 and math.isfinite(self.tolerance_m)):
            raise ValueError(f"tolerance_m must be finite and positive, got {self.tolerance_m}")
        if self.consensus_floor < 0:
            raise ValueError(f"consensus_floor must be >= 0 (0 = off), got {self.consensus_floor}")

    def to_dict(self) -> dict:
        return {"tolerance_m": self.tolerance_m, "consensus_floor": self.consensus_floor, "gated": self.gated}


GATED_SUFFIX = "_g"


def agreement_name(decision_arm: str, tolerance_m: float, consensus_floor: int = 0, gated: bool = False) -> str:
    floor = f"_c{consensus_floor}" if consensus_floor > 0 else ""
    return f"{decision_arm}{AGREEMENT_SUFFIX}{tolerance_m:g}{floor}{GATED_SUFFIX if gated else ''}"


def _wrap(angle: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def disagreement_m(soft, exact, heading_lambda: float) -> Tensor:
    """``(B,)`` metres: translation gap plus ``heading_lambda`` times the heading gap.

    The same metre-equivalent the heading augmentation uses, so one
    tolerance covers both components.
    """
    translation = (exact.t - soft.t).norm(dim=-1)
    heading = _wrap(exact.psi - soft.psi).abs()
    return translation + heading_lambda * heading


def agree(soft, exact, fallback, config: AgreementConfig, heading_lambda: float):
    """A **new** estimate: the exact fit where the two agree, zero where they
    disagree, ``fallback`` where the re-solve did not engage."""
    if exact.refined is None:
        raise ValueError("the agreement rule needs an estimate the re-solve produced (refined is None)")
    engaged = exact.refined
    agreeing = engaged & (disagreement_m(soft, exact, heading_lambda) <= config.tolerance_m)
    if config.consensus_floor > 0:
        if exact.consensus is None:
            raise ValueError("a consensus floor needs an estimate that carries its consensus (consensus is None)")
        agreeing = agreeing | (engaged & (exact.consensus >= config.consensus_floor))
    if config.gated:
        corrected = (fallback.t != 0).any(dim=-1) | (fallback.psi != 0)
        agreeing = agreeing & corrected
    zero_t, zero_psi = torch.zeros_like(exact.t), torch.zeros_like(exact.psi)
    t = torch.where(engaged.unsqueeze(-1), torch.where(agreeing.unsqueeze(-1), exact.t, zero_t), fallback.t)
    psi = torch.where(engaged, torch.where(agreeing, exact.psi, zero_psi), fallback.psi)
    return replace(fallback, psi=psi, t=t)


__all__ = [
    "AGREEMENT_SUFFIX",
    "AgreementConfig",
    "agree",
    "agreement_name",
    "disagreement_m",
    "pair_weights",
    "GATED_SUFFIX",
    "WEIGHTED_SUFFIX",
    "DEFAULT_GATES_M",
    "DEFAULT_INLIER_M",
    "DEFAULT_MIN_PAIRS",
    "ICP",
    "ICP_RANSAC",
    "RANSAC_SUFFIX",
    "SEARCH_OFF",
    "candidate_search",
    "robust_exact_solve",
    "se2_from_two_pairs",
    "se2_from_pair_samples",
    "REFINED_SUFFIX",
    "REFINE_MODES",
    "REFINE_NONE",
    "RefineConfig",
    "icp_refine",
    "mutual_nearest",
    "refined_name",
]
