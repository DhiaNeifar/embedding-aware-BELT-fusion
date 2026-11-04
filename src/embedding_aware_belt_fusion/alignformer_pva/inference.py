"""Inference-time glue between the evaluation harness and AlignFormer-PVA.

``belt_fusion.py`` (the shared OPV2V evaluation pipeline) assembles one
``simple_detections_by_cav`` dict per frame: ``{"boxes": (N, 7), "scores":
(N,), "agent_id": cav_id, "oracle_object_ids": [...]}`` per CAV, already
projected into the ego frame via each CAV's own reported (possibly noisy)
pose. The existing "AlignFormer (ours)" row corrects that per-source pose
error with trusted PQ-STF embedding matches feeding ``estimate_se2_ransac``
(``integration/localization.py``). This module is the geometry-only,
embedding-free replacement: box geometry alone feeds AlignFormerB's
Propose (Sinkhorn/Kabsch) stage, the RANSAC-based Verify stage
(``alignformer_pva.refine.icp_refine``) hardens it, and the Wald-test
Abstain stage (``alignformer_pva.abstain.decide``) decides whether the
resulting correction is trusted at all. The accepted correction is applied
with the exact same ``apply_se2_correction`` mechanism the RANSAC path
uses, so the corrected ``simple_detections_by_cav`` flows through the
existing, unchanged association/fusion/AP code.

Nothing here needs ground-truth match labels (``ego_match``/``cav_match``):
those are read only by ``alignformer_pva.dataset`` during training.
``AlignFormerB.forward`` itself never reads them -- it only consumes
``*_boxes``/``*_scores``/``*_mask``/``*_embeddings``, confirmed by
``model.py``'s ``_tokens``/``_encode``.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from embedding_aware_belt_fusion.alignformer_pva import abstain
from embedding_aware_belt_fusion.alignformer_pva.model import AlignFormerB, PoseEstimate
from embedding_aware_belt_fusion.alignformer_pva.refine import (
    AgreementConfig,
    RefineConfig,
    agree,
    icp_refine,
)
from embedding_aware_belt_fusion.alignformer_pva.robust import RobustSolveConfig
from embedding_aware_belt_fusion.alignformer_pva.shrinkage import (
    ShrinkageCalibration,
    shrink,
)
from embedding_aware_belt_fusion.embeddings.context_embedding import ContextEmbedding
from embedding_aware_belt_fusion.integration.belt_fusion import (
    _constrained_assignment,
    fuse_groups_score_weighted,
)
from embedding_aware_belt_fusion.integration.localization import apply_se2_correction


def build_pairwise_batch(
    ego_detection: Mapping[str, Tensor],
    cav_detection: Mapping[str, Tensor],
    ego_embeddings: Optional[Tensor] = None,
    cav_embeddings: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Build one ``AlignFormerB.forward`` batch (size 1) from live detections.

    ``ego_detection``/``cav_detection`` are one CAV's entry in
    ``simple_detections_by_cav``: ``{"boxes": (N, 7), "scores": (N,), ...}``,
    boxes already in the ego frame. This is inference, not training, so:

    * No noise is injected (the detections are already whatever the live
      detector + current pose produced).
    * No ``ego_match``/``cav_match`` ground-truth assignment is built --
      ``AlignFormerB.forward`` never reads those fields (only
      ``dataset.py``'s training path does).
    * Masks are all-True: a single live pair needs no padding, and
      ``AlignFormerTrunk``/``AlignFormerB`` tolerate batch size 1 with an
      all-True mask (``tests/test_alignformer_pva_model.py``) as well as a
      zero-length object set on either side (``_is_empty`` guard).

    ``ego_embeddings``/``cav_embeddings``: optional ``(N, embed_dim)``
    precomputed per-object descriptors (:func:`estimate_correction`'s
    ``embedding_encoder`` path, Phase 1 of the richer-embedding plan).
    ``None`` (the default on every existing call site) reproduces the
    original zero-width, geometry-only behavior bit for bit -- this backend
    stays geometry-only unless a caller explicitly opts into a real
    embedding.
    """
    ego_boxes = ego_detection["boxes"]
    cav_boxes = cav_detection["boxes"]
    ego_scores = ego_detection["scores"]
    cav_scores = cav_detection["scores"]
    ego_count, cav_count = len(ego_boxes), len(cav_boxes)
    device, dtype = ego_boxes.device, ego_boxes.dtype
    if ego_embeddings is None:
        ego_embeddings = ego_boxes.new_zeros((ego_count, 0))
    if cav_embeddings is None:
        cav_embeddings = ego_boxes.new_zeros((cav_count, 0))
    return {
        "ego_boxes": ego_boxes.unsqueeze(0),
        "cav_boxes": cav_boxes.to(device=device, dtype=dtype).unsqueeze(0),
        "ego_scores": ego_scores.unsqueeze(0),
        "cav_scores": cav_scores.to(device=device, dtype=dtype).unsqueeze(0),
        "ego_mask": torch.ones(1, ego_count, dtype=torch.bool, device=device),
        "cav_mask": torch.ones(1, cav_count, dtype=torch.bool, device=device),
        "ego_embeddings": ego_embeddings.to(device=device, dtype=dtype).unsqueeze(0),
        "cav_embeddings": cav_embeddings.to(device=device, dtype=dtype).unsqueeze(0),
    }


def _embed_detection(encoder: ContextEmbedding, detection: Mapping[str, Tensor]) -> Tensor:
    """Run ``encoder`` over one agent's own padded cell set.

    Requires ``detection`` to carry ``cell_tokens``/``cell_positions``/
    ``cell_footprint``/``cell_mask`` alongside its ``boxes``/``scores`` --
    ``evaluation/belt_fusion.py``'s ``_context_embedding_cells`` (or any
    equivalent live source) attaches these. A missing key here raises a
    plain ``KeyError`` immediately: a caller that passes an
    ``embedding_encoder`` without actually attaching cell data is a caller
    bug worth surfacing right away, not a case to silently fall back to
    geometry-only for.
    """
    return encoder(
        detection["cell_tokens"],
        detection["cell_positions"],
        detection["cell_footprint"],
        detection["cell_mask"],
        detection["boxes"],
        detection["scores"],
    )


class CorrectionDiagnostics(dict):
    """Lightweight record of one CAV's Propose-Verify-Abstain outcome.

    A plain ``dict`` subclass (rather than a dataclass) so it round-trips
    into ``metrics.json`` without a custom encoder; keys are ``attempted``
    (bool), ``accepted`` (bool), ``statistic``/``dof`` (float or ``None``,
    the Wald test's own quantities), and ``refined`` (bool, whether the
    Verify/ICP stage actually engaged for this pair).
    """


def estimate_correction(
    model: AlignFormerB,
    ego_detection: Mapping[str, Tensor],
    cav_detection: Mapping[str, Tensor],
    *,
    refine_config: RefineConfig,
    abstain_config: abstain.AbstentionConfig,
    agreement_config: Optional[AgreementConfig] = None,
    embedding_encoder: Optional[ContextEmbedding] = None,
    robust_config: Optional[RobustSolveConfig] = None,
    shrinkage_config: Optional[ShrinkageCalibration] = None,
) -> Tuple[dict | None, CorrectionDiagnostics]:
    """Run Propose -> Verify -> Abstain for one ego/CAV pair.

    Propose: ``model(batch, statistic=True)`` -- the soft Sinkhorn
    correspondence and closed-form weighted Kabsch solve, plus the Wald
    statistic/dof/precision needed to gate it.
    Verify: ``refine.icp_refine`` hardens ``(psi, t)`` with mutually-nearest
    hard correspondences and (in ``icp_ransac`` mode) a RANSAC consensus fit,
    while leaving the soft fit's decision inputs (``offset_statistic`` /
    ``offset_dof`` / ``offset_precision``) untouched -- see ``icp_refine``'s
    own docstring: it changes only ``(psi, t)``.
    Abstain: two rules, in sequence, mirroring the labmate's own
    ``sec/3_method.tex`` "Abstain" section.

    * The Wald test (``abstain.decide``) always runs against the Propose
      stage's own statistic (unchanged by Verify).
    * ``agreement_config``, when given, additionally runs the Verify/Propose
      **agreement rule** (``refine.agree``): the Verify-stage exact fit
      replaces the Wald-gated proposal only where the two agree within
      ``agreement_config.tolerance_m`` (or the exact fit's own hard-pair
      consensus clears ``agreement_config.consensus_floor``); otherwise the
      Wald-gated proposal (the ``fallback``) is used -- exactly ``agree``'s
      own contract. ``None`` (the default, every pre-existing call site)
      skips this rule entirely and reproduces the original, simpler
      behavior bit for bit: the Wald test alone gates whatever ``icp_refine``
      produced.

    ``embedding_encoder``, when given (Phase 1 of the richer-embedding
    plan), computes real per-object embeddings from each side's own cell
    data (:func:`_embed_detection`) instead of the zero-width geometry-only
    placeholders -- ``None`` (the default, every pre-existing call site)
    reproduces the original geometry-only behavior exactly.

    Returns ``(correction, diagnostics)``. ``correction`` is ``None`` only
    when there is nothing to estimate from at all (either side detected zero
    objects this frame); otherwise it is always a dict with the same
    ``rotation``/``translation``/``yaw_correction`` keys
    ``apply_se2_correction`` expects (mirroring ``estimate_se2_ransac``'s
    output shape), so the caller can apply it unconditionally when
    ``diagnostics["accepted"]`` is true and must leave the detection
    untouched otherwise.
    """
    ego_embeddings = cav_embeddings = None
    if embedding_encoder is not None:
        ego_embeddings = _embed_detection(embedding_encoder, ego_detection)
        cav_embeddings = _embed_detection(embedding_encoder, cav_detection)
    batch = build_pairwise_batch(ego_detection, cav_detection, ego_embeddings, cav_embeddings)
    if batch["ego_boxes"].shape[1] == 0 or batch["cav_boxes"].shape[1] == 0:
        return None, CorrectionDiagnostics(
            attempted=False, accepted=False, statistic=None, dof=None, refined=False
        )

    estimate: PoseEstimate = model(batch, robust=robust_config, statistic=True)
    if shrinkage_config is not None:
        estimate = shrink(estimate, shrinkage_config)
    refined = icp_refine(
        estimate,
        batch,
        refine_config,
        heading_lambda=model.heading_lambda,
        variance_model=model.variance_model if refine_config.weighted else None,
    )
    if agreement_config is not None:
        fallback = abstain.decide(estimate, abstain_config)
        decided = agree(
            estimate, refined, fallback, agreement_config, heading_lambda=model.heading_lambda
        )
        accepted = bool(torch.any(decided.psi != 0.0) or torch.any(decided.t != 0.0))
    else:
        decided = abstain.decide(refined, abstain_config)
        accepted = (
            bool(torch.any(decided.psi != 0.0) or torch.any(decided.t != 0.0))
            if abstain_config.enabled
            else True
        )

    psi = decided.psi[0]
    cosine, sine = torch.cos(psi), torch.sin(psi)
    rotation = torch.stack(
        [torch.stack([cosine, -sine]), torch.stack([sine, cosine])]
    )
    correction = {
        "rotation": rotation,
        "translation": decided.t[0],
        "yaw_correction": psi,
    }
    diagnostics = CorrectionDiagnostics(
        attempted=True,
        accepted=accepted,
        statistic=(
            float(refined.offset_statistic[0])
            if refined.offset_statistic is not None
            else None
        ),
        dof=(
            float(refined.offset_dof[0])
            if refined.offset_dof is not None
            else None
        ),
        refined=bool(refined.refined[0]) if refined.refined is not None else False,
    )
    return correction, diagnostics


def correct_simple_detections(
    model: AlignFormerB,
    simple_detections_by_cav: Mapping[str, Mapping[str, Tensor]],
    cav_order: Sequence[str],
    *,
    refine_config: RefineConfig,
    abstain_config: abstain.AbstentionConfig,
    agreement_config: Optional[AgreementConfig] = None,
    embedding_encoder: Optional[ContextEmbedding] = None,
    robust_config: Optional[RobustSolveConfig] = None,
    shrinkage_config: Optional[ShrinkageCalibration] = None,
) -> Tuple[Dict[str, Mapping[str, Tensor]], Dict[str, CorrectionDiagnostics]]:
    """Apply a per-CAV AlignFormer-PVA correction to one frame's simple detections.

    Ego's own entry is never corrected (there is nothing to correct it
    against) and is passed through unchanged, exactly like the existing
    RANSAC path's ``ego_detection``. Every other CAV goes through
    :func:`estimate_correction` against ego; a correction is applied with
    ``apply_se2_correction`` -- the same mechanism ``--ransac-localization-
    correction`` uses -- only when the Abstain stage accepted it. A CAV whose
    correction is rejected (or that had nothing to estimate from) keeps its
    original, uncorrected entry.

    ``agreement_config`` is forwarded to :func:`estimate_correction`
    unchanged; ``None`` (the default) skips the agreement rule and preserves
    the original Wald-only behavior exactly, as every pre-existing call site
    still gets.

    ``embedding_encoder`` is forwarded to :func:`estimate_correction`
    unchanged; ``None`` (the default) preserves the original geometry-only
    behavior exactly, as every pre-existing call site still gets.
    """
    if "ego" not in simple_detections_by_cav:
        raise RuntimeError("Late-fusion batch has no ego agent")
    ego_detection = simple_detections_by_cav["ego"]
    corrected = dict(simple_detections_by_cav)
    diagnostics: Dict[str, CorrectionDiagnostics] = {}
    for cav_id in cav_order:
        if cav_id == "ego":
            continue
        cav_detection = simple_detections_by_cav[cav_id]
        correction, stats = estimate_correction(
            model,
            ego_detection,
            cav_detection,
            refine_config=refine_config,
            abstain_config=abstain_config,
            agreement_config=agreement_config,
            embedding_encoder=embedding_encoder,
            robust_config=robust_config,
            shrinkage_config=shrinkage_config,
        )
        diagnostics[cav_id] = stats
        if correction is not None and stats["accepted"]:
            corrected[cav_id] = apply_se2_correction(cav_detection, correction)
    return corrected, diagnostics


def associate_simple_by_geometry(
    detections: Sequence[Mapping[str, Tensor]],
    *,
    maximum_distance: float = 5.0,
) -> list:
    """Greedy one-to-one multi-agent grouping by ego-frame centre distance.

    This is the plain-geometry analogue of
    ``integration.belt_fusion.associate_by_geometry`` for detections that
    carry only ``boxes``/``scores`` (no BELT evidence/covariance): a group's
    representative is the unweighted mean of its members' centres rather
    than a precision-weighted one, since simple detections have no precision
    to weight by. Reuses the same Hungarian-with-explicit-no-match solver
    (``_constrained_assignment``) the embedding- and covariance-based
    associators already use, so no new assignment logic is introduced.

    AlignFormer-PVA corrects each source CAV's pose but does not itself
    decide cross-agent duplicate grouping (that is a separate concern from
    pose correction); this is the grouping step its corrected detections
    need before ``fuse_groups_score_weighted`` can produce one box per
    physical object, exactly as the existing simple/score-weighted fusion
    rows already do for their own association method.
    """
    groups: list = []
    for detection in detections:
        members = [
            {
                "box": detection["boxes"][index],
                "score": detection["scores"][index],
                "agent_id": detection.get("agent_id"),
            }
            for index in range(len(detection["boxes"]))
        ]
        if not groups:
            groups = [[member] for member in members]
            continue
        if not members:
            continue
        representatives = torch.stack(
            [
                torch.stack([member["box"][:2] for member in group]).mean(dim=0)
                for group in groups
            ]
        )
        centers = torch.stack([member["box"][:2] for member in members])
        distance = torch.cdist(representatives, centers)
        valid = distance <= maximum_distance
        matched_members = set()
        for row, column in _constrained_assignment(
            distance, valid, unmatched_cost=maximum_distance + 1e-3
        ):
            groups[row].append(members[column])
            matched_members.add(column)
        groups.extend(
            [member]
            for index, member in enumerate(members)
            if index not in matched_members
        )
    return groups


def fuse_corrected_simple_detections(
    corrected_detections_by_cav: Mapping[str, Mapping[str, Tensor]],
    cav_order: Sequence[str],
    *,
    maximum_distance: float = 5.0,
) -> Dict[str, Tensor]:
    """Associate corrected simple detections and fuse by detector-score weighting.

    Thin wrapper over :func:`associate_simple_by_geometry` +
    ``fuse_groups_score_weighted`` -- the same, unmodified fusion function
    the existing ``--trackformer-box-fusion score-weighted`` row uses -- so
    producing a comparable AP row for this backend requires no new fusion
    logic, only the new association step above.
    """
    detections = [corrected_detections_by_cav[cav_id] for cav_id in cav_order]
    groups = associate_simple_by_geometry(detections, maximum_distance=maximum_distance)
    return fuse_groups_score_weighted(groups)
