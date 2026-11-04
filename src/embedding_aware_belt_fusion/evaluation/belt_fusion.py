"""Evaluate OpenCOOD naive and BELT-style late fusion on the same frames."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from embedding_aware_belt_fusion.features import (
    assign_proposals_to_ground_truth,
    decode_local_proposals,
    proposal_roi_cell_indices,
)
from embedding_aware_belt_fusion.embeddings.spatial_trackformer import (
    SpatialTrackFormer,
)
from embedding_aware_belt_fusion.embeddings.all_agent_association import (
    SymmetricQueryStatePairHead,
)
from embedding_aware_belt_fusion.embeddings.geometry import transform_boxes_to_ego
from embedding_aware_belt_fusion.integration.localization import (
    apply_se2_correction,
    correspondences_by_source,
    estimate_se2_ransac,
)
from embedding_aware_belt_fusion.alignformer_pva.abstain import AbstentionConfig
from embedding_aware_belt_fusion.alignformer_pva.refine import (
    AgreementConfig,
    DEFAULT_GATES_M as PVA_DEFAULT_GATES_M,
    DEFAULT_INLIER_M as PVA_DEFAULT_INLIER_M,
    DEFAULT_MIN_PAIRS as PVA_DEFAULT_MIN_PAIRS,
    ICP_RANSAC as PVA_ICP_RANSAC,
    REFINE_MODES as PVA_REFINE_MODES,
    RefineConfig,
)
from embedding_aware_belt_fusion.alignformer_pva.train import (
    load_alignformer_pva,
    load_embedding_encoder,
)
from embedding_aware_belt_fusion.alignformer_pva.inference import (
    correct_simple_detections,
    fuse_corrected_simple_detections,
)
from embedding_aware_belt_fusion.alignformer_pva.robust import RobustSolveConfig
from embedding_aware_belt_fusion.alignformer_pva.shrinkage import ShrinkageCalibration
from embedding_aware_belt_fusion.freealign.freealign import (
    DEFAULT_MIN_NODES as FREEALIGN_DEFAULT_MIN_NODES,
    EDGE_DISTANCE as FREEALIGN_EDGE_DISTANCE,
    EDGE_DISTANCE_YAW as FREEALIGN_EDGE_DISTANCE_YAW,
)
from embedding_aware_belt_fusion.freealign.freealign_calibration import (
    freealign_config_from_args,
)
from embedding_aware_belt_fusion.freealign.inference import (
    DEFAULT_ASSOCIATION_DISTANCE_M as FREEALIGN_DEFAULT_ASSOCIATION_DISTANCE_M,
    correct_simple_detections as freealign_correct_simple_detections,
    fuse_corrected_simple_detections as freealign_fuse_corrected_simple_detections,
)
from embedding_aware_belt_fusion.data.materialize_context_roi_cache import (
    padded_roi_cell_indices_and_flags,
)
from embedding_aware_belt_fusion.communication.product_codebook import (
    load_product_codebook,
    load_residual_product_codebook,
)
from embedding_aware_belt_fusion.integration.belt_fusion import (
    associate_by_embedding,
    associate_by_symmetric_pair_head,
    associate_ego_with_propagated_sources,
    associate_ego_with_propagated_sources_simple,
    fuse_detections,
    fuse_groups,
    fuse_groups_max_score,
    fuse_groups_score_weighted,
    proposal_uncertainty,
    simple_singleton_groups,
    singleton_groups,
)
from embedding_aware_belt_fusion.integration.opencood_uncertainty import (
    FrozenPointPillarWithUncertainty,
)
from embedding_aware_belt_fusion.integration.train_uncertainty import (
    _load_detector_checkpoint,
)
from embedding_aware_belt_fusion.evaluation.frame_cache import (
    EvaluationFrameCache,
    build_provenance as build_frame_cache_provenance,
    to_device as cached_to_device,
)
from embedding_aware_belt_fusion.evaluation.component_diagnostics import (
    association_counts,
    empty_counts as empty_component_counts,
    group_counts,
    merge_counts as merge_component_counts,
    pair_head_counts,
    summarize_counts as summarize_component_counts,
    summarize_group_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opencood-config", type=Path, required=True)
    parser.add_argument("--detector-dir", type=Path, required=True)
    parser.add_argument("--uncertainty-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--scenario-split-file", type=Path)
    parser.add_argument(
        "--scenario-role", choices=("train", "validation")
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--frame-cache-dir",
        type=Path,
        help=(
            "Reusable detector/proposal/ROI cache. It is independent of the "
            "PQ-STF checkpoint, pair head, RANSAC settings, fusion rule, and "
            "message codebook."
        ),
    )
    parser.add_argument(
        "--frame-cache-mode",
        choices=("off", "read-write", "read-only", "refresh"),
        default="off",
        help=(
            "read-write builds missing artifacts; read-only replays a complete "
            "compatible cache without loading LiDAR frames or running PointPillars; "
            "refresh rebuilds it."
        ),
    )
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--position-noise-std", type=float, default=0.0)
    parser.add_argument("--heading-noise-std-deg", type=float, default=0.0)
    parser.add_argument("--time-delay-ms", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=20)
    parser.add_argument(
        "--noise-mode",
        choices=("shared", "independent"),
        default="shared",
        help=(
            "shared: one (dx, dy, dyaw) draw per (position_std, heading_std, "
            "seed), reused identically for every agent and frame in the run "
            "(current default OpenCOOD behavior, formalized). independent: a "
            "fresh, independently-seeded draw per (scenario, frame, agent), "
            "deterministic given --noise-seed."
        ),
    )
    parser.add_argument(
        "--covariance-noise-variant",
        choices=("oracle", "fixed-spec"),
        default="fixed-spec",
        help=(
            "oracle: BELT's uncertainty covariance is handed the exact "
            "injected --position-noise-std/--heading-noise-std-deg at every "
            "sweep point (oracle knowledge of the ground-truth noise, kept "
            "for reference only). fixed-spec (default): covariance instead "
            "uses a single assumed spec set once via "
            "--assumed-position-noise-std/--assumed-heading-noise-std-deg, "
            "not refreshed per sweep severity -- fairer against AlignFormer, "
            "which gets no such oracle input."
        ),
    )
    parser.add_argument("--assumed-position-noise-std", type=float, default=0.2)
    parser.add_argument("--assumed-heading-noise-std-deg", type=float, default=2.0)
    parser.add_argument("--association-distance", type=float, default=5.0)
    parser.add_argument("--trackformer-checkpoint", type=Path)
    parser.add_argument(
        "--trackformer-protocol",
        choices=("propagated", "independent"),
        default="propagated",
        help="Association message protocol used by the TrackFormer checkpoint.",
    )
    parser.add_argument(
        "--message-codebook", type=Path,
        help="Shared product-codebook checkpoint used to quantize source messages.",
    )
    parser.add_argument(
        "--residual-message-codebook",
        type=Path,
        help="Progressive residual codebook for 256-D propagated query states.",
    )
    parser.add_argument(
        "--residual-stages",
        type=int,
        help="Fixed prefix length of --residual-message-codebook to transmit.",
    )
    parser.add_argument(
        "--adaptive-rate-thresholds",
        nargs=2,
        type=float,
        metavar=("LOW_TO_MEDIUM", "MEDIUM_TO_HIGH"),
        help="Use 1/2/3 residual stages based on source-local uncertainty rank.",
    )
    parser.add_argument(
        "--trackformer-root", type=Path, default=Path("external/trackformer")
    )
    parser.add_argument("--embedding-distance", type=float, default=8.0)
    parser.add_argument("--embedding-min-similarity", type=float, default=0.5)
    parser.add_argument(
        "--message-content-ablation",
        choices=("none", "zero", "permute"),
        default="none",
        help=(
            "Diagnostic for propagated PQ-STF: preserve the message shape "
            "but replace its query-state content with zeros or a deterministic "
            "within-CAV permutation."
        ),
    )
    parser.add_argument(
        "--ransac-localization-correction",
        action="store_true",
        help=(
            "Estimate a per-source planar translation/yaw correction from "
            "trusted PQ-STF ego--source matches before final association."
        ),
    )
    parser.add_argument(
        "--ransac-seed-distance",
        type=float,
        default=10.0,
        help="Permissive geometric gate used only to obtain RANSAC seed matches.",
    )
    parser.add_argument(
        "--ransac-pair-min-probability",
        type=float,
        default=None,
        help=(
            "Minimum symmetric pair-head probability used only for RANSAC "
            "seed correspondences. Defaults to --pair-head-min-probability, "
            "so final grouping and RANSAC seeds retain the historical shared gate."
        ),
    )
    parser.add_argument(
        "--ransac-inlier-threshold",
        type=float,
        default=1.0,
        help="Maximum center residual in metres for a RANSAC inlier.",
    )
    parser.add_argument(
        "--ransac-min-inliers",
        type=int,
        default=3,
        help="Minimum retained correspondences needed to apply a correction.",
    )
    parser.add_argument(
        "--ransac-min-relative-improvement",
        type=float,
        default=0.5,
        help=(
            "Require this fractional reduction in mean residual versus the "
            "identity transform before correcting a source CAV."
        ),
    )
    parser.add_argument(
        "--trackformer-box-fusion",
        choices=("belt", "score-weighted", "max-score"),
        default="belt",
        help=(
            "Use BELT uncertainty fusion (belt), plain detector-score weighted "
            "box averaging (score-weighted), or keep the highest-confidence "
            "box in each PQ-STF group (max-score)."
        ),
    )
    parser.add_argument(
        "--trackformer-grouping",
        choices=(
            "ego-centric",
            "all-agent-direct",
            "all-agent-symmetric-pair",
        ),
        default="ego-centric",
        help=(
            "For propagated PQ-STF plain fusion, use the original "
            "ego-to-source query-propagation grouping, direct all-agent cosine "
            "grouping, or learned symmetric all-agent query-state grouping."
        ),
    )
    parser.add_argument(
        "--all-agent-pair-head-checkpoint",
        type=Path,
        help=(
            "Learned symmetric PQ query-state pair scorer required by "
            "--trackformer-grouping all-agent-symmetric-pair."
        ),
    )
    parser.add_argument(
        "--pair-head-min-probability",
        type=float,
        default=0.9,
        help=(
            "Minimum calibrated same-object probability for a learned "
            "pair-head association; 0.9 is the fixed default."
        ),
    )
    parser.add_argument(
        "--track-query-score-threshold", type=float, default=0.5
    )
    parser.add_argument(
        "--oracle-grouping",
        action="store_true",
        help=(
            "Evaluation-only upper bound: group local detector proposals by "
            "their OPV2V physical object IDs before score-weighted box fusion. "
            "Physical IDs are never available to a deployable method."
        ),
    )
    parser.add_argument(
        "--oracle-ransac-localization-correction",
        action="store_true",
        help=(
            "Evaluation-only: use physical-ID matched detector boxes to "
            "estimate a per-source SE(2) RANSAC correction before oracle "
            "grouping and fusion. Requires --oracle-grouping."
        ),
    )
    parser.add_argument(
        "--alignformer-pva-checkpoint",
        type=Path,
        help=(
            "Opt-in, additive alternative to --ransac-localization-correction: "
            "a trained AlignFormer Propose-Verify-Abstain checkpoint "
            "(embedding_aware_belt_fusion.alignformer_pva). Geometry-only "
            "(no PQ-STF/TrackFormer embeddings): per source CAV, solves a "
            "Sinkhorn/Kabsch pose correction from simple_detections_by_cav "
            "box geometry alone, hardens it with a RANSAC Verify re-solve, "
            "and gates it with a Wald-test Abstain decision, before applying "
            "it with the same apply_se2_correction mechanism the RANSAC path "
            "uses. Produces its own 'alignformer_pva' row; it does not alter "
            "any existing flag's behavior when omitted."
        ),
    )
    parser.add_argument(
        "--alignformer-pva-refine-mode",
        choices=PVA_REFINE_MODES,
        default=PVA_ICP_RANSAC,
        help=(
            "Verify-stage hardening of the soft Sinkhorn/Kabsch solve "
            "(alignformer_pva.refine.icp_refine): 'none' keeps the soft "
            "estimate, 'icp' re-solves exactly over mutually-nearest hard "
            "pairs, 'icp_ransac' (default) additionally takes a RANSAC "
            "consensus over those hard pairs."
        ),
    )
    parser.add_argument(
        "--alignformer-pva-refine-min-pairs",
        type=int,
        default=PVA_DEFAULT_MIN_PAIRS,
        help="Minimum hard correspondences before the Verify re-solve engages.",
    )
    parser.add_argument(
        "--alignformer-pva-refine-inlier-m",
        type=float,
        default=PVA_DEFAULT_INLIER_M,
        help="RANSAC inlier gate (metres) for the Verify stage's hard-pair fit.",
    )
    parser.add_argument(
        "--alignformer-pva-refine-gates-m",
        type=str,
        default=",".join(str(gate) for gate in PVA_DEFAULT_GATES_M),
        help=(
            "Comma-separated Verify-stage gate schedule in metres, widest "
            f"first (default {','.join(str(gate) for gate in PVA_DEFAULT_GATES_M)}"
            ", matching alignformer_pva.refine.DEFAULT_GATES_M). Chenyi's own "
            "tuned schedule is '6,3,1.5,0.75,0.5' -- wider because, per his "
            "own paper, 'the proposal can be metres off on the pairs that "
            "matter most'."
        ),
    )
    parser.add_argument(
        "--alignformer-pva-abstain-level",
        type=float,
        default=0.05,
        help=(
            "False-fire level of the Abstain stage's Wald test "
            "(alignformer_pva.abstain.decide, mode='abstain'): a source "
            "CAV's correction is zeroed unless its own Wald statistic clears "
            "this level's threshold. The geometry-only analogue of "
            "--ransac-min-inliers/--ransac-min-relative-improvement. 0 "
            "disables the gate (always apply the Verify-stage correction)."
        ),
    )
    parser.add_argument(
        "--alignformer-pva-agreement-tolerance-m",
        type=float,
        default=None,
        help=(
            "Opt-in second Abstain-stage rule (alignformer_pva.refine.agree), "
            "Chenyi's 'agreement with a consensus floor': when given, the "
            "Verify-stage exact fit replaces the Wald-gated proposal only "
            "where the two agree within this many metres of each other (or "
            "the exact fit's own hard-pair consensus clears "
            "--alignformer-pva-consensus-floor); otherwise the Wald-gated "
            "proposal is used. Omitted (default, every existing invocation) "
            "skips this rule entirely and reproduces today's simpler "
            "behavior bit for bit: the Wald test alone gates the Verify-stage "
            "correction."
        ),
    )
    parser.add_argument(
        "--alignformer-pva-consensus-floor",
        type=int,
        default=0,
        help=(
            "Hard-pair consensus (alignformer_pva.refine.AgreementConfig."
            "consensus_floor) above which the agreement rule accepts the "
            "exact fit even when it disagrees with the proposal by more than "
            "--alignformer-pva-agreement-tolerance-m. 0 disables the floor "
            "(default, matching AgreementConfig's own default). Only takes "
            "effect when --alignformer-pva-agreement-tolerance-m is given."
        ),
    )
    parser.add_argument(
        "--alignformer-pva-association-distance",
        type=float,
        default=5.0,
        help=(
            "Geometric gate for associating corrected simple detections "
            "across agents before score-weighted fusion of the "
            "'alignformer_pva' row."
        ),
    )
    parser.add_argument(
        "--alignformer-pva-context-margin-m",
        type=float,
        default=1.75,
        help=(
            "Context-padding margin (metres) for live per-object embedding "
            "cells, ONLY used when --alignformer-pva-checkpoint was trained "
            "with --embed-dim > 0 (data/materialize_context_roi_cache.py's "
            "training-time cache uses the same convention). Ignored for a "
            "geometry-only (embed_dim=0) checkpoint. Should match the margin "
            "the checkpoint's embedding encoder was actually trained with, "
            "or the live embedding distribution will not match training."
        ),
    )
    parser.add_argument(
        "--alignformer-pva-robust-mode",
        choices=("none", "huber"),
        default="none",
        help=(
            "Optional IRLS re-weighting of the Propose stage's weighted "
            "Kabsch solve (alignformer_pva.robust.RobustSolveConfig), run "
            "before Verify/Abstain. 'none' (default) reproduces the "
            "original single weighted-Kabsch solve exactly. 'huber' "
            "matches Chenyi's own robust_solve_config (iterations=2, "
            "huber_constant=1.345, scale_floor_m=0.05, min_evidence=3.0; "
            "override iterations with --alignformer-pva-robust-iterations)."
        ),
    )
    parser.add_argument(
        "--alignformer-pva-robust-iterations",
        type=int,
        default=2,
        help="IRLS re-weight/re-solve rounds when --alignformer-pva-robust-mode huber.",
    )
    parser.add_argument(
        "--alignformer-pva-shrinkage-tau-translation-m",
        type=float,
        default=None,
        help=(
            "Opt-in James-Stein-style shrinkage of the Propose-stage "
            "estimate toward the identity (alignformer_pva.shrinkage), "
            "applied BEFORE Verify/Abstain. Requires both this and "
            "--alignformer-pva-shrinkage-tau-yaw-deg. Omitted (default) "
            "skips shrinkage entirely, reproducing existing behavior; "
            "mutually exclusive in spirit with Abstain's per-pair decision "
            "(Chenyi's own protocol uses one or the other, not both, as "
            "alternative treatments of the same base estimate)."
        ),
    )
    parser.add_argument(
        "--alignformer-pva-shrinkage-tau-yaw-deg",
        type=float,
        default=None,
        help="Paired with --alignformer-pva-shrinkage-tau-translation-m; see its help.",
    )
    parser.add_argument(
        "--freealign",
        action="store_true",
        help=(
            "Opt-in, additive competitor row: Lei et al.'s FreeAlign (ICRA "
            "2024), REIMPLEMENTED on our own detections "
            "(embedding_aware_belt_fusion.freealign). Training-free: per "
            "source CAV, builds each side's salient-object distance graph "
            "from simple_detections_by_cav box geometry alone, finds the "
            "common subgraph by MASS, and fits a robust (RANSAC/LMedS) "
            "SE(2) correction over it, applied with the same "
            "apply_se2_correction mechanism every other alignment method "
            "uses. Produces its own 'freealign' row; it does not alter any "
            "existing flag's behavior when omitted."
        ),
    )
    parser.add_argument(
        "--freealign-calibration",
        type=Path,
        default=None,
        help=(
            "Path to a freealign/calibrate_freealign.py result JSON; its "
            "'selected' block sets EVERY FreeAlignConfig field (the edge "
            "threshold, anchor limit, epsilon power/offset and robust "
            "estimator this project calibrated on our own scenario-disjoint "
            "validation role -- never the labmate's own fitted numbers). "
            "Requires --freealign. Mutually exclusive with "
            "--freealign-edge-feature/--freealign-min-nodes (a calibration "
            "file already sets those two)."
        ),
    )
    parser.add_argument(
        "--freealign-edge-feature",
        choices=(FREEALIGN_EDGE_DISTANCE, FREEALIGN_EDGE_DISTANCE_YAW),
        default=FREEALIGN_EDGE_DISTANCE,
        help=(
            "FreeAlign's salient-object edge feature absent --freealign-"
            "calibration: 'distance' (default, the paper's training-free "
            "path and their shipped gnn:false config) or 'distance_yaw' "
            "(their shipped greedy_match.py's 2-channel edge)."
        ),
    )
    parser.add_argument(
        "--freealign-min-nodes",
        type=int,
        default=FREEALIGN_DEFAULT_MIN_NODES,
        help=(
            "FreeAlign's abstain threshold absent --freealign-calibration: "
            "below this many matched objects the collaborative message is "
            "discarded (fused uncorrected), mirroring alignformer_pva's own "
            "MIN_MATCH_MASS fallback."
        ),
    )
    parser.add_argument(
        "--freealign-association-distance",
        type=float,
        default=FREEALIGN_DEFAULT_ASSOCIATION_DISTANCE_M,
        help=(
            "Geometric gate for associating corrected simple detections "
            "across agents before score-weighted fusion of the "
            "'freealign' row."
        ),
    )
    parser.add_argument("--global-sort-detections", action="store_true")
    parser.add_argument(
        "--skip-belt",
        action="store_true",
        help=(
            "Skip BELT's uncertainty-weighted association/fusion entirely "
            "(no belt_geometry/belt_association in the output). Additive, "
            "off by default so existing behavior is unchanged; use this when "
            "BELT's numbers aren't needed, since its association step is a "
            "real per-frame cost."
        ),
    )
    parser.add_argument(
        "--fast-ap",
        action="store_true",
        help=(
            "Route NMS, range filtering and TP/FP matching through the "
            "compiled embedding_aware_belt_fusion.fusion_core extension "
            "(vendored from GenCPM/OpenCPM, exact double-precision polygon "
            "IoU, same math as box_utils.nms_rotated + eval_utils."
            "caluclate_tp_fp) instead of OpenCOOD's shapely path, which "
            "recomputes per-pair polygon IoU once per IoU threshold. "
            "Numerically equivalent; only the naive, ego_only, oracle, "
            "freealign and alignformer_pva rows use it today. Requires "
            "building fusion_core once (see _load_fusion_core's error "
            "message for the command)."
        ),
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def _statistics():
    return {
        threshold: {"tp": [], "fp": [], "gt": 0, "score": []}
        for threshold in (0.3, 0.5, 0.7)
    }


def _ablate_message_content(message_state, mode: str):
    """Destroy query-state identity while retaining message count and shape."""
    if mode == "none" or len(message_state) < 2:
        return message_state
    if mode == "zero":
        return torch.zeros_like(message_state)
    if mode == "permute":
        return message_state.roll(shifts=1, dims=0)
    raise ValueError(f"Unknown message-content ablation: {mode}")


def _query_geometry_from_ego_boxes(boxes, scores):
    """Rebuild the 9-D PQ geometry after an ego-frame SE(2) correction."""
    scales = boxes.new_tensor([70.4, 40.0, 4.0, 4.0, 4.0, 10.0])
    return torch.cat(
        [
            boxes[:, :6] / scales,
            torch.sin(boxes[:, 6:7]),
            torch.cos(boxes[:, 6:7]),
            scores.float().reshape(-1, 1).clamp(0.0, 1.0),
        ],
        dim=-1,
    )


def _apply_symmetric_ransac_correction(detection, correction):
    """Correct boxes and the explicit geometry given to the PQ pair head.

    The transmitted 256-D state is intentionally not altered: it is the
    source CAV's already-transmitted appearance/query message.  Only its
    separately transmitted ego-frame box geometry is updated after ego has
    estimated the source pose error.
    """
    corrected = apply_se2_correction(detection, correction)
    if "query_geometry" in corrected:
        corrected["query_geometry"] = _query_geometry_from_ego_boxes(
            corrected["boxes"], corrected["scores"]
        )
    return corrected


def _oracle_groups(detections):
    """Group only detector proposals assigned to the same physical GT ID.

    This is an evaluation oracle, not an association method.  A proposal that
    cannot be assigned one-to-one to a local ground-truth object remains a
    singleton, preserving detector false positives in the upper-bound result.
    """
    groups_by_object_id = OrderedDict()
    singleton_groups_out = []
    for agent_index, detection in enumerate(detections):
        for proposal_index, object_id in enumerate(detection["oracle_object_ids"]):
            member = {
                "agent_index": agent_index,
                "box": detection["boxes"][proposal_index],
                "score": detection["scores"][proposal_index],
            }
            if object_id is None:
                singleton_groups_out.append([member])
            else:
                groups_by_object_id.setdefault(str(object_id), []).append(member)
    return list(groups_by_object_id.values()) + singleton_groups_out


def _oracle_object_ids(proposal, cav_content, postprocessor):
    """Assign each local prediction to a local labelled physical object."""
    gt_mask = cav_content["object_bbx_mask"][0] > 0
    assignment = assign_proposals_to_ground_truth(
        proposal["corners"],
        cav_content["object_bbx_center"][0][gt_mask],
        cav_content["object_ids"],
        order=postprocessor.params["order"],
        minimum_iou=0.5,
    )
    return assignment["gt_ids"]


def _ego_ground_truth_reference(cav_content):
    """Return the AP-scope ego ground-truth boxes and physical IDs per frame."""
    mask = cav_content["object_bbx_mask"][0] > 0
    object_ids = list(cav_content["object_ids"])
    valid = mask.nonzero().flatten().tolist()
    return {
        "boxes": cav_content["object_bbx_center"][0][mask],
        "object_ids": [str(object_ids[index]) for index in valid],
    }


def _coverage_categories(detections_by_cav, ego_ground_truth, final_corners, postprocessor):
    """Diagnose which GT objects can be contributed by source CAVs.

    Counts are per frame-instance, not unique tracks across an entire scenario.
    A detector proposal is considered to see a GT object when its local
    proposal-to-GT assignment has IoU >= 0.5.
    """
    ground_truth_ids = set(ego_ground_truth["object_ids"])
    ego_detected = {
        object_id
        for object_id in detections_by_cav["ego"]["oracle_object_ids"]
        if object_id is not None and object_id in ground_truth_ids
    }
    source_detected = set()
    for cav_id, detection in detections_by_cav.items():
        if cav_id == "ego":
            continue
        source_detected.update(
            object_id
            for object_id in detection["oracle_object_ids"]
            if object_id is not None and object_id in ground_truth_ids
        )
    categories = {
        "ego_detected": ego_detected,
        "source_only_detectable": source_detected - ego_detected,
        "undetected_by_everyone": ground_truth_ids - ego_detected - source_detected,
    }
    recovered = set()
    if final_corners is not None and len(final_corners):
        assignment = assign_proposals_to_ground_truth(
            final_corners,
            ego_ground_truth["boxes"],
            ego_ground_truth["object_ids"],
            order=postprocessor.params["order"],
            minimum_iou=0.5,
        )
        recovered = {
            object_id for object_id in assignment["gt_ids"] if object_id is not None
        }
    return {
        name: {"total": len(ids), "recovered": len(ids & recovered)}
        for name, ids in categories.items()
    }


def _merge_coverage(total, update):
    for category, values in update.items():
        for key, value in values.items():
            total[category][key] += value


def _summarize_coverage(counts):
    return {
        category: {
            **values,
            "recall": values["recovered"] / max(values["total"], 1),
        }
        for category, values in counts.items()
    }


def _oracle_prediction_correspondences(detections_by_cav):
    """Use physical IDs only to expose trusted predicted-box correspondences."""
    ego = detections_by_cav["ego"]
    ego_by_id = {
        object_id: index
        for index, object_id in enumerate(ego["oracle_object_ids"])
        if object_id is not None
    }
    result = {}
    for cav_id, detection in detections_by_cav.items():
        if cav_id == "ego":
            continue
        source_indices, ego_indices = [], []
        for source_index, object_id in enumerate(
            detection["oracle_object_ids"]
        ):
            ego_index = ego_by_id.get(object_id)
            if ego_index is not None:
                source_indices.append(source_index)
                ego_indices.append(ego_index)
        if len(source_indices) >= 2:
            result[cav_id] = (
                detection["boxes"][source_indices, :2],
                ego["boxes"][ego_indices, :2],
            )
    return result


def _evaluate_prediction(prediction, score, ground_truth, statistics):
    from opencood.utils import eval_utils

    for threshold in statistics:
        eval_utils.caluclate_tp_fp(
            prediction, score, ground_truth, statistics, threshold
        )


def _ap(statistics, global_sort):
    from opencood.utils import eval_utils

    return {
        f"ap_{int(threshold * 100):02d}": eval_utils.calculate_ap(
            copy.deepcopy(statistics), threshold, global_sort
        )[0]
        for threshold in statistics
    }


def _postprocess_fused_boxes(boxes, scores, postprocessor):
    """Apply OpenCOOD's final range filters and rotated global NMS."""
    from opencood.utils import box_utils

    if len(boxes) == 0:
        return None, None
    corners = box_utils.boxes_to_corners_3d(
        boxes, order=postprocessor.params["order"]
    )
    valid = box_utils.remove_large_pred_bbx(corners)
    valid &= box_utils.remove_bbx_abnormal_z(corners)
    corners, scores = corners[valid], scores[valid]
    if len(corners) == 0:
        return None, None
    keep = box_utils.nms_rotated(
        corners, scores, postprocessor.params["nms_thresh"]
    )
    corners, scores = corners[keep], scores[keep]
    in_range = box_utils.get_mask_for_boxes_within_range_torch(corners)
    return corners[in_range], scores[in_range]


_FUSION_CORE = None


def _load_fusion_core():
    """Import the compiled embedding_aware_belt_fusion.fusion_core extension.

    Vendored from GenCPM/OpenCPM's tools/fusion_core (same math as
    opencood.utils.box_utils.nms_rotated + opencood.utils.eval_utils.
    caluclate_tp_fp, exact double-precision polygon-clip IoU), built once
    per Python ABI into fusion_core/build next to this package.
    """
    global _FUSION_CORE
    if _FUSION_CORE is None:
        import sys

        build_dir = Path(__file__).resolve().parents[1] / "fusion_core" / "build"
        if str(build_dir) not in sys.path:
            sys.path.insert(0, str(build_dir))
        try:
            import fusion_core
        except ImportError as exc:
            raise ImportError(
                "fusion_core extension not found or built for a different "
                "Python ABI. Build it once with:\n"
                "  cmake -S src/embedding_aware_belt_fusion/fusion_core "
                "-B src/embedding_aware_belt_fusion/fusion_core/build "
                f"-DPython_EXECUTABLE={sys.executable}\n"
                "  cmake --build src/embedding_aware_belt_fusion/fusion_core/build -j"
            ) from exc
        _FUSION_CORE = fusion_core
    return _FUSION_CORE


def _evaluate_prediction_fast(boxes, scores, ground_truth, postprocessor, statistics):
    """fusion_core-backed replacement for _postprocess_fused_boxes +
    _evaluate_prediction: one compiled NMS + range-filter + greedy TP/FP
    pass shared across every IoU threshold, instead of OpenCOOD's shapely
    path recomputing polygon IoUs once per threshold."""
    import numpy as np
    from opencood.utils import box_utils, common_utils

    fusion_core = _load_fusion_core()
    thresholds = sorted(statistics)
    thresholds_arr = np.asarray(thresholds, dtype=np.float64)
    gt = np.ascontiguousarray(
        common_utils.torch_tensor_to_numpy(ground_truth), dtype=np.float32
    )
    if boxes is None or len(boxes) == 0:
        sc = np.zeros(0, dtype=np.float32)
        tp = np.zeros((len(thresholds), 0), dtype=np.int8)
    else:
        if len(boxes.shape) == 2:  # (N, 7) center format -> corners
            corners = box_utils.boxes_to_corners_3d(
                boxes, order=postprocessor.params["order"]
            )
        else:  # already (N, 8, 3) corners (e.g. OpenCOOD's own post_process)
            corners = boxes
        corners = np.ascontiguousarray(
            common_utils.torch_tensor_to_numpy(corners), dtype=np.float32
        )
        sc_in = np.ascontiguousarray(
            common_utils.torch_tensor_to_numpy(scores), dtype=np.float32
        )
        sc, tp = fusion_core.fuse_and_score(
            corners,
            sc_in,
            gt,
            float(postprocessor.params["nms_thresh"]),
            thresholds_arr,
        )
    for index, threshold in enumerate(thresholds):
        bucket = statistics[threshold]
        tp_i = tp[index].astype(int)
        bucket["tp"] += tp_i.tolist()
        bucket["fp"] += (1 - tp_i).tolist()
        bucket["score"] += sc.tolist()
        bucket["gt"] += int(gt.shape[0])


def _postprocess_and_evaluate(
    boxes, scores, ground_truth, postprocessor, statistics, fast=False
):
    """_postprocess_fused_boxes + _evaluate_prediction, or the fusion_core
    fast path when fast=True. Behaviorally identical to the two-call form
    when fast=False."""
    if not fast:
        boxes, scores = _postprocess_fused_boxes(boxes, scores, postprocessor)
        _evaluate_prediction(boxes, scores, ground_truth, statistics)
        return
    _evaluate_prediction_fast(boxes, scores, ground_truth, postprocessor, statistics)


def _set_noise(hypes, args):
    hypes["validate_dir"] = str((args.data_root / args.split).resolve())
    hypes.setdefault("wild_setting", {})
    hypes["wild_setting"].update(
        {
            "seed": args.noise_seed,
            "async": args.time_delay_ms > 0,
            "async_mode": "sim",
            "async_overhead": args.time_delay_ms,
            "loc_err": (
                args.position_noise_std > 0
                or args.heading_noise_std_deg > 0
            ),
            "xyz_std": args.position_noise_std,
            "ryp_std": args.heading_noise_std_deg,
            "noise_mode": args.noise_mode,
        }
    )


def _scenario_id(dataset, index):
    scenario_index = next(
        position
        for position, end in enumerate(dataset.len_record)
        if index < end
    )
    scenario = dataset.scenario_database[scenario_index]
    timestamps = next(iter(scenario.values()))
    return Path(next(iter(timestamps.values()))["yaml"]).parents[1].name


def _select_detection(detection, indices):
    count = len(detection["boxes"])
    return {
        key: (
            value[indices]
            if isinstance(value, torch.Tensor) and len(value) == count
            else value
        )
        for key, value in detection.items()
    }


def _simple_detection(proposal, cav_content):
    """Transform only PointPillars boxes/scores to ego coordinates.

    No uncertainty, covariance, or evidential value is included.  This is the
    message used by the score-weighted TrackFormer ablation.
    """
    return {
        "boxes": transform_boxes_to_ego(
            proposal["boxes"], cav_content["transformation_matrix"]
        ),
        "scores": proposal["scores"],
    }


def _select_trackformer_agent(agent, indices):
    return {
        "boxes": agent["boxes"][indices],
        "scores": agent["scores"][indices],
        "tokens": agent["tokens"][indices],
        "positions": agent["positions"][indices],
        "mask": agent["mask"][indices],
        "transformation_matrix": agent["transformation_matrix"],
    }


def _trackformer_agent(proposal, cav_content, output, hypes):
    """Materialize every PointPillars ROI cell for live TrackFormer inference."""
    features = output["spatial_features_2d"][0]
    roi_indices = proposal_roi_cell_indices(
        proposal["boxes"],
        height=features.shape[-2],
        width=features.shape[-1],
        lidar_range=hypes["model"]["args"]["lidar_range"],
    )
    count = len(roi_indices)
    maximum = max((len(indices) for indices in roi_indices), default=0)
    flattened = features.flatten(1).transpose(0, 1)
    tokens = features.new_zeros((count, maximum, flattened.shape[-1]))
    positions = features.new_zeros((count, maximum, 2))
    mask = torch.ones(
        (count, maximum), dtype=torch.bool, device=features.device
    )
    for proposal_index, indices in enumerate(roi_indices):
        indices = indices.to(features.device, dtype=torch.long)
        length = len(indices)
        tokens[proposal_index, :length] = flattened[indices]
        rows = torch.div(
            indices, features.shape[-1], rounding_mode="floor"
        ).float()
        columns = (indices % features.shape[-1]).float()
        coordinates = torch.stack([rows, columns], dim=-1)
        coordinates -= coordinates.mean(dim=0, keepdim=True)
        coordinates /= coordinates.abs().amax(
            dim=0, keepdim=True
        ).clamp_min(1.0)
        positions[proposal_index, :length] = coordinates
        mask[proposal_index, :length] = False
    return {
        "boxes": proposal["boxes"],
        "scores": proposal["scores"],
        "tokens": tokens,
        "positions": positions,
        "mask": mask,
        "transformation_matrix": cav_content["transformation_matrix"],
    }


def _context_embedding_cells(proposal, output, hypes, *, margin):
    """Context-padded PointPillars ROI cells for live AlignFormer-PVA embedding inference.

    Phase 1 of the richer-embedding plan. Mirrors ``_trackformer_agent``'s
    structure exactly (same live ``spatial_features_2d`` source, same
    relative cell-position convention) but pads each proposal's cell set by
    ``margin`` metres beyond its exact footprint and tags every cell
    ``cell_footprint`` (own footprint vs. padding context), using
    ``data/materialize_context_roi_cache.py``'s own
    ``padded_roi_cell_indices_and_flags`` so the live and offline-cache paths
    agree bit-for-bit (both call the identical function).

    Returns ONLY the new ``cell_tokens``/``cell_positions``/``cell_footprint``/
    ``cell_mask`` keys -- never ``boxes``/``scores``. The caller merges this
    into an EXISTING ``simple_detections_by_cav[cav_id]`` entry (built by
    ``_simple_detection``), reusing its ego-frame ``boxes``/``scores`` as-is:
    that is exactly what ``AlignFormerPVAContextPairDataset``/``train.py`` fed
    the embedding encoder during training (ego-frame boxes, not the local-frame
    ones ``_trackformer_agent`` itself keeps for its own, unrelated purpose).
    """
    features = output["spatial_features_2d"][0]
    height, width = features.shape[-2], features.shape[-1]
    lidar_range = hypes["model"]["args"]["lidar_range"]
    cells = padded_roi_cell_indices_and_flags(
        proposal["boxes"], height=height, width=width, lidar_range=lidar_range, margin=margin
    )
    count = len(cells)
    maximum = max((len(indices) for indices, _ in cells), default=0)
    flattened = features.flatten(1).transpose(0, 1)
    tokens = features.new_zeros((count, maximum, flattened.shape[-1]))
    positions = features.new_zeros((count, maximum, 2))
    footprint = torch.zeros((count, maximum), dtype=torch.bool, device=features.device)
    mask = torch.ones((count, maximum), dtype=torch.bool, device=features.device)
    for proposal_index, (indices, flags) in enumerate(cells):
        indices = indices.to(features.device, dtype=torch.long)
        length = len(indices)
        tokens[proposal_index, :length] = flattened[indices]
        rows = torch.div(indices, width, rounding_mode="floor").float()
        columns = (indices % width).float()
        coordinates = torch.stack([rows, columns], dim=-1)
        coordinates -= coordinates.mean(dim=0, keepdim=True)
        coordinates /= coordinates.abs().amax(dim=0, keepdim=True).clamp_min(1.0)
        positions[proposal_index, :length] = coordinates
        footprint[proposal_index, :length] = flags.to(device=features.device)
        mask[proposal_index, :length] = False
    return {
        "cell_tokens": tokens,
        "cell_positions": positions,
        "cell_footprint": footprint,
        "cell_mask": mask,
    }


def _has_valid_trackformer_roi(agent):
    """Whether an agent has at least one BEV cell usable by TrackFormer.

    A detector may produce boxes whose rotated ROI lies completely outside the
    native BEV map.  Such a frame remains a valid detection/fusion example,
    but TrackFormer has no feature token from which to form an embedding.
    """
    if not len(agent["boxes"]):
        return True
    mask = agent["mask"]
    return bool(mask.numel() and (~mask.bool()).any())


def _load_trackformer(args, feature_dim, device):
    if args.trackformer_checkpoint is None:
        return None, None
    saved = torch.load(
        args.trackformer_checkpoint, map_location="cpu", weights_only=False
    )
    if saved.get("pipeline") != "spatial_trackformer_v2_geometry_association":
        raise ValueError(
            "TrackFormer checkpoint must be the geometry-association v2 model"
        )
    training = saved["arguments"]
    checkpoint_protocol = saved.get("association_protocol", "propagated")
    if checkpoint_protocol != args.trackformer_protocol:
        raise ValueError(
            f"Checkpoint was trained for {checkpoint_protocol!r}; "
            f"requested {args.trackformer_protocol!r} evaluation"
        )
    model = SpatialTrackFormer(
        trackformer_root=args.trackformer_root,
        input_dim=feature_dim,
        d_model=int(training["d_model"]),
        embedding_dim=int(training["embedding_dim"]),
        heads=int(training["heads"]),
        encoder_layers=int(training["encoder_layers"]),
        decoder_layers=int(training["decoder_layers"]),
        feedforward_dim=int(training["feedforward_dim"]),
        dropout=float(training["dropout"]),
        geometry_association=True,
    ).to(device)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval()
    return model, int(saved["epoch"])


def _load_all_agent_pair_head(args, trackformer, device):
    """Load the learned source-state scorer and verify its PQ-STF parent."""
    if args.trackformer_grouping != "all-agent-symmetric-pair":
        return None, None
    saved = torch.load(
        args.all_agent_pair_head_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    expected_base = str(args.trackformer_checkpoint.resolve())
    if saved.get("base_checkpoint") != expected_base:
        raise ValueError(
            "The all-agent pair head was trained from a different "
            "--trackformer-checkpoint"
        )
    if saved.get("pipeline") != "calibrated_pq_query_state_pair_head_v2":
        raise ValueError(
            "The supplied pair head is the retired retrieval-only version. "
            "Train and use calibrated_pq_query_state_pair_head_v2."
        )
    state_dim = int(saved["state_dim"])
    if state_dim != int(trackformer.embedding_head.in_features):
        raise ValueError("Pair-head query-state dimension is incompatible")
    head = SymmetricQueryStatePairHead(state_dim).to(device)
    head.load_state_dict(saved["model_state_dict"], strict=True)
    head.eval()
    return head, int(saved["epoch"])


def _load_message_codebook(args, trackformer, device):
    if args.message_codebook is not None and args.residual_message_codebook is not None:
        raise ValueError("Use only one of --message-codebook and --residual-message-codebook")
    codebook_path = args.message_codebook or args.residual_message_codebook
    if codebook_path is None:
        if args.residual_stages is not None or args.adaptive_rate_thresholds is not None:
            raise ValueError("Adaptive/residual-rate options require --residual-message-codebook")
        return None, None
    if trackformer is None:
        raise ValueError("A message codebook requires --trackformer-checkpoint")
    if args.residual_message_codebook is not None:
        codebook, metadata = load_residual_product_codebook(codebook_path, device)
        if args.residual_stages is not None and not 1 <= args.residual_stages <= len(codebook.stages):
            raise ValueError("--residual-stages must select a valid residual-codebook prefix")
        if args.adaptive_rate_thresholds is not None:
            low, high = args.adaptive_rate_thresholds
            if not 0.0 <= low < high <= 1.0:
                raise ValueError("Adaptive thresholds must satisfy 0 <= low < high <= 1")
    else:
        if args.residual_stages is not None or args.adaptive_rate_thresholds is not None:
            raise ValueError("Residual-rate options require --residual-message-codebook")
        codebook, metadata = load_product_codebook(codebook_path, device)
    expected = (
        "query-state"
        if args.trackformer_protocol == "propagated"
        else "embedding"
    )
    if metadata["representation"] != expected:
        raise ValueError(
            f"{codebook_path} encodes {metadata['representation']!r}; "
            f"protocol {args.trackformer_protocol!r} requires {expected!r}"
        )
    expected_dimension = (
        trackformer.embedding_head.in_features
        if expected == "query-state"
        else trackformer.embedding_head.out_features
    )
    if codebook.dimension != expected_dimension:
        raise ValueError("Codebook dimension does not match TrackFormer message dimension")
    source_checkpoint = metadata.get("source_checkpoint")
    if source_checkpoint is not None and Path(source_checkpoint).resolve() != args.trackformer_checkpoint.resolve():
        raise ValueError(
            "The codebook was fit from a different --trackformer-checkpoint; "
            "fit a codebook from the model being evaluated."
        )
    return codebook, metadata


def parse_alignformer_pva_gates_m(spec: str) -> Tuple[float, ...]:
    """Parse ``--alignformer-pva-refine-gates-m``'s comma-separated schedule.

    ``"6,3,1.5,0.75,0.5"`` -> ``(6.0, 3.0, 1.5, 0.75, 0.5)``. Raises
    ``ValueError`` with a message naming the offending spec on anything that
    is not a non-empty, comma-separated list of finite numbers -- the same
    construction ``RefineConfig.__post_init__`` itself enforces on the
    resulting tuple (at least one gate, every gate finite and positive), so a
    bad CLI value is caught here with a clear message rather than surfacing
    later as an opaque ``RefineConfig`` error.
    """
    pieces = [piece.strip() for piece in str(spec).split(",")]
    if not pieces or any(piece == "" for piece in pieces):
        raise ValueError(
            f"{spec!r} is not a comma-separated gate schedule, e.g. '6,3,1.5,0.75,0.5'"
        )
    try:
        return tuple(float(piece) for piece in pieces)
    except ValueError as error:
        raise ValueError(
            f"{spec!r} is not a comma-separated gate schedule, e.g. '6,3,1.5,0.75,0.5'"
        ) from error


def _load_alignformer_pva(args, device):
    """Load the opt-in Propose-Verify-Abstain backend, if requested.

    Returns ``(model, checkpoint, refine_config, abstain_config,
    agreement_config, embedding_encoder)``, all ``None`` when
    ``--alignformer-pva-checkpoint`` was not given -- the only way this ever
    changes what ``main`` does for an existing invocation.

    ``refine_config.gates_m`` comes from ``--alignformer-pva-refine-gates-m``
    (:func:`parse_alignformer_pva_gates_m`), whose default string reproduces
    ``RefineConfig``'s own ``DEFAULT_GATES_M`` exactly, so an invocation that
    omits the flag is unaffected.

    ``agreement_config`` is ``None`` unless
    ``--alignformer-pva-agreement-tolerance-m`` was given -- omitting it
    skips the agreement rule entirely (see ``inference.estimate_correction``)
    and reproduces the original Wald-only decision bit for bit.

    ``embedding_encoder`` (Phase 1 of the richer-embedding plan) is loaded
    additively via ``load_embedding_encoder``: it is ``None`` for every
    checkpoint trained with ``--embed-dim 0`` (every checkpoint that existed
    before embeddings were added, and any new one that keeps the default),
    so passing a geometry-only checkpoint here behaves exactly as before.
    """
    if args.alignformer_pva_checkpoint is None:
        return None, None, None, None, None, None, None, None
    model, checkpoint = load_alignformer_pva(args.alignformer_pva_checkpoint, device)
    embedding_encoder = load_embedding_encoder(args.alignformer_pva_checkpoint, device)
    refine_config = RefineConfig(
        mode=args.alignformer_pva_refine_mode,
        gates_m=parse_alignformer_pva_gates_m(args.alignformer_pva_refine_gates_m),
        min_pairs=args.alignformer_pva_refine_min_pairs,
        inlier_m=args.alignformer_pva_refine_inlier_m,
    )
    abstain_config = (
        AbstentionConfig(mode="abstain", level=args.alignformer_pva_abstain_level)
        if args.alignformer_pva_abstain_level > 0.0
        else AbstentionConfig(mode="none")
    )
    agreement_config = (
        AgreementConfig(
            tolerance_m=args.alignformer_pva_agreement_tolerance_m,
            consensus_floor=args.alignformer_pva_consensus_floor,
        )
        if args.alignformer_pva_agreement_tolerance_m is not None
        else None
    )
    robust_config = (
        RobustSolveConfig(
            mode="huber", iterations=args.alignformer_pva_robust_iterations
        )
        if args.alignformer_pva_robust_mode == "huber"
        else None
    )
    shrinkage_config = (
        ShrinkageCalibration(
            tau_translation_m=args.alignformer_pva_shrinkage_tau_translation_m,
            tau_yaw_rad=math.radians(args.alignformer_pva_shrinkage_tau_yaw_deg),
            pairs=0,
            split="--alignformer-pva-shrinkage-tau-* (CLI-provided)",
            sigma_m=0.0,
        )
        if args.alignformer_pva_shrinkage_tau_translation_m is not None
        and args.alignformer_pva_shrinkage_tau_yaw_deg is not None
        else None
    )
    return (
        model,
        checkpoint,
        refine_config,
        abstain_config,
        agreement_config,
        embedding_encoder,
        robust_config,
        shrinkage_config,
    )


def _load_freealign(args):
    """Load the opt-in FreeAlign competitor row's configuration, if requested.

    Returns ``None`` unless ``--freealign`` was given -- the only way this
    ever changes what ``main`` does for an existing invocation.
    ``freealign_config_from_args`` (freealign/freealign_calibration.py)
    itself raises on an inconsistent combination of flags (a calibration
    file given without ``--freealign``, or alongside a moved
    ``--freealign-edge-feature``/``--freealign-min-nodes``), so those checks
    are not duplicated here.
    """
    return freealign_config_from_args(args)


def _rank_unit_interval(values):
    """Return deterministic within-source ranks in [0, 1]."""
    if len(values) <= 1:
        return torch.full_like(values, 0.5)
    order = values.argsort()
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(
        len(values), device=values.device, dtype=torch.float32
    )
    return ranks / float(len(values) - 1)


def _adaptive_stage_counts(detection, thresholds):
    """Choose more residual stages for locally more uncertain source objects.

    Classification uncertainty, ego-frame position standard deviation, and
    heading standard deviation have incompatible physical units.  We rank each
    quantity within the current source CAV's object messages, then average the
    three ranks.  This supplies a source-local, unit-free rate decision.
    """
    count = len(detection["boxes"])
    if not count:
        return torch.empty(0, dtype=torch.long, device=detection["boxes"].device)
    class_rank = _rank_unit_interval(detection["class_uncertainty"])
    diagonal = detection["covariances"].diagonal(dim1=-2, dim2=-1)
    position_std = diagonal[:, :2].mean(dim=-1).clamp_min(0).sqrt()
    heading_std = diagonal[:, 6].clamp_min(0).sqrt()
    uncertainty_rank = (
        class_rank
        + _rank_unit_interval(position_std)
        + _rank_unit_interval(heading_std)
    ) / 3.0
    low, high = thresholds
    return (
        1
        + (uncertainty_rank >= low).long()
        + (uncertainty_rank >= high).long()
    )


def _message_overhead_bits(detection, query_geometry=None):
    """Bits needed to deliver a source's raw box geometry and detector score
    alongside its query-state message, plus TrackFormer query geometry when
    the receiving side needs it for association/RANSAC. ``_transmit`` only
    meters the query-state vector itself; every other field a real receiver
    needs to fuse or associate the message must be counted here too."""
    count = len(detection["boxes"])
    if not count:
        return 0
    bits = count * detection["boxes"].shape[-1] * 32  # raw box geometry
    bits += count * 32  # detector confidence score
    if query_geometry is not None:
        bits += query_geometry.numel() * 32
    return bits


def _transmit(codebook, vectors, *, normalize=False, stages=None):
    """Round-trip source messages and return reconstruction, bits, and IDs."""
    if codebook is None:
        return vectors, int(len(vectors) * vectors.shape[-1] * 32), 0
    if hasattr(codebook, "cumulative_bits"):
        if stages is None:
            stages = len(codebook.stages)
        if isinstance(stages, int):
            restored, codes = codebook.roundtrip(vectors, stages=stages)
            bits = len(vectors) * codebook.cumulative_bits[stages - 1]
            index_count = sum(code.numel() for code in codes)
        else:
            stages = stages.to(device=vectors.device, dtype=torch.long)
            if len(stages) != len(vectors):
                raise ValueError("One adaptive stage count is required per message")
            restored = torch.empty_like(vectors)
            bits = index_count = 0
            for stage_count in stages.unique(sorted=True).tolist():
                selected = stages == stage_count
                decoded, codes = codebook.roundtrip(
                    vectors[selected], stages=int(stage_count)
                )
                restored[selected] = decoded
                bits += int(selected.sum()) * codebook.cumulative_bits[stage_count - 1]
                index_count += sum(code.numel() for code in codes)
    else:
        restored, codes = codebook.roundtrip(vectors)
        bits = len(vectors) * codebook.bits_per_vector
        index_count = codes.numel()
    if normalize:
        restored = torch.nn.functional.normalize(restored, dim=-1)
    return restored, bits, int(index_count)


def main() -> None:
    args = parse_args()
    if args.frame_cache_mode != "off" and args.frame_cache_dir is None:
        raise ValueError("--frame-cache-mode requires --frame-cache-dir")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if (
        args.trackformer_box_fusion in {"score-weighted", "max-score"}
        and args.trackformer_protocol != "propagated"
    ):
        raise ValueError(
            "--trackformer-box-fusion score-weighted/max-score currently supports "
            "the propagated PQ-STF protocol only"
        )
    if (
        args.trackformer_grouping != "ego-centric"
        and (
            args.trackformer_protocol != "propagated"
            or args.trackformer_box_fusion not in {"score-weighted", "max-score"}
        )
    ):
        raise ValueError(
            "all-agent grouping requires propagated "
            "PQ-STF with --trackformer-box-fusion score-weighted or max-score"
        )
    if (
        args.trackformer_grouping == "all-agent-direct"
        and (
            args.message_codebook is not None
            or args.residual_message_codebook is not None
        )
    ):
        raise ValueError(
            "All-agent direct grouping transmits 128-D local embeddings, but "
            "the available codebooks encode 256-D propagated query states."
        )
    if (
        args.trackformer_grouping == "all-agent-symmetric-pair"
        and args.all_agent_pair_head_checkpoint is None
    ):
        raise ValueError(
            "--all-agent-pair-head-checkpoint is required for "
            "--trackformer-grouping all-agent-symmetric-pair"
        )
    if (
        args.ransac_localization_correction
        and (
            args.trackformer_protocol != "propagated"
            or args.trackformer_box_fusion not in {"score-weighted", "max-score"}
            or args.trackformer_grouping not in {
                "ego-centric", "all-agent-symmetric-pair"
            }
        )
    ):
        raise ValueError(
            "--ransac-localization-correction requires propagated PQ-STF "
            "with plain fusion and ego-centric or calibrated all-agent grouping"
        )
    if args.ransac_min_inliers < 2:
        raise ValueError("--ransac-min-inliers must be at least 2")
    if (
        args.oracle_ransac_localization_correction
        and not args.oracle_grouping
    ):
        raise ValueError(
            "--oracle-ransac-localization-correction requires "
            "--oracle-grouping"
        )
    if not 0.0 <= args.ransac_min_relative_improvement <= 1.0:
        raise ValueError(
            "--ransac-min-relative-improvement must lie in [0, 1]"
        )
    if (
        args.ransac_pair_min_probability is not None
        and not 0.0 <= args.ransac_pair_min_probability <= 1.0
    ):
        raise ValueError("--ransac-pair-min-probability must lie in [0, 1]")
    if (
        args.message_content_ablation != "none"
        and args.trackformer_protocol != "propagated"
    ):
        raise ValueError(
            "--message-content-ablation requires propagated PQ-STF"
        )
    if not 0.0 <= args.alignformer_pva_abstain_level <= 1.0:
        raise ValueError("--alignformer-pva-abstain-level must lie in [0, 1]")
    if args.alignformer_pva_refine_min_pairs < 1:
        raise ValueError("--alignformer-pva-refine-min-pairs must be at least 1")
    parse_alignformer_pva_gates_m(args.alignformer_pva_refine_gates_m)
    if (
        args.alignformer_pva_agreement_tolerance_m is not None
        and args.alignformer_pva_agreement_tolerance_m <= 0.0
    ):
        raise ValueError("--alignformer-pva-agreement-tolerance-m must be positive")
    if args.alignformer_pva_consensus_floor < 0:
        raise ValueError("--alignformer-pva-consensus-floor must be >= 0")
    freealign_config = _load_freealign(args)
    device = torch.device(args.device)

    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.tools.train_utils import create_model, to_device

    hypes = load_yaml(str(args.opencood_config), None)
    _set_noise(hypes, args)
    opencood_dataset = build_dataset(hypes, visualize=False, train=False)
    dataset = opencood_dataset
    if (args.scenario_split_file is None) != (args.scenario_role is None):
        raise ValueError(
            "--scenario-split-file and --scenario-role must be used together"
        )
    scenario_subset = None
    if args.scenario_split_file is not None:
        split = json.loads(args.scenario_split_file.read_text())
        scenario_subset = set(split[f"{args.scenario_role}_scenarios"])
        frame_indices = [
            index
            for index in range(len(dataset))
            if _scenario_id(opencood_dataset, index) in scenario_subset
        ]
        dataset = Subset(dataset, frame_indices)
    detector = create_model(hypes)
    detector_checkpoint = _load_detector_checkpoint(args.detector_dir, detector)
    model = FrozenPointPillarWithUncertainty(
        detector,
        anchor_count=int(hypes["model"]["args"]["anchor_number"]),
        hidden_channels=args.hidden_channels,
    ).to(device)
    uncertainty_epoch = model.load_uncertainty_checkpoint(
        args.uncertainty_checkpoint, map_location="cpu"
    )
    model.eval()
    trackformer, trackformer_epoch = _load_trackformer(
        args, int(detector.backbone.num_bev_features), device
    )
    pair_head, pair_head_epoch = _load_all_agent_pair_head(
        args, trackformer, device
    )
    message_codebook, message_metadata = _load_message_codebook(
        args, trackformer, device
    )
    (
        alignformer_pva_model,
        alignformer_pva_checkpoint,
        alignformer_pva_refine_config,
        alignformer_pva_abstain_config,
        alignformer_pva_agreement_config,
        alignformer_pva_embedding_encoder,
        alignformer_pva_robust_config,
        alignformer_pva_shrinkage_config,
    ) = _load_alignformer_pva(args, device)
    expected_frames = min(
        len(dataset), args.max_frames if args.max_frames is not None else len(dataset)
    )
    cache_subset = (
        {
            "role": args.scenario_role,
            "scenarios": sorted(scenario_subset),
        }
        if scenario_subset is not None
        else None
    )
    frame_cache = EvaluationFrameCache(
        args.frame_cache_dir,
        mode=args.frame_cache_mode,
        provenance=build_frame_cache_provenance(
            opencood_config=args.opencood_config,
            detector_checkpoint=detector_checkpoint,
            uncertainty_checkpoint=args.uncertainty_checkpoint,
            data_root=args.data_root,
            split=args.split,
            noise={
                "position_std_m": args.position_noise_std,
                "heading_std_deg": args.heading_noise_std_deg,
                "time_delay_ms": args.time_delay_ms,
                "seed": args.noise_seed,
            },
            scenario_subset=cache_subset,
        ),
        expected_frames=expected_frames,
    )
    if alignformer_pva_embedding_encoder is not None and frame_cache.replay:
        raise ValueError(
            "--alignformer-pva-checkpoint's embedding encoder (--embed-dim > 0 "
            "at training time) needs live detector output (spatial_features_2d) "
            "to compute context-padded cell embeddings matching training. A "
            "replayed --frame-cache-mode read-only artifact only ever stores "
            "zero-margin trackformer_agents cell tokens (for the unrelated "
            "TrackFormer/PQ-STF path), not the dense feature map this needs. "
            "Rerun with a fresh --frame-cache-dir in read-write mode (or "
            "without --frame-cache-dir at all) so the live branch runs."
        )
    loader = None
    if not frame_cache.replay:
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=opencood_dataset.collate_batch_test,
            pin_memory=args.device == "cuda",
            persistent_workers=args.workers > 0,
        )

    naive_statistics = _statistics()
    ego_only_statistics = _statistics()
    belt_statistics = _statistics()
    belt_association_counts = empty_component_counts()
    oracle_statistics = _statistics() if args.oracle_grouping else None
    oracle_max_score_statistics = (
        _statistics() if args.oracle_grouping else None
    )
    oracle_ransac_average_statistics = (
        _statistics() if args.oracle_ransac_localization_correction else None
    )
    oracle_ransac_max_score_statistics = (
        _statistics() if args.oracle_ransac_localization_correction else None
    )
    embedding_statistics = _statistics() if trackformer is not None else None
    alignformer_pva_statistics = (
        _statistics() if alignformer_pva_model is not None else None
    )
    alignformer_pva_fused_count = 0
    alignformer_pva_attempts = alignformer_pva_accepted = 0
    alignformer_pva_statistic_sum = 0.0
    alignformer_pva_statistic_count = 0
    freealign_statistics = _statistics() if freealign_config is not None else None
    freealign_fused_count = 0
    freealign_attempts = freealign_accepted = 0
    freealign_matched_nodes_sum = 0.0
    freealign_matched_nodes_count = 0
    frame_count = candidate_count = fused_count = matched_groups = 0
    oracle_fused_count = oracle_matched_groups = 0
    oracle_max_score_fused_count = 0
    oracle_ransac_fused_count = oracle_ransac_max_score_fused_count = 0
    oracle_ransac_attempts = oracle_ransac_applied = oracle_ransac_inliers = 0
    oracle_ransac_residual_sum = 0.0
    embedding_fused_count = embedding_matched_groups = 0
    trackformer_roi_fallback_frames = 0
    track_query_count = selected_track_query_count = 0
    track_query_score_sum = 0.0
    track_query_score_min = float("inf")
    track_query_score_max = float("-inf")
    message_code_count = 0
    transmitted_message_count = 0
    transmitted_message_bits = 0
    ransac_source_attempts = ransac_source_candidates = 0
    ransac_source_corrected = ransac_source_rejected = ransac_inliers = 0
    ransac_inlier_residual_sum = 0.0
    ransac_relative_improvement_sum = 0.0
    pair_head_component_counts = empty_component_counts()
    ransac_seed_group_counts = empty_component_counts()
    ransac_pair_min_probability = (
        args.pair_head_min_probability
        if args.ransac_pair_min_probability is None
        else args.ransac_pair_min_probability
    )
    final_group_counts = empty_component_counts()
    coverage_counts = {
        name: {"total": 0, "recovered": 0}
        for name in (
            "ego_detected",
            "source_only_detectable",
            "undetected_by_everyone",
        )
    }
    progress = tqdm(
        range(expected_frames) if frame_cache.replay else loader,
        total=expected_frames,
        desc=(
            "evaluating cosine association + plain box fusion"
            if args.trackformer_box_fusion in {"score-weighted", "max-score"}
            else "evaluating BELT fusion"
        ),
    )
    with torch.no_grad():
        for batch_index, cached_index_or_batch in enumerate(progress):
            if args.max_frames is not None and batch_index >= args.max_frames:
                break
            if frame_cache.replay:
                artifact = cached_to_device(frame_cache.load(batch_index), device)
                naive_boxes = artifact["naive_boxes"]
                naive_scores = artifact["naive_scores"]
                ground_truth = artifact["ground_truth"]
                detections_by_cav = artifact["detections_by_cav"]
                simple_detections_by_cav = artifact["simple_detections_by_cav"]
                trackformer_agents = artifact["trackformer_agents"]
                cav_order = artifact["cav_order"]
                ego_ground_truth = artifact["ego_ground_truth"]
                detections = [detections_by_cav[cav_id] for cav_id in cav_order]
            else:
                batch = to_device(cached_index_or_batch, device)
                outputs = OrderedDict(
                    (cav_id, model(cav_content))
                    for cav_id, cav_content in batch.items()
                )
                naive_boxes, naive_scores, ground_truth = opencood_dataset.post_process(
                    batch, outputs
                )

                detections = []
                detections_by_cav = {}
                simple_detections_by_cav = {}
                trackformer_agents = {}
                cav_order = list(batch)
                ego_ground_truth = _ego_ground_truth_reference(batch["ego"])
                for cav_id, cav_content in batch.items():
                    proposal = decode_local_proposals(
                        outputs[cav_id], cav_content, opencood_dataset.post_processor
                    )
                    # Cache ROI tokens regardless of whether this particular
                    # run enables PQ-STF.  Future association experiments can
                    # then reuse the detector-side artifact without rerunning
                    # PointPillars or reading the LiDAR frames again.
                    trackformer_agents[cav_id] = _trackformer_agent(
                        proposal, cav_content, outputs[cav_id], hypes
                    )
                    if args.covariance_noise_variant == "oracle":
                        covariance_position_std = args.position_noise_std
                        covariance_heading_std_deg = args.heading_noise_std_deg
                    else:
                        covariance_position_std = args.assumed_position_noise_std
                        covariance_heading_std_deg = args.assumed_heading_noise_std_deg
                    detection = proposal_uncertainty(
                        outputs[cav_id],
                        proposal,
                        cav_content,
                        position_noise_std=(
                            0.0 if cav_id == "ego" else covariance_position_std
                        ),
                        heading_noise_std_deg=(
                            0.0
                            if cav_id == "ego"
                            else covariance_heading_std_deg
                        ),
                    )
                    oracle_ids = _oracle_object_ids(
                        proposal, cav_content, opencood_dataset.post_processor
                    )
                    detection["agent_id"] = cav_id
                    detection["oracle_object_ids"] = oracle_ids
                    detections.append(detection)
                    detections_by_cav[cav_id] = detection
                    simple_detections_by_cav[cav_id] = _simple_detection(
                        proposal, cav_content
                    )
                    simple_detections_by_cav[cav_id]["agent_id"] = cav_id
                    simple_detections_by_cav[cav_id]["oracle_object_ids"] = oracle_ids
                    if alignformer_pva_embedding_encoder is not None:
                        # Additive: only runs (and only changes the stored
                        # artifact's shape) when an embed_dim>0 checkpoint was
                        # loaded. Every other run -- including a geometry-only
                        # --alignformer-pva-checkpoint and every run without
                        # one at all -- never executes this branch, so
                        # simple_detections_by_cav (and the frame_cache
                        # artifact built from it below) stay byte-identical.
                        simple_detections_by_cav[cav_id].update(
                            _context_embedding_cells(
                                proposal, outputs[cav_id], hypes,
                                margin=args.alignformer_pva_context_margin_m,
                            )
                        )
                frame_cache.store(
                    batch_index,
                    {
                        "naive_boxes": naive_boxes,
                        "naive_scores": naive_scores,
                        "ground_truth": ground_truth,
                        "detections_by_cav": detections_by_cav,
                        "simple_detections_by_cav": simple_detections_by_cav,
                        "trackformer_agents": trackformer_agents,
                        "cav_order": cav_order,
                        "ego_ground_truth": ego_ground_truth,
                    },
                )
            if args.fast_ap:
                _evaluate_prediction_fast(
                    naive_boxes,
                    naive_scores,
                    ground_truth,
                    opencood_dataset.post_processor,
                    naive_statistics,
                )
            else:
                _evaluate_prediction(
                    naive_boxes,
                    naive_scores,
                    ground_truth,
                    naive_statistics,
                )
            _postprocess_and_evaluate(
                simple_detections_by_cav["ego"]["boxes"],
                simple_detections_by_cav["ego"]["scores"],
                ground_truth,
                opencood_dataset.post_processor,
                ego_only_statistics,
                fast=args.fast_ap,
            )
            fused = fuse_detections(
                detections, maximum_distance=args.association_distance
            )
            if not args.skip_belt:
                merge_component_counts(
                    belt_association_counts,
                    association_counts(
                        detections,
                        fused["groups"],
                        maximum_distance=args.association_distance,
                    ),
                )
                belt_boxes, belt_scores = _postprocess_fused_boxes(
                    fused["boxes"],
                    fused["scores"],
                    opencood_dataset.post_processor,
                )
                _evaluate_prediction(
                    belt_boxes, belt_scores, ground_truth, belt_statistics
                )
            if alignformer_pva_model is not None:
                # Propose-Verify-Abstain alternative to
                # --ransac-localization-correction: geometry-only by default
                # (operates entirely on simple_detections_by_cav, no
                # PQ-STF/TrackFormer required, so it runs unconditionally here
                # rather than inside the trackformer-gated branches below),
                # or fed a real per-object embedding when
                # alignformer_pva_embedding_encoder is not None (Phase 1 of
                # the richer-embedding plan) -- simple_detections_by_cav was
                # additively augmented with cell_* fields above for exactly
                # that case.
                corrected_simple_detections, pva_diagnostics = (
                    correct_simple_detections(
                        alignformer_pva_model,
                        simple_detections_by_cav,
                        cav_order,
                        refine_config=alignformer_pva_refine_config,
                        abstain_config=alignformer_pva_abstain_config,
                        agreement_config=alignformer_pva_agreement_config,
                        embedding_encoder=alignformer_pva_embedding_encoder,
                        robust_config=alignformer_pva_robust_config,
                        shrinkage_config=alignformer_pva_shrinkage_config,
                    )
                )
                pva_fused = fuse_corrected_simple_detections(
                    corrected_simple_detections,
                    cav_order,
                    maximum_distance=args.alignformer_pva_association_distance,
                )
                _postprocess_and_evaluate(
                    pva_fused["boxes"],
                    pva_fused["scores"],
                    ground_truth,
                    opencood_dataset.post_processor,
                    alignformer_pva_statistics,
                    fast=args.fast_ap,
                )
                alignformer_pva_fused_count += len(pva_fused["boxes"])
                for pva_stats in pva_diagnostics.values():
                    alignformer_pva_attempts += int(pva_stats["attempted"])
                    alignformer_pva_accepted += int(pva_stats["accepted"])
                    if pva_stats["statistic"] is not None:
                        alignformer_pva_statistic_sum += pva_stats["statistic"]
                        alignformer_pva_statistic_count += 1
            if freealign_config is not None:
                # Training-free competitor row (embedding_aware_belt_fusion.
                # freealign): same simple_detections_by_cav input every other
                # corrected-detections row reads, so the detections, the
                # noise draws and the fusion step are shared by construction
                # and only the alignment algorithm differs. Runs
                # unconditionally here (not gated behind the trackformer
                # branches below), exactly like the alignformer_pva row.
                freealign_corrected, freealign_diagnostics = (
                    freealign_correct_simple_detections(
                        simple_detections_by_cav, cav_order, freealign_config
                    )
                )
                freealign_fused = freealign_fuse_corrected_simple_detections(
                    freealign_corrected,
                    cav_order,
                    maximum_distance=args.freealign_association_distance,
                )
                _postprocess_and_evaluate(
                    freealign_fused["boxes"],
                    freealign_fused["scores"],
                    ground_truth,
                    opencood_dataset.post_processor,
                    freealign_statistics,
                    fast=args.fast_ap,
                )
                freealign_fused_count += len(freealign_fused["boxes"])
                for free_stats in freealign_diagnostics.values():
                    freealign_attempts += int(free_stats["attempted"])
                    freealign_accepted += int(free_stats["accepted"])
                    if free_stats["attempted"]:
                        freealign_matched_nodes_sum += free_stats["matched_nodes"]
                        freealign_matched_nodes_count += 1
            if args.oracle_grouping:
                oracle_groups = _oracle_groups(simple_detections_by_cav.values())
                oracle = fuse_groups_score_weighted(oracle_groups)
                _postprocess_and_evaluate(
                    oracle["boxes"],
                    oracle["scores"],
                    ground_truth,
                    opencood_dataset.post_processor,
                    oracle_statistics,
                    fast=args.fast_ap,
                )
                oracle_fused_count += len(oracle["boxes"])
                oracle_matched_groups += int(
                    (oracle["member_counts"] > 1).sum()
                )
                oracle_max_score = fuse_groups_max_score(oracle_groups)
                oracle_max_score_boxes, oracle_max_score_scores = (
                    _postprocess_fused_boxes(
                        oracle_max_score["boxes"],
                        oracle_max_score["scores"],
                        opencood_dataset.post_processor,
                    )
                )
                _evaluate_prediction(
                    oracle_max_score_boxes,
                    oracle_max_score_scores,
                    ground_truth,
                    oracle_max_score_statistics,
                )
                oracle_max_score_fused_count += len(oracle_max_score["boxes"])
                if args.oracle_ransac_localization_correction:
                    corrections = {}
                    for cav_id, (source_centers, ego_centers) in (
                        _oracle_prediction_correspondences(
                            simple_detections_by_cav
                        ).items()
                    ):
                        oracle_ransac_attempts += 1
                        correction = estimate_se2_ransac(
                            source_centers,
                            ego_centers,
                            inlier_threshold=args.ransac_inlier_threshold,
                        )
                        if correction is None:
                            continue
                        inlier_count = int(correction["inliers"].sum())
                        if inlier_count < args.ransac_min_inliers:
                            continue
                        corrections[cav_id] = correction
                        oracle_ransac_applied += 1
                        oracle_ransac_inliers += inlier_count
                        oracle_ransac_residual_sum += float(
                            correction["residual"][correction["inliers"]].sum()
                        )
                    corrected_oracle_detections = [
                        (
                            apply_se2_correction(detection, corrections[cav_id])
                            if cav_id in corrections
                            else detection
                        )
                        for cav_id, detection in simple_detections_by_cav.items()
                    ]
                    corrected_oracle_groups = _oracle_groups(
                        corrected_oracle_detections
                    )
                    corrected_oracle = fuse_groups_score_weighted(
                        corrected_oracle_groups
                    )
                    corrected_boxes, corrected_scores = _postprocess_fused_boxes(
                        corrected_oracle["boxes"],
                        corrected_oracle["scores"],
                        opencood_dataset.post_processor,
                    )
                    _evaluate_prediction(
                        corrected_boxes,
                        corrected_scores,
                        ground_truth,
                        oracle_ransac_average_statistics,
                    )
                    oracle_ransac_fused_count += len(corrected_oracle["boxes"])
                    corrected_oracle_max = fuse_groups_max_score(
                        corrected_oracle_groups
                    )
                    corrected_max_boxes, corrected_max_scores = (
                        _postprocess_fused_boxes(
                            corrected_oracle_max["boxes"],
                            corrected_oracle_max["scores"],
                            opencood_dataset.post_processor,
                        )
                    )
                    _evaluate_prediction(
                        corrected_max_boxes,
                        corrected_max_scores,
                        ground_truth,
                        oracle_ransac_max_score_statistics,
                    )
                    oracle_ransac_max_score_fused_count += len(
                        corrected_oracle_max["boxes"]
                    )
            if trackformer is not None and args.trackformer_protocol == "independent":
                ego_detection = detections_by_cav.get("ego")
                if ego_detection is None:
                    raise RuntimeError("Late-fusion batch has no ego agent")
                # A proposal can fall completely outside the BEV feature map.
                # It still has a valid detector box, but cannot obtain an
                # ROI-derived embedding.  Preserve the frame through the
                # geometry-only BELT result rather than aborting evaluation or
                # silently discarding its detections.
                if (
                    not len(ego_detection["boxes"])
                    or not all(
                        _has_valid_trackformer_roi(agent)
                        for agent in trackformer_agents.values()
                    )
                ):
                    embedded = fused
                    trackformer_roi_fallback_frames += 1
                else:
                    ego_output = trackformer.forward_agent(
                        trackformer_agents["ego"]
                    )
                    ego_detection["embeddings"] = ego_output["embeddings"][0]
                    independent_detections = [ego_detection]
                    for cav_id in cav_order:
                        if cav_id == "ego":
                            continue
                        detection = detections_by_cav[cav_id]
                        source_agent = trackformer_agents[cav_id]
                        if not len(source_agent["boxes"]):
                            independent_detections.append(detection)
                            continue
                        source_output = trackformer.forward_agent(source_agent)
                        stage_counts = (
                            _adaptive_stage_counts(
                                detection, args.adaptive_rate_thresholds
                            )
                            if args.adaptive_rate_thresholds is not None
                            else args.residual_stages
                        )
                        embedding, message_bits, code_count = _transmit(
                            message_codebook,
                            source_output["embeddings"][0],
                            normalize=message_codebook is not None,
                            stages=stage_counts,
                        )
                        transmitted_message_bits += message_bits
                        transmitted_message_bits += _message_overhead_bits(detection)
                        message_code_count += code_count
                        transmitted_message_count += len(embedding)
                        detection["embeddings"] = embedding
                        independent_detections.append(detection)
                    embedded = fuse_detections(
                        independent_detections,
                        maximum_distance=args.embedding_distance,
                        association="embedding",
                        minimum_similarity=args.embedding_min_similarity,
                    )
                embedding_boxes, embedding_scores = _postprocess_fused_boxes(
                    embedded["boxes"],
                    embedded["scores"],
                    opencood_dataset.post_processor,
                )
                _evaluate_prediction(
                    embedding_boxes,
                    embedding_scores,
                    ground_truth,
                    embedding_statistics,
                )
                embedding_fused_count += len(embedded["boxes"])
                embedding_matched_groups += int(
                    (embedded["member_counts"] > 1).sum()
                )
            if trackformer is not None and args.trackformer_protocol == "propagated":
                if args.trackformer_grouping in {
                    "all-agent-direct", "all-agent-symmetric-pair"
                }:
                    # All-agent PQ-STF ablations: form groups over ego and all
                    # source CAVs before plain score-weighted box fusion.
                    # Neither branch uses BELT evidence or covariance weighting.
                    direct_detections = []
                    fallback_detections = []
                    embedding_dim = int(trackformer.embedding_head.out_features)
                    state_dim = int(trackformer.embedding_head.in_features)
                    for cav_id in cav_order:
                        detection = detections_by_cav[cav_id]
                        agent = trackformer_agents[cav_id]
                        if not len(agent["boxes"]):
                            if args.trackformer_grouping == "all-agent-direct":
                                detection["embeddings"] = detection["boxes"].new_empty(
                                    (0, embedding_dim)
                                )
                            else:
                                detection["query_states"] = detection["boxes"].new_empty(
                                    (0, state_dim)
                                )
                                detection["query_geometry"] = detection["boxes"].new_empty(
                                    (0, 9)
                                )
                            direct_detections.append(detection)
                            continue
                        if not _has_valid_trackformer_roi(agent):
                            # Do not silently discard a detector proposal for
                            # which no BEV ROI cell exists.  It remains an
                            # unassociated singleton in the final output.
                            fallback_detections.append(detection)
                            continue
                        local_output = trackformer.forward_agent(agent)
                        if args.trackformer_grouping == "all-agent-direct":
                            detection["embeddings"] = local_output["embeddings"][0]
                            if cav_id != "ego":
                                transmitted_message_count += len(detection["boxes"])
                                transmitted_message_bits += (
                                    len(detection["boxes"]) * embedding_dim * 32
                                )
                                transmitted_message_bits += _message_overhead_bits(
                                    detection
                                )
                        else:
                            states = local_output["hs_embed"][0]
                            if cav_id != "ego":
                                stage_counts = (
                                    _adaptive_stage_counts(
                                        detection, args.adaptive_rate_thresholds
                                    )
                                    if args.adaptive_rate_thresholds is not None
                                    else args.residual_stages
                                )
                                states, message_bits, code_count = _transmit(
                                    message_codebook, states, stages=stage_counts
                                )
                                # This is a causal message test: geometry,
                                # boxes, scores, grouping, and fusion remain
                                # unchanged while only the transmitted PQ
                                # query-state content is destroyed.
                                states = _ablate_message_content(
                                    states, args.message_content_ablation
                                )
                                transmitted_message_bits += message_bits
                                transmitted_message_bits += _message_overhead_bits(
                                    detection, local_output["query_geometry"]
                                )
                                message_code_count += code_count
                                transmitted_message_count += len(states)
                            detection["query_states"] = states
                            detection["query_geometry"] = local_output["query_geometry"]
                        direct_detections.append(detection)
                    if direct_detections:
                        if args.trackformer_grouping == "all-agent-direct":
                            groups = associate_by_embedding(
                                direct_detections,
                                maximum_distance=args.embedding_distance,
                                minimum_similarity=args.embedding_min_similarity,
                            )
                        else:
                            merge_component_counts(
                                pair_head_component_counts,
                                pair_head_counts(
                                    direct_detections,
                                    pair_head,
                                    maximum_distance=args.embedding_distance,
                                    minimum_probability=args.pair_head_min_probability,
                                ),
                            )
                            groups = associate_by_symmetric_pair_head(
                                direct_detections,
                                pair_head,
                                maximum_distance=args.embedding_distance,
                                minimum_probability=args.pair_head_min_probability,
                            )
                    else:
                        groups = []
                    corrected_fallback_detections = fallback_detections
                    if (
                        args.ransac_localization_correction
                        and args.trackformer_grouping == "all-agent-symmetric-pair"
                    ):
                        # Seed correspondences use a wider positional gate only
                        # to estimate each source CAV's shared pose error. The
                        # final grouping below always returns to the normal 5 m
                        # candidate gate after applying accepted corrections.
                        seed_groups = associate_by_symmetric_pair_head(
                            direct_detections,
                            pair_head,
                            maximum_distance=args.ransac_seed_distance,
                            minimum_probability=ransac_pair_min_probability,
                        ) if direct_detections else []
                        merge_component_counts(
                            ransac_seed_group_counts, group_counts(seed_groups)
                        )
                        corrections = {}
                        for source_id, (source_centers, ego_centers) in (
                            correspondences_by_source(seed_groups).items()
                        ):
                            ransac_source_attempts += 1
                            correction = estimate_se2_ransac(
                                source_centers,
                                ego_centers,
                                inlier_threshold=args.ransac_inlier_threshold,
                            )
                            if correction is None:
                                continue
                            ransac_source_candidates += 1
                            inlier_count = int(correction["inliers"].sum())
                            relative_improvement = float(
                                correction["relative_improvement"]
                            )
                            if (
                                inlier_count < args.ransac_min_inliers
                                or relative_improvement
                                < args.ransac_min_relative_improvement
                            ):
                                ransac_source_rejected += 1
                                continue
                            corrections[source_id] = correction
                            ransac_source_corrected += 1
                            ransac_inliers += inlier_count
                            ransac_relative_improvement_sum += relative_improvement
                            ransac_inlier_residual_sum += float(
                                correction["residual"][correction["inliers"]].sum()
                            )
                        direct_detections = [
                            (
                                _apply_symmetric_ransac_correction(
                                    detection, corrections[detection["agent_id"]]
                                )
                                if detection.get("agent_id") in corrections
                                else detection
                            )
                            for detection in direct_detections
                        ]
                        corrected_fallback_detections = [
                            (
                                _apply_symmetric_ransac_correction(
                                    detection, corrections[detection["agent_id"]]
                                )
                                if detection.get("agent_id") in corrections
                                else detection
                            )
                            for detection in fallback_detections
                        ]
                        groups = associate_by_symmetric_pair_head(
                            direct_detections,
                            pair_head,
                            maximum_distance=args.embedding_distance,
                            minimum_probability=args.pair_head_min_probability,
                        ) if direct_detections else []
                    for detection in corrected_fallback_detections:
                        groups.extend(simple_singleton_groups(detection))
                    if args.trackformer_grouping == "all-agent-symmetric-pair":
                        merge_component_counts(final_group_counts, group_counts(groups))
                    embedded = (
                        fuse_groups_score_weighted(groups)
                        if args.trackformer_box_fusion == "score-weighted"
                        else fuse_groups_max_score(groups)
                    )
                    embedding_boxes, embedding_scores = _postprocess_fused_boxes(
                        embedded["boxes"],
                        embedded["scores"],
                        opencood_dataset.post_processor,
                    )
                    _evaluate_prediction(
                        embedding_boxes,
                        embedding_scores,
                        ground_truth,
                        embedding_statistics,
                    )
                    if args.trackformer_grouping == "all-agent-symmetric-pair":
                        _merge_coverage(
                            coverage_counts,
                            _coverage_categories(
                                detections_by_cav,
                                ego_ground_truth,
                                embedding_boxes,
                                opencood_dataset.post_processor,
                            ),
                        )
                    embedding_fused_count += len(embedded["boxes"])
                    embedding_matched_groups += int(
                        (embedded["member_counts"] > 1).sum()
                    )
                    frame_count += 1
                    candidate_count += sum(
                        len(item["boxes"]) for item in detections
                    )
                    fused_count += len(fused["boxes"])
                    matched_groups += int((fused["member_counts"] > 1).sum())
                    progress.set_postfix(
                        candidates=candidate_count // frame_count,
                        fused=fused_count // frame_count,
                        groups=matched_groups // frame_count,
                    )
                    continue
                if args.trackformer_box_fusion in {"score-weighted", "max-score"}:
                    association_detections = simple_detections_by_cav
                else:
                    association_detections = detections_by_cav
                ego_detection = association_detections.get("ego")
                propagated_sources = []
                unselected_sources = []
                for cav_id in cav_order:
                    if cav_id == "ego":
                        continue
                    detection = association_detections[cav_id]
                    source_agent = trackformer_agents[cav_id]
                    ego_agent = trackformer_agents["ego"]
                    if (
                        not len(source_agent["boxes"])
                        or not len(ego_agent["boxes"])
                    ):
                        unselected_sources.append(detection)
                        continue
                    source_probe = trackformer.forward_agent(source_agent)
                    object_score = source_probe["pred_logits"][0].softmax(-1)[:, 0]
                    selected = torch.nonzero(
                        object_score >= args.track_query_score_threshold
                    ).squeeze(1)
                    unselected = torch.nonzero(
                        object_score < args.track_query_score_threshold
                    ).squeeze(1)
                    track_query_count += len(object_score)
                    selected_track_query_count += len(selected)
                    track_query_score_sum += float(object_score.sum())
                    track_query_score_min = min(
                        track_query_score_min, float(object_score.min())
                    )
                    track_query_score_max = max(
                        track_query_score_max, float(object_score.max())
                    )
                    if len(unselected):
                        unselected_sources.append(
                            _select_detection(detection, unselected)
                        )
                    if not len(selected):
                        continue
                    source_agent = _select_trackformer_agent(
                        source_agent, selected
                    )
                    source_output = trackformer.forward_agent(source_agent)
                    source = _select_detection(detection, selected)
                    stage_counts = (
                        _adaptive_stage_counts(
                            source, args.adaptive_rate_thresholds
                        )
                        if args.adaptive_rate_thresholds is not None
                        else args.residual_stages
                    )
                    message_state, message_bits, code_count = _transmit(
                        message_codebook,
                        source_output["hs_embed"][0],
                        stages=stage_counts,
                    )
                    message_state = _ablate_message_content(
                        message_state, args.message_content_ablation
                    )
                    transmitted_message_bits += message_bits
                    transmitted_message_bits += _message_overhead_bits(
                        source, source_output["query_geometry"]
                    )
                    message_code_count += code_count
                    transmitted_message_count += len(message_state)
                    source_embeddings = trackformer.association_embeddings(
                        message_state, source_output["query_geometry"]
                    )
                    target_output = trackformer.forward_agent(
                        ego_agent,
                        track_states=message_state,
                        track_reference_boxes=source_output["pred_boxes"][0],
                        track_reference_geometry=source_output["query_geometry"],
                    )
                    source["embeddings"] = source_embeddings
                    source["ego_embeddings"] = target_output["embeddings"][0][
                        target_output["track_count"] :
                    ]
                    propagated_sources.append(source)
                if ego_detection is None:
                    raise RuntimeError("Late-fusion batch has no ego agent")
                if propagated_sources:
                    association = (
                        associate_ego_with_propagated_sources_simple
                        if args.trackformer_box_fusion in {"score-weighted", "max-score"}
                        else associate_ego_with_propagated_sources
                    )
                    seed_groups = association(
                        ego_detection,
                        propagated_sources,
                        maximum_distance=(
                            args.ransac_seed_distance
                            if args.ransac_localization_correction
                            else args.embedding_distance
                        ),
                        minimum_similarity=args.embedding_min_similarity,
                    )
                    corrections = {}
                    if args.ransac_localization_correction:
                        for source_id, (source_centers, ego_centers) in (
                            correspondences_by_source(seed_groups).items()
                        ):
                            ransac_source_attempts += 1
                            correction = estimate_se2_ransac(
                                source_centers,
                                ego_centers,
                                inlier_threshold=args.ransac_inlier_threshold,
                            )
                            if correction is None:
                                continue
                            ransac_source_candidates += 1
                            inlier_count = int(correction["inliers"].sum())
                            relative_improvement = float(
                                correction["relative_improvement"]
                            )
                            if (
                                inlier_count < args.ransac_min_inliers
                                or relative_improvement
                                < args.ransac_min_relative_improvement
                            ):
                                ransac_source_rejected += 1
                                continue
                            corrections[source_id] = correction
                            ransac_source_corrected += 1
                            ransac_inliers += inlier_count
                            ransac_relative_improvement_sum += relative_improvement
                            ransac_inlier_residual_sum += float(
                                correction["residual"][correction["inliers"]].sum()
                            )
                        corrected_sources = [
                            apply_se2_correction(source, corrections[source["agent_id"]])
                            if source.get("agent_id") in corrections
                            else source
                            for source in propagated_sources
                        ]
                        unselected_sources = [
                            apply_se2_correction(source, corrections[source["agent_id"]])
                            if source.get("agent_id") in corrections
                            else source
                            for source in unselected_sources
                        ]
                        groups = association(
                            ego_detection,
                            corrected_sources,
                            maximum_distance=args.embedding_distance,
                            minimum_similarity=args.embedding_min_similarity,
                        )
                    else:
                        groups = seed_groups
                else:
                    groups = (
                        simple_singleton_groups(ego_detection)
                        if args.trackformer_box_fusion in {"score-weighted", "max-score"}
                        else singleton_groups(ego_detection)
                    )
                for source in unselected_sources:
                    groups.extend(
                        simple_singleton_groups(source)
                        if args.trackformer_box_fusion in {"score-weighted", "max-score"}
                        else singleton_groups(source)
                    )
                embedded = (
                    fuse_groups_score_weighted(groups)
                    if args.trackformer_box_fusion == "score-weighted"
                    else (
                        fuse_groups_max_score(groups)
                        if args.trackformer_box_fusion == "max-score"
                        else fuse_groups(groups)
                    )
                )
                embedding_boxes, embedding_scores = _postprocess_fused_boxes(
                    embedded["boxes"],
                    embedded["scores"],
                    opencood_dataset.post_processor,
                )
                _evaluate_prediction(
                    embedding_boxes,
                    embedding_scores,
                    ground_truth,
                    embedding_statistics,
                )
                embedding_fused_count += len(embedded["boxes"])
                embedding_matched_groups += int(
                    (embedded["member_counts"] > 1).sum()
                )
            frame_count += 1
            candidate_count += sum(len(item["boxes"]) for item in detections)
            fused_count += len(fused["boxes"])
            matched_groups += int((fused["member_counts"] > 1).sum())
            progress.set_postfix(
                candidates=candidate_count // frame_count,
                fused=fused_count // frame_count,
                groups=matched_groups // frame_count,
            )

    result = {
        "frames": frame_count,
        "ap_protocol": (
            "global-score-sorted"
            if args.global_sort_detections
            else "per-frame-score-order"
        ),
        "detector_checkpoint": str(detector_checkpoint.resolve()),
        "uncertainty_checkpoint": str(args.uncertainty_checkpoint.resolve()),
        "uncertainty_epoch": uncertainty_epoch,
        "noise": {
            "position_std_m": args.position_noise_std,
            "heading_std_deg": args.heading_noise_std_deg,
            "time_delay_ms": args.time_delay_ms,
            "seed": args.noise_seed,
            "mode": args.noise_mode,
        },
        "covariance_noise_variant": args.covariance_noise_variant,
        "assumed_position_noise_std_m": args.assumed_position_noise_std,
        "assumed_heading_noise_std_deg": args.assumed_heading_noise_std_deg,
        "association_distance_m": args.association_distance,
        "embedding_distance_m": args.embedding_distance,
        "embedding_min_similarity": args.embedding_min_similarity,
        "message_content_ablation": args.message_content_ablation,
        "track_query_score_threshold": args.track_query_score_threshold,
        "frame_cache": {
            "directory": (
                str(args.frame_cache_dir.resolve())
                if args.frame_cache_dir is not None
                else None
            ),
            "mode": args.frame_cache_mode,
            "replayed": frame_cache.replay,
            "artifact_boundary": (
                "PointPillars decoded proposals, uncertainty, and ROI tokens; "
                "PQ-STF states are recomputed for the requested checkpoint"
            ),
        },
        "ransac_localization": {
            "enabled": args.ransac_localization_correction,
            "seed_distance_m": args.ransac_seed_distance,
            "seed_pair_min_probability": ransac_pair_min_probability,
            "inlier_threshold_m": args.ransac_inlier_threshold,
            "minimum_inliers": args.ransac_min_inliers,
            "minimum_relative_improvement": args.ransac_min_relative_improvement,
            "source_estimation_attempts": ransac_source_attempts,
            "source_correction_candidates": ransac_source_candidates,
            "source_corrections_applied": ransac_source_corrected,
            "source_corrections_rejected": ransac_source_rejected,
            "mean_inliers_per_applied_correction": (
                ransac_inliers / max(ransac_source_corrected, 1)
            ),
            "mean_relative_improvement_per_applied_correction": (
                ransac_relative_improvement_sum
                / max(ransac_source_corrected, 1)
            ),
            "mean_inlier_residual_m": (
                ransac_inlier_residual_sum / max(ransac_inliers, 1)
            ),
        },
        "component_diagnostics": (
            {
                "pair_head": summarize_component_counts(
                    pair_head_component_counts
                ),
                "ransac_seed_grouping": summarize_group_counts(
                    ransac_seed_group_counts
                ),
                "final_grouping": summarize_group_counts(final_group_counts),
                "ground_truth_coverage": _summarize_coverage(coverage_counts),
                "label_source": (
                    "OPV2V physical object IDs, evaluation only; never used "
                    "by deployed association, RANSAC, or fusion"
                ),
            }
            if args.trackformer_grouping == "all-agent-symmetric-pair"
            else None
        ),
        "trackformer_box_fusion": (
            args.trackformer_box_fusion if trackformer is not None else None
        ),
        "trackformer_protocol": (
            args.trackformer_protocol if trackformer is not None else None
        ),
        "trackformer_grouping": (
            args.trackformer_grouping if trackformer is not None else None
        ),
        "pair_head_min_probability": (
            args.pair_head_min_probability
            if args.trackformer_grouping == "all-agent-symmetric-pair"
            else None
        ),
        "association_message": (
            {
                "codebook": (
                    str((args.message_codebook or args.residual_message_codebook).resolve())
                    if (args.message_codebook or args.residual_message_codebook) is not None
                    else None
                ),
                "representation": (
                    "direct-local-embedding"
                    if args.trackformer_grouping == "all-agent-direct"
                    else (
                        message_metadata["representation"]
                        if message_metadata is not None
                        else (
                            "query-state"
                            if args.trackformer_protocol == "propagated"
                            else "embedding"
                        )
                    )
                ),
                "bits_per_source_object": (
                    int(trackformer.embedding_head.out_features) * 32
                    if args.trackformer_grouping == "all-agent-direct"
                    else (
                        int(message_metadata["bits_per_vector"])
                        if message_metadata is not None
                        and "bits_per_vector" in message_metadata
                        else (
                            message_metadata["cumulative_bits"][
                                (args.residual_stages or len(message_codebook.stages)) - 1
                            ]
                            if message_metadata is not None
                            and args.adaptive_rate_thresholds is None
                            else (
                                None
                                if message_metadata is not None
                                else (
                                    int(trackformer.embedding_head.in_features)
                                    if args.trackformer_protocol == "propagated"
                                    else int(trackformer.embedding_head.out_features)
                                ) * 32
                            )
                        )
                    )
                ),
                "mean_source_objects_per_frame": (
                    transmitted_message_count / max(frame_count, 1)
                ),
                "mean_association_message_bytes_per_frame": (
                    transmitted_message_bits / 8 / max(frame_count, 1)
                ),
                "mean_association_bits_per_source_object": (
                    transmitted_message_bits / max(transmitted_message_count, 1)
                ),
                "residual_stages": args.residual_stages,
                "adaptive_rate_thresholds": args.adaptive_rate_thresholds,
                "total_code_indices": message_code_count,
            }
            if trackformer is not None
            else None
        ),
        "scenario_subset": (
            {
                "file": str(args.scenario_split_file.resolve()),
                "role": args.scenario_role,
                "scenario_count": len(scenario_subset),
            }
            if scenario_subset is not None
            else None
        ),
        "mean_candidates_per_frame": candidate_count / max(frame_count, 1),
        "mean_fused_boxes_per_frame": fused_count / max(frame_count, 1),
        "mean_multi_agent_groups_per_frame": matched_groups / max(
            frame_count, 1
        ),
        "naive_late": _ap(naive_statistics, args.global_sort_detections),
        "ego_only": _ap(ego_only_statistics, args.global_sort_detections),
        "belt_geometry": (
            None if args.skip_belt else _ap(belt_statistics, args.global_sort_detections)
        ),
        "belt_association": (
            None if args.skip_belt else summarize_component_counts(belt_association_counts)
        ),
        **(
            {
                "alignformer_pva": _ap(
                    alignformer_pva_statistics, args.global_sort_detections
                ),
                "alignformer_pva_checkpoint": str(
                    args.alignformer_pva_checkpoint.resolve()
                ),
                "alignformer_pva_refine_mode": args.alignformer_pva_refine_mode,
                "alignformer_pva_refine_min_pairs": (
                    args.alignformer_pva_refine_min_pairs
                ),
                "alignformer_pva_refine_inlier_m": (
                    args.alignformer_pva_refine_inlier_m
                ),
                "alignformer_pva_refine_gates_m": list(
                    alignformer_pva_refine_config.gates_m
                ),
                "alignformer_pva_abstain_level": args.alignformer_pva_abstain_level,
                "alignformer_pva_agreement_tolerance_m": (
                    args.alignformer_pva_agreement_tolerance_m
                ),
                "alignformer_pva_consensus_floor": args.alignformer_pva_consensus_floor,
                "alignformer_pva_association_distance_m": (
                    args.alignformer_pva_association_distance
                ),
                "mean_alignformer_pva_fused_boxes_per_frame": (
                    alignformer_pva_fused_count / max(frame_count, 1)
                ),
                "alignformer_pva_correction": {
                    "attempts": alignformer_pva_attempts,
                    "accepted": alignformer_pva_accepted,
                    "mean_offset_statistic": (
                        alignformer_pva_statistic_sum
                        / max(alignformer_pva_statistic_count, 1)
                    ),
                },
            }
            if alignformer_pva_statistics is not None
            else {}
        ),
        **(
            {
                "freealign": _ap(freealign_statistics, args.global_sort_detections),
                "freealign_config": freealign_config.to_dict(),
                "freealign_calibration": (
                    str(args.freealign_calibration.resolve())
                    if args.freealign_calibration is not None
                    else None
                ),
                "freealign_association_distance_m": (
                    args.freealign_association_distance
                ),
                "mean_freealign_fused_boxes_per_frame": (
                    freealign_fused_count / max(frame_count, 1)
                ),
                "freealign_correction": {
                    "attempts": freealign_attempts,
                    "accepted": freealign_accepted,
                    "mean_matched_nodes": (
                        freealign_matched_nodes_sum
                        / max(freealign_matched_nodes_count, 1)
                    ),
                },
            }
            if freealign_statistics is not None
            else {}
        ),
        **(
            {
                "oracle_physical_id_score_weighted": _ap(
                    oracle_statistics, args.global_sort_detections
                ),
                "mean_oracle_fused_boxes_per_frame": (
                    oracle_fused_count / max(frame_count, 1)
                ),
                "mean_oracle_multi_agent_groups_per_frame": (
                    oracle_matched_groups / max(frame_count, 1)
                ),
                "oracle_physical_id_max_score": _ap(
                    oracle_max_score_statistics, args.global_sort_detections
                ),
                "mean_oracle_max_score_fused_boxes_per_frame": (
                    oracle_max_score_fused_count / max(frame_count, 1)
                ),
                **(
                    {
                        "oracle_id_ransac_score_weighted": _ap(
                            oracle_ransac_average_statistics,
                            args.global_sort_detections,
                        ),
                        "oracle_id_ransac_max_score": _ap(
                            oracle_ransac_max_score_statistics,
                            args.global_sort_detections,
                        ),
                        "oracle_id_ransac": {
                            "attempts": oracle_ransac_attempts,
                            "corrections_applied": oracle_ransac_applied,
                            "mean_inliers_per_applied_correction": (
                                oracle_ransac_inliers
                                / max(oracle_ransac_applied, 1)
                            ),
                            "mean_inlier_residual_m": (
                                oracle_ransac_residual_sum
                                / max(oracle_ransac_inliers, 1)
                            ),
                            "inlier_threshold_m": args.ransac_inlier_threshold,
                            "minimum_inliers": args.ransac_min_inliers,
                        },
                        "mean_oracle_id_ransac_fused_boxes_per_frame": (
                            oracle_ransac_fused_count / max(frame_count, 1)
                        ),
                        "mean_oracle_id_ransac_max_score_fused_boxes_per_frame": (
                            oracle_ransac_max_score_fused_count
                            / max(frame_count, 1)
                        ),
                    }
                    if args.oracle_ransac_localization_correction
                    else {}
                ),
            }
            if oracle_statistics is not None
            else {}
        ),
        **(
            {
                "trackformer_checkpoint": str(
                    args.trackformer_checkpoint.resolve()
                ),
                "trackformer_epoch": trackformer_epoch,
                "trackformer_roi_fallback_frames": (
                    trackformer_roi_fallback_frames
                ),
                "mean_trackformer_fused_boxes_per_frame": (
                    embedding_fused_count / max(frame_count, 1)
                ),
                "mean_trackformer_multi_agent_groups_per_frame": (
                    embedding_matched_groups / max(frame_count, 1)
                ),
                "mean_trackformer_source_queries_per_frame": (
                    None
                    if args.trackformer_grouping != "ego-centric"
                    else track_query_count / max(frame_count, 1)
                ),
                "mean_selected_track_queries_per_frame": (
                    None
                    if args.trackformer_grouping != "ego-centric"
                    else selected_track_query_count / max(frame_count, 1)
                ),
                "mean_trackformer_source_object_score": (
                    None
                    if args.trackformer_grouping != "ego-centric"
                    else track_query_score_sum / max(track_query_count, 1)
                ),
                "min_trackformer_source_object_score": (
                    None
                    if args.trackformer_grouping != "ego-centric"
                    else track_query_score_min
                ),
                "max_trackformer_source_object_score": (
                    None
                    if args.trackformer_grouping != "ego-centric"
                    else track_query_score_max
                ),
                "all_agent_pair_head_checkpoint": (
                    str(args.all_agent_pair_head_checkpoint.resolve())
                    if args.trackformer_grouping == "all-agent-symmetric-pair"
                    else None
                ),
                "all_agent_pair_head_epoch": pair_head_epoch,
                **{
                    (
                        "simple_trackformer"
                        if args.trackformer_box_fusion in {"score-weighted", "max-score"}
                        else "belt_trackformer"
                    ): _ap(embedding_statistics, args.global_sort_detections)
                },
            }
            if embedding_statistics is not None
            else {}
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    label = (
        "SIMPLE_TRACKFORMER"
        if args.trackformer_box_fusion in {"score-weighted", "max-score"}
        else "BELT"
    )
    print(label + " " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
