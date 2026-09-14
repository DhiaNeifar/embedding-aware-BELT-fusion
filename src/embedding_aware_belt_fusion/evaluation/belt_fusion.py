"""Evaluate OpenCOOD naive and BELT-style late fusion on the same frames."""

from __future__ import annotations

import argparse
import copy
import json
from collections import OrderedDict
from pathlib import Path

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
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--position-noise-std", type=float, default=0.0)
    parser.add_argument("--heading-noise-std-deg", type=float, default=0.0)
    parser.add_argument("--time-delay-ms", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=20)
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
    parser.add_argument("--global-sort-detections", action="store_true")
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
        args.message_content_ablation != "none"
        and args.trackformer_protocol != "propagated"
    ):
        raise ValueError(
            "--message-content-ablation requires propagated PQ-STF"
        )
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
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=opencood_dataset.collate_batch_test,
        pin_memory=args.device == "cuda",
        persistent_workers=args.workers > 0,
    )
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

    naive_statistics = _statistics()
    ego_only_statistics = _statistics()
    belt_statistics = _statistics()
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
    progress = tqdm(
        loader,
        desc=(
            "evaluating cosine association + plain box fusion"
            if args.trackformer_box_fusion in {"score-weighted", "max-score"}
            else "evaluating BELT fusion"
        ),
    )
    with torch.no_grad():
        for batch_index, batch in enumerate(progress):
            if args.max_frames is not None and batch_index >= args.max_frames:
                break
            batch = to_device(batch, device)
            outputs = OrderedDict(
                (cav_id, model(cav_content))
                for cav_id, cav_content in batch.items()
            )
            naive_boxes, naive_scores, ground_truth = opencood_dataset.post_process(
                batch, outputs
            )
            _evaluate_prediction(
                naive_boxes,
                naive_scores,
                ground_truth,
                naive_statistics,
            )

            detections = []
            detections_by_cav = {}
            simple_detections_by_cav = {}
            trackformer_agents = {}
            for cav_id, cav_content in batch.items():
                proposal = decode_local_proposals(
                    outputs[cav_id], cav_content, opencood_dataset.post_processor
                )
                if trackformer is not None:
                    trackformer_agents[cav_id] = _trackformer_agent(
                        proposal, cav_content, outputs[cav_id], hypes
                    )
                detection = proposal_uncertainty(
                    outputs[cav_id],
                    proposal,
                    cav_content,
                    position_noise_std=(
                        0.0 if cav_id == "ego" else args.position_noise_std
                    ),
                    heading_noise_std_deg=(
                        0.0
                        if cav_id == "ego"
                        else args.heading_noise_std_deg
                    ),
                )
                detection["agent_id"] = cav_id
                detections.append(detection)
                detections_by_cav[cav_id] = detection
                simple_detections_by_cav[cav_id] = _simple_detection(
                    proposal, cav_content
                )
                simple_detections_by_cav[cav_id]["agent_id"] = cav_id
                if args.oracle_grouping:
                    simple_detections_by_cav[cav_id]["oracle_object_ids"] = (
                        _oracle_object_ids(
                            proposal,
                            cav_content,
                            opencood_dataset.post_processor,
                        )
                    )
            ego_simple_boxes, ego_simple_scores = _postprocess_fused_boxes(
                simple_detections_by_cav["ego"]["boxes"],
                simple_detections_by_cav["ego"]["scores"],
                opencood_dataset.post_processor,
            )
            _evaluate_prediction(
                ego_simple_boxes,
                ego_simple_scores,
                ground_truth,
                ego_only_statistics,
            )
            fused = fuse_detections(
                detections, maximum_distance=args.association_distance
            )
            belt_boxes, belt_scores = _postprocess_fused_boxes(
                fused["boxes"],
                fused["scores"],
                opencood_dataset.post_processor,
            )
            _evaluate_prediction(
                belt_boxes, belt_scores, ground_truth, belt_statistics
            )
            if args.oracle_grouping:
                oracle_groups = _oracle_groups(simple_detections_by_cav.values())
                oracle = fuse_groups_score_weighted(oracle_groups)
                oracle_boxes, oracle_scores = _postprocess_fused_boxes(
                    oracle["boxes"],
                    oracle["scores"],
                    opencood_dataset.post_processor,
                )
                _evaluate_prediction(
                    oracle_boxes,
                    oracle_scores,
                    ground_truth,
                    oracle_statistics,
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
                    for cav_id, _ in batch.items():
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
                    for cav_id, _ in batch.items():
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
                            minimum_probability=args.pair_head_min_probability,
                        ) if direct_detections else []
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
                for cav_id, _ in batch.items():
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
        },
        "association_distance_m": args.association_distance,
        "embedding_distance_m": args.embedding_distance,
        "embedding_min_similarity": args.embedding_min_similarity,
        "message_content_ablation": args.message_content_ablation,
        "track_query_score_threshold": args.track_query_score_threshold,
        "ransac_localization": {
            "enabled": args.ransac_localization_correction,
            "seed_distance_m": args.ransac_seed_distance,
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
        "belt_geometry": _ap(belt_statistics, args.global_sort_detections),
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
