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
    decode_local_proposals,
    proposal_roi_cell_indices,
)
from embedding_aware_belt_fusion.embeddings.spatial_trackformer import (
    SpatialTrackFormer,
)
from embedding_aware_belt_fusion.embeddings.geometry import transform_boxes_to_ego
from embedding_aware_belt_fusion.communication.product_codebook import (
    load_product_codebook,
    load_residual_product_codebook,
)
from embedding_aware_belt_fusion.integration.belt_fusion import (
    associate_by_embedding,
    associate_ego_with_propagated_sources,
    associate_ego_with_propagated_sources_simple,
    fuse_detections,
    fuse_groups,
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
        "--trackformer-box-fusion",
        choices=("belt", "score-weighted"),
        default="belt",
        help=(
            "Use BELT uncertainty fusion (belt), or plain detector-score "
            "weighted box averaging after cosine/Hungarian association."
        ),
    )
    parser.add_argument(
        "--track-query-score-threshold", type=float, default=0.5
    )
    parser.add_argument("--global-sort-detections", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def _statistics():
    return {
        threshold: {"tp": [], "fp": [], "gt": 0, "score": []}
        for threshold in (0.3, 0.5, 0.7)
    }


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
        args.trackformer_box_fusion == "score-weighted"
        and args.trackformer_protocol != "propagated"
    ):
        raise ValueError(
            "--trackformer-box-fusion score-weighted currently supports "
            "the propagated PQ-STF protocol only"
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
    message_codebook, message_metadata = _load_message_codebook(
        args, trackformer, device
    )

    naive_statistics = _statistics()
    belt_statistics = _statistics()
    embedding_statistics = _statistics() if trackformer is not None else None
    frame_count = candidate_count = fused_count = matched_groups = 0
    embedding_fused_count = embedding_matched_groups = 0
    trackformer_roi_fallback_frames = 0
    track_query_count = selected_track_query_count = 0
    track_query_score_sum = 0.0
    track_query_score_min = float("inf")
    track_query_score_max = float("-inf")
    message_code_count = 0
    transmitted_message_count = 0
    transmitted_message_bits = 0
    progress = tqdm(
        loader,
        desc=(
            "evaluating cosine association + score-weighted fusion"
            if args.trackformer_box_fusion == "score-weighted"
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
                detections.append(detection)
                detections_by_cav[cav_id] = detection
                simple_detections_by_cav[cav_id] = _simple_detection(
                    proposal, cav_content
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
                if args.trackformer_box_fusion == "score-weighted":
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
                        if args.trackformer_box_fusion == "score-weighted"
                        else associate_ego_with_propagated_sources
                    )
                    groups = association(
                        ego_detection,
                        propagated_sources,
                        maximum_distance=args.embedding_distance,
                        minimum_similarity=args.embedding_min_similarity,
                    )
                else:
                    groups = (
                        simple_singleton_groups(ego_detection)
                        if args.trackformer_box_fusion == "score-weighted"
                        else singleton_groups(ego_detection)
                    )
                for source in unselected_sources:
                    groups.extend(
                        simple_singleton_groups(source)
                        if args.trackformer_box_fusion == "score-weighted"
                        else singleton_groups(source)
                    )
                embedded = (
                    fuse_groups_score_weighted(groups)
                    if args.trackformer_box_fusion == "score-weighted"
                    else fuse_groups(groups)
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
        "track_query_score_threshold": args.track_query_score_threshold,
        "trackformer_box_fusion": (
            args.trackformer_box_fusion if trackformer is not None else None
        ),
        "trackformer_protocol": (
            args.trackformer_protocol if trackformer is not None else None
        ),
        "association_message": (
            {
                "codebook": (
                    str((args.message_codebook or args.residual_message_codebook).resolve())
                    if (args.message_codebook or args.residual_message_codebook) is not None
                    else None
                ),
                "representation": (
                    message_metadata["representation"]
                    if message_metadata is not None
                    else (
                        "query-state"
                        if args.trackformer_protocol == "propagated"
                        else "embedding"
                    )
                ),
                "bits_per_source_object": (
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
        "belt_geometry": _ap(belt_statistics, args.global_sort_detections),
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
                    track_query_count / max(frame_count, 1)
                ),
                "mean_selected_track_queries_per_frame": (
                    selected_track_query_count / max(frame_count, 1)
                ),
                "mean_trackformer_source_object_score": (
                    track_query_score_sum / max(track_query_count, 1)
                ),
                "min_trackformer_source_object_score": track_query_score_min,
                "max_trackformer_source_object_score": track_query_score_max,
                **{
                    (
                        "simple_trackformer"
                        if args.trackformer_box_fusion == "score-weighted"
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
        if args.trackformer_box_fusion == "score-weighted"
        else "BELT"
    )
    print(label + " " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
