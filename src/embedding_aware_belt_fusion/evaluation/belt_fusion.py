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
from embedding_aware_belt_fusion.integration.belt_fusion import (
    associate_ego_with_propagated_sources,
    fuse_detections,
    fuse_groups,
    proposal_uncertainty,
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
        "--trackformer-root", type=Path, default=Path("external/trackformer")
    )
    parser.add_argument("--embedding-distance", type=float, default=8.0)
    parser.add_argument("--embedding-min-similarity", type=float, default=0.5)
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


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
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

    naive_statistics = _statistics()
    belt_statistics = _statistics()
    embedding_statistics = _statistics() if trackformer is not None else None
    frame_count = candidate_count = fused_count = matched_groups = 0
    embedding_fused_count = embedding_matched_groups = 0
    track_query_count = selected_track_query_count = 0
    track_query_score_sum = 0.0
    track_query_score_min = float("inf")
    track_query_score_max = float("-inf")
    progress = tqdm(loader, desc="evaluating BELT fusion")
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
            if trackformer is not None:
                ego_detection = detections_by_cav.get("ego")
                propagated_sources = []
                unselected_sources = []
                for cav_id, _ in batch.items():
                    if cav_id == "ego":
                        continue
                    detection = detections_by_cav[cav_id]
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
                    target_output = trackformer.forward_agent(
                        ego_agent,
                        track_states=source_output["hs_embed"][0],
                        track_reference_boxes=source_output["pred_boxes"][0],
                        track_reference_geometry=source_output["query_geometry"],
                    )
                    source = _select_detection(detection, selected)
                    source["embeddings"] = source_output["embeddings"][0]
                    source["ego_embeddings"] = target_output["embeddings"][0][
                        target_output["track_count"] :
                    ]
                    propagated_sources.append(source)
                if ego_detection is None:
                    raise RuntimeError("Late-fusion batch has no ego agent")
                if propagated_sources:
                    groups = associate_ego_with_propagated_sources(
                        ego_detection,
                        propagated_sources,
                        maximum_distance=args.embedding_distance,
                        minimum_similarity=args.embedding_min_similarity,
                    )
                else:
                    groups = singleton_groups(ego_detection)
                for source in unselected_sources:
                    groups.extend(singleton_groups(source))
                embedded = fuse_groups(groups)
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
                "belt_trackformer": _ap(
                    embedding_statistics, args.global_sort_detections
                ),
            }
            if embedding_statistics is not None
            else {}
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("BELT " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
