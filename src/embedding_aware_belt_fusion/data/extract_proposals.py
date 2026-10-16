"""Cache multi-agent OpenCOOD proposals and pre-head BEV features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from embedding_aware_belt_fusion.features import (
    assign_proposals_to_ground_truth,
    decode_local_proposals,
    pointpillar_forward_with_features,
    proposal_roi_cell_indices,
)
from embedding_aware_belt_fusion.integration.train_uncertainty import (
    _load_detector_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opencood-config", type=Path, required=True)
    parser.add_argument("--detector-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--minimum-gt-iou", type=float, default=0.5)
    parser.add_argument("--shard-frames", type=int, default=100)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--position-noise-std", type=float, default=0.0)
    parser.add_argument(
        "--heading-noise-std-deg", type=float, default=0.0
    )
    parser.add_argument("--time-delay-ms", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=20)
    parser.add_argument(
        "--scenario-split-file",
        type=Path,
        help="Scenario split JSON produced by TrackFormer training",
    )
    parser.add_argument(
        "--scenario-role",
        choices=("train", "validation"),
        help="Select scenarios from --scenario-split-file",
    )
    parser.add_argument("--bev-height", type=int)
    parser.add_argument("--bev-width", type=int)
    parser.add_argument(
        "--cache-mode",
        choices=(
            "proposals",
            "pooled",
            "complete",
            "complete_roi_training",
        ),
        default="proposals",
        help=(
            "complete stores the full unpooled BEV map, dense heads, ROI-cell "
            "indices, transforms, and source references; "
            "complete_roi_training directly stores only compact ROI tokens"
        ),
    )
    parser.add_argument(
        "--feature-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


class _IndexedDataset(Dataset):
    """Load selected dataset frames in workers while retaining their indices."""

    def __init__(self, dataset, frame_indices):
        self.dataset = dataset
        self.frame_indices = frame_indices

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, position):
        frame_index = self.frame_indices[position]
        return frame_index, self.dataset[frame_index]


def _collate_indexed(batch):
    # OpenCOOD late-fusion testing only supports one frame per batch.
    assert len(batch) == 1
    return batch[0]


def _load_existing_shards(output_dir: Path, split: str):
    """Return valid shards and their frame, agent, and proposal counts."""
    paths = sorted(output_dir.glob(f"{split}_*.pt"))
    shards = []
    completed_frames = 0
    completed_agents = 0
    completed_proposals = 0
    for expected_index, path in enumerate(paths):
        expected_name = f"{split}_{expected_index:05d}.pt"
        if path.name != expected_name:
            raise RuntimeError(
                f"Cannot resume: expected {expected_name}, found {path.name}"
            )
        try:
            records = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            raise RuntimeError(
                f"Cannot resume from unreadable shard {path}; move that shard "
                "out of the output directory and retry"
            ) from error
        if not isinstance(records, list) or not records:
            raise RuntimeError(f"Cannot resume from empty or invalid shard {path}")
        shards.append({"path": path.name, "frames": len(records)})
        completed_frames += len(records)
        completed_agents += sum(len(frame["agents"]) for frame in records)
        completed_proposals += sum(
            len(agent["gt_ids"])
            for frame in records
            for agent in frame["agents"]
        )
    return (
        shards,
        completed_frames,
        completed_agents,
        completed_proposals,
    )


def _frame_metadata(dataset, index: int):
    scenario_index = next(i for i, end in enumerate(dataset.len_record) if index < end)
    previous_end = 0 if scenario_index == 0 else dataset.len_record[scenario_index - 1]
    frame_offset = index - previous_end
    scenario = dataset.scenario_database[scenario_index]
    timestamp = list(next(iter(scenario.values())).keys())[frame_offset]
    ego_id = next(cav_id for cav_id, content in scenario.items() if content["ego"])
    scenario_path = Path(next(iter(scenario.values()))[timestamp]["yaml"]).parents[1]
    return str(scenario_path.name), str(timestamp), str(ego_id)


def main() -> None:
    args = parse_args()
    if (args.bev_height is None) != (args.bev_width is None):
        raise ValueError("--bev-height and --bev-width must be provided together")
    if args.cache_mode == "pooled" and args.bev_height is None:
        raise ValueError("pooled mode requires --bev-height and --bev-width")
    if (
        args.cache_mode in {"complete", "complete_roi_training"}
        and args.bev_height is not None
    ):
        raise ValueError(
            f"{args.cache_mode} preserves native resolution; omit BEV size"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)

    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.tools.train_utils import create_model, to_device

    hypes = load_yaml(str(args.opencood_config), None)
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
    dataset = build_dataset(hypes, visualize=False, train=False)
    detector = create_model(hypes)
    detector_checkpoint = _load_detector_checkpoint(args.detector_dir, detector)
    detector.to(device).eval().requires_grad_(False)
    feature_dtype = (
        torch.float16 if args.feature_dtype == "float16" else torch.float32
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard = []
    frame_indices = list(range(0, len(dataset), args.frame_stride))
    selected_scenarios = None
    if (args.scenario_split_file is None) != (args.scenario_role is None):
        raise ValueError(
            "--scenario-split-file and --scenario-role must be used together"
        )
    if args.scenario_split_file is not None:
        scenario_split = json.loads(args.scenario_split_file.read_text())
        selected_scenarios = set(
            scenario_split[f"{args.scenario_role}_scenarios"]
        )
        frame_indices = [
            index
            for index in frame_indices
            if _frame_metadata(dataset, index)[0] in selected_scenarios
        ]
    if args.max_frames is not None:
        frame_indices = frame_indices[: args.max_frames]
    requested_frame_count = len(frame_indices)
    if args.resume:
        (
            shards,
            completed_frames,
            agent_count,
            proposal_count,
        ) = _load_existing_shards(args.output_dir, args.split)
        if completed_frames > len(frame_indices):
            raise RuntimeError(
                "Existing shards contain more frames than the requested extraction"
            )
        frame_indices = frame_indices[completed_frames:]
        print(
            f"Resuming after {completed_frames} completed frames "
            f"from {len(shards)} shards."
        )
    else:
        shards = []
        agent_count = 0
        proposal_count = 0

    indexed_dataset = _IndexedDataset(dataset, frame_indices)
    loader = DataLoader(
        indexed_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=_collate_indexed,
        pin_memory=args.device == "cuda",
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    native_bev_shape = None
    if args.cache_mode == "complete" and shards:
        existing = torch.load(
            args.output_dir / shards[0]["path"],
            map_location="cpu",
            weights_only=False,
        )
        native_bev_shape = list(
            existing[0]["agents"][0]["bev_features"].shape
        )
    with torch.no_grad():
        for extracted_index, (frame_index, raw_frame) in enumerate(
            tqdm(loader, desc="extracting proposals", total=len(frame_indices))
        ):
            scenario_id, frame_id, ego_id = _frame_metadata(dataset, frame_index)
            batch = dataset.collate_batch_test([raw_frame])
            batch = to_device(batch, device)
            frame_record = {
                "scenario_id": scenario_id,
                "frame_id": frame_id,
                "agents": [],
            }
            for batch_agent_id, cav_content in batch.items():
                output = pointpillar_forward_with_features(detector, cav_content)
                proposal = decode_local_proposals(
                    output, cav_content, dataset.post_processor
                )
                gt_mask = cav_content["object_bbx_mask"][0].bool()
                gt_boxes = cav_content["object_bbx_center"][0][gt_mask]
                gt_ids = cav_content["object_ids"]
                assignment = assign_proposals_to_ground_truth(
                    proposal["corners"],
                    gt_boxes,
                    gt_ids,
                    order=dataset.post_processor.params["order"],
                    minimum_iou=args.minimum_gt_iou,
                )
                matched_gt_boxes = torch.zeros_like(proposal["boxes"])
                matched = assignment["gt_indices"] >= 0
                if matched.any():
                    proposal_rows = torch.nonzero(matched).squeeze(1)
                    gt_rows = assignment["gt_indices"][proposal_rows]
                    matched_gt_boxes[
                        proposal_rows.to(matched_gt_boxes.device)
                    ] = gt_boxes[gt_rows.to(gt_boxes.device)].to(
                        device=matched_gt_boxes.device,
                        dtype=matched_gt_boxes.dtype,
                    )
                actual_agent_id = ego_id if batch_agent_id == "ego" else str(batch_agent_id)
                full_bev = output["spatial_features_2d"][0]
                native_bev_shape = list(full_bev.shape)
                if args.cache_mode in {
                    "complete",
                    "complete_roi_training",
                }:
                    roi_indices = proposal_roi_cell_indices(
                        proposal["boxes"],
                        height=full_bev.shape[-2],
                        width=full_bev.shape[-1],
                        lidar_range=hypes["model"]["args"]["lidar_range"],
                    )
                else:
                    roi_indices = None
                compact_roi = {}
                if args.cache_mode == "complete_roi_training":
                    flattened = full_bev.flatten(1).transpose(0, 1)
                    maximum = max(
                        (len(item) for item in roi_indices), default=0
                    )
                    count = len(roi_indices)
                    tokens = torch.zeros(
                        count,
                        maximum,
                        flattened.shape[-1],
                        device=flattened.device,
                        dtype=flattened.dtype,
                    )
                    positions = torch.zeros(
                        count, maximum, 2, device=flattened.device
                    )
                    padding_mask = torch.ones(
                        count,
                        maximum,
                        dtype=torch.bool,
                        device=flattened.device,
                    )
                    for proposal_index, indices in enumerate(roi_indices):
                        indices = indices.long()
                        length = len(indices)
                        tokens[proposal_index, :length] = flattened[indices]
                        rows = torch.div(
                            indices,
                            full_bev.shape[-1],
                            rounding_mode="floor",
                        ).float()
                        columns = (indices % full_bev.shape[-1]).float()
                        coordinates = torch.stack(
                            [rows, columns], dim=-1
                        )
                        coordinates -= coordinates.mean(
                            dim=0, keepdim=True
                        )
                        scale = coordinates.abs().amax(
                            dim=0, keepdim=True
                        ).clamp_min(1.0)
                        positions[proposal_index, :length] = (
                            coordinates / scale
                        )
                        padding_mask[proposal_index, :length] = False
                    compact_roi = {
                        "tokens": tokens.cpu().to(feature_dtype),
                        "positions": positions.cpu().float(),
                        "mask": padding_mask.cpu(),
                    }
                agent_record = {
                    "agent_id": actual_agent_id,
                    "boxes": proposal["boxes"].cpu().float(),
                    "scores": proposal["scores"].cpu().float(),
                    "features": proposal["features"].cpu().to(feature_dtype),
                    "anchor_indices": proposal["anchor_indices"].cpu(),
                    "gt_ids": assignment["gt_ids"],
                    "gt_ious": assignment["gt_ious"],
                    "gt_boxes": matched_gt_boxes.cpu().float(),
                    **compact_roi,
                    **(
                        {
                            "bev_features": F.adaptive_avg_pool2d(
                                output["spatial_features_2d"],
                                (args.bev_height, args.bev_width),
                            )[0]
                            .cpu()
                            .to(feature_dtype)
                        }
                        if args.cache_mode == "pooled"
                        else {}
                    ),
                }
                if args.cache_mode in {"complete", "complete_roi_training"}:
                    agent_record["transformation_matrix"] = cav_content[
                        "transformation_matrix"
                    ].cpu().float()
                if args.cache_mode == "complete":
                    source_base = (
                        args.data_root
                        / args.split
                        / scenario_id
                        / actual_agent_id
                        / frame_id
                    )
                    agent_record.update(
                        {
                            "bev_features": full_bev.cpu().to(feature_dtype),
                            "classification_map": output["psm"][0]
                            .cpu()
                            .to(feature_dtype),
                            "regression_map": output["rm"][0]
                            .cpu()
                            .to(feature_dtype),
                            "proposal_roi_cell_indices": roi_indices,
                            "source_yaml": str(
                                source_base.with_suffix(".yaml")
                            ),
                            "source_lidar": str(
                                source_base.with_suffix(".pcd")
                            ),
                        }
                    )
                frame_record["agents"].append(agent_record)
                agent_count += 1
                proposal_count += len(assignment["gt_ids"])
            shard.append(frame_record)
            if (
                len(shard) == args.shard_frames
                or extracted_index + 1 == len(frame_indices)
            ):
                shard_name = f"{args.split}_{len(shards):05d}.pt"
                torch.save(shard, args.output_dir / shard_name)
                shards.append({"path": shard_name, "frames": len(shard)})
                shard = []

    manifest = {
        "schema_version": (
            3
            if args.cache_mode == "complete"
            else (1 if args.cache_mode == "complete_roi_training" else 2)
        ),
        "split": args.split,
        "frame_count": requested_frame_count,
        "agent_count": agent_count,
        "proposal_count": proposal_count,
        "frame_stride": args.frame_stride,
        "feature_dim": int(detector.backbone.num_bev_features),
        "minimum_gt_iou": args.minimum_gt_iou,
        "detector_checkpoint": str(detector_checkpoint.resolve()),
        "opencood_config": str(args.opencood_config.resolve()),
        "noise_disabled": not (
            args.position_noise_std
            or args.heading_noise_std_deg
            or args.time_delay_ms
        ),
        "noise": {
            "position_std_m": args.position_noise_std,
            "heading_std_deg": args.heading_noise_std_deg,
            "time_delay_ms": args.time_delay_ms,
            "seed": args.noise_seed,
            "implementation": "OpenCOOD BaseDataset wild_setting",
        },
        "scenario_subset": (
            {
                "file": str(args.scenario_split_file.resolve()),
                "role": args.scenario_role,
                "scenario_count": len(selected_scenarios),
            }
            if selected_scenarios is not None
            else None
        ),
        "cache_mode": args.cache_mode,
        "complete": args.cache_mode == "complete_roi_training",
        "feature_dtype": args.feature_dtype,
        "bev_shape": (
            [int(detector.backbone.num_bev_features), args.bev_height, args.bev_width]
            if args.cache_mode == "pooled"
            else (
                native_bev_shape
                if args.cache_mode in {
                    "complete",
                    "complete_roi_training",
                }
                else None
            )
        ),
        "lidar_range": list(hypes["model"]["args"]["lidar_range"]),
        "voxel_size": list(hypes["preprocess"]["args"]["voxel_size"]),
        "box_order": dataset.post_processor.params["order"],
        "roi_index_layout": "row_major_flat_index = row * bev_width + column",
        "fields": {
            "scenario_id": "OPV2V scenario directory name",
            "frame_id": "same-timestamp OPV2V frame key",
            "agent_id": "physical CAV directory identifier",
            "boxes": "decoded local PointPillars proposals [x,y,z,h,w,l,yaw]",
            "scores": "sigmoid vehicle confidence for each proposal",
            "features": "384-D native BEV cell feature selected by anchor index",
            "anchor_indices": "dense prediction indices retained through NMS",
            "gt_ids": "matched physical OPV2V object ID or null",
            "gt_ious": "rotated-BEV proposal-to-GT IoU",
            "gt_boxes": "matched local GT box; zero row for unmatched proposal",
            **(
                {
                    "bev_features": (
                        "complete native unpooled PointPillars BEV tensor"
                    ),
                    "classification_map": (
                        "dense pre-sigmoid PointPillars class logits"
                    ),
                    "regression_map": (
                        "dense PointPillars anchor box deltas"
                    ),
                    "proposal_roi_cell_indices": (
                        "all native BEV cells covered by each rotated "
                        "proposal footprint"
                    ),
                    "source_yaml": "original OPV2V annotation path",
                    "source_lidar": "original OPV2V point-cloud path",
                }
                if args.cache_mode == "complete"
                else {}
            ),
            "transformation_matrix": "this agent's local coordinates to ego",
            "tokens": "every native BEV cell covered by every proposal",
            "positions": "relative layout of every covered BEV cell",
            "mask": "padding mask; false entries are real covered cells",
        },
        "shards": shards,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
