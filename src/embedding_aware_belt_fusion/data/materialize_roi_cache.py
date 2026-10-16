"""Materialize a compact all-proposal ROI view from a complete schema-v3 cache."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
from pathlib import Path
from time import perf_counter

import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-shards-per-output", type=int, default=20)
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="CPU threads that compact agents already loaded into RAM",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Intra-op threads per compacting operation",
    )
    parser.add_argument(
        "--prefetch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read the next shard while CPUs compact the current shard",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _relative_positions(indices, width):
    rows = torch.div(indices, width, rounding_mode="floor").float()
    columns = (indices % width).float()
    coordinates = torch.stack([rows, columns], dim=-1)
    coordinates -= coordinates.mean(dim=0, keepdim=True)
    scale = coordinates.abs().amax(dim=0, keepdim=True).clamp_min(1.0)
    return coordinates / scale


def _compact_agent(agent, width):
    bev = agent["bev_features"].float()
    flattened = bev.flatten(1).transpose(0, 1)
    cell_indices = [
        item.long() for item in agent["proposal_roi_cell_indices"]
    ]
    proposal_count = len(cell_indices)
    maximum = max((len(item) for item in cell_indices), default=0)
    tokens = torch.zeros(
        proposal_count,
        maximum,
        flattened.shape[-1],
        dtype=torch.float32,
    )
    positions = torch.zeros(proposal_count, maximum, 2)
    mask = torch.ones(proposal_count, maximum, dtype=torch.bool)
    for proposal_index, indices in enumerate(cell_indices):
        length = len(indices)
        tokens[proposal_index, :length] = flattened[indices]
        positions[proposal_index, :length] = _relative_positions(
            indices, width
        )
        mask[proposal_index, :length] = False
    return {
        "agent_id": str(agent["agent_id"]),
        "tokens": tokens,
        "positions": positions,
        "mask": mask,
        "boxes": agent["boxes"].float(),
        "scores": agent["scores"].float(),
        "gt_ids": agent["gt_ids"],
        "gt_ious": agent["gt_ious"].float(),
        "gt_boxes": agent["gt_boxes"].float(),
        "anchor_indices": agent["anchor_indices"].long(),
        "transformation_matrix": agent["transformation_matrix"].float(),
    }


def _load_full_shard(path):
    # Eager loading creates one sequential HDD stream. Sparse mmap page faults
    # are substantially slower on a rotational disk for this conversion.
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=False,
    )


def _compact_shard(frames, width, executor):
    counts = [len(frame["agents"]) for frame in frames]
    agents = [
        agent for frame in frames for agent in frame["agents"]
    ]
    compacted = list(
        executor.map(partial(_compact_agent, width=width), agents)
    )
    output = []
    offset = 0
    for frame, count in zip(frames, counts):
        output.append(
            {
                "scenario_id": str(frame["scenario_id"]),
                "frame_id": str(frame["frame_id"]),
                "agents": compacted[offset : offset + count],
            }
        )
        offset += count
    return output


def _write_state(path, state):
    path.write_text(json.dumps(state, indent=2) + "\n")


def main():
    args = parse_args()
    if args.source_shards_per_output < 1:
        raise ValueError("--source-shards-per-output must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    source_manifest = json.loads(
        (args.source_cache / "manifest.json").read_text()
    )
    if source_manifest.get("cache_mode") != "complete":
        raise ValueError("Source cache must use cache_mode=complete")
    if int(source_manifest.get("schema_version", 0)) < 3:
        raise ValueError("Source cache schema 3 or newer is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.json"
    if progress_path.is_file():
        if not args.resume:
            raise FileExistsError(
                f"{progress_path} already exists; pass --resume"
            )
        state = json.loads(progress_path.read_text())
    else:
        state = {
            "schema_version": 1,
            "source_cache": str(args.source_cache.resolve()),
            "source_shards_per_output": args.source_shards_per_output,
            "processed_source_shards": 0,
            "frame_count": 0,
            "agent_count": 0,
            "proposal_count": 0,
            "shards": [],
            "frames": [],
        }
        _write_state(progress_path, state)
    if state["source_cache"] != str(args.source_cache.resolve()):
        raise ValueError("Resume state points to a different source cache")
    if (
        state["source_shards_per_output"]
        != args.source_shards_per_output
    ):
        raise ValueError(
            "Resume requires the original --source-shards-per-output"
        )

    _, _, width = source_manifest["bev_shape"]
    source_shards = source_manifest["shards"]
    start = int(state["processed_source_shards"])
    groups = range(
        start,
        len(source_shards),
        args.source_shards_per_output,
    )
    progress = tqdm(
        groups,
        total=(
            len(source_shards) - start
            + args.source_shards_per_output
            - 1
        )
        // args.source_shards_per_output,
        desc="materializing compact ROI cache",
    )
    with (
        ThreadPoolExecutor(max_workers=args.workers) as compact_executor,
        ThreadPoolExecutor(max_workers=1) as read_executor,
    ):
        for group_start in progress:
            group_started = perf_counter()
            group_end = min(
                group_start + args.source_shards_per_output,
                len(source_shards),
            )
            compact_frames = []

            def submit_read(source_shard_index):
                source_path = (
                    args.source_cache
                    / source_shards[source_shard_index]["path"]
                )
                return read_executor.submit(_load_full_shard, source_path)

            next_read = submit_read(group_start)
            for source_shard_index in range(group_start, group_end):
                frames = next_read.result()
                if args.prefetch and source_shard_index + 1 < group_end:
                    next_read = submit_read(source_shard_index + 1)
                compact_frames.extend(
                    _compact_shard(
                        frames, width, compact_executor
                    )
                )
                if (
                    not args.prefetch
                    and source_shard_index + 1 < group_end
                ):
                    next_read = submit_read(source_shard_index + 1)
                del frames
            output_index = len(state["shards"])
            output_name = f"roi_{output_index:05d}.pt"
            torch.save(compact_frames, args.output_dir / output_name)
            for frame_index, frame in enumerate(compact_frames):
                state["frames"].append(
                    {
                        "scenario_id": frame["scenario_id"],
                        "frame_id": frame["frame_id"],
                        "shard_index": output_index,
                        "frame_index": frame_index,
                    }
                )
            agent_count = sum(
                len(frame["agents"]) for frame in compact_frames
            )
            proposal_count = sum(
                len(agent["gt_ids"])
                for frame in compact_frames
                for agent in frame["agents"]
            )
            state["frame_count"] += len(compact_frames)
            state["agent_count"] += agent_count
            state["proposal_count"] += proposal_count
            state["processed_source_shards"] = group_end
            state["shards"].append(
                {
                    "path": output_name,
                    "frames": len(compact_frames),
                    "source_shard_start": group_start,
                    "source_shard_end": group_end,
                }
            )
            _write_state(progress_path, state)
            elapsed = perf_counter() - group_started
            progress.set_postfix(
                frames=state["frame_count"],
                fps=f"{len(compact_frames) / elapsed:.2f}",
                proposals=state["proposal_count"],
            )

    if state["frame_count"] != source_manifest["frame_count"]:
        raise RuntimeError(
            "Compact cache frame count does not match the source cache"
        )
    manifest = {
        "schema_version": 1,
        "cache_mode": "complete_roi_training",
        "complete": True,
        "source_cache": state["source_cache"],
        "source_schema_version": source_manifest["schema_version"],
        "split": source_manifest["split"],
        "feature_dim": source_manifest["feature_dim"],
        "bev_shape": source_manifest["bev_shape"],
        "frame_count": state["frame_count"],
        "agent_count": state["agent_count"],
        "proposal_count": state["proposal_count"],
        "minimum_gt_iou": source_manifest["minimum_gt_iou"],
        "fields": {
            "tokens": "every native BEV cell covered by every proposal",
            "positions": "relative layout of every covered BEV cell",
            "mask": "padding mask; false entries are real covered cells",
            "boxes": "all decoded proposals [x,y,z,h,w,l,yaw]",
            "scores": "all proposal confidence scores",
            "gt_ids": "matched object identity or null for every proposal",
            "gt_ious": "proposal-to-ground-truth rotated BEV IoU",
            "gt_boxes": "matched ground-truth boxes",
            "anchor_indices": "dense PointPillars indices retained through NMS",
            "transformation_matrix": "agent-local LiDAR coordinates to ego LiDAR coordinates",
        },
        "shards": state["shards"],
    }
    _write_state(args.output_dir / "manifest.json", manifest)
    scenario_index = {
        "schema_version": 1,
        "cache_schema_version": manifest["schema_version"],
        "shards": [item["path"] for item in manifest["shards"]],
        "frames": state["frames"],
    }
    _write_state(
        args.output_dir / "scenario_index.json", scenario_index
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
