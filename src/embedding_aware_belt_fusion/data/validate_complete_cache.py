"""Validate and inventory a complete PointPillars research cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import torch
from tqdm import tqdm


REQUIRED_AGENT_FIELDS = {
    "agent_id",
    "boxes",
    "scores",
    "features",
    "anchor_indices",
    "gt_ids",
    "gt_ious",
    "gt_boxes",
    "bev_features",
    "classification_map",
    "regression_map",
    "proposal_roi_cell_indices",
    "transformation_matrix",
    "source_yaml",
    "source_lidar",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--max-shards", type=int)
    args = parser.parse_args()
    manifest_path = args.cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 3:
        raise ValueError("Expected complete-cache schema_version=3")
    if manifest.get("cache_mode") != "complete":
        raise ValueError("Expected cache_mode=complete")

    shard_items = manifest["shards"]
    if args.max_shards is not None:
        shard_items = shard_items[: args.max_shards]
    frame_count = agent_count = proposal_count = matched_count = 0
    roi_counts, shard_bytes = [], 0
    for item in tqdm(shard_items, desc="validating complete cache"):
        path = args.cache_dir / item["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        shard_bytes += path.stat().st_size
        frames = torch.load(path, map_location="cpu", weights_only=False)
        if len(frames) != item["frames"]:
            raise RuntimeError(f"Frame count mismatch in {path}")
        for frame in frames:
            frame_count += 1
            if not {"scenario_id", "frame_id", "agents"} <= set(frame):
                raise RuntimeError(f"Invalid frame record in {path}")
            for agent in frame["agents"]:
                missing = REQUIRED_AGENT_FIELDS - set(agent)
                if missing:
                    raise RuntimeError(f"{path} agent missing {sorted(missing)}")
                agent_count += 1
                count = len(agent["boxes"])
                proposal_count += count
                aligned = (
                    len(agent["scores"]),
                    len(agent["features"]),
                    len(agent["anchor_indices"]),
                    len(agent["gt_ids"]),
                    len(agent["gt_ious"]),
                    len(agent["gt_boxes"]),
                    len(agent["proposal_roi_cell_indices"]),
                )
                if any(value != count for value in aligned):
                    raise RuntimeError(f"Proposal field misalignment in {path}")
                if list(agent["bev_features"].shape) != manifest["bev_shape"]:
                    raise RuntimeError(f"BEV shape mismatch in {path}")
                cells = agent["bev_features"].shape[-2] * agent["bev_features"].shape[-1]
                for proposal_index, indices in enumerate(
                    agent["proposal_roi_cell_indices"]
                ):
                    if not len(indices) or indices.min() < 0 or indices.max() >= cells:
                        raise RuntimeError(
                            f"Invalid ROI indices in {path}, proposal {proposal_index}"
                        )
                    roi_counts.append(len(indices))
                for proposal_index, object_id in enumerate(agent["gt_ids"]):
                    if object_id is not None:
                        matched_count += 1
                        if agent["gt_boxes"][proposal_index].abs().sum() == 0:
                            raise RuntimeError(f"Matched proposal has zero GT in {path}")
                if tuple(agent["transformation_matrix"].shape) != (4, 4):
                    raise RuntimeError(f"Invalid transformation in {path}")
                if not Path(agent["source_yaml"]).is_file():
                    raise FileNotFoundError(agent["source_yaml"])
                if not Path(agent["source_lidar"]).is_file():
                    raise FileNotFoundError(agent["source_lidar"])

    inventory = {
        "validated": True,
        "schema_version": manifest["schema_version"],
        "cache_mode": manifest["cache_mode"],
        "validated_shards": len(shard_items),
        "frames": frame_count,
        "agents": agent_count,
        "proposals": proposal_count,
        "matched_proposals": matched_count,
        "bev_shape": manifest["bev_shape"],
        "feature_dtype": manifest["feature_dtype"],
        "mean_roi_cells_per_proposal": mean(roi_counts),
        "min_roi_cells_per_proposal": min(roi_counts),
        "max_roi_cells_per_proposal": max(roi_counts),
        "validated_shard_bytes": shard_bytes,
        "validated_shard_gib": shard_bytes / (1024**3),
    }
    (args.cache_dir / "inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n"
    )
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
