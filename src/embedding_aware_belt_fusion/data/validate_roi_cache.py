"""Validate a materialized complete-ROI training cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from embedding_aware_belt_fusion.data.complete_cache import _load_shard


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--max-shards", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    if manifest.get("cache_mode") != "complete_roi_training":
        raise ValueError("Expected cache_mode=complete_roi_training")
    if not manifest.get("complete"):
        raise ValueError("Compact cache is marked incomplete")
    source_inventory = {}
    if manifest.get("source_cache"):
        source_cache = Path(manifest["source_cache"])
        source_manifest = json.loads(
            (source_cache / "manifest.json").read_text()
        )
        source_inventory_path = source_cache / "inventory.json"
        source_inventory = (
            json.loads(source_inventory_path.read_text())
            if source_inventory_path.is_file()
            else {}
        )
        expected_counts = {
            "frame_count": source_manifest["frame_count"],
            "agent_count": source_inventory.get("agents"),
            "proposal_count": source_inventory.get("proposals"),
        }
        for field, expected in expected_counts.items():
            if expected is not None and manifest[field] != expected:
                raise ValueError(f"{field} does not match the source cache")

    shards = manifest["shards"]
    if args.max_shards is not None:
        shards = shards[: args.max_shards]
    frames = agents = proposals = matched = real_cells = 0
    minimum_cells = None
    maximum_cells = 0
    validated_bytes = 0
    feature_dim = int(manifest["feature_dim"])
    progress = tqdm(shards, desc="validating compact ROI cache")
    for shard in progress:
        path = args.cache_dir / shard["path"]
        records = _load_shard(path)
        if len(records) != shard["frames"]:
            raise ValueError(f"Frame-count mismatch in {path}")
        validated_bytes += path.stat().st_size
        frames += len(records)
        for frame in records:
            if not frame["scenario_id"] or not frame["frame_id"]:
                raise ValueError(f"Missing frame identity in {path}")
            agents += len(frame["agents"])
            for agent in frame["agents"]:
                transformation = agent.get("transformation_matrix")
                if (
                    transformation is None
                    or transformation.shape != (4, 4)
                    or not torch.isfinite(transformation).all()
                ):
                    raise ValueError(
                        f"Invalid transformation matrix in {path}"
                    )
                count = len(agent["gt_ids"])
                expected = {
                    "tokens": count,
                    "positions": count,
                    "mask": count,
                    "boxes": count,
                    "scores": count,
                    "gt_ious": count,
                    "gt_boxes": count,
                    "anchor_indices": count,
                }
                for field, length in expected.items():
                    if len(agent[field]) != length:
                        raise ValueError(
                            f"{field} length mismatch in {path}"
                        )
                tokens = agent["tokens"]
                positions = agent["positions"]
                mask = agent["mask"]
                if tokens.ndim != 3 or tokens.shape[-1] != feature_dim:
                    raise ValueError(f"Invalid token shape in {path}")
                if positions.shape != (*tokens.shape[:2], 2):
                    raise ValueError(f"Invalid position shape in {path}")
                if mask.shape != tokens.shape[:2] or mask.dtype != torch.bool:
                    raise ValueError(f"Invalid mask shape/type in {path}")
                if count:
                    cell_counts = (~mask).sum(dim=1)
                    if (cell_counts <= 0).any():
                        raise ValueError(f"Empty proposal ROI in {path}")
                    current_min = int(cell_counts.min())
                    current_max = int(cell_counts.max())
                    minimum_cells = (
                        current_min
                        if minimum_cells is None
                        else min(minimum_cells, current_min)
                    )
                    maximum_cells = max(maximum_cells, current_max)
                    real_cells += int(cell_counts.sum())
                    if not torch.isfinite(
                        tokens.masked_select((~mask).unsqueeze(-1))
                    ).all():
                        raise ValueError(f"Non-finite ROI token in {path}")
                    if not torch.isfinite(positions).all():
                        raise ValueError(f"Non-finite position in {path}")
                proposals += count
                matched += sum(
                    object_id is not None
                    for object_id in agent["gt_ids"]
                )

    full_validation = len(shards) == len(manifest["shards"])
    if full_validation:
        if frames != manifest["frame_count"]:
            raise ValueError("Validated frame count does not match manifest")
        if agents != manifest["agent_count"]:
            raise ValueError("Validated agent count does not match manifest")
        if proposals != manifest["proposal_count"]:
            raise ValueError(
                "Validated proposal count does not match manifest"
            )
        expected_matched = source_inventory.get("matched_proposals")
        if expected_matched is not None and matched != expected_matched:
            raise ValueError(
                "Matched-proposal count does not match source inventory"
            )
    result = {
        "validated": True,
        "complete_validation": full_validation,
        "validated_shards": len(shards),
        "frames": frames,
        "agents": agents,
        "proposals": proposals,
        "matched_proposals": matched,
        "feature_dim": feature_dim,
        "mean_cells_per_proposal": real_cells / max(proposals, 1),
        "min_cells_per_proposal": minimum_cells,
        "max_cells_per_proposal": maximum_cells,
        "validated_bytes": validated_bytes,
        "validated_gib": validated_bytes / 1024**3,
    }
    (args.cache_dir / "inventory.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
