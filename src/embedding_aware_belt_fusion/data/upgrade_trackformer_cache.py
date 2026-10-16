"""Add GT-box supervision to an existing spatial cache without rerunning PointPillars."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opencood-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    source_manifest = json.loads(
        (args.source_cache / "manifest.json").read_text()
    )
    if not source_manifest.get("bev_shape"):
        raise ValueError("Source cache has no spatial BEV maps")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from opencood.data_utils.post_processor import build_postprocessor
    from opencood.hypes_yaml.yaml_utils import load_yaml

    hypes = load_yaml(str(args.opencood_config), None)
    postprocessor = build_postprocessor(hypes["postprocess"], train=False)

    def upgrade_frame(frame):
        for agent in frame["agents"]:
            annotation = (
                args.data_root
                / args.split
                / frame["scenario_id"]
                / str(agent["agent_id"])
                / f"{frame['frame_id']}.yaml"
            )
            if not annotation.is_file():
                raise FileNotFoundError(f"Missing OPV2V annotation {annotation}")
            params = load_yaml(str(annotation), None)
            centers, mask, object_ids = postprocessor.generate_object_center(
                [{"params": params}], params["lidar_pose"]
            )
            centers = torch.from_numpy(centers[mask.astype(bool)]).float()
            box_by_id = {
                str(object_id): centers[index]
                for index, object_id in enumerate(object_ids)
            }
            gt_boxes = torch.zeros_like(agent["boxes"]).float()
            for proposal_index, object_id in enumerate(agent["gt_ids"]):
                if object_id is None:
                    continue
                box = box_by_id.get(str(object_id))
                if box is None:
                    raise RuntimeError(
                        f"Object {object_id} missing from {annotation}"
                    )
                gt_boxes[proposal_index] = box
            agent["gt_boxes"] = gt_boxes
        return frame

    for shard in tqdm(source_manifest["shards"], desc="upgrading cache shards"):
        source_path = args.source_cache / shard["path"]
        frames = torch.load(source_path, map_location="cpu", weights_only=False)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            upgraded = list(executor.map(upgrade_frame, frames))
        torch.save(upgraded, args.output_dir / shard["path"])

    upgraded_manifest = {
        **source_manifest,
        "schema_version": 2,
        "upgraded_from": str(args.source_cache.resolve()),
        "ground_truth_boxes": True,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(upgraded_manifest, indent=2) + "\n"
    )
    print(json.dumps(upgraded_manifest, indent=2))


if __name__ == "__main__":
    main()
