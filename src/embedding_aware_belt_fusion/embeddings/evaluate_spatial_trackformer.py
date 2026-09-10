"""Evaluate one frozen spatial TrackFormer checkpoint on a complete ROI cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from embedding_aware_belt_fusion.data.complete_cache import (
    IndexedCompleteFrameDataset,
    load_or_build_scenario_index,
)
from embedding_aware_belt_fusion.embeddings.spatial_trackformer import (
    SpatialTrackFormer,
)
from embedding_aware_belt_fusion.embeddings.spatial_trackformer_loss import (
    SpatialHungarianMatcher,
    SpatialTrackFormerCriterion,
)
from embedding_aware_belt_fusion.embeddings.train_spatial_trackformer import (
    _collate,
    _run_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--trackformer-root",
        type=Path,
        default=Path("external/trackformer"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--index-workers", type=int, default=8)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
    device = torch.device(args.device)
    saved = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    pipeline = saved.get("pipeline")
    supported = {
        "spatial_trackformer_v1",
        "spatial_trackformer_v2_geometry_association",
    }
    if pipeline not in supported:
        raise ValueError("Checkpoint is not a supported spatial TrackFormer")
    training = dict(saved["arguments"])
    cache_manifest = json.loads(
        (args.cache_dir / "manifest.json").read_text()
    )
    model = SpatialTrackFormer(
        trackformer_root=args.trackformer_root,
        input_dim=int(cache_manifest["feature_dim"]),
        d_model=int(training["d_model"]),
        embedding_dim=int(training["embedding_dim"]),
        heads=int(training["heads"]),
        encoder_layers=int(training["encoder_layers"]),
        decoder_layers=int(training["decoder_layers"]),
        feedforward_dim=int(training["feedforward_dim"]),
        dropout=float(training["dropout"]),
        geometry_association=(
            pipeline == "spatial_trackformer_v2_geometry_association"
        ),
    ).to(device)
    model.load_state_dict(
        saved["model_state_dict"],
        strict=(
            pipeline == "spatial_trackformer_v2_geometry_association"
        ),
    )

    index = load_or_build_scenario_index(
        args.cache_dir, workers=args.index_workers
    )
    records = index["frames"]
    if args.max_frames is not None:
        records = records[: args.max_frames]
    dataset = IndexedCompleteFrameDataset(args.cache_dir, records)
    scenario_role = (
        dataset.manifest.get("scenario_subset") or {}
    ).get("role")
    evaluation_split = (
        f"{dataset.manifest.get('split')}:{scenario_role}_scenarios"
        if scenario_role
        else dataset.manifest.get("split")
    )
    if dataset.manifest.get("split") != "test" and scenario_role != "validation":
        print(
            "WARNING: cache manifest split is "
            f"{dataset.manifest.get('split')!r}, not 'test'"
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=_collate,
        pin_memory=False,
        persistent_workers=args.workers > 0,
        prefetch_factor=(
            args.prefetch_factor if args.workers > 0 else None
        ),
    )
    evaluation_args = SimpleNamespace(
        amp=args.amp,
        false_track_queries=int(training["false_track_queries"]),
        false_negative_probability=0.0,
        embedding_weight=float(training["embedding_weight"]),
        temperature=float(training["temperature"]),
        hard_negative_weight=float(
            training.get("hard_negative_weight", 0.0)
        ),
        hard_negative_radius=float(
            training.get("hard_negative_radius", 15.0)
        ),
        hard_negative_size_difference=float(
            training.get("hard_negative_size_difference", 2.0)
        ),
        hard_negative_margin=float(
            training.get("hard_negative_margin", 0.2)
        ),
        training_stage=training.get("training_stage", "full"),
        association_protocol=saved.get("association_protocol", "propagated"),
    )
    matcher = SpatialHungarianMatcher()
    criterion = SpatialTrackFormerCriterion(matcher)
    metrics = _run_epoch(
        model,
        loader,
        matcher,
        criterion,
        device,
        evaluation_args,
        None,
        evaluation_split,
    )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(saved["epoch"]),
        "cache": str(args.cache_dir.resolve()),
        "split": evaluation_split,
        "frames": len(dataset),
        **metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("TEST " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
