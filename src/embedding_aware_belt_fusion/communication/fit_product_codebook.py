"""Fit a shared CodeFilling-style product codebook to frozen CAV messages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from embedding_aware_belt_fusion.communication.product_codebook import (
    ProductCodebook,
    checkpoint_payload,
)
from embedding_aware_belt_fusion.data.complete_cache import (
    IndexedCompleteFrameDataset,
    load_or_build_scenario_index,
)
from embedding_aware_belt_fusion.embeddings.spatial_trackformer import (
    SpatialTrackFormer,
)
from embedding_aware_belt_fusion.embeddings.train_spatial_trackformer import (
    _collate,
    _ego_and_sources,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--trackformer-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trackformer-root", type=Path, default=Path("external/trackformer")
    )
    parser.add_argument(
        "--representation", choices=("query-state", "embedding"), required=True
    )
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--entries", type=int, default=256)
    parser.add_argument("--max-vectors", type=int, default=100_000)
    parser.add_argument("--kmeans-iterations", type=int, default=25)
    parser.add_argument("--kmeans-chunk-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--index-workers", type=int, default=8)
    parser.add_argument("--scenario-split-file", type=Path, required=True)
    parser.add_argument("--scenario-role", choices=("train",), default="train")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--track-query-score-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def _load_model(checkpoint, cache_dir, trackformer_root, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("pipeline") != "spatial_trackformer_v2_geometry_association":
        raise ValueError("Expected a geometry-association Spatial TrackFormer checkpoint")
    training = saved["arguments"]
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    model = SpatialTrackFormer(
        trackformer_root=trackformer_root,
        input_dim=int(manifest["feature_dim"]),
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
    return model, saved


def _reservoir_update(reservoir, values, seen, generator):
    """Uniformly retain at most reservoir.shape[0] rows without disk caching."""
    values = values.detach().float().cpu()
    capacity = len(reservoir)
    for value in values:
        if seen < capacity:
            reservoir[seen] = value
        else:
            replacement = int(
                torch.randint(seen + 1, (), generator=generator).item()
            )
            if replacement < capacity:
                reservoir[replacement] = value
        seen += 1
    return seen


def main():
    args = parse_args()
    if args.workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
    device = torch.device(args.device)
    model, checkpoint = _load_model(
        args.trackformer_checkpoint, args.cache_dir, args.trackformer_root, device
    )
    split = json.loads(args.scenario_split_file.read_text())
    requested_scenarios = set(split[f"{args.scenario_role}_scenarios"])
    index = load_or_build_scenario_index(args.cache_dir, workers=args.index_workers)
    records = [
        record
        for record in index["frames"]
        if record["scenario_id"] in requested_scenarios
    ]
    if args.max_frames is not None:
        records = records[: args.max_frames]
    if not records:
        raise RuntimeError("No records matched the requested scenario role")
    dataset = IndexedCompleteFrameDataset(args.cache_dir, records)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=_collate,
        pin_memory=False,
        persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers else None,
    )
    dimension = (
        int(checkpoint["arguments"]["d_model"])
        if args.representation == "query-state"
        else int(checkpoint["arguments"]["embedding_dim"])
    )
    if args.max_vectors < args.entries:
        raise ValueError("--max-vectors must be at least --entries")
    reservoir = torch.empty(args.max_vectors, dimension)
    generator = torch.Generator().manual_seed(args.seed)
    seen = 0
    with torch.no_grad():
        for frames in tqdm(loader, desc=f"collecting {args.representation} messages"):
            _, sources = _ego_and_sources(frames[0])
            for source in sources:
                output = model.forward_agent(source)
                if args.representation == "query-state":
                    object_score = output["pred_logits"][0].softmax(-1)[:, 0]
                    selected = object_score >= args.track_query_score_threshold
                    values = output["hs_embed"][0][selected]
                else:
                    values = output["embeddings"][0]
                if len(values):
                    seen = _reservoir_update(reservoir, values, seen, generator)
    retained = min(seen, args.max_vectors)
    if retained < args.entries:
        raise RuntimeError(
            f"Only {retained} source messages were available; need {args.entries}"
        )
    quantizer = ProductCodebook(dimension, args.groups, args.entries).to(device)
    fit_metrics = quantizer.fit(
        reservoir[:retained],
        iterations=args.kmeans_iterations,
        chunk_size=args.kmeans_chunk_size,
        seed=args.seed,
    )
    result = checkpoint_payload(
        quantizer,
        representation=args.representation,
        source_checkpoint=str(args.trackformer_checkpoint.resolve()),
        fit_metrics={
            **fit_metrics,
            "seen_vectors": seen,
            "retained_vectors": retained,
            "scenario_role": args.scenario_role,
            "scenario_count": len(requested_scenarios),
            "track_query_score_threshold": args.track_query_score_threshold,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print("CODEBOOK " + json.dumps({k: v for k, v in result.items() if k != "state_dict"}, sort_keys=True))


if __name__ == "__main__":
    main()

