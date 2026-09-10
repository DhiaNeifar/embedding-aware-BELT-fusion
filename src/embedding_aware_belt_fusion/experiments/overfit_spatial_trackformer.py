"""Overfit one CAV pair with the complete spatial TrackFormer pipeline."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm import trange

from embedding_aware_belt_fusion.data.complete_cache import (
    CompleteFrameDataset,
    IndexedCompleteFrameDataset,
    load_or_build_scenario_index,
)
from embedding_aware_belt_fusion.embeddings.spatial_trackformer import (
    SpatialTrackFormer,
    spatial_target,
)
from embedding_aware_belt_fusion.embeddings.spatial_trackformer_loss import (
    SpatialHungarianMatcher,
    SpatialTrackFormerCriterion,
    pairwise_generalized_iou,
    spatial_embedding_loss,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--trackformer-root",
        type=Path,
        default=Path("external/trackformer"),
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--object-queries", type=int, default=64)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--feedforward-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--false-track-queries", type=int, default=2)
    parser.add_argument("--minimum-shared-objects", type=int, default=4)
    parser.add_argument("--max-search-frames", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def _ids(agent):
    return {
        str(object_id)
        for object_id in agent["gt_ids"]
        if object_id is not None
    }


def _find_pair(dataset, minimum_shared, maximum_frames):
    for frame_index in range(min(len(dataset), maximum_frames)):
        frame = dataset[frame_index]
        for source, target in combinations(frame["agents"], 2):
            shared = _ids(source).intersection(_ids(target))
            if len(shared) >= minimum_shared:
                return frame, source, target, sorted(shared)
    raise RuntimeError(
        f"No CAV pair with at least {minimum_shared} shared objects found"
    )


@torch.no_grad()
def _metrics(run, matcher):
    output, target = run["target_output"], run["target"]
    logits, boxes = output["pred_logits"][0], output["pred_boxes"][0]
    match_ids = target.get(
        "track_query_match_ids",
        torch.empty(0, dtype=torch.long, device=logits.device),
    )
    valid = match_ids >= 0
    false = ~valid
    track_count = len(match_ids)
    valid_indices = torch.arange(track_count, device=logits.device)[valid]
    false_indices = torch.arange(track_count, device=logits.device)[false]
    track_object_accuracy = (
        float((logits[valid_indices].argmax(-1) == 0).float().mean())
        if len(valid_indices)
        else 0.0
    )
    false_no_object_accuracy = (
        float((logits[false_indices].argmax(-1) == 1).float().mean())
        if len(false_indices)
        else 1.0
    )
    if len(valid_indices):
        expected = target["boxes"][match_ids[valid]]
        predicted = boxes[valid_indices]
        track_l1 = float(F.l1_loss(predicted, expected))
        track_giou = float(
            pairwise_generalized_iou(predicted, expected).diag().mean()
        )
    else:
        track_l1, track_giou = 0.0, 0.0
    prediction_indices, target_indices = matcher(output, target)
    matched_object_accuracy = (
        float(
            (
                logits[prediction_indices].argmax(-1)
                == target["labels"][target_indices]
            )
            .float()
            .mean()
        )
        if len(prediction_indices)
        else 1.0
    )
    return {
        "track_object_accuracy": track_object_accuracy,
        "false_track_no_object_accuracy": false_no_object_accuracy,
        "track_l1": track_l1,
        "track_giou": track_giou,
        "matched_object_accuracy": matched_object_accuracy,
        "track_queries": track_count,
        "target_objects": len(target["boxes"]),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    if manifest.get("cache_mode") == "complete":
        dataset = CompleteFrameDataset(args.cache_dir)
    elif manifest.get("cache_mode") == "complete_roi_training":
        index = load_or_build_scenario_index(args.cache_dir, workers=1)
        dataset = IndexedCompleteFrameDataset(
            args.cache_dir, index["frames"]
        )
    else:
        raise ValueError(
            "Spatial TrackFormer requires a complete or complete-ROI cache"
        )
    frame, source, target, shared = _find_pair(
        dataset,
        args.minimum_shared_objects,
        args.max_search_frames,
    )
    model = SpatialTrackFormer(
        trackformer_root=args.trackformer_root,
        d_model=args.d_model,
        embedding_dim=args.embedding_dim,
        num_object_queries=args.object_queries,
        heads=args.heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
    ).to(device)
    matcher = SpatialHungarianMatcher()
    criterion = SpatialTrackFormerCriterion(matcher)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    progress = trange(1, args.epochs + 1, desc="spatial TrackFormer overfit")
    for epoch in progress:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        run = model.forward_pair(
            source,
            target,
            matcher,
            false_track_queries=args.false_track_queries,
        )
        source_losses = criterion(
            run["source_output"], run["source_target"]
        )
        target_losses = criterion(run["target_output"], run["target"])
        embedding_loss, _, _ = spatial_embedding_loss(
            run["source_output"],
            source,
            run["target_output"],
            target,
        )
        loss = (
            source_losses["loss"]
            + target_losses["loss"]
            + embedding_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        evaluation = model.forward_pair(
            source,
            target,
            matcher,
            false_track_queries=args.false_track_queries,
        )
        metrics = _metrics(evaluation, matcher)
        evaluation_embedding_loss, embedding_top1, embedding_objects = (
            spatial_embedding_loss(
                evaluation["source_output"],
                source,
                evaluation["target_output"],
                target,
            )
        )
        record = {
            "epoch": epoch,
            "loss": float(loss.detach()),
            **{
                f"source_{name}": float(value.detach())
                for name, value in source_losses.items()
                if name.startswith("loss_")
            },
            **{
                f"target_{name}": float(value.detach())
                for name, value in target_losses.items()
                if name.startswith("loss_")
            },
            **metrics,
            "embedding_loss": float(evaluation_embedding_loss),
            "embedding_top1": float(embedding_top1),
            "embedding_objects": embedding_objects,
        }
        history.append(record)
        progress.set_postfix(
            loss=f"{record['loss']:.3f}",
            track=f"{record['track_object_accuracy']:.3f}",
            false=f"{record['false_track_no_object_accuracy']:.3f}",
            l1=f"{record['track_l1']:.3f}",
        )
        if (
            metrics["track_object_accuracy"] == 1.0
            and metrics["false_track_no_object_accuracy"] == 1.0
            and metrics["matched_object_accuracy"] == 1.0
            and metrics["track_l1"] < 0.01
            and float(embedding_top1) == 1.0
        ):
            break
    result = {
        "passed": (
            history[-1]["track_object_accuracy"] == 1.0
            and history[-1]["false_track_no_object_accuracy"] == 1.0
            and history[-1]["matched_object_accuracy"] == 1.0
            and history[-1]["track_l1"] < 0.01
            and history[-1]["embedding_top1"] == 1.0
        ),
        "scenario_id": str(frame["scenario_id"]),
        "frame_id": str(frame["frame_id"]),
        "source_agent": str(source["agent_id"]),
        "target_agent": str(target["agent_id"]),
        "shared_objects": len(shared),
        "final": history[-1],
    }
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n"
    )
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    torch.save(
        {"model_state_dict": model.state_dict(), "arguments": vars(args)},
        args.output_dir / "spatial_trackformer_overfit.pth",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
