"""Train PointPillars–TrackFormer with propagated cross-agent track queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from embedding_aware_belt_fusion.data.trackformer_cache import (
    TrackFormerProposalDataset,
    collate_trackformer_frames,
)
from embedding_aware_belt_fusion.embeddings.trackformer import (
    PointPillarsTrackFormer,
)
from embedding_aware_belt_fusion.embeddings.trackformer_loss import (
    trackformer_pair_loss,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trackformer-root", type=Path, default=Path("external/trackformer"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--feedforward-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--false-track-queries", type=int, default=2)
    parser.add_argument("--association-weight", type=float, default=1.0)
    parser.add_argument("--variance-weight", type=float, default=1.0)
    parser.add_argument("--association-temperature", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--overfit-frames",
        type=int,
        help="Train and validate on the same N frames as a correctness test",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def _to_device(agent, device):
    return {key: value.to(device) for key, value in agent.items()}


def _group_agents(agents, *, shuffle):
    grouped = {}
    for agent in agents:
        group = int(agent["groups"][0])
        grouped.setdefault(group, []).append(agent)
    pairs = []
    for frame_agents in grouped.values():
        if len(frame_agents) < 2:
            continue
        if shuffle:
            order = torch.randperm(len(frame_agents)).tolist()
            frame_agents = [frame_agents[index] for index in order]
        pairs.extend(
            (frame_agents[index], frame_agents[(index + 1) % len(frame_agents)])
            for index in range(len(frame_agents))
        )
    return pairs


def _source_states(model, source, false_track_queries):
    false_count = min(false_track_queries, len(source["false_scores"]))
    if false_count:
        indices = torch.topk(source["false_scores"], false_count).indices
        features = torch.cat(
            [source["query_features"], source["false_query_features"][indices]]
        )
        boxes = torch.cat([source["boxes"], source["false_boxes"][indices]])
        scores = torch.cat([source["scores"], source["false_scores"][indices]])
    else:
        features, boxes, scores = (
            source["query_features"],
            source["boxes"],
            source["scores"],
        )
    states = model.forward_agent_states(
        source["bev"], features, boxes, scores
    )
    labels = torch.cat(
        [
            source["labels"],
            torch.full(
                (false_count,), -1, dtype=torch.long, device=states.device
            ),
        ]
    )
    return states, labels


def _retrieval(output, source_labels, target_labels):
    track_count = len(source_labels)
    true_source = source_labels >= 0
    target_by_label = {
        int(label): index for index, label in enumerate(target_labels.tolist())
    }
    query_indices, target_indices = [], []
    for index in torch.nonzero(true_source).flatten().tolist():
        target = target_by_label.get(int(source_labels[index]))
        if target is not None:
            query_indices.append(index)
            target_indices.append(target)
    if not query_indices:
        return None
    query_indices = torch.tensor(query_indices, device=source_labels.device)
    target_indices = torch.tensor(target_indices, device=source_labels.device)
    tracks = output["embeddings"][query_indices]
    objects = output["embeddings"][track_count:]
    similarity = tracks @ objects.T
    top1 = (similarity.argmax(dim=1) == target_indices).float().mean()
    positive = similarity[
        torch.arange(len(query_indices), device=similarity.device), target_indices
    ]
    negative_mask = torch.ones_like(similarity, dtype=torch.bool)
    negative_mask[
        torch.arange(len(query_indices), device=similarity.device), target_indices
    ] = False
    return {
        "top1": float(top1),
        "random_top1": 1.0 / max(len(objects), 1),
        "positive_cosine": float(positive.mean()),
        "negative_cosine": float(similarity[negative_mask].mean())
        if negative_mask.any()
        else 0.0,
    }


def _run_batch(model, agents, device, args, *, training):
    pair_losses, records = [], []
    for source_cpu, target_cpu in _group_agents(agents, shuffle=training):
        source = _to_device(source_cpu, device)
        target = _to_device(target_cpu, device)
        source_states, source_labels = _source_states(
            model, source, args.false_track_queries
        )
        output = model.forward_with_track_queries(
            target["bev"],
            target["query_features"],
            target["boxes"],
            target["scores"],
            source_states,
        )
        losses = trackformer_pair_loss(
            output,
            source_labels,
            target["labels"],
            model.normalize_boxes(target["target_boxes"]),
            association_weight=args.association_weight,
            variance_weight=args.variance_weight,
            association_temperature=args.association_temperature,
        )
        pair_losses.append(losses["loss"])
        record = {
            "classification": float(losses["classification"]),
            "l1": float(losses["l1"]),
            "giou": float(losses["giou"]),
            "association": float(losses["association"]),
            "variance": float(losses["variance"]),
        }
        retrieval = _retrieval(output, source_labels, target["labels"])
        if retrieval is not None:
            record.update(retrieval)
        records.append(record)
    return torch.stack(pair_losses).mean(), records


def _validate(model, loader, device, args):
    model.eval()
    records = []
    with torch.no_grad():
        for agents in loader:
            loss, batch_records = _run_batch(
                model, agents, device, args, training=False
            )
            for record in batch_records:
                record["loss"] = float(loss)
            records.extend(batch_records)
    keys = set().union(*(set(record) for record in records))
    return {
        key: mean(record[key] for record in records if key in record)
        for key in keys
    }


def _save_figure(history, path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epochs = [item["epoch"] for item in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="Train")
    axes[0].plot(epochs, [item["validation_loss"] for item in history], label="Val")
    axes[0].set_title("TrackFormer loss")
    axes[0].legend()
    axes[1].plot(
        epochs,
        [item["validation_top1"] for item in history],
        label="TrackFormer",
    )
    axes[1].plot(
        epochs,
        [item["validation_random_top1"] for item in history],
        "--",
        label="Random",
    )
    axes[1].set_title("Cross-agent retrieval")
    axes[1].legend()
    axes[2].plot(
        epochs,
        [item["validation_positive_cosine"] for item in history],
        label="Same",
    )
    axes[2].plot(
        epochs,
        [item["validation_negative_cosine"] for item in history],
        label="Different",
    )
    axes[2].set_title("Decoder embedding similarity")
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.multiprocessing.set_sharing_strategy("file_system")
    device = torch.device(args.device)
    train_data = TrackFormerProposalDataset(
        args.cache_dir, split="train", validation_fraction=args.validation_fraction
    )
    validation_data = TrackFormerProposalDataset(
        args.cache_dir,
        split="validation",
        validation_fraction=args.validation_fraction,
    )
    manifest = train_data.manifest
    if args.overfit_frames is not None:
        if args.overfit_frames < 1:
            raise ValueError("--overfit-frames must be positive")
        count = min(args.overfit_frames, len(train_data))
        train_data = Subset(train_data, range(count))
        validation_data = Subset(train_data.dataset, range(count))
        print(
            f"OVERFIT SANITY MODE: training and validating on the same "
            f"{count} frames"
        )
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_trackformer_frames,
        pin_memory=False,
        persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
    )
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_args)
    model = PointPillarsTrackFormer(
        args.trackformer_root,
        input_channels=int(manifest["bev_shape"][0]),
        d_model=args.d_model,
        embedding_dim=args.embedding_dim,
        nhead=args.heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history, best, start_epoch = [], -1.0, 0
    if args.resume is not None:
        checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        validation = _validate(
            model, validation_loader, device, args
        )
        best = validation["top1"]
        history.append(
            {
                "epoch": start_epoch,
                "train_loss": float("nan"),
                **{
                    f"validation_{key}": value
                    for key, value in validation.items()
                },
            }
        )
        print(
            f"Resuming after epoch {start_epoch}: "
            f"validation_top1={validation['top1']:.6f}"
        )
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        progress = tqdm(
            train_loader, desc=f"TrackFormer epoch {epoch + 1}/{args.epochs}"
        )
        for agents in progress:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _run_batch(
                model,
                agents,
                device,
                args,
                training=True,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            losses.append(float(loss))
            progress.set_postfix(loss=f"{float(loss):.4f}")
        scheduler.step()
        validation = _validate(
            model, validation_loader, device, args
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": mean(losses),
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True))
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "arguments": vars(args),
            "bev_shape": manifest["bev_shape"],
            "metrics": record,
        }
        torch.save(checkpoint, args.output_dir / f"trackformer_epoch{epoch + 1}.pth")
        if validation["top1"] > best:
            best = validation["top1"]
            torch.save(checkpoint, args.output_dir / "trackformer_best.pth")
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2, default=str) + "\n"
        )
        _save_figure(history, args.output_dir / "training_metrics.png")
        collapse_check_epoch = (
            9 if args.overfit_frames is not None else max(1, start_epoch + 1)
        )
        if (
            epoch >= collapse_check_epoch
            and abs(
                validation["positive_cosine"]
                - validation["negative_cosine"]
            )
            < 1e-3
            and validation["top1"] <= validation["random_top1"]
        ):
            raise RuntimeError(
                "Embedding collapse detected after two epochs; training was "
                "stopped automatically instead of wasting the remaining epochs"
            )


if __name__ == "__main__":
    main()
