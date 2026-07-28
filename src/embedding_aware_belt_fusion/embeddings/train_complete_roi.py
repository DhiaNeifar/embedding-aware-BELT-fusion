"""Train complete-ROI TrackFormer embeddings with scenario-separated validation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from statistics import mean

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from embedding_aware_belt_fusion.data.complete_cache import (
    IndexedCompleteFrameDataset,
    load_or_build_scenario_index,
)
from embedding_aware_belt_fusion.experiments.overfit_complete_roi import (
    CompleteROIEncoder,
    _agent_rois,
    _best_proposals,
    _pair_loss,
)
from embedding_aware_belt_fusion.embeddings.geometry import (
    GeometryAwareROIEncoder,
    normalized_ego_geometry,
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--index-workers", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-frames", type=int)
    parser.add_argument("--max-validation-frames", type=int)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--feedforward-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--variance-weight", type=float, default=5.0)
    parser.add_argument("--variance-target", type=float, default=0.05)
    parser.add_argument(
        "--geometry-aware",
        action="store_true",
        help=(
            "Fuse ROI appearance with ego-frame box position, dimensions, "
            "orientation, and confidence"
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume",
        type=Path,
        help="Resume model, optimizer, scheduler, and epoch exactly",
    )
    checkpoint_group.add_argument(
        "--initialize-from",
        type=Path,
        help="Load model weights only and start a fresh optimizer schedule",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA bfloat16 autocast",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


class ShardGroupedSampler(Sampler):
    """Shuffle without repeatedly reopening the 429-GiB cache shards."""

    def __init__(self, dataset, *, seed):
        self.seed = seed
        self.epoch = 0
        grouped = defaultdict(list)
        for dataset_index, (shard_index, _) in enumerate(dataset.references):
            grouped[shard_index].append(dataset_index)
        self.groups = dict(grouped)

    def __len__(self):
        return sum(len(indices) for indices in self.groups.values())

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        shard_ids = list(self.groups)
        shard_order = torch.randperm(
            len(shard_ids), generator=generator
        ).tolist()
        for shard_offset in shard_order:
            indices = self.groups[shard_ids[shard_offset]]
            order = torch.randperm(
                len(indices), generator=generator
            ).tolist()
            yield from (indices[offset] for offset in order)


class CompactROIDataset:
    """Gather only labeled ROI cells before a frame leaves a CPU worker."""

    def __init__(self, dataset, *, geometry_aware=False):
        self.dataset = dataset
        self.references = dataset.references
        self.manifest = dataset.manifest
        self.geometry_aware = geometry_aware
        _, self.height, self.width = self.manifest["bev_shape"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        frame = self.dataset[index]
        agents = []
        for agent in frame["agents"]:
            proposals = _best_proposals(agent)
            if not proposals:
                continue
            object_ids = sorted(proposals)
            if "tokens" in agent:
                indices = torch.tensor(
                    [proposals[object_id] for object_id in object_ids],
                    dtype=torch.long,
                )
                tokens = agent["tokens"][indices].float()
                positions = agent["positions"][indices].float()
                mask = agent["mask"][indices].bool()
            else:
                tokens, positions, mask, object_ids = _agent_rois(
                    agent,
                    proposals,
                    height=self.height,
                    width=self.width,
                    device=torch.device("cpu"),
                )
                indices = torch.tensor(
                    [proposals[object_id] for object_id in object_ids],
                    dtype=torch.long,
                )
            geometry = None
            if self.geometry_aware:
                if "transformation_matrix" not in agent:
                    raise KeyError(
                        "Geometry-aware training requires each cached agent's "
                        "transformation_matrix"
                    )
                geometry = normalized_ego_geometry(
                    agent["boxes"][indices],
                    agent["scores"][indices],
                    agent["transformation_matrix"],
                )
            agents.append(
                {
                    "tokens": tokens,
                    "positions": positions,
                    "mask": mask,
                    "geometry": geometry,
                    "object_ids": object_ids,
                }
            )
        return {"agents": agents}


def _collate_frames(frames):
    agents = [
        agent for frame in frames for agent in frame["agents"]
    ]
    frame_agent_counts = [len(frame["agents"]) for frame in frames]
    if not agents:
        return {
            "tokens": None,
            "positions": None,
            "mask": None,
            "geometry": None,
            "proposal_counts": [],
            "object_ids": [],
            "frame_agent_counts": frame_agent_counts,
        }
    maximum = max(agent["tokens"].shape[1] for agent in agents)
    tokens, positions, masks = [], [], []
    for agent in agents:
        padding = maximum - agent["tokens"].shape[1]
        tokens.append(F.pad(agent["tokens"], (0, 0, 0, padding)))
        positions.append(F.pad(agent["positions"], (0, 0, 0, padding)))
        masks.append(F.pad(agent["mask"], (0, padding), value=True))
    geometries = [agent["geometry"] for agent in agents]
    geometry = (
        torch.cat(geometries)
        if geometries[0] is not None
        else None
    )
    return {
        "tokens": torch.cat(tokens),
        "positions": torch.cat(positions),
        "mask": torch.cat(masks),
        "geometry": geometry,
        "proposal_counts": [
            len(agent["object_ids"]) for agent in agents
        ],
        "object_ids": [agent["object_ids"] for agent in agents],
        "frame_agent_counts": frame_agent_counts,
    }


def _encode_frame_batch(model, batch, device):
    if batch["tokens"] is None:
        return [[] for _ in batch["frame_agent_counts"]]
    tokens = batch["tokens"].to(device, non_blocking=True)
    positions = batch["positions"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    if batch["geometry"] is None:
        embeddings = model(tokens, positions, mask)
    else:
        geometry = batch["geometry"].to(device, non_blocking=True)
        embeddings = model(tokens, positions, mask, geometry)
    encoded_agents = []
    offset = 0
    for object_ids, count in zip(
        batch["object_ids"], batch["proposal_counts"]
    ):
        encoded_agents.append(
            (embeddings[offset : offset + count], object_ids)
        )
        offset += count
    encoded_frames = []
    agent_offset = 0
    for count in batch["frame_agent_counts"]:
        encoded_frames.append(
            encoded_agents[agent_offset : agent_offset + count]
        )
        agent_offset += count
    return encoded_frames


def _batch_records(
    model,
    batch,
    device,
    temperature,
    *,
    variance_weight,
    variance_target,
):
    encoded_frames = _encode_frame_batch(model, batch, device)
    records = []
    for agents in encoded_frames:
        for source_index in range(len(agents)):
            for target_index in range(source_index + 1, len(agents)):
                record = _pair_loss(
                    agents[source_index],
                    agents[target_index],
                    temperature,
                    variance_weight=variance_weight,
                    variance_target=variance_target,
                )
                if record is not None:
                    records.append(record)
    return records


def _run_batched_epoch(
    model,
    loader,
    device,
    args,
    *,
    optimizer=None,
    description,
):
    training = optimizer is not None
    model.train(training)
    records = []
    progress = tqdm(loader, desc=description)
    gradient_context = torch.enable_grad() if training else torch.no_grad()
    autocast_context = (
        lambda: torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        )
        if args.amp and device.type == "cuda"
        else nullcontext()
    )
    with gradient_context:
        for batch in progress:
            if training:
                optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                batch_records = _batch_records(
                    model,
                    batch,
                    device,
                    args.temperature,
                    variance_weight=args.variance_weight,
                    variance_target=args.variance_target,
                )
                if not batch_records:
                    continue
                loss = torch.stack(
                    [item["loss"] for item in batch_records]
                ).mean()
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            progress.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                top1=f"{mean(float(item['top1']) for item in batch_records):.3f}",
            )
            records.extend(
                {
                    key: float(value.detach())
                    if isinstance(value, torch.Tensor)
                    else float(value)
                    for key, value in item.items()
                }
                for item in batch_records
            )
    if not records:
        raise RuntimeError(
            "No batch contained identities shared by two agents"
        )
    return {
        key: mean(item[key] for item in records)
        for key in records[0]
    }


def _split_records(index, *, validation_fraction, seed):
    scenarios = sorted(
        {record["scenario_id"] for record in index["frames"]}
    )
    if len(scenarios) < 2:
        raise ValueError(
            "Scenario-separated training requires at least two scenarios"
        )
    if not 0 < validation_fraction < 1:
        raise ValueError("--validation-fraction must be between zero and one")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(scenarios), generator=generator).tolist()
    shuffled = [scenarios[index] for index in order]
    validation_count = max(1, round(len(scenarios) * validation_fraction))
    validation_count = min(validation_count, len(scenarios) - 1)
    validation_scenarios = set(shuffled[:validation_count])
    train_scenarios = set(shuffled[validation_count:])
    train_records = [
        item
        for item in index["frames"]
        if item["scenario_id"] in train_scenarios
    ]
    validation_records = [
        item
        for item in index["frames"]
        if item["scenario_id"] in validation_scenarios
    ]
    return (
        train_records,
        validation_records,
        sorted(train_scenarios),
        sorted(validation_scenarios),
    )


def _limit_records(records, maximum):
    if maximum is None:
        return records
    if maximum < 1:
        raise ValueError("Frame limits must be positive")
    return records[:maximum]


def _save_figure(history, path):
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epochs = [item["epoch"] for item in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(
        epochs, [item["train_loss"] for item in history], label="Train"
    )
    axes[0].plot(
        epochs,
        [item["validation_loss"] for item in history],
        label="Validation",
    )
    axes[0].set_title("Association objective")
    axes[1].plot(
        epochs, [item["train_top1"] for item in history], label="Train"
    )
    axes[1].plot(
        epochs,
        [item["validation_top1"] for item in history],
        label="Validation",
    )
    axes[1].plot(
        epochs,
        [item["validation_random_top1"] for item in history],
        "--",
        label="Random",
    )
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Cross-agent top-1")
    axes[2].plot(
        epochs,
        [item["validation_positive_cosine"] for item in history],
        label="Same object",
    )
    axes[2].plot(
        epochs,
        [item["validation_negative_cosine"] for item in history],
        label="Different object",
    )
    axes[2].set_title("Unseen-scenario similarity")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _checkpoint(
    *,
    epoch,
    model,
    optimizer,
    scheduler,
    args,
    manifest,
    split,
    metrics,
):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "arguments": vars(args),
        "manifest": manifest,
        "split": split,
        "metrics": metrics,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    index = load_or_build_scenario_index(
        args.cache_dir, workers=args.index_workers
    )
    (
        train_records,
        validation_records,
        train_scenarios,
        validation_scenarios,
    ) = _split_records(
        index,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    train_records = _limit_records(
        train_records, args.max_train_frames
    )
    validation_records = _limit_records(
        validation_records, args.max_validation_frames
    )
    train_data = CompactROIDataset(
        IndexedCompleteFrameDataset(args.cache_dir, train_records),
        geometry_aware=args.geometry_aware,
    )
    validation_data = CompactROIDataset(
        IndexedCompleteFrameDataset(args.cache_dir, validation_records),
        geometry_aware=args.geometry_aware,
    )
    train_sampler = ShardGroupedSampler(train_data, seed=args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "collate_fn": _collate_frames,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
        "prefetch_factor": (
            args.prefetch_factor if args.workers > 0 else None
        ),
    }
    train_loader = DataLoader(
        train_data, sampler=train_sampler, **loader_options
    )
    validation_loader = DataLoader(
        validation_data, shuffle=False, **loader_options
    )
    manifest = train_data.manifest
    model_class = (
        GeometryAwareROIEncoder
        if args.geometry_aware
        else CompleteROIEncoder
    )
    model = model_class(
        trackformer_root=args.trackformer_root,
        input_dim=int(manifest["feature_dim"]),
        d_model=args.d_model,
        embedding_dim=args.embedding_dim,
        heads=args.heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.minimum_learning_rate,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = {
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "train_scenarios": train_scenarios,
        "validation_scenarios": validation_scenarios,
        "train_frames": len(train_data),
        "validation_frames": len(validation_data),
    }
    (args.output_dir / "scenario_split.json").write_text(
        json.dumps(split, indent=2) + "\n"
    )
    print(
        f"Scenario split: {len(train_scenarios)} train scenarios/"
        f"{len(train_data)} frames; "
        f"{len(validation_scenarios)} validation scenarios/"
        f"{len(validation_data)} frames"
    )

    history = []
    best_top1 = -1.0
    epochs_without_improvement = 0
    start_epoch = 0
    if args.resume is not None:
        saved = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        if saved.get("split") != split:
            raise ValueError(
                "Resume checkpoint uses a different scenario split"
            )
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        start_epoch = int(saved["epoch"])
        best_top1 = float(
            saved.get("metrics", {}).get("validation_top1", -1.0)
        )
        print(
            f"Exactly resumed epoch {start_epoch} from {args.resume}; "
            "optimizer and learning-rate schedule were restored"
        )
    elif args.initialize_from is not None:
        saved = torch.load(
            args.initialize_from, map_location="cpu", weights_only=False
        )
        model.load_state_dict(saved["model_state_dict"])
        print(
            f"Initialized model weights from {args.initialize_from}; "
            "optimizer and learning-rate schedule start fresh"
        )

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_metrics = _run_batched_epoch(
            model,
            train_loader,
            device,
            args,
            optimizer=optimizer,
            description=f"training epoch {epoch}/{args.epochs}",
        )
        validation_metrics = _run_batched_epoch(
            model,
            validation_loader,
            device,
            args,
            description=f"validation epoch {epoch}/{args.epochs}",
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{
                f"train_{key}": value
                for key, value in train_metrics.items()
            },
            **{
                f"validation_{key}": value
                for key, value in validation_metrics.items()
            },
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True))
        saved = _checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            manifest=manifest,
            split=split,
            metrics=record,
        )
        torch.save(saved, args.output_dir / "complete_roi_latest.pth")
        if validation_metrics["top1"] > best_top1:
            best_top1 = validation_metrics["top1"]
            epochs_without_improvement = 0
            torch.save(saved, args.output_dir / "complete_roi_best.pth")
        else:
            epochs_without_improvement += 1
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n"
        )
        _save_figure(history, args.output_dir / "training_metrics.png")
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement
            >= args.early_stopping_patience
        ):
            print(
                "Early stopping: validation top-1 did not improve for "
                f"{args.early_stopping_patience} epochs"
            )
            break

    result = {
        "best_validation_top1": best_top1,
        "completed_epochs": history[-1]["epoch"] if history else start_epoch,
        "best_checkpoint": str(
            args.output_dir / "complete_roi_best.pth"
        ),
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("FINAL " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
