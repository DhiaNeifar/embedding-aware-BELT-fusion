"""Train the complete TrackFormer pipeline across same-timestamp CAVs."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from statistics import mean

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    spatial_embedding_loss,
    spatial_hard_negative_loss,
)
from embedding_aware_belt_fusion.embeddings.train_complete_roi import (
    ShardGroupedSampler,
    _limit_records,
    _split_records,
)
from embedding_aware_belt_fusion.experiments.overfit_spatial_trackformer import (
    _metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--validation-cache-dir",
        type=Path,
        help=(
            "Separate held-out cache for validation. This is useful for "
            "noise fine-tuning: train and validate on the predefined "
            "scenario split without extracting the validation frames twice."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--trackformer-root",
        type=Path,
        default=Path("external/trackformer"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
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
    parser.add_argument("--false-track-queries", type=int, default=2)
    parser.add_argument("--false-negative-probability", type=float, default=0.1)
    parser.add_argument("--embedding-weight", type=float, default=1.0)
    parser.add_argument("--hard-negative-weight", type=float, default=1.0)
    parser.add_argument("--hard-negative-radius", type=float, default=15.0)
    parser.add_argument(
        "--hard-negative-size-difference", type=float, default=2.0
    )
    parser.add_argument("--hard-negative-margin", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--resume", type=Path)
    checkpoint.add_argument("--initialize-from", type=Path)
    parser.add_argument(
        "--training-stage",
        choices=("full", "association-head", "joint"),
        default="full",
    )
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def _collate(frames):
    return frames


def _has_valid_proposals(agent):
    """Return whether an agent can be consumed by the transformer."""
    if len(agent.get("boxes", ())) == 0:
        return False
    tokens = agent.get("tokens")
    mask = agent.get("mask")
    if tokens is None or mask is None or tokens.numel() == 0:
        return False
    return bool((~mask.bool()).any())


def _ego_and_sources(frame):
    identity = torch.eye(4)
    agents = frame["agents"]
    ego_index = min(
        range(len(agents)),
        key=lambda index: float(
            (
                agents[index]["transformation_matrix"].float() - identity
            ).abs().max()
        ),
    )
    ego = agents[ego_index]
    if not _has_valid_proposals(ego):
        return None, []
    return ego, [
        agent
        for index, agent in enumerate(agents)
        if index != ego_index and _has_valid_proposals(agent)
    ]


def _configure_training_stage(model, stage):
    if stage == "association-head":
        model.requires_grad_(False)
        model.association_head.requires_grad_(True)
    elif stage in {"full", "joint"}:
        model.requires_grad_(True)
    else:
        raise ValueError(f"Unknown training stage: {stage}")


def _pair_record(model, matcher, criterion, source, target, args, training):
    run = model.forward_pair(
        source,
        target,
        matcher,
        false_track_queries=args.false_track_queries,
        false_negative_probability=(
            args.false_negative_probability if training else 0.0
        ),
    )
    source_losses = criterion(run["source_output"], run["source_target"])
    target_losses = criterion(run["target_output"], run["target"])
    embedding_loss, embedding_top1, embedding_objects = (
        spatial_embedding_loss(
            run["source_output"],
            source,
            run["target_output"],
            target,
            temperature=args.temperature,
        )
    )
    hard_negative_loss, hard_negative_anchors = spatial_hard_negative_loss(
        run["source_output"],
        source,
        run["target_output"],
        target,
        radius=args.hard_negative_radius,
        size_difference=args.hard_negative_size_difference,
        margin=args.hard_negative_margin,
    )
    association_loss = (
        args.embedding_weight * embedding_loss
        + args.hard_negative_weight * hard_negative_loss
    )
    total = association_loss
    if args.training_stage != "association-head":
        total = total + source_losses["loss"] + target_losses["loss"]
    metrics = _metrics(run, matcher)
    return {
        "tensor_loss": total,
        "loss": float(total.detach()),
        "source_loss": float(source_losses["loss"].detach()),
        "target_loss": float(target_losses["loss"].detach()),
        "embedding_loss": float(embedding_loss.detach()),
        "embedding_top1": float(embedding_top1.detach()),
        "embedding_objects": float(embedding_objects),
        "hard_negative_loss": float(hard_negative_loss.detach()),
        "hard_negative_anchors": float(hard_negative_anchors),
        **metrics,
    }


def _run_epoch(model, loader, matcher, criterion, device, args, optimizer, label):
    training = optimizer is not None
    model.train(training)
    if training and args.training_stage == "association-head":
        model.eval()
        model.association_head.train()
    records = []
    context = torch.enable_grad() if training else torch.no_grad()
    autocast = (
        lambda: torch.autocast("cuda", dtype=torch.bfloat16)
        if args.amp and device.type == "cuda"
        else nullcontext()
    )
    progress = tqdm(loader, desc=label)
    with context:
        for frames in progress:
            if training:
                optimizer.zero_grad(set_to_none=True)
            batch = []
            with autocast():
                for frame in frames:
                    ego, sources = _ego_and_sources(frame)
                    if ego is None:
                        continue
                    for source in sources:
                        batch.append(
                            _pair_record(
                                model,
                                matcher,
                                criterion,
                                source,
                                ego,
                                args,
                                training,
                            )
                        )
                if not batch:
                    continue
                loss = torch.stack(
                    [record["tensor_loss"] for record in batch]
                ).mean()
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            for record in batch:
                record.pop("tensor_loss")
            records.extend(batch)
            progress.set_postfix(
                loss=f"{mean(x['loss'] for x in batch):.3f}",
                track=f"{mean(x['track_object_accuracy'] for x in batch):.3f}",
                embed=f"{mean(x['embedding_top1'] for x in batch):.3f}",
            )
    if not records:
        raise RuntimeError("No cooperative source-to-ego pairs were found")
    return {key: mean(record[key] for record in records) for key in records[0]}


def _checkpoint(epoch, model, optimizer, scheduler, args, split, metrics):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "arguments": vars(args),
        "split": split,
        "metrics": metrics,
        "pipeline": "spatial_trackformer_v2_geometry_association",
    }


def main():
    args = parse_args()
    # Each cached frame contains many tensors. PyTorch's default
    # ``file_descriptor`` multiprocessing transport retains one descriptor
    # per shared storage and eventually exceeds the process soft limit.
    # ``file_system`` keeps multi-worker loading without that descriptor
    # accumulation.
    if args.workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    initial_checkpoint = None
    checkpoint_path = args.resume or args.initialize_from
    if checkpoint_path is not None:
        initial_checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        architecture = initial_checkpoint["arguments"]
        for name in (
            "embedding_dim",
            "d_model",
            "heads",
            "encoder_layers",
            "decoder_layers",
            "feedforward_dim",
            "dropout",
        ):
            setattr(args, name, architecture[name])
    index = load_or_build_scenario_index(
        args.cache_dir, workers=args.index_workers
    )
    if args.validation_cache_dir is None:
        train, validation, train_scenarios, validation_scenarios = _split_records(
            index,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        validation_cache_dir = args.cache_dir
    else:
        validation_cache_dir = args.validation_cache_dir
        train_manifest = json.loads(
            (args.cache_dir / "manifest.json").read_text()
        )
        validation_manifest = json.loads(
            (validation_cache_dir / "manifest.json").read_text()
        )
        if train_manifest.get("feature_dim") != validation_manifest.get(
            "feature_dim"
        ):
            raise ValueError(
                "Training and validation caches use different feature dimensions"
            )
        if train_manifest.get("noise") != validation_manifest.get("noise"):
            raise ValueError(
                "Training and validation caches must use identical noise settings"
            )
        validation_index = load_or_build_scenario_index(
            validation_cache_dir, workers=args.index_workers
        )
        train, validation = index["frames"], validation_index["frames"]
        train_scenarios = sorted(
            {record["scenario_id"] for record in train}
        )
        validation_scenarios = sorted(
            {record["scenario_id"] for record in validation}
        )
        overlap = set(train_scenarios) & set(validation_scenarios)
        if overlap:
            raise ValueError(
                "Training and validation caches overlap in scenarios: "
                f"{sorted(overlap)}"
            )
    train = _limit_records(train, args.max_train_frames)
    validation = _limit_records(validation, args.max_validation_frames)
    train_data = IndexedCompleteFrameDataset(args.cache_dir, train)
    validation_data = IndexedCompleteFrameDataset(
        validation_cache_dir, validation
    )
    options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "collate_fn": _collate,
        "pin_memory": False,
        "persistent_workers": args.workers > 0,
        "prefetch_factor": args.prefetch_factor if args.workers > 0 else None,
    }
    train_loader = DataLoader(
        train_data,
        sampler=ShardGroupedSampler(train_data, seed=args.seed),
        **options,
    )
    validation_loader = DataLoader(
        validation_data, shuffle=False, **options
    )
    split = {
        "seed": args.seed,
        "train_scenarios": train_scenarios,
        "validation_scenarios": validation_scenarios,
        "train_frames": len(train_data),
        "validation_frames": len(validation_data),
        "validation_cache_dir": str(Path(validation_cache_dir).resolve()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scenario_split.json").write_text(
        json.dumps(split, indent=2) + "\n"
    )
    model = SpatialTrackFormer(
        trackformer_root=args.trackformer_root,
        input_dim=int(train_data.manifest["feature_dim"]),
        d_model=args.d_model,
        embedding_dim=args.embedding_dim,
        heads=args.heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
    ).to(device)
    if initial_checkpoint is not None:
        missing, unexpected = model.load_state_dict(
            initial_checkpoint["model_state_dict"], strict=False
        )
        allowed_missing = {
            name
            for name in model.state_dict()
            if name.startswith("association_head.")
        }
        if set(missing) - allowed_missing or unexpected:
            raise RuntimeError(
                f"Incompatible initialization: missing={missing}, "
                f"unexpected={unexpected}"
            )
        print(
            f"Initialized architecture and weights from {checkpoint_path}; "
            f"new parameters={sorted(missing)}"
        )
    _configure_training_stage(model, args.training_stage)
    matcher = SpatialHungarianMatcher()
    criterion = SpatialTrackFormerCriterion(matcher)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.minimum_learning_rate,
    )
    start_epoch, best, stale, history = 0, -1.0, 0, []
    if args.resume:
        saved = initial_checkpoint
        if saved["split"] != split:
            raise ValueError("Resume checkpoint uses a different split")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        start_epoch = int(saved["epoch"])
        best = float(saved["metrics"]["validation_embedding_top1"])
    elif args.initialize_from:
        # Checkpoints may have been selected on clean validation data.  Their
        # stored score is not comparable to this run's noisy validation set,
        # so establish the baseline on the current held-out cache instead.
        baseline = _run_epoch(
            model,
            validation_loader,
            matcher,
            criterion,
            device,
            args,
            None,
            "initial noisy validation",
        )
        best = float(baseline["embedding_top1"])
        baseline_record = {
            "epoch": 0,
            **{f"validation_{key}": value for key, value in baseline.items()},
        }
        initialized_best = _checkpoint(
            0,
            model,
            optimizer,
            scheduler,
            args,
            split,
            baseline_record,
        )
        initialized_best["initialized_from"] = str(args.initialize_from)
        torch.save(
            initialized_best,
            args.output_dir / "spatial_trackformer_best.pth",
        )
        print(
            f"Starting fresh {args.training_stage!r} optimization from "
            f"{args.initialize_from}; noisy initialization score="
            f"{best:.6f}"
        )
    print(
        f"Spatial TrackFormer: {len(train_data)} train frames, "
        f"{len(validation_data)} unseen-scenario validation frames; "
        f"stage={args.training_stage}"
    )
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            matcher,
            criterion,
            device,
            args,
            optimizer,
            f"training {epoch}/{args.epochs}",
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            matcher,
            criterion,
            device,
            args,
            None,
            f"validation {epoch}/{args.epochs}",
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"validation_{k}": v for k, v in validation_metrics.items()},
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True))
        saved = _checkpoint(
            epoch, model, optimizer, scheduler, args, split, record
        )
        torch.save(saved, args.output_dir / "spatial_trackformer_latest.pth")
        score = validation_metrics["embedding_top1"]
        if score > best:
            best, stale = score, 0
            torch.save(
                saved, args.output_dir / "spatial_trackformer_best.pth"
            )
        else:
            stale += 1
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n"
        )
        if args.early_stopping_patience and stale >= args.early_stopping_patience:
            print("Early stopping on validation embedding top-1")
            break
    print(
        "FINAL "
        + json.dumps(
            {
                "best_validation_embedding_top1": best,
                "completed_epochs": history[-1]["epoch"],
            }
        )
    )


if __name__ == "__main__":
    main()
