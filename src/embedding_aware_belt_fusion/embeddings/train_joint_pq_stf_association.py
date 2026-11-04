"""Jointly fine-tune the PQ-STF message encoder and all-agent pair head.

The deployed PQ-STF association path is

    ROI tokens + proposal geometry -> Transformer -> 256-D query state h
    (h_a, g_a, h_b, g_b) -> symmetric same-object/no-match pair head.

Earlier experiments trained these two parts independently.  This trainer
back-propagates the pair-classification loss through the pair head and the
complete query-state path at the same time.  PointPillars remains frozen: the
cached PointPillars ROI tokens are the fixed detector input to PQ-STF.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from itertools import combinations
import json
from pathlib import Path
from statistics import mean

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from embedding_aware_belt_fusion.data.complete_cache import IndexedCompleteFrameDataset
from embedding_aware_belt_fusion.embeddings.all_agent_association import (
    SymmetricQueryStatePairHead,
    binary_pair_loss_and_metrics,
    has_valid_proposals,
    pair_head_scores,
    records_for_role,
)
from embedding_aware_belt_fusion.embeddings.spatial_trackformer import (
    SpatialTrackFormer,
    spatial_target,
)
from embedding_aware_belt_fusion.embeddings.spatial_trackformer_loss import (
    SpatialHungarianMatcher,
    SpatialTrackFormerCriterion,
    spatial_embedding_loss,
    spatial_hard_negative_loss,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--validation-cache-dir", type=Path, required=True)
    parser.add_argument("--scenario-split-file", type=Path, required=True)
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help=(
            "Optional propagated PQ-STF checkpoint. Omit for the clean, "
            "random-initialization stage; provide the previous JPA-STF base "
            "checkpoint for a subsequent annealing stage."
        ),
    )
    parser.add_argument("--initialize-pair-head", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trackformer-root", type=Path, default=Path("external/trackformer"))
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--feedforward-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--index-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--pair-head-learning-rate",
        type=float,
        help=(
            "Optional separate learning rate for the new pair head. "
            "Use a larger value than --learning-rate when training from scratch."
        ),
    )
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--candidate-distance", type=float, default=10.0)
    parser.add_argument("--match-probability", type=float, default=0.5)
    parser.add_argument("--hard-negative-ratio", type=int, default=4)
    parser.add_argument(
        "--embedding-regularizer-weight",
        type=float,
        default=0.25,
        help="Weight of the existing 128-D cross-agent embedding loss.",
    )
    parser.add_argument(
        "--hard-negative-regularizer-weight",
        type=float,
        default=0.10,
        help="Weight of the existing hard-negative embedding regularizer.",
    )
    parser.add_argument(
        "--detection-regularizer-weight",
        type=float,
        default=0.10,
        help=(
            "Weight of the original PointPillars-proposal classification and "
            "box-regression criterion. This anchors a random PQ-STF decoder "
            "to real object queries during joint training."
        ),
    )
    parser.add_argument("--hard-negative-radius", type=float, default=15.0)
    parser.add_argument("--hard-negative-size-difference", type=float, default=2.0)
    parser.add_argument("--hard-negative-margin", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument(
        "--selection-metric",
        choices=("pair-loss", "f1"),
        default="pair-loss",
        help=(
            "Validation criterion for best-checkpoint selection and early "
            "stopping. pair-loss is threshold-free; F1 depends on the "
            "temporary --match-probability gate."
        ),
    )
    parser.add_argument("--max-train-frames", type=int)
    parser.add_argument("--max-validation-frames", type=int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def _collate(frames):
    return frames


def _loader(dataset, args, *, shuffle):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        collate_fn=_collate,
        pin_memory=False,
        persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
    )


def _architecture(saved=None, args=None):
    arguments = saved["arguments"] if saved is not None else vars(args)
    return {
        name: arguments[name]
        for name in (
            "embedding_dim",
            "d_model",
            "heads",
            "encoder_layers",
            "decoder_layers",
            "feedforward_dim",
            "dropout",
        )
    }


def _build_base(saved, args, feature_dim, device):
    architecture = _architecture(saved)
    base = SpatialTrackFormer(
        trackformer_root=args.trackformer_root,
        input_dim=feature_dim,
        d_model=int(architecture["d_model"]),
        embedding_dim=int(architecture["embedding_dim"]),
        heads=int(architecture["heads"]),
        encoder_layers=int(architecture["encoder_layers"]),
        decoder_layers=int(architecture["decoder_layers"]),
        feedforward_dim=int(architecture["feedforward_dim"]),
        dropout=float(architecture["dropout"]),
        geometry_association=True,
    ).to(device)
    base.load_state_dict(saved["model_state_dict"], strict=True)

    return base, architecture


def _load_pair_head(path, state_dim, base_checkpoint, device):
    head = SymmetricQueryStatePairHead(state_dim).to(device)
    if path is None:
        return head
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("pipeline") != "calibrated_pq_query_state_pair_head_v2":
        raise ValueError("--initialize-pair-head is not a calibrated PQ pair head")
    if int(saved.get("state_dim", -1)) != state_dim:
        raise ValueError("--initialize-pair-head has an incompatible state dimension")
    if base_checkpoint is None:
        raise ValueError(
            "--initialize-pair-head requires --initialize-from; a pair head "
            "cannot be warm-started on a random PQ-STF encoder"
        )
    if saved.get("base_checkpoint") != str(base_checkpoint.resolve()):
        raise ValueError(
            "--initialize-pair-head was trained from a different "
            "--initialize-from PQ-STF checkpoint"
        )
    head.load_state_dict(saved["model_state_dict"], strict=True)
    print(f"Warm-started pair head from {path}")
    return head


def _accumulate(metrics, record):
    for key in ("true_positive", "false_positive", "false_negative", "pairs", "positive_pairs"):
        metrics[key] += int(record[key])
    metrics["losses"].append(float(record["loss"].detach()))


def _summarize(metrics):
    precision = metrics["true_positive"] / max(
        metrics["true_positive"] + metrics["false_positive"], 1
    )
    recall = metrics["true_positive"] / max(
        metrics["true_positive"] + metrics["false_negative"], 1
    )
    return {
        "pair_loss": mean(metrics["losses"]),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-8),
        "candidate_pairs": metrics["pairs"],
        "positive_pairs": metrics["positive_pairs"],
    }


def _frame_loss(base, head, matcher, criterion, frame, args):
    agents = [agent for agent in frame["agents"] if has_valid_proposals(agent)]
    if len(agents) < 2:
        return None
    outputs = {id(agent): base.forward_agent(agent) for agent in agents}
    pair_losses, embedding_losses, hard_losses, detection_losses = [], [], [], []
    metrics = {
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "pairs": 0,
        "positive_pairs": 0,
        "losses": [],
    }
    if args.detection_regularizer_weight:
        device = next(base.parameters()).device
        for agent in agents:
            detection_losses.append(
                criterion(outputs[id(agent)], spatial_target(agent, device))["loss"]
            )
    for agent_a, agent_b in combinations(agents, 2):
        output_a, output_b = outputs[id(agent_a)], outputs[id(agent_b)]
        record = binary_pair_loss_and_metrics(
            pair_head_scores(head, output_a, output_b),
            agent_a,
            agent_b,
            maximum_distance=args.candidate_distance,
            probability_threshold=args.match_probability,
            negative_ratio=args.hard_negative_ratio,
        )
        if not record["pairs"]:
            continue
        pair_losses.append(record["loss"])
        _accumulate(metrics, record)
        if args.embedding_regularizer_weight:
            value, _, _ = spatial_embedding_loss(
                output_a, agent_a, output_b, agent_b, temperature=args.temperature
            )
            embedding_losses.append(value)
        if args.hard_negative_regularizer_weight:
            value, _ = spatial_hard_negative_loss(
                output_a,
                agent_a,
                output_b,
                agent_b,
                radius=args.hard_negative_radius,
                size_difference=args.hard_negative_size_difference,
                margin=args.hard_negative_margin,
            )
            hard_losses.append(value)
    if not pair_losses:
        return None
    pair_loss = torch.stack(pair_losses).mean()
    embedding_loss = (
        torch.stack(embedding_losses).mean()
        if embedding_losses else pair_loss * 0
    )
    hard_loss = torch.stack(hard_losses).mean() if hard_losses else pair_loss * 0
    detection_loss = (
        torch.stack(detection_losses).mean()
        if detection_losses else pair_loss * 0
    )
    total = (
        pair_loss
        + args.embedding_regularizer_weight * embedding_loss
        + args.hard_negative_regularizer_weight * hard_loss
        + args.detection_regularizer_weight * detection_loss
    )
    metrics.update(
        tensor_loss=total,
        # Keep the pair BCE separately for optimization diagnostics.  The
        # total also contains the two optional representation regularizers.
        loss=pair_loss,
        embedding_loss=float(embedding_loss.detach()),
        hard_negative_loss=float(hard_loss.detach()),
        detection_loss=float(detection_loss.detach()),
    )
    return metrics


def _run_epoch(base, head, loader, matcher, criterion, optimizer, args, device, label):
    training = optimizer is not None
    base.train(training)
    head.train(training)
    autocast = (
        lambda: torch.autocast("cuda", dtype=torch.bfloat16)
        if args.amp and device.type == "cuda"
        else nullcontext()
    )
    total = {
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "pairs": 0,
        "positive_pairs": 0,
        "losses": [],
        "embedding_losses": [],
        "hard_losses": [],
        "detection_losses": [],
    }
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        progress = tqdm(loader, desc=label)
        for frames in progress:
            if training:
                optimizer.zero_grad(set_to_none=True)
            usable = 0
            for frame in frames:
                with autocast():
                    record = _frame_loss(base, head, matcher, criterion, frame, args)
                if record is None:
                    continue
                usable += 1
                if training:
                    # Frame-by-frame backward keeps a requested batch size of
                    # 32 practical: this is gradient accumulation over frames,
                    # followed by exactly one optimizer step per DataLoader batch.
                    (record["tensor_loss"] / len(frames)).backward()
                _accumulate(total, record)
                total["embedding_losses"].append(record["embedding_loss"])
                total["hard_losses"].append(record["hard_negative_loss"])
                total["detection_losses"].append(record["detection_loss"])
            if training and usable:
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for module in (base, head)
                        for parameter in module.parameters()
                        if parameter.requires_grad
                    ],
                    1.0,
                )
                optimizer.step()
            if total["losses"]:
                precision = total["true_positive"] / max(
                    total["true_positive"] + total["false_positive"], 1
                )
                recall = total["true_positive"] / max(
                    total["true_positive"] + total["false_negative"], 1
                )
                progress.set_postfix(
                    loss=f"{total['losses'][-1]:.3f}",
                    precision=f"{precision:.3f}",
                    recall=f"{recall:.3f}",
                )
    if not total["losses"]:
        raise RuntimeError("No geometrically valid all-agent proposal pairs were found")
    result = _summarize(total)
    result["embedding_regularizer"] = mean(total["embedding_losses"])
    result["hard_negative_regularizer"] = mean(total["hard_losses"])
    result["detection_regularizer"] = mean(total["detection_losses"])
    return result


def _base_checkpoint(
    epoch, base, optimizer, scheduler, architecture, args, split, metrics
):
    return {
        "epoch": epoch,
        "model_state_dict": base.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "arguments": architecture,
        "split": split,
        "metrics": metrics,
        "pipeline": "spatial_trackformer_v2_geometry_association",
        "association_protocol": "propagated",
        "joint_association_training": True,
    }


def _pair_checkpoint(epoch, head, base_path, state_dim, args, split, metrics):
    return {
        "epoch": epoch,
        "model_state_dict": head.state_dict(),
        "base_checkpoint": str(base_path.resolve()),
        "state_dim": state_dim,
        "arguments": vars(args),
        "split": split,
        "metrics": metrics,
        "pipeline": "calibrated_pq_query_state_pair_head_v2",
        "joint_association_training": True,
    }


def main():
    args = parse_args()
    if args.workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    initial = None
    if args.initialize_from is not None:
        initial = torch.load(
            args.initialize_from, map_location="cpu", weights_only=False
        )
        if initial.get("pipeline") != "spatial_trackformer_v2_geometry_association":
            raise ValueError("--initialize-from must be a propagated PQ-STF checkpoint")
        if initial.get("association_protocol", "propagated") != "propagated":
            raise ValueError("JPA-STF requires a propagated PQ-STF checkpoint")

    train_records = records_for_role(
        args.cache_dir, args.scenario_split_file, "train", args.index_workers
    )
    validation_records = records_for_role(
        args.validation_cache_dir,
        args.scenario_split_file,
        "validation",
        args.index_workers,
    )
    if args.max_train_frames is not None:
        train_records = train_records[: args.max_train_frames]
    if args.max_validation_frames is not None:
        validation_records = validation_records[: args.max_validation_frames]
    train_data = IndexedCompleteFrameDataset(args.cache_dir, train_records)
    validation_data = IndexedCompleteFrameDataset(
        args.validation_cache_dir, validation_records
    )
    if train_data.manifest["feature_dim"] != validation_data.manifest["feature_dim"]:
        raise ValueError("Training and validation caches have different feature dimensions")

    architecture = _architecture(initial, args)
    if initial is None:
        base = SpatialTrackFormer(
            trackformer_root=args.trackformer_root,
            input_dim=int(train_data.manifest["feature_dim"]),
            d_model=int(architecture["d_model"]),
            embedding_dim=int(architecture["embedding_dim"]),
            heads=int(architecture["heads"]),
            encoder_layers=int(architecture["encoder_layers"]),
            decoder_layers=int(architecture["decoder_layers"]),
            feedforward_dim=int(architecture["feedforward_dim"]),
            dropout=float(architecture["dropout"]),
            geometry_association=True,
        ).to(device)
        print("Starting JPA-STF from random initialization")
    else:
        base, architecture = _build_base(
            initial, args, int(train_data.manifest["feature_dim"]), device
        )
    head = _load_pair_head(
        args.initialize_pair_head,
        int(architecture["d_model"]),
        args.initialize_from,
        device,
    )
    pair_head_learning_rate = args.pair_head_learning_rate or args.learning_rate
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    parameter for parameter in base.parameters()
                    if parameter.requires_grad
                ],
                "lr": args.learning_rate,
            },
            {"params": head.parameters(), "lr": pair_head_learning_rate},
        ],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.minimum_learning_rate
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = {
        "scenario_split_file": str(args.scenario_split_file.resolve()),
        "train_cache_dir": str(args.cache_dir.resolve()),
        "validation_cache_dir": str(args.validation_cache_dir.resolve()),
        "train_frames": len(train_data),
        "validation_frames": len(validation_data),
    }
    (args.output_dir / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    train_loader = _loader(train_data, args, shuffle=True)
    validation_loader = _loader(validation_data, args, shuffle=False)
    matcher = SpatialHungarianMatcher()
    criterion = SpatialTrackFormerCriterion(matcher)
    print(
        f"JPA-STF: {len(train_data)} train frames, {len(validation_data)} "
        f"validation frames; full PQ query-state path + pair head are trainable"
    )

    best = float("inf") if args.selection_metric == "pair-loss" else -1.0
    stale, history = 0, []
    best_base_path = args.output_dir / "spatial_trackformer_best.pth"
    best_head_path = args.output_dir / "all_agent_pair_head_best.pth"
    for epoch in range(1, args.epochs + 1):
        train = _run_epoch(
            base, head, train_loader, matcher, criterion, optimizer, args, device,
            f"training {epoch}/{args.epochs}",
        )
        validation = _run_epoch(
            base, head, validation_loader, matcher, criterion, None, args, device,
            f"validation {epoch}/{args.epochs}",
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "pair_head_learning_rate": optimizer.param_groups[1]["lr"],
            **{f"train_{key}": value for key, value in train.items()},
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True))
        base_saved = _base_checkpoint(
            epoch, base, optimizer, scheduler, architecture, args, split, record
        )
        latest_base_path = args.output_dir / "spatial_trackformer_latest.pth"
        torch.save(base_saved, latest_base_path)
        torch.save(
            _pair_checkpoint(epoch, head, latest_base_path, int(architecture["d_model"]), args, split, record),
            args.output_dir / "all_agent_pair_head_latest.pth",
        )
        selection_value = (
            validation["pair_loss"]
            if args.selection_metric == "pair-loss"
            else validation["f1"]
        )
        improved = (
            selection_value < best
            if args.selection_metric == "pair-loss"
            else selection_value > best
        )
        if improved:
            best, stale = selection_value, 0
            torch.save(base_saved, best_base_path)
            torch.save(
                _pair_checkpoint(epoch, head, best_base_path, int(architecture["d_model"]), args, split, record),
                best_head_path,
            )
        else:
            stale += 1
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if args.early_stopping_patience and stale >= args.early_stopping_patience:
            print(f"Early stopping on validation {args.selection_metric}")
            break
    print(json.dumps({
        "selection_metric": args.selection_metric,
        "best_validation_value": best,
        "epochs": history[-1]["epoch"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
