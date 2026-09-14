"""Train a symmetric PQ query-state head for source-to-source association."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from statistics import mean

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from embedding_aware_belt_fusion.data.complete_cache import IndexedCompleteFrameDataset
from embedding_aware_belt_fusion.embeddings.all_agent_association import (
    SymmetricQueryStatePairHead,
    all_agent_pairs,
    binary_pair_loss_and_metrics,
    load_pq_stf,
    pair_head_scores,
    records_for_role,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--validation-cache-dir",
        type=Path,
        help=(
            "Optional separately materialized validation cache. Use this for "
            "a noisy training curriculum whose train and validation scenarios "
            "were cached independently."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help=(
            "Optional calibrated pair-head checkpoint to warm-start. Its "
            "PQ-STF parent checkpoint must equal --checkpoint."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trackformer-root", type=Path, default=Path("external/trackformer"))
    parser.add_argument("--scenario-split-file", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--index-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--candidate-distance", type=float, default=5.0)
    parser.add_argument("--match-probability", type=float, default=0.90)
    parser.add_argument("--hard-negative-ratio", type=int, default=4)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--max-train-frames", type=int)
    parser.add_argument("--max-validation-frames", type=int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def _collate(frames):
    return frames


def _limit(records, maximum):
    return records if maximum is None else records[:maximum]


def _run_epoch(base, head, loader, optimizer, args, device, label):
    training = optimizer is not None
    base.eval()
    head.train(training)
    autocast = (
        lambda: torch.autocast("cuda", dtype=torch.bfloat16)
        if args.amp and device.type == "cuda"
        else nullcontext()
    )
    total_loss, true_positive, false_positive, false_negative = [], 0, 0, 0
    candidate_pairs = positive_pairs = agent_pairs = 0
    progress = tqdm(loader, desc=label)
    for frames in progress:
        if training:
            optimizer.zero_grad(set_to_none=True)
        losses = []
        with torch.no_grad():
            encoded = []
            for frame in frames:
                frame_pairs = []
                for agent_a, agent_b in all_agent_pairs(frame):
                    with autocast():
                        output_a = base.forward_agent(agent_a)
                        output_b = base.forward_agent(agent_b)
                    frame_pairs.append((agent_a, output_a, agent_b, output_b))
                encoded.extend(frame_pairs)
        with autocast():
            for agent_a, output_a, agent_b, output_b in encoded:
                scores = pair_head_scores(head, output_a, output_b)
                record = binary_pair_loss_and_metrics(
                    scores,
                    agent_a,
                    agent_b,
                    maximum_distance=args.candidate_distance,
                    probability_threshold=args.match_probability,
                    negative_ratio=args.hard_negative_ratio,
                )
                if record["pairs"]:
                    losses.append(record["loss"])
                    true_positive += record["true_positive"]
                    false_positive += record["false_positive"]
                    false_negative += record["false_negative"]
                    candidate_pairs += record["pairs"]
                    positive_pairs += record["positive_pairs"]
                    agent_pairs += 1
        if not losses:
            continue
        loss = torch.stack(losses).mean()
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
        total_loss.append(float(loss.detach()))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        progress.set_postfix(
            loss=f"{total_loss[-1]:.3f}",
            precision=f"{precision:.3f}",
            recall=f"{recall:.3f}",
        )
    if not candidate_pairs:
        raise RuntimeError("No geometrically valid CAV proposal pairs were found")
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "loss": mean(total_loss),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "candidate_pairs": candidate_pairs,
        "positive_pairs": positive_pairs,
        "agent_pairs": agent_pairs,
    }


def main():
    args = parse_args()
    if args.workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_records = _limit(
        records_for_role(args.cache_dir, args.scenario_split_file, "train", args.index_workers),
        args.max_train_frames,
    )
    validation_cache_dir = args.validation_cache_dir or args.cache_dir
    validation_records = _limit(
        records_for_role(
            validation_cache_dir,
            args.scenario_split_file,
            "validation",
            args.index_workers,
        ),
        args.max_validation_frames,
    )
    train_data = IndexedCompleteFrameDataset(args.cache_dir, train_records)
    validation_data = IndexedCompleteFrameDataset(
        validation_cache_dir, validation_records
    )
    options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "collate_fn": _collate,
        "pin_memory": False,
        "persistent_workers": args.workers > 0,
        "prefetch_factor": args.prefetch_factor if args.workers > 0 else None,
    }
    train_loader = DataLoader(train_data, shuffle=True, **options)
    validation_loader = DataLoader(validation_data, shuffle=False, **options)
    base, saved = load_pq_stf(
        args.checkpoint,
        args.trackformer_root,
        int(train_data.manifest["feature_dim"]),
        device,
    )
    state_dim = int(saved["arguments"]["d_model"])
    head = SymmetricQueryStatePairHead(state_dim).to(device)
    if args.initialize_from is not None:
        initial = torch.load(
            args.initialize_from, map_location="cpu", weights_only=False
        )
        if initial.get("pipeline") != "calibrated_pq_query_state_pair_head_v2":
            raise ValueError(
                "--initialize-from must be a calibrated same-object/no-match "
                "pair-head checkpoint"
            )
        if initial.get("base_checkpoint") != str(args.checkpoint.resolve()):
            raise ValueError(
                "--initialize-from was trained from a different PQ-STF "
                "--checkpoint"
            )
        if int(initial.get("state_dim", -1)) != state_dim:
            raise ValueError("--initialize-from has an incompatible state dimension")
        head.load_state_dict(initial["model_state_dict"], strict=True)
        print(f"Warm-started calibrated pair head from {args.initialize_from}")
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate / 100
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = {
        "scenario_split_file": str(args.scenario_split_file.resolve()),
        "validation_cache_dir": str(validation_cache_dir.resolve()),
        "train_frames": len(train_data),
        "validation_frames": len(validation_data),
    }
    (args.output_dir / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    print(
        f"All-agent calibrated PQ pair head: {len(train_data)} train frames, "
        f"{len(validation_data)} validation frames; frozen backbone={args.checkpoint}"
    )
    best, stale, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        train = _run_epoch(base, head, train_loader, optimizer, args, device, f"training {epoch}/{args.epochs}")
        validation = _run_epoch(base, head, validation_loader, None, args, device, f"validation {epoch}/{args.epochs}")
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train.items()},
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True))
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": head.state_dict(),
            "base_checkpoint": str(args.checkpoint.resolve()),
            "state_dim": state_dim,
            "arguments": vars(args),
            "split": split,
            "metrics": record,
            "pipeline": "calibrated_pq_query_state_pair_head_v2",
        }
        torch.save(checkpoint, args.output_dir / "all_agent_pair_head_latest.pth")
        if validation["f1"] > best:
            best, stale = validation["f1"], 0
            torch.save(checkpoint, args.output_dir / "all_agent_pair_head_best.pth")
        else:
            stale += 1
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if args.early_stopping_patience and stale >= args.early_stopping_patience:
            print("Early stopping on validation same-object/no-match F1")
            break
    print(json.dumps({"best_validation_f1": best, "epochs": history[-1]["epoch"]}, sort_keys=True))


if __name__ == "__main__":
    main()
