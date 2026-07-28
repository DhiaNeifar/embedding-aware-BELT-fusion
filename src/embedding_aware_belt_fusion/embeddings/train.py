"""Train compact cross-agent proposal embeddings from an extracted cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from embedding_aware_belt_fusion.data.proposal_cache import (
    CrossAgentProposalDataset,
    collate_proposal_frames,
)
from embedding_aware_belt_fusion.embeddings.model import (
    CrossAgentSupervisedContrastiveLoss,
    ProposalEmbeddingHead,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16, help="Frames per batch")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def _retrieval_metrics(embeddings, labels, agents, groups):
    similarity = embeddings @ embeddings.T
    eligible = (
        (groups[:, None] == groups[None, :])
        & (agents[:, None] != agents[None, :])
    )
    positive = (labels[:, None] == labels[None, :]) & eligible
    negative = (labels[:, None] != labels[None, :]) & eligible
    valid_anchor = positive.any(dim=1)
    similarity = similarity.masked_fill(~eligible, float("-inf"))
    nearest = similarity.argmax(dim=1)
    correct = labels[nearest] == labels
    random_accuracy = (
        positive.sum(dim=1).float()
        / eligible.sum(dim=1).clamp_min(1).float()
    )
    return {
        "top1_cross_agent_accuracy": float(
            correct[valid_anchor].float().mean()
        ),
        "random_top1_accuracy": float(random_accuracy[valid_anchor].mean()),
        "positive_cosine": float(
            similarity.masked_select(positive).mean()
        ),
        "negative_cosine": float(
            similarity.masked_select(negative).mean()
        ),
    }


def _validate(model, criterion, loader, device):
    model.eval()
    losses = []
    metrics = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            embeddings = model(batch["features"])
            losses.append(
                float(
                    criterion(
                        embeddings, batch["labels"], batch["agents"]
                    ).item()
                )
            )
            metrics.append(
                _retrieval_metrics(
                    embeddings,
                    batch["labels"],
                    batch["agents"],
                    batch["groups"],
                )
            )
    result = {"loss": mean(losses)}
    for key in metrics[0]:
        result[key] = mean(item[key] for item in metrics)
    return result


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)

    train_dataset = CrossAgentProposalDataset(
        args.cache_dir,
        split="train",
        validation_fraction=args.validation_fraction,
    )
    validation_dataset = CrossAgentProposalDataset(
        args.cache_dir,
        split="validation",
        validation_fraction=args.validation_fraction,
    )
    if not train_dataset or not validation_dataset:
        raise RuntimeError("Cache does not contain enough eligible scenarios")
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "collate_fn": collate_proposal_frames,
        "pin_memory": args.device == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_args)

    input_dim = int(train_dataset.manifest["feature_dim"])
    model = ProposalEmbeddingHead(
        input_dim=input_dim, embedding_dim=args.embedding_dim
    ).to(device)
    criterion = CrossAgentSupervisedContrastiveLoss(args.temperature)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"embedding epoch {epoch + 1}/{args.epochs}")
        train_losses = []
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            embeddings = model(batch["features"])
            loss = criterion(embeddings, batch["labels"], batch["agents"])
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            progress.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        validation = _validate(model, criterion, validation_loader, device)
        record = {
            "epoch": epoch + 1,
            "train_loss": mean(train_losses),
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True))
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "embedding_dim": args.embedding_dim,
                "temperature": args.temperature,
            },
            args.output_dir / f"embedding_epoch{epoch + 1}.pth",
        )
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
