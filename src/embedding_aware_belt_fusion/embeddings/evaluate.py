"""Evaluate saved proposal-embedding checkpoints with frame-local retrieval."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from embedding_aware_belt_fusion.data.proposal_cache import (
    CrossAgentProposalDataset,
    collate_proposal_frames,
)
from embedding_aware_belt_fusion.embeddings.model import (
    CrossAgentSupervisedContrastiveLoss,
    ProposalEmbeddingHead,
)
from embedding_aware_belt_fusion.embeddings.train import _validate


def _epoch(path: Path) -> int:
    match = re.search(r"epoch(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Checkpoint name does not contain an epoch: {path}")
    return int(match.group(1))


def _save_figure(results, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = [item["epoch"] for item in results]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [item["loss"] for item in results])
    axes[0].set(title="Validation loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(
        epochs,
        [item["top1_cross_agent_accuracy"] for item in results],
        label="Embedding",
    )
    axes[1].plot(
        epochs,
        [item["random_top1_accuracy"] for item in results],
        linestyle="--",
        label="Random",
    )
    axes[1].set(
        title="Frame-local cross-agent retrieval",
        xlabel="Epoch",
        ylabel="Top-1 accuracy",
    )
    axes[1].legend()
    axes[2].plot(
        epochs,
        [item["positive_cosine"] for item in results],
        label="Same object",
    )
    axes[2].plot(
        epochs,
        [item["negative_cosine"] for item in results],
        label="Different object",
    )
    axes[2].set(title="Cosine similarity", xlabel="Epoch", ylabel="Cosine")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = CrossAgentProposalDataset(
        args.cache_dir,
        split="validation",
        validation_fraction=args.validation_fraction,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_proposal_frames,
        pin_memory=args.device == "cuda",
        persistent_workers=args.workers > 0,
    )
    checkpoints = sorted(args.model_dir.glob("embedding_epoch*.pth"), key=_epoch)
    if not checkpoints:
        raise RuntimeError(f"No embedding checkpoints found in {args.model_dir}")

    results = []
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = ProposalEmbeddingHead(
            input_dim=int(checkpoint["input_dim"]),
            embedding_dim=int(checkpoint["embedding_dim"]),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        criterion = CrossAgentSupervisedContrastiveLoss(
            float(checkpoint["temperature"])
        )
        metrics = _validate(model, criterion, loader, device)
        record = {"epoch": _epoch(path), **metrics}
        results.append(record)
        print(json.dumps(record, sort_keys=True))

    best = max(results, key=lambda item: item["top1_cross_agent_accuracy"])
    (args.model_dir / "evaluation_frame_local.json").write_text(
        json.dumps({"best": best, "epochs": results}, indent=2) + "\n"
    )
    _save_figure(results, args.model_dir / "evaluation_frame_local.png")
    print("best:", json.dumps(best, sort_keys=True))


if __name__ == "__main__":
    main()
