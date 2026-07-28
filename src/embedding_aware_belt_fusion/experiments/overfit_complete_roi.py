"""Overfit cross-agent identity association using every proposal ROI cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from embedding_aware_belt_fusion.data.complete_cache import CompleteFrameDataset
from embedding_aware_belt_fusion.embeddings.trackformer import _load_transformer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--trackformer-root",
        type=Path,
        default=Path("external/trackformer"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--feedforward-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--variance-weight", type=float, default=5.0)
    parser.add_argument("--variance-target", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--collapse-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


class CompleteROIEncoder(nn.Module):
    """Encode all cells in a proposal with self- and cross-attention."""

    def __init__(
        self,
        *,
        trackformer_root,
        input_dim,
        d_model,
        embedding_dim,
        heads,
        encoder_layers,
        decoder_layers,
        feedforward_dim,
        dropout,
    ):
        super().__init__()
        self.token_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
        )
        self.relative_position = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        Transformer = _load_transformer(trackformer_root)
        self.transformer = Transformer(
            d_model=d_model,
            nhead=heads,
            num_encoder_layers=encoder_layers,
            num_decoder_layers=decoder_layers,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            return_intermediate_dec=True,
        )
        self.object_query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.embedding = nn.Linear(d_model, embedding_dim)

    def forward(self, tokens, positions, padding_mask):
        memory = self.token_projection(tokens)
        source = memory.transpose(1, 2).unsqueeze(2)
        position = self.relative_position(positions).transpose(1, 2).unsqueeze(2)
        mask = padding_mask.unsqueeze(1)
        query = self.object_query[:, None, :].expand(-1, len(tokens), -1)
        states, _, _ = self.transformer(
            source,
            mask,
            query,
            position,
            torch.zeros_like(query),
        )
        state = states[-1, :, 0]
        return F.normalize(self.embedding(state), dim=-1)


def _best_proposals(agent):
    best = {}
    for proposal_index, object_id in enumerate(agent["gt_ids"]):
        if object_id is None:
            continue
        object_id = str(object_id)
        iou = float(agent["gt_ious"][proposal_index])
        if object_id not in best or iou > best[object_id][0]:
            best[object_id] = (iou, proposal_index)
    return {object_id: item[1] for object_id, item in best.items()}


def _agent_rois(agent, proposal_by_id, *, height, width, device):
    bev = agent["bev_features"].to(dtype=torch.float32)
    flattened = bev.flatten(1).transpose(0, 1)
    sequences, positions = [], []
    for object_id in sorted(proposal_by_id):
        proposal_index = proposal_by_id[object_id]
        indices = agent["proposal_roi_cell_indices"][proposal_index].long()
        sequences.append(flattened[indices])
        rows = torch.div(indices, width, rounding_mode="floor").float()
        columns = (indices % width).float()
        coordinates = torch.stack([rows, columns], dim=-1)
        # Relative coordinates retain the layout within the vehicle while
        # preventing absolute location from becoming an identity shortcut.
        coordinates = coordinates - coordinates.mean(dim=0, keepdim=True)
        scale = coordinates.abs().amax(dim=0, keepdim=True).clamp_min(1.0)
        positions.append(coordinates / scale)
    maximum = max(len(item) for item in sequences)
    token_batch = torch.zeros(
        len(sequences),
        maximum,
        sequences[0].shape[-1],
    )
    position_batch = torch.zeros(
        len(sequences), maximum, 2
    )
    padding_mask = torch.ones(
        len(sequences), maximum, dtype=torch.bool
    )
    for index, (tokens, coordinates) in enumerate(zip(sequences, positions)):
        length = len(tokens)
        token_batch[index, :length] = tokens
        position_batch[index, :length] = coordinates
        padding_mask[index, :length] = False
    return (
        token_batch.to(device),
        position_batch.to(device),
        padding_mask.to(device),
        sorted(proposal_by_id),
    )


def _agent_embeddings(model, agent, *, height, width, device):
    proposals = _best_proposals(agent)
    if not proposals:
        return None
    tokens, positions, mask, object_ids = _agent_rois(
        agent,
        proposals,
        height=height,
        width=width,
        device=device,
    )
    return model(tokens, positions, mask), object_ids


def _pair_loss(
    source,
    target,
    temperature,
    *,
    variance_weight,
    variance_target,
):
    source_embeddings, source_ids = source
    target_embeddings, target_ids = target
    target_by_id = {object_id: index for index, object_id in enumerate(target_ids)}
    source_by_id = {object_id: index for index, object_id in enumerate(source_ids)}
    shared = sorted(set(source_ids).intersection(target_ids))
    if len(shared) < 2:
        return None
    source_indices = torch.tensor(
        [source_by_id[object_id] for object_id in shared],
        device=source_embeddings.device,
    )
    target_indices = torch.tensor(
        [target_by_id[object_id] for object_id in shared],
        device=target_embeddings.device,
    )
    forward_logits = (
        source_embeddings[source_indices] @ target_embeddings.T
    ) / temperature
    reverse_logits = (
        target_embeddings[target_indices] @ source_embeddings.T
    ) / temperature
    association = (
        F.cross_entropy(forward_logits, target_indices)
        + F.cross_entropy(reverse_logits, source_indices)
    ) / 2
    embedding_batch = torch.cat(
        [source_embeddings, target_embeddings], dim=0
    )
    embedding_std = torch.sqrt(
        embedding_batch.var(dim=0, unbiased=False) + 1e-4
    )
    variance = F.relu(variance_target - embedding_std).mean()
    loss = association + variance_weight * variance
    forward_top1 = (
        forward_logits.argmax(dim=1) == target_indices
    ).float()
    reverse_top1 = (
        reverse_logits.argmax(dim=1) == source_indices
    ).float()
    positive = source_embeddings[source_indices] @ target_embeddings[target_indices].T
    positive = positive.diagonal()
    all_similarity = source_embeddings[source_indices] @ target_embeddings.T
    negative_mask = torch.ones_like(all_similarity, dtype=torch.bool)
    negative_mask[
        torch.arange(len(source_indices), device=source_indices.device),
        target_indices,
    ] = False
    return {
        "loss": loss,
        "association": association,
        "variance": variance,
        "top1": torch.cat([forward_top1, reverse_top1]).mean(),
        "positive_cosine": positive.mean(),
        "negative_cosine": all_similarity[negative_mask].mean(),
        "random_top1": (
            0.5 / len(target_embeddings) + 0.5 / len(source_embeddings)
        ),
        "shared_objects": len(shared),
    }


def _frame_records(
    model,
    frame,
    manifest,
    device,
    temperature,
    *,
    variance_weight,
    variance_target,
):
    height, width = manifest["bev_shape"][1:]
    encoded = [
        _agent_embeddings(
            model,
            agent,
            height=height,
            width=width,
            device=device,
        )
        for agent in frame["agents"]
    ]
    records = []
    for source_index in range(len(encoded)):
        if encoded[source_index] is None:
            continue
        for target_index in range(source_index + 1, len(encoded)):
            if encoded[target_index] is None:
                continue
            record = _pair_loss(
                encoded[source_index],
                encoded[target_index],
                temperature,
                variance_weight=variance_weight,
                variance_target=variance_target,
            )
            if record is not None:
                records.append(record)
    return records


def _run_epoch(
    model,
    loader,
    manifest,
    device,
    temperature,
    *,
    variance_weight,
    variance_target,
    optimizer=None,
    description=None,
):
    training = optimizer is not None
    model.train(training)
    records = []
    context = torch.enable_grad() if training else torch.no_grad()
    iterator = (
        tqdm(loader, desc=description) if description is not None else loader
    )
    with context:
        for frame in iterator:
            if training:
                optimizer.zero_grad(set_to_none=True)
            frame_records = _frame_records(
                model,
                frame,
                manifest,
                device,
                temperature,
                variance_weight=variance_weight,
                variance_target=variance_target,
            )
            if not frame_records:
                continue
            loss = torch.stack([item["loss"] for item in frame_records]).mean()
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if description is not None:
                iterator.set_postfix(
                    loss=f"{float(loss.detach()):.4f}",
                    top1=f"{mean(float(item['top1']) for item in frame_records):.3f}",
                )
            for item in frame_records:
                records.append(
                    {
                        key: float(value.detach())
                        if isinstance(value, torch.Tensor)
                        else float(value)
                        for key, value in item.items()
                    }
                )
    if not records:
        raise RuntimeError(
            "No frame contained at least two identities shared by two agents"
        )
    return {
        key: mean(item[key] for item in records)
        for key in records[0]
    }


def _save_figure(history, path):
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epochs = [item["epoch"] for item in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [item["train_loss"] for item in history])
    axes[0].set_title("Association loss")
    axes[1].plot(
        epochs,
        [item["train_top1"] for item in history],
        label="Top-1",
    )
    axes[1].plot(
        epochs,
        [item["train_random_top1"] for item in history],
        "--",
        label="Random",
    )
    axes[1].set_title("Cross-agent identity")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()
    axes[2].plot(
        epochs,
        [item["train_positive_cosine"] for item in history],
        label="Same object",
    )
    axes[2].plot(
        epochs,
        [item["train_negative_cosine"] for item in history],
        label="Different object",
    )
    axes[2].set_title("Embedding similarity")
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    if args.max_frames < 1:
        raise ValueError("--max-frames must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = CompleteFrameDataset(args.cache_dir)
    frame_count = min(args.max_frames, len(dataset))
    dataset = Subset(dataset, range(frame_count))
    loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        persistent_workers=args.workers > 0,
    )
    manifest = dataset.dataset.manifest
    model = CompleteROIEncoder(
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
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    print(
        f"OVERFIT SANITY TEST: {frame_count} schema-v3 frames; "
        "training and evaluation use the same frames"
    )
    progress = tqdm(range(1, args.epochs + 1), desc="complete-ROI overfit")
    for epoch in progress:
        metrics = _run_epoch(
            model,
            loader,
            manifest,
            device,
            args.temperature,
            variance_weight=args.variance_weight,
            variance_target=args.variance_target,
            optimizer=optimizer,
        )
        record = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in metrics.items()},
        }
        history.append(record)
        progress.set_postfix(
            loss=f"{metrics['loss']:.4f}",
            top1=f"{metrics['top1']:.3f}",
        )
        print(json.dumps(record, sort_keys=True))
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "arguments": vars(args),
            "manifest": manifest,
            "metrics": record,
        }
        torch.save(checkpoint, args.output_dir / "roi_overfit_latest.pth")
        if metrics["top1"] >= 0.99:
            torch.save(checkpoint, args.output_dir / "roi_overfit_success.pth")
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2, default=str) + "\n"
        )
        _save_figure(history, args.output_dir / "training_metrics.png")
        if len(history) >= args.collapse_patience:
            recent = history[-args.collapse_patience :]
            collapsed = all(
                abs(
                    item["train_positive_cosine"]
                    - item["train_negative_cosine"]
                )
                < 1e-4
                and item["train_variance"]
                >= args.variance_target * 0.75
                for item in recent
            )
            if collapsed:
                raise RuntimeError(
                    "Embedding collapse persisted for "
                    f"{args.collapse_patience} epochs; stopped early"
                )
    final_metrics = _run_epoch(
        model,
        loader,
        manifest,
        device,
        args.temperature,
        variance_weight=args.variance_weight,
        variance_target=args.variance_target,
    )
    result = {
        "passed": final_metrics["top1"] >= 0.95,
        "criterion": (
            f"top1 >= 0.95 on the same {frame_count} overfit frame(s)"
        ),
        "final": final_metrics,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("FINAL " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
