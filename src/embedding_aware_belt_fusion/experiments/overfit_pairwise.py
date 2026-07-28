"""Overfit learned matchers on one real two-object cross-agent assignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from embedding_aware_belt_fusion.data.trackformer_cache import (
    TrackFormerProposalDataset,
    collate_trackformer_frames,
)
from embedding_aware_belt_fusion.embeddings.trackformer import PointPillarsTrackFormer
from embedding_aware_belt_fusion.embeddings.train_trackformer import (
    _group_agents,
    _to_device,
)


class PairwiseMatcher(nn.Module):
    def __init__(self, state_dim: int, geometry_dim: int = 0):
        super().__init__()
        input_dim = state_dim * 4 + geometry_dim
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, source, target, geometry=None):
        source = source[:, None, :].expand(-1, len(target), -1)
        target = target[None, :, :].expand(len(source), -1, -1)
        descriptor = [source, target, (source - target).abs(), source * target]
        if geometry is not None:
            descriptor.append(geometry)
        return self.network(torch.cat(descriptor, dim=-1)).squeeze(-1)


class GeometryMatcher(nn.Module):
    def __init__(self, geometry_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(geometry_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, geometry):
        return self.network(geometry).squeeze(-1)


def _subset(agent, indices):
    keys = {"query_features", "boxes", "target_boxes", "scores", "labels"}
    return {
        key: value[indices] if key in keys else value
        for key, value in agent.items()
    }


def _load_two_objects(cache_dir, checkpoint_path, trackformer_root, device):
    dataset = TrackFormerProposalDataset(
        cache_dir, split="train", validation_fraction=0.2
    )
    agents = collate_trackformer_frames([dataset[0]])
    source, target = _group_agents(agents, shuffle=False)[0]
    source, target = _to_device(source, device), _to_device(target, device)
    target_by_label = {
        int(label): index for index, label in enumerate(target["labels"].tolist())
    }
    shared = [
        (source_index, target_by_label[int(label)], int(label))
        for source_index, label in enumerate(source["labels"].tolist())
        if int(label) in target_by_label
    ][:2]
    if len(shared) != 2:
        raise RuntimeError("The selected frame does not contain two shared objects")
    source_indices = torch.tensor(
        [item[0] for item in shared], device=device
    )
    target_indices = torch.tensor(
        [item[1] for item in shared], device=device
    )
    source, target = _subset(source, source_indices), _subset(target, target_indices)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    saved = checkpoint["arguments"]
    model = PointPillarsTrackFormer(
        trackformer_root,
        input_channels=int(checkpoint["bev_shape"][0]),
        d_model=int(saved["d_model"]),
        embedding_dim=int(saved["embedding_dim"]),
        nhead=int(saved["heads"]),
        encoder_layers=int(saved["encoder_layers"]),
        decoder_layers=int(saved["decoder_layers"]),
        feedforward_dim=int(saved["feedforward_dim"]),
        dropout=float(saved["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        source_states = model.forward_agent_states(
            source["bev"],
            source["query_features"],
            source["boxes"],
            source["scores"],
        )
        output = model.forward_with_track_queries(
            target["bev"],
            target["query_features"],
            target["boxes"],
            target["scores"],
            source_states,
        )
    track_states = output["states"][:2].detach()
    object_states = output["states"][2:].detach()
    source_size = source["boxes"][:, 3:6]
    target_size = target["boxes"][:, 3:6]
    geometry = (
        source_size[:, None, :] - target_size[None, :, :]
    ).abs()
    return track_states, object_states, geometry, [item[2] for item in shared]


def _metrics(logits, labels):
    predictions = (logits > 0).to(labels.dtype)
    return {
        "binary_accuracy": float((predictions == labels).float().mean()),
        "row_top1_accuracy": float(
            (logits.argmax(dim=1) == torch.arange(2, device=logits.device))
            .float()
            .mean()
        ),
        "positive_probability": float(logits.sigmoid().diagonal().mean()),
        "negative_probability": float(
            logits.sigmoid()[~torch.eye(2, dtype=torch.bool, device=logits.device)]
            .mean()
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trackformer-root", type=Path, default=Path("external/trackformer"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    track_states, object_states, geometry, object_ids = _load_two_objects(
        args.cache_dir, args.checkpoint, args.trackformer_root, device
    )
    labels = torch.eye(2, device=device)
    matchers = {
        "states": PairwiseMatcher(track_states.shape[-1]).to(device),
        "geometry": GeometryMatcher(geometry.shape[-1]).to(device),
        "states_geometry": PairwiseMatcher(
            track_states.shape[-1], geometry.shape[-1]
        ).to(device),
    }
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        for name, model in matchers.items()
    }
    history = []
    for step in range(args.steps + 1):
        record = {"step": step}
        for name, matcher in matchers.items():
            logits = (
                matcher(geometry)
                if name == "geometry"
                else matcher(
                    track_states,
                    object_states,
                    geometry if name == "states_geometry" else None,
                )
            )
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            if step < args.steps:
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[name].step()
            record[f"{name}_loss"] = float(loss)
            record.update(
                {
                    f"{name}_{key}": value
                    for key, value in _metrics(logits.detach(), labels).items()
                }
            )
        history.append(record)
        if step % 50 == 0 or step == args.steps:
            print(json.dumps(record, sort_keys=True))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "object_ids": object_ids,
        "checkpoint": str(args.checkpoint),
        "final": history[-1],
        "history": history,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    torch.save(
        {
            name: matcher.state_dict() for name, matcher in matchers.items()
        },
        args.output_dir / "pairwise_matchers.pth",
    )
    try:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        for name in matchers:
            axes[0].plot(
                [item["step"] for item in history],
                [item[f"{name}_loss"] for item in history],
                label=name,
            )
            axes[1].plot(
                [item["step"] for item in history],
                [item[f"{name}_row_top1_accuracy"] for item in history],
                label=name,
            )
        axes[0].set(title="2x2 assignment loss", xlabel="Step")
        axes[1].set(title="2x2 row top-1", xlabel="Step", ylim=(-0.05, 1.05))
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend()
        figure.tight_layout()
        figure.savefig(args.output_dir / "pairwise_overfit.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
