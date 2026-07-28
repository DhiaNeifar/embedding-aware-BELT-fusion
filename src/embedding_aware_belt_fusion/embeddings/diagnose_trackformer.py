"""Diagnose one real TrackFormer source-target association pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from embedding_aware_belt_fusion.data.trackformer_cache import (
    TrackFormerProposalDataset,
    collate_trackformer_frames,
)
from embedding_aware_belt_fusion.embeddings.trackformer import PointPillarsTrackFormer
from embedding_aware_belt_fusion.embeddings.trackformer_loss import trackformer_pair_loss
from embedding_aware_belt_fusion.embeddings.train_trackformer import (
    _group_agents,
    _source_states,
    _to_device,
)


def _off_diagonal_cosine(values):
    values = F.normalize(values, dim=-1)
    similarity = values @ values.T
    mask = ~torch.eye(len(values), dtype=torch.bool, device=values.device)
    selected = similarity[mask]
    return {
        "mean": float(selected.mean()),
        "std": float(selected.std(unbiased=False)),
        "min": float(selected.min()),
        "max": float(selected.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trackformer-root", type=Path, default=Path("external/trackformer"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)

    dataset = TrackFormerProposalDataset(
        args.cache_dir, split="train", validation_fraction=0.2
    )
    agents = collate_trackformer_frames([dataset[0]])
    source_cpu, target_cpu = _group_agents(agents, shuffle=False)[0]
    source, target = _to_device(source_cpu, device), _to_device(target_cpu, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint["arguments"]
    model = PointPillarsTrackFormer(
        args.trackformer_root,
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
    source_states, source_labels = _source_states(model, source, 2)
    output = model.forward_with_track_queries(
        target["bev"],
        target["query_features"],
        target["boxes"],
        target["scores"],
        source_states,
    )
    raw_projection = model.embedding_projection(output["states"])
    target_by_label = {
        int(label): index for index, label in enumerate(target["labels"].tolist())
    }
    query_indices, target_indices = [], []
    for index, label in enumerate(source_labels.tolist()):
        if label >= 0 and label in target_by_label:
            query_indices.append(index)
            target_indices.append(target_by_label[label])
    track_count = len(source_labels)
    similarity = (
        output["embeddings"][query_indices]
        @ output["embeddings"][track_count:].T
    )
    centered = F.normalize(
        raw_projection - raw_projection.mean(dim=0, keepdim=True), dim=-1
    )
    centered_similarity = (
        centered[query_indices] @ centered[track_count:].T
    )
    losses = trackformer_pair_loss(
        output,
        source_labels,
        target["labels"],
        model.normalize_boxes(target["target_boxes"]),
        association_weight=5.0,
        variance_weight=10.0,
        association_temperature=0.1,
    )
    model.zero_grad(set_to_none=True)
    losses["loss"].backward()
    report = {
        "source_true_queries": int((source_labels >= 0).sum()),
        "source_false_queries": int((source_labels < 0).sum()),
        "target_objects": len(target["labels"]),
        "shared_objects": len(query_indices),
        "decoder_state_cosine": _off_diagonal_cosine(output["states"].detach()),
        "raw_projection_cosine": _off_diagonal_cosine(raw_projection.detach()),
        "normalized_embedding_cosine": _off_diagonal_cosine(
            output["embeddings"].detach()
        ),
        "association_logit_std": float((similarity / 0.1).std(unbiased=False)),
        "association_probability_max_mean": float(
            (similarity / 0.1).softmax(-1).max(-1).values.mean()
        ),
        "centered_embedding_cosine": _off_diagonal_cosine(centered.detach()),
        "centered_top1": float(
            (
                centered_similarity.argmax(dim=1)
                == torch.tensor(target_indices, device=device)
            )
            .float()
            .mean()
        ),
        "embedding_projection_gradient": float(
            model.embedding_projection.weight.grad.norm()
        ),
        "query_content_gradient": float(model.query_content.weight.grad.norm()),
        "association_loss": float(losses["association"]),
        "variance_loss": float(losses["variance"]),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
