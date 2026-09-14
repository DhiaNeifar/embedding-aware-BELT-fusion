"""Shared utilities for all-agent, source-to-source PQ-STF association.

The existing propagated-query PQ-STF checkpoint is trained for source-to-ego
association.  This module deliberately keeps that backbone frozen while
testing two source-to-source grouping mechanisms:

* ``direct``: cosine similarity between the checkpoint's exported local
  association embeddings; and
* ``symmetric-pair``: a learned, order-invariant scorer over two transmitted
  256-D query states and their 9-D geometry/context vectors.

Both methods receive exactly the same per-object source messages.  This makes
the first comparison an association experiment, not a detector/fusion change.
"""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from embedding_aware_belt_fusion.data.complete_cache import (
    load_or_build_scenario_index,
)
from embedding_aware_belt_fusion.embeddings.spatial_trackformer import (
    SpatialTrackFormer,
)
from embedding_aware_belt_fusion.embeddings.geometry import transform_boxes_to_ego


def has_valid_proposals(agent) -> bool:
    """Return whether an agent has proposal ROI cells usable by PQ-STF."""
    if len(agent.get("boxes", ())) == 0:
        return False
    tokens, mask = agent.get("tokens"), agent.get("mask")
    return bool(
        tokens is not None
        and mask is not None
        and tokens.numel() > 0
        and (~mask.bool()).any()
    )


def ego_index(frame) -> int:
    """Locate the physical ego agent from its identity transform."""
    identity = torch.eye(4)
    return min(
        range(len(frame["agents"])),
        key=lambda index: float(
            (frame["agents"][index]["transformation_matrix"].float() - identity)
            .abs()
            .max()
        ),
    )


def source_agent_pairs(frame) -> Iterable[tuple[dict, dict]]:
    """Yield every usable non-ego CAV pair in one same-timestamp frame."""
    ego = ego_index(frame)
    sources = [
        agent
        for index, agent in enumerate(frame["agents"])
        if index != ego and has_valid_proposals(agent)
    ]
    return combinations(sources, 2)


def all_agent_pairs(frame) -> Iterable[tuple[dict, dict]]:
    """Yield every usable CAV pair, including ego--source pairs."""
    agents = [agent for agent in frame["agents"] if has_valid_proposals(agent)]
    return combinations(agents, 2)


def best_identity_indices(agent) -> dict[str, int]:
    """Select the highest-IoU proposal for every labelled physical object."""
    best: dict[str, tuple[float, int]] = {}
    for index, (object_id, iou) in enumerate(
        zip(agent["gt_ids"], agent["gt_ious"])
    ):
        if object_id is None:
            continue
        key, score = str(object_id), float(iou)
        if key not in best or score > best[key][0]:
            best[key] = (score, index)
    return {key: value[1] for key, value in best.items()}


def load_pq_stf(checkpoint: Path, trackformer_root: Path, feature_dim: int, device):
    """Load a frozen geometry-conditioned propagated-query checkpoint."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("pipeline") != "spatial_trackformer_v2_geometry_association":
        raise ValueError(
            "All-agent PQ-STF evaluation requires a geometry-association "
            "SpatialTrackFormer checkpoint"
        )
    if saved.get("association_protocol", "propagated") != "propagated":
        raise ValueError(
            "All-agent source-state comparison requires a propagated PQ-STF checkpoint"
        )
    arguments = saved["arguments"]
    model = SpatialTrackFormer(
        trackformer_root=trackformer_root,
        input_dim=feature_dim,
        d_model=int(arguments["d_model"]),
        embedding_dim=int(arguments["embedding_dim"]),
        heads=int(arguments["heads"]),
        encoder_layers=int(arguments["encoder_layers"]),
        decoder_layers=int(arguments["decoder_layers"]),
        feedforward_dim=int(arguments["feedforward_dim"]),
        dropout=float(arguments["dropout"]),
        geometry_association=True,
    ).to(device)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.requires_grad_(False)
    model.eval()
    return model, saved


class SymmetricQueryStatePairHead(nn.Module):
    """Order-invariant same-object score for two PQ-STF source messages."""

    def __init__(self, state_dim: int, geometry_dim: int = 9):
        super().__init__()
        input_dim = 2 * state_dim + 2 * geometry_dim
        hidden = max(128, state_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, states_a, geometry_a, states_b, geometry_b):
        """Return a score matrix of shape ``[len(a), len(b)]``.

        Absolute differences and element-wise products are symmetric, hence
        exchanging the two source messages leaves a pair's score unchanged.
        """
        a_state, b_state = states_a[:, None, :], states_b[None, :, :]
        a_geometry, b_geometry = geometry_a[:, None, :], geometry_b[None, :, :]
        features = torch.cat(
            [
                (a_state - b_state).abs(),
                a_state * b_state,
                (a_geometry - b_geometry).abs(),
                a_geometry * b_geometry,
            ],
            dim=-1,
        )
        return self.network(features).squeeze(-1)


def direct_scores(output_a, output_b):
    """IE-like baseline: direct cosine comparison of exported embeddings."""
    return output_a["embeddings"][0] @ output_b["embeddings"][0].T


def pair_head_scores(head, output_a, output_b):
    """Source-state scorer over the same information transmitted by PQ-STF."""
    return head(
        output_a["hs_embed"][0],
        output_a["query_geometry"],
        output_b["hs_embed"][0],
        output_b["query_geometry"],
    )


def shared_identity_loss_and_metrics(scores, agent_a, agent_b, temperature=0.07):
    """Symmetric retrieval loss/top-1 for a source--source score matrix.

    Anchors are the best labelled proposal for each shared OPV2V physical ID.
    Every proposal at the other source, including unmatched detector proposals,
    remains a candidate negative.  This mirrors deployed association more
    closely than evaluating only ground-truth-filtered boxes.
    """
    best_a, best_b = best_identity_indices(agent_a), best_identity_indices(agent_b)
    shared = sorted(set(best_a).intersection(best_b))
    zero = scores.sum() * 0
    if not shared:
        return {
            "loss": zero,
            "top1_correct": 0.0,
            "top1_total": 0,
            "shared_objects": 0,
        }
    device = scores.device
    rows = torch.tensor([best_a[key] for key in shared], device=device)
    columns = torch.tensor([best_b[key] for key in shared], device=device)
    forward = scores[rows]
    backward = scores.T[columns]
    loss = (
        nn.functional.cross_entropy(forward / temperature, columns)
        + nn.functional.cross_entropy(backward / temperature, rows)
    ) / 2
    correct = int((forward.argmax(dim=1) == columns).sum())
    correct += int((backward.argmax(dim=1) == rows).sum())
    return {
        "loss": loss,
        "top1_correct": float(correct),
        "top1_total": 2 * len(shared),
        "shared_objects": len(shared),
    }


def binary_pair_loss_and_metrics(
    logits,
    agent_a,
    agent_b,
    *,
    maximum_distance: float,
    probability_threshold: float,
    negative_ratio: int,
    maximum_no_match_negatives: int = 32,
):
    """Calibrated same-object/no-match loss over geometrically valid pairs.

    Unlike retrieval loss, every nearby pair is explicitly labelled same or
    different.  This includes false detections and source-only/ego-only
    proposals, which are negative pairs and teach the head that ``no match``
    is a valid outcome.
    """
    if logits.ndim != 2:
        raise ValueError("pair logits must have shape [objects_a, objects_b]")
    boxes_a = transform_boxes_to_ego(
        agent_a["boxes"], agent_a["transformation_matrix"]
    ).to(logits.device)
    boxes_b = transform_boxes_to_ego(
        agent_b["boxes"], agent_b["transformation_matrix"]
    ).to(logits.device)
    distance = torch.cdist(boxes_a[:, :2], boxes_b[:, :2])
    candidate = distance <= maximum_distance
    ids_a = list(agent_a["gt_ids"])
    ids_b = list(agent_b["gt_ids"])
    labels = torch.zeros_like(logits, dtype=torch.bool)
    for index_a, object_id_a in enumerate(ids_a):
        if object_id_a is None:
            continue
        for index_b, object_id_b in enumerate(ids_b):
            labels[index_a, index_b] = object_id_a == object_id_b
    positive = candidate & labels
    negative = candidate & ~labels
    positive_indices = positive.flatten().nonzero().squeeze(1)
    negative_indices = negative.flatten().nonzero().squeeze(1)
    if not len(positive_indices) and not len(negative_indices):
        zero = logits.sum() * 0
        return {
            "loss": zero,
            "pairs": 0,
            "positive_pairs": 0,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
        }

    # Keep all positives and the closest hard negatives.  This prevents the
    # huge easy-negative population from overwhelming the no-match objective.
    if len(positive_indices):
        negative_limit = min(
            len(negative_indices), len(positive_indices) * negative_ratio
        )
    else:
        negative_limit = min(len(negative_indices), maximum_no_match_negatives)
    if negative_limit:
        flat_distance = distance.flatten()[negative_indices]
        hard_order = flat_distance.argsort()[:negative_limit]
        negative_indices = negative_indices[hard_order]
    selected = torch.cat([positive_indices, negative_indices])
    selected_logits = logits.flatten()[selected]
    selected_labels = positive.flatten()[selected].to(selected_logits.dtype)
    positive_weight = torch.tensor(
        min(
            max(len(negative_indices) / max(len(positive_indices), 1), 1.0),
            10.0,
        ),
        device=logits.device,
        dtype=logits.dtype,
    )
    loss = nn.functional.binary_cross_entropy_with_logits(
        selected_logits, selected_labels, pos_weight=positive_weight
    )
    predicted = torch.sigmoid(logits) >= probability_threshold
    evaluated = candidate
    true_positive = int((predicted & positive).sum())
    false_positive = int((predicted & negative).sum())
    false_negative = int(((~predicted) & positive).sum())
    return {
        "loss": loss,
        "pairs": int(evaluated.sum()),
        "positive_pairs": int(positive.sum()),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def records_for_role(cache_dir: Path, split_file: Path | None, role: str | None, workers: int):
    """Load all cache records or the named scenario role from a saved split."""
    index = load_or_build_scenario_index(cache_dir, workers=workers)
    records = index["frames"]
    if split_file is None:
        return records
    split = json.loads(Path(split_file).read_text())
    if role not in {"train", "validation"}:
        raise ValueError("--scenario-role must be train or validation with --scenario-split-file")
    scenario_ids = set(split[f"{role}_scenarios"])
    return [record for record in records if record["scenario_id"] in scenario_ids]
