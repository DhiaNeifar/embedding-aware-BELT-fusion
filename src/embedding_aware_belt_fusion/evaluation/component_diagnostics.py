"""Ground-truth diagnostics for PQ-STF association and grouping.

These functions are evaluation-only.  ``oracle_object_ids`` originate from
OPV2V labels and are never used by the deployed association/RANSAC path.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Mapping

import torch


def empty_counts() -> dict[str, float]:
    return {
        "candidate_pairs": 0,
        "positive_pairs": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "positive_probability_sum": 0.0,
        "negative_probability_sum": 0.0,
        "positive_probability_count": 0,
        "negative_probability_count": 0,
    }


def merge_counts(total: dict[str, float], update: Mapping[str, float]) -> None:
    for key, value in update.items():
        total[key] = total.get(key, 0) + value


def summarize_counts(counts: Mapping[str, float]) -> dict[str, float | int]:
    tp = int(counts.get("true_positive", 0))
    fp = int(counts.get("false_positive", 0))
    fn = int(counts.get("false_negative", 0))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "candidate_pairs": int(counts.get("candidate_pairs", 0)),
        "positive_pairs": int(counts.get("positive_pairs", 0)),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-8),
        "mean_probability_true_pair": (
            counts.get("positive_probability_sum", 0.0)
            / max(counts.get("positive_probability_count", 0), 1)
        ),
        "mean_probability_nonmatch": (
            counts.get("negative_probability_sum", 0.0)
            / max(counts.get("negative_probability_count", 0), 1)
        ),
    }


def pair_head_counts(
    detections: Iterable[Mapping],
    pair_head,
    *,
    maximum_distance: float,
    minimum_probability: float,
) -> dict[str, float]:
    """Measure all cross-CAV candidate pairs against physical object IDs."""
    output = empty_counts()
    for first, second in combinations(list(detections), 2):
        first_state, second_state = first["query_states"], second["query_states"]
        if not len(first_state) or not len(second_state):
            continue
        probability = torch.sigmoid(
            pair_head(
                first_state,
                first["query_geometry"],
                second_state,
                second["query_geometry"],
            )
        )
        distance = torch.cdist(first["boxes"][:, :2], second["boxes"][:, :2])
        candidate = distance <= maximum_distance
        labels = torch.zeros_like(candidate)
        first_ids = first["oracle_object_ids"]
        second_ids = second["oracle_object_ids"]
        for row, object_id in enumerate(first_ids):
            if object_id is None:
                continue
            for column, other_id in enumerate(second_ids):
                labels[row, column] = object_id == other_id
        positive = candidate & labels
        negative = candidate & ~labels
        predicted = candidate & (probability >= minimum_probability)
        output["candidate_pairs"] += int(candidate.sum())
        output["positive_pairs"] += int(positive.sum())
        output["true_positive"] += int((predicted & positive).sum())
        output["false_positive"] += int((predicted & negative).sum())
        output["false_negative"] += int(((~predicted) & positive).sum())
        output["positive_probability_sum"] += float(probability[positive].sum())
        output["negative_probability_sum"] += float(probability[negative].sum())
        output["positive_probability_count"] += int(positive.sum())
        output["negative_probability_count"] += int(negative.sum())
    return output


def group_counts(groups: Iterable[list[Mapping]]) -> dict[str, float]:
    """Measure group purity using only labelled cross-agent member pairs."""
    output = empty_counts()
    for group in groups:
        for first, second in combinations(group, 2):
            if first.get("agent_id") == second.get("agent_id"):
                continue
            first_id = first.get("oracle_object_id")
            second_id = second.get("oracle_object_id")
            if first_id is None or second_id is None:
                continue
            same = first_id == second_id
            output["candidate_pairs"] += 1
            output["positive_pairs"] += int(same)
            output["true_positive"] += int(same)
            output["false_positive"] += int(not same)
    # A grouped labelled positive not present in a group is not observable from
    # groups alone; pair-head recall is reported separately above.
    return output


def association_counts(
    detections: Iterable[Mapping],
    groups: Iterable[list[Mapping]],
    *,
    maximum_distance: float,
) -> dict[str, float]:
    """Full precision/recall/F1 for a non-learned association (e.g. nearest-
    distance grouping), against physical object IDs.

    Unlike ``group_counts``, this also counts false negatives: candidate
    pairs (within ``maximum_distance``, both physical IDs known) that are
    truly the same object but were *not* placed in the same group. That
    requires knowing every candidate pair up front, not just the ones a
    one-to-one assignment happened to admit into a group.
    """
    output = empty_counts()
    detections = list(detections)
    group_of: dict[tuple, int] = {}
    for group_index, group in enumerate(groups):
        for member in group:
            group_of[(member["agent_index"], member["proposal_index"])] = group_index

    for first_index in range(len(detections)):
        first = detections[first_index]
        first_ids = first.get("oracle_object_ids")
        if first_ids is None or not len(first["boxes"]):
            continue
        for second_index in range(first_index + 1, len(detections)):
            second = detections[second_index]
            second_ids = second.get("oracle_object_ids")
            if second_ids is None or not len(second["boxes"]):
                continue
            distance = torch.cdist(first["boxes"][:, :2], second["boxes"][:, :2])
            for row, first_id in enumerate(first_ids):
                if first_id is None:
                    continue
                for column, second_id in enumerate(second_ids):
                    if second_id is None or distance[row, column] > maximum_distance:
                        continue
                    same = first_id == second_id
                    grouped_together = (
                        group_of.get((first_index, row)) is not None
                        and group_of.get((first_index, row))
                        == group_of.get((second_index, column))
                    )
                    output["candidate_pairs"] += 1
                    output["positive_pairs"] += int(same)
                    if same and grouped_together:
                        output["true_positive"] += 1
                    elif (not same) and grouped_together:
                        output["false_positive"] += 1
                    elif same and not grouped_together:
                        output["false_negative"] += 1
    return output


def summarize_group_counts(counts: Mapping[str, float]) -> dict[str, float | int]:
    """Summarize the labelled edges actually admitted into groups."""
    correct = int(counts.get("true_positive", 0))
    incorrect = int(counts.get("false_positive", 0))
    return {
        "labelled_group_edges": correct + incorrect,
        "same_object_edges": correct,
        "different_object_edges": incorrect,
        "edge_purity": correct / max(correct + incorrect, 1),
    }
