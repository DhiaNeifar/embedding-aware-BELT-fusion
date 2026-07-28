"""TrackFormer-style assignment, classification, and box losses."""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F


def _xyxy(boxes):
    x, y, width, length = boxes[:, 0], boxes[:, 1], boxes[:, 4], boxes[:, 5]
    return torch.stack(
        [x - width / 2, y - length / 2, x + width / 2, y + length / 2],
        dim=-1,
    )


def generalized_iou(predicted, target):
    predicted, target = _xyxy(predicted), _xyxy(target)
    left_top = torch.maximum(predicted[:, :2], target[:, :2])
    right_bottom = torch.minimum(predicted[:, 2:], target[:, 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
    pred_area = (predicted[:, 2:] - predicted[:, :2]).clamp_min(0).prod(dim=-1)
    target_area = (target[:, 2:] - target[:, :2]).clamp_min(0).prod(dim=-1)
    union = pred_area + target_area - intersection
    iou = intersection / union.clamp_min(1e-6)
    cover_left = torch.minimum(predicted[:, :2], target[:, :2])
    cover_right = torch.maximum(predicted[:, 2:], target[:, 2:])
    cover = (cover_right - cover_left).clamp_min(0).prod(dim=-1)
    return iou - (cover - union) / cover.clamp_min(1e-6)


def pairwise_generalized_iou(predicted, target):
    predicted, target = _xyxy(predicted), _xyxy(target)
    pred = predicted[:, None, :]
    truth = target[None, :, :]
    left_top = torch.maximum(pred[..., :2], truth[..., :2])
    right_bottom = torch.minimum(pred[..., 2:], truth[..., 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
    pred_area = (pred[..., 2:] - pred[..., :2]).clamp_min(0).prod(dim=-1)
    truth_area = (truth[..., 2:] - truth[..., :2]).clamp_min(0).prod(dim=-1)
    union = pred_area + truth_area - intersection
    iou = intersection / union.clamp_min(1e-6)
    cover_left = torch.minimum(pred[..., :2], truth[..., :2])
    cover_right = torch.maximum(pred[..., 2:], truth[..., 2:])
    cover = (cover_right - cover_left).clamp_min(0).prod(dim=-1)
    return iou - (cover - union) / cover.clamp_min(1e-6)


def trackformer_pair_loss(
    output,
    source_labels,
    target_labels,
    target_boxes,
    *,
    no_object_weight=0.1,
    classification_weight=1.0,
    l1_weight=5.0,
    giou_weight=2.0,
    association_weight=1.0,
    variance_weight=1.0,
    association_temperature=0.1,
):
    """Hard-assign propagated tracks; Hungarian-match ordinary object queries."""
    logits, predicted_boxes = output["logits"], output["boxes"]
    track_count = output["track_count"]
    device = logits.device
    target_boxes = target_boxes.to(device)
    normalized_targets = target_boxes
    classes = torch.zeros(len(logits), dtype=torch.long, device=device)
    pred_indices, gt_indices = [], []

    target_by_label = {
        int(label): index for index, label in enumerate(target_labels.tolist())
    }
    used_targets = set()
    for source_index, label in enumerate(source_labels.tolist()):
        target_index = target_by_label.get(int(label))
        if target_index is not None:
            classes[source_index] = 1
            pred_indices.append(source_index)
            gt_indices.append(target_index)
            used_targets.add(target_index)

    ordinary = torch.arange(
        track_count, len(logits), dtype=torch.long, device=device
    )
    remaining = [
        index for index in range(len(target_labels)) if index not in used_targets
    ]
    if len(ordinary) and remaining:
        remaining_tensor = torch.tensor(remaining, device=device)
        probability = logits[ordinary].softmax(-1)[:, 1]
        l1 = torch.cdist(
            predicted_boxes[ordinary], normalized_targets[remaining_tensor], p=1
        )
        giou_cost = 1 - pairwise_generalized_iou(
            predicted_boxes[ordinary], normalized_targets[remaining_tensor]
        )
        cost = -probability[:, None] + 5.0 * l1 + 2.0 * giou_cost
        rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
        for row, column in zip(rows, columns):
            prediction = int(ordinary[row])
            target = int(remaining_tensor[column])
            classes[prediction] = 1
            pred_indices.append(prediction)
            gt_indices.append(target)

    class_weights = torch.tensor(
        [no_object_weight, 1.0], device=device, dtype=logits.dtype
    )
    classification = F.cross_entropy(logits, classes, weight=class_weights)
    if pred_indices:
        pred_indices = torch.tensor(pred_indices, device=device)
        gt_indices = torch.tensor(gt_indices, device=device)
        prediction = predicted_boxes[pred_indices]
        target = normalized_targets[gt_indices]
        l1 = F.l1_loss(prediction, target)
        giou = (1 - generalized_iou(prediction, target)).mean()
    else:
        l1 = predicted_boxes.sum() * 0
        giou = predicted_boxes.sum() * 0
    embeddings = output["embeddings"]
    object_embeddings = embeddings[track_count:]
    association_queries, association_targets = [], []
    for source_index, label in enumerate(source_labels.tolist()):
        target_index = target_by_label.get(int(label))
        if label >= 0 and target_index is not None:
            association_queries.append(source_index)
            association_targets.append(target_index)
    if association_queries and len(object_embeddings):
        association_queries = torch.tensor(association_queries, device=device)
        association_targets = torch.tensor(association_targets, device=device)
        association_logits = (
            embeddings[association_queries] @ object_embeddings.T
        ) / association_temperature
        association = F.cross_entropy(
            association_logits, association_targets
        )
    else:
        association = embeddings.sum() * 0
    # The native detection heads do not directly constrain the exported
    # normalized embedding. This VICReg-style term prevents a constant vector.
    embedding_std = torch.sqrt(
        embeddings.var(dim=0, unbiased=False) + 1e-4
    )
    variance = F.relu(0.05 - embedding_std).mean()

    total = (
        classification_weight * classification
        + l1_weight * l1
        + giou_weight * giou
        + association_weight * association
        + variance_weight * variance
    )
    return {
        "loss": total,
        "classification": classification,
        "l1": l1,
        "giou": giou,
        "association": association,
        "variance": variance,
    }
