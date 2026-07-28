"""Hungarian assignment and complete DETR/TrackFormer supervision."""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from embedding_aware_belt_fusion.embeddings.geometry import (
    transform_boxes_to_ego,
)


def _best_identity_indices(agent):
    best = {}
    for index, (object_id, iou) in enumerate(
        zip(agent["gt_ids"], agent["gt_ious"])
    ):
        if object_id is None:
            continue
        key = str(object_id)
        score = float(iou)
        if key not in best or score > best[key][0]:
            best[key] = (score, index)
    return {key: value[1] for key, value in best.items()}


def spatial_embedding_loss(
    source_output,
    source_agent,
    target_output,
    target_agent,
    *,
    temperature=0.07,
):
    """Supervise the exported proposal embeddings with physical GT IDs."""
    source = _best_identity_indices(source_agent)
    target = _best_identity_indices(target_agent)
    shared = sorted(set(source).intersection(target))
    if len(shared) < 2:
        zero = source_output["embeddings"].sum() * 0
        return zero, zero.detach(), len(shared)
    device = source_output["embeddings"].device
    source_indices = torch.tensor(
        [source[key] for key in shared], dtype=torch.long, device=device
    )
    target_indices = torch.tensor(
        [target[key] for key in shared], dtype=torch.long, device=device
    )
    target_offset = int(target_output["track_count"])
    source_embeddings = source_output["embeddings"][0, source_indices]
    target_embeddings = target_output["embeddings"][
        0, target_offset + target_indices
    ]
    similarities = source_embeddings @ target_embeddings.T / temperature
    labels = torch.arange(len(shared), device=device)
    loss = (
        F.cross_entropy(similarities, labels)
        + F.cross_entropy(similarities.T, labels)
    ) / 2
    accuracy = (
        (similarities.argmax(1) == labels).float().mean()
        + (similarities.argmax(0) == labels).float().mean()
    ) / 2
    return loss, accuracy, len(shared)


def spatial_hard_negative_loss(
    source_output,
    source_agent,
    target_output,
    target_agent,
    *,
    radius=15.0,
    size_difference=2.0,
    margin=0.2,
):
    """Separate nearby, similar-size wrong vehicles by a cosine margin."""
    source = _best_identity_indices(source_agent)
    target = _best_identity_indices(target_agent)
    shared = sorted(set(source).intersection(target))
    if len(shared) < 2:
        zero = source_output["embeddings"].sum() * 0
        return zero, 0
    device = source_output["embeddings"].device
    source_indices = torch.tensor(
        [source[key] for key in shared], dtype=torch.long, device=device
    )
    target_indices = torch.tensor(
        [target[key] for key in shared], dtype=torch.long, device=device
    )
    target_offset = int(target_output["track_count"])
    source_embeddings = source_output["embeddings"][0, source_indices]
    target_embeddings = target_output["embeddings"][
        0, target_offset + target_indices
    ]
    source_boxes = transform_boxes_to_ego(
        source_agent["boxes"][source_indices.cpu()],
        source_agent["transformation_matrix"],
    ).to(device)
    target_boxes = transform_boxes_to_ego(
        target_agent["boxes"][target_indices.cpu()],
        target_agent["transformation_matrix"],
    ).to(device)
    similarities = source_embeddings @ target_embeddings.T
    position_distance = torch.cdist(
        source_boxes[:, :2], target_boxes[:, :2]
    )
    dimension_difference = torch.cdist(
        source_boxes[:, 3:6], target_boxes[:, 3:6], p=1
    )
    diagonal = torch.eye(
        len(shared), dtype=torch.bool, device=device
    )
    hard_mask = (
        (position_distance <= radius)
        & (dimension_difference <= size_difference)
        & ~diagonal
    )

    def directional_loss(scores, mask):
        valid = mask.any(dim=1)
        if not valid.any():
            return scores.sum() * 0, 0
        hardest = scores.masked_fill(~mask, -torch.inf).max(dim=1).values
        positives = scores.diag()
        return (
            F.relu(margin + hardest[valid] - positives[valid]).mean(),
            int(valid.sum()),
        )

    forward, forward_count = directional_loss(similarities, hard_mask)
    backward, backward_count = directional_loss(
        similarities.T, hard_mask.T
    )
    count = forward_count + backward_count
    if forward_count and backward_count:
        loss = (forward + backward) / 2
    else:
        loss = forward + backward
    return loss, count


def _xyxy(boxes):
    x, y, width, length = boxes[..., 0], boxes[..., 1], boxes[..., 4], boxes[..., 5]
    return torch.stack(
        [x - length / 2, y - width / 2, x + length / 2, y + width / 2],
        dim=-1,
    )


def pairwise_generalized_iou(predicted, target):
    predicted, target = _xyxy(predicted), _xyxy(target)
    pred, truth = predicted[:, None], target[None]
    intersection_size = (
        torch.minimum(pred[..., 2:], truth[..., 2:])
        - torch.maximum(pred[..., :2], truth[..., :2])
    ).clamp_min(0)
    intersection = intersection_size.prod(-1)
    pred_area = (pred[..., 2:] - pred[..., :2]).clamp_min(0).prod(-1)
    truth_area = (truth[..., 2:] - truth[..., :2]).clamp_min(0).prod(-1)
    union = pred_area + truth_area - intersection
    enclosing = (
        torch.maximum(pred[..., 2:], truth[..., 2:])
        - torch.minimum(pred[..., :2], truth[..., :2])
    ).clamp_min(0).prod(-1)
    return (
        intersection / union.clamp_min(1e-6)
        - (enclosing - union) / enclosing.clamp_min(1e-6)
    )


class SpatialHungarianMatcher(nn.Module):
    def __init__(self, *, class_cost=1.0, box_cost=5.0, giou_cost=2.0):
        super().__init__()
        self.class_cost = class_cost
        self.box_cost = box_cost
        self.giou_cost = giou_cost

    @torch.no_grad()
    def forward(self, output, target):
        probabilities = output["pred_logits"][0].softmax(-1)[:, 0]
        boxes = output["pred_boxes"][0]
        target_boxes = target["boxes"]
        if not len(target_boxes):
            empty = torch.zeros(0, dtype=torch.long, device=boxes.device)
            return empty, empty
        cost = (
            -self.class_cost * probabilities[:, None]
            + self.box_cost * torch.cdist(boxes, target_boxes, p=1)
            - self.giou_cost * pairwise_generalized_iou(boxes, target_boxes)
        )
        track_matches = target.get("track_query_match_ids")
        if track_matches is not None:
            large = 1e6
            for query_index, target_index in enumerate(track_matches.tolist()):
                if target_index < 0:
                    cost[query_index] = large
                else:
                    cost[query_index] = large
                    cost[:, target_index] = large
                    cost[query_index, target_index] = -large
        rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
        return (
            torch.as_tensor(rows, dtype=torch.long, device=boxes.device),
            torch.as_tensor(columns, dtype=torch.long, device=boxes.device),
        )


class SpatialTrackFormerCriterion(nn.Module):
    def __init__(
        self,
        matcher,
        *,
        no_object_weight=0.1,
        class_weight=1.0,
        box_weight=5.0,
        giou_weight=2.0,
        auxiliary_loss=True,
    ):
        super().__init__()
        self.matcher = matcher
        self.no_object_weight = no_object_weight
        self.weights = {
            "loss_ce": class_weight,
            "loss_bbox": box_weight,
            "loss_giou": giou_weight,
        }
        self.auxiliary_loss = auxiliary_loss

    def _layer_losses(self, output, target):
        prediction_indices, target_indices = self.matcher(output, target)
        logits, boxes = output["pred_logits"][0], output["pred_boxes"][0]
        classes = torch.ones(
            len(logits), dtype=torch.long, device=logits.device
        )
        classes[prediction_indices] = 0
        class_weights = logits.new_tensor([1.0, self.no_object_weight])
        per_query = F.cross_entropy(
            logits, classes, weight=class_weights, reduction="none"
        )
        false_mask = target.get("track_queries_false_positive_mask")
        if false_mask is not None and false_mask.any():
            per_query[false_mask] /= self.no_object_weight
        loss_ce = per_query.mean()
        if len(prediction_indices):
            predicted = boxes[prediction_indices]
            expected = target["boxes"][target_indices]
            loss_bbox = F.l1_loss(predicted, expected, reduction="sum")
            loss_bbox = loss_bbox / max(1, len(expected))
            giou = pairwise_generalized_iou(predicted, expected).diag()
            loss_giou = (1 - giou).mean()
        else:
            loss_bbox = boxes.sum() * 0
            loss_giou = boxes.sum() * 0
        return {
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }, (prediction_indices, target_indices)

    def forward(self, output, target):
        losses, indices = self._layer_losses(output, target)
        if self.auxiliary_loss:
            for layer_index, auxiliary in enumerate(
                output.get("aux_outputs", [])
            ):
                layer_losses, _ = self._layer_losses(auxiliary, target)
                losses.update(
                    {
                        f"{name}_{layer_index}": value
                        for name, value in layer_losses.items()
                    }
                )
        total = self.weights["loss_ce"] * losses["loss_ce"]
        total = total + self.weights["loss_bbox"] * losses["loss_bbox"]
        total = total + self.weights["loss_giou"] * losses["loss_giou"]
        for name, value in losses.items():
            if name.startswith("loss_ce_"):
                total = total + self.weights["loss_ce"] * value
            elif name.startswith("loss_bbox_"):
                total = total + self.weights["loss_bbox"] * value
            elif name.startswith("loss_giou_"):
                total = total + self.weights["loss_giou"] * value
        losses["loss"] = total
        losses["indices"] = indices
        return losses
