"""BELT-style proposal uncertainty decoding and adaptive box fusion.

The upstream BELT-Fusion repository provides the evidential and Mahalanobis
formulas, but its released fusion path contains placeholder NMS and does not
propagate anchor uncertainty through PointPillars decoding.  This module
implements that missing OpenCOOD adapter while retaining the published
evidential mass fusion.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from embedding_aware_belt_fusion.embeddings.geometry import (
    transform_boxes_to_ego,
)
from embedding_aware_belt_fusion.integration.uncertainty_loss import (
    _anchor_evidence,
    _anchor_regression,
)


def _decoded_local_variance(
    encoded_log_variance: Tensor,
    anchors: Tensor,
    decoded_boxes: Tensor,
) -> Tensor:
    """Propagate diagonal anchor-delta variance into decoded box space."""
    encoded_variance = encoded_log_variance.exp()
    anchor_diagonal = torch.sqrt(
        anchors[:, 4].square() + anchors[:, 5].square()
    )
    jacobian = torch.stack(
        [
            anchor_diagonal,
            anchor_diagonal,
            anchors[:, 3],
            decoded_boxes[:, 3],
            decoded_boxes[:, 4],
            decoded_boxes[:, 5],
            torch.ones_like(anchor_diagonal),
        ],
        dim=-1,
    )
    return encoded_variance * jacobian.square()


def _transform_diagonal_covariance(
    local_variance: Tensor,
    transformation_matrix: Tensor,
    *,
    position_noise_std: float,
    heading_noise_std_deg: float,
) -> Tensor:
    """Transform local diagonal covariance into ego box coordinates."""
    count = len(local_variance)
    covariance = local_variance.new_zeros((count, 7, 7))
    rotation = transformation_matrix[:3, :3].to(
        device=local_variance.device,
        dtype=local_variance.dtype,
    )
    local_center = torch.diag_embed(local_variance[:, :3])
    covariance[:, :3, :3] = (
        rotation.unsqueeze(0)
        @ local_center
        @ rotation.T.unsqueeze(0)
    )
    dimensions = torch.arange(3, 6, device=local_variance.device)
    covariance[:, dimensions, dimensions] = local_variance[:, 3:6]
    covariance[:, 6, 6] = local_variance[:, 6]
    if position_noise_std:
        center = torch.arange(3, device=local_variance.device)
        covariance[:, center, center] += position_noise_std**2
    if heading_noise_std_deg:
        covariance[:, 6, 6] += math.radians(heading_noise_std_deg) ** 2
    return covariance


def proposal_uncertainty(
    output: Mapping[str, Tensor],
    proposal: Mapping[str, Tensor],
    cav_content: Mapping[str, Tensor],
    *,
    position_noise_std: float = 0.0,
    heading_noise_std_deg: float = 0.0,
) -> Dict[str, Tensor]:
    """Attach proposal-aligned evidence and decoded ego-frame covariance."""
    indices = proposal["anchor_indices"].long()
    boxes = proposal["boxes"]
    transformation = cav_content["transformation_matrix"].to(boxes.device)
    if indices.numel() == 0:
        return {
            "boxes": transform_boxes_to_ego(boxes, transformation),
            "scores": proposal["scores"],
            "evidence": boxes.new_empty((0, 2)),
            "class_uncertainty": boxes.new_empty((0,)),
            "covariances": boxes.new_empty((0, 7, 7)),
        }

    alpha = _anchor_evidence(output["alpha"])[0, indices]
    evidence = (alpha - 1.0).clamp_min(0.0)
    class_uncertainty = alpha.shape[-1] / alpha.sum(dim=-1)
    encoded_log_variance = _anchor_regression(output["reg_log_var"])[
        0, indices
    ]
    anchors = cav_content["anchor_box"].reshape(-1, 7)[indices].to(
        device=boxes.device,
        dtype=boxes.dtype,
    )
    local_variance = _decoded_local_variance(
        encoded_log_variance, anchors, boxes
    )
    covariance = _transform_diagonal_covariance(
        local_variance,
        transformation,
        position_noise_std=position_noise_std,
        heading_noise_std_deg=heading_noise_std_deg,
    )
    return {
        "boxes": transform_boxes_to_ego(boxes, transformation),
        "scores": proposal["scores"],
        "evidence": evidence,
        "class_uncertainty": class_uncertainty,
        "covariances": covariance,
    }


def evidence_to_mass(evidence: Tensor) -> tuple[Tensor, Tensor]:
    """Convert non-negative evidence into belief and uncertainty masses."""
    class_count = evidence.shape[-1]
    strength = (evidence + 1.0).sum(dim=-1, keepdim=True)
    return evidence / strength, class_count / strength.squeeze(-1)


def fuse_two_masses(
    belief_a: Tensor,
    uncertainty_a: Tensor,
    belief_b: Tensor,
    uncertainty_b: Tensor,
) -> tuple[Tensor, Tensor]:
    """Dempster-Shafer fusion for two subjective-logic masses."""
    conflict = (
        belief_a.sum(dim=-1, keepdim=True)
        * belief_b.sum(dim=-1, keepdim=True)
        - (belief_a * belief_b).sum(dim=-1, keepdim=True)
    )
    normalizer = (1.0 - conflict).clamp_min(1e-8)
    belief = (
        belief_a * belief_b
        + belief_a * uncertainty_b.unsqueeze(-1)
        + belief_b * uncertainty_a.unsqueeze(-1)
    ) / normalizer
    uncertainty = (
        uncertainty_a * uncertainty_b / normalizer.squeeze(-1)
    )
    return belief, uncertainty


def _group_representative(group: List[Dict[str, Tensor]]) -> Tensor:
    weights = torch.stack(
        [
            1.0
            / member["covariance"][:2, :2]
            .diagonal()
            .sum()
            .clamp_min(1e-4)
            for member in group
        ]
    )
    centers = torch.stack([member["box"][:2] for member in group])
    return (centers * weights[:, None]).sum(dim=0) / weights.sum()


def _group_embedding(group: List[Dict[str, Tensor]]) -> Tensor:
    embeddings = torch.stack([member["embedding"] for member in group])
    return torch.nn.functional.normalize(embeddings.mean(dim=0), dim=0)


def associate_by_geometry(
    detections: Iterable[Mapping[str, Tensor]],
    *,
    maximum_distance: float = 5.0,
) -> List[List[Dict[str, Tensor]]]:
    """Create one-to-one multi-agent groups using ego-frame center distance."""
    groups: List[List[Dict[str, Tensor]]] = []
    for agent_index, detection in enumerate(detections):
        members = [
            {
                "agent_index": agent_index,
                "box": detection["boxes"][index],
                "score": detection["scores"][index],
                "evidence": detection["evidence"][index],
                "class_uncertainty": detection["class_uncertainty"][index],
                "covariance": detection["covariances"][index],
            }
            for index in range(len(detection["boxes"]))
        ]
        if not groups:
            groups = [[member] for member in members]
            continue
        if not members:
            continue
        representatives = torch.stack(
            [_group_representative(group) for group in groups]
        )
        centers = torch.stack([member["box"][:2] for member in members])
        distance = torch.cdist(representatives, centers)
        rows, columns = linear_sum_assignment(distance.detach().cpu().numpy())
        matched_members = set()
        for row, column in zip(rows.tolist(), columns.tolist()):
            if float(distance[row, column]) <= maximum_distance:
                groups[row].append(members[column])
                matched_members.add(column)
        groups.extend(
            [member]
            for index, member in enumerate(members)
            if index not in matched_members
        )
    return groups


def associate_by_embedding(
    detections: Iterable[Mapping[str, Tensor]],
    *,
    maximum_distance: float = 8.0,
    minimum_similarity: float = 0.5,
) -> List[List[Dict[str, Tensor]]]:
    """Associate CAV proposals with TrackFormer embeddings and a safety gate.

    Embeddings choose the assignment.  A permissive ego-frame distance gate
    prevents physically implausible matches if an appearance embedding fails.
    """
    groups: List[List[Dict[str, Tensor]]] = []
    for agent_index, detection in enumerate(detections):
        members = [
            {
                "agent_index": agent_index,
                "box": detection["boxes"][index],
                "score": detection["scores"][index],
                "evidence": detection["evidence"][index],
                "class_uncertainty": detection["class_uncertainty"][index],
                "covariance": detection["covariances"][index],
                "embedding": detection["embeddings"][index],
            }
            for index in range(len(detection["boxes"]))
        ]
        if not groups:
            groups = [[member] for member in members]
            continue
        if not members:
            continue
        representatives = torch.stack(
            [_group_representative(group) for group in groups]
        )
        embeddings = torch.stack([_group_embedding(group) for group in groups])
        centers = torch.stack([member["box"][:2] for member in members])
        source_embeddings = torch.stack(
            [member["embedding"] for member in members]
        )
        similarity = embeddings @ source_embeddings.T
        distance = torch.cdist(representatives, centers)
        # The tiny distance term resolves equal-appearance assignments without
        # returning to a geometry-only matcher.
        cost = 1.0 - similarity + 0.01 * distance
        rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
        matched_members = set()
        for row, column in zip(rows.tolist(), columns.tolist()):
            if (
                float(similarity[row, column]) >= minimum_similarity
                and float(distance[row, column]) <= maximum_distance
            ):
                groups[row].append(members[column])
                matched_members.add(column)
        groups.extend(
            [member]
            for index, member in enumerate(members)
            if index not in matched_members
        )
    return groups


def fuse_group(group: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Fuse one associated object group using covariance precision."""
    boxes = torch.stack([member["box"] for member in group])
    scores = torch.stack([member["score"] for member in group])
    covariances = torch.stack([member["covariance"] for member in group])
    variances = covariances.diagonal(dim1=-2, dim2=-1).clamp(1e-4, 1e2)
    precision = variances.reciprocal()

    fused_box = (boxes * precision).sum(dim=0) / precision.sum(dim=0)
    yaw_weight = precision[:, 6]
    fused_box[6] = torch.atan2(
        (torch.sin(boxes[:, 6]) * yaw_weight).sum(),
        (torch.cos(boxes[:, 6]) * yaw_weight).sum(),
    )
    fused_variance = precision.sum(dim=0).reciprocal()
    fused_covariance = torch.diag(fused_variance)

    belief, uncertainty = evidence_to_mass(group[0]["evidence"][None])
    for member in group[1:]:
        next_belief, next_uncertainty = evidence_to_mass(
            member["evidence"][None]
        )
        belief, uncertainty = fuse_two_masses(
            belief, uncertainty, next_belief, next_uncertainty
        )
    return {
        "box": fused_box,
        # Keep detector confidence ranking controlled while testing whether
        # uncertainty-weighted geometry itself improves AP.
        "score": scores.max(),
        "covariance": fused_covariance,
        "belief": belief[0],
        "class_uncertainty": uncertainty[0],
        "member_count": boxes.new_tensor(len(group), dtype=torch.long),
    }


def fuse_groups(groups: Iterable[List[Dict[str, Tensor]]]) -> Dict[str, Tensor]:
    """Fuse already-associated groups into one output tensor dictionary."""
    groups = list(groups)
    if not groups:
        return {
            "boxes": torch.empty((0, 7)),
            "scores": torch.empty((0,)),
            "covariances": torch.empty((0, 7, 7)),
            "class_uncertainty": torch.empty((0,)),
            "member_counts": torch.empty((0,), dtype=torch.long),
        }
    fused = [fuse_group(group) for group in groups]
    return {
        "boxes": torch.stack([item["box"] for item in fused]),
        "scores": torch.stack([item["score"] for item in fused]),
        "covariances": torch.stack(
            [item["covariance"] for item in fused]
        ),
        "class_uncertainty": torch.stack(
            [item["class_uncertainty"] for item in fused]
        ),
        "member_counts": torch.stack(
            [item["member_count"] for item in fused]
        ),
    }


def associate_ego_with_propagated_sources(
    ego: Mapping[str, Tensor],
    sources: Iterable[Mapping[str, Tensor]],
    *,
    maximum_distance: float = 8.0,
    minimum_similarity: float = 0.5,
) -> List[List[Dict[str, Tensor]]]:
    """Associate sources to ego with TrackFormer-propagated embeddings.

    Every source mapping provides its own ``ego_embeddings``.  Those vectors
    are produced after propagating that source's object queries through the
    ego TrackFormer decoder, exactly mirroring the spatial adaptation of
    TrackFormer's temporal query propagation.
    """
    groups = [
        [
            {
                "agent_index": 0,
                "box": ego["boxes"][index],
                "score": ego["scores"][index],
                "evidence": ego["evidence"][index],
                "class_uncertainty": ego["class_uncertainty"][index],
                "covariance": ego["covariances"][index],
            }
        ]
        for index in range(len(ego["boxes"]))
    ]
    ego_centers = ego["boxes"][:, :2]
    for source_index, source in enumerate(sources, start=1):
        count = len(source["boxes"])
        if not count:
            continue
        members = [
            {
                "agent_index": source_index,
                "box": source["boxes"][index],
                "score": source["scores"][index],
                "evidence": source["evidence"][index],
                "class_uncertainty": source["class_uncertainty"][index],
                "covariance": source["covariances"][index],
            }
            for index in range(count)
        ]
        if not len(ego_centers):
            groups.extend([[member] for member in members])
            continue
        target_embeddings = source["ego_embeddings"]
        source_embeddings = source["embeddings"]
        similarity = target_embeddings @ source_embeddings.T
        distance = torch.cdist(ego_centers, source["boxes"][:, :2])
        cost = 1.0 - similarity + 0.01 * distance
        rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
        matched = set()
        for row, column in zip(rows.tolist(), columns.tolist()):
            if (
                float(similarity[row, column]) >= minimum_similarity
                and float(distance[row, column]) <= maximum_distance
            ):
                groups[row].append(members[column])
                matched.add(column)
        groups.extend(
            [member]
            for index, member in enumerate(members)
            if index not in matched
        )
    return groups


def singleton_groups(detection: Mapping[str, Tensor]) -> List[List[Dict[str, Tensor]]]:
    """Return one fusion group per unassociated detection."""
    return [
        [
            {
                "agent_index": -1,
                "box": detection["boxes"][index],
                "score": detection["scores"][index],
                "evidence": detection["evidence"][index],
                "class_uncertainty": detection["class_uncertainty"][index],
                "covariance": detection["covariances"][index],
            }
        ]
        for index in range(len(detection["boxes"]))
    ]


def fuse_detections(
    detections: Iterable[Mapping[str, Tensor]],
    *,
    maximum_distance: float = 5.0,
    association: str = "geometry",
    minimum_similarity: float = 0.5,
) -> Dict[str, Tensor]:
    """Associate and fuse all detections from one cooperative frame."""
    detections = list(detections)
    if association == "geometry":
        groups = associate_by_geometry(
            detections, maximum_distance=maximum_distance
        )
    elif association == "embedding":
        groups = associate_by_embedding(
            detections,
            maximum_distance=maximum_distance,
            minimum_similarity=minimum_similarity,
        )
    else:
        raise ValueError(f"Unknown association mode: {association}")
    return fuse_groups(groups)
