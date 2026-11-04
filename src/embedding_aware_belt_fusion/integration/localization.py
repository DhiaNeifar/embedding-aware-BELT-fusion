"""Robust per-CAV planar localization correction from matched detections."""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import torch
from torch import Tensor


def _wrap_angle(angle: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _se2_from_two_pairs(source: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Estimate planar rigid transform mapping ``source`` points to ``target``."""
    source_delta = source[1] - source[0]
    target_delta = target[1] - target[0]
    angle = torch.atan2(
        source_delta[0] * target_delta[1] - source_delta[1] * target_delta[0],
        (source_delta * target_delta).sum(),
    )
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotation = torch.stack(
        [torch.stack([cosine, -sine]), torch.stack([sine, cosine])]
    )
    translation = target[0] - rotation @ source[0]
    return rotation, translation


def _refine_se2(source: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Least-squares rigid transform over RANSAC inliers (2-D Kabsch)."""
    source_center = source.mean(dim=0)
    target_center = target.mean(dim=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_t = torch.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if torch.linalg.det(rotation) < 0:
        right_t = right_t.clone()
        right_t[-1] *= -1
        rotation = right_t.T @ left.T
    return rotation, target_center - rotation @ source_center


def _batched_se2_from_pairs(
    source_first: Tensor,
    source_second: Tensor,
    target_first: Tensor,
    target_second: Tensor,
) -> tuple[Tensor, Tensor]:
    """Batched planar rigid transform from one point-pair per hypothesis.

    Vectorized form of :func:`_se2_from_two_pairs` over a leading hypothesis
    dimension ``H``. All inputs are ``(H, 2)``; returns rotation ``(H, 2, 2)``
    and translation ``(H, 2)``.
    """
    source_delta = source_second - source_first
    target_delta = target_second - target_first
    angle = torch.atan2(
        source_delta[:, 0] * target_delta[:, 1] - source_delta[:, 1] * target_delta[:, 0],
        (source_delta * target_delta).sum(dim=-1),
    )
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotation = torch.stack(
        [
            torch.stack([cosine, -sine], dim=-1),
            torch.stack([sine, cosine], dim=-1),
        ],
        dim=-2,
    )
    translation = target_first - torch.einsum("hij,hj->hi", rotation, source_first)
    return rotation, translation


def estimate_se2_ransac(
    source_centers: Tensor,
    ego_centers: Tensor,
    *,
    inlier_threshold: float,
    max_hypotheses: int = 128,
) -> dict | None:
    """Estimate ``ego_center ≈ R @ source_center + t`` with RANSAC.

    At least two spatially distinct matched boxes are required.  The returned
    correction is relative: it says how the source's current ego-frame boxes
    should move to agree with the ego CAV, not which vehicle is absolutely
    wrong in the real world.

    Hypothesis generation and scoring are vectorized over all candidate pairs
    at once (previously a pure-Python loop, the dominant cost of a sweep at
    thousands of frames per point) rather than iterating hypotheses one at a
    time. Numerically and in tie-breaking this is pinned equivalent to that
    original loop by ``tests/test_localization_ransac.py``: the loop kept the
    *first* hypothesis reaching the best ``(inlier_count, -residual_sum)``
    tuple under strict ``>``, which the two-step ``argmax``/``argmin`` below
    reproduces exactly, since both return the first occurrence of their
    extreme value on ties.
    """
    if len(source_centers) < 2:
        return None
    source = source_centers.detach().float()
    target = ego_centers.detach().float()
    n = len(source)

    pair_index = torch.combinations(torch.arange(n, device=source.device), r=2)
    pair_index = pair_index[:max_hypotheses]
    first_index, second_index = pair_index[:, 0], pair_index[:, 1]

    source_first, source_second = source[first_index], source[second_index]
    target_first, target_second = target[first_index], target[second_index]

    valid = (
        torch.linalg.vector_norm(source_second - source_first, dim=-1) >= 0.25
    )

    rotation, translation = _batched_se2_from_pairs(
        source_first, source_second, target_first, target_second
    )
    # (H, N, 2): every hypothesis's correction applied to every source point.
    transformed = (
        torch.einsum("hij,nj->hni", rotation, source) + translation.unsqueeze(1)
    )
    residual = torch.linalg.vector_norm(transformed - target.unsqueeze(0), dim=-1)
    inlier_mask = residual <= inlier_threshold
    inlier_count = inlier_mask.sum(dim=-1)
    inlier_residual_sum = (residual * inlier_mask).sum(dim=-1)

    # Invalidate near-duplicate-point hypotheses by giving them a count no
    # real hypothesis can reach, so they never win the argmax below.
    inlier_count = torch.where(valid, inlier_count, torch.full_like(inlier_count, -1))

    best_count = inlier_count.max()
    if best_count.item() < 0:
        return None
    tied = inlier_count == best_count
    tie_break = torch.where(
        tied, inlier_residual_sum, torch.full_like(inlier_residual_sum, float("inf"))
    )
    best_index = int(torch.argmin(tie_break))

    if int(best_count) < 2:
        return None
    rotation = rotation[best_index]
    translation = translation[best_index]
    inliers = inlier_mask[best_index]
    rotation, translation = _refine_se2(source[inliers], target[inliers])
    residual = torch.linalg.vector_norm(
        source @ rotation.T + translation - target, dim=-1
    )
    inliers = residual <= inlier_threshold
    if int(inliers.sum()) >= 2:
        rotation, translation = _refine_se2(source[inliers], target[inliers])
        residual = torch.linalg.vector_norm(
            source @ rotation.T + translation - target, dim=-1
        )
        inliers = residual <= inlier_threshold
    identity_residual = torch.linalg.vector_norm(source - target, dim=-1)
    identity_mean = identity_residual.mean()
    corrected_mean = residual.mean()
    identity_inlier_mean = identity_residual[inliers].mean()
    corrected_inlier_mean = residual[inliers].mean()
    angle = torch.atan2(rotation[1, 0], rotation[0, 0])
    return {
        "rotation": rotation,
        "translation": translation,
        "yaw_correction": angle,
        "inliers": inliers,
        "residual": residual,
        "identity_residual": identity_residual,
        "mean_identity_residual": identity_mean,
        "mean_corrected_residual": corrected_mean,
        "mean_identity_inlier_residual": identity_inlier_mean,
        "mean_corrected_inlier_residual": corrected_inlier_mean,
        "relative_improvement": (
            (identity_inlier_mean - corrected_inlier_mean)
            / identity_inlier_mean.clamp_min(1e-6)
        ),
    }


def correspondences_by_source(groups: Iterable[list[Mapping]]) -> dict[str, tuple[Tensor, Tensor]]:
    """Extract source-to-ego center correspondences from association groups."""
    collected: dict[str, tuple[list[Tensor], list[Tensor]]] = {}
    for group in groups:
        ego_members = [member for member in group if member.get("agent_id") == "ego"]
        if len(ego_members) != 1:
            continue
        ego_center = ego_members[0]["box"][:2]
        for member in group:
            source_id = member.get("agent_id")
            if source_id in (None, "ego"):
                continue
            sources, targets = collected.setdefault(source_id, ([], []))
            sources.append(member["box"][:2])
            targets.append(ego_center)
    return {
        source_id: (torch.stack(sources), torch.stack(targets))
        for source_id, (sources, targets) in collected.items()
        if len(sources) >= 2
    }


def apply_se2_correction(detection: Mapping[str, Tensor], correction: Mapping) -> dict:
    """Return a copy with box centers and yaw corrected in ego coordinates."""
    corrected = dict(detection)
    boxes = detection["boxes"].clone()
    rotation = correction["rotation"].to(device=boxes.device, dtype=boxes.dtype)
    translation = correction["translation"].to(device=boxes.device, dtype=boxes.dtype)
    boxes[:, :2] = boxes[:, :2] @ rotation.T + translation
    boxes[:, 6] = _wrap_angle(
        boxes[:, 6] + correction["yaw_correction"].to(boxes)
    )
    corrected["boxes"] = boxes
    return corrected
