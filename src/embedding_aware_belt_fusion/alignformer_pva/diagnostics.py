"""Falsification-first diagnostics for soft correspondence pose estimation."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

_EPS = 1e-12


class AssignmentDiagnostics(NamedTuple):
    matched_mass: Tensor
    effective_rows: Tensor
    mean_row_entropy: Tensor
    mean_row_confidence: Tensor
    mean_top2_margin: Tensor
    mean_spatial_dispersion: Tensor


def assignment_diagnostics(
    weights: Tensor, cav_centres: Tensor, *, row_mask: Tensor | None = None, eps: float = _EPS
) -> AssignmentDiagnostics:
    """Summarize the real-object transport block, excluding empty rows."""
    if weights.dim() != 3:
        raise ValueError(f"weights must be (B,M,N), got {tuple(weights.shape)}")
    if cav_centres.shape != (weights.shape[0], weights.shape[2], 2):
        raise ValueError("cav_centres and weights disagree")
    weights = weights.clamp_min(0.0)
    mass = weights.sum(dim=-1)
    valid = mass > eps
    if row_mask is not None:
        if row_mask.shape != mass.shape:
            raise ValueError(f"row_mask must be {tuple(mass.shape)}")
        valid = valid & row_mask.bool()
    probabilities = weights / mass.clamp_min(eps).unsqueeze(-1)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=-1)
    confidence = probabilities.max(dim=-1).values
    if weights.shape[-1] >= 2:
        top2 = probabilities.topk(2, dim=-1).values
        margin = top2[..., 0] - top2[..., 1]
    else:
        margin = confidence
    mean = probabilities @ cav_centres
    squared_distance = (cav_centres.unsqueeze(1) - mean.unsqueeze(2)).square().sum(dim=-1)
    dispersion = (probabilities * squared_distance).sum(dim=-1)
    count = valid.sum(dim=-1).clamp_min(1)

    def valid_mean(value: Tensor) -> Tensor:
        return (value * valid.to(value.dtype)).sum(dim=-1) / count

    return AssignmentDiagnostics(
        matched_mass=(mass * valid.to(mass.dtype)).sum(dim=-1),
        effective_rows=valid.sum(dim=-1),
        mean_row_entropy=valid_mean(entropy),
        mean_row_confidence=valid_mean(confidence),
        mean_top2_margin=valid_mean(margin),
        mean_spatial_dispersion=valid_mean(dispersion),
    )


def apply_se2(points: Tensor, psi: Tensor, translation: Tensor) -> Tensor:
    """Apply batched SE(2) to ``(B,...,2)`` row-vector points."""
    cosine, sine = torch.cos(psi), torch.sin(psi)
    rotation = torch.stack([cosine, -sine, sine, cosine], dim=-1).reshape(-1, 2, 2)
    view = (translation.shape[0],) + (1,) * (points.dim() - 2) + (2,)
    return points @ rotation.transpose(-1, -2) + translation.reshape(view)


def centre_barycentric_costs(
    ego_centres: Tensor,
    cav_centres: Tensor,
    weights: Tensor,
    psi: Tensor,
    translation: Tensor,
    *,
    eps: float = _EPS,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return expanded cost, virtual cost, and pose-independent dispersion."""
    if ego_centres.shape != weights.shape[:2] + (2,):
        raise ValueError("ego_centres and weights disagree")
    if cav_centres.shape != (weights.shape[0], weights.shape[2], 2):
        raise ValueError("cav_centres and weights disagree")
    mass = weights.sum(dim=-1)
    barycentre = weights @ cav_centres / mass.clamp_min(eps).unsqueeze(-1)
    transformed_candidates = apply_se2(cav_centres, psi, translation)
    pair_residual = ego_centres.unsqueeze(2) - transformed_candidates.unsqueeze(1)
    expanded = (weights * pair_residual.square().sum(dim=-1)).sum(dim=(1, 2))
    transformed_barycentre = apply_se2(barycentre, psi, translation)
    virtual = (mass * (ego_centres - transformed_barycentre).square().sum(dim=-1)).sum(dim=-1)
    candidate_delta = cav_centres.unsqueeze(1) - barycentre.unsqueeze(2)
    dispersion = (weights * candidate_delta.square().sum(dim=-1)).sum(dim=(1, 2))
    return expanded, virtual, dispersion


def marginalized_information(
    hypothesis_probability: Tensor, gradients: Tensor, hessians: Tensor, *, symmetrize: bool = True
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute observed, complete, and missing correspondence information."""
    if gradients.shape[:2] != hypothesis_probability.shape:
        raise ValueError("gradients and hypothesis probabilities disagree")
    if hessians.shape[:3] != gradients.shape or hessians.shape[-1] != gradients.shape[-1]:
        raise ValueError("hessians must be (B,K,D,D)")
    probability = hypothesis_probability.clamp_min(0.0)
    probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(_EPS)
    complete = (probability[..., None, None] * hessians).sum(dim=1)
    mean_gradient = (probability[..., None] * gradients).sum(dim=1)
    centered = gradients - mean_gradient.unsqueeze(1)
    missing = (
        probability[..., None, None] * centered.unsqueeze(-1) * centered.unsqueeze(-2)
    ).sum(dim=1)
    observed = complete - missing
    if symmetrize:
        complete = 0.5 * (complete + complete.transpose(-1, -2))
        missing = 0.5 * (missing + missing.transpose(-1, -2))
        observed = 0.5 * (observed + observed.transpose(-1, -2))
    return observed, complete, missing


__all__ = [
    "AssignmentDiagnostics",
    "apply_se2",
    "assignment_diagnostics",
    "centre_barycentric_costs",
    "marginalized_information",
]
