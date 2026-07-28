"""Geometry-aware proposal embeddings in the shared ego coordinate frame."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from embedding_aware_belt_fusion.experiments.overfit_complete_roi import (
    CompleteROIEncoder,
)


GEOMETRY_DIM = 9


def transform_boxes_to_ego(boxes, transformation_matrix):
    """Transform ``[x,y,z,h,w,l,yaw]`` boxes from one agent to ego."""
    if boxes.ndim != 2 or boxes.shape[-1] != 7:
        raise ValueError("boxes must have shape [proposal_count, 7]")
    if transformation_matrix.shape != (4, 4):
        raise ValueError("transformation_matrix must have shape [4, 4]")
    boxes = boxes.float()
    transformation_matrix = transformation_matrix.to(
        device=boxes.device,
        dtype=boxes.dtype,
    )
    homogeneous_centers = F.pad(
        boxes[:, :3], (0, 1), mode="constant", value=1
    )
    ego_centers = homogeneous_centers @ transformation_matrix.T
    rotation_yaw = torch.atan2(
        transformation_matrix[1, 0],
        transformation_matrix[0, 0],
    )
    ego_yaw = boxes[:, 6] + rotation_yaw
    ego_yaw = torch.atan2(torch.sin(ego_yaw), torch.cos(ego_yaw))
    return torch.cat(
        [ego_centers[:, :3], boxes[:, 3:6], ego_yaw[:, None]],
        dim=-1,
    )


def normalized_ego_geometry(boxes, scores, transformation_matrix):
    """Return stable geometry features used by the association encoder."""
    ego_boxes = transform_boxes_to_ego(boxes, transformation_matrix)
    scores = scores.float().reshape(-1, 1)
    if len(scores) != len(ego_boxes):
        raise ValueError("scores and boxes must contain the same proposals")
    # OPV2V/PointPillars uses approximately +/-70.4 m longitudinal,
    # +/-40 m lateral, and a 4 m vertical span. Vehicle sizes are normalized
    # separately so every input component has a comparable numerical scale.
    scales = ego_boxes.new_tensor(
        [70.4, 40.0, 4.0, 4.0, 4.0, 10.0]
    )
    return torch.cat(
        [
            ego_boxes[:, :6] / scales,
            torch.sin(ego_boxes[:, 6:7]),
            torch.cos(ego_boxes[:, 6:7]),
            scores.clamp(0.0, 1.0),
        ],
        dim=-1,
    )


class GeometryAwareROIEncoder(nn.Module):
    """Fuse complete-ROI appearance with ego-frame proposal geometry."""

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
        self.appearance = CompleteROIEncoder(
            trackformer_root=trackformer_root,
            input_dim=input_dim,
            d_model=d_model,
            embedding_dim=embedding_dim,
            heads=heads,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        self.geometry = nn.Sequential(
            nn.LayerNorm(GEOMETRY_DIM),
            nn.Linear(GEOMETRY_DIM, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, embedding_dim),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, embedding_dim),
        )

    def forward(self, tokens, positions, padding_mask, geometry):
        if geometry.ndim != 2 or geometry.shape[-1] != GEOMETRY_DIM:
            raise ValueError(
                f"geometry must have shape [proposal_count, {GEOMETRY_DIM}]"
            )
        appearance = self.appearance(tokens, positions, padding_mask)
        geometry_embedding = self.geometry(geometry)
        fused = self.fusion(
            torch.cat([appearance, geometry_embedding], dim=-1)
        )
        return F.normalize(fused, dim=-1)
