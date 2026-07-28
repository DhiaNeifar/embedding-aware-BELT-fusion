"""Trainable uncertainty heads around a frozen OpenCOOD PointPillar detector."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class UncertaintyHeads(nn.Module):
    """Anchor-aligned evidential classification and regression variance heads."""

    def __init__(
        self,
        in_channels: int,
        anchor_count: int,
        hidden_channels: int = 128,
        class_count: int = 2,
        min_log_variance: float = -10.0,
        max_log_variance: float = 10.0,
    ) -> None:
        super().__init__()
        if class_count < 2:
            raise ValueError("class_count must include background and at least one object class")
        self.anchor_count = anchor_count
        self.class_count = class_count
        self.min_log_variance = min_log_variance
        self.max_log_variance = max_log_variance

        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.evidence_head = nn.Conv2d(
            hidden_channels, anchor_count * class_count, kernel_size=1
        )
        self.log_variance_head = nn.Conv2d(
            hidden_channels, anchor_count * 7, kernel_size=1
        )
        nn.init.zeros_(self.evidence_head.weight)
        nn.init.zeros_(self.evidence_head.bias)
        nn.init.zeros_(self.log_variance_head.weight)
        nn.init.zeros_(self.log_variance_head.bias)

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        hidden = self.shared(features)
        batch_size, _, height, width = hidden.shape
        evidence = F.softplus(self.evidence_head(hidden))
        evidence = evidence.view(
            batch_size, self.anchor_count, self.class_count, height, width
        )
        alpha = evidence + 1.0
        class_uncertainty = self.class_count / alpha.sum(dim=2)

        regression_log_variance = self.log_variance_head(hidden).clamp(
            self.min_log_variance, self.max_log_variance
        )
        return {
            "evidence": evidence,
            "alpha": alpha,
            "cls_uncertainty": class_uncertainty,
            "reg_log_var": regression_log_variance,
        }


class FrozenPointPillarWithUncertainty(nn.Module):
    """Run a frozen OpenCOOD detector and train only uncertainty heads."""

    def __init__(
        self,
        detector: nn.Module,
        anchor_count: int,
        hidden_channels: int = 128,
        freeze_detector: bool = True,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.freeze_detector = freeze_detector
        in_channels = int(detector.backbone.num_bev_features)
        self.uncertainty_heads = UncertaintyHeads(
            in_channels=in_channels,
            anchor_count=anchor_count,
            hidden_channels=hidden_channels,
        )
        if freeze_detector:
            self._freeze_detector()

    def _freeze_detector(self) -> None:
        self.detector.requires_grad_(False)
        self.detector.eval()

    def train(self, mode: bool = True) -> "FrozenPointPillarWithUncertainty":
        super().train(mode)
        if self.freeze_detector:
            self.detector.eval()
        return self

    def _extract_bev_features(self, data_dict: Mapping[str, Mapping[str, Tensor]]) -> Tensor:
        processed_lidar = data_dict["processed_lidar"]
        batch_dict = {
            "voxel_features": processed_lidar["voxel_features"],
            "voxel_coords": processed_lidar["voxel_coords"],
            "voxel_num_points": processed_lidar["voxel_num_points"],
        }
        context = torch.no_grad() if self.freeze_detector else torch.enable_grad()
        with context:
            batch_dict = self.detector.pillar_vfe(batch_dict)
            batch_dict = self.detector.scatter(batch_dict)
            batch_dict = self.detector.backbone(batch_dict)
        return batch_dict["spatial_features_2d"]

    def forward(self, data_dict: Mapping[str, Mapping[str, Tensor]]) -> Dict[str, Tensor]:
        features = self._extract_bev_features(data_dict)
        context = torch.no_grad() if self.freeze_detector else torch.enable_grad()
        with context:
            psm = self.detector.cls_head(features)
            rm = self.detector.reg_head(features)
        output = {"psm": psm, "rm": rm, "spatial_features_2d": features}
        output.update(self.uncertainty_heads(features))
        return output

    def uncertainty_state_dict(self) -> Dict[str, Tensor]:
        return self.uncertainty_heads.state_dict()

    def save_uncertainty_checkpoint(
        self, path: Path, *, epoch: int, optimizer: torch.optim.Optimizer | None = None
    ) -> None:
        payload = {
            "epoch": epoch,
            "uncertainty_state_dict": self.uncertainty_state_dict(),
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_uncertainty_checkpoint(self, path: Path, map_location: str = "cpu") -> int:
        payload = torch.load(path, map_location=map_location)
        self.uncertainty_heads.load_state_dict(payload["uncertainty_state_dict"])
        return int(payload.get("epoch", 0))

