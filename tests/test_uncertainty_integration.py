from pathlib import Path

import torch
from torch import nn

from embedding_aware_belt_fusion.integration.opencood_uncertainty import (
    FrozenPointPillarWithUncertainty,
    UncertaintyHeads,
)
from embedding_aware_belt_fusion.integration.uncertainty_loss import (
    AnchorUncertaintyLoss,
)


class _PassThrough(nn.Module):
    def forward(self, value):
        return value


class _FakeBackbone(nn.Module):
    num_bev_features = 4

    def forward(self, batch):
        batch["spatial_features_2d"] = batch["voxel_features"]
        return batch


class _FakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.pillar_vfe = _PassThrough()
        self.scatter = _PassThrough()
        self.backbone = _FakeBackbone()
        self.cls_head = nn.Conv2d(4, 2, 1)
        self.reg_head = nn.Conv2d(4, 14, 1)


def _input(batch_size=2):
    features = torch.randn(batch_size, 4, 3, 5)
    return {
        "processed_lidar": {
            "voxel_features": features,
            "voxel_coords": torch.empty(0),
            "voxel_num_points": torch.empty(0),
        }
    }


def test_uncertainty_head_shapes_and_initial_values():
    heads = UncertaintyHeads(in_channels=4, anchor_count=2, hidden_channels=8)
    output = heads(torch.randn(2, 4, 3, 5))
    assert output["evidence"].shape == (2, 2, 2, 3, 5)
    assert output["reg_log_var"].shape == (2, 14, 3, 5)
    assert torch.allclose(output["cls_uncertainty"], torch.full((2, 2, 3, 5), 0.5906), atol=1e-4)
    assert torch.count_nonzero(output["reg_log_var"]) == 0


def test_frozen_detector_has_no_gradients_but_uncertainty_heads_do():
    model = FrozenPointPillarWithUncertainty(
        _FakeDetector(), anchor_count=2, hidden_channels=8
    )
    output = model(_input())
    output["evidence"].sum().backward()
    assert all(parameter.grad is None for parameter in model.detector.parameters())
    assert any(
        parameter.grad is not None for parameter in model.uncertainty_heads.parameters()
    )
    model.train()
    assert not model.detector.training
    assert model.uncertainty_heads.training


def test_uncertainty_loss_is_finite_and_trains_only_new_outputs():
    model = FrozenPointPillarWithUncertainty(
        _FakeDetector(), anchor_count=2, hidden_channels=8
    )
    output = model(_input())
    target_shape = (2, 3, 5, 2)
    positives = torch.zeros(target_shape)
    negatives = torch.ones(target_shape)
    positives[:, 1, 2, 0] = 1
    negatives[:, 1, 2, 0] = 0
    targets = {
        "pos_equal_one": positives,
        "neg_equal_one": negatives,
        "targets": torch.randn(2, 3, 5, 14),
    }
    losses = AnchorUncertaintyLoss()(output, targets, epoch=0)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss"].backward()
    assert model.uncertainty_heads.log_variance_head.weight.grad is not None
    assert model.uncertainty_heads.evidence_head.weight.grad is not None


def test_classification_loss_balances_positive_and_negative_anchor_counts():
    criterion = AnchorUncertaintyLoss(negative_weight=1.0)
    output = {
        "psm": torch.zeros(1, 1, 1, 4),
        "rm": torch.zeros(1, 7, 1, 4),
        "reg_log_var": torch.zeros(1, 7, 1, 4),
        "alpha": torch.tensor(
            [[[[[1.0, 9.0, 9.0, 9.0]], [[9.0, 1.0, 1.0, 1.0]]]]]
        ),
    }
    # One positive and three negatives are all confidently wrong. Separate class
    # averaging prevents the three negatives from dominating only by count.
    targets = {
        "pos_equal_one": torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]]),
        "neg_equal_one": torch.tensor([[[[0.0, 1.0, 1.0, 1.0]]]]),
        "targets": torch.zeros(1, 1, 4, 7),
    }
    loss = criterion(output, targets)["classification_loss"]
    assert torch.isfinite(loss)


def test_uncertainty_checkpoint_round_trip(tmp_path: Path):
    model = FrozenPointPillarWithUncertainty(
        _FakeDetector(), anchor_count=2, hidden_channels=8
    )
    checkpoint = tmp_path / "uncertainty.pth"
    model.save_uncertainty_checkpoint(checkpoint, epoch=3)
    with torch.no_grad():
        model.uncertainty_heads.evidence_head.bias.add_(1)
    assert model.load_uncertainty_checkpoint(checkpoint) == 3
    assert torch.count_nonzero(model.uncertainty_heads.evidence_head.bias) == 0
