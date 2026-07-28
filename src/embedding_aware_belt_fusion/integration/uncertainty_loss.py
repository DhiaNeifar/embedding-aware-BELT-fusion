"""Losses for anchor-aligned BELT-style uncertainty prediction."""

from __future__ import annotations

from typing import Dict, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _anchor_regression(tensor: Tensor) -> Tensor:
    """Convert `[B, A*7, H, W]` to `[B, H*W*A, 7]`."""
    return tensor.permute(0, 2, 3, 1).contiguous().view(tensor.shape[0], -1, 7)


def _anchor_evidence(tensor: Tensor) -> Tensor:
    """Convert `[B, A, K, H, W]` to `[B, H*W*A, K]`."""
    return tensor.permute(0, 3, 4, 1, 2).contiguous().view(
        tensor.shape[0], -1, tensor.shape[2]
    )


class AnchorUncertaintyLoss(nn.Module):
    """Heteroscedastic box loss plus two-outcome evidential classification."""

    def __init__(
        self,
        regression_weight: float = 1.0,
        classification_weight: float = 1.0,
        kl_weight: float = 1e-3,
        anneal_epochs: int = 10,
        negative_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        self.kl_weight = kl_weight
        self.anneal_epochs = anneal_epochs
        self.negative_weight = negative_weight

    @staticmethod
    def _dirichlet_kl_to_uniform(alpha: Tensor) -> Tensor:
        class_count = alpha.shape[-1]
        sum_alpha = alpha.sum(dim=-1, keepdim=True)
        log_normalizer_ratio = torch.lgamma(sum_alpha.squeeze(-1)) - torch.lgamma(
            alpha
        ).sum(dim=-1) - torch.lgamma(
            torch.tensor(float(class_count), device=alpha.device, dtype=alpha.dtype)
        )
        digamma = ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(sum_alpha)))
        return log_normalizer_ratio + digamma.sum(dim=-1)

    def forward(
        self,
        outputs: Mapping[str, Tensor],
        targets: Mapping[str, Tensor],
        *,
        epoch: int = 0,
    ) -> Dict[str, Tensor]:
        positive = targets["pos_equal_one"].reshape(outputs["psm"].shape[0], -1) > 0
        negative = targets["neg_equal_one"].reshape(outputs["psm"].shape[0], -1) > 0

        mean = _anchor_regression(outputs["rm"].detach())
        log_variance = _anchor_regression(outputs["reg_log_var"])
        regression_target = targets["targets"].reshape(mean.shape)
        regression_error = (mean - regression_target).square()
        heteroscedastic = 0.5 * (
            torch.exp(-log_variance) * regression_error + log_variance
        ).sum(dim=-1)
        positive_count = positive.sum().clamp(min=1)
        regression_loss = (heteroscedastic * positive).sum() / positive_count

        alpha = _anchor_evidence(outputs["alpha"])
        labels = positive.long()
        one_hot = F.one_hot(labels, num_classes=alpha.shape[-1]).to(alpha.dtype)
        strength = alpha.sum(dim=-1, keepdim=True)
        expected_probability = alpha / strength
        edl_error = ((one_hot - expected_probability) ** 2).sum(dim=-1)
        edl_variance = (
            alpha * (strength - alpha) / (strength.square() * (strength + 1.0))
        ).sum(dim=-1)

        valid = positive | negative
        edl = edl_error + edl_variance
        positive_denominator = positive.sum().clamp(min=1)
        negative_denominator = negative.sum().clamp(min=1)
        positive_edl = (edl * positive).sum() / positive_denominator
        negative_edl = (edl * negative).sum() / negative_denominator
        classification_loss = (
            positive_edl + self.negative_weight * negative_edl
        ) / (1.0 + self.negative_weight)

        non_target_alpha = one_hot + (1.0 - one_hot) * alpha
        kl = self._dirichlet_kl_to_uniform(non_target_alpha)
        anneal = min(1.0, float(epoch + 1) / max(1, self.anneal_epochs))
        positive_kl = (kl * positive).sum() / positive_denominator
        negative_kl = (kl * negative).sum() / negative_denominator
        kl_loss = (
            positive_kl + self.negative_weight * negative_kl
        ) / (1.0 + self.negative_weight)

        total = (
            self.regression_weight * regression_loss
            + self.classification_weight * classification_loss
            + self.kl_weight * anneal * kl_loss
        )
        return {
            "loss": total,
            "regression_loss": regression_loss,
            "classification_loss": classification_loss,
            "kl_loss": kl_loss,
            "positive_anchors": positive_count.detach(),
        }
