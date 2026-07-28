"""Compact proposal embedding model."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ProposalEmbeddingHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dims=(256, 128),
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dimensions = (input_dim, *hidden_dims)
        layers = []
        for input_size, output_size in zip(dimensions[:-1], dimensions[1:]):
            layers.extend(
                [
                    nn.Linear(input_size, output_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(dimensions[-1], embedding_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        return F.normalize(self.network(features), dim=-1)


class CrossAgentSupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: Tensor, labels: Tensor, agents: Tensor) -> Tensor:
        count = embeddings.shape[0]
        identity = torch.eye(count, dtype=torch.bool, device=embeddings.device)
        cross_agent = agents[:, None] != agents[None, :]
        positives = (labels[:, None] == labels[None, :]) & cross_agent & ~identity
        valid_anchor = positives.any(dim=1)
        if not valid_anchor.any():
            return embeddings.sum() * 0.0

        logits = embeddings @ embeddings.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * ~identity
        log_probability = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
        )
        positive_count = positives.sum(dim=1).clamp_min(1)
        mean_positive_log_probability = (
            log_probability * positives
        ).sum(dim=1) / positive_count
        return -mean_positive_log_probability[valid_anchor].mean()
