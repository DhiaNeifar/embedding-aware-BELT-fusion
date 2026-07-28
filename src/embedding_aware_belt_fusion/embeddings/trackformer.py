"""PointPillars adapter around TrackFormer's actual Transformer module."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def _load_transformer(trackformer_root: Path):
    source = trackformer_root / "src/trackformer/models/transformer.py"
    spec = importlib.util.spec_from_file_location("_trackformer_transformer", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load TrackFormer transformer from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Transformer


def _position_encoding(height, width, channels, device):
    if channels % 4:
        raise ValueError("d_model must be divisible by four")
    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    y = y.float() / max(height - 1, 1) * (2 * math.pi)
    x = x.float() / max(width - 1, 1) * (2 * math.pi)
    dim = torch.arange(channels // 4, device=device).float()
    dim = 10000 ** (2 * torch.div(dim, 2, rounding_mode="floor") / (channels // 4))
    x = x[..., None] / dim
    y = y[..., None] / dim
    encoding = torch.cat(
        [x.sin(), x.cos(), y.sin(), y.cos()], dim=-1
    )[..., :channels]
    return encoding.permute(2, 0, 1).unsqueeze(0)


class PointPillarsTrackFormer(nn.Module):
    def __init__(
        self,
        trackformer_root: Path,
        *,
        input_channels=384,
        d_model=256,
        embedding_dim=128,
        nhead=8,
        encoder_layers=6,
        decoder_layers=6,
        feedforward_dim=2048,
        dropout=0.1,
    ):
        super().__init__()
        Transformer = _load_transformer(trackformer_root)
        self.memory_projection = nn.Conv2d(input_channels, d_model, 1)
        self.query_content = nn.Linear(input_channels + 1, d_model)
        self.query_position = nn.Sequential(
            nn.Linear(7, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.register_buffer(
            "box_offset",
            torch.tensor([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
        )
        self.register_buffer(
            "box_scale",
            torch.tensor([140.8, 80.0, 4.0, 4.0, 4.0, 8.0, math.pi]),
        )
        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=encoder_layers,
            num_decoder_layers=decoder_layers,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            return_intermediate_dec=True,
        )
        self.embedding_projection = nn.Linear(d_model, embedding_dim)
        self.class_head = nn.Linear(d_model, 2)
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 7),
        )

    def _memory_and_queries(self, bev, query_features, boxes, scores):
        src = self.memory_projection(bev.unsqueeze(0))
        _, _, height, width = src.shape
        mask = torch.zeros((1, height, width), dtype=torch.bool, device=src.device)
        position = _position_encoding(
            height, width, src.shape[1], src.device
        ).to(src.dtype)
        normalized_boxes = (boxes - self.box_offset) / self.box_scale
        query_position = self.query_position(normalized_boxes).unsqueeze(1)
        target = self.query_content(
            torch.cat([query_features, scores[:, None]], dim=-1)
        ).unsqueeze(1)
        return src, mask, position, query_position, target

    def forward_agent_states(self, bev, query_features, boxes, scores):
        src, mask, position, query_position, target = self._memory_and_queries(
            bev, query_features, boxes, scores
        )
        _, states, _ = self.transformer(
            src, mask, query_position, position, target
        )
        return states[-1, 0]

    def forward_agent(self, bev, query_features, boxes, scores):
        states = self.forward_agent_states(
            bev, query_features, boxes, scores
        )
        return F.normalize(
            self.embedding_projection(states), dim=-1
        )

    def forward_with_track_queries(
        self,
        bev,
        query_features,
        boxes,
        scores,
        track_states,
    ):
        src, mask, position, query_position, target = self._memory_and_queries(
            bev, query_features, boxes, scores
        )
        track_count = track_states.shape[0]
        track_position = torch.zeros(
            (track_count, 1, query_position.shape[-1]),
            device=query_position.device,
            dtype=query_position.dtype,
        )
        query_position = torch.cat([track_position, query_position], dim=0)
        target = torch.cat([track_states.detach().unsqueeze(1), target], dim=0)
        _, states, _ = self.transformer(
            src, mask, query_position, position, target
        )
        states = states[-1, 0]
        return {
            "states": states,
            "embeddings": F.normalize(
                self.embedding_projection(states), dim=-1
            ),
            "logits": self.class_head(states),
            "boxes": self.box_head(states).sigmoid(),
            "track_count": track_count,
        }

    def normalize_boxes(self, boxes):
        box_min = torch.tensor(
            [-70.4, -40.0, -3.0, 0.0, 0.0, 0.0, -math.pi],
            device=boxes.device,
            dtype=boxes.dtype,
        )
        box_max = torch.tensor(
            [70.4, 40.0, 1.0, 4.0, 4.0, 8.0, math.pi],
            device=boxes.device,
            dtype=boxes.dtype,
        )
        return ((boxes - box_min) / (box_max - box_min)).clamp(0, 1)
