"""Shared transformer over the ego and CAV object sets.

Both heads consume this trunk unchanged, so an A-vs-B comparison isolates the
head. Cross-attention between the two sets supplies the multiplicative ego x CAV
interaction whose absence made CoLoca-QuA's linear tokenization unable to
register; see docs/coloca_qua_baseline.md.

Object sets are unordered, so the trunk carries no positional encoding - spatial
information enters only through each token's own box geometry.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

DEFAULT_MODEL_DIM = 256
DEFAULT_LAYERS = 4
DEFAULT_HEADS = 4
# Token budget per agent; objects beyond this are dropped by descending score.
MAX_OBJECTS = 64

# 8 geometry features plus the detector score.
GEOMETRY_FEATURES = 9
# Index of the yaw field in our detector's box layout: [x, y, z, h, w, l, yaw].
# Hardcoded here (rather than imported from a shared boxes module, as the
# original alignformer backend does) since this backend's detector pipeline
# already commits to this exact "hwl" layout, confirmed compatible end to end
# (boxes + scores + embeddings is exactly the (boxes, scores, embeddings)
# shape tokenize() expects below). If that layout ever changes, update this
# constant to match.
BOX_YAW = 6


def tokenize(boxes: Tensor, scores: Tensor, embeddings: Tensor) -> Tensor:
    """Build raw token features from boxes, scores and embeddings.

    Yaw is encoded as ``(cos, sin)`` so the representation is continuous across
    the +/-pi wrap.

    Parameters
    ----------
    boxes: ``(B, N, 7)`` in ``hwl`` order, already in the ego frame.
    scores: ``(B, N)`` detector confidences.
    embeddings: ``(B, N, d)`` unit-norm per-object descriptors.

    Returns
    -------
    Tensor
        ``(B, N, GEOMETRY_FEATURES + d)``.
    """
    yaw = boxes[..., BOX_YAW]
    geometry = torch.cat(
        [
            boxes[..., :6],
            torch.cos(yaw).unsqueeze(-1),
            torch.sin(yaw).unsqueeze(-1),
            scores.unsqueeze(-1),
        ],
        dim=-1,
    )
    return torch.cat([geometry, embeddings], dim=-1)


def _attend_without_degenerate_rows(
    attention: nn.MultiheadAttention, query: Tensor, key: Tensor, value: Tensor, valid_keys: Tensor
) -> Tensor:
    """Run ``attention`` while avoiding NaN for batch items with no valid keys.

    R27: ``nn.MultiheadAttention`` computes softmax over an all ``-inf`` row
    when every key in a batch item is masked out -- 0/0 -- which is a real
    case here (an agent that detected nothing in this frame), not a synthetic
    edge case. That NaN is not just in the forward output, where it could be
    patched after the call with e.g. ``torch.nan_to_num``: it is inside the
    attention weights themselves, which the backward pass differentiates
    through directly, so gradients for shared parameters (the projections,
    norms, feed-forward) come back NaN too -- confirmed empirically, and it
    poisons every batch item's gradient, not just the degenerate one's, since
    those parameters are shared across the batch.

    The fix is to never let the pathological all-masked row reach softmax at
    all: unmask one arbitrary key (position 0) for exactly those batch items,
    run attention, then zero the (now finite, but meaningless) output for
    those same items. Zeroing is the semantically correct value too -- an
    empty key set should contribute no attention update -- and unlike
    ``nan_to_num`` this keeps every intermediate value finite, so gradients
    computed through it are finite as well.

    Requires ``valid_keys.shape[1] > 0`` (a real, non-empty key dimension):
    the caller (``AlignFormerA``/``AlignFormerB`` via ``_is_empty``) must
    already have excluded a wholly empty ego/cav set before invoking the
    trunk at all, since there is no key position 0 to unmask otherwise.
    """
    if valid_keys.shape[1] == 0:
        raise ValueError(
            "valid_keys has zero keys (shape "
            f"{tuple(valid_keys.shape)}); the caller must exclude a wholly "
            "empty object set before calling attention -- see _is_empty in "
            "model.py, which every caller of the trunk already checks"
        )

    fully_masked = ~valid_keys.any(dim=1)
    # ``~valid_keys`` already allocates a fresh tensor, so no further .clone()
    # is needed before writing into it below.
    key_padding_mask = ~valid_keys
    if fully_masked.any():
        key_padding_mask[fully_masked, 0] = False

    attended, _ = attention(
        query, key, value, key_padding_mask=key_padding_mask, need_weights=False
    )
    if fully_masked.any():
        attended = attended.masked_fill(fully_masked.view(-1, 1, 1), 0.0)
    return attended


class _Layer(nn.Module):
    """One self-attention pass within each set, then cross-attention across."""

    def __init__(self, model_dim: int, heads: int) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(model_dim, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(model_dim, heads, batch_first=True)
        self.norm_self = nn.LayerNorm(model_dim)
        self.norm_cross = nn.LayerNorm(model_dim)
        self.norm_ff = nn.LayerNorm(model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Linear(4 * model_dim, model_dim),
        )

    def forward(self, x: Tensor, y: Tensor, x_mask: Tensor, y_mask: Tensor) -> Tensor:
        normed = self.norm_self(x)
        attended = _attend_without_degenerate_rows(
            self.self_attention, normed, normed, normed, x_mask
        )
        x = x + attended

        normed_x, normed_y = self.norm_cross(x), self.norm_cross(y)
        # A batch item whose OWN set (x) is empty also has no valid queries
        # here, so its cross-attention output does not matter; a batch item
        # whose *other* set (y) is empty is exactly the degenerate-row case
        # _attend_without_degenerate_rows exists for, and it is handled the
        # same way regardless of which side is empty.
        attended = _attend_without_degenerate_rows(
            self.cross_attention, normed_x, normed_y, normed_y, y_mask
        )
        x = x + attended

        return x + self.feed_forward(self.norm_ff(x))


class AlignFormerTrunk(nn.Module):
    """Interleaved self- and cross-attention over the two object sets."""

    def __init__(
        self,
        embed_dim: int,
        model_dim: int = DEFAULT_MODEL_DIM,
        layers: int = DEFAULT_LAYERS,
        heads: int = DEFAULT_HEADS,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(GEOMETRY_FEATURES + embed_dim, model_dim)
        self.set_embedding = nn.Parameter(torch.zeros(2, model_dim))
        self.ego_layers = nn.ModuleList(_Layer(model_dim, heads) for _ in range(layers))
        self.cav_layers = nn.ModuleList(_Layer(model_dim, heads) for _ in range(layers))
        self.output_norm = nn.LayerNorm(model_dim)
        self.model_dim = model_dim

    def forward(
        self, ego_tokens: Tensor, cav_tokens: Tensor, ego_mask: Tensor, cav_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Encode both sets, each attending to itself and to the other."""
        ego = self.input_projection(ego_tokens) + self.set_embedding[0]
        cav = self.input_projection(cav_tokens) + self.set_embedding[1]

        for ego_layer, cav_layer in zip(self.ego_layers, self.cav_layers):
            next_ego = ego_layer(ego, cav, ego_mask, cav_mask)
            next_cav = cav_layer(cav, ego, cav_mask, ego_mask)
            ego, cav = next_ego, next_cav

        ego = self.output_norm(ego) * ego_mask.unsqueeze(-1)
        cav = self.output_norm(cav) * cav_mask.unsqueeze(-1)
        return ego, cav
