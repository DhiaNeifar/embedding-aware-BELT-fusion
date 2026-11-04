"""AlignFormerTrunk extended with CAV-CAV context cross-attention.

Additive, new file: does not subclass or modify ``trunk.py``'s
``AlignFormerTrunk``. Built to attack the dense-scene ("blur") weakness
measured on OPV2V frames with 3+ CAVs (see the plan's 2026-10-29 update):
AlignFormer loses to FreeAlign there by a fixed ~0.02 AP@0.7 regardless of
noise, traced to the Sinkhorn soft-assignment blurring over more candidate
objects in denser scenes, not to any imprecision an after-the-fact fix
(``evaluation``-side ``cav_cav_consensus``) could clean up.

This is the conservative version of "joint multi-agent Sinkhorn": rather
than redesigning the trunk for a true N-way exchange among every agent, the
CAV being solved for (and, symmetrically, ego) additionally cross-attends to
one extra token set -- the concatenation of every OTHER CAV's own tokens in
the same frame ("context"). The existing ego/CAV cross-attention, and the
Sinkhorn matching that follows it, are unchanged in shape and meaning; only
the tokens feeding them have seen more of the scene first.

Reuses ``trunk.py``'s ``_Layer``/``_attend_without_degenerate_rows`` directly
(imported, not copied): both already operate on generic ``(B, N, D)`` token
tensors plus a boolean mask with no assumption that there are only ever two
sets in the picture, confirmed by reading them before writing this file.

A frame with exactly one CAV has an empty (``K=0``) context set. Concatenating
a ``(B, 0, D)`` tensor onto ``cav``/``ego`` is a true no-op (zero rows,
dimensions line up, nothing is added), so the single-CAV case -- the large
majority of frames -- is architecturally IDENTICAL to ``AlignFormerTrunk``,
not just expected to behave similarly to it.

Context tokens reuse the "CAV" role embedding (set_embedding[1]): a context
token is the same *kind* of thing as the CAV being solved for (another CAV's
own detection), just serving as side information rather than the query being
matched, so introducing a third role embedding was judged unnecessary extra
capacity for what is deliberately a minimal extension.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from embedding_aware_belt_fusion.alignformer_pva.trunk import (
    DEFAULT_HEADS,
    DEFAULT_LAYERS,
    DEFAULT_MODEL_DIM,
    GEOMETRY_FEATURES,
    _Layer,
)


class AlignFormerTrunkWithContext(nn.Module):
    """Like ``AlignFormerTrunk``, plus a CAV-CAV context cross-attention step."""

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
        self,
        ego_tokens: Tensor,
        cav_tokens: Tensor,
        ego_mask: Tensor,
        cav_mask: Tensor,
        context_tokens: Tensor,
        context_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Encode ego and the query CAV, each also seeing the other CAVs in the frame.

        ``context_tokens``/``context_mask``: ``(B, K, GEOMETRY_FEATURES +
        embed_dim)`` / ``(B, K)``, the concatenated raw tokens of every OTHER
        CAV in the same frame as the query CAV (``K = 0`` for a single-CAV
        frame). Projected and role-tagged exactly like ``cav_tokens``, then
        concatenated onto ego's and the query CAV's cross-attention targets
        so both can attend to it -- context itself is not re-embedded or
        updated across layers, it stays a fixed side signal.
        """
        ego = self.input_projection(ego_tokens) + self.set_embedding[0]
        cav = self.input_projection(cav_tokens) + self.set_embedding[1]
        context = self.input_projection(context_tokens) + self.set_embedding[1]

        for ego_layer, cav_layer in zip(self.ego_layers, self.cav_layers):
            cav_with_context = torch.cat([cav, context], dim=1)
            cav_with_context_mask = torch.cat([cav_mask, context_mask], dim=1)
            next_ego = ego_layer(ego, cav_with_context, ego_mask, cav_with_context_mask)

            ego_with_context = torch.cat([ego, context], dim=1)
            ego_with_context_mask = torch.cat([ego_mask, context_mask], dim=1)
            next_cav = cav_layer(cav, ego_with_context, cav_mask, ego_with_context_mask)

            ego, cav = next_ego, next_cav

        ego = self.output_norm(ego) * ego_mask.unsqueeze(-1)
        cav = self.output_norm(cav) * cav_mask.unsqueeze(-1)
        return ego, cav
