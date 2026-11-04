"""Context-padded per-object embedding encoder for ``alignformer_pva``.

Phase 1 of the richer-embedding plan: reads the per-proposal padded-cell
token/position/footprint-flag tensors produced by
``data/materialize_context_roi_cache.py`` (own exact box footprint PLUS a
configurable metre margin of surrounding context) and produces one fixed-size
embedding vector per proposal.

Reuses the feature pattern already proven out in
``embeddings/spatial_trackformer.py``/``experiments/overfit_complete_roi.py``:
project native BEV cell tokens, add relative cell-position and per-proposal
geometry embeddings, then attend over the cell set. The difference here is a
richer INPUT (padded cells plus an own-footprint-vs-context flag per cell),
not a different attention mechanism.

Deliberately does NOT reuse ``SpatialTrackFormer``/``CompleteROIEncoder``'s
vendored DETR/TrackFormer transformer (``embeddings/trackformer.py:
_load_transformer``). That machinery exists to propagate DECODER queries
across time or across CAVs for its own detection/association objective
(learned object queries, a Hungarian matcher, box/class heads) -- none of
which this module needs: ``alignformer_pva`` already supplies its own
Sinkhorn soft-matching and closed-form Kabsch pose solve
(``alignformer_pva/model.py``) over whatever per-object descriptor is handed
to it via ``alignformer_pva/trunk.py:tokenize``. All this module has to do is
turn ONE proposal's own padded cell set into ONE fixed-size vector, which is
exactly what a plain ``nn.TransformerEncoder`` self-attention stack does,
with no decoder, no queries, and no extra loss of its own -- it trains
end-to-end purely from the signal that already flows back through
``alignformer_pva``'s ``match_nll`` + ``corner_loss`` (``alignformer_pva/
losses.py``). This keeps the embedding path a strict dependency-weight
*decrease* relative to the vendored-DETR alternative, while reusing the same
proven cell-token/position/geometry feature recipe.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

# Matches alignformer_pva.trunk.GEOMETRY_FEATURES / tokenize()'s geometry
# half exactly (6 box dims + cos/sin yaw + score). Not imported from there:
# the dependency direction is the other way around (alignformer_pva/
# dataset.py and train.py read THIS module's output), so alignformer_pva must
# never be an import-time dependency of embeddings/context_embedding.py.
GEOMETRY_FEATURES = 9
# Index of the yaw field in this repo's [x, y, z, h, w, l, yaw] box layout.
BOX_YAW = 6


def proposal_geometry(boxes: Tensor, scores: Tensor) -> Tensor:
    """9-D per-proposal geometry vector: boxes[:6], cos(yaw), sin(yaw), score.

    Same convention as ``alignformer_pva.trunk.tokenize``'s geometry half,
    reimplemented here as a tiny free function rather than imported, per this
    module's one-way dependency direction (see module docstring).
    """
    yaw = boxes[..., BOX_YAW]
    return torch.cat(
        [
            boxes[..., :6],
            torch.cos(yaw).unsqueeze(-1),
            torch.sin(yaw).unsqueeze(-1),
            scores.unsqueeze(-1),
        ],
        dim=-1,
    )


class ContextEmbedding(nn.Module):
    """Self-attention encoder over one proposal's padded (footprint + context) cells.

    Call :meth:`forward` once per agent, batching that agent's proposals
    along dimension 0 (``P``); it tolerates ``P == 0`` (an agent with no
    proposals) by returning an empty ``(0, embed_dim)`` tensor.

    Parameters to :meth:`forward`
    ------------------------------
    tokens: ``(P, K, input_dim)`` native BEV cell features, zero-padded to
        this agent's own max cell count ``K``.
    positions: ``(P, K, 2)`` relative cell layout (``materialize_context_roi_
        cache.py``'s own centred/scaled row-column convention), zero-padded.
    footprint_flag: ``(P, K)`` bool -- True for a cell inside the proposal's
        own exact rotated box footprint, False for a cell only inside the
        padding margin. Meaningless at padded slots (``pad_mask`` excludes
        them from every reduction below).
    pad_mask: ``(P, K)`` bool -- True marks a PADDED (invalid) slot, matching
        the convention ``materialize_roi_cache.py``/``materialize_context_roi_
        cache.py`` both already use ("false entries are the real cells").
    boxes: ``(P, 7)``, scores: ``(P,)`` -- this proposal's own geometry/score,
        feeding the per-proposal geometry embedding (not the cell attention
        keys/values themselves).

    Returns
    -------
    Tensor
        ``(P, embed_dim)``, L2-normalized. Normalization is NOT required by
        anything downstream in this repository today: ``alignformer_pva``'s
        trunk (``alignformer_pva/trunk.py:tokenize``) concatenates whatever
        is handed to it with the 9-D geometry vector and learns its own input
        projection from there, so the trunk itself is scale-agnostic to this
        embedding. The one place raw embedding scale WOULD matter is
        ``AlignFormerB``'s test-only ``use_raw_embedding_scores`` hook
        (``alignformer_pva/model.py:404,423-425``), which bypasses the
        trunk's projection entirely and scores raw embeddings against each
        other directly -- that hook expects directly comparable (and, in
        practice, unit-scale) descriptors. This encoder is not required to
        satisfy that property for ordinary (trunk-fed) training or inference,
        but normalizing is a single free `F.normalize` call that costs
        nothing and keeps the embedding well-scaled for the trunk's own input
        projection regardless, and incidentally also makes it compatible with
        that raw-score test hook if it is ever pointed at this embedding
        instead of the geometry-only one. Kept on for both reasons.
    """

    def __init__(
        self,
        input_dim: int = 384,
        d_model: int = 128,
        embed_dim: int = 64,
        heads: int = 4,
        layers: int = 2,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_projection = nn.Linear(input_dim, d_model)
        self.cell_position = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        # Index 0 = padding-context cell, index 1 = own-footprint cell.
        self.footprint_embedding = nn.Embedding(2, d_model)
        self.proposal_geometry_embedding = nn.Sequential(
            nn.Linear(GEOMETRY_FEATURES, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, embed_dim)
        self.d_model = d_model
        self.embed_dim = embed_dim

    def forward(
        self,
        tokens: Tensor,
        positions: Tensor,
        footprint_flag: Tensor,
        pad_mask: Tensor,
        boxes: Tensor,
        scores: Tensor,
    ) -> Tensor:
        proposal_count = tokens.shape[0]
        if proposal_count == 0:
            return tokens.new_zeros((0, self.embed_dim))

        geometry = proposal_geometry(boxes.to(tokens.dtype), scores.to(tokens.dtype))
        cell_state = (
            self.token_projection(tokens)
            + self.cell_position(positions.to(tokens.dtype))
            + self.footprint_embedding(footprint_flag.long())
            + self.proposal_geometry_embedding(geometry).unsqueeze(1)
        )

        valid = ~pad_mask
        # A proposal with literally zero valid cells cannot happen by
        # construction (materialize_context_roi_cache.py's fallback always
        # keeps at least one real cell per proposal, mirroring
        # proposal_roi_cell_indices' own single-nearest-cell fallback), but
        # nn.TransformerEncoderLayer's self-attention has the exact same
        # fully-masked-row softmax/NaN hazard alignformer_pva.trunk documents
        # and guards against (_attend_without_degenerate_rows): if every key
        # in a row is masked, the normalization is 0/0 and the resulting NaN
        # poisons the backward pass for every row in the batch, not just the
        # degenerate one, since the encoder's weights are shared across rows.
        # Guard defensively the same way: unmask position 0 for any such row,
        # run the encoder, then zero its output.
        fully_masked = ~valid.any(dim=1)
        key_padding_mask = pad_mask.clone()
        if fully_masked.any():
            key_padding_mask[fully_masked, 0] = False

        encoded = self.encoder(cell_state, src_key_padding_mask=key_padding_mask)
        if fully_masked.any():
            encoded = encoded.masked_fill(fully_masked.view(-1, 1, 1), 0.0)

        weights = valid.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        pooled = self.output_norm(pooled)
        embedding = self.output_projection(pooled)
        return nn.functional.normalize(embedding, dim=-1)
