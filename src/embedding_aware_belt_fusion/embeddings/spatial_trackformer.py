"""Complete TrackFormer query propagation adapted from time to CAV space."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from embedding_aware_belt_fusion.embeddings.geometry import (
    normalized_ego_geometry,
    transform_boxes_to_ego,
)
from embedding_aware_belt_fusion.embeddings.trackformer import _load_transformer
from embedding_aware_belt_fusion.experiments.overfit_complete_roi import (
    _best_proposals,
)


def normalize_ego_boxes(boxes):
    """Encode ego-frame ``[x,y,z,h,w,l,yaw]`` boxes into DETR targets."""
    scale = boxes.new_tensor([140.8, 80.0, 4.0, 4.0, 80.0, 140.8])
    offset = boxes.new_tensor([70.4, 40.0, 3.0, 0.0, 0.0, 0.0])
    linear = (boxes[:, :6] + offset) / scale
    return torch.cat(
        [
            linear,
            (torch.sin(boxes[:, 6:7]) + 1.0) / 2.0,
            (torch.cos(boxes[:, 6:7]) + 1.0) / 2.0,
        ],
        dim=-1,
    ).clamp(0.0, 1.0)


def spatial_target(agent, device):
    """Build unique vehicle targets and stable physical IDs for one CAV."""
    proposals = _best_proposals(agent)
    object_ids = sorted(proposals)
    if not object_ids:
        return {
            "labels": torch.zeros(0, dtype=torch.long, device=device),
            "boxes": torch.zeros(0, 8, device=device),
            "object_ids": [],
        }
    indices = torch.tensor(
        [proposals[object_id] for object_id in object_ids],
        dtype=torch.long,
    )
    boxes = transform_boxes_to_ego(
        agent["gt_boxes"][indices],
        agent["transformation_matrix"],
    ).to(device)
    return {
        "labels": torch.zeros(len(indices), dtype=torch.long, device=device),
        "boxes": normalize_ego_boxes(boxes),
        "object_ids": object_ids,
    }


class SpatialTrackFormer(nn.Module):
    """DETR/TrackFormer over all PointPillars ROI cells from one agent."""

    def __init__(
        self,
        *,
        trackformer_root,
        input_dim=384,
        d_model=128,
        embedding_dim=128,
        num_object_queries=64,
        heads=8,
        encoder_layers=2,
        decoder_layers=2,
        feedforward_dim=512,
        dropout=0.1,
        auxiliary_loss=True,
        geometry_association=True,
    ):
        super().__init__()
        Transformer = _load_transformer(trackformer_root)
        # Kept in the public signature for checkpoint/config compatibility.
        # In this PointPillars adaptation, every detector proposal is an
        # object query; arbitrary learned DETR queries would discard the
        # detector's proposal identity and geometry.
        self.num_object_queries = num_object_queries
        self.auxiliary_loss = auxiliary_loss
        self.geometry_association = geometry_association
        self.token_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
        )
        self.cell_position = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.proposal_position = nn.Sequential(
            nn.Linear(9, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.object_query_content = nn.Sequential(
            nn.Linear(9, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.object_query_position = nn.Sequential(
            nn.Linear(8, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.track_box_position = nn.Sequential(
            nn.Linear(8, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.transformer = Transformer(
            d_model=d_model,
            nhead=heads,
            num_encoder_layers=encoder_layers,
            num_decoder_layers=decoder_layers,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            return_intermediate_dec=True,
        )
        self.class_head = nn.Linear(d_model, 2)
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 8),
        )
        self.embedding_head = nn.Linear(d_model, embedding_dim)
        self.association_head = nn.Sequential(
            nn.LayerNorm(d_model + 9),
            nn.Linear(d_model + 9, d_model),
            nn.GELU(),
            nn.Linear(d_model, embedding_dim),
        )
        nn.init.zeros_(self.association_head[-1].weight)
        nn.init.zeros_(self.association_head[-1].bias)

    def _proposal_queries(self, agent, device):
        """Turn each actual PointPillars proposal into one object query."""
        geometry = normalized_ego_geometry(
            agent["boxes"],
            agent["scores"],
            agent["transformation_matrix"],
        ).to(device)
        ego_boxes = transform_boxes_to_ego(
            agent["boxes"],
            agent["transformation_matrix"],
        ).to(device)
        references = normalize_ego_boxes(ego_boxes)
        content = self.object_query_content(geometry)
        positions = self.object_query_position(references)
        return content, positions, references, geometry

    def _memory(self, agent, device):
        tokens = agent["tokens"].to(device)
        positions = agent["positions"].to(device)
        valid = ~agent["mask"].to(device)
        if not valid.any():
            raise ValueError("Agent contains no valid proposal ROI cells")
        proposal_indices = (
            torch.arange(len(tokens), device=device)[:, None]
            .expand_as(valid)[valid]
        )
        geometry = normalized_ego_geometry(
            agent["boxes"],
            agent["scores"],
            agent["transformation_matrix"],
        ).to(device)
        source = self.token_projection(tokens[valid])
        position = self.cell_position(positions[valid])
        position = position + self.proposal_position(
            geometry[proposal_indices]
        )
        source = source.T[None, :, None, :]
        position = position.T[None, :, None, :]
        mask = torch.zeros(
            (1, 1, source.shape[-1]),
            dtype=torch.bool,
            device=device,
        )
        return source, mask, position

    def forward_agent(
        self,
        agent,
        *,
        track_states=None,
        track_reference_boxes=None,
        track_reference_geometry=None,
    ):
        device = next(self.parameters()).device
        source, mask, position = self._memory(agent, device)
        (
            object_content,
            object_query_positions,
            object_references,
            object_geometry,
        ) = (
            self._proposal_queries(agent, device)
        )
        object_content = object_content[:, None, :]
        object_query_positions = object_query_positions[:, None, :]
        track_count = 0
        if track_states is not None:
            track_count = len(track_states)
            if track_reference_boxes is None:
                raise ValueError(
                    "Propagated track states require reference boxes"
                )
            if track_reference_geometry is None:
                raise ValueError(
                    "Propagated track states require ego-frame geometry"
                )
            track_positions = self.track_box_position(
                track_reference_boxes
            )[:, None, :]
            query_positions = torch.cat(
                [track_positions, object_query_positions], dim=0
            )
            query_content = torch.cat(
                [track_states[:, None, :], object_content], dim=0
            )
            query_references = torch.cat(
                [track_reference_boxes, object_references], dim=0
            )
            query_geometry = torch.cat(
                [track_reference_geometry, object_geometry], dim=0
            )
        else:
            query_positions = object_query_positions
            query_content = object_content
            query_references = object_references
            query_geometry = object_geometry
        states, states_without_norm, _ = self.transformer(
            source,
            mask,
            query_positions,
            position,
            query_content,
        )
        logits = self.class_head(states)
        box_deltas = self.box_head(states)
        reference_logits = torch.logit(
            query_references.clamp(1e-4, 1 - 1e-4)
        )
        boxes = (
            box_deltas + reference_logits[None, None]
        ).sigmoid()
        if self.geometry_association:
            expanded_geometry = query_geometry[None, None].expand(
                states.shape[0], states.shape[1], -1, -1
            )
            association_input = torch.cat(
                [states, expanded_geometry.to(states.dtype)], dim=-1
            )
            embeddings = F.normalize(
                self.embedding_head(states)
                + self.association_head(association_input),
                dim=-1,
            )
        else:
            embeddings = F.normalize(
                self.embedding_head(states), dim=-1
            )
        output = {
            "pred_logits": logits[-1],
            "pred_boxes": boxes[-1],
            "embeddings": embeddings[-1],
            "hs_embed": states_without_norm[-1],
            "track_count": track_count,
            "object_query_count": len(object_references),
            "query_geometry": query_geometry,
        }
        if self.auxiliary_loss:
            output["aux_outputs"] = [
                {"pred_logits": layer_logits, "pred_boxes": layer_boxes}
                for layer_logits, layer_boxes in zip(logits[:-1], boxes[:-1])
            ]
        return output

    def forward_pair(
        self,
        source_agent,
        target_agent,
        matcher,
        *,
        false_track_queries=2,
        false_negative_probability=0.0,
        backpropagate_source=True,
    ):
        """Run source detection, propagate its queries, then decode target."""
        device = next(self.parameters()).device
        source_target = spatial_target(source_agent, device)
        target = spatial_target(target_agent, device)
        source_output = self.forward_agent(source_agent)
        source_indices, source_target_indices = matcher(
            source_output, source_target
        )
        if false_negative_probability:
            keep = (
                torch.rand(len(source_indices), device=device)
                >= false_negative_probability
            )
            source_indices = source_indices[keep]
            source_target_indices = source_target_indices[keep]
        track_states = source_output["hs_embed"][0, source_indices]
        track_boxes = source_output["pred_boxes"][0, source_indices]
        track_geometry = source_output["query_geometry"][source_indices]
        track_ids = [
            source_target["object_ids"][index]
            for index in source_target_indices.tolist()
        ]
        selected = set(source_indices.tolist())
        unmatched = torch.tensor(
            [
                index
                for index in range(
                    source_output["object_query_count"]
                )
                if index not in selected
            ],
            dtype=torch.long,
            device=device,
        )
        false_count = min(false_track_queries, len(unmatched))
        if false_count:
            false_scores = source_output["pred_logits"][
                0, unmatched
            ].softmax(-1)[:, 0]
            false_indices = unmatched[
                false_scores.topk(false_count).indices
            ]
            track_states = torch.cat(
                [track_states, source_output["hs_embed"][0, false_indices]]
            )
            track_boxes = torch.cat(
                [track_boxes, source_output["pred_boxes"][0, false_indices]]
            )
            track_geometry = torch.cat(
                [
                    track_geometry,
                    source_output["query_geometry"][false_indices],
                ]
            )
            track_ids.extend([None] * false_count)
        if not backpropagate_source:
            track_states = track_states.detach()
            track_boxes = track_boxes.detach()
            track_geometry = track_geometry.detach()
        target_by_id = {
            object_id: index
            for index, object_id in enumerate(target["object_ids"])
        }
        match_ids = torch.tensor(
            [
                target_by_id.get(object_id, -1)
                if object_id is not None
                else -1
                for object_id in track_ids
            ],
            dtype=torch.long,
            device=device,
        )
        target_output = self.forward_agent(
            target_agent,
            track_states=track_states,
            track_reference_boxes=track_boxes,
            track_reference_geometry=track_geometry,
        )
        query_count = len(target_output["pred_logits"][0])
        track_mask = torch.zeros(
            query_count, dtype=torch.bool, device=device
        )
        track_mask[: len(track_states)] = True
        false_mask = torch.zeros_like(track_mask)
        false_mask[: len(track_states)] = match_ids < 0
        target["track_query_match_ids"] = match_ids
        target["track_queries_mask"] = track_mask
        target["track_queries_false_positive_mask"] = false_mask
        return {
            "source_output": source_output,
            "source_target": source_target,
            "target_output": target_output,
            "target": target,
            "track_ids": track_ids,
        }
