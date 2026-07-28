import torch

from embedding_aware_belt_fusion.embeddings.spatial_trackformer import (
    normalize_ego_boxes,
)
from embedding_aware_belt_fusion.embeddings.spatial_trackformer_loss import (
    SpatialHungarianMatcher,
    SpatialTrackFormerCriterion,
    spatial_embedding_loss,
    spatial_hard_negative_loss,
)
from embedding_aware_belt_fusion.embeddings.train_spatial_trackformer import (
    _configure_training_stage,
    _ego_and_sources,
    _has_valid_proposals,
)


def test_normalize_ego_boxes_encodes_periodic_yaw():
    boxes = torch.tensor(
        [
            [0.0, 0.0, -1.0, 2.0, 2.0, 4.0, 0.0],
            [0.0, 0.0, -1.0, 2.0, 2.0, 4.0, 3.14159265],
        ]
    )
    normalized = normalize_ego_boxes(boxes)
    assert normalized.shape == (2, 8)
    assert torch.allclose(normalized[0, 6:], torch.tensor([0.5, 1.0]))
    assert torch.allclose(
        normalized[1, 6:], torch.tensor([0.5, 0.0]), atol=1e-6
    )


def test_matcher_forces_valid_tracks_and_rejects_false_tracks():
    output = {
        "pred_logits": torch.tensor(
            [
                [
                    [0.0, 5.0],
                    [5.0, 0.0],
                    [5.0, 0.0],
                    [5.0, 0.0],
                ]
            ]
        ),
        "pred_boxes": torch.tensor(
            [
                [
                    [0.8] * 8,
                    [0.2] * 8,
                    [0.2] * 8,
                    [0.8] * 8,
                ]
            ]
        ),
    }
    target = {
        "labels": torch.tensor([0, 0]),
        "boxes": torch.tensor([[0.2] * 8, [0.8] * 8]),
        "track_query_match_ids": torch.tensor([0, -1]),
        "track_queries_false_positive_mask": torch.tensor(
            [False, True, False, False]
        ),
    }
    prediction, truth = SpatialHungarianMatcher()(output, target)
    matches = dict(zip(prediction.tolist(), truth.tolist()))
    assert matches[0] == 0
    assert 1 not in matches
    assert 1 in matches.values()


def test_criterion_supervises_auxiliary_layers_and_false_tracks():
    logits = torch.tensor(
        [[[5.0, 0.0], [0.0, 5.0], [0.0, 5.0]]],
        requires_grad=True,
    )
    boxes = torch.tensor(
        [[[0.2] * 8, [0.8] * 8, [0.5] * 8]],
        requires_grad=True,
    )
    output = {
        "pred_logits": logits,
        "pred_boxes": boxes,
        "aux_outputs": [
            {"pred_logits": logits.clone(), "pred_boxes": boxes.clone()}
        ],
    }
    target = {
        "labels": torch.tensor([0]),
        "boxes": torch.tensor([[0.2] * 8]),
        "track_query_match_ids": torch.tensor([0, -1]),
        "track_queries_false_positive_mask": torch.tensor(
            [False, True, False]
        ),
    }
    losses = SpatialTrackFormerCriterion(
        SpatialHungarianMatcher()
    )(output, target)
    assert {"loss_ce_0", "loss_bbox_0", "loss_giou_0"} <= set(losses)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert logits.grad is not None
    assert boxes.grad is not None


def test_embedding_head_receives_cross_agent_identity_supervision():
    source_embeddings = torch.nn.functional.normalize(
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True),
        dim=-1,
    )
    target_embeddings = torch.nn.functional.normalize(
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True),
        dim=-1,
    )
    output_source = {"embeddings": source_embeddings}
    output_target = {"embeddings": target_embeddings, "track_count": 0}
    agent = {
        "gt_ids": ["a", "b"],
        "gt_ious": torch.tensor([0.9, 0.8]),
    }
    loss, accuracy, count = spatial_embedding_loss(
        output_source, agent, output_target, agent
    )
    assert count == 2
    assert float(accuracy) == 1.0
    loss.backward()
    assert source_embeddings.grad_fn is not None


def _cached_agent(agent_id, *, valid, ego):
    return {
        "agent_id": agent_id,
        "boxes": torch.zeros(1, 7),
        "tokens": torch.zeros(1, 2, 384),
        "mask": torch.tensor([[False, True]]) if valid else torch.ones(1, 2, dtype=torch.bool),
        "transformation_matrix": (
            torch.eye(4)
            if ego
            else torch.tensor(
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        ),
    }


def test_empty_agent_is_skipped_without_discarding_valid_pair():
    ego = _cached_agent("ego", valid=True, ego=True)
    empty = _cached_agent("empty", valid=False, ego=False)
    source = _cached_agent("source", valid=True, ego=False)
    selected_ego, sources = _ego_and_sources(
        {"agents": [empty, source, ego]}
    )
    assert selected_ego["agent_id"] == "ego"
    assert [agent["agent_id"] for agent in sources] == ["source"]
    assert not _has_valid_proposals(empty)


def test_empty_ego_skips_frame():
    ego = _cached_agent("ego", valid=False, ego=True)
    source = _cached_agent("source", valid=True, ego=False)
    selected_ego, sources = _ego_and_sources({"agents": [source, ego]})
    assert selected_ego is None
    assert sources == []


def test_hard_negative_loss_emphasizes_nearby_similar_vehicle():
    source_values = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True
    )
    target_values = torch.tensor(
        [[[0.6, 0.8], [1.0, 0.0]]], requires_grad=True
    )
    output_source = {"embeddings": source_values}
    output_target = {"embeddings": target_values, "track_count": 0}
    agent = {
        "gt_ids": ["a", "b"],
        "gt_ious": torch.tensor([0.9, 0.9]),
        "boxes": torch.tensor(
            [
                [0.0, 0.0, -1.0, 2.0, 2.0, 4.0, 0.0],
                [2.0, 0.0, -1.0, 2.0, 2.0, 4.0, 0.0],
            ]
        ),
        "transformation_matrix": torch.eye(4),
    }
    loss, anchors = spatial_hard_negative_loss(
        output_source, agent, output_target, agent
    )
    assert anchors == 4
    assert float(loss) > 0
    loss.backward()
    assert source_values.grad is not None
    assert target_values.grad is not None


def test_head_only_stage_freezes_every_other_parameter():
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(2, 2)
    model.association_head = torch.nn.Linear(2, 2)
    _configure_training_stage(model, "association-head")
    assert not any(
        parameter.requires_grad for parameter in model.backbone.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.association_head.parameters()
    )
