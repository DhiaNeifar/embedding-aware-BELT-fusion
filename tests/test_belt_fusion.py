import torch

from embedding_aware_belt_fusion.integration.belt_fusion import (
    _decoded_local_variance,
    associate_by_embedding,
    associate_by_geometry,
    associate_ego_with_propagated_sources_simple,
    evidence_to_mass,
    fuse_detections,
    fuse_groups_score_weighted,
    fuse_two_masses,
)


def _detection(boxes, variance=1.0):
    count = len(boxes)
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "scores": torch.full((count,), 0.8),
        "evidence": torch.tensor([[1.0, 4.0]]).repeat(count, 1),
        "class_uncertainty": torch.full((count,), 0.25),
        "covariances": torch.eye(7).repeat(count, 1, 1) * variance,
    }


def test_delta_variance_is_scaled_into_decoded_box_coordinates():
    log_variance = torch.zeros(1, 7)
    anchors = torch.tensor([[0.0, 0.0, 0.0, 2.0, 3.0, 4.0, 0.0]])
    boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 3.0, 4.0, 0.0]])
    variance = _decoded_local_variance(log_variance, anchors, boxes)
    assert torch.allclose(
        variance,
        torch.tensor([[25.0, 25.0, 4.0, 4.0, 9.0, 16.0, 1.0]]),
    )


def test_geometry_association_keeps_distinct_objects_separate():
    ego = _detection(
        [[0.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0], [20.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]]
    )
    source = _detection(
        [[0.2, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0], [20.1, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]]
    )
    groups = associate_by_geometry([ego, source], maximum_distance=1.0)
    assert sorted(len(group) for group in groups) == [2, 2]


def test_precision_fusion_favors_the_lower_variance_box():
    first = _detection([[0.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]], variance=1.0)
    second = _detection([[10.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]], variance=9.0)
    fused = fuse_detections([first, second], maximum_distance=20.0)
    assert fused["boxes"].shape == (1, 7)
    assert torch.allclose(fused["boxes"][0, 0], torch.tensor(1.0))
    assert torch.allclose(fused["covariances"][0, 0, 0], torch.tensor(0.9))
    assert int(fused["member_counts"][0]) == 2


def test_dempster_shafer_fusion_reduces_uncertainty_for_agreement():
    belief, uncertainty = evidence_to_mass(torch.tensor([[0.0, 4.0]]))
    _, fused_uncertainty = fuse_two_masses(
        belief, uncertainty, belief, uncertainty
    )
    assert float(fused_uncertainty) < float(uncertainty)


def test_embedding_association_selects_identity_over_nearer_wrong_box():
    ego = _detection(
        [[0.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0], [5.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]]
    )
    source = _detection(
        [[0.1, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0], [5.1, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]]
    )
    ego["embeddings"] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    source["embeddings"] = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    groups = associate_by_embedding(
        [ego, source], maximum_distance=10.0, minimum_similarity=0.9
    )
    paired_centers = sorted(
        sorted(round(float(member["box"][0]), 1) for member in group)
        for group in groups
    )
    assert paired_centers == [[0.0, 5.1], [0.1, 5.0]]


def test_simple_propagated_association_and_score_weighted_fusion():
    ego = {
        "boxes": torch.tensor(
            [[0.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]], dtype=torch.float32
        ),
        "scores": torch.tensor([0.8]),
    }
    source = {
        "boxes": torch.tensor(
            [[2.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]], dtype=torch.float32
        ),
        "scores": torch.tensor([0.2]),
        "ego_embeddings": torch.tensor([[1.0, 0.0]]),
        "embeddings": torch.tensor([[1.0, 0.0]]),
    }
    groups = associate_ego_with_propagated_sources_simple(
        ego, [source], maximum_distance=5.0, minimum_similarity=0.9
    )
    fused = fuse_groups_score_weighted(groups)
    assert int(fused["member_counts"][0]) == 2
    # Score weights 0.8 and 0.2 give x=(0*0.8+2*0.2)/(0.8+0.2)=0.4.
    assert torch.allclose(fused["boxes"][0, 0], torch.tensor(0.4))
    assert torch.allclose(fused["scores"], torch.tensor([0.8]))
