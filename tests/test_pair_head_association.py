import torch

from embedding_aware_belt_fusion.evaluation.component_diagnostics import (
    association_counts,
)
from embedding_aware_belt_fusion.integration.belt_fusion import (
    _constrained_assignment,
    associate_by_geometry,
    associate_by_symmetric_pair_head,
    fuse_detections,
)


def _pair_head_detection(agent_id, boxes, states):
    boxes = torch.tensor(boxes, dtype=torch.float32)
    count = len(boxes)
    return {
        "agent_id": agent_id,
        "boxes": boxes,
        "scores": torch.full((count,), 0.9),
        "evidence": torch.tensor([[1.0, 4.0]]).repeat(count, 1),
        "class_uncertainty": torch.full((count,), 0.25),
        "covariances": torch.eye(7).repeat(count, 1, 1),
        "query_states": torch.tensor(states, dtype=torch.float32),
        "query_geometry": torch.zeros(count, 9),
    }


def _fake_pair_head(group_states, group_geometry, member_states, member_geometry):
    """Deterministic stand-in for the trained pair head: same-object states
    (small L2 distance) score near +10 (sigmoid ~1); different-object states
    score near -40 (sigmoid ~0). Lets the test control matches exactly,
    independent of any trained model."""
    diff = group_states.unsqueeze(1) - member_states.unsqueeze(0)
    return 10.0 - 50.0 * diff.pow(2).sum(dim=-1)


def test_all_agent_symmetric_pair_head_matches_source_to_source_without_ego():
    # Object A is seen by ego and source1. Object C is occluded from ego but
    # seen by both source1 and source2 -- the all-agent grouping mode must
    # associate C across the two sources directly, with no ego mediation.
    ego = _pair_head_detection(
        "ego", boxes=[[0.0, 0.0, 0, 1, 1, 1, 0.0]], states=[[1.0, 0.0]]
    )
    source1 = _pair_head_detection(
        "source1",
        boxes=[[0.1, 0.0, 0, 1, 1, 1, 0.0], [20.0, 0.0, 0, 1, 1, 1, 0.0]],
        states=[[1.0, 0.0], [0.0, 1.0]],
    )
    source2 = _pair_head_detection(
        "source2", boxes=[[20.1, 0.0, 0, 1, 1, 1, 0.0]], states=[[0.0, 1.0]]
    )

    groups = associate_by_symmetric_pair_head(
        [ego, source1, source2],
        _fake_pair_head,
        maximum_distance=8.0,
        minimum_probability=0.9,
    )

    agent_sets = [frozenset(member["agent_id"] for member in group) for group in groups]
    assert frozenset({"ego", "source1"}) in agent_sets
    assert frozenset({"source1", "source2"}) in agent_sets
    for group in groups:
        ids = {member["agent_id"] for member in group}
        if "ego" in ids:
            assert "source2" not in ids


def test_constrained_assignment_does_not_let_an_invalid_pair_block_a_valid_one():
    # The bug fixed in 0497cf4: an unconstrained Hungarian solve minimizes
    # total cost over the *full* matrix, including invalid entries, before
    # filtering. Here the cross pairs (row0-col1, row1-col0) are cheaper but
    # invalid; the old post-hoc-filter approach would pick them, then drop
    # both, losing the two valid diagonal pairs entirely.
    cost = torch.tensor([[5.0, 1.0], [1.0, 5.0]])
    valid = torch.tensor([[True, False], [False, True]])
    pairs = _constrained_assignment(cost, valid, unmatched_cost=10.0)
    assert set(pairs) == {(0, 0), (1, 1)}


def test_associate_by_geometry_recovers_both_valid_pairs_under_adversarial_cost():
    # End-to-end version of the same adversarial layout: two ego-frame
    # objects where the cross assignment is geometrically invalid (too far)
    # but numerically cheaper than the correct diagonal pairing would be if
    # validity weren't enforced before solving.
    ego = _pair_head_detection(
        "ego",
        boxes=[[0.0, 0.0, 0, 1, 1, 1, 0.0], [100.0, 0.0, 0, 1, 1, 1, 0.0]],
        states=[[0.0], [0.0]],
    )
    source = _pair_head_detection(
        "source",
        boxes=[[0.2, 0.0, 0, 1, 1, 1, 0.0], [100.2, 0.0, 0, 1, 1, 1, 0.0]],
        states=[[0.0], [0.0]],
    )
    groups = associate_by_geometry([ego, source], maximum_distance=5.0)
    agent_sets = [frozenset(member["agent_id"] for member in group) for group in groups]
    assert agent_sets.count(frozenset({"ego", "source"})) == 2


def _detection_with_ids(agent_id, boxes, oracle_object_ids):
    boxes = torch.tensor(boxes, dtype=torch.float32)
    count = len(boxes)
    return {
        "agent_id": agent_id,
        "boxes": boxes,
        "scores": torch.full((count,), 0.9),
        "evidence": torch.tensor([[1.0, 4.0]]).repeat(count, 1),
        "class_uncertainty": torch.full((count,), 0.25),
        "covariances": torch.eye(7).repeat(count, 1, 1),
        "oracle_object_ids": oracle_object_ids,
    }


def test_association_counts_catches_a_false_negative_group_counts_would_miss():
    # Agent a has one box (object "obj1"). Agent b has two candidates: the
    # true match "obj1" a bit further away, and a different physical object
    # "obj2" that happens to sit closer. One-to-one Hungarian assignment
    # picks the closer (wrong) one, leaving the true match as its own
    # singleton group -- a false negative that never enters any group, so
    # group_counts() alone (which only looks inside existing groups) cannot
    # see it. association_counts() must still catch it as a missed candidate.
    a = _detection_with_ids(
        "a", boxes=[[0.0, 0.0, 0, 1, 1, 1, 0.0]], oracle_object_ids=["obj1"]
    )
    b = _detection_with_ids(
        "b",
        boxes=[[0.1, 0.0, 0, 1, 1, 1, 0.0], [0.05, 0.0, 0, 1, 1, 1, 0.0]],
        oracle_object_ids=["obj1", "obj2"],
    )
    fused = fuse_detections([a, b], maximum_distance=5.0, association="geometry")
    counts = association_counts(
        [a, b], fused["groups"], maximum_distance=5.0
    )
    assert counts["candidate_pairs"] == 2
    assert counts["positive_pairs"] == 1
    assert counts["true_positive"] == 0
    assert counts["false_positive"] == 1
    assert counts["false_negative"] == 1
