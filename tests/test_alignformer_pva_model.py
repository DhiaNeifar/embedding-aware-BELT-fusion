import math

import torch

from embedding_aware_belt_fusion.alignformer_pva.model import (
    AlignFormerA,
    AlignFormerB,
    Correspondence,
    PoseEstimate,
    _is_empty,
    _zero_estimate,
    reduce_correspondence,
    solve_pose,
)
from embedding_aware_belt_fusion.alignformer_pva.variance import UNWEIGHTED


def _rotation(yaw: float) -> torch.Tensor:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return torch.tensor([[cosine, -sine], [sine, cosine]])


def _batch(ego_centres, ego_yaws, cav_centres, cav_yaws, embed_dim=0):
    m, n = ego_centres.shape[0], cav_centres.shape[0]

    def make_boxes(centres, yaws):
        boxes = torch.zeros(1, centres.shape[0], 7)
        boxes[0, :, :2] = centres
        boxes[0, :, 3:6] = torch.tensor([1.5, 1.5, 4.0])
        boxes[0, :, 6] = yaws
        return boxes

    return {
        "ego_boxes": make_boxes(ego_centres, ego_yaws),
        "cav_boxes": make_boxes(cav_centres, cav_yaws),
        "ego_scores": torch.full((1, m), 0.9),
        "cav_scores": torch.full((1, n), 0.9),
        "ego_mask": torch.ones(1, m, dtype=torch.bool),
        "cav_mask": torch.ones(1, n, dtype=torch.bool),
        "ego_embeddings": torch.zeros(1, m, embed_dim),
        "cav_embeddings": torch.zeros(1, n, embed_dim),
    }


# ---- reduce_correspondence + solve_pose (isolates the pose math from the
# untrained trunk, mirroring the source repo's own oracle-testing philosophy:
# a perfect one-hot assignment should let solve_pose recover the exact
# transform, regardless of how that assignment was produced). ------------


def test_reduce_and_solve_recover_exact_transform_from_oracle_assignment():
    true_psi = 0.25
    translation = torch.tensor([1.2, -0.7])
    rotation = _rotation(true_psi)

    ego_centres = torch.randn(5, 2) * 4
    ego_yaws = torch.rand(5) * 2 * math.pi - math.pi
    # cav sees the same objects, rotated into its own (noisy) frame: moving
    # ego's boxes by R/t should land on the cav's reported boxes, i.e.
    # cav = (ego - t) @ R (matching weighted_se2_kabsch's own convention,
    # already pinned by tests/test_alignformer_pva_math.py).
    cav_centres = (ego_centres - translation) @ rotation
    cav_yaws = ego_yaws - true_psi

    batch = _batch(ego_centres, ego_yaws, cav_centres, cav_yaws)
    # One-hot oracle assignment: ego object i <-> cav object i.
    weights = torch.eye(5).unsqueeze(0)

    correspondence = reduce_correspondence(weights, batch, UNWEIGHTED)
    estimate = solve_pose(correspondence, batch, heading_lambda=2.0, variance_model=UNWEIGHTED)

    assert math.isclose(float(estimate.psi[0]), true_psi, abs_tol=1e-3)
    assert torch.allclose(estimate.t[0], translation, atol=1e-3)
    assert float(estimate.confidence[0]) == 5.0


def test_reduce_correspondence_drops_unmatched_ego_rows():
    ego_centres = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    ego_yaws = torch.zeros(2)
    cav_centres = torch.tensor([[0.0, 0.0]])
    cav_yaws = torch.zeros(1)
    batch = _batch(ego_centres, ego_yaws, cav_centres, cav_yaws)
    # Only ego row 0 has a match; row 1 (unmatched) must get zero mass.
    weights = torch.tensor([[[1.0], [0.0]]])
    correspondence = reduce_correspondence(weights, batch, UNWEIGHTED)
    assert float(correspondence.mass[0, 0]) == 1.0
    assert float(correspondence.mass[0, 1]) == 0.0


# ---- empty-set guards -------------------------------------------------------


def test_is_empty_detects_either_side():
    empty_cav = _batch(torch.zeros(2, 2), torch.zeros(2), torch.zeros(0, 2), torch.zeros(0))
    assert _is_empty(empty_cav)
    nonempty = _batch(torch.zeros(2, 2), torch.zeros(2), torch.zeros(3, 2), torch.zeros(3))
    assert not _is_empty(nonempty)


def test_zero_estimate_has_zero_confidence_and_identity_pose():
    batch = _batch(torch.zeros(2, 2), torch.zeros(2), torch.zeros(0, 2), torch.zeros(0))
    estimate = _zero_estimate(batch, statistic=True)
    assert torch.equal(estimate.psi, torch.zeros(1))
    assert torch.equal(estimate.t, torch.zeros(1, 2))
    assert torch.equal(estimate.confidence, torch.zeros(1))
    assert torch.equal(estimate.offset_statistic, torch.zeros(1))


# ---- end-to-end module forward (untrained weights: shape/guard correctness,
# not pose accuracy -- accuracy is solve_pose's job, tested above against a
# clean oracle assignment, isolating the head from an untrained trunk). ----


def test_alignformer_b_forward_runs_end_to_end_and_respects_empty_guard():
    torch.manual_seed(0)
    model = AlignFormerB(embed_dim=0, layers=1, heads=2, model_dim=16)
    batch = _batch(torch.randn(3, 2) * 3, torch.rand(3) * 2 * math.pi - math.pi,
                    torch.randn(4, 2) * 3, torch.rand(4) * 2 * math.pi - math.pi)
    estimate = model(batch)
    assert estimate.psi.shape == (1,)
    assert estimate.t.shape == (1, 2)
    assert torch.isfinite(estimate.psi).all()
    assert torch.isfinite(estimate.t).all()

    empty_batch = _batch(torch.randn(3, 2), torch.zeros(3), torch.zeros(0, 2), torch.zeros(0))
    zero_estimate = model(empty_batch)
    assert float(zero_estimate.confidence[0]) == 0.0
    assert float(zero_estimate.psi[0]) == 0.0


def test_alignformer_b_statistic_flag_populates_wald_fields():
    torch.manual_seed(0)
    model = AlignFormerB(embed_dim=0, layers=1, heads=2, model_dim=16)
    batch = _batch(torch.randn(4, 2) * 3, torch.rand(4) * 2 * math.pi - math.pi,
                    torch.randn(4, 2) * 3, torch.rand(4) * 2 * math.pi - math.pi)
    estimate = model(batch, statistic=True)
    assert estimate.offset_statistic is not None
    assert estimate.offset_dof is not None
    assert torch.isfinite(estimate.offset_statistic).all()


def test_alignformer_a_forward_runs_end_to_end_and_respects_empty_guard():
    torch.manual_seed(0)
    model = AlignFormerA(embed_dim=0, layers=1, heads=2, model_dim=16)
    batch = _batch(torch.randn(3, 2) * 3, torch.zeros(3), torch.randn(4, 2) * 3, torch.zeros(4))
    estimate = model(batch)
    assert estimate.psi.shape == (1,)
    assert torch.isfinite(estimate.psi).all()
    assert torch.isfinite(estimate.t).all()

    empty_batch = _batch(torch.randn(3, 2), torch.zeros(3), torch.zeros(0, 2), torch.zeros(0))
    zero_estimate = model(empty_batch)
    assert float(zero_estimate.confidence[0]) == 0.0
