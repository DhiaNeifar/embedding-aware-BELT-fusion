import math

import torch

from embedding_aware_belt_fusion.alignformer_pva.losses import (
    apply_se2,
    bev_corners,
    corner_displacements,
    corner_loss,
    match_nll,
)


def _box(x, y, length, width, yaw):
    return [x, y, 0.0, 1.5, width, length, yaw]


def test_bev_corners_axis_aligned_box():
    boxes = torch.tensor([_box(0.0, 0.0, length=4.0, width=2.0, yaw=0.0)]).unsqueeze(0)
    corners = bev_corners(boxes)[0, 0]
    expected = torch.tensor([[2.0, 1.0], [2.0, -1.0], [-2.0, -1.0], [-2.0, 1.0]])
    assert torch.allclose(corners, expected, atol=1e-5)


def test_bev_corners_rotated_box_matches_manual_rotation():
    yaw = math.pi / 2
    boxes = torch.tensor([_box(1.0, 1.0, length=4.0, width=2.0, yaw=yaw)]).unsqueeze(0)
    corners = bev_corners(boxes)[0, 0]
    # At 90 degrees, length runs along y and width along x.
    expected = torch.tensor([[0.0, 3.0], [2.0, 3.0], [2.0, -1.0], [0.0, -1.0]])
    assert torch.allclose(corners, expected, atol=1e-5)


def test_apply_se2_rotates_and_translates():
    points = torch.tensor([[[1.0, 0.0]]])
    psi = torch.tensor([math.pi / 2])
    t = torch.tensor([[1.0, 1.0]])
    result = apply_se2(points, psi, t)
    assert torch.allclose(result[0, 0], torch.tensor([1.0, 2.0]), atol=1e-5)


def test_corner_loss_zero_when_prediction_matches_truth():
    boxes = torch.tensor([_box(2.0, 3.0, length=4.0, width=2.0, yaw=0.3)]).unsqueeze(0)
    psi = torch.tensor([0.5])
    t = torch.tensor([[1.0, -1.0]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    loss = corner_loss(boxes, psi, t, psi, t, mask)
    assert math.isclose(float(loss), 0.0, abs_tol=1e-6)


def test_corner_loss_grows_with_displacement_and_ignores_masked_objects():
    boxes = torch.tensor(
        [_box(0.0, 0.0, 4.0, 2.0, 0.0), _box(5.0, 5.0, 4.0, 2.0, 0.0)]
    ).unsqueeze(0)
    psi_true = torch.tensor([0.0])
    t_true = torch.tensor([[0.0, 0.0]])
    psi_pred = torch.tensor([0.0])
    t_pred_small = torch.tensor([[0.1, 0.0]])
    t_pred_large = torch.tensor([[2.0, 0.0]])
    mask = torch.ones(1, 2, dtype=torch.bool)

    small_loss = corner_loss(boxes, psi_pred, t_pred_small, psi_true, t_true, mask)
    large_loss = corner_loss(boxes, psi_pred, t_pred_large, psi_true, t_true, mask)
    assert float(large_loss) > float(small_loss) > 0.0

    # Masking out the only object must not divide by zero and must zero the loss.
    empty_mask = torch.zeros(1, 2, dtype=torch.bool)
    zero_loss = corner_loss(boxes, psi_pred, t_pred_large, psi_true, t_true, empty_mask)
    assert math.isclose(float(zero_loss), 0.0, abs_tol=1e-6)


def test_corner_displacements_matches_corner_loss_reduction():
    boxes = torch.tensor(
        [_box(0.0, 0.0, 4.0, 2.0, 0.0), _box(3.0, 1.0, 4.0, 2.0, 0.1)]
    ).unsqueeze(0)
    psi_pred = torch.tensor([0.05])
    t_pred = torch.tensor([[0.2, -0.1]])
    psi_true = torch.tensor([0.0])
    t_true = torch.tensor([[0.0, 0.0]])
    mask = torch.ones(1, 2, dtype=torch.bool)

    per_object = corner_displacements(boxes, psi_pred, t_pred, psi_true, t_true)
    manual_mean = per_object[0].mean()
    loss = corner_loss(boxes, psi_pred, t_pred, psi_true, t_true, mask)
    assert math.isclose(float(manual_mean), float(loss), abs_tol=1e-6)


# ---- match_nll --------------------------------------------------------------


def test_match_nll_is_near_zero_for_a_confident_correct_assignment():
    # A log-assignment that puts almost all mass on the correct cells.
    log_assignment = torch.full((1, 3, 3), math.log(1e-6))
    log_assignment[0, 0, 0] = math.log(0.98)
    log_assignment[0, 1, 1] = math.log(0.98)
    ego_match = torch.tensor([[0, 1]])
    cav_match = torch.tensor([[0, 1]])
    loss = match_nll(log_assignment, ego_match, cav_match)
    assert float(loss) < 0.1


def test_match_nll_penalizes_confident_wrong_assignment():
    log_assignment = torch.full((1, 3, 3), math.log(1e-6))
    log_assignment[0, 0, 1] = math.log(0.98)  # confidently wrong: truth is 0->0
    ego_match = torch.tensor([[0, -1]])
    cav_match = torch.tensor([[0, -1]])
    loss = match_nll(log_assignment, ego_match, cav_match)
    assert float(loss) > 5.0


def test_match_nll_unmatched_objects_are_supervised_onto_dustbin():
    # 1 ego object, 1 cav object, genuinely no match (e.g. different physical
    # objects) -- both should be confidently assigned to their dustbin.
    log_assignment = torch.full((1, 2, 2), math.log(1e-6))
    log_assignment[0, 0, 1] = math.log(0.99)  # ego[0] -> dustbin (col index 1)
    log_assignment[0, 1, 0] = math.log(0.99)  # cav[0] -> dustbin (row index 1)
    ego_match = torch.tensor([[-1]])
    cav_match = torch.tensor([[-1]])
    loss = match_nll(log_assignment, ego_match, cav_match)
    assert float(loss) < 0.1
