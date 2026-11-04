import math

import torch

from embedding_aware_belt_fusion.alignformer_pva.head_match import log_sinkhorn
from embedding_aware_belt_fusion.alignformer_pva.procrustes import (
    augment_with_heading,
    heading_orientation,
    weighted_se2_kabsch,
)


def _rotation(yaw: float) -> torch.Tensor:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return torch.tensor([[cosine, -sine], [sine, cosine]])


# ---- log_sinkhorn ----------------------------------------------------------


def test_sinkhorn_real_rows_and_columns_sum_to_one():
    # Only the REAL rows/columns individually sum to 1. The dustbin row/
    # column is a shared bucket that can absorb leftover mass from several
    # real rows/columns at once, so it sums to the *count* of the other
    # side's real entries (columns for the dustbin row, rows for the dustbin
    # column), not to 1 -- verified separately below.
    torch.manual_seed(0)
    rows, columns = 4, 5
    scores = torch.randn(2, rows, columns)
    alpha = torch.tensor(-1.0)
    assignment = log_sinkhorn(scores, alpha, iterations=50).exp()

    real_row_sums = assignment[:, :rows, :].sum(dim=2)
    real_column_sums = assignment[:, :, :columns].sum(dim=1)
    assert torch.allclose(real_row_sums, torch.ones(2, rows), atol=1e-3)
    assert torch.allclose(real_column_sums, torch.ones(2, columns), atol=1e-3)

    dustbin_row_sum = assignment[:, rows, :].sum(dim=1)
    dustbin_column_sum = assignment[:, :, columns].sum(dim=1)
    assert torch.allclose(
        dustbin_row_sum, torch.full((2,), float(columns)), atol=1e-3
    )
    assert torch.allclose(
        dustbin_column_sum, torch.full((2,), float(rows)), atol=1e-3
    )


def test_sinkhorn_favors_the_diagonal_on_a_peaked_score_matrix():
    # A strongly diagonal score matrix should produce a near-identity soft
    # assignment on its real (non-dustbin) block.
    scores = torch.full((1, 3, 3), -5.0)
    scores[0] += torch.eye(3) * 10.0
    assignment = log_sinkhorn(scores, torch.tensor(-10.0), iterations=50).exp()
    real_block = assignment[0, :3, :3]
    assert torch.allclose(real_block, torch.eye(3), atol=0.05)


def test_sinkhorn_degenerate_empty_side_returns_finite_zero_gradient():
    scores = torch.zeros(1, 0, 3, requires_grad=True)
    alpha = torch.tensor(-1.0, requires_grad=True)
    assignment = log_sinkhorn(scores, alpha, iterations=20)
    assert torch.isfinite(assignment).all()
    assert assignment.shape == (1, 1, 4)
    assignment.sum().backward()
    assert alpha.grad is not None
    assert torch.isfinite(alpha.grad).all()


# ---- weighted_se2_kabsch ----------------------------------------------------


def test_kabsch_recovers_exact_rotation_and_translation():
    torch.manual_seed(1)
    true_psi = 0.6
    true_translation = torch.tensor([1.5, -2.0])
    source = torch.randn(1, 6, 2) * 3
    target = source @ _rotation(true_psi).T + true_translation
    weights = torch.ones(1, 6)

    psi, t = weighted_se2_kabsch(target, source, weights)
    assert math.isclose(float(psi[0]), true_psi, abs_tol=1e-4)
    assert torch.allclose(t[0], true_translation, atol=1e-4)


def test_kabsch_negated_sign_would_fail_this_check():
    # Regression guard for the exact sign bug caught during implementation:
    # the wrong-sign formula recovers -true_psi, not true_psi.
    true_psi = 0.6
    source = torch.randn(1, 5, 2) * 3
    target = source @ _rotation(true_psi).T + torch.tensor([1.0, 1.0])
    psi, _ = weighted_se2_kabsch(target, source, torch.ones(1, 5))
    assert not math.isclose(float(psi[0]), -true_psi, abs_tol=1e-3)
    assert math.isclose(float(psi[0]), true_psi, abs_tol=1e-3)


def test_kabsch_downweights_outlier_points():
    true_psi = 0.1
    translation = torch.tensor([0.5, 0.5])
    inlier_source = torch.randn(1, 8, 2) * 3
    inlier_target = inlier_source @ _rotation(true_psi).T + translation
    # One wild outlier correspondence that does not follow the true transform.
    outlier_source = torch.tensor([[[50.0, 50.0]]])
    outlier_target = torch.tensor([[[-50.0, -50.0]]])

    source = torch.cat([inlier_source, outlier_source], dim=1)
    target = torch.cat([inlier_target, outlier_target], dim=1)
    weights = torch.cat([torch.ones(1, 8), torch.zeros(1, 1)], dim=1)

    psi, t = weighted_se2_kabsch(target, source, weights)
    assert math.isclose(float(psi[0]), true_psi, abs_tol=1e-3)
    assert torch.allclose(t[0], translation, atol=1e-3)


def test_kabsch_single_point_returns_identity_not_a_guessed_translation():
    # A single point pins translation ONLY under an assumed zero rotation --
    # but with one correspondence, rotation is genuinely undetermined (any
    # psi admits some t that fits this one point exactly). The implementation
    # correctly refuses to guess: both psi and t come back as identity
    # (finite, not NaN) rather than silently assuming zero rotation.
    source = torch.tensor([[[1.0, 2.0]]])
    target = torch.tensor([[[4.0, 6.0]]])
    psi, t = weighted_se2_kabsch(target, source, torch.ones(1, 1))
    assert torch.isfinite(psi).all()
    assert torch.isfinite(t).all()
    assert math.isclose(float(psi[0]), 0.0, abs_tol=1e-6)
    assert torch.allclose(t[0], torch.zeros(2), atol=1e-6)


# ---- augment_with_heading / heading_orientation ----------------------------


def test_augment_with_heading_appends_offset_tip_point():
    centres = torch.tensor([[[0.0, 0.0], [2.0, 0.0]]])
    yaws = torch.tensor([[0.0, math.pi / 2]])
    augmented = augment_with_heading(centres, yaws, lam=1.0)
    assert augmented.shape == (1, 4, 2)
    assert torch.allclose(augmented[0, :2], centres[0], atol=1e-6)
    assert torch.allclose(augmented[0, 2], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert torch.allclose(augmented[0, 3], torch.tensor([2.0, 1.0]), atol=1e-6)


def test_heading_orientation_flips_opposite_headings_only():
    reference = torch.tensor([[0.0]])  # pointing along +x
    candidates = torch.tensor([[0.0, math.pi, math.pi / 2 - 0.01, math.pi / 2 + 0.01]])
    sign = heading_orientation(reference, candidates)[0, 0]
    assert float(sign[0]) == 1.0  # same direction
    assert float(sign[1]) == -1.0  # exactly opposite
    assert float(sign[2]) == 1.0  # just within 90 degrees
    assert float(sign[3]) == -1.0  # just past 90 degrees


def test_heading_fold_prevents_cancellation_when_averaging():
    # Two candidates for the same object, one correctly oriented, one flipped
    # 180 degrees -- averaging raw headings would partially cancel; folding
    # first should make them reinforce instead.
    reference = torch.tensor([[0.0]])
    candidates = torch.tensor([[0.05, math.pi - 0.05]])
    sign = heading_orientation(reference, candidates)
    direction = torch.stack(
        [torch.cos(candidates), torch.sin(candidates)], dim=-1
    )
    folded_mean = (sign.squeeze(1).unsqueeze(-1) * direction).mean(dim=1)
    raw_mean = direction.mean(dim=1)
    assert torch.linalg.vector_norm(folded_mean[0]) > torch.linalg.vector_norm(
        raw_mean[0]
    )
