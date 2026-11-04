import math

import torch

from embedding_aware_belt_fusion.alignformer_pva.procrustes import weighted_se2_kabsch
from embedding_aware_belt_fusion.alignformer_pva.robust import (
    DISABLED,
    GEMAN_MCCLURE,
    HUBER,
    RobustSolveConfig,
    residual_scale,
    robust_factor,
    robust_se2_kabsch,
    weighted_median,
)


def _rotation(yaw: float) -> torch.Tensor:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return torch.tensor([[cosine, -sine], [sine, cosine]])


def _outlier_case():
    """14 exact inliers under a known SE(2) transform, plus one bad correspondence.

    The outlier is displaced by a fixed offset from where the true transform
    would put it -- large enough to stand out once a roughly-correct fit is in
    hand, but not so extreme (relative to the inlier spread and count) that it
    hijacks the very first least-squares solve by leverage alone. A wild,
    far-away outlier instead masks itself: with equal weights it can dominate
    the LS normal equations and make the *naive* solve nearly interpolate it,
    which is a real masking effect but not what this test is checking.
    """
    torch.manual_seed(3)
    true_psi = 0.3
    translation = torch.tensor([1.0, -0.5])
    n_inliers = 14
    inlier_source = torch.randn(1, n_inliers, 2) * 3
    inlier_target = inlier_source @ _rotation(true_psi).T + translation

    outlier_source = torch.tensor([[[2.0, 2.0]]])
    outlier_target = (
        outlier_source @ _rotation(true_psi).T
        + translation
        + torch.tensor([5.0, -4.0])
    )

    source = torch.cat([inlier_source, outlier_source], dim=1)
    target = torch.cat([inlier_target, outlier_target], dim=1)
    weights = torch.ones(1, n_inliers + 1)
    return target, source, weights, true_psi, translation


# ---- wraps, does not replace -------------------------------------------------


def test_disabled_config_reproduces_weighted_se2_kabsch_bit_for_bit():
    target, source, weights, _, _ = _outlier_case()
    evidence = torch.tensor([15.0])
    baseline_psi, baseline_t = weighted_se2_kabsch(target, source, weights)
    solution = robust_se2_kabsch(
        target, source, weights, evidence=evidence, config=DISABLED
    )
    assert torch.equal(solution.psi, baseline_psi)
    assert torch.equal(solution.t, baseline_t)
    assert torch.equal(solution.weights, weights)
    assert torch.equal(solution.engaged, torch.zeros_like(solution.engaged))


def test_zero_iterations_disables_regardless_of_mode():
    target, source, weights, _, _ = _outlier_case()
    config = RobustSolveConfig(mode=HUBER, iterations=0)
    assert not config.enabled
    evidence = torch.tensor([15.0])
    baseline_psi, baseline_t = weighted_se2_kabsch(target, source, weights)
    solution = robust_se2_kabsch(
        target, source, weights, evidence=evidence, config=config
    )
    assert torch.equal(solution.psi, baseline_psi)
    assert torch.equal(solution.t, baseline_t)


# ---- the core robustness property --------------------------------------------


def test_huber_irls_recovers_truth_more_accurately_than_naive_uniform_weights():
    target, source, weights, true_psi, translation = _outlier_case()
    evidence = torch.tensor([15.0])
    config = RobustSolveConfig(mode=HUBER, iterations=3, min_evidence=3.0)

    naive_psi, naive_t = weighted_se2_kabsch(target, source, weights)
    solution = robust_se2_kabsch(
        target, source, weights, evidence=evidence, config=config
    )

    naive_psi_error = abs(float(naive_psi[0]) - true_psi)
    robust_psi_error = abs(float(solution.psi[0]) - true_psi)
    naive_t_error = float(torch.linalg.vector_norm(naive_t[0] - translation))
    robust_t_error = float(torch.linalg.vector_norm(solution.t[0] - translation))

    # The outlier biases the naive (uniform-weight) solve measurably; IRLS
    # must recover the true transform strictly more accurately.
    assert robust_psi_error < naive_psi_error
    assert robust_t_error < naive_t_error
    assert robust_psi_error < 0.01
    assert robust_t_error < 0.1

    # The outlier's final weight must have been pushed well below an inlier's.
    assert float(solution.weights[0, -1]) < 0.1 * float(solution.weights[0, 0])
    assert bool(solution.engaged[0])


def test_geman_mcclure_suppresses_outlier_at_least_as_hard_as_huber():
    target, source, weights, _, _ = _outlier_case()
    evidence = torch.tensor([15.0])
    huber_cfg = RobustSolveConfig(mode=HUBER, iterations=3)
    gm_cfg = RobustSolveConfig(mode=GEMAN_MCCLURE, iterations=3)
    huber_solution = robust_se2_kabsch(
        target, source, weights, evidence=evidence, config=huber_cfg
    )
    gm_solution = robust_se2_kabsch(
        target, source, weights, evidence=evidence, config=gm_cfg
    )
    # Geman-McClure is redescending (outlier pull -> 0) while Huber is only
    # bounded-influence (a constant, small pull remains), so GM's outlier
    # weight should be no larger than Huber's.
    assert float(gm_solution.weights[0, -1]) <= float(huber_solution.weights[0, -1]) + 1e-6


# ---- the evidence guard -------------------------------------------------------


def test_guard_skips_the_loop_below_min_evidence():
    target, source, weights, _, _ = _outlier_case()
    config = RobustSolveConfig(mode=HUBER, iterations=3, min_evidence=50.0)
    evidence = torch.tensor([15.0])  # below the 50 threshold
    baseline_psi, baseline_t = weighted_se2_kabsch(target, source, weights)
    solution = robust_se2_kabsch(
        target, source, weights, evidence=evidence, config=config
    )
    assert torch.equal(solution.psi, baseline_psi)
    assert torch.equal(solution.t, baseline_t)
    assert torch.equal(solution.weights, weights)
    assert not bool(solution.engaged[0])


# ---- weighted_median -----------------------------------------------------------


def test_weighted_median_matches_known_value_under_equal_weights():
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    weights = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    # Lower weighted median of 4 equally-weighted points: the smallest value
    # whose cumulative weight reaches half the total (2.0) is the second one.
    median = weighted_median(values, weights)
    assert math.isclose(float(median[0]), 2.0)


def test_weighted_median_moves_toward_the_heavily_weighted_point():
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    weights_skewed = torch.tensor([[0.01, 0.01, 0.01, 10.0]])
    median_skewed = weighted_median(values, weights_skewed)
    assert math.isclose(float(median_skewed[0]), 4.0)


# ---- the gradient path is unchanged (load-bearing per the module docstring) ---


def test_residual_scale_and_robust_factor_are_detached_from_autograd():
    residual = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    weights = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True)
    config = RobustSolveConfig(mode=HUBER)
    scale = residual_scale(residual, weights, config)
    factor = robust_factor(residual, scale, config)
    assert not scale.requires_grad
    assert not factor.requires_grad
