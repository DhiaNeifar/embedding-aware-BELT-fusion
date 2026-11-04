"""Numeric tests for :mod:`embedding_aware_belt_fusion.alignformer_pva.abstain`.

Two properties are checked against independently-constructed ground truth
rather than against the module's own internals wherever possible:

1. The Wald statistic discriminates strong, consistent correspondence
   evidence (should NOT abstain) from weak, sparse, high-variance evidence
   (should abstain), in the documented direction.
2. The shipped sandwich covariance ``A^-1 B A^-1`` is the one that matches
   reality -- verified by Monte Carlo simulation of the actual estimator
   (:func:`weighted_se2_kabsch`) under a known, correlated noise model -- and
   the documented-wrong ``sigma^2 A^-1`` does not. This is the exact bug the
   module's docstring says an earlier draft had.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from embedding_aware_belt_fusion.alignformer_pva import abstain
from embedding_aware_belt_fusion.alignformer_pva.procrustes import (
    augment_with_heading,
    weighted_se2_kabsch,
)


def _rotation(yaw: float) -> torch.Tensor:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return torch.tensor([[cosine, -sine], [sine, cosine]])


def _make_case(M, true_psi, true_t, var_centre_val, var_heading_val, lam, noise=0.0):
    """A heading-augmented SE(2) correspondence set with known noise model."""
    centres = torch.randn(1, M, 2) * 3
    yaws = torch.rand(1, M) * 2 * math.pi - math.pi
    q = augment_with_heading(centres, yaws, lam)
    rotation = _rotation(true_psi)
    p = q @ rotation.T + true_t
    if noise > 0:
        p = p + torch.randn_like(p) * noise
    weights = torch.ones(1, 2 * M)
    variance_centre = torch.full((1, M), var_centre_val)
    variance_heading = torch.full((1, M), var_heading_val)
    psi = torch.tensor([true_psi])
    t = true_t.unsqueeze(0)
    return p, q, weights, psi, t, variance_centre, variance_heading


# ---- the core discrimination property -----------------------------------------


def test_wald_statistic_discriminates_strong_from_weak_evidence():
    torch.manual_seed(0)
    # Strong, consistent evidence: many matched objects, tight (low) variance,
    # and a perfect (zero-residual) fit -- the module should have very high
    # confidence that the emitted correction is real.
    p, q, w, psi, t, vc, vh = _make_case(
        M=8, true_psi=0.25, true_t=torch.tensor([1.0, -0.5]),
        var_centre_val=0.05, var_heading_val=0.08, lam=1.0, noise=0.0,
    )
    strong = abstain.wald_parts(
        p, q, w, psi, t, variance_centre=vc, variance_heading=vh, heading_lambda=1.0
    )

    # Weak, sparse, inconsistent evidence: a single matched object, large
    # variance, and a small, noisy correction -- there is essentially nothing
    # to support correcting at all.
    p2, q2, w2, psi2, t2, vc2, vh2 = _make_case(
        M=1, true_psi=0.01, true_t=torch.tensor([0.02, -0.01]),
        var_centre_val=4.0, var_heading_val=4.0, lam=1.0, noise=0.5,
    )
    weak = abstain.wald_parts(
        p2, q2, w2, psi2, t2, variance_centre=vc2, variance_heading=vh2,
        heading_lambda=1.0,
    )

    # The statistic itself must discriminate, and by a wide margin.
    assert float(strong.statistic[0]) > 1000.0 * float(weak.statistic[0]) + 1.0

    config = abstain.AbstentionConfig(mode="abstain", level=0.05)
    threshold_strong = abstain.abstention_threshold(strong.dof, config)
    threshold_weak = abstain.abstention_threshold(weak.dof, config)

    # The strong case clears its own (dof-dependent) threshold: do not abstain.
    assert bool((strong.statistic > threshold_strong)[0])
    # The weak case does not clear its threshold: abstain.
    assert not bool((weak.statistic > threshold_weak)[0])

    factor_strong = abstain.decision_factor(strong.statistic, strong.dof, config)
    factor_weak = abstain.decision_factor(weak.statistic, weak.dof, config)
    assert math.isclose(float(factor_strong[0]), 1.0, abs_tol=1e-6)
    assert math.isclose(float(factor_weak[0]), 0.0, abs_tol=1e-6)


def test_identity_correction_scores_exactly_zero():
    # Documented invariant: an identity correction (psi=0, t=0) scores exactly
    # zero and therefore abstains under every rule, regardless of geometry.
    p, q, w, _, _, vc, vh = _make_case(
        M=5, true_psi=0.0, true_t=torch.tensor([0.0, 0.0]),
        var_centre_val=0.2, var_heading_val=0.3, lam=1.0, noise=0.0,
    )
    zero_psi = torch.zeros(1)
    zero_t = torch.zeros(1, 2)
    statistic, dof = abstain.wald_statistic(
        p, q, w, zero_psi, zero_t, variance_centre=vc, variance_heading=vh,
        heading_lambda=1.0,
    )
    assert math.isclose(float(statistic[0]), 0.0, abs_tol=1e-9)


# ---- the documented bug: sigma^2 A^-1 vs the sandwich A^-1 B A^-1 -------------


def test_sandwich_covariance_matches_monte_carlo_ground_truth_but_naive_does_not():
    """The sandwich prediction must match simulated reality; the naive one must not.

    This is a from-scratch numerical re-derivation, independent of the
    module's internal wiring: simulate many noisy correspondence sets under a
    KNOWN, heavily-correlated noise model (centre and heading residuals
    sharing almost all of their randomness, as the module docstring
    describes), re-solve with the real, already-verified
    :func:`weighted_se2_kabsch` each time, and take the empirical sample
    covariance of the resulting (psi, t) as ground truth.
    """
    torch.manual_seed(42)
    M = 2  # the sparse regime the docstring says is worst-affected
    lam = 1.0
    true_psi = 0.2
    true_t = torch.tensor([0.8, -0.3])

    centres = torch.randn(1, M, 2) * 3
    yaws = torch.rand(1, M) * 2 * math.pi - math.pi
    q = augment_with_heading(centres, yaws, lam)
    directions = torch.stack([torch.cos(yaws), torch.sin(yaws)], dim=-1)
    perp_d = torch.stack([-directions[..., 1], directions[..., 0]], dim=-1)

    rotation = _rotation(true_psi)
    true_p = q @ rotation.T + true_t

    # Near-maximal correlation between an object's centre and heading noise
    # (tiny angular variance g relative to the centre variance a): this is the
    # regime the docstring says the naive "2M independent points" assumption
    # is most wrong in.
    a_val = 1.0
    g_val = 0.01
    variance_centre = torch.full((1, M), a_val)
    variance_heading = torch.full((1, M), a_val + lam ** 2 * g_val)

    per_axis_centre_sd = math.sqrt(a_val / 2.0)
    heading_extra_sd = math.sqrt(g_val)

    n_trials = 20000
    weights = torch.ones(1, 2 * M)
    theta_samples = torch.zeros(n_trials, 3)
    for trial in range(n_trials):
        eps_c = torch.randn(1, M, 2) * per_axis_centre_sd
        eps_h_extra = (torch.randn(1, M, 1) * heading_extra_sd) * perp_d
        eps_h = eps_c + lam * eps_h_extra
        noisy_p = true_p.clone()
        noisy_p[:, :M] += eps_c
        noisy_p[:, M:] += eps_h
        psi_hat, t_hat = weighted_se2_kabsch(noisy_p, q, weights)
        theta_samples[trial, 0] = t_hat[0, 0]
        theta_samples[trial, 1] = t_hat[0, 1]
        theta_samples[trial, 2] = psi_hat[0]

    centred = theta_samples - theta_samples.mean(dim=0)
    empirical_cov = (centred.T @ centred) / (n_trials - 1)

    # Model predictions at the true transform.
    psi_t = torch.tensor([true_psi])
    t_t = true_t.unsqueeze(0)
    cos, sin = torch.cos(psi_t).unsqueeze(-1), torch.sin(psi_t).unsqueeze(-1)
    x, y = q[..., 0], q[..., 1]
    rotated = torch.stack([cos * x - sin * y, sin * x + cos * y], dim=-1)
    normalized = abstain._normalized(weights)
    normal_matrix = abstain._normal_matrix(rotated, normalized)
    middle = abstain.sandwich_middle(
        rotated, normalized, variance_centre, variance_heading, lam
    )
    inverse = torch.linalg.inv(normal_matrix)
    cov_sandwich = (inverse @ middle @ inverse)[0]

    sigma_sq_per_axis = a_val / 2.0  # genuine isotropic per-point variance
    cov_naive = (sigma_sq_per_axis * inverse)[0]

    def relative_error(predicted: Tensor, truth: Tensor) -> float:
        return float((predicted - truth).abs().sum() / truth.abs().sum())

    sandwich_error = relative_error(cov_sandwich, empirical_cov)
    naive_error = relative_error(cov_naive, empirical_cov)

    # The sandwich formula should track the simulated ground truth closely...
    assert sandwich_error < 0.1
    # ...while the naive sigma^2 A^-1 formula should be measurably wrong, and
    # much worse than the sandwich formula on the SAME ground truth.
    assert naive_error > 0.5
    assert naive_error > 5.0 * sandwich_error

    # And in the documented direction: naive UNDERSTATES the covariance (so it
    # OVERSTATES confidence / the Wald statistic) for the true correction.
    theta_vec = torch.cat([true_t, torch.tensor([true_psi])])
    naive_precision = torch.linalg.inv(cov_naive)
    sandwich_precision = torch.linalg.inv(cov_sandwich)
    naive_statistic = float(theta_vec @ naive_precision @ theta_vec)
    sandwich_statistic = float(theta_vec @ sandwich_precision @ theta_vec)
    empirical_precision = torch.linalg.inv(empirical_cov)
    empirical_statistic = float(theta_vec @ empirical_precision @ theta_vec)

    assert naive_statistic > sandwich_statistic
    # The sandwich statistic should be close to what the empirical (simulated)
    # covariance itself implies; the naive one should not.
    assert abs(sandwich_statistic - empirical_statistic) < abs(
        naive_statistic - empirical_statistic
    )


def test_shipped_precision_is_proportional_to_the_sandwich_formula_not_to_A_alone():
    """Directly pins the formula :func:`wald_parts` actually computes.

    ``wald_parts`` is documented to return ``precision = A B^-1 A / tau^2``.
    This recomputes ``A`` and ``B`` independently via the module's own public
    (``sandwich_middle``) and private (``_normal_matrix``) building blocks and
    checks that the shipped ``precision`` is an exact scalar multiple of
    ``A B^-1 A`` -- and explicitly NOT a scalar multiple of ``A`` alone, which
    is the shape ``sigma^2 A^-1`` (the documented bug) would imply.
    """
    torch.manual_seed(7)
    M = 2
    lam = 1.0
    true_psi = 0.2
    true_t = torch.tensor([0.8, -0.3])
    p, q, w, psi, t, vc, vh = _make_case(
        M=M, true_psi=true_psi, true_t=true_t,
        var_centre_val=1.0, var_heading_val=1.0 + lam ** 2 * 0.01, lam=lam, noise=0.0,
    )
    parts = abstain.wald_parts(
        p, q, w, psi, t, variance_centre=vc, variance_heading=vh, heading_lambda=lam
    )

    cos, sin = torch.cos(psi).unsqueeze(-1), torch.sin(psi).unsqueeze(-1)
    x, y = q[..., 0], q[..., 1]
    rotated = torch.stack([cos * x - sin * y, sin * x + cos * y], dim=-1)
    normalized = abstain._normalized(w)
    normal_matrix = abstain._normal_matrix(rotated, normalized)
    middle = abstain.sandwich_middle(rotated, normalized, vc, vh, lam)
    sandwich_shape = torch.matmul(normal_matrix, torch.linalg.solve(middle, normal_matrix))

    # Recover the implied tau^2 from one well-conditioned entry (the psi-psi
    # one, which is informative here), then check the WHOLE matrix agrees.
    tau_scale = float(sandwich_shape[0, 2, 2] / parts.precision[0, 2, 2])
    rescaled = parts.precision[0] * tau_scale
    assert torch.allclose(rescaled, sandwich_shape[0], rtol=1e-3, atol=1e-3)

    # The same rescaling must NOT reproduce plain A (the naive sigma^2 A^-1
    # shape): if it did, the shipped formula would be the documented bug.
    assert not torch.allclose(rescaled, normal_matrix[0], rtol=1e-2, atol=1e-2)


# ---- documented equivalences and small building blocks ------------------------


def test_both_at_level_one_equals_per_pair():
    # "``both`` at level 1.0 -- a threshold of zero -- IS ``per_pair``."
    statistic = torch.tensor([0.1, 5.0, 100.0])
    dof = torch.tensor([2.0, 5.0, 50.0])
    per_pair_factor = abstain.decision_factor(
        statistic, None, abstain.AbstentionConfig(mode="per_pair")
    )
    both_factor = abstain.decision_factor(
        statistic, dof, abstain.AbstentionConfig(mode="both", level=1.0)
    )
    assert torch.allclose(per_pair_factor, both_factor, atol=1e-5)


def test_effective_sample_size_matches_kish_formula_by_hand():
    weights = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0], [1.0, 3.0, 0.0, 0.0]]
    )
    ess = abstain.effective_sample_size(weights)
    expected = torch.tensor([4.0, 1.0, 2.0, 1.6])
    assert torch.allclose(ess, expected, atol=1e-6)


def test_abstention_config_rejects_level_with_scalar_free_modes():
    import pytest

    with pytest.raises(ValueError):
        abstain.AbstentionConfig(mode="per_pair", level=0.1)
    with pytest.raises(ValueError):
        abstain.AbstentionConfig(mode="abstain", level=0.0)
    with pytest.raises(ValueError):
        abstain.AbstentionConfig(mode="not_a_mode")


def test_abstention_config_parse_round_trips():
    parsed = abstain.AbstentionConfig.parse("abstain:0.05")
    assert parsed.mode == "abstain"
    assert math.isclose(parsed.level, 0.05)

    parsed_both = abstain.AbstentionConfig.parse("both:0.01")
    assert parsed_both.mode == "both"
    assert math.isclose(parsed_both.level, 0.01)

    parsed_per_pair = abstain.AbstentionConfig.parse("per_pair")
    assert parsed_per_pair.mode == "per_pair"
    assert parsed_per_pair.level == 0.0


# ---- the public decide() API ---------------------------------------------------


@dataclass(frozen=True)
class _FakeEstimate:
    psi: Tensor
    t: Tensor
    offset_statistic: Optional[Tensor] = None
    offset_dof: Optional[Tensor] = None
    offset_precision: Optional[Tensor] = None


def test_decide_abstain_zeros_weak_and_passes_through_strong():
    config = abstain.AbstentionConfig(mode="abstain", level=0.05)
    # threshold at dof=10, level=0.05 is ~11.1 (computed from the F reference);
    # 0.5 is clearly below it and 1e6 is clearly above it.
    weak = _FakeEstimate(
        psi=torch.tensor([0.05]),
        t=torch.tensor([[0.1, -0.1]]),
        offset_statistic=torch.tensor([0.5]),
        offset_dof=torch.tensor([10.0]),
    )
    strong = _FakeEstimate(
        psi=torch.tensor([0.05]),
        t=torch.tensor([[0.1, -0.1]]),
        offset_statistic=torch.tensor([1.0e6]),
        offset_dof=torch.tensor([10.0]),
    )
    decided_weak = abstain.decide(weak, config)
    decided_strong = abstain.decide(strong, config)

    assert torch.allclose(decided_weak.psi, torch.zeros(1))
    assert torch.allclose(decided_weak.t, torch.zeros(1, 2))
    assert torch.allclose(decided_strong.psi, strong.psi)
    assert torch.allclose(decided_strong.t, strong.t)


def test_decide_per_pair_applies_the_exact_positive_part_factor():
    config = abstain.AbstentionConfig(mode="per_pair")
    statistic = torch.tensor([12.0])  # max(0, 1 - 3/12) = 0.75, by hand
    estimate = _FakeEstimate(
        psi=torch.tensor([0.2]), t=torch.tensor([[1.0, -1.0]]),
        offset_statistic=statistic,
    )
    decided = abstain.decide(estimate, config)
    assert math.isclose(float(decided.psi[0]), 0.2 * 0.75, abs_tol=1e-6)
    assert torch.allclose(decided.t[0], torch.tensor([0.75, -0.75]), atol=1e-6)


def test_decide_disabled_config_returns_the_same_object():
    config = abstain.AbstentionConfig(mode="none")
    estimate = _FakeEstimate(psi=torch.tensor([1.0]), t=torch.tensor([[2.0, 3.0]]))
    assert abstain.decide(estimate, config) is estimate
