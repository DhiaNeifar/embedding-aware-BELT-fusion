"""alignformer_ext.temporal: odometry carry covariance and the robust correlated GLS window estimate."""

import numpy as np

from embedding_aware_belt_fusion.alignformer_ext import temporal
from embedding_aware_belt_fusion.alignformer_ext.se2 import compose


def test_carry_covariance_matches_monte_carlo():
    rng = np.random.default_rng(0)
    a, x, b = np.array([0.6, 0.05, 0.01]), np.array([35.0, -12.0, 0.4]), np.array([-0.5, 0.02, -0.02])
    cov = np.diag([0.02, 0.03, 1e-5])
    q = np.diag([0.01 ** 2, 0.01 ** 2, np.radians(0.05) ** 2])
    mean, predicted = temporal._carry(a, x, b, cov, q, q)
    samples = []
    for _ in range(40000):
        xs = x + rng.multivariate_normal(np.zeros(3), cov)
        samples.append(compose(compose(a + rng.multivariate_normal(np.zeros(3), q), xs), b + rng.multivariate_normal(np.zeros(3), q)))
    samples = np.array(samples)
    np.testing.assert_allclose(mean, compose(compose(a, x), b))
    empirical = np.cov(samples.T)
    scale = np.sqrt(np.outer(np.diag(predicted), np.diag(predicted)))   # compare on the correlation scale
    np.testing.assert_allclose(empirical / scale, predicted / scale, atol=0.03)


def test_gls_is_the_mean_without_odometry_and_ignores_an_outlier():
    noise = np.diag([0.05 ** 2, 0.05 ** 2, np.radians(0.1) ** 2])
    truth = np.array([20.0, 5.0, 0.3])
    offsets = np.array([[0.03, -0.02, 0.001], [-0.03, 0.02, -0.001], [0.0, 0.0, 0.0]])
    zero = [np.zeros((3, 3))] * 3
    np.testing.assert_allclose(temporal._robust_gls(truth + offsets, zero, noise), truth, atol=1e-6)
    with_outlier = np.vstack([truth + offsets, truth + [3.0, -2.0, 0.2]])
    estimate = temporal._robust_gls(with_outlier, zero + [np.zeros((3, 3))], noise)
    assert np.hypot(*(estimate[:2] - truth[:2])) < 0.05


def test_gls_trusts_the_newest_measurement_when_old_ones_carry_large_odometry_error():
    noise = np.diag([0.05 ** 2, 0.05 ** 2, np.radians(0.1) ** 2])
    newest = np.array([20.0, 5.0, 0.3])
    old = newest + [0.4, 0.4, 0.0]
    big = np.diag([4.0, 4.0, 1e-2])
    estimate = temporal._robust_gls(np.array([old, old, newest]), [big, big, np.zeros((3, 3))], noise)
    np.testing.assert_allclose(estimate, newest, atol=0.01)


def test_window_agreement_flags_an_inconsistent_window():
    noise = np.diag([0.05 ** 2, 0.05 ** 2, np.radians(0.1) ** 2])
    rng = np.random.default_rng(1)
    zero = [np.zeros((3, 3))] * 6
    consistent = np.array([20.0, 5.0, 0.3]) + rng.normal(0, [0.05, 0.05, np.radians(0.1)], (6, 3))
    scattered = np.array([20.0, 5.0, 0.3]) + rng.normal(0, [0.5, 0.5, np.radians(1.0)], (6, 3))
    assert temporal._robust_gls_agreement(consistent, zero, noise)[1] > 0.5
    assert temporal._robust_gls_agreement(scattered, zero, noise)[1] <= 0.5


def test_odometry_drift_scales_with_distance_and_vanishes_when_parked():
    settings = temporal.TemporalSettings(odometry_sd=(0.0, 0.0), odometry_rel=temporal.LOAM_KITTI)
    _, parked = temporal._odometry(np.zeros(3), settings, 0, 0, "rsu")
    _, moving = temporal._odometry(np.array([10.0, 0.0, 0.0]), settings, 0, 0, "cav")
    assert parked[0, 0] < 1e-10 and np.isclose(np.sqrt(moving[0, 0]), 0.055)
    assert np.isclose(np.degrees(np.sqrt(moving[2, 2])), 0.013)
