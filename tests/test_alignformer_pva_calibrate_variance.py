import math

import numpy as np
import torch

from embedding_aware_belt_fusion.alignformer_pva.calibrate_variance import (
    collect_correspondences,
    collect_pair_correspondences,
    fit_score_additive_mle,
    score_decile_table,
)
from embedding_aware_belt_fusion.alignformer_pva.variance import variance_model_from_config


def _agent(centers, yaws, gt_ids, scores=None, transformation_matrix=None):
    n = len(centers)
    boxes = torch.zeros(n, 7)
    if n:
        boxes[:, 0:2] = torch.tensor(centers, dtype=torch.float32)
        boxes[:, 3] = 1.5
        boxes[:, 4] = 1.5
        boxes[:, 5] = 4.0
        boxes[:, 6] = torch.tensor(yaws, dtype=torch.float32)
    if scores is None:
        scores = torch.linspace(0.9, 0.5, n) if n else torch.zeros(0)
    else:
        scores = torch.tensor(scores, dtype=torch.float32)
    return {
        "boxes": boxes,
        "scores": scores,
        "gt_ids": list(gt_ids),
        "transformation_matrix": (
            transformation_matrix if transformation_matrix is not None else torch.eye(4)
        ),
    }


def _planar_transform(yaw, tx, ty):
    matrix = torch.eye(4)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    matrix[0, 0], matrix[0, 1] = cosine, -sine
    matrix[1, 0], matrix[1, 1] = sine, cosine
    matrix[0, 3], matrix[1, 3] = tx, ty
    return matrix


# ---- fit_score_additive_mle: recovers a KNOWN power law from synthetic data ----


def _synthesize_additive_residuals(sigma1, exponent, score_reference, n, rng):
    """Sample ``n`` correspondences under the EXACT model
    ``v(s) = sigma1^2 (s/s_ref)^-2b``, ``Var = v(ego) + v(cav)``, as a 2-D
    centre error (so the residual is the Euclidean norm of two independent
    per-axis Gaussian components, each carrying half the total variance)."""
    ego_score = rng.uniform(0.2, 1.0, size=n)
    cav_score = rng.uniform(0.2, 1.0, size=n)
    v_ego = sigma1 ** 2 * (ego_score / score_reference) ** (-2 * exponent)
    v_cav = sigma1 ** 2 * (cav_score / score_reference) ** (-2 * exponent)
    total_variance = v_ego + v_cav
    dx = rng.normal(0.0, np.sqrt(total_variance / 2.0))
    dy = rng.normal(0.0, np.sqrt(total_variance / 2.0))
    residual = np.sqrt(dx ** 2 + dy ** 2)
    return residual, ego_score, cav_score


def test_fit_score_additive_mle_recovers_known_sigma_and_exponent():
    rng = np.random.default_rng(42)
    true_sigma1, true_exponent, score_reference = 0.3, 1.15, 0.4
    residual, ego_score, cav_score = _synthesize_additive_residuals(
        true_sigma1, true_exponent, score_reference, n=60000, rng=rng
    )

    fitted_sigma1, fitted_exponent, converged = fit_score_additive_mle(
        residual, ego_score, cav_score, score_reference
    )

    assert converged
    assert math.isclose(fitted_sigma1, true_sigma1, rel_tol=0.15)
    assert math.isclose(fitted_exponent, true_exponent, rel_tol=0.2)


def test_fit_score_additive_mle_recovers_a_different_known_power_law():
    # A second, distinct ground truth (larger sigma, steeper exponent -- the
    # heading-channel regime per variance.py's own docstring) to make sure
    # the first test's recovery wasn't a coincidence of one parameter choice.
    rng = np.random.default_rng(7)
    true_sigma1, true_exponent, score_reference = math.radians(5.0), 1.9, 0.4
    residual, ego_score, cav_score = _synthesize_additive_residuals(
        true_sigma1, true_exponent, score_reference, n=60000, rng=rng
    )

    fitted_sigma1, fitted_exponent, converged = fit_score_additive_mle(
        residual, ego_score, cav_score, score_reference
    )

    assert converged
    assert math.isclose(fitted_sigma1, true_sigma1, rel_tol=0.15)
    assert math.isclose(fitted_exponent, true_exponent, rel_tol=0.2)


def test_fit_score_additive_mle_rejects_mismatched_shapes():
    import pytest

    with pytest.raises(ValueError):
        fit_score_additive_mle(
            np.ones(3), np.ones(3), np.ones(4), score_reference=0.4
        )


# ---- score_decile_table -----------------------------------------------------


def test_score_decile_table_shows_rms_falling_as_confidence_rises():
    rng = np.random.default_rng(0)
    n = 20000
    min_score = np.sort(rng.uniform(0.2, 1.0, size=n))
    # Deterministic residual (no sampling noise) so the decile RMS trend is
    # guaranteed monotonic -- this test checks the table-building/binning
    # logic, not the stochastic fit.
    centre_residual = 0.3 * (min_score / 0.4) ** (-1.1)
    yaw_residual_rad = math.radians(5.0) * (min_score / 0.4) ** (-1.8)

    table = score_decile_table(min_score, centre_residual, yaw_residual_rad, n_bins=10)

    assert len(table) == 10
    assert sum(row["n"] for row in table) == n
    rms_centre = [row["rms_centre_m"] for row in table]
    rms_yaw = [row["rms_yaw_deg"] for row in table]
    assert all(a >= b for a, b in zip(rms_centre, rms_centre[1:]))
    assert all(a >= b for a, b in zip(rms_yaw, rms_yaw[1:]))


# ---- collect_pair_correspondences: reuses dataset.py / procrustes.py --------


def test_collect_pair_correspondences_measures_true_pose_disagreement():
    # Ego sees "o1" at (1.0, 1.0); the CAV, under an identity (TRUE) pose,
    # reports the same physical object 0.3 m away in x. The residual must be
    # exactly that disagreement, not zero (which it would be if gt_ids were
    # ignored) and not something route through injected noise (there is none
    # here -- this function never injects any).
    ego_agent = _agent([[1.0, 1.0]], [0.0], ["o1"])
    cav_agent = _agent([[1.3, 1.0]], [0.0], ["o1"])

    result = collect_pair_correspondences(ego_agent, cav_agent)

    assert result is not None
    assert result["centre_residual_m"].shape == (1,)
    assert math.isclose(float(result["centre_residual_m"][0]), 0.3, abs_tol=1e-5)
    assert math.isclose(float(result["yaw_residual_rad"][0]), 0.0, abs_tol=1e-5)


def test_collect_pair_correspondences_folds_180_degree_heading_ambiguity():
    # Same physical object, same centre; the CAV's detector fixed the box's
    # axis but reports it pointing the opposite way (yaw differs by ~pi).
    # The fold must recognize these as a near-zero disagreement, not ~pi.
    ego_agent = _agent([[0.0, 0.0]], [0.0], ["o1"])
    cav_agent = _agent([[0.0, 0.0]], [math.pi - 0.05], ["o1"])

    result = collect_pair_correspondences(ego_agent, cav_agent)

    assert result is not None
    yaw_residual_deg = math.degrees(float(result["yaw_residual_rad"][0]))
    assert yaw_residual_deg < 5.0  # folded residual, NOT ~175 degrees


def test_collect_pair_correspondences_folds_each_pair_independently():
    # Regression test for a real bug: with only ONE matched object, applying
    # the fold sign elementwise vs. via a full (1, K, K) cross-matrix and
    # taking row 0 happen to agree (K=1 collapses the distinction), so a
    # single-pair test cannot catch a bug that reads the wrong row/diagonal
    # of that cross-matrix. This test uses TWO matched objects with DIFFERENT
    # fold requirements: object "o1" needs a 180-degree fold (CAV reports it
    # pointing the opposite way), object "o2" needs no fold at all (CAV
    # already agrees). A bug that broadcasts object 0's fold sign onto every
    # pair would wrongly flip "o2" too, inflating its residual from ~0 to ~pi.
    ego_agent = _agent([[0.0, 0.0], [5.0, 5.0]], [0.0, 1.0], ["o1", "o2"])
    cav_agent = _agent(
        [[0.0, 0.0], [5.0, 5.0]], [math.pi - 0.05, 1.0], ["o1", "o2"]
    )

    result = collect_pair_correspondences(ego_agent, cav_agent)

    assert result is not None
    assert result["yaw_residual_rad"].shape == (2,)
    yaw_residual_deg = np.degrees(result["yaw_residual_rad"])
    # "o1" needed the fold: small residual, not ~175 degrees.
    assert yaw_residual_deg[0] < 5.0
    # "o2" needed NO fold: must stay small too, not corrupted by "o1"'s fold.
    assert yaw_residual_deg[1] < 5.0


def test_collect_pair_correspondences_returns_none_when_nothing_matches():
    ego_agent = _agent([[1.0, 1.0]], [0.0], ["o1"])
    cav_agent = _agent([[1.3, 1.0]], [0.0], ["o2"])  # disjoint physical ids
    assert collect_pair_correspondences(ego_agent, cav_agent) is None


def test_collect_pair_correspondences_returns_none_for_an_empty_side():
    ego_agent = _agent([], [], [])
    cav_agent = _agent([[1.3, 1.0]], [0.0], ["o1"])
    assert collect_pair_correspondences(ego_agent, cav_agent) is None


def test_collect_pair_correspondences_projects_through_the_true_transform():
    # The CAV's boxes are in ITS OWN local frame; _project_true must carry
    # them into the ego frame via transformation_matrix before the residual
    # is measured. Here the CAV-local box at the origin, under a +1m x
    # translation, lands at ego-frame (1.0, 0.5); the ego's own box is
    # reported directly at (1.0, 0.5) (identity transform) -- so the
    # residual must be ~0 despite the CAV's raw local coordinate being (0, 0).
    ego_agent = _agent([[1.0, 0.5]], [0.0], ["o1"])
    cav_agent = _agent(
        [[0.0, 0.0]], [0.0], ["o1"],
        transformation_matrix=_planar_transform(0.0, 1.0, 0.5),
    )

    result = collect_pair_correspondences(ego_agent, cav_agent)

    assert result is not None
    assert math.isclose(float(result["centre_residual_m"][0]), 0.0, abs_tol=1e-5)


def test_collect_correspondences_aggregates_across_frames_and_agents():
    frame_a = {
        "agents": [
            _agent([[1.0, 1.0], [4.0, -2.0]], [0.0, 0.4], ["o1", "o2"]),
            _agent([[1.3, 1.0], [4.0, -1.5]], [0.0, 0.4], ["o1", "o2"]),
        ]
    }
    frame_b = {
        "agents": [
            _agent([[0.0, 0.0]], [0.0], ["p1"]),
            _agent([[0.2, 0.0]], [0.0], ["p1"]),
        ]
    }
    data = collect_correspondences([frame_a, frame_b])

    assert data["pairs_seen"] == 2
    assert data["pairs_with_matches"] == 2
    assert data["centre_residual_m"].shape == (3,)  # 2 matches in frame_a + 1 in frame_b
    assert data["ego_score"].shape == data["centre_residual_m"].shape
    assert data["cav_score"].shape == data["centre_residual_m"].shape


# ---- integration: a calibrate_variance.py JSON payload round-trips --------


def test_fitted_correspondence_variance_block_round_trips_through_variance_model_from_config():
    # Exactly the shape run_calibration()/main() writes under the
    # "correspondence_variance" key.
    payload = {
        "correspondence_variance": {
            "mode": "split",
            "sigma_translation_m": 0.31,
            "translation_exponent": 1.21,
            "sigma_yaw_rad": math.radians(4.4),
            "sigma_yaw_deg": 4.4,
            "yaw_exponent": 1.87,
            "score_reference": 0.4,
        }
    }

    model = variance_model_from_config(payload)

    assert model.mode == "split"
    assert model.enabled
    scores = torch.tensor([0.05, 0.2, 0.4, 0.6, 0.9, 1.0])
    centre_var, heading_var = model.detection_variances(scores)
    assert torch.isfinite(centre_var).all() and torch.all(centre_var > 0)
    assert torch.isfinite(heading_var).all() and torch.all(heading_var > 0)
    # Lower confidence must predict higher disagreement variance (the whole
    # point of the model), same invariant variance.py's own tests check.
    assert torch.all(centre_var[1:] < centre_var[:-1])
    assert torch.all(heading_var[1:] < heading_var[:-1])
