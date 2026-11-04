import math

import pytest
import torch

from embedding_aware_belt_fusion.alignformer_pva.variance import (
    SCORE_FLOOR,
    UNWEIGHTED,
    VARIANCE_MODES,
    CorrespondenceVarianceModel,
    variance_model_from_config,
)


def _fitted_model(mode: str = "split") -> CorrespondenceVarianceModel:
    # Placeholder-but-structurally-realistic constants (same shape as the
    # labmate's fitted values -- exponents > 0, sigma > 0 -- just not the
    # exact numbers, which are flagged in variance.py as needing
    # recalibration against our own detector).
    return CorrespondenceVarianceModel(
        mode=mode,
        sigma_translation_m=0.3,
        translation_exponent=1.13,
        sigma_yaw_rad=math.radians(5.0),
        yaw_exponent=1.93,
        score_reference=0.4,
    )


# ---- CorrespondenceVarianceModel construction / validation -----------------


def test_rejects_unknown_mode():
    with pytest.raises(ValueError):
        CorrespondenceVarianceModel(
            mode="bogus",
            sigma_translation_m=1.0,
            translation_exponent=0.0,
            sigma_yaw_rad=1.0,
            yaw_exponent=0.0,
        )


@pytest.mark.parametrize("field", ["sigma_translation_m", "sigma_yaw_rad", "score_reference"])
def test_rejects_non_positive_sigmas_and_reference(field):
    kwargs = dict(
        mode="split",
        sigma_translation_m=1.0,
        translation_exponent=0.5,
        sigma_yaw_rad=1.0,
        yaw_exponent=0.5,
        score_reference=0.4,
    )
    kwargs[field] = 0.0
    with pytest.raises(ValueError):
        CorrespondenceVarianceModel(**kwargs)


@pytest.mark.parametrize("field", ["translation_exponent", "yaw_exponent"])
def test_rejects_negative_exponents(field):
    kwargs = dict(
        mode="split",
        sigma_translation_m=1.0,
        translation_exponent=0.5,
        sigma_yaw_rad=1.0,
        yaw_exponent=0.5,
        score_reference=0.4,
    )
    kwargs[field] = -0.1
    with pytest.raises(ValueError):
        CorrespondenceVarianceModel(**kwargs)


def test_unweighted_constant_is_disabled_and_identity_on_weights():
    assert UNWEIGHTED.mode == "none"
    assert not UNWEIGHTED.enabled
    mass = torch.tensor([[0.2, 0.5, 0.9]])
    weights = UNWEIGHTED.augmented_weights(
        mass,
        ego_scores=torch.tensor([[0.3, 0.6, 0.9]]),
        cav_variance_centre=torch.ones(1, 3),
        cav_variance_heading=torch.ones(1, 3),
        heading_lambda=1.0,
    )
    # mode="none" must return cat([mass, mass]) bit for bit (per docstring).
    assert torch.equal(weights, torch.cat([mass, mass], dim=1))


# ---- detection_variances: direction, positivity, finiteness, shape --------


def test_variance_decreases_as_confidence_increases():
    # Lower detection confidence -> higher predicted per-detection variance
    # (the power law has a NEGATIVE exponent on normalized score: see
    # `normalized.pow(-translation_exponent)` in detection_variances).
    model = _fitted_model()
    scores = torch.tensor([0.05, 0.2, 0.4, 0.6, 0.9, 1.0])
    centre_var, heading_var = model.detection_variances(scores)

    # Strictly decreasing as score increases.
    assert torch.all(centre_var[1:] < centre_var[:-1])
    assert torch.all(heading_var[1:] < heading_var[:-1])


def test_variance_equals_sigma_squared_at_score_reference():
    # By construction (score_reference docstring): a detection AT the
    # reference confidence has centre variance exactly sigma_translation_m^2
    # and heading variance exactly sigma_yaw_rad^2 (normalized == 1, so the
    # power term is 1).
    model = _fitted_model()
    scores = torch.tensor([model.score_reference])
    centre_var, heading_var = model.detection_variances(scores)
    assert math.isclose(float(centre_var[0]), model.sigma_translation_m ** 2, rel_tol=1e-6)
    assert math.isclose(float(heading_var[0]), model.sigma_yaw_rad ** 2, rel_tol=1e-6)


def test_detection_variances_finite_and_positive_at_score_zero_and_one():
    # score=0 is clamped to SCORE_FLOOR internally (padded/absent detections
    # in a collated batch carry score 0, and a raw power of 0 with a negative
    # exponent would be infinite).
    model = _fitted_model()
    scores = torch.tensor([0.0, SCORE_FLOOR, 1.0])
    centre_var, heading_var = model.detection_variances(scores)
    assert torch.isfinite(centre_var).all()
    assert torch.isfinite(heading_var).all()
    assert torch.all(centre_var > 0)
    assert torch.all(heading_var > 0)
    # score=0 and score=SCORE_FLOOR must agree exactly: both clamp to the floor.
    assert torch.equal(centre_var[0], centre_var[1])
    assert torch.equal(heading_var[0], heading_var[1])


def test_detection_variances_output_shape_matches_input_shape():
    model = _fitted_model()
    for shape in [(3,), (2, 4), (2, 3, 5)]:
        scores = torch.rand(shape).clamp_min(SCORE_FLOOR)
        centre_var, heading_var = model.detection_variances(scores)
        assert centre_var.shape == shape
        assert heading_var.shape == shape


def test_detection_variances_finite_over_reasonable_score_range():
    model = _fitted_model()
    scores = torch.linspace(0.0, 1.0, steps=201)
    centre_var, heading_var = model.detection_variances(scores)
    assert torch.isfinite(centre_var).all()
    assert torch.isfinite(heading_var).all()
    assert torch.all(centre_var > 0)
    assert torch.all(heading_var > 0)


# ---- correspondence_variances / precisions ---------------------------------


def test_correspondence_variance_heading_includes_centre_term():
    # Var_heading = Var_centre + lam^2 * (v_psi_ego + v_psi_cav), so for any
    # positive lambda, Var_heading > Var_centre (strictly, since heading
    # variances are positive).
    model = _fitted_model()
    ego_scores = torch.tensor([[0.3, 0.7]])
    cav_centre = torch.tensor([[0.1, 0.2]])
    cav_heading = torch.tensor([[0.05, 0.05]])
    var_centre, var_heading = model.correspondence_variances(
        ego_scores, cav_centre, cav_heading, heading_lambda=1.0
    )
    assert torch.all(var_heading > var_centre)

    # lambda=0 collapses heading variance onto centre variance exactly.
    var_centre0, var_heading0 = model.correspondence_variances(
        ego_scores, cav_centre, cav_heading, heading_lambda=0.0
    )
    assert torch.allclose(var_heading0, var_centre0)


def test_precisions_are_inverse_of_correspondence_variances_up_to_reference():
    model = _fitted_model()
    ego_scores = torch.tensor([[0.3, 0.7, 0.9]])
    cav_centre = torch.tensor([[0.1, 0.2, 0.3]])
    cav_heading = torch.tensor([[0.05, 0.05, 0.05]])
    lam = 0.5

    rho_centre, rho_heading = model.precisions(ego_scores, cav_centre, cav_heading, lam)
    var_centre, var_heading = model.correspondence_variances(
        ego_scores, cav_centre, cav_heading, lam
    )
    reference = 2.0 * model.sigma_translation_m ** 2

    assert torch.allclose(rho_centre, reference / var_centre)
    assert torch.allclose(rho_heading, reference / var_heading)
    # Precisions must be positive and finite (inverse-variance weights).
    assert torch.isfinite(rho_centre).all() and torch.all(rho_centre > 0)
    assert torch.isfinite(rho_heading).all() and torch.all(rho_heading > 0)


def test_precision_decreases_as_ego_score_decreases():
    # Lower confidence -> larger variance -> smaller precision (weight).
    model = _fitted_model()
    cav_centre = torch.tensor([[0.1, 0.1]])
    cav_heading = torch.tensor([[0.05, 0.05]])
    ego_scores = torch.tensor([[0.2, 0.9]])
    rho_centre, rho_heading = model.precisions(ego_scores, cav_centre, cav_heading, 1.0)
    assert rho_centre[0, 0] < rho_centre[0, 1]
    assert rho_heading[0, 0] < rho_heading[0, 1]


# ---- augmented_weights -------------------------------------------------


def test_augmented_weights_shape_and_finiteness():
    model = _fitted_model("split")
    batch, m = 2, 4
    mass = torch.rand(batch, m)
    ego_scores = torch.rand(batch, m).clamp_min(SCORE_FLOOR)
    cav_centre = torch.rand(batch, m) + 0.05
    cav_heading = torch.rand(batch, m) * 0.1 + 0.01
    weights = model.augmented_weights(mass, ego_scores, cav_centre, cav_heading, heading_lambda=1.0)
    assert weights.shape == (batch, 2 * m)
    assert torch.isfinite(weights).all()
    assert torch.all(weights >= 0)


def test_scalar_mode_uses_same_weight_for_centre_and_heading_halves_up_to_rescale():
    # mode="scalar" sets rho_heading = rho_centre before the halves are
    # concatenated and jointly rescaled, so the two halves of the weight
    # vector must be in the same mass-weighted proportion.
    model = _fitted_model("scalar")
    mass = torch.tensor([[1.0, 1.0]])
    ego_scores = torch.tensor([[0.3, 0.8]])
    cav_centre = torch.tensor([[0.1, 0.1]])
    cav_heading = torch.tensor([[0.05, 0.05]])
    weights = model.augmented_weights(mass, ego_scores, cav_centre, cav_heading, heading_lambda=1.0)
    centre_half, heading_half = weights[:, :2], weights[:, 2:]
    assert torch.allclose(centre_half, heading_half)


def test_augmented_weights_preserves_total_mass():
    # The docstring/comment promise: the reweighted vector is renormalized so
    # its total equals 2 * sum(mass), exactly what mode="none" would also
    # produce as a total (cat([mass, mass]).sum() == 2 * mass.sum()).
    model = _fitted_model("split")
    mass = torch.tensor([[0.3, 0.6, 0.9]])
    ego_scores = torch.tensor([[0.2, 0.5, 0.95]])
    cav_centre = torch.tensor([[0.2, 0.1, 0.05]])
    cav_heading = torch.tensor([[0.1, 0.05, 0.02]])
    weights = model.augmented_weights(mass, ego_scores, cav_centre, cav_heading, heading_lambda=1.0)
    assert math.isclose(float(weights.sum()), float(2.0 * mass.sum()), rel_tol=1e-5)


def test_augmented_weights_handles_all_zero_mass_row_without_nan():
    # total == 0 for a row must not divide-by-zero; `scale` falls back to 1.
    model = _fitted_model("split")
    mass = torch.zeros(1, 3)
    ego_scores = torch.tensor([[0.2, 0.5, 0.9]])
    cav_centre = torch.tensor([[0.2, 0.1, 0.05]])
    cav_heading = torch.tensor([[0.1, 0.05, 0.02]])
    weights = model.augmented_weights(mass, ego_scores, cav_centre, cav_heading, heading_lambda=1.0)
    assert torch.isfinite(weights).all()
    assert torch.all(weights == 0)


# ---- to_dict / variance_model_from_config ----------------------------------


def test_to_dict_round_trips_sigma_yaw_deg():
    model = _fitted_model()
    as_dict = model.to_dict()
    assert math.isclose(as_dict["sigma_yaw_deg"], math.degrees(model.sigma_yaw_rad), rel_tol=1e-9)
    assert as_dict["mode"] == model.mode


def test_variance_model_from_config_defaults_to_unweighted_when_block_absent():
    model = variance_model_from_config({})
    assert model.mode == "none"
    assert model == UNWEIGHTED


def test_variance_model_from_config_errors_on_weighted_mode_without_block():
    with pytest.raises(ValueError):
        variance_model_from_config({}, mode="scalar")


def test_variance_model_from_config_builds_from_sigma_yaw_deg():
    config = {
        "correspondence_variance": {
            "mode": "split",
            "sigma_translation_m": 0.3,
            "translation_exponent": 1.13,
            "sigma_yaw_deg": 5.0,
            "yaw_exponent": 1.93,
        }
    }
    model = variance_model_from_config(config)
    assert model.mode == "split"
    assert math.isclose(model.sigma_yaw_rad, math.radians(5.0), rel_tol=1e-9)
    assert math.isclose(model.score_reference, 0.4)  # default


def test_variance_model_from_config_prefers_sigma_yaw_rad_when_both_present():
    config = {
        "correspondence_variance": {
            "mode": "split",
            "sigma_translation_m": 0.3,
            "translation_exponent": 1.13,
            "sigma_yaw_rad": 0.1,
            "sigma_yaw_deg": 999.0,  # must be ignored
            "yaw_exponent": 1.93,
        }
    }
    model = variance_model_from_config(config)
    assert math.isclose(model.sigma_yaw_rad, 0.1)


def test_variance_model_from_config_mode_override_takes_precedence():
    config = {
        "correspondence_variance": {
            "mode": "split",
            "sigma_translation_m": 0.3,
            "translation_exponent": 1.13,
            "sigma_yaw_deg": 5.0,
            "yaw_exponent": 1.93,
        }
    }
    model = variance_model_from_config(config, mode="scalar")
    assert model.mode == "scalar"


def test_variance_modes_tuple_contents():
    assert VARIANCE_MODES == ("none", "scalar", "split")
