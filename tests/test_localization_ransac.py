import math

import torch

from embedding_aware_belt_fusion.integration.localization import (
    apply_se2_correction,
    correspondences_by_source,
    estimate_se2_ransac,
)

_POINTS = torch.tensor(
    [[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [5.0, 5.0], [2.5, 8.0], [8.0, 2.5]]
)


def _rotation(yaw):
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return torch.tensor([[cosine, -sine], [sine, cosine]])


def test_identity_at_zero_noise():
    result = estimate_se2_ransac(_POINTS, _POINTS, inlier_threshold=0.2)
    assert result is not None
    assert torch.allclose(result["rotation"], torch.eye(2), atol=1e-4)
    assert torch.allclose(result["translation"], torch.zeros(2), atol=1e-4)
    assert bool(result["inliers"].all())
    assert float(result["mean_corrected_residual"]) < 1e-4


def test_pure_translation_recovery():
    translation = torch.tensor([2.0, -3.0])
    ego = _POINTS + translation
    result = estimate_se2_ransac(_POINTS, ego, inlier_threshold=0.2)
    assert result is not None
    assert torch.allclose(result["rotation"], torch.eye(2), atol=1e-3)
    assert torch.allclose(result["translation"], translation, atol=1e-3)


def test_pure_yaw_recovery():
    yaw = 0.3
    ego = _POINTS @ _rotation(yaw).T
    result = estimate_se2_ransac(_POINTS, ego, inlier_threshold=0.2)
    assert result is not None
    assert math.isclose(float(result["yaw_correction"]), yaw, abs_tol=1e-3)
    assert torch.allclose(result["translation"], torch.zeros(2), atol=1e-3)


def test_combined_translation_and_yaw_recovery():
    yaw = -0.5
    translation = torch.tensor([1.5, 4.0])
    ego = _POINTS @ _rotation(yaw).T + translation
    result = estimate_se2_ransac(_POINTS, ego, inlier_threshold=0.2)
    assert result is not None
    assert math.isclose(float(result["yaw_correction"]), yaw, abs_tol=1e-3)
    assert torch.allclose(result["translation"], translation, atol=1e-3)


def test_inlier_outlier_rejection():
    yaw = 0.15
    translation = torch.tensor([1.0, -1.0])
    rotation = _rotation(yaw)
    inlier_source = _POINTS
    inlier_ego = inlier_source @ rotation.T + translation
    # Two outlier correspondences that do not follow the true transform at all.
    outlier_source = torch.tensor([[20.0, 0.0], [0.0, 20.0]])
    outlier_ego = torch.tensor([[-5.0, -5.0], [30.0, 30.0]])

    source = torch.cat([inlier_source, outlier_source])
    ego = torch.cat([inlier_ego, outlier_ego])

    result = estimate_se2_ransac(source, ego, inlier_threshold=0.2)
    assert result is not None
    inliers = result["inliers"]
    assert bool(inliers[: len(inlier_source)].all())
    assert not bool(inliers[len(inlier_source) :].any())
    assert math.isclose(float(result["yaw_correction"]), yaw, abs_tol=1e-3)
    assert torch.allclose(result["translation"], translation, atol=1e-3)


def test_returns_none_with_fewer_than_two_points():
    single = torch.tensor([[0.0, 0.0]])
    assert estimate_se2_ransac(single, single, inlier_threshold=0.2) is None


def _member(agent_id, xy):
    return {"agent_id": agent_id, "box": torch.tensor([xy[0], xy[1], 0.0, 1.0, 1.0, 1.0, 0.0])}


def test_correspondences_by_source_filters_incomplete_and_single_matches():
    groups = [
        [_member("ego", (0.0, 0.0)), _member("source1", (0.1, 0.0))],
        [_member("ego", (5.0, 0.0)), _member("source1", (5.2, 0.0))],
        [_member("source1", (9.0, 0.0))],  # no ego member: ignored
        [_member("ego", (1.0, 1.0)), _member("source2", (1.1, 1.0))],  # only 1 match
    ]
    correspondences = correspondences_by_source(groups)
    assert set(correspondences.keys()) == {"source1"}
    sources, targets = correspondences["source1"]
    assert sources.shape == (2, 2)
    assert targets.shape == (2, 2)


def test_apply_se2_correction_transforms_center_and_yaw():
    detection = {"boxes": torch.tensor([[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.2]])}
    yaw = math.pi / 2
    rotation = _rotation(yaw)
    translation = torch.tensor([3.0, 4.0])
    correction = {
        "rotation": rotation,
        "translation": translation,
        "yaw_correction": torch.tensor(yaw),
    }
    corrected = apply_se2_correction(detection, correction)
    expected_center = rotation @ torch.tensor([1.0, 0.0]) + translation
    expected_yaw = math.atan2(math.sin(0.2 + yaw), math.cos(0.2 + yaw))
    assert torch.allclose(corrected["boxes"][0, :2], expected_center, atol=1e-5)
    assert math.isclose(float(corrected["boxes"][0, 6]), expected_yaw, abs_tol=1e-5)
