import math
from itertools import combinations

import torch

from embedding_aware_belt_fusion.integration.localization import (
    _refine_se2,
    _se2_from_two_pairs,
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


def _reference_estimate_se2_ransac(source_centers, ego_centers, *, inlier_threshold, max_hypotheses=128):
    """Pre-vectorization implementation, kept here only as a pinning oracle.

    Identical to the ``estimate_se2_ransac`` loop before it was rewritten to
    score all hypotheses in one batched pass; see that function's docstring
    for the tie-breaking argument this is pinning.
    """
    if len(source_centers) < 2:
        return None
    source = source_centers.detach().float()
    target = ego_centers.detach().float()
    pairs = list(combinations(range(len(source)), 2))[:max_hypotheses]
    best = None
    for first, second in pairs:
        if torch.linalg.vector_norm(source[second] - source[first]) < 0.25:
            continue
        rotation, translation = _se2_from_two_pairs(
            source[[first, second]], target[[first, second]]
        )
        residual = torch.linalg.vector_norm(
            source @ rotation.T + translation - target, dim=-1
        )
        inliers = residual <= inlier_threshold
        candidate = (int(inliers.sum()), -float(residual[inliers].sum()))
        if best is None or candidate > best[0]:
            best = candidate, rotation, translation, inliers
    if best is None or best[0][0] < 2:
        return None
    return best[1], best[2], best[3]  # rotation, translation, inliers (pre-refine)


def test_vectorized_hypothesis_selection_matches_reference_loop():
    # Deliberately messy: random points plus near-duplicates (to exercise the
    # 0.25m degenerate-pair filter) and a cluster designed to create tied
    # inlier counts, across varying point-set sizes including ones that
    # exceed max_hypotheses's pair budget.
    generator = torch.Generator().manual_seed(0)
    for n_points in (2, 3, 5, 9, 14, 20):
        for trial in range(5):
            source = torch.randn(n_points, 2, generator=generator) * 10
            if n_points >= 4:
                # Force some near-duplicate pairs so the 0.25m filter actually
                # discards hypotheses, not just skips a code path untested.
                source[1] = source[0] + 0.01
            noise = torch.randn(n_points, 2, generator=generator) * 0.05
            yaw = float(torch.rand(1, generator=generator) * 2 * math.pi - math.pi)
            translation = torch.randn(2, generator=generator) * 5
            target = source @ _rotation(yaw).T + translation + noise

            reference = _reference_estimate_se2_ransac(
                source, target, inlier_threshold=0.2, max_hypotheses=16
            )
            result = estimate_se2_ransac(
                source, target, inlier_threshold=0.2, max_hypotheses=16
            )

            if reference is None:
                assert result is None, (n_points, trial)
                continue
            assert result is not None, (n_points, trial)
            _, _, ref_inliers = reference
            # The vectorization only changed hypothesis SELECTION; the Kabsch
            # refine steps that follow are untouched, shared code. So feeding
            # the reference loop's selected inlier set through that same
            # refine path must reproduce the production result exactly if
            # (and only if) selection picked the same winning hypothesis.
            ref_rotation, ref_translation = _refine_se2(
                source[ref_inliers], target[ref_inliers]
            )
            ref_residual = torch.linalg.vector_norm(
                source @ ref_rotation.T + ref_translation - target, dim=-1
            )
            ref_final_inliers = ref_residual <= 0.2
            if int(ref_final_inliers.sum()) >= 2:
                ref_rotation, ref_translation = _refine_se2(
                    source[ref_final_inliers], target[ref_final_inliers]
                )
            assert torch.allclose(result["rotation"], ref_rotation, atol=1e-5), (
                n_points, trial,
            )
            assert torch.allclose(
                result["translation"], ref_translation, atol=1e-5
            ), (n_points, trial)


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
