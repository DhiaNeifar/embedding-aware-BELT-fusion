import math
from dataclasses import dataclass
from typing import Optional

import torch

from embedding_aware_belt_fusion.alignformer_pva.refine import (
    AgreementConfig,
    ICP,
    ICP_RANSAC,
    RefineConfig,
    agree,
    disagreement_m,
    icp_refine,
    mutual_nearest,
    robust_exact_solve,
    se2_from_two_pairs,
)


def _rotation(yaw: float) -> torch.Tensor:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return torch.tensor([[cosine, -sine], [sine, cosine]])


def _box(x, y, yaw=0.0):
    return [x, y, 0.0, 1.5, 1.5, 4.0, yaw]


@dataclass(frozen=True)
class _Estimate:
    """Minimal stand-in for model.py's PoseEstimate -- refine.py only ever
    duck-types against this shape via dataclasses.replace, so testing refine
    in isolation with this is a faithful test of its actual contract."""

    psi: torch.Tensor
    t: torch.Tensor
    confidence: torch.Tensor
    refined: Optional[torch.Tensor] = None
    consensus: Optional[torch.Tensor] = None


# ---- mutual_nearest ---------------------------------------------------------


def test_mutual_nearest_is_one_to_one_and_gated():
    ego = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    cav = torch.tensor([[0.1, 0.0], [10.2, 0.0], [50.0, 50.0]])
    ego_idx, cav_idx = mutual_nearest(ego, cav, gate_m=1.0)
    assert ego_idx.tolist() == [0, 1]
    assert cav_idx.tolist() == [0, 1]  # the far cav[2] is correctly excluded


def test_mutual_nearest_breaks_non_mutual_matches():
    # cav[0] is nearest to ego[0], but ego[0] is not cav[0]'s nearest ego --
    # not a valid pair.
    ego = torch.tensor([[0.0, 0.0], [0.3, 0.0]])
    cav = torch.tensor([[0.2, 0.0]])
    ego_idx, cav_idx = mutual_nearest(ego, cav, gate_m=1.0)
    assert ego_idx.tolist() == [1]
    assert cav_idx.tolist() == [0]


# ---- se2_from_two_pairs / robust_exact_solve -------------------------------


def test_se2_from_two_pairs_recovers_exact_transform():
    true_psi = 0.4
    translation = torch.tensor([2.0, -1.0])
    source = torch.tensor([[0.0, 0.0], [5.0, 2.0]])
    target = source @ _rotation(true_psi).T + translation
    psi, t = se2_from_two_pairs(target, source)
    assert math.isclose(float(psi), true_psi, abs_tol=1e-5)
    assert torch.allclose(t, translation, atol=1e-5)


def test_robust_exact_solve_rejects_a_bad_correspondence():
    torch.manual_seed(0)
    true_psi = 0.2
    translation = torch.tensor([1.0, 1.0])
    rotation = _rotation(true_psi)
    ego_centres = torch.randn(6, 2) * 5
    # Physically consistent headings: a CAV observes the same object's yaw
    # rotated by -psi relative to ego, since the two frames differ by psi.
    # Leaving both at a constant (e.g. 0) feeds augment_with_heading's tip
    # points inconsistent data and visibly biases the fit -- caught by this
    # test itself before this fix.
    ego_yaws = torch.rand(6) * 2 * math.pi - math.pi
    cav_yaws_true = ego_yaws - true_psi
    cav_centres_true = (ego_centres - translation) @ rotation  # inverse of R@cav+t=ego
    # Swap one correspondence to be wrong (point to a different ego object).
    cav_centres = cav_centres_true.clone()
    cav_centres[5] = cav_centres_true[0]
    cav_yaws = cav_yaws_true.clone()
    cav_yaws[5] = cav_yaws_true[0]

    ego_boxes = torch.tensor(
        [_box(x.item(), y.item(), yaw.item()) for (x, y), yaw in zip(ego_centres, ego_yaws)]
    )
    cav_boxes = torch.tensor(
        [_box(x.item(), y.item(), yaw.item()) for (x, y), yaw in zip(cav_centres, cav_yaws)]
    )
    ego_idx = torch.arange(6)
    cav_idx = torch.arange(6)

    psi, t, inliers = robust_exact_solve(
        ego_boxes, cav_boxes, ego_idx, cav_idx, heading_lambda=2.0, inlier_m=0.5
    )
    assert math.isclose(float(psi), true_psi, abs_tol=1e-3)
    assert torch.allclose(t, translation, atol=1e-3)
    assert not bool(inliers[5])  # the swapped correspondence is rejected
    assert bool(inliers[:5].all())


# ---- icp_refine: sharpens a blurry soft estimate ---------------------------


def test_icp_refine_improves_on_a_noisy_soft_estimate():
    torch.manual_seed(1)
    true_psi = 0.15
    translation = torch.tensor([0.8, -0.5])
    rotation = _rotation(true_psi)
    ego_centres = torch.randn(6, 2) * 5
    ego_yaws = torch.rand(6) * 2 * math.pi - math.pi
    cav_yaws = ego_yaws - true_psi
    cav_centres = (ego_centres - translation) @ rotation

    ego_boxes = torch.tensor(
        [_box(x.item(), y.item(), yaw.item()) for (x, y), yaw in zip(ego_centres, ego_yaws)]
    )
    cav_boxes = torch.tensor(
        [_box(x.item(), y.item(), yaw.item()) for (x, y), yaw in zip(cav_centres, cav_yaws)]
    )

    # A deliberately blurry starting estimate, off the true transform by
    # enough to need refining but not so much that it exceeds the first gate
    # (points sit ~5-10m out, so a 0.3 rad rotation error alone would move
    # them several metres -- past a 3m gate entirely, which is the
    # documented "nothing a gate can reach" failure mode, not what this test
    # is checking).
    soft_psi = torch.tensor([true_psi + 0.05])
    soft_t = translation.unsqueeze(0) + torch.tensor([[0.4, -0.4]])
    estimate = _Estimate(
        psi=soft_psi, t=soft_t, confidence=torch.tensor([6.0])
    )
    batch = {
        "ego_boxes": ego_boxes.unsqueeze(0),
        "cav_boxes": cav_boxes.unsqueeze(0),
    }

    config = RefineConfig(mode=ICP, gates_m=(3.0, 1.0, 0.3), min_pairs=3)
    refined = icp_refine(estimate, batch, config, heading_lambda=2.0)

    soft_error = abs(float(soft_psi[0]) - true_psi) + float(
        (soft_t[0] - translation).norm()
    )
    refined_error = abs(float(refined.psi[0]) - true_psi) + float(
        (refined.t[0] - translation).norm()
    )
    assert refined_error < soft_error
    assert refined_error < 1e-2
    assert bool(refined.refined[0])


def test_icp_refine_disabled_mode_is_a_no_op():
    estimate = _Estimate(
        psi=torch.tensor([0.1]), t=torch.tensor([[1.0, 2.0]]), confidence=torch.tensor([3.0])
    )
    config = RefineConfig(mode="none")
    batch = {"ego_boxes": torch.zeros(1, 3, 7), "cav_boxes": torch.zeros(1, 3, 7)}
    result = icp_refine(estimate, batch, config, heading_lambda=2.0)
    assert result is estimate


def test_icp_refine_below_evidence_floor_keeps_soft_estimate():
    # Only one mutually-nearest pair possible -- below min_pairs=3, so the
    # soft (psi, t) must come back untouched.
    ego_boxes = torch.tensor([_box(0.0, 0.0)])
    cav_boxes = torch.tensor([_box(0.05, 0.0)])
    soft_psi = torch.tensor([0.42])
    soft_t = torch.tensor([[1.23, -4.56]])
    estimate = _Estimate(psi=soft_psi, t=soft_t, confidence=torch.tensor([1.0]))
    batch = {"ego_boxes": ego_boxes.unsqueeze(0), "cav_boxes": cav_boxes.unsqueeze(0)}
    config = RefineConfig(mode=ICP, min_pairs=3)
    refined = icp_refine(estimate, batch, config, heading_lambda=2.0)
    assert torch.allclose(refined.psi, soft_psi)
    assert torch.allclose(refined.t, soft_t)
    assert not bool(refined.refined[0])


# ---- the agreement rule -----------------------------------------------------


def test_agreement_rule_keeps_exact_fit_when_fits_agree():
    soft = _Estimate(psi=torch.tensor([0.10]), t=torch.tensor([[1.0, 1.0]]), confidence=torch.tensor([5.0]))
    exact = _Estimate(
        psi=torch.tensor([0.11]), t=torch.tensor([[1.02, 0.98]]), confidence=torch.tensor([5.0]),
        refined=torch.tensor([True]), consensus=torch.tensor([5]),
    )
    fallback = soft
    config = AgreementConfig(tolerance_m=0.5)
    result = agree(soft, exact, fallback, config, heading_lambda=2.0)
    assert torch.allclose(result.psi, exact.psi)
    assert torch.allclose(result.t, exact.t)


def test_agreement_rule_zeroes_correction_when_fits_disagree():
    soft = _Estimate(psi=torch.tensor([0.10]), t=torch.tensor([[1.0, 1.0]]), confidence=torch.tensor([5.0]))
    exact = _Estimate(
        psi=torch.tensor([1.5]), t=torch.tensor([[20.0, -20.0]]), confidence=torch.tensor([5.0]),
        refined=torch.tensor([True]), consensus=torch.tensor([0]),
    )
    fallback = soft
    config = AgreementConfig(tolerance_m=0.5)
    result = agree(soft, exact, fallback, config, heading_lambda=2.0)
    assert float(result.psi[0]) == 0.0
    assert torch.allclose(result.t[0], torch.zeros(2))


def test_agreement_rule_falls_back_when_not_engaged():
    soft = _Estimate(psi=torch.tensor([0.10]), t=torch.tensor([[1.0, 1.0]]), confidence=torch.tensor([5.0]))
    exact = _Estimate(
        psi=torch.tensor([9.9]), t=torch.tensor([[99.0, 99.0]]), confidence=torch.tensor([0.0]),
        refined=torch.tensor([False]), consensus=torch.tensor([0]),
    )
    fallback = _Estimate(
        psi=torch.tensor([0.33]), t=torch.tensor([[7.0, 7.0]]), confidence=torch.tensor([5.0])
    )
    config = AgreementConfig(tolerance_m=0.5)
    result = agree(soft, exact, fallback, config, heading_lambda=2.0)
    assert torch.allclose(result.psi, fallback.psi)
    assert torch.allclose(result.t, fallback.t)


def test_agreement_rule_consensus_floor_overrides_tolerance():
    # The two fits disagree by more than tolerance, but the exact fit's own
    # consensus is large enough to trust it anyway.
    soft = _Estimate(psi=torch.tensor([0.0]), t=torch.tensor([[0.0, 0.0]]), confidence=torch.tensor([5.0]))
    exact = _Estimate(
        psi=torch.tensor([0.0]), t=torch.tensor([[2.0, 0.0]]), confidence=torch.tensor([5.0]),
        refined=torch.tensor([True]), consensus=torch.tensor([10]),
    )
    fallback = soft
    config = AgreementConfig(tolerance_m=0.5, consensus_floor=5)
    result = agree(soft, exact, fallback, config, heading_lambda=2.0)
    assert torch.allclose(result.t, exact.t)


def test_disagreement_m_combines_translation_and_heading():
    soft = _Estimate(psi=torch.tensor([0.0]), t=torch.tensor([[0.0, 0.0]]), confidence=torch.tensor([1.0]))
    exact = _Estimate(
        psi=torch.tensor([math.pi / 2]), t=torch.tensor([[3.0, 4.0]]), confidence=torch.tensor([1.0]),
        refined=torch.tensor([True]), consensus=torch.tensor([3]),
    )
    gap = disagreement_m(soft, exact, heading_lambda=2.0)
    expected = 5.0 + 2.0 * (math.pi / 2)  # |t| gap + lambda * heading gap
    assert math.isclose(float(gap[0]), expected, abs_tol=1e-5)
