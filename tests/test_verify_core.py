"""verify_core (C++) must reproduce the Python Verify + Abstain + agreement chain of the deployed AlignFormer."""

import math

import numpy as np
import pytest
import torch

from embedding_aware_belt_fusion.alignformer_pva import abstain, refine
from embedding_aware_belt_fusion.alignformer_pva.model import PoseEstimate
from embedding_aware_belt_fusion.alignformer_pva.variance import CorrespondenceVarianceModel

verify_cpp = pytest.importorskip("embedding_aware_belt_fusion.alignformer_ext.verify_cpp")

HEADING_LAMBDA = 2.0
VARIANCE = CorrespondenceVarianceModel(
    mode="scalar", sigma_translation_m=0.25, translation_exponent=1.13,
    sigma_yaw_rad=math.radians(4.6), yaw_exponent=1.93, score_reference=0.4,
)
REFINE = refine.RefineConfig(mode=refine.ICP_RANSAC, gates_m=(6.0, 3.0, 1.5, 0.75, 0.5), weighted=True)
ABSTAIN = abstain.AbstentionConfig(mode=abstain.ABSTAIN, level=0.2)
AGREE = refine.AgreementConfig(tolerance_m=1.0, consensus_floor=4)


def _scene(rng, n_ego, n_cav, shared):
    ego = torch.zeros(n_ego, 7)
    ego[:, :2] = torch.from_numpy(rng.uniform(-60, 60, (n_ego, 2)))
    ego[:, 6] = torch.from_numpy(rng.uniform(-math.pi, math.pi, n_ego))
    psi, t = rng.normal(0, 0.03), rng.normal(0, 1.0, 2)
    cav = torch.zeros(n_cav, 7)
    cav[:, :2] = torch.from_numpy(rng.uniform(-60, 60, (n_cav, 2)))
    cav[:, 6] = torch.from_numpy(rng.uniform(-math.pi, math.pi, n_cav))
    k = min(shared, n_ego, n_cav)
    c, s = math.cos(-psi), math.sin(-psi)
    src = ego[:k, :2] - torch.tensor(t, dtype=torch.float32)
    cav[:k, 0] = c * src[:, 0] - s * src[:, 1] + torch.from_numpy(rng.normal(0, 0.15, k)).float()
    cav[:k, 1] = s * src[:, 0] + c * src[:, 1] + torch.from_numpy(rng.normal(0, 0.15, k)).float()
    cav[:k, 6] = ego[:k, 6] - psi + torch.from_numpy(rng.choice([0.0, math.pi], k)).float()
    scores = lambda n: torch.from_numpy(rng.uniform(0.2, 0.95, n)).float()
    soft = (float(psi + rng.normal(0, 0.01)), *(t + rng.normal(0, 0.3, 2)))
    return ego, scores(n_ego), cav, scores(n_cav), soft


def test_cpp_matches_python_chain_on_random_scenes():
    rng = np.random.default_rng(0)
    cases = []
    for _ in range(300):
        n_ego, n_cav = rng.integers(1, 40), rng.integers(1, 40)
        cases.append(_scene(rng, n_ego, n_cav, rng.integers(0, 30)))
    statistic = rng.uniform(0, 30, len(cases)).astype(np.float32)
    dof = rng.uniform(1, 40, len(cases)).astype(np.float32)

    python = []
    for (ego, es, cav, cs, soft), stat, d in zip(cases, statistic, dof):
        batch = {
            "ego_boxes": ego[None], "cav_boxes": cav[None], "ego_scores": es[None], "cav_scores": cs[None],
            "ego_mask": torch.ones(1, len(ego), dtype=torch.bool), "cav_mask": torch.ones(1, len(cav), dtype=torch.bool),
        }
        estimate = PoseEstimate(
            psi=torch.tensor([soft[0]], dtype=torch.float32), t=torch.tensor([soft[1:]], dtype=torch.float32),
            confidence=torch.ones(1), offset_statistic=torch.tensor([stat]), offset_dof=torch.tensor([d]),
        )
        refined = refine.icp_refine(estimate, batch, REFINE, HEADING_LAMBDA, VARIANCE)
        decided = refine.agree(estimate, refined, abstain.decide(estimate, ABSTAIN), AGREE, HEADING_LAMBDA)
        python.append([float(decided.psi[0]), float(decided.t[0, 0]), float(decided.t[0, 1])])

    out = verify_cpp.verify(
        [verify_cpp.box_rows(e, s, VARIANCE) for e, s, _, _, _ in cases],
        [verify_cpp.box_rows(c, s, VARIANCE) for _, _, c, s, _ in cases],
        np.array([soft for *_, soft in cases], np.float32), statistic, dof, VARIANCE, HEADING_LAMBDA,
        verify_cpp.VerifySettings.from_configs(REFINE, ABSTAIN, AGREE),
    )
    python = np.array(python, np.float32)
    assert np.array_equal(np.any(out[0] != 0, 1), np.any(python != 0, 1))
    np.testing.assert_allclose(out[0], python, atol=1e-4)
