"""Python side of ``verify_core``: AlignFormer's Verify + Abstain + agreement over cached Propose outputs.

Default arguments reproduce ``alignformer_pva.inference.estimate_correction``
with the deployed configs (``FULL_REFINE_CONFIG`` / ``FULL_ABSTAIN_CONFIG`` /
``FULL_AGREEMENT_CONFIG`` in ``scripts/reproduce_opv2v_sweep.py``). Three
inputs are computed here with the ORIGINAL Python code rather than
re-implemented in C++, so each has a single implementation:

* the per-detection centre variances (``variance_model.detection_variances``)
  that the weighted re-solve turns into Kabsch weights;
* the Wald decision on the Propose estimate (``abstain.abstention_threshold``);
* the RANSAC minimal samples for more than 16 hard pairs (``refine._minimal_samples``,
  which draws from torch's generator seeded by the pair count).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch

from embedding_aware_belt_fusion.alignformer_pva import abstain, refine

_BUILD = Path(__file__).resolve().parents[1] / "verify_core" / "build"
if str(_BUILD) not in sys.path:
    sys.path.insert(0, str(_BUILD))
import verify_core  # noqa: E402

BOX_X, BOX_Y, BOX_YAW = 0, 1, 6


@dataclass(frozen=True)
class VerifySettings:
    """Everything ``verify_core.verify_pairs`` reads, defaulting to the deployed AlignFormer."""

    gates_m: Tuple[float, ...] = (6.0, 3.0, 1.5, 0.75, 0.5)
    min_pairs: int = refine.DEFAULT_MIN_PAIRS
    inlier_m: float = refine.DEFAULT_INLIER_M
    weighted: bool = True
    agree_tolerance_m: float = 1.0
    consensus_floor: int = 4
    abstain_level: float = 0.2
    # Not part of the deployed method: scales the heading-tip points' weight in Verify's exact fits
    # (1.0 = AlignFormer as deployed).
    heading_weight: float = 1.0
    # Not part of the deployed method: refine.AgreementConfig.gated (Verify's fit only replaces a Wald-accepted proposal).
    gated: bool = False

    @classmethod
    def from_configs(cls, refine_config, abstain_config, agreement_config) -> "VerifySettings":
        if refine_config.mode != refine.ICP_RANSAC or refine_config.search_m != refine.SEARCH_OFF:
            raise ValueError("verify_core ports icp_ransac with the search step off only")
        if abstain_config.mode != abstain.ABSTAIN or agreement_config.gated:
            raise ValueError("verify_core ports the 'abstain' Wald arm with the ungated agreement rule only")
        return cls(
            gates_m=tuple(refine_config.gates_m), min_pairs=refine_config.min_pairs,
            inlier_m=refine_config.inlier_m, weighted=refine_config.weighted,
            agree_tolerance_m=agreement_config.tolerance_m, consensus_floor=agreement_config.consensus_floor,
            abstain_level=abstain_config.level,
        )


@lru_cache(maxsize=None)
def ransac_table(max_pairs: int) -> Tuple[Tuple[int, ...], ...]:
    """Index ``k``: refine._minimal_samples(k) flattened, for k > EXHAUSTIVE_PAIRS (empty below)."""
    table = [()] * (max_pairs + 1)
    for count in range(refine.EXHAUSTIVE_PAIRS + 1, max_pairs + 1):
        table[count] = tuple(refine._minimal_samples(count, "cpu").reshape(-1).tolist())
    return tuple(table)


def wald_answered(statistic: np.ndarray, dof: np.ndarray, level: float) -> np.ndarray:
    """abstain.decide's 'abstain' rule on the Propose estimate: statistic > threshold(dof)."""
    config = abstain.AbstentionConfig(mode=abstain.ABSTAIN, level=level)
    statistic_t = torch.as_tensor(statistic, dtype=torch.float32)
    threshold = abstain.abstention_threshold(torch.as_tensor(dof, dtype=torch.float32), config)
    return (statistic_t > threshold).numpy()


def box_rows(boxes: torch.Tensor, scores: torch.Tensor, variance_model) -> np.ndarray:
    """(N, 4) float32: x, y, yaw, per-detection centre variance."""
    centre_variance, _ = variance_model.detection_variances(scores.unsqueeze(0))
    return np.ascontiguousarray(
        torch.stack(
            [boxes[:, BOX_X], boxes[:, BOX_Y], boxes[:, BOX_YAW], centre_variance[0]], dim=1
        ).numpy(),
        dtype=np.float32,
    )


def verify(
    ego_rows: Sequence[np.ndarray],
    cav_rows: Sequence[np.ndarray],
    propose: np.ndarray,
    statistic: np.ndarray,
    dof: np.ndarray,
    variance_model,
    heading_lambda: float,
    settings: VerifySettings = VerifySettings(),
):
    """Run Verify + Abstain + agreement over P pairs; returns verify_core's (decided, refined, engaged, consensus)."""
    ego_off = np.concatenate([[0], np.cumsum([len(r) for r in ego_rows])]).astype(np.int64)
    cav_off = np.concatenate([[0], np.cumsum([len(r) for r in cav_rows])]).astype(np.int64)
    max_pairs = max([min(len(e), len(c)) for e, c in zip(ego_rows, cav_rows)] + [0])
    weight_ref = 2.0 * variance_model.sigma_translation_m ** 2 if settings.weighted else 1.0
    return verify_core.verify_pairs(
        np.concatenate(list(ego_rows) or [np.zeros((0, 4), np.float32)]),
        ego_off,
        np.concatenate(list(cav_rows) or [np.zeros((0, 4), np.float32)]),
        cav_off,
        np.ascontiguousarray(propose, dtype=np.float32),
        wald_answered(statistic, dof, settings.abstain_level),
        list(settings.gates_m), settings.min_pairs, settings.inlier_m, settings.weighted,
        float(heading_lambda), float(weight_ref), settings.agree_tolerance_m, settings.consensus_floor,
        [list(t) for t in ransac_table(max(max_pairs, refine.EXHAUSTIVE_PAIRS + 1))],
        float(settings.heading_weight), bool(settings.gated),
    )
