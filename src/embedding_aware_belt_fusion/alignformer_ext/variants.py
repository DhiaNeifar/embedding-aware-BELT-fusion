"""Named AlignFormer variants, each mapping one cached condition to per-pair decided corrections.

``alignformer`` is the deployed one-frame method, unchanged (verified against the Python path by
``scripts/check_verify_core.py``). ``alignformer_c`` is the one-frame method the paper reports (Verify fits centres
only), and ``alignformer_tk`` is the full method over time. The other entries exist only to select a setting on
V2X-Real validation (window length, assumed odometry accuracy). No variant reads the noise level.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, List

import numpy as np
import torch

from embedding_aware_belt_fusion.alignformer_ext import verify_cpp
from embedding_aware_belt_fusion.alignformer_pva.inference import build_pairwise_batch


class PairInputs:
    """Per-condition inputs every variant reads: attempted pairs' box rows and Propose outputs."""

    def __init__(self, records: List, cache: Dict[str, np.ndarray], model, robust_config=None, base=None):
        self.cache, self.model, self.robust_config = cache, model, robust_config
        self.records, self.base = records, base
        self.attempted = np.flatnonzero(cache["attempted"])
        self.batches = []
        self.ego_rows, self.cav_rows = [], []
        ego_rows_by_frame: Dict[int, np.ndarray] = {}
        for p in self.attempted:
            frame = int(cache["frame"][p])
            record = records[frame]
            if frame not in ego_rows_by_frame:
                ego_rows_by_frame[frame] = verify_cpp.box_rows(record.ego_boxes, record.ego_scores, model.variance_model)
            cav_id = sorted(record.cav_boxes)[int(cache["cav_rank"][p])]
            rows = slice(cache["cav_off"][p], cache["cav_off"][p + 1])
            cav = torch.from_numpy(cache["cav_boxes"][rows])
            # an asynchronous cache (async_cache.py) carries the scores of the stale message itself
            cav_scores = torch.from_numpy(cache["cav_scores"][rows]) if "cav_scores" in cache else record.cav_scores[cav_id]
            self.ego_rows.append(ego_rows_by_frame[frame])
            self.cav_rows.append(verify_cpp.box_rows(cav, cav_scores, model.variance_model))
            self.batches.append(build_pairwise_batch(
                {"boxes": record.ego_boxes, "scores": record.ego_scores},
                {"boxes": cav, "scores": cav_scores},
            ))

    def scatter(self, decided_attempted: np.ndarray) -> np.ndarray:
        """(P, 3) over every cached pair, zero (= the reported pose) where nothing was attempted."""
        out = np.zeros((len(self.cache["frame"]), 3), np.float32)
        out[self.attempted] = decided_attempted
        return out


def _one_frame(inputs: PairInputs, settings: verify_cpp.VerifySettings = None) -> np.ndarray:
    c, a = inputs.cache, inputs.attempted
    decided = verify_cpp.verify(
        inputs.ego_rows, inputs.cav_rows, c["propose"][a], c["statistic"][a], c["dof"][a],
        inputs.model.variance_model, inputs.model.heading_lambda, *([settings] if settings is not None else []),
    )[0]
    return inputs.scatter(decided)


def alignformer(inputs: PairInputs) -> np.ndarray:
    """The deployed AlignFormer: Propose -> Verify (icp_ransac, weighted) -> Wald 0.2 + agreement 1.0 m / floor 4."""
    return _one_frame(inputs)


def alignformer_c(inputs: PairInputs) -> np.ndarray:
    """Verify's exact fits on box centres only (heading-tip weight 0); Propose, gates, RANSAC, Abstain and the
    agreement rule unchanged. Detector headings are far noisier than the lever arm of >=3 centres spread over tens
    of metres, so the heading tips cost yaw accuracy on dense pairs."""
    return _one_frame(inputs, replace(verify_cpp.VerifySettings(), heading_weight=0.0))


def _temporal(doc: str, **settings):
    def run(inputs: PairInputs) -> np.ndarray:
        from embedding_aware_belt_fusion.alignformer_ext import temporal

        return temporal.temporal_decisions(inputs, temporal.TemporalSettings(**settings))

    run.__doc__ = f"alignformer_c's decisions tracked over time (temporal.py): {doc}"
    return run


_LOAM = (0.0055, 0.0013)   # temporal.LOAM_KITTI, without importing temporal at module load
_ODO_FLOOR = (0.003, 0.002)   # standstill jitter per frame: m, deg

alignformer_tk = _temporal("30-frame window, adaptive window Abstain, odometry accuracy assumed at LOAM/KITTI "
                           "(0.55 %, 0.0013 deg/m).", odometry_sd=(0.0, 0.0), odometry_rel=_LOAM, consistency="adaptive")

# Selection on V2X-Real validation only: the odometry accuracy to assume with real LiDAR odometry (--odometry) ...
alignformer_odo1 = _temporal("alignformer_tk with a standstill floor, LOAM accuracy assumed.", odometry_sd=_ODO_FLOOR,
                             odometry_rel=_LOAM, consistency="adaptive")
alignformer_odo4 = _temporal("alignformer_odo1 at 4x LOAM.", odometry_sd=_ODO_FLOOR,
                             odometry_rel=(4 * _LOAM[0], 4 * _LOAM[1]), consistency="adaptive")
alignformer_odo8 = _temporal("alignformer_odo1 at 8x LOAM.", odometry_sd=_ODO_FLOOR,
                             odometry_rel=(8 * _LOAM[0], 8 * _LOAM[1]), consistency="adaptive")
# ... and the window length.
alignformer_tk_w3 = _temporal("alignformer_tk, 3-frame window.", odometry_sd=(0.0, 0.0), odometry_rel=_LOAM,
                              consistency="adaptive", window=3)
alignformer_tk_w10 = _temporal("alignformer_tk, 10-frame window.", odometry_sd=(0.0, 0.0), odometry_rel=_LOAM,
                               consistency="adaptive", window=10)
alignformer_tk_w50 = _temporal("alignformer_tk, 50-frame window.", odometry_sd=(0.0, 0.0), odometry_rel=_LOAM,
                               consistency="adaptive", window=50)


VARIANTS: Dict[str, Callable[[PairInputs], np.ndarray]] = {
    "alignformer": alignformer,
    "alignformer_c": alignformer_c,
    "alignformer_tk": alignformer_tk,
    "alignformer_odo1": alignformer_odo1,
    "alignformer_odo4": alignformer_odo4,
    "alignformer_odo8": alignformer_odo8,
    "alignformer_tk_w3": alignformer_tk_w3,
    "alignformer_tk_w10": alignformer_tk_w10,
    "alignformer_tk_w50": alignformer_tk_w50,
}
