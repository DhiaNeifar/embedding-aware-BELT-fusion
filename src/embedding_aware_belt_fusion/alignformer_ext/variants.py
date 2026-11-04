"""Named Verify/Abstain variants, each mapping one cached condition to per-pair decided corrections.

``alignformer`` is the deployed method, unchanged (verified against the
Python path by ``scripts/check_verify_core.py``); every other entry is an
opt-in experiment under its own name and never replaces it.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import torch

from embedding_aware_belt_fusion.alignformer_ext import coupling, uncertainty, verify_cpp
from embedding_aware_belt_fusion.alignformer_pva.inference import build_pairwise_batch


class PairInputs:
    """Per-condition inputs every variant reads: attempted pairs' box rows and Propose outputs."""

    def __init__(self, records: List, cache: Dict[str, np.ndarray], model, robust_config=None, cavcav=None, base=None):
        self.cache, self.model, self.robust_config = cache, model, robust_config
        self.records, self.cavcav, self.base = records, cavcav, base
        self.attempted = np.flatnonzero(cache["attempted"])
        sizes = cache["plan_shape"].prod(axis=1)
        self.plan_off = np.concatenate([[0], np.cumsum(sizes)])
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

    def plan(self, p: int) -> np.ndarray:
        """Cached log Sinkhorn plan of pair ``p`` (dustbin row/column included)."""
        return self.cache["plan"][self.plan_off[p]:self.plan_off[p + 1]].reshape(self.cache["plan_shape"][p])

    def repropose(self, spread: bool) -> np.ndarray:
        """(A, 5) psi, tx, ty, statistic, dof over the attempted pairs, re-solved from the cached plans."""
        with torch.no_grad():
            return np.array(
                [uncertainty.propose(self.plan(p), batch, self.model, self.robust_config, spread=spread)
                 for p, batch in zip(self.attempted, self.batches)],
                np.float32,
            ).reshape(-1, 5)

    def scatter(self, decided_attempted: np.ndarray) -> np.ndarray:
        """(P, 3) over every cached pair, zero (= uncorrected) where nothing was attempted."""
        out = np.zeros((len(self.cache["frame"]), 3), np.float32)
        out[self.attempted] = decided_attempted
        return out


def alignformer(inputs: PairInputs) -> np.ndarray:
    """The deployed AlignFormer: Propose -> Verify (icp_ransac, weighted) -> Wald 0.2 + agreement 1.0 m / floor 4."""
    c = inputs.cache
    decided = verify_cpp.verify(
        inputs.ego_rows, inputs.cav_rows, c["propose"][inputs.attempted], c["statistic"][inputs.attempted],
        c["dof"][inputs.attempted], inputs.model.variance_model, inputs.model.heading_lambda,
    )[0]
    return inputs.scatter(decided)


def alignformer_u(inputs: PairInputs) -> np.ndarray:
    """Idea 1: Propose with the soft match's spread added to each correspondence's variance
    (alignformer_ext.uncertainty), then the deployed Verify / Abstain / agreement unchanged."""
    solved = inputs.repropose(spread=True)
    decided = verify_cpp.verify(
        inputs.ego_rows, inputs.cav_rows, solved[:, :3], solved[:, 3], solved[:, 4],
        inputs.model.variance_model, inputs.model.heading_lambda,
    )[0]
    return inputs.scatter(decided)


def alignformer_c(inputs: PairInputs) -> np.ndarray:
    """Verify's exact fits on box centres only (heading-tip weight 0); Propose, gates, RANSAC, Abstain and the
    agreement rule unchanged. Found by measuring that the dense-frame gap to FreeAlign is AlignFormer's yaw
    (swapping in FreeAlign's rotation recovers it), and that the yaw error comes from the heading tips: detector
    headings are far noisier than the lever arm of >=3 centres spread over tens of metres."""
    from dataclasses import replace

    c = inputs.cache
    decided = verify_cpp.verify(
        inputs.ego_rows, inputs.cav_rows, c["propose"][inputs.attempted], c["statistic"][inputs.attempted],
        c["dof"][inputs.attempted], inputs.model.variance_model, inputs.model.heading_lambda,
        replace(verify_cpp.VerifySettings(), heading_weight=0.0),
    )[0]
    return inputs.scatter(decided)


def alignformer_cw(inputs: PairInputs) -> np.ndarray:
    """alignformer_c, plus: Verify's fit is applied only where the Wald test on Propose's fit also accepts a
    correction (refine.AgreementConfig.gated). In the deployed rule the >=4-inlier consensus floor overrides the
    Wald test, so on real data (noisy detections) a confidently-wrong fit was applied even when Propose's own
    statistic said the correction was indistinguishable from zero."""
    from dataclasses import replace

    c = inputs.cache
    decided = verify_cpp.verify(
        inputs.ego_rows, inputs.cav_rows, c["propose"][inputs.attempted], c["statistic"][inputs.attempted],
        c["dof"][inputs.attempted], inputs.model.variance_model, inputs.model.heading_lambda,
        replace(verify_cpp.VerifySettings(), heading_weight=0.0, gated=True),
    )[0]
    return inputs.scatter(decided)


def _coupled(inputs: PairInputs, consistency: bool, min_support: int = 3, verify_wald: bool = True) -> np.ndarray:
    from dataclasses import replace

    c, model, a = inputs.cache, inputs.model, inputs.attempted
    _, refined, engaged, _, inliers, inlier_off = verify_cpp.verify(
        inputs.ego_rows, inputs.cav_rows, c["propose"][a], c["statistic"][a], c["dof"][a],
        model.variance_model, model.heading_lambda, replace(verify_cpp.VerifySettings(), heading_weight=0.0),
    )
    answered = verify_cpp.wald_answered(c["statistic"][a], c["dof"][a], 0.2)
    weight_ref = 2.0 * model.variance_model.sigma_translation_m ** 2
    decided = np.zeros((len(a), 3), np.float32)
    for k, p in enumerate(a):
        plan = inputs.plan(p)
        stats = coupling.verify_fit_stats(
            inputs.ego_rows[k], inputs.cav_rows[k], refined[k], inliers[inlier_off[k]:inlier_off[k + 1]], weight_ref, plan
        )
        precision = None
        if consistency and engaged[k]:
            with torch.no_grad():
                _, precision = uncertainty.propose(plan, inputs.batches[k], model, inputs.robust_config,
                                                   spread=False, with_precision=True)
        decided[k] = coupling.decide(stats, refined[k], c["propose"][p], bool(answered[k]), bool(engaged[k]),
                                     propose_precision=precision, min_support=min_support, verify_wald=verify_wald)
    return inputs.scatter(decided)


def alignformer_pv(inputs: PairInputs) -> np.ndarray:
    """Idea 2 (coupling.py): Verify's centres-only fit applied only if significant under its OWN uncertainty
    (Abstain's Wald/F test, level 0.2) and backed by >= 3 of Propose's decided Sinkhorn matches; Propose's
    Wald-gated fallback where Verify did not engage. Replaces the 1 m tolerance and the >= 4-inlier override."""
    return _coupled(inputs, consistency=False)


def alignformer_pvc(inputs: PairInputs) -> np.ndarray:
    """alignformer_pv plus: Verify's and Propose's corrections must agree under their combined covariance."""
    return _coupled(inputs, consistency="adaptive")


def alignformer_cwj(inputs: PairInputs) -> np.ndarray:
    """Joint multi-CAV estimation (joint.py) over alignformer_cw-accepted ego<->CAV and CAV<->CAV edges.
    Needs the CAV<->CAV cache (build_propose_cache.py --cavcav)."""
    from embedding_aware_belt_fusion.alignformer_ext import joint

    if inputs.cavcav is None:
        raise FileNotFoundError("alignformer_cwj needs the CAV<->CAV cache: build_propose_cache.py --cavcav")
    return joint.joint_decisions(inputs, inputs.cavcav, inputs.records)


def alignformer_cwa(inputs: PairInputs) -> np.ndarray:
    """alignformer_cwj without fill-in: CAV<->CAV evidence only refines CAVs whose own ego edge was accepted."""
    from embedding_aware_belt_fusion.alignformer_ext import joint

    if inputs.cavcav is None:
        raise FileNotFoundError("alignformer_cwa needs the CAV<->CAV cache: build_propose_cache.py --cavcav")
    return joint.joint_decisions(inputs, inputs.cavcav, inputs.records, fill=False)


def _temporal(doc: str, **settings):
    def run(inputs: PairInputs) -> np.ndarray:
        from embedding_aware_belt_fusion.alignformer_ext import temporal

        return temporal.temporal_decisions(inputs, temporal.TemporalSettings(**settings))

    run.__doc__ = f"Per-frame decisions tracked over time (temporal.py): {doc}"
    return run


_LOAM = (0.0055, 0.0013)   # temporal.LOAM_KITTI, without importing temporal at module load
alignformer_t = _temporal("alignformer_c, odometry drift 1 cm / 0.02 deg per frame per vehicle.", odometry_sd=(0.01, 0.02))
alignformer_t2 = _temporal("alignformer_c, odometry drift 2 cm / 0.05 deg per frame.", odometry_sd=(0.02, 0.05))
alignformer_t5 = _temporal("alignformer_c, odometry drift 5 cm / 0.1 deg per frame.", odometry_sd=(0.05, 0.1))
alignformer_tc = _temporal("alignformer_t + window-majority Abstain (abstain on every failing window).",
                           odometry_sd=(0.01, 0.02), consistency="majority")
alignformer_ta = _temporal("alignformer_t + adaptive window Abstain (failing window applied only if its spread is "
                           "below the estimated localization error).", odometry_sd=(0.01, 0.02), consistency="adaptive")
alignformer_tk = _temporal("alignformer_ta with distance-proportional odometry drift at LOAM/KITTI accuracy "
                           "(0.55 %, 0.0013 deg/m).", odometry_sd=(0.0, 0.0), odometry_rel=_LOAM, consistency="adaptive")
alignformer_tk4 = _temporal("alignformer_tk with 4x LOAM drift (2.2 %, 0.0052 deg/m).", odometry_sd=(0.0, 0.0),
                            odometry_rel=(4 * _LOAM[0], 4 * _LOAM[1]), consistency="adaptive")
# Real LiDAR odometry (--odometry): which fixed accuracy to assume is chosen once, on V2X-Real validation.
_ODO_FLOOR = (0.003, 0.002)   # standstill jitter per frame: m, deg
alignformer_odo1 = _temporal("alignformer_tk with a standstill floor, LOAM accuracy assumed.", odometry_sd=_ODO_FLOOR,
                             odometry_rel=_LOAM, consistency="adaptive")
alignformer_odo4 = _temporal("alignformer_odo1 at 4x LOAM.", odometry_sd=_ODO_FLOOR,
                             odometry_rel=(4 * _LOAM[0], 4 * _LOAM[1]), consistency="adaptive")
alignformer_odo8 = _temporal("alignformer_odo1 at 8x LOAM.", odometry_sd=_ODO_FLOOR,
                             odometry_rel=(8 * _LOAM[0], 8 * _LOAM[1]), consistency="adaptive")
alignformer_tk_w3 = _temporal("alignformer_tk, 3-frame window (CoAnchor's history length).", odometry_sd=(0.0, 0.0),
                              odometry_rel=_LOAM, consistency="adaptive", window=3)
alignformer_tk_w10 = _temporal("alignformer_tk, 10-frame window.", odometry_sd=(0.0, 0.0), odometry_rel=_LOAM,
                               consistency="adaptive", window=10)
alignformer_tk_w50 = _temporal("alignformer_tk, 50-frame window.", odometry_sd=(0.0, 0.0), odometry_rel=_LOAM,
                               consistency="adaptive", window=50)
def alignformer_tkr(inputs: PairInputs):
    """alignformer_tk + alignment-aware confidence (confidence.py): each CAV box's score times the probability that
    its alignment error stays within the IoU-0.7 shift, from the window's own uncertainty. Returns (decided, scale)."""
    from embedding_aware_belt_fusion.alignformer_ext import confidence, temporal

    uncertainty: dict = {}
    settings = temporal.TemporalSettings(odometry_sd=(0.0, 0.0), odometry_rel=_LOAM, consistency="adaptive")
    decided = temporal.temporal_decisions(inputs, settings, uncertainty=uncertainty)
    return decided, confidence.alignment_confidence(inputs, decided, uncertainty)


def _with_reported_confidence(scale: float, calibrate: bool = False, remove_offset: bool = False, offset_se: str = "iid"):
    def run(inputs: PairInputs) -> np.ndarray:
        from embedding_aware_belt_fusion.alignformer_ext import localization_prior, temporal

        settings = temporal.TemporalSettings(odometry_sd=(0.0, 0.0), odometry_rel=_LOAM, consistency="adaptive")
        return localization_prior.fuse_with_reported(inputs, settings, confidence_scale=scale, calibrate=calibrate,
                                                     remove_offset=remove_offset, offset_se=offset_se)

    run.__doc__ = (f"alignformer_tk fused with the CAV's self-reported localization covariance "
                   f"(localization_prior.py), reported sigma x{scale:g}.")
    return run


alignformer_tg = _with_reported_confidence(1.0)
alignformer_tg_x2 = _with_reported_confidence(2.0)
alignformer_tg_half = _with_reported_confidence(0.5)
alignformer_tgo = _with_reported_confidence(1.0, remove_offset=True)
alignformer_tgow = _with_reported_confidence(1.0, remove_offset=True, offset_se="window")
alignformer_tgob = _with_reported_confidence(1.0, remove_offset=True, offset_se="batch")
alignformer_tgc = _with_reported_confidence(1.0, calibrate=True)
alignformer_tgc_x2 = _with_reported_confidence(2.0, calibrate=True)
alignformer_tgc_half = _with_reported_confidence(0.5, calibrate=True)


def alignformer_tg_cal(inputs: PairInputs) -> np.ndarray:
    """alignformer_tg with the dataset's scan-vs-pose offsets given as calibration (calibration.py): per-frame
    estimates are converted to the dataset's pose convention before the window and the confidence fusion."""
    from embedding_aware_belt_fusion.alignformer_ext import calibration, localization_prior, temporal

    settings = temporal.TemporalSettings(odometry_sd=(0.0, 0.0), odometry_rel=_LOAM, consistency="adaptive",
                                         calibration=calibration.active())
    return localization_prior.fuse_with_reported(inputs, settings)


def freealign_cal(inputs: PairInputs) -> np.ndarray:
    """FreeAlign's per-frame corrections with the same dataset calibration as alignformer_tg_cal."""
    from embedding_aware_belt_fusion.alignformer_ext import calibration, temporal

    return calibration.to_recorded_convention(inputs, temporal._freealign_measurements(inputs), calibration.active())


def _multiway(base_variant: Callable[[PairInputs], np.ndarray], register: bool,
              relay_mass: float = None, temper: bool = False) -> Callable[[PairInputs], np.ndarray]:
    def run(inputs: PairInputs) -> np.ndarray:
        from embedding_aware_belt_fusion.alignformer_ext import multiway

        if register:
            joint = multiway.with_joint_registration(inputs, relay_mass=relay_mass)
        else:
            joint = multiway.with_joint_plans(inputs, temper=temper)
        return base_variant(joint)

    run.__doc__ = (f"{base_variant.__name__} on the joint (multi-marginal Sinkhorn) matching of all agents in a frame"
                   + (" plus the joint pose solve through other CAVs" if register else "") + " (multiway.py).")
    return run


alignformer_mm = _multiway(alignformer, register=False)
alignformer_mmj = _multiway(alignformer, register=True)
alignformer_tg_mm = _multiway(alignformer_tg, register=False)
alignformer_tg_mmj = _multiway(alignformer_tg, register=True)
alignformer_mmt = _multiway(alignformer, register=False, temper=True)
alignformer_tg_mmt = _multiway(alignformer_tg, register=False, temper=True)
alignformer_mmr = _multiway(alignformer, register=True, relay_mass=3.0)
alignformer_tg_mmr = _multiway(alignformer_tg, register=True, relay_mass=3.0)


freealign_t = _temporal("FreeAlign's per-frame corrections, otherwise alignformer_t (fairness ablation).",
                        odometry_sd=(0.01, 0.02), source="freealign")
freealign_ta = _temporal("FreeAlign's per-frame corrections, otherwise alignformer_ta (fairness ablation).",
                         odometry_sd=(0.01, 0.02), consistency="adaptive", source="freealign")
freealign_tk = _temporal("FreeAlign's per-frame corrections, otherwise alignformer_tk (fairness ablation).",
                         odometry_sd=(0.0, 0.0), odometry_rel=_LOAM, consistency="adaptive", source="freealign")


VARIANTS: Dict[str, Callable[[PairInputs], np.ndarray]] = {
    "alignformer": alignformer,
    "alignformer_u": alignformer_u,
    "alignformer_c": alignformer_c,
    "alignformer_cw": alignformer_cw,
    "alignformer_pv": alignformer_pv,
    "alignformer_pvc": alignformer_pvc,
    "alignformer_cwj": alignformer_cwj,
    "alignformer_cwa": alignformer_cwa,
    "alignformer_t": alignformer_t,
    "alignformer_t2": alignformer_t2,
    "alignformer_t5": alignformer_t5,
    "alignformer_tc": alignformer_tc,
    "alignformer_tk": alignformer_tk,
    "alignformer_odo1": alignformer_odo1,
    "alignformer_odo4": alignformer_odo4,
    "alignformer_odo8": alignformer_odo8,
    "alignformer_tk4": alignformer_tk4,
    "alignformer_ta": alignformer_ta,
    "alignformer_tk_w3": alignformer_tk_w3,
    "alignformer_tk_w10": alignformer_tk_w10,
    "alignformer_tk_w50": alignformer_tk_w50,
    "alignformer_tkr": alignformer_tkr,
    "alignformer_tg": alignformer_tg,
    "alignformer_tg_x2": alignformer_tg_x2,
    "alignformer_tg_half": alignformer_tg_half,
    "alignformer_tgo": alignformer_tgo,
    "alignformer_tg_cal": alignformer_tg_cal,
    "alignformer_mm": alignformer_mm,
    "alignformer_mmj": alignformer_mmj,
    "alignformer_tg_mm": alignformer_tg_mm,
    "alignformer_tg_mmj": alignformer_tg_mmj,
    "alignformer_mmt": alignformer_mmt,
    "alignformer_tg_mmt": alignformer_tg_mmt,
    "alignformer_mmr": alignformer_mmr,
    "alignformer_tg_mmr": alignformer_tg_mmr,
    "freealign_cal": freealign_cal,
    "alignformer_tgow": alignformer_tgow,
    "alignformer_tgob": alignformer_tgob,
    "alignformer_tgc": alignformer_tgc,
    "alignformer_tgc_x2": alignformer_tgc_x2,
    "alignformer_tgc_half": alignformer_tgc_half,
    "freealign_t": freealign_t,
    "freealign_ta": freealign_ta,
    "freealign_tk": freealign_tk,
}
