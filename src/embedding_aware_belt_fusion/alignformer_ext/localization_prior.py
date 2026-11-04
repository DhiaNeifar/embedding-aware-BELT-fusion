"""Use the localization system's own confidence: fuse the reported pose with AlignFormer's estimate.

Real vehicles' localization stacks report a covariance with every pose (GNSS/INS fix quality). Each CAV sends
it with its reported pose (here diag(sigma^2, sigma^2, sigma_yaw^2), the benchmark's noise model, optionally
mis-scaled to test a badly calibrated report). The ego then has two estimates of the CAV's pose:

* the reported pose ``N`` with covariance ``G`` (the CAV's self-report), and
* AlignFormer's temporal estimate ``Xhat`` with covariance ``M`` (the window's own spread, temporal.py),

and combines them by their uncertainties (the Gaussian posterior mean):

    X = N + K (Xhat - N),   K = G (G + M)^-1.

With accurate GNSS (G small) the result stays on the reported pose, so weak or biased box evidence cannot pull
it away; with poor GNSS (G large) it follows the box evidence. Where AlignFormer applied no correction, the
reported pose is kept.

With ``calibrate``, the reported covariance also calibrates AlignFormer's own uncertainty. The window spread
under-states AlignFormer's error (errors that persist across frames, detector offsets). But the gap between the
estimate and the reported pose has covariance ``G + M_true`` when the two errors are independent, and ``G`` is
known, so ``M_true = cov(Xhat - N) - G`` per axis, estimated robustly (median / 0.455) over the most recent
pairs before the current one (causal, no ground truth). ``M`` is then at least that level.

With ``remove_offset``, the part of the correction that stays constant over time is removed before fusing.
The gap between AlignFormer's estimate and the reported pose, expressed in the CAV's own frame, contains GNSS
noise (zero-mean over time), AlignFormer's own noise, and a constant: the calibration offset between that
vehicle's sensor data and its pose (measured on CooperScene: up to 0.8 m and 1.5 deg, stable to a few cm per
recording). The running median of the gap per vehicle (causal, reset at scenario boundaries, after
``offset_min_frames`` frames) estimates that constant; subtracting it keeps the GNSS-noise correction but not the
calibration offset. The median of N gaps still carries noise of variance (pi/2) (G + M) / N, which the reported
covariance G makes computable. The offset is applied only if it is significant at the same chi2(3) 0.99 level
the window agreement test uses (sum c^2 / se^2), and then each axis is shrunk by the positive-part James-Stein
factor max(0, 1 - se^2 / c^2): an "offset" that is only GNSS noise (no real offset, or too few frames yet) is not
applied, and a real offset well above its standard error is applied in full. With ``offset_se="window"`` the
standard error accounts for AlignFormer's own error being shared by consecutive estimates (one window, ``settings.window``
frames): its part averages over N / window blocks only, not N frames; ``"batch"`` additionally takes the observed
spread of the per-block medians when there are at least two blocks (whichever is larger).

Poses are (x, y, yaw) of the CAV in the ego frame; a heading error turns the CAV about its
own origin, so G is diagonal in this parametrization.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from embedding_aware_belt_fusion.alignformer_ext import temporal
from embedding_aware_belt_fusion.alignformer_ext.joint import compose, inverse

_FLOOR = np.diag([1e-6, 1e-6, 1e-10])


def reported_covariance(sigma_xy: float, yaw_deg_per_m: float, scale: float = 1.0) -> np.ndarray:
    s = sigma_xy * scale
    return np.diag([s ** 2, s ** 2, np.radians(s * yaw_deg_per_m) ** 2]) + _FLOOR


def fuse_with_reported(inputs, settings: temporal.TemporalSettings, confidence_scale: float = 1.0,
                       calibrate: bool = False, memory: int = 2000, remove_offset: bool = False,
                       offset_min_frames: int = 20, offset_se: str = "iid") -> np.ndarray:
    """(P, 3) psi, tx, ty: AlignFormer's temporal estimate fused with the reported pose by their covariances."""
    from embedding_aware_belt_fusion.alignformer_ext.propose_cache import noisy_poses

    base, records, cache = inputs.base, inputs.records, inputs.cache
    sigma, seed = float(cache["sigma"]), int(cache["seed"])
    uncertainty: dict = {}
    decided = temporal.temporal_decisions(inputs, settings, uncertainty=uncertainty)
    g = reported_covariance(sigma, base.YAW_STD_PER_XY_STD, confidence_scale)
    out = np.zeros_like(decided)
    reported = {}
    gaps_t, gaps_y = deque(maxlen=2 * memory), deque(maxlen=memory)
    offsets: dict = {}          # (segment, cav_id) -> list of gaps in the CAV's own frame
    segment, last_ego = 0, None
    last_frame = -1
    for p in range(len(cache["frame"])):
        if not np.any(decided[p] != 0):
            continue
        f = int(cache["frame"][p])
        record = records[f]
        cav_id = sorted(record.cav_boxes)[int(cache["cav_rank"][p])]
        source = int(cache["past_frame"][p]) if "past_frame" in cache else f   # a delayed message carries its own pose
        if source not in reported:
            reported = {source: noisy_poses(base, records[source], source, sigma, seed)}
        n = temporal._pose2d(base.x1_to_x2(reported[source][cav_id], record.ego_pose))
        d = decided[p]
        estimate = compose(np.array([d[1], d[2], d[0]], np.float64), n)
        m = uncertainty.get("window", {}).get(p, uncertainty["reported"])
        if remove_offset:
            if f != last_frame:
                for k in range(last_frame + 1, f + 1):
                    ego = records[k].ego_pose
                    if last_ego is not None and np.hypot(ego[0] - last_ego[0], ego[1] - last_ego[1]) > temporal.SCENARIO_JUMP_M:
                        segment += 1
                    last_ego = ego
                last_frame = f
            local = compose(inverse(n), estimate)                      # estimate = n o local
            local[2] = temporal._wrap(local[2])
            history = offsets.setdefault((segment, cav_id), [])
            if len(history) >= offset_min_frames:
                h = np.array(history)
                constant = np.array([np.median(h[:, 0]), np.median(h[:, 1]),
                                     np.arctan2(np.median(np.sin(h[:, 2])), np.median(np.cos(h[:, 2])))])
                se2 = (np.pi / 2.0) * np.diag(g + m) / len(h)
                if offset_se != "iid":
                    # consecutive estimates share most of their window, so AlignFormer's own error averages only
                    # over N / window independent blocks (the reported-pose noise is drawn fresh every frame)
                    blocks = max(len(h) / settings.window, 1.0)
                    se2 = (np.pi / 2.0) * (np.diag(g) / len(h) + np.diag(m) / blocks)
                    k = len(h) // settings.window
                    if offset_se == "batch" and k >= 2:
                        means = np.median(h[len(h) - k * settings.window:].reshape(k, settings.window, 3), axis=1)
                        se2 = np.maximum(se2, means.var(axis=0, ddof=1) / k)
                if float(np.sum(constant ** 2 / se2)) > temporal._CONSISTENCY_BOUND:
                    constant = constant * np.maximum(0.0, 1.0 - se2 / np.maximum(constant ** 2, 1e-12))
                    estimate = compose(n, compose(local, inverse(constant)))
            history.append(local)
        delta = estimate - n
        delta[2] = temporal._wrap(delta[2])
        if calibrate:
            if len(gaps_y) >= 50:
                level_t = max(float(np.median(gaps_t)) / temporal.CHI2_1_MEDIAN - g[0, 0], 0.0)
                level_y = max(float(np.median(gaps_y)) / temporal.CHI2_1_MEDIAN - g[2, 2], 0.0)
                m = np.maximum(m, np.diag([level_t, level_t, level_y]))
            gaps_t.extend(delta[:2] ** 2)
            gaps_y.append(delta[2] ** 2)
        fused = n + g @ np.linalg.solve(g + m, delta)
        correction = compose(fused, inverse(n))
        out[p] = (correction[2], correction[0], correction[1])
    return out
