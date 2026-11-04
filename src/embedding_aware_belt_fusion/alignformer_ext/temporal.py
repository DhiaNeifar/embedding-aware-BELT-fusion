"""alignformer_t: Propose -> Verify -> Abstain over time, as a robust fixed-lag estimate of each CAV's pose.

Single-frame box registration is at its ceiling (alignformer_c matches a least-squares fit on the TRUE
correspondences), and pairs with too few shared objects get no correction at all. Both are limits of using one
frame. Each CAV's pose relative to ego is instead estimated from the last ``window`` frames:

* measurement at frame t: the per-frame method's decided correction ``D_t`` gives ``Xhat_t = D_t o N_t``
  (``N_t`` = the CAV->ego transform from the CAV's reported, noisy pose); abstained frames give none. The
  per-frame method is alignformer_c;
* every stored measurement is carried to the current frame by both vehicles' own odometry between consecutive
  frames, ``X <- A o X o B`` (``A`` ego motion f-1 -> f, ``B`` CAV motion f -> f-1), each a noisy random walk.
  Per step and vehicle the drift sd is ``odometry_sd`` (fixed per frame) plus ``odometry_rel`` times the
  distance moved (translation: fraction of distance; yaw: deg per metre) -- the form in which odometry accuracy
  is reported (KITTI: LOAM 0.55 %, 0.0013 deg/m), so a parked vehicle or a roadside unit does not drift. Its odometry covariance ``C`` is propagated through the exact
  Jacobians of that composition, so an odometry yaw error correctly costs translation in proportion to the
  CAV's distance (a flat ``k Q`` growth over-trusts old measurements);
* estimate: robust generalised least squares on SE(2) over the window. Carried measurements are correlated:
  ages i < j share the last i odometry steps, so their joint covariance is ``C_younger + delta_ab S``;
  treating them as independent over-counts old measurements. Outliers are down-weighted by iteratively
  reweighting each measurement's ``S`` with a Cauchy kernel (standard 2.385 constant) on its residual
  standardised by ``S + C``. ``S`` = per-measurement noise
  (diagonal: translation, yaw), estimated online and causally from consecutive measurements of the same CAV:
  ``r = Xhat_t - carried Xhat_{t-1}`` has variance ``2 S + C_1`` (one frame of odometry), so
  ``S = (median(r^2) / 0.455 - C_1) / 2`` (robust to wrong fits; residuals against the window solution
  would absorb odometry error and inflate ``S``). Verify's own fit covariances are NOT used: they miss
  errors common to all of a CAV's boxes and are badly over-confident (median NIS 4 on OPV2V, 17 on V2X-Real
  vs 2.4 calibrated), and weighting by them measured worse than equal weights;
* Abstain becomes a soft decision: a wrong single-frame fit is down-weighted by the kernel instead of being
  applied or discarded wholesale, and a frame alignformer_c abstains on is still corrected from the window;
* with ``consistency``, Abstain also tests the window itself: the estimate is applied only if a majority of the
  window's measurements agree with it (standardised residual within chi2(3) at 0.99). The agreement scale is
  the measurements' total spread about their window estimate (robust, pooled over all CAVs, causal), not
  ``S``: measurement errors are correlated from frame to frame (V2X-Real: error 0.31 m vs frame-to-frame change
  0.15 m), so ``S`` sees only their fast part and would reject good windows (8 % on OPV2V, 24 % on V2X-Real).
  ``consistency="majority"`` abstains on every window that fails. That is right when the reported pose is good
  (sigma 0) and wrong when it is not: windows that fail on OPV2V at sigma 1 still have 0.25 m error vs 1.21 m
  uncorrected. Abstain is a choice between two estimates, the reported pose and the window's, so
  ``consistency="adaptive"`` applies a failing window only if its spread (median distance of its measurements
  from the estimate, at the CAV's origin) is below the reported poses' error level, estimated without ground
  truth as the median distance between reported pose and window estimate over recent windows that pass
  (assumes localization quality is similar across CAVs and slowly varying); until that level is known it
  abstains. A pair with no real
  overlap yields corrections that change from frame to frame (CooperScene CAV 3: 0.47 m / 0.66 deg per frame
  vs 0.08 m on OPV2V, median 0 objects shared with ego); single frames cannot tell, the window can. Majority is
  the robust estimate's own breakdown point: below it the estimate is the outliers';
* output at frame t: ``D'_t = X_t o N_t^-1``, applied like any decided correction.

Causal (past and current frames only). Windows reset at scenario boundaries and when the CAV leaves the frame
table. Odometry is each vehicle's real LiDAR odometry (odometry.py) when the sweep passes --odometry; otherwise each
vehicle's true motion plus seeded random-walk noise, which is not valid for reported results. No alignformer_pva code
changes.
"""

from __future__ import annotations

import zlib
from collections import deque
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import chi2

from embedding_aware_belt_fusion.alignformer_ext import odometry, verify_cpp
from embedding_aware_belt_fusion.alignformer_ext.se2 import compose, inverse

SCENARIO_JUMP_M = 5.0          # ego displacement between consecutive frame-table rows that marks a new scenario
CAUCHY_C = 2.385               # Cauchy kernel constant (95% efficiency under Gaussian noise)
CHI2_1_MEDIAN = 0.455          # median of chi2(1): robust variance from squared residuals
SCALE_WARMUP = (0.1, 0.2)      # (m, deg) per-measurement noise before any residual has been seen
_IRLS_ITERATIONS = 10


CONSISTENCY_LEVEL = 0.99       # chi2(3) quantile a measurement must fall within to agree with the window estimate
LOAM_KITTI = (0.0055, 0.0013)  # LOAM on the KITTI odometry benchmark: 0.55 % of distance, 0.0013 deg/m
_CONSISTENCY_BOUND = float(chi2.ppf(CONSISTENCY_LEVEL, 3))


@dataclass(frozen=True)
class TemporalSettings:
    odometry_sd: Tuple[float, float] = (0.01, 0.02)  # per frame per vehicle: translation (m), yaw (deg)
    odometry_rel: Tuple[float, float] = (0.0, 0.0)   # per metre moved: translation (fraction), yaw (deg/m)
    window: int = 30                                  # frames
    scale_memory: int = 2000                          # residuals kept for the noise-scale estimate
    consistency: str = "off"                          # off | majority | adaptive (window-agreement Abstain)


def _pose2d(matrix) -> np.ndarray:
    m = np.asarray(matrix, np.float64)
    return np.array([m[0, 3], m[1, 3], np.arctan2(m[1, 0], m[0, 0])])


def _carry(a: np.ndarray, x: np.ndarray, b: np.ndarray, cov: np.ndarray, q_a: np.ndarray, q_b: np.ndarray):
    """``A o X o B`` and its covariance from ``cov`` (of X) and odometry noise ``q_a`` on A and ``q_b`` on B
    (additive on their x, y, yaw), by the Jacobians of ``compose``."""
    def d_left(m, r):   # d compose(m, r) / d m
        c, s = np.cos(m[2]), np.sin(m[2])
        return np.array([[1.0, 0.0, -s * r[0] - c * r[1]], [0.0, 1.0, c * r[0] - s * r[1]], [0.0, 0.0, 1.0]])

    def d_right(m):     # d compose(m, r) / d r
        c, s = np.cos(m[2]), np.sin(m[2])
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    m = compose(a, x)
    j_m = d_left(m, b)
    j_a, j_x = j_m @ d_left(a, x), j_m @ d_right(a)
    j_b = d_right(m)
    return compose(m, b), j_x @ cov @ j_x.T + j_a @ q_a @ j_a.T + j_b @ q_b @ j_b.T


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def _odometry_sd(step: np.ndarray, settings: TemporalSettings) -> np.ndarray:
    moved = float(np.hypot(step[0], step[1]))
    t_sd = settings.odometry_sd[0] + settings.odometry_rel[0] * moved
    y_sd = np.radians(settings.odometry_sd[1] + settings.odometry_rel[1] * moved)
    return np.array([t_sd, t_sd, max(y_sd, 1e-9)])


def _odometry(true_step: np.ndarray, settings: TemporalSettings, seed: int, frame: int, agent: str):
    """Noisy odometry step and its covariance: true motion + drift, deterministic per (seed, frame, vehicle)."""
    sd = _odometry_sd(true_step, settings)
    noise = np.random.default_rng([int(seed), int(frame), zlib.crc32(agent.encode())]).normal(0.0, sd)
    return true_step + noise, np.diag(np.maximum(sd, 1e-6) ** 2)


def _measured_step(table, source: int, target: int, agent: str, settings: TemporalSettings):
    """A step of the vehicle's real LiDAR odometry and its assumed covariance (fixed accuracy, never the simulator's);
    (None, None) without both scans."""
    step = table.between(source, target, agent)
    if step is None:
        return None, None
    return step, np.diag(np.maximum(_odometry_sd(step, settings), 1e-6) ** 2)


class _NoiseScale:
    """Robust online per-measurement noise (translation, yaw variances) from consecutive-measurement differences,
    and the measurements' total spread about their window estimate (for the agreement test)."""

    def __init__(self, memory: int):
        self.translation, self.yaw = deque(maxlen=2 * memory), deque(maxlen=memory)
        self.spread_translation, self.spread_yaw = deque(maxlen=2 * memory), deque(maxlen=memory)
        self.reported_error = deque(maxlen=memory)
        self.reported_yaw_error = deque(maxlen=memory)

    def localization_level(self):
        """Median distance (m) between reported poses and passing window estimates; None until known."""
        return float(np.median(self.reported_error)) if len(self.reported_error) >= 50 else None

    def spread(self) -> np.ndarray:
        if len(self.spread_yaw) < 20:
            return self.matrix()
        t = max(float(np.median(self.spread_translation)) / CHI2_1_MEDIAN, 1e-6)
        y = max(float(np.median(self.spread_yaw)) / CHI2_1_MEDIAN, 1e-10)
        return np.diag([t, t, y])

    def observe_spread(self, residual: np.ndarray) -> None:
        self.spread_translation.extend(residual[:2] ** 2)
        self.spread_yaw.append(residual[2] ** 2)

    def matrix(self) -> np.ndarray:
        if len(self.yaw) < 20:
            t, y = SCALE_WARMUP[0] ** 2, np.radians(SCALE_WARMUP[1]) ** 2
        else:
            t = max(float(np.median(self.translation)) / CHI2_1_MEDIAN / 2.0, 1e-6)
            y = max(float(np.median(self.yaw)) / CHI2_1_MEDIAN / 2.0, 1e-10)
        return np.diag([t, t, y])

    def observe(self, difference: np.ndarray, odometry_cov: np.ndarray) -> None:
        """``difference`` = new measurement - previous frame's carried measurement, ``odometry_cov`` its C_1."""
        excess = difference ** 2 - np.diag(odometry_cov) * CHI2_1_MEDIAN
        self.translation.extend(excess[:2])
        self.yaw.append(excess[2])


def _robust_gls(estimates: np.ndarray, odometry_covs: List[np.ndarray], noise: np.ndarray) -> np.ndarray:
    """Robust GLS mean of window measurements ordered oldest -> newest (shared odometry -> correlated errors)."""
    return _robust_gls_agreement(estimates, odometry_covs, noise)[0]


def _robust_gls_agreement(estimates: np.ndarray, odometry_covs: List[np.ndarray], noise: np.ndarray,
                          spread: np.ndarray = None):
    """``_robust_gls`` and the fraction of measurements agreeing with it: residual standardised by ``spread + C``
    (default ``noise + C``) within chi2(3) at ``CONSISTENCY_LEVEL``."""
    x = np.array([np.median(estimates[:, 0]), np.median(estimates[:, 1]),
                  np.arctan2(np.median(np.sin(estimates[:, 2])), np.median(np.cos(estimates[:, 2])))])
    n = len(estimates)
    if n == 1:
        return estimates[0].copy(), 1.0
    shared = np.empty((3 * n, 3 * n))
    for a in range(n):
        for b in range(a, n):
            block = odometry_covs[b]          # b is the younger of the two: their shared odometry
            shared[3 * a:3 * a + 3, 3 * b:3 * b + 3] = block
            shared[3 * b:3 * b + 3, 3 * a:3 * a + 3] = block
    marginal = [np.linalg.inv(noise + cov) for cov in odometry_covs]
    stacked = np.tile(np.eye(3), (n, 1))
    for _ in range(_IRLS_ITERATIONS):
        r = estimates - x
        r[:, 2] = _wrap(r[:, 2])
        kernel = np.array([1.0 / (1.0 + float(ra @ m @ ra) / CAUCHY_C ** 2) for ra, m in zip(r, marginal)])
        sigma = shared.copy()
        for a in range(n):
            sigma[3 * a:3 * a + 3, 3 * a:3 * a + 3] += noise / kernel[a]
        info = np.linalg.solve(sigma, stacked)                 # Sigma^-1 H
        normal = stacked.T @ info
        step = np.linalg.solve(normal, info.T @ r.reshape(-1))
        x = x + step
        x[2] = _wrap(x[2])
        if np.abs(step).max() < 1e-7:
            break
    r = estimates - x
    r[:, 2] = _wrap(r[:, 2])
    if spread is not None:
        marginal = [np.linalg.inv(spread + cov) for cov in odometry_covs]
    agree = np.mean([float(ra @ m @ ra) <= _CONSISTENCY_BOUND for ra, m in zip(r, marginal)])
    return x, float(agree)


def _measurements(inputs) -> np.ndarray:
    """alignformer_c's decided corrections (P, 3: psi, tx, ty); zero = abstained / not attempted."""
    c, model, a = inputs.cache, inputs.model, inputs.attempted
    decided = verify_cpp.verify(
        inputs.ego_rows, inputs.cav_rows, c["propose"][a], c["statistic"][a], c["dof"][a],
        model.variance_model, model.heading_lambda, replace(verify_cpp.VerifySettings(), heading_weight=0.0),
    )[0]
    return inputs.scatter(decided)


def temporal_decisions(inputs, settings: TemporalSettings = TemporalSettings()) -> np.ndarray:
    """(P, 3) psi, tx, ty per cached pair from each CAV's robust fixed-lag pose estimate.

    Delayed messages (an asynchronous cache, async_cache.py, carries ``past_frame``): the reported pose and the CAV's
    odometry are the ones IN THE MESSAGE, i.e. of the frame it was produced at, placed in the ego's current frame;
    the delay itself is never read."""
    base, records, cache = inputs.base, inputs.records, inputs.cache
    if base is None:
        raise ValueError("alignformer_t needs PairInputs(base=...) for the CAVs' reported (noisy) poses")
    from embedding_aware_belt_fusion.alignformer_ext.propose_cache import noisy_poses

    sigma, seed = float(cache["sigma"]), int(cache["seed"])
    decided = _measurements(inputs)
    scale = _NoiseScale(settings.scale_memory)
    table = odometry.active()

    pair_of: Dict[Tuple[int, str], int] = {}
    for p in range(len(cache["frame"])):
        f = int(cache["frame"][p])
        pair_of[(f, sorted(records[f].cav_boxes)[int(cache["cav_rank"][p])])] = p

    out = np.zeros((len(cache["frame"]), 3), np.float32)
    message_poses: Dict[int, dict] = {}
    windows: Dict[str, deque] = {}   # cav_id -> deque of [carried estimate (current ego frame), its odometry covariance, age]
    for f, record in enumerate(records):
        prev = records[f - 1] if f else None
        if prev is None or np.hypot(record.ego_pose[0] - prev.ego_pose[0], record.ego_pose[1] - prev.ego_pose[1]) > SCENARIO_JUMP_M:
            windows = {}
        if windows and table is not None:
            ego_odo, q_ego = _measured_step(table, f - 1, f, odometry.EGO, settings)
            if ego_odo is None:
                windows = {}
        elif windows:
            ego_odo, q_ego = _odometry(_pose2d(base.x1_to_x2(prev.ego_pose, record.ego_pose)), settings, seed, f, "ego")
        alive: Dict[str, deque] = {}
        for cav_id, pose in noisy_poses(base, record, f, sigma, seed).items():
            window = windows.get(cav_id)
            cav_now, cav_prev = record.cav_poses.get(cav_id), prev.cav_poses.get(cav_id) if prev is not None else None
            message_frame = f
            if "past_frame" in cache:          # delayed message: its own pose and odometry, of the frame it was made at
                q = pair_of.get((f, cav_id))
                if q is None:
                    continue
                g = int(cache["past_frame"][q])
                if g not in message_poses:
                    message_poses.clear()
                    message_poses[g] = noisy_poses(base, records[g], g, sigma, seed)
                pose = message_poses[g][cav_id]
                cav_now = records[g].cav_poses.get(cav_id)
                cav_prev = records[g - 1].cav_poses.get(cav_id) if g > 0 else None
                message_frame = g
            if window is not None and cav_now is not None and cav_prev is not None:
                if table is not None:
                    cav_odo, q_cav = _measured_step(table, message_frame, message_frame - 1, cav_id, settings)
                else:
                    cav_odo, q_cav = _odometry(_pose2d(base.x1_to_x2(cav_now, cav_prev)), settings, seed, f, cav_id)
            else:
                cav_odo = None
            if cav_odo is not None:
                window = deque(([*_carry(ego_odo, x, cav_odo, cov, q_ego, q_cav), age + 1]
                                for x, cov, age in window if age + 1 < settings.window), maxlen=settings.window)
            else:
                window = deque(maxlen=settings.window)
            n = _pose2d(base.x1_to_x2(pose, record.ego_pose))
            p = pair_of.get((f, cav_id))
            if p is not None and np.any(decided[p] != 0):
                d = decided[p]
                z = compose(np.array([d[1], d[2], d[0]], np.float64), n)
                if window and window[-1][2] == 1:
                    difference = z - window[-1][0]
                    difference[2] = _wrap(difference[2])
                    scale.observe(difference, window[-1][1])
                window.append([z, np.zeros((3, 3)), 0])
            if not window:
                continue
            alive[cav_id] = window
            noise = scale.matrix()
            measured = np.array([w for w, _, _ in window])
            x, agree = _robust_gls_agreement(measured, [cov for _, cov, _ in window], noise, scale.spread())
            if len(window) > 1 and window[-1][2] == 0:
                residual = window[-1][0] - x
                residual[2] = _wrap(residual[2])
                scale.observe_spread(residual)
            if settings.consistency != "off":
                if agree > 0.5:
                    scale.reported_error.append(float(np.hypot(*(x[:2] - n[:2]))))
                    scale.reported_yaw_error.append(float(abs(_wrap(x[2] - n[2]))))
                else:
                    level = scale.localization_level()
                    window_spread = float(np.median(np.hypot(*(np.array([z for z, _, _ in window])[:, :2] - x[:2]).T)))
                    if settings.consistency == "majority" or level is None or window_spread >= level:
                        continue
            if p is not None:
                correction = compose(x, inverse(n))
                out[p] = (correction[2], correction[0], correction[1])
        windows = alive
    return out
