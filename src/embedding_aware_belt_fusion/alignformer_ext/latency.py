"""Idea A: estimate a CAV's pose error AND its unknown latency together, from boxes and sender velocities.

A message that is tau frames old shows every object where it was tau frames ago. With the sender's per-box velocity
v (m per frame, in the ego frame after naive placement), the object is now near b + v tau, so the ego's current box
of the same object satisfies

    e ~ D (b + v tau),      D in SE(2) the pose correction, tau the latency.

Static objects (v = 0) pin D whatever tau is; moving objects with different velocities pin tau, since a wrong tau
moves each of them by its own v (tau - tau_true) and no single rigid D can undo that. The receiver is never told
tau. For each candidate tau on a grid, the boxes are moved forward by v tau and the unchanged per-frame AlignFormer
(Propose network + C++ Verify) runs on them; the candidate whose Verify consensus is largest (most objects agreeing
within the inlier threshold; ties broken by the inliers' residual) is the estimate. Latency is a property of the
sender's pipeline and link, so it is also pooled over time per CAV (causal running median over recent frames).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from embedding_aware_belt_fusion.alignformer_ext import verify_cpp


def moved_cache(cache: Dict[str, np.ndarray], tau: float) -> Dict[str, np.ndarray]:
    """A copy of an asynchronous cache whose CAV boxes are moved forward by velocity x tau (frames)."""
    out = dict(cache)
    boxes = cache["cav_boxes"].copy()
    boxes[:, :2] += cache["cav_velocity"] * float(tau)
    out["cav_boxes"] = boxes
    return out


def _propose(inputs, model, robust_config) -> None:
    """Run the UNMODIFIED Propose network on every attempted pair of ``inputs`` (its boxes may have moved) and write
    propose / statistic / dof into its cache, as propose_cache.build_condition does."""
    cache = inputs.cache
    cache["propose"], cache["statistic"], cache["dof"] = cache["propose"].copy(), cache["statistic"].copy(), cache["dof"].copy()
    with torch.no_grad():
        for i, p in enumerate(inputs.attempted):
            estimate = model(inputs.batches[i], robust=robust_config, statistic=True)
            cache["propose"][p] = (float(estimate.psi[0]), float(estimate.t[0, 0]), float(estimate.t[0, 1]))
            cache["statistic"][p] = float(estimate.offset_statistic[0])
            cache["dof"][p] = float(estimate.offset_dof[0])


def _residuals(inputs, refined: np.ndarray, inliers: np.ndarray, inlier_off: np.ndarray) -> np.ndarray:
    """Mean centre residual of each attempted pair's Verify inliers under its refined pose (inf without inliers)."""
    cache, out = inputs.cache, np.full(len(inputs.attempted), np.inf)
    for i, p in enumerate(inputs.attempted):
        pairs = inliers[inlier_off[i]:inlier_off[i + 1]]
        if not len(pairs):
            continue
        record = inputs.records[int(cache["frame"][p])]
        ego = np.asarray(record.ego_boxes, np.float64)[pairs[:, 0], :2]
        cav = cache["cav_boxes"][cache["cav_off"][p]:cache["cav_off"][p + 1]][pairs[:, 1], :2].astype(np.float64)
        psi, tx, ty = map(float, refined[i])
        c, s = np.cos(psi), np.sin(psi)
        moved = np.stack([c * cav[:, 0] - s * cav[:, 1] + tx, s * cav[:, 0] + c * cav[:, 1] + ty], 1)
        out[i] = float(np.linalg.norm(moved - ego, axis=1).mean())
    return out


def latency_grid(records, cache, model, robust_config, base, taus: Sequence[int]) -> Dict[int, dict]:
    """Per candidate latency: the moved cache, its pair inputs, and AlignFormer's decision on it."""
    from embedding_aware_belt_fusion.alignformer_ext import variants

    out = {}
    for tau in taus:
        moved = moved_cache(cache, tau)
        inputs = variants.PairInputs(records, moved, model, robust_config, base=base)
        _propose(inputs, model, robust_config)
        decided, refined, engaged, consensus, inliers, inlier_off = verify_cpp.verify(
            inputs.ego_rows, inputs.cav_rows, moved["propose"][inputs.attempted], moved["statistic"][inputs.attempted],
            moved["dof"][inputs.attempted], model.variance_model, model.heading_lambda,
        )
        out[tau] = {"cache": moved, "inputs": inputs, "decided": inputs.scatter(decided), "engaged": np.asarray(engaged, bool),
                    "consensus": np.asarray(consensus, np.int64), "residual": _residuals(inputs, refined, inliers, inlier_off),
                    "inliers": np.asarray(inliers), "inlier_off": np.asarray(inlier_off)}
    return out


def choose_latency(grid: Dict[int, dict], cache, scenarios: List[str] = None, pool: int = 30,
                   min_history: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """(per-frame estimate, pooled estimate) of every cached pair's latency in frames.

    Per frame: the candidate with the largest Verify consensus among those where Verify engaged, ties broken by the
    smaller inlier residual; no engaged candidate -> 0. Pooled: causal median of the per-frame estimates of the same
    CAV over its last ``pool`` engaged frames (same scenario), once ``min_history`` exist; before that the per-frame one.
    """
    taus = sorted(grid)
    attempted = grid[taus[0]]["inputs"].attempted
    n = len(cache["frame"])
    per_frame, engaged_any = np.zeros(n, np.int64), np.zeros(n, bool)
    for i, p in enumerate(attempted):
        best = None
        for tau in taus:
            g = grid[tau]
            if not g["engaged"][i]:
                continue
            key = (int(g["consensus"][i]), -float(g["residual"][i]))
            if best is None or key > best[0]:
                best = (key, tau)
        if best is not None:
            per_frame[p], engaged_any[p] = best[1], True
    pooled = per_frame.copy()
    history: Dict[Tuple, List[int]] = {}
    records = grid[taus[0]]["inputs"].records
    order = np.argsort(cache["frame"], kind="stable")
    for p in order:
        frame = int(cache["frame"][p])
        cav_id = sorted(records[frame].cav_boxes)[int(cache["cav_rank"][p])]
        key = (scenarios[frame] if scenarios is not None else 0, cav_id)
        past = history.setdefault(key, [])
        if len(past) >= min_history:
            pooled[p] = int(round(float(np.median(past[-pool:]))))
        if engaged_any[p]:
            past.append(int(per_frame[p]))
    return per_frame, pooled


def assemble(grid: Dict[int, dict], cache, choice: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """The cache whose pair p carries the boxes moved by ``choice[p]`` and the decision made at that latency."""
    out = dict(cache)
    boxes = cache["cav_boxes"].copy()
    decided = np.zeros((len(cache["frame"]), 3), np.float32)
    for p in range(len(cache["frame"])):
        g = grid[int(choice[p])]
        rows = slice(cache["cav_off"][p], cache["cav_off"][p + 1])
        boxes[rows] = g["cache"]["cav_boxes"][rows]
        decided[p] = g["decided"][p]
    out["cav_boxes"] = boxes
    return out, decided


# ---------------------------------------------------------------------------------------------------------------
# A2: continuous joint fit, significance test against "no delay", information-weighted pooling over time
# ---------------------------------------------------------------------------------------------------------------

SIGNIFICANCE = 0.99


def _weighted_kabsch(src: np.ndarray, dst: np.ndarray, w: np.ndarray) -> Tuple[float, np.ndarray]:
    """psi, t of the weighted least-squares rigid map dst ~ R(psi) src + t."""
    w = w / max(float(w.sum()), 1e-12)
    ms, md = w @ src, w @ dst
    a, b = src - ms, dst - md
    h = (a * w[:, None]).T @ b
    psi = float(np.arctan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1]))
    c, s = np.cos(psi), np.sin(psi)
    return psi, md - np.array([c * ms[0] - s * ms[1], s * ms[0] + c * ms[1]])


def joint_fit(ego: np.ndarray, cav: np.ndarray, vel: np.ndarray, tau0: float, iterations: int = 20,
              huber_m: float = 0.5) -> dict:
    """Robust least squares for ego ~ R(psi) (cav + vel tau) + t over matched centres (Huber IRLS), alternating the
    rigid fit (given tau) and the closed-form tau (given the rigid fit). Also the pose-only fit (tau = 0) on the SAME
    pairs with the same final weights, for the significance test, and the information about tau that the pairs carry:
    the weighted energy of the velocities left after removing their rigid (translation) part, which is zero when
    every object moves alike (then tau is indistinguishable from a translation)."""
    w = np.ones(len(ego))
    tau = float(tau0)
    for _ in range(iterations):
        psi, t = _weighted_kabsch(cav + vel * tau, ego, w)
        c, s = np.cos(psi), np.sin(psi)
        rot = np.array([[c, -s], [s, c]])
        ru = vel @ rot.T
        base_residual = ego - cav @ rot.T - t
        den = float((w[:, None] * ru * ru).sum())
        if den > 1e-9:
            tau = float((w[:, None] * ru * base_residual).sum() / den)
        r = np.linalg.norm(ego - (cav + vel * tau) @ rot.T - t, axis=1)
        w = np.where(r <= huber_m, 1.0, huber_m / np.maximum(r, 1e-9))
    psi, t = _weighted_kabsch(cav + vel * tau, ego, w)
    c, s = np.cos(psi), np.sin(psi)
    rot = np.array([[c, -s], [s, c]])
    rss1 = float((w * np.sum((ego - (cav + vel * tau) @ rot.T - t) ** 2, axis=1)).sum())
    psi0, t0 = _weighted_kabsch(cav, ego, w)
    c0, s0 = np.cos(psi0), np.sin(psi0)
    rss0 = float((w * np.sum((ego - cav @ np.array([[c0, -s0], [s0, c0]]).T - t0) ** 2, axis=1)).sum())
    wn = w / max(float(w.sum()), 1e-12)
    centred = vel - wn @ vel
    info = float((w[:, None] * centred * centred).sum())
    return {"psi": psi, "t": t, "tau": tau, "rss1": rss1, "rss0": rss0, "info": info, "pairs": len(ego)}


def refine_latency(grid: Dict[int, dict], cache, scenarios: List[str] = None, pool: int = 30, min_history: int = 5,
                   max_tau: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per cached pair: (per-frame latency estimate, latency used, its standard error), continuous, in frames.

    Per frame, the joint fit starts from the grid candidate with the largest Verify consensus and uses that
    candidate's Verify inliers; it gives tau_f and its variance (residual variance / information about tau, the
    information being the energy of the inliers' velocities beyond a common translation: zero when everything moves
    alike, then tau is unobservable and the frame is skipped). Latency is a property of the link, so the estimates
    are POOLED FIRST, then TESTED: the causal precision-weighted median of the CAV's last ``pool`` estimates (negative
    ones included, so noise around zero cancels) is used only if it exceeds zero significantly at the 0.99 level
    (one-sided z on the pooled standard error); otherwise "no delay". Before ``min_history`` estimates exist, the
    frame's own estimate is tested the same way (F test of the joint fit against the pose-only fit on the same pairs).
    """
    from scipy.stats import f as f_dist, norm

    taus = sorted(grid)
    max_tau = float(max(taus)) if max_tau is None else max_tau
    first = grid[taus[0]]["inputs"]
    attempted, records = first.attempted, first.records
    n = len(cache["frame"])
    per_frame, variance, significant = np.zeros(n), np.full(n, np.inf), np.zeros(n, bool)
    for i, p in enumerate(attempted):
        best = None
        for tau in taus:
            g = grid[tau]
            if g["engaged"][i]:
                key = (int(g["consensus"][i]), -float(g["residual"][i]))
                if best is None or key > best[0]:
                    best = (key, tau)
        if best is None:
            continue
        g = grid[best[1]]
        pairs = g["inliers"][g["inlier_off"][i]:g["inlier_off"][i + 1]]
        if len(pairs) < 3:
            continue
        record = records[int(cache["frame"][p])]
        rows = slice(cache["cav_off"][p], cache["cav_off"][p + 1])
        ego = np.asarray(record.ego_boxes, np.float64)[pairs[:, 0], :2]
        cav = cache["cav_boxes"][rows][pairs[:, 1], :2].astype(np.float64)
        vel = cache["cav_velocity"][rows][pairs[:, 1]].astype(np.float64)
        fit = joint_fit(ego, cav, vel, best[1])
        dof = 2 * fit["pairs"] - 4
        if dof <= 0 or fit["info"] <= 1e-9 or fit["rss1"] <= 0.0:
            continue
        per_frame[p] = float(np.clip(fit["tau"], -max_tau, 2.0 * max_tau))
        variance[p] = (fit["rss1"] / dof) / fit["info"]
        statistic = (fit["rss0"] - fit["rss1"]) / (fit["rss1"] / dof)
        significant[p] = fit["tau"] > 0.0 and statistic > f_dist.ppf(SIGNIFICANCE, 1, dof)
    z = norm.ppf(SIGNIFICANCE)
    used, error = np.zeros(n), np.full(n, np.inf)
    history: Dict[Tuple, List[Tuple[float, float]]] = {}
    for p in np.argsort(cache["frame"], kind="stable"):
        frame = int(cache["frame"][p])
        cav_id = sorted(records[frame].cav_boxes)[int(cache["cav_rank"][p])]
        key = (scenarios[frame] if scenarios is not None else 0, cav_id)
        past = history.setdefault(key, [])
        if np.isfinite(variance[p]):
            past.append((float(per_frame[p]), float(variance[p])))
        recent = past[-pool:]
        if len(recent) >= min_history:
            values = np.array([v for v, _ in recent])
            weights = 1.0 / np.array([v for _, v in recent])
            order = np.argsort(values)
            cumulative = np.cumsum(weights[order])
            estimate = float(values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])
            se = float(np.sqrt(np.pi / 2.0 / weights.sum()))          # weighted median: sqrt(pi / 2) x the mean's SE
            error[p] = se
            used[p] = min(estimate, max_tau) if estimate > z * se else 0.0
        elif significant[p]:
            used[p], error[p] = min(per_frame[p], max_tau), float(np.sqrt(variance[p]))
    return per_frame, used, error


def decide_at(records, cache, latency_frames: np.ndarray, model, robust_config, base):
    """AlignFormer (Propose network + Verify + Abstain, unchanged) on the boxes moved by a per-pair latency."""
    from embedding_aware_belt_fusion.alignformer_ext import variants

    moved = dict(cache)
    boxes = cache["cav_boxes"].copy()
    for p in range(len(cache["frame"])):
        rows = slice(cache["cav_off"][p], cache["cav_off"][p + 1])
        boxes[rows, :2] += cache["cav_velocity"][rows] * float(latency_frames[p])
    moved["cav_boxes"] = boxes
    inputs = variants.PairInputs(records, moved, model, robust_config, base=base)
    _propose(inputs, model, robust_config)
    return moved, variants.alignformer(inputs), inputs


def absorb_residual_motion(inputs, decided: np.ndarray, tau_used: np.ndarray, tau_error: np.ndarray, table: np.ndarray,
                           k: float = 3.0) -> np.ndarray:
    """After the pose decision (temporal layer + GPS confidence), absorb what the latency estimate got wrong.

    GPS confidence governs the POSE: with an accurate GPS the decision stays on the reported pose. But when a delay
    was detected and the boxes were moved forward by v tau, the boxes still carry the error of that prediction:
    (tau_true - tau) v plus the sender's own prediction error, which no pose estimate contains. So, only for pairs
    whose latency is non-zero, Verify (the deployed C++ RANSAC) is run from the pose decision against the ego's
    CURRENT boxes, and its refit replaces the decision only if it moves every CAV box by less than k standard
    deviations of that expected motion error (latency standard error x speed, plus the sender's prediction-error
    variance at that horizon, fusion.prediction_error_table). Without a detected delay nothing changes."""
    from embedding_aware_belt_fusion.alignformer_ext import evaluate, fusion

    cache = inputs.cache
    a = inputs.attempted
    start = decided[a].astype(np.float32)
    refined, engaged = verify_cpp.verify(
        inputs.ego_rows, inputs.cav_rows, start, np.full(len(a), 1e6, np.float32), np.full(len(a), 3.0, np.float32),
        inputs.model.variance_model, inputs.model.heading_lambda,
    )[1:3]
    out = decided.copy()
    for i, p in enumerate(a):
        if tau_used[p] <= 0.0 or not engaged[i]:
            continue
        rows = slice(cache["cav_off"][p], cache["cav_off"][p + 1])
        speed = np.linalg.norm(cache["cav_velocity"][rows], axis=1)
        h = float(np.clip(tau_used[p], 0, table.shape[1] - 1))
        lo, frac = int(np.floor(h)), h - np.floor(h)
        hi = min(lo + 1, table.shape[1] - 1)
        bins = np.clip(np.searchsorted(fusion.SPEED_EDGES, speed, side="right") - 1, 0, table.shape[0] - 1)
        prediction_var = (1.0 - frac) * table[bins, lo] + frac * table[bins, hi]
        se = float(tau_error[p]) if np.isfinite(tau_error[p]) else float(tau_used[p])
        tolerance = k * float(np.sqrt(np.median(se ** 2 * speed ** 2 + 2.0 * prediction_var)))
        boxes = torch.from_numpy(cache["cav_boxes"][rows])
        before = evaluate.apply_correction(boxes, *map(float, decided[p]))[:, :2]
        after = evaluate.apply_correction(boxes, *map(float, refined[i]))[:, :2]
        if float((before - after).norm(dim=1).max()) <= tolerance:
            out[p] = refined[i]
    return out
