"""Staleness-aware scores for delayed messages in late fusion.

Each CAV box's score is scaled by the probability that its predicted position stays within the IoU-0.7 tolerance,
from the prediction errors the sender measures on its own tracks (``prediction_error_table``).
"""

from __future__ import annotations

import numpy as np


SPEED_EDGES = (0.0, 0.1, 0.5, 1.0, np.inf)       # m per frame: <1, 1-5, 5-10, >10 m/s
IOU_SHIFT = (1.0 - 0.7) / (1.0 + 0.7)             # lateral shift (fraction of box width) that loses IoU 0.7


def prediction_error_table(base, records, scenarios, horizons=range(0, 7), causal: bool = False) -> np.ndarray:
    """(speed bins, horizons) per-axis variance (m^2) of constant-velocity prediction from a vehicle's OWN tracks.

    Every CAV tracks its detections (async_cache's tracker, its own odometry: odometry.ACTIVE) and compares each track's
    position ``h`` frames later with the prediction p + v h made from the 3-frame velocity it would have sent:
    no ground truth, only its own sensor history. Horizon 0 is the detector's own noise (frame-to-frame jitter
    of static tracks is not separated out; the 0-horizon entry uses the residual of a 1-frame-smoothed position).

    ``causal=True`` (audit X1) returns one table per frame, (frames, speed bins, horizons): frame F's table uses only
    the prediction errors already observed before F (an error comparing frames f and f + h is known at f + h), so no
    frame is scored with a table fitted on its own or later frames. A bin with no error yet has variance 0, which
    leaves those boxes' scores unscaled.
    """
    from embedding_aware_belt_fusion.alignformer_ext.async_cache import GATE_M, MAX_MISS, VELOCITY_FRAMES, sender_odometry

    odometry_table = sender_odometry()

    horizons = list(horizons)
    sums = np.zeros((len(SPEED_EDGES) - 1, len(horizons)))
    counts = np.zeros_like(sums)
    if causal:                     # contributions by the frame at which they become known
        sums = np.zeros((len(records) + 1,) + sums.shape)
        counts = np.zeros_like(sums)

    def add(known_at, b, k, value, count):
        if causal:
            known_at = min(known_at, len(records))     # known only after the last frame: never used
            sums[known_at, b, k] += value
            counts[known_at, b, k] += count
        else:
            sums[b, k] += value
            counts[b, k] += count
    agents = sorted({a for r in records for a in r.cav_boxes})
    for agent in agents:
        tracks, active, prev = [], [], None
        for f, record in enumerate(records):
            if scenarios[f] != prev:
                active, prev = [], scenarios[f]
            pose = odometry_table.pose2d(f, agent) if agent in record.cav_boxes else None
            if pose is None or len(record.cav_boxes[agent]) == 0:
                for t in active:
                    t["miss"] += 1
                active = [t for t in active if t["miss"] <= MAX_MISS]
                continue
            local = np.asarray(record.cav_boxes[agent], np.float64)[:, :2]
            c, s = np.cos(pose[2]), np.sin(pose[2])
            world = np.stack([pose[0] + c * local[:, 0] - s * local[:, 1], pose[1] + s * local[:, 0] + c * local[:, 1]], 1)
            predicted = []
            for t in active:
                fr = sorted(t["obs"])
                last = t["obs"][fr[-1]]
                v = (last - t["obs"][fr[-2]]) / (fr[-1] - fr[-2]) if len(fr) > 1 else np.zeros(2)
                predicted.append(last + v * (f - fr[-1]))
            used_t, used_d = set(), set()
            if predicted:
                d = np.linalg.norm(np.array(predicted)[:, None] - world[None], axis=2)
                for i, j in sorted(np.ndindex(*d.shape), key=lambda ij: d[ij]):
                    if d[i, j] > GATE_M:
                        break
                    if i in used_t or j in used_d:
                        continue
                    active[i]["obs"][f] = world[j]
                    active[i]["miss"] = 0
                    used_t.add(i)
                    used_d.add(j)
            for i, t in enumerate(active):
                if i not in used_t:
                    t["miss"] += 1
            for j in range(len(world)):
                if j not in used_d:
                    t = {"obs": {f: world[j]}, "miss": 0}
                    active.append(t)
                    tracks.append(t)
            active = [t for t in active if t["miss"] <= MAX_MISS]
        for t in tracks:
            obs = t["obs"]
            for f in obs:
                past = [h for h in sorted(obs) if f - VELOCITY_FRAMES <= h < f]
                if not past:
                    continue
                v = (obs[f] - obs[past[0]]) / (f - past[0])
                b = int(np.searchsorted(SPEED_EDGES, np.linalg.norm(v), side="right") - 1)
                for k, h in enumerate(horizons):
                    if h == 0:
                        if f - 1 in obs and f + 1 in obs:
                            e = obs[f] - 0.5 * (obs[f - 1] + obs[f + 1])
                            add(f + 1, b, k, float(e @ e) * (2.0 / 3.0), 2.0)   # jitter of one detection, 3-point residual
                    elif f + h in obs:
                        e = obs[f + h] - (obs[f] + v * h)
                        add(f + h, b, k, float(e @ e), 2.0)
    if causal:                     # frame F sees everything known at frames < F
        sums = np.cumsum(sums, axis=0)[:-1]
        counts = np.cumsum(counts, axis=0)[:-1]
        sums = np.concatenate([np.zeros_like(sums[:1]), sums[:-1]]) if len(sums) else sums
        counts = np.concatenate([np.zeros_like(counts[:1]), counts[:-1]]) if len(counts) else counts
    return sums / np.maximum(counts, 1.0)


def staleness_confidence(cache, table: np.ndarray, delays) -> np.ndarray:
    """Per cached CAV box: P(its predicted position is within the IoU-0.7 tolerance) given its speed and the delay
    (``delays``: one latency in frames per cached pair) from the sender's own prediction-error table."""
    out = np.ones(len(cache["cav_boxes"]), np.float32)
    speed = np.linalg.norm(cache["cav_velocity"], axis=1)
    n_bins = table.shape[-2]
    bins = np.clip(np.searchsorted(SPEED_EDGES, speed, side="right") - 1, 0, n_bins - 1)
    for p in range(len(cache["frame"])):
        rows = slice(cache["cav_off"][p], cache["cav_off"][p + 1])
        frame_table = table[int(cache["frame"][p])] if table.ndim == 3 else table   # a causal table: the pair's frame
        h = float(np.clip(delays[p], 0, frame_table.shape[1] - 1))  # continuous latency: linear between horizons
        lo, frac = int(np.floor(h)), h - np.floor(h)
        hi = min(lo + 1, frame_table.shape[1] - 1)
        var = (1.0 - frac) * frame_table[bins[rows], lo] + frac * frame_table[bins[rows], hi]
        b = cache["cav_boxes"][rows]
        tol = IOU_SHIFT * np.minimum(b[:, 4], b[:, 5])
        out[rows] = 1.0 - np.exp(-tol ** 2 / (2.0 * np.maximum(var, 1e-12)))
    return out
