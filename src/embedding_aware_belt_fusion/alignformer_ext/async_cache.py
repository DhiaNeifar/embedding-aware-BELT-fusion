"""Asynchronous late fusion: every CAV message is ``delay`` frames old when the ego fuses it.

At frame t the ego has its own boxes of frame t; CAV k's message was produced at g = t - delay and carries its boxes
of frame g (its own LiDAR frame), its scores, its reported (noisy) pose of frame g (the same draws as the
synchronous cache at frame g), and a timestamp, so the delay is known. A receiver that ignores the delay places the
stale boxes with the stale pose into its current frame (``x1_to_x2(N_g, ego_t)``) and aligns them against its current
boxes: object motion over the delay is then read as pose error. The cache stores, per (frame t, CAV):

* ``cav_boxes``: the stale boxes placed naively (what every per-frame method, FreeAlign included, aligns);
* ``cav_scores``: their scores (frame g); ``oracle_boxes``: the stale boxes placed with the TRUE pose of frame g;
* AlignFormer's Propose on (ego boxes t, naive stale boxes) and FreeAlign's corrected boxes on the same pair;
* ``past_frame`` / ``past_rank``: the synchronous pair of frame g (ego boxes g <-> CAV boxes g), where a receiver that
  registers IN THE PAST, against its own boxes of the message's timestamp, finds the CAV's pose free of motion bias;
* ``ego_motion``: the ego's own motion from frame g to frame t (x, y, yaw), which moves that estimate to now.

With ``velocities`` (``cav_velocities``: each CAV tracks its own detections with its own odometry and sends one
velocity per box, as V2X sensor-sharing messages do), every stale box is first moved forward by velocity x delay in
the sender's frame ("compensated"); the oracle boxes get the same prediction.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from embedding_aware_belt_fusion.alignformer_ext.propose_cache import noisy_poses
from embedding_aware_belt_fusion.alignformer_pva.inference import build_pairwise_batch
from embedding_aware_belt_fusion.freealign.inference import correct_simple_detections as freealign_correct

GATE_M, MAX_MISS, VELOCITY_FRAMES = 2.0, 2, 3


def scenario_ids(geometry_dir: Path) -> List[str]:
    """Scenario id of every frame, in frame-table order (the geometry cache the table was built from)."""
    out = []
    for shard in sorted(glob.glob(str(Path(geometry_dir) / "*.pt"))):
        out += [frame["scenario_id"] for frame in torch.load(shard, weights_only=False)]
    return out


def _pose2d(matrix) -> np.ndarray:
    m = np.asarray(matrix, np.float64)
    return np.array([m[0, 3], m[1, 3], np.arctan2(m[1, 0], m[0, 0])])


def sender_odometry():
    """The odometry table every sender tracks with; refuses to run without one rather than fall back on true poses."""
    from embedding_aware_belt_fusion.alignformer_ext import odometry

    if odometry.ACTIVE is None:
        raise RuntimeError("sender tracking needs odometry.ACTIVE (pass --odometry); it never uses true poses")
    return odometry.ACTIVE


def cav_velocities(base, records, scenarios: List[str]) -> Dict[Tuple[int, str, int], np.ndarray]:
    """(frame, CAV id, box index) -> velocity (m per frame) in that CAV's own frame at that frame.

    Each CAV tracks its own detections (greedy constant-velocity association, 2 m gate) in a frame made stationary
    by its own odometry (odometry.ACTIVE: real LiDAR odometry on V2X-Real, the recorded motion on OPV2V; never the
    true pose) and reports the displacement over up to ``VELOCITY_FRAMES`` past frames; a box without history gets
    zero velocity.
    """
    table = sender_odometry()
    out: Dict[Tuple[int, str, int], np.ndarray] = {}
    agents = sorted({a for r in records for a in r.cav_boxes})
    for agent in agents:
        active, prev = [], None
        for f, record in enumerate(records):
            if scenarios[f] != prev:
                active, prev = [], scenarios[f]
            pose = table.pose2d(f, agent) if agent in record.cav_boxes else None
            if pose is None or len(record.cav_boxes[agent]) == 0:
                for track in active:
                    track["miss"] += 1
                active = [t for t in active if t["miss"] <= MAX_MISS]
                continue
            local = np.asarray(record.cav_boxes[agent], np.float64)[:, :2]
            c, s = np.cos(pose[2]), np.sin(pose[2])
            world = np.stack([pose[0] + c * local[:, 0] - s * local[:, 1], pose[1] + s * local[:, 0] + c * local[:, 1]], 1)
            predicted = []
            for track in active:
                frames = sorted(track["obs"])
                last = track["obs"][frames[-1]]
                v = (last - track["obs"][frames[-2]]) / (frames[-1] - frames[-2]) if len(frames) > 1 else np.zeros(2)
                predicted.append(last + v * (f - frames[-1]))
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
            for i, track in enumerate(active):
                if i not in used_t:
                    track["miss"] += 1
            for j in range(len(world)):
                if j not in used_d:
                    active.append({"obs": {f: world[j]}, "miss": 0})
                    used_d.add(j)
            # velocity of every box of this frame, from its track's past (causal)
            index_of = {}
            for track in active:
                if f in track["obs"]:
                    j = int(np.argmin(np.linalg.norm(world - track["obs"][f], axis=1)))
                    index_of[j] = track
            for j in range(len(world)):
                track = index_of.get(j)
                v_world = np.zeros(2)
                if track is not None:
                    past = [h for h in sorted(track["obs"]) if f - VELOCITY_FRAMES <= h < f]
                    if past:
                        v_world = (track["obs"][f] - track["obs"][past[0]]) / (f - past[0])
                out[(f, agent, j)] = np.array([c * v_world[0] + s * v_world[1], -s * v_world[0] + c * v_world[1]])
            active = [t for t in active if t["miss"] <= MAX_MISS]
    return out


def _compensated(boxes: torch.Tensor, frame: int, agent: str, delay: int, velocities) -> torch.Tensor:
    out = boxes.clone()
    for j in range(len(out)):
        v = velocities.get((frame, agent, j))
        if v is not None:
            out[j, 0] += float(v[0] * delay)
            out[j, 1] += float(v[1] * delay)
    return out


def build_async_condition(base, records, model, robust_config, sigma: float, seed: int, delay: int,
                          scenarios: List[str], velocities=None, freealign_config=None,
                          compensate: bool = False) -> Dict[str, np.ndarray]:
    """Per (frame t, CAV present at t and at t - delay): the stale message, placed naively, with Propose and
    FreeAlign on it, the oracle placement, and the link to the synchronous pair of frame t - delay. With
    ``velocities``, each box's sender velocity is stored too (``cav_velocity``: m per frame, rotated into the ego
    frame by the naive placement); ``compensate`` also moves every stale box forward by velocity x delay."""
    frame, cav_rank, attempted, past_frame, past_rank, ego_motion = [], [], [], [], [], []
    cav_boxes, cav_scores, oracle_boxes, freealign_boxes, counts, cav_velocity = [], [], [], [], [], []
    propose, statistic, dof, plans, plan_shape = [], [], [], [], []
    with torch.no_grad():
        for t, record in enumerate(records):
            g = t - delay
            if g < 0 or scenarios[g] != scenarios[t]:
                continue
            old = records[g]
            reported = noisy_poses(base, old, g, sigma, seed)
            ids_t, ids_g = sorted(record.cav_boxes), sorted(old.cav_boxes)
            motion = _pose2d(base.x1_to_x2(old.ego_pose, record.ego_pose))
            for cav_id in ids_g:
                if cav_id not in reported or cav_id not in record.cav_boxes or cav_id not in old.cav_poses:
                    continue
                local = old.cav_boxes[cav_id]
                if compensate:
                    local = _compensated(local, g, cav_id, delay, velocities)
                placement = base.x1_to_x2(reported[cav_id], record.ego_pose)
                naive = base.transform_boxes_to_ego(local, torch.as_tensor(placement, dtype=torch.float32))
                if velocities is not None:
                    yaw = float(np.arctan2(placement[1, 0], placement[0, 0]))
                    c_, s_ = np.cos(yaw), np.sin(yaw)
                    v = np.array([velocities.get((g, cav_id, j), np.zeros(2)) for j in range(len(local))]).reshape(-1, 2)
                    cav_velocity.append(np.stack([c_ * v[:, 0] - s_ * v[:, 1], s_ * v[:, 0] + c_ * v[:, 1]], 1).astype(np.float32))
                oracle = base.transform_boxes_to_ego(local, torch.as_tensor(base.x1_to_x2(old.cav_poses[cav_id], record.ego_pose), dtype=torch.float32))
                scores = old.cav_scores[cav_id]
                frame.append(t); cav_rank.append(ids_t.index(cav_id))
                past_frame.append(g); past_rank.append(ids_g.index(cav_id)); ego_motion.append(motion)
                cav_boxes.append(naive.numpy().astype(np.float32)); oracle_boxes.append(oracle.numpy().astype(np.float32))
                cav_scores.append(np.asarray(scores, np.float32)); counts.append(int(naive.shape[0]))
                if freealign_config is not None:
                    simple = {"ego": {"boxes": record.ego_boxes, "scores": record.ego_scores},
                              cav_id: {"boxes": naive, "scores": scores}}
                    corrected, _ = freealign_correct(simple, ["ego", cav_id], freealign_config)
                    freealign_boxes.append(corrected[cav_id]["boxes"].numpy().astype(np.float32))
                ok = record.ego_boxes.shape[0] > 0 and naive.shape[0] > 0
                attempted.append(ok)
                if not ok:
                    propose.append((0.0, 0.0, 0.0)); statistic.append(0.0); dof.append(1.0); plan_shape.append((0, 0))
                    continue
                batch = build_pairwise_batch({"boxes": record.ego_boxes, "scores": record.ego_scores},
                                             {"boxes": naive, "scores": scores})
                estimate = model(batch, robust=robust_config, statistic=True)
                propose.append((float(estimate.psi[0]), float(estimate.t[0, 0]), float(estimate.t[0, 1])))
                statistic.append(float(estimate.offset_statistic[0])); dof.append(float(estimate.offset_dof[0]))
                plan = estimate.log_assignment[0].numpy().astype(np.float32)
                plans.append(plan.reshape(-1)); plan_shape.append(plan.shape)
    out = {
        "sigma": np.float64(sigma), "seed": np.int64(seed), "delay": np.int64(delay),
        "compensated": np.bool_(compensate),
        "frame": np.asarray(frame, np.int64), "cav_rank": np.asarray(cav_rank, np.int64),
        "attempted": np.asarray(attempted, bool),
        "past_frame": np.asarray(past_frame, np.int64), "past_rank": np.asarray(past_rank, np.int64),
        "ego_motion": np.asarray(ego_motion, np.float64).reshape(-1, 3),
        "cav_boxes": np.concatenate(cav_boxes) if cav_boxes else np.zeros((0, 7), np.float32),
        "cav_scores": np.concatenate(cav_scores) if cav_scores else np.zeros(0, np.float32),
        "oracle_boxes": np.concatenate(oracle_boxes) if oracle_boxes else np.zeros((0, 7), np.float32),
        "cav_off": np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        "propose": np.asarray(propose, np.float32).reshape(-1, 3),
        "statistic": np.asarray(statistic, np.float32), "dof": np.asarray(dof, np.float32),
        "plan": np.concatenate(plans) if plans else np.zeros(0, np.float32),
        "plan_shape": np.asarray(plan_shape, np.int64).reshape(-1, 2),
    }
    if freealign_config is not None:
        out["freealign_boxes"] = np.concatenate(freealign_boxes) if freealign_boxes else np.zeros((0, 7), np.float32)
    if velocities is not None:
        out["cav_velocity"] = np.concatenate(cav_velocity) if cav_velocity else np.zeros((0, 2), np.float32)
    return out


def condition_path(cache_dir: Path, sigma: float, seed: int, delay: int, compensated: bool) -> Path:
    return Path(cache_dir) / f"async_d{delay}{'_comp' if compensated else ''}_sigma_{sigma:g}_seed_{seed}.npz"


def from_the_past(decided_sync: np.ndarray, sync_index: Dict[Tuple[int, int], int], cache) -> np.ndarray:
    """(P, 3) psi, tx, ty for the async pairs: the decision a receiver makes at the message's timestamp (pair of
    frame g in the synchronous cache, ego boxes g <-> CAV boxes g) moved to now by the ego's own motion E:
    D_t = E o D_g o E^-1, so that D_t o (E o N_g) = E o (D_g o N_g)."""
    from embedding_aware_belt_fusion.alignformer_ext.se2 import compose, inverse

    out = np.zeros((len(cache["frame"]), 3), np.float32)
    for p in range(len(cache["frame"])):
        q = sync_index.get((int(cache["past_frame"][p]), int(cache["past_rank"][p])))
        if q is None:
            continue
        d = decided_sync[q]
        if not np.any(d != 0):
            continue
        e = cache["ego_motion"][p]
        moved = compose(compose(e, np.array([d[1], d[2], d[0]], np.float64)), inverse(e))
        out[p] = (moved[2], moved[0], moved[1])
    return out
