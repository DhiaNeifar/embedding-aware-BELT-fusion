#!/usr/bin/env python3
"""Prevalence gate for symmetry-induced multi-modal cooperative pose."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_verify_core  # noqa: E402
import reproduce_opv2v_sweep as base  # noqa: E402
from test_structured_pose_marginalization import kabsch, sampled_matchings, se2_matrix  # noqa: E402


CONFIG = {
    "opv2v": dict(table="outputs/opv2v_frame_table.pt", position="outputs/propose_cache_opv2v",
                  delay="outputs/async/opv2v"),
    "v2xreal": dict(table="outputs/v2xreal_test_frame_table.pt", position="outputs/propose_cache_v2xreal",
                    delay="outputs/async/v2xreal"),
}
_RECORDS = None


def unpack_plans(cache):
    pointer, output = 0, []
    for rows, columns in cache["plan_shape"]:
        size = int(rows * columns)
        output.append(cache["plan"][pointer:pointer + size].reshape(int(rows), int(columns)) if size else None)
        pointer += size
    return output


def angle_distance(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def pose_modes(log_plan, ego, cav, rng, hypotheses):
    matchings, utility = sampled_matchings(log_plan, hypotheses, rng)
    modes, scores = [], []
    for assignment in matchings:
        pairs = [(r, c) for r, c in enumerate(assignment) if c >= 0]
        if len(pairs) < 2:
            continue
        ri = np.asarray([p[0] for p in pairs]); ci = np.asarray([p[1] for p in pairs])
        solved = kabsch(ego[ri], cav[ci])
        if solved is None:
            continue
        theta, residual = solved
        assignment_score = sum(utility[r, c if c >= 0 else cav.shape[0] + r]
                               for r, c in enumerate(assignment))
        geometric_score = -0.5 * len(pairs) * residual / (0.75 ** 2)
        prior_score = -0.5 * ((theta[1] ** 2 + theta[2] ** 2) / 3.0 ** 2
                              + theta[0] ** 2 / math.radians(20) ** 2)
        modes.append(theta); scores.append(assignment_score + geometric_score + prior_score)
    if not modes:
        return None
    modes, scores = np.asarray(modes), np.asarray(scores)
    probability = np.exp(scores - scores.max()); probability /= probability.sum()
    order = np.argsort(-probability)
    clusters = []
    for index in order:
        theta, weight = modes[index], probability[index]
        assigned = False
        for cluster in clusters:
            centre = cluster["centre"]
            if np.linalg.norm(theta[1:] - centre[1:]) < 0.5 and angle_distance(theta[0], centre[0]) < math.radians(3):
                cluster["mass"] += float(weight)
                assigned = True
                break
        if not assigned:
            clusters.append({"centre": theta, "mass": float(weight)})
    major = [cluster for cluster in clusters if cluster["mass"] >= 0.1]
    separation = 0.0
    for first in range(len(major)):
        for second in range(first + 1, len(major)):
            translation = np.linalg.norm(major[first]["centre"][1:] - major[second]["centre"][1:])
            yaw_equivalent = angle_distance(major[first]["centre"][0], major[second]["centre"][0]) / math.radians(3) * 0.5
            separation = max(separation, float(max(translation, yaw_equivalent)))
    entropy = -sum(cluster["mass"] * math.log(max(cluster["mass"], 1e-12)) for cluster in clusters)
    return len(major), separation, entropy


def true_correction_position(record, cav_id, frame, rank, sigma, seed):
    if sigma:
        noisy_pose = base.perturb_pose(
            record.cav_poses[cav_id], sigma, sigma * base.YAW_STD_PER_XY_STD,
            base.sweep_rng(seed, sigma, frame, rank),
        )
    else:
        noisy_pose = record.cav_poses[cav_id]
    reported = base.x1_to_x2(noisy_pose, record.ego_pose)
    truth = base.x1_to_x2(record.cav_poses[cav_id], record.ego_pose)
    return truth @ np.linalg.inv(reported)


def error(theta, truth):
    residual = np.linalg.inv(se2_matrix(theta)) @ truth
    return float(np.linalg.norm(residual[:2, 3]))


def analyze_path(job):
    path, kind, hypotheses = job
    with np.load(path) as z:
        cache = {key: z[key] for key in z.files}
    plans = unpack_plans(cache)
    rows = []
    sigma, seed = float(cache.get("sigma", 0.0)), int(cache.get("seed", 0))
    for index, (frame, rank, plan) in enumerate(zip(cache["frame"], cache["cav_rank"], plans)):
        if plan is None:
            continue
        record = _RECORDS[int(frame)]
        cav_id = sorted(record.cav_boxes)[int(rank)]
        start, end = cache["cav_off"][index:index + 2]
        cav = cache["cav_boxes"][start:end, :2].astype(np.float64)
        ego = record.ego_boxes[:, :2].numpy().astype(np.float64)
        result = pose_modes(plan.astype(np.float64), ego, cav,
                            np.random.default_rng(seed * 1000003 + index), hypotheses)
        if result is None:
            continue
        if kind == "position":
            truth = true_correction_position(record, cav_id, int(frame), int(rank), sigma, seed)
        elif "oracle_boxes" in cache:
            oracle = cache["oracle_boxes"][start:end, :2].astype(np.float64)
            solved = kabsch(oracle, cav)
            truth = se2_matrix(solved[0]) if solved is not None else np.eye(4)
        else:
            truth = np.eye(4)
        propose_error = error(cache["propose"][index], truth)
        bare_error = error(np.zeros(3), truth)
        major, separation, entropy = result
        rows.append((int(frame), int(rank), major >= 2, major, separation, entropy,
                     propose_error, bare_error, propose_error > bare_error + 0.05))
    ambiguous = {(row[0], row[1]) for row in rows if row[2]}
    next_persistent = [((frame + 1, rank) in ambiguous) for frame, rank in ambiguous]
    array = np.asarray([[r[2], r[3], r[4], r[5], r[6], r[7], r[8]] for r in rows], dtype=float)
    multi = array[:, 0].astype(bool)
    summary = {
        "pairs": len(rows), "multimodal_fraction": float(multi.mean()),
        "persistence_next_frame": float(np.mean(next_persistent)) if next_persistent else 0.0,
        "mean_major_modes_multimodal": float(array[multi, 1].mean()) if multi.any() else 0.0,
        "median_mode_separation": float(np.median(array[multi, 2])) if multi.any() else 0.0,
        "median_propose_error_unimodal": float(np.median(array[~multi, 4])) if (~multi).any() else None,
        "median_propose_error_multimodal": float(np.median(array[multi, 4])) if multi.any() else None,
        "failure_rate_unimodal": float(np.mean(array[~multi, 4] > 0.5)) if (~multi).any() else None,
        "failure_rate_multimodal": float(np.mean(array[multi, 4] > 0.5)) if multi.any() else None,
        "harm_rate_unimodal": float(array[~multi, 6].mean()) if (~multi).any() else None,
        "harm_rate_multimodal": float(array[multi, 6].mean()) if multi.any() else None,
    }
    return path.stem, summary


def paths_for(cfg, kind):
    root = Path(cfg[kind])
    if kind == "position":
        return [path for path in sorted(root.glob("sigma_*_seed_0.npz")) if not path.name.startswith("cavcav_")]
    return [path for path in sorted(root.glob("async_d*_sigma_*_seed_0.npz")) if "_comp_" not in path.name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(CONFIG), default=sorted(CONFIG))
    parser.add_argument("--hypotheses", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("outputs/diagnostics/symmetry_modes_seed0.json"))
    args = parser.parse_args()
    output = {"hypotheses": args.hypotheses, "datasets": {}}
    global _RECORDS
    for dataset in args.datasets:
        cfg = CONFIG[dataset]
        _RECORDS = check_verify_core.load_frame_table(Path(cfg["table"]))
        output["datasets"][dataset] = {}
        for kind in ("position", "delay"):
            jobs = [(path, kind, args.hypotheses) for path in paths_for(cfg, kind)]
            summaries = {}
            with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
                for name, summary in pool.map(analyze_path, jobs):
                    summaries[name] = summary
                    print(dataset, kind, name, summary)
            output["datasets"][dataset][kind] = summaries
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
