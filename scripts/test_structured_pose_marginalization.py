#!/usr/bin/env python3
"""One-seed go/no-go test for structured correspondence pose marginalization."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_verify_core  # noqa: E402
import reproduce_opv2v_sweep as base  # noqa: E402


CONFIG = {
    "opv2v": ("outputs/opv2v_frame_table.pt", "outputs/propose_cache_opv2v"),
    "v2xreal": ("outputs/v2xreal_test_frame_table.pt", "outputs/propose_cache_v2xreal"),
}
YAW_SCALE = math.radians(10.0)


def unpack_plans(cache):
    pointer, output = 0, []
    for rows, columns in cache["plan_shape"]:
        size = int(rows * columns)
        output.append(cache["plan"][pointer:pointer + size].reshape(int(rows), int(columns)) if size else None)
        pointer += size
    return output


def se2_matrix(theta):
    yaw, tx, ty = theta
    c, s = math.cos(yaw), math.sin(yaw)
    matrix = np.eye(4)
    matrix[:2, :2] = [[c, -s], [s, c]]
    matrix[:2, 3] = [tx, ty]
    return matrix


def theta_of(matrix):
    return np.array([math.atan2(matrix[1, 0], matrix[0, 0]), matrix[0, 3], matrix[1, 3]])


def pose_error(theta, truth):
    residual = np.linalg.inv(se2_matrix(theta)) @ truth
    return float(np.linalg.norm(residual[:2, 3])), abs(math.degrees(math.atan2(residual[1, 0], residual[0, 0])))


def kabsch(target, source):
    if len(target) < 2:
        return None
    pbar, qbar = target.mean(0), source.mean(0)
    covariance = (source - qbar).T @ (target - pbar)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = pbar - rotation @ qbar
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    transformed = source @ rotation.T + translation
    residual = float(np.mean(np.sum((target - transformed) ** 2, axis=1)))
    return np.array([yaw, translation[0], translation[1]]), residual


def sampled_matchings(log_plan, count, rng):
    rows, columns = log_plan.shape[0] - 1, log_plan.shape[1] - 1
    utility = np.full((rows, columns + rows), -1e6, dtype=np.float64)
    utility[:, :columns] = log_plan[:-1, :-1]
    utility[np.arange(rows), columns + np.arange(rows)] = log_plan[:-1, -1]
    output, seen = [], set()
    for sample in range(count * 4):
        perturbed = utility.copy()
        if sample:
            uniform = np.clip(rng.random(perturbed.shape), 1e-9, 1 - 1e-9)
            perturbed += 0.65 * (-np.log(-np.log(uniform)))
            perturbed[utility < -1e5] = -1e6
        row, column = linear_sum_assignment(-perturbed)
        assignment = tuple(int(c) if c < columns else -1 for c in column[np.argsort(row)])
        if assignment not in seen:
            seen.add(assignment)
            output.append(assignment)
        if len(output) >= count:
            break
    return output, utility


def structured_estimate(log_plan, ego, cav, rng, hypotheses=16):
    matchings, utility = sampled_matchings(log_plan, hypotheses, rng)
    modes, scores = [], []
    for assignment in matchings:
        pairs = [(row, column) for row, column in enumerate(assignment) if column >= 0]
        if len(pairs) < 2:
            continue
        ego_index = np.array([pair[0] for pair in pairs])
        cav_index = np.array([pair[1] for pair in pairs])
        solved = kabsch(ego[ego_index], cav[cav_index])
        if solved is None:
            continue
        theta, residual = solved
        assignment_score = sum(utility[row, column if column >= 0 else cav.shape[0] + row]
                               for row, column in enumerate(assignment))
        # Fixed likelihood scales, not fitted on the evaluated data.
        geometric_score = -0.5 * len(pairs) * residual / (0.75 ** 2)
        prior_score = -0.5 * ((theta[1] ** 2 + theta[2] ** 2) / 3.0 ** 2 + theta[0] ** 2 / math.radians(20) ** 2)
        modes.append(theta)
        scores.append(assignment_score + geometric_score + prior_score)
    if not modes:
        return None
    modes, scores = np.asarray(modes), np.asarray(scores)
    probability = np.exp(scores - scores.max())
    probability /= probability.sum()
    mean = np.array([
        math.atan2(np.sum(probability * np.sin(modes[:, 0])), np.sum(probability * np.cos(modes[:, 0]))),
        np.sum(probability * modes[:, 1]), np.sum(probability * modes[:, 2]),
    ])
    scaled_modes = np.column_stack([modes[:, 1], modes[:, 2], modes[:, 0] / YAW_SCALE])
    scaled_mean = np.array([mean[1], mean[2], mean[0] / YAW_SCALE])
    risk_mean = float(np.sum(probability * np.sum((scaled_modes - scaled_mean) ** 2, axis=1)))
    risk_identity = float(np.sum(probability * np.sum(scaled_modes ** 2, axis=1)))
    accepted = risk_mean < 0.9 * risk_identity
    return {
        "mean": mean if accepted else np.zeros(3),
        "map": modes[int(np.argmax(probability))],
        "hungarian": modes[0],
        "accepted": accepted,
        "modes": len(modes),
        "effective_modes": float(1.0 / np.sum(probability ** 2)),
    }


def evaluate(dataset, sigma, seed, hypotheses):
    table_path, cache_dir = CONFIG[dataset]
    records = check_verify_core.load_frame_table(Path(table_path))
    with np.load(Path(cache_dir) / f"sigma_{sigma:g}_seed_{seed}.npz") as z:
        cache = {key: z[key] for key in z.files}
    plans = unpack_plans(cache)
    values = {name: {"t": [], "yaw": [], "harm": [], "better": [], "answered": []}
              for name in ["bare", "propose", "hungarian", "structured_map", "structured_marginal"]}
    effective_modes = []
    for index, (frame_index, rank, plan) in enumerate(zip(cache["frame"], cache["cav_rank"], plans)):
        if plan is None:
            continue
        record = records[int(frame_index)]
        cav_id = sorted(record.cav_boxes)[int(rank)]
        reported = base.x1_to_x2(
            base.perturb_pose(record.cav_poses[cav_id], sigma, sigma * base.YAW_STD_PER_XY_STD,
                              base.sweep_rng(seed, sigma, int(frame_index), int(rank))),
            record.ego_pose,
        ) if sigma else base.x1_to_x2(record.cav_poses[cav_id], record.ego_pose)
        truth_transform = base.x1_to_x2(record.cav_poses[cav_id], record.ego_pose)
        truth_correction = truth_transform @ np.linalg.inv(reported)
        start, end = cache["cav_off"][index:index + 2]
        cav = cache["cav_boxes"][start:end, :2].astype(np.float64)
        ego = record.ego_boxes[:, :2].numpy().astype(np.float64)
        result = structured_estimate(plan.astype(np.float64), ego, cav,
                                     np.random.default_rng(seed * 1000003 + index), hypotheses)
        if result is None:
            continue
        estimates = {
            "bare": np.zeros(3),
            "propose": cache["propose"][index].astype(np.float64),
            "hungarian": result["hungarian"],
            "structured_map": result["map"],
            "structured_marginal": result["mean"],
        }
        bare_t, _ = pose_error(estimates["bare"], truth_correction)
        for name, estimate in estimates.items():
            t, yaw = pose_error(estimate, truth_correction)
            values[name]["t"].append(t)
            values[name]["yaw"].append(yaw)
            values[name]["harm"].append(t > bare_t + 0.05)
            values[name]["better"].append(t + 0.05 < bare_t)
            values[name]["answered"].append(bool(np.any(np.abs(estimate) > 1e-9)))
        effective_modes.append(result["effective_modes"])
    summary = {}
    for name, row in values.items():
        t, yaw = np.asarray(row["t"]), np.asarray(row["yaw"])
        summary[name] = {
            "pairs": len(t), "median_m": float(np.median(t)), "p90_m": float(np.quantile(t, .9)),
            "median_yaw_deg": float(np.median(yaw)), "harm_rate": float(np.mean(row["harm"])),
            "benefit_rate": float(np.mean(row["better"])), "coverage": float(np.mean(row["answered"])),
        }
    return {"summary": summary, "mean_effective_modes": float(np.mean(effective_modes))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(CONFIG), default=sorted(CONFIG))
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hypotheses", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("outputs/diagnostics/structured_pose_one_seed.json"))
    args = parser.parse_args()
    result = {"sigma": args.sigma, "seed": args.seed, "hypotheses": args.hypotheses, "datasets": {}}
    for dataset in args.datasets:
        result["datasets"][dataset] = evaluate(dataset, args.sigma, args.seed, args.hypotheses)
        print(dataset)
        for name, row in result["datasets"][dataset]["summary"].items():
            print(f"  {name:22s} med {row['median_m']:.3f} p90 {row['p90_m']:.3f} "
                  f"harm {100*row['harm_rate']:.2f}% benefit {100*row['benefit_rate']:.1f}% cov {100*row['coverage']:.1f}%")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
