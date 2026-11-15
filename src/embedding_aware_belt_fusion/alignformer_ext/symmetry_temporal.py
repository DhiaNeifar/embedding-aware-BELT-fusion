"""Symmetry-conditioned projection of the full temporal AlignFormer estimate."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def _angle(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def _kabsch(target, source):
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
    transformed = source @ rotation.T + translation
    residual = float(np.mean(np.sum((target - transformed) ** 2, axis=1)))
    return np.array([math.atan2(rotation[1, 0], rotation[0, 0]), *translation]), residual


def _matchings(log_plan, count, rng):
    rows, columns = log_plan.shape[0] - 1, log_plan.shape[1] - 1
    utility = np.full((rows, columns + rows), -1e6)
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
            output.append(assignment); seen.add(assignment)
        if len(output) >= count:
            break
    return output, utility


def _classes(log_plan, ego, cav, rng, count):
    assignments, utility = _matchings(log_plan, count, rng)
    modes, scores = [], []
    for assignment in assignments:
        pairs = [(r, c) for r, c in enumerate(assignment) if c >= 0]
        if len(pairs) < 2:
            continue
        ri = np.asarray([p[0] for p in pairs]); ci = np.asarray([p[1] for p in pairs])
        solved = _kabsch(ego[ri], cav[ci])
        if solved is None:
            continue
        theta, residual = solved
        assignment_score = sum(utility[r, c if c >= 0 else cav.shape[0] + r]
                               for r, c in enumerate(assignment))
        score = assignment_score - 0.5 * len(pairs) * residual / 0.75 ** 2
        score -= 0.5 * ((theta[1] ** 2 + theta[2] ** 2) / 3.0 ** 2
                        + theta[0] ** 2 / math.radians(20) ** 2)
        modes.append(theta); scores.append(score)
    if not modes:
        return []
    modes, scores = np.asarray(modes), np.asarray(scores)
    probability = np.exp(scores - scores.max()); probability /= probability.sum()
    clusters = []
    for index in np.argsort(-probability):
        theta, weight = modes[index], float(probability[index])
        if weight <= 1e-15:
            continue
        destination = None
        for cluster in clusters:
            if np.linalg.norm(theta[1:] - cluster["centre"][1:]) < 0.5 and _angle(theta[0], cluster["centre"][0]) < math.radians(3):
                destination = cluster; break
        if destination is None:
            clusters.append({"members": [theta], "weights": [weight], "centre": theta.copy(), "mass": weight})
        else:
            destination["members"].append(theta); destination["weights"].append(weight); destination["mass"] += weight
    for cluster in clusters:
        members, weights = np.asarray(cluster["members"]), np.asarray(cluster["weights"])
        weights /= max(weights.sum(), 1e-15)
        cluster["centre"] = np.array([
            math.atan2(np.sum(weights * np.sin(members[:, 0])), np.sum(weights * np.cos(members[:, 0]))),
            np.sum(weights * members[:, 1]), np.sum(weights * members[:, 2]),
        ])
    return [cluster for cluster in clusters if cluster["mass"] >= 0.1]


def _plans(cache):
    pointer, output = 0, []
    for rows, columns in cache["plan_shape"]:
        size = int(rows * columns)
        output.append(cache["plan"][pointer:pointer + size].reshape(int(rows), int(columns)) if size else None)
        pointer += size
    return output


def decide(inputs, temporal_decided: np.ndarray, *, hypotheses: int = 16) -> np.ndarray:
    """Project temporal estimates onto significant pose-symmetry classes.

    Unimodal pairs are unchanged. For multimodal pairs, the odometry-aware
    temporal estimate chooses the compatible discrete class; the class mean
    replaces the potentially between-class soft estimate. An abstained
    temporal estimate remains abstained unless one class has >= .75 mass.
    """
    output = temporal_decided.copy()
    plans = _plans(inputs.cache)
    for local, pair_index in enumerate(inputs.attempted):
        plan = plans[pair_index]
        if plan is None:
            continue
        batch = inputs.batches[local]
        ego = batch["ego_boxes"][0, :, :2].detach().cpu().numpy().astype(np.float64)
        cav = batch["cav_boxes"][0, :, :2].detach().cpu().numpy().astype(np.float64)
        classes = _classes(plan.astype(np.float64), ego, cav,
                           np.random.default_rng(int(inputs.cache.get("seed", 0)) * 1000003 + int(pair_index)),
                           hypotheses)
        if len(classes) < 2:
            continue
        reference = temporal_decided[pair_index].astype(np.float64)
        if not np.any(np.abs(reference) > 1e-9):
            best = max(classes, key=lambda cluster: cluster["mass"])
            if best["mass"] >= 0.75:
                output[pair_index] = best["centre"]
            continue
        def objective(cluster):
            centre = cluster["centre"]
            distance = np.linalg.norm(centre[1:] - reference[1:]) / 0.5
            distance += _angle(centre[0], reference[0]) / math.radians(3)
            return distance - math.log(max(cluster["mass"], 1e-12))
        output[pair_index] = min(classes, key=objective)["centre"]
    return output


__all__ = ["decide"]
