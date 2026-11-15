#!/usr/bin/env python3
"""Compare ambiguity, observability, and Wald signals on cached AlignFormer pairs.

This is evaluation-only: it reads the immutable Propose caches and exported
final poses, then predicts whether an *accepted* AlignFormer correction is
harmful relative to retaining the reported pose. Train/test splitting is by
scenario so repeated frames and noise seeds never cross the split.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_verify_core  # noqa: E402
import reproduce_opv2v_sweep as base  # noqa: E402


CONFIG = {
    "opv2v": ("outputs/opv2v_frame_table.pt", "outputs/propose_cache_opv2v"),
    "v2xreal": ("outputs/v2xreal_test_frame_table.pt", "outputs/propose_cache_v2xreal"),
}
YAW_SCALE_RAD = math.radians(10.0)


def yaw_of(matrix):
    return float(math.atan2(matrix[1, 0], matrix[0, 0]))


def transform_error(transform, truth):
    residual = np.linalg.inv(transform) @ truth
    return float(np.linalg.norm(residual[:2, 3])), abs(yaw_of(residual))


def unpack_plans(cache):
    pointer = 0
    plans = []
    for rows, columns in cache["plan_shape"]:
        size = int(rows * columns)
        if size:
            plans.append(cache["plan"][pointer:pointer + size].reshape(int(rows), int(columns)))
            pointer += size
        else:
            plans.append(None)
    if pointer != len(cache["plan"]):
        raise RuntimeError(f"read {pointer} plan values, cache has {len(cache['plan'])}")
    return plans


def ambiguity_and_information(plan, ego_centres, cav_centres, propose):
    if plan is None or plan.shape[0] <= 1 or plan.shape[1] <= 1 or len(cav_centres) == 0:
        return [np.nan] * 15
    weights = np.exp(plan[:-1, :-1].astype(np.float64))
    mass = weights.sum(axis=1)
    valid = mass > 1e-9
    if not np.any(valid):
        return [np.nan] * 15
    probability = weights[valid] / mass[valid, None]
    entropy = -(probability * np.log(np.maximum(probability, 1e-12))).sum(axis=1)
    confidence = probability.max(axis=1)
    if probability.shape[1] > 1:
        ordered = np.sort(probability, axis=1)
        margin = ordered[:, -1] - ordered[:, -2]
    else:
        margin = confidence.copy()
    qbar = probability @ cav_centres
    delta = cav_centres[None, :, :] - qbar[:, None, :]
    dispersion = (probability * np.sum(delta * delta, axis=-1)).sum(axis=1)

    psi, tx, ty = map(float, propose)
    cosine, sine = math.cos(psi), math.sin(psi)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    rotated = qbar @ rotation.T
    # Residual convention affects signs but not J'J eigenvalues.
    jacobian = np.zeros((len(rotated), 2, 3), dtype=np.float64)
    jacobian[:, 0, 0] = 1.0
    jacobian[:, 1, 1] = 1.0
    jacobian[:, 0, 2] = -rotated[:, 1]
    jacobian[:, 1, 2] = rotated[:, 0]
    hessian = np.einsum("n,nki,nkj->ij", mass[valid], jacobian, jacobian)
    scale = np.diag([1.0, 1.0, YAW_SCALE_RAD])
    whitened = scale @ hessian @ scale
    eigenvalues = np.maximum(np.linalg.eigvalsh(whitened), 1e-12)
    condition = eigenvalues[-1] / eigenvalues[0]

    # Row-factorized approximation of correspondence-marginalized observed
    # information. Each valid row is a latent categorical match. This ignores
    # cross-row one-to-one coupling, but unlike entropy it measures whether
    # plausible candidates pull the pose score in different directions.
    transformed_all = cav_centres @ rotation.T + np.array([tx, ty])
    complete = np.zeros((3, 3), dtype=np.float64)
    missing = np.zeros((3, 3), dtype=np.float64)
    valid_ego = ego_centres[valid]
    for row_probability, row_mass, target in zip(probability, mass[valid], valid_ego):
        row_jacobian = np.zeros((len(cav_centres), 2, 3), dtype=np.float64)
        row_jacobian[:, 0, 0] = 1.0
        row_jacobian[:, 1, 1] = 1.0
        row_jacobian[:, 0, 2] = -transformed_all[:, 1]
        row_jacobian[:, 1, 2] = transformed_all[:, 0]
        residual = transformed_all - target
        row_gradient = np.einsum("nki,nk->ni", row_jacobian, residual)
        mean_gradient = np.einsum("n,ni->i", row_probability, row_gradient)
        centered_gradient = row_gradient - mean_gradient
        complete += row_mass * np.einsum(
            "n,nki,nkj->ij", row_probability, row_jacobian, row_jacobian
        )
        missing += row_mass * row_mass * np.einsum(
            "n,ni,nj->ij", row_probability, centered_gradient, centered_gradient
        )
    complete_white = scale @ complete @ scale
    missing_white = scale @ missing @ scale
    observed_white = complete_white - missing_white
    complete_eig, complete_vec = np.linalg.eigh(complete_white + np.eye(3) * 1e-9)
    inverse_sqrt = complete_vec @ np.diag(1.0 / np.sqrt(np.maximum(complete_eig, 1e-9))) @ complete_vec.T
    lost_fraction = np.linalg.eigvalsh(inverse_sqrt @ missing_white @ inverse_sqrt)
    observed_eig = np.linalg.eigvalsh(observed_white)
    observed_positive = np.maximum(observed_eig, 1e-12)
    dustbin_fraction = 1.0 - float(mass[valid].sum()) / max(
        float(np.exp(plan[:-1, :]).sum()), 1e-12
    )
    return [
        float(entropy.mean()), float(confidence.mean()), float(margin.mean()),
        float(dispersion.mean()), float(np.quantile(dispersion, 0.9)),
        float(np.sum(mass[valid])), float(np.sum(valid)), float(dustbin_fraction),
        float(np.log(eigenvalues[0])), float(np.log(condition)),
        float(np.log(eigenvalues).sum()),
        float(np.trace(missing_white) / max(np.trace(complete_white), 1e-12)),
        float(lost_fraction[-1]), float(np.log(observed_positive[0])),
        float(observed_eig[0] <= 0.0),
    ]


def collect(dataset, sigma, seeds):
    table_path, cache_dir = CONFIG[dataset]
    records = check_verify_core.load_frame_table(Path(table_path))
    rows = []
    for seed in range(seeds):
        cache_path = Path(cache_dir) / f"sigma_{sigma:g}_seed_{seed}.npz"
        pose_path = Path("outputs/poses") / dataset / "alignformer" / f"sigma_{sigma:g}_seed_{seed}.npz"
        with np.load(cache_path) as z:
            cache = {key: z[key] for key in z.files}
        with np.load(pose_path) as poses:
            pose = {key: poses[key] for key in poses.files}
        if len(cache["frame"]) != len(pose["frame_index"]):
            raise RuntimeError(f"pair count mismatch for {dataset} seed {seed}")
        plans = unpack_plans(cache)
        for index, (frame_index, cav_id) in enumerate(zip(pose["frame_index"], pose["cav_id"])):
            record = records[int(frame_index)]
            cav_id = str(cav_id)
            truth = base.x1_to_x2(record.cav_poses[cav_id], record.ego_pose)
            corrected_m, corrected_yaw = transform_error(pose["transform"][index], truth)
            reported_transform = base.x1_to_x2(
                list(pose["reported_pose"][index]), list(pose["ego_pose"][index])
            )
            bare_m, bare_yaw = transform_error(reported_transform, truth)
            correction = pose["correction"][index]
            answered = bool(np.any(np.abs(correction) > 1e-9))
            start, end = cache["cav_off"][index:index + 2]
            cav_centres = cache["cav_boxes"][start:end, :2].astype(np.float64)
            features = ambiguity_and_information(
                plans[index], record.ego_boxes[:, :2].numpy().astype(np.float64),
                cav_centres, cache["propose"][index]
            )
            scenario = str(pose["scenario"][index])
            rows.append(features + [
                math.log1p(max(float(cache["statistic"][index]), 0.0)),
                math.log1p(max(float(cache["dof"][index]), 0.0)),
                bare_m, bare_yaw, corrected_m, corrected_yaw,
                float(answered), float(corrected_m > bare_m + 0.05),
                scenario, seed,
            ])
    return rows


FEATURES = [
    "entropy", "confidence", "top2_margin", "dispersion_mean", "dispersion_p90",
    "matched_mass", "effective_rows", "dustbin_fraction", "log_lambda_min",
    "log_condition", "log_determinant",
    "missing_trace_fraction", "max_lost_fraction", "log_observed_lambda_min",
    "observed_nonpositive", "log_wald", "log_dof",
]
MODELS = {
    "wald": ["log_wald", "log_dof"],
    "ambiguity": ["entropy", "confidence", "top2_margin", "dispersion_mean", "dispersion_p90", "dustbin_fraction"],
    "observability": ["log_lambda_min", "log_condition", "log_determinant", "matched_mass", "effective_rows"],
    "ambiguity_observability": FEATURES[:11],
    "missing_information": ["missing_trace_fraction", "max_lost_fraction", "log_observed_lambda_min", "observed_nonpositive"],
    "missing_plus_wald": ["log_wald", "log_dof", "missing_trace_fraction", "max_lost_fraction", "log_observed_lambda_min", "observed_nonpositive"],
    "all_plus_wald": FEATURES,
}


def scenario_split(scenarios):
    unique = np.array(sorted(set(scenarios)))
    rng = np.random.default_rng(20261109)
    rng.shuffle(unique)
    test_scenarios = set(unique[: max(1, round(0.4 * len(unique)))])
    return np.array([scenario not in test_scenarios for scenario in scenarios]), np.array(
        [scenario in test_scenarios for scenario in scenarios]
    ), test_scenarios


def metrics(y, probability):
    return {
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
    }


def bootstrap_ci(y, probability, scenarios, draws=300):
    unique = np.array(sorted(set(scenarios)))
    rng = np.random.default_rng(9917)
    values = []
    for _ in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(scenarios == group) for group in sampled])
        if len(np.unique(y[indices])) < 2:
            continue
        values.append([
            roc_auc_score(y[indices], probability[indices]),
            average_precision_score(y[indices], probability[indices]),
        ])
    values = np.asarray(values)
    return {
        "auroc_95ci": [float(x) for x in np.quantile(values[:, 0], [0.025, 0.975])],
        "auprc_95ci": [float(x) for x in np.quantile(values[:, 1], [0.025, 0.975])],
    }


def paired_bootstrap_delta(y, candidate, reference, scenarios, draws=1000):
    unique = np.array(sorted(set(scenarios)))
    rng = np.random.default_rng(271828)
    values = []
    for _ in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(scenarios == group) for group in sampled])
        if len(np.unique(y[indices])) < 2:
            continue
        values.append([
            roc_auc_score(y[indices], candidate[indices]) - roc_auc_score(y[indices], reference[indices]),
            average_precision_score(y[indices], candidate[indices]) - average_precision_score(y[indices], reference[indices]),
        ])
    values = np.asarray(values)
    return {
        "delta_auroc_vs_wald": float(roc_auc_score(y, candidate) - roc_auc_score(y, reference)),
        "delta_auroc_95ci": [float(x) for x in np.quantile(values[:, 0], [0.025, 0.975])],
        "delta_auprc_vs_wald": float(average_precision_score(y, candidate) - average_precision_score(y, reference)),
        "delta_auprc_95ci": [float(x) for x in np.quantile(values[:, 1], [0.025, 0.975])],
        "bootstrap_probability_auprc_better": float(np.mean(values[:, 1] > 0.0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(CONFIG), default=sorted(CONFIG))
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("outputs/diagnostics/alignment_hypotheses_sigma1.json"))
    args = parser.parse_args()
    result = {"sigma": args.sigma, "seeds": args.seeds, "datasets": {}}
    for dataset in args.datasets:
        rows = collect(dataset, args.sigma, args.seeds)
        matrix = np.asarray([row[:len(FEATURES)] for row in rows], dtype=np.float64)
        answered = np.asarray([row[len(FEATURES) + 4] for row in rows], dtype=bool)
        harmful = np.asarray([row[len(FEATURES) + 5] for row in rows], dtype=np.int64)
        scenarios = np.asarray([row[len(FEATURES) + 6] for row in rows])
        bare = np.asarray([row[len(FEATURES)] for row in rows])
        corrected = np.asarray([row[len(FEATURES) + 2] for row in rows])
        eligible = answered & np.isfinite(matrix).all(axis=1)
        train, test, test_scenarios = scenario_split(scenarios)
        train &= eligible
        test &= eligible
        dataset_result = {
            "pairs": len(rows), "answered_pairs": int(answered.sum()),
            "eligible_answered_pairs": int(eligible.sum()),
            "test_scenarios": len(test_scenarios), "test_pairs": int(test.sum()),
            "harm_rate_answered": float(harmful[eligible].mean()),
            "median_bare_m": float(np.median(bare[eligible])),
            "median_corrected_m": float(np.median(corrected[eligible])),
            "models": {},
        }
        probabilities = {}
        for name, columns in MODELS.items():
            indices = [FEATURES.index(column) for column in columns]
            model = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", C=0.3),
            )
            model.fit(matrix[train][:, indices], harmful[train])
            probability = model.predict_proba(matrix[test][:, indices])[:, 1]
            probabilities[name] = probability
            report = metrics(harmful[test], probability)
            report.update(bootstrap_ci(harmful[test], probability, scenarios[test]))
            order = np.argsort(probability)
            keep = order[: max(1, round(0.9 * len(order)))]
            report["harm_rate_at_90pct_coverage"] = float(harmful[test][keep].mean())
            dataset_result["models"][name] = report
        for name, probability in probabilities.items():
            if name != "wald":
                dataset_result["models"][name].update(
                    paired_bootstrap_delta(
                        harmful[test], probability, probabilities["wald"], scenarios[test]
                    )
                )
        result["datasets"][dataset] = dataset_result
        print(f"{dataset}: {eligible.sum()} accepted analyzable pairs, harm {100*harmful[eligible].mean():.2f}%")
        for name, report in dataset_result["models"].items():
            print(f"  {name:25s} AUROC {report['auroc']:.3f} AUPRC {report['auprc']:.3f} "
                  f"harm@90% {100*report['harm_rate_at_90pct_coverage']:.2f}%")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
