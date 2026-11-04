"""Choose FreeAlign's free parameters on the scenario-disjoint VALIDATION role.

A reimplementation that loses can always be dismissed as a strawman, and the
paper leaves several quantities unspecified (the edge-discrepancy threshold,
the anchor-list limit ``gamma``, the exponent ``p`` and the constant in the
selection score ``eps = (offset + sum_e eps_e) / r^p``, and whether RANSAC or
LMedS is used). This script picks them the same way every other
hyperparameter in this project is picked (``alignformer_pva/
calibrate_variance.py``, ``evaluation/belt_fusion.py``'s own gate/consensus
retuning): by grid search on the scenario-disjoint 15% validation slice,
never on test, and never by copying the labmate's own fitted numbers for his
detector/dataset (``embedding-aware-BELT-fusion``'s
``scripts/calibrate_freealign.py``, inspected read-only via ``git show`` for
METHODOLOGY only -- its grid shape and scoring criterion -- never for its
selected values).

**Adapted to this repo's own infrastructure rather than ported line for
line**, because the data path differs: the labmate's script re-runs his
V2X-Real detector live, per the task's instruction this script instead
reuses:

* ``embeddings/all_agent_association.py:records_for_role`` +
  ``data/complete_cache.py:IndexedCompleteFrameDataset`` over the existing,
  noise-free ``outputs/alignformer_pva_geometry_train_cache`` (6,374 official
  OPV2V train frames, ``noise_disabled: true`` -- the SAME cache
  ``alignformer_pva/calibrate_variance.py`` calibrated the variance model on)
  and the same scenario-disjoint split file
  (``outputs/spatial_trackformer_association_head_128d/scenario_split.json``),
  restricted to ``validation_scenarios``.
* ``alignformer_pva/dataset.py:AlignFormerPVAPairDataset`` +
  ``NoiseSchedule.fixed`` for the noise injection itself: it already does
  exactly what a FreeAlign calibration needs (project each CAV's TRUE boxes
  into the ego frame, perturb by a sampled planar SE(2) noise, and hand back
  both the noisy boxes and the exact correction that undoes the noise), so no
  second noise-injection implementation is written here. This is
  infrastructure reuse, not an architecture dependency: this script imports
  ``alignformer_pva`` for its noise/cache plumbing only, never for pose
  fitting -- :mod:`freealign.freealign` itself still imports nothing from
  ``alignformer_pva`` (see that module's own docstring).

**The criterion is deliberately not sigma = 0.** At zero localization noise
the true correction is exactly the identity, so a method that abstains on
every pair scores a perfect zero -- calibrating there would select for
silence (same reasoning the labmate's own script states). Scored at TWO
noise points simultaneously -- (1.0 m, 0 deg) and (2.0 m, 10 deg), the same
two non-trivial points ``evaluation/belt_fusion.py``'s own gate/consensus
retuning used for the SAME reason: a single sigma's winner need not be the
best compromise across the sweep this project actually reports. The grid is
scored on translation MAE over **all** pairs at both points, averaged,
counting a discarded message as the uncorrected (identity) pose it actually
produces, so coverage and accuracy trade off against each other the way they
do in the real sweep.

Usage::

    python -m embedding_aware_belt_fusion.freealign.calibrate_freealign \\
        --cache-dir outputs/alignformer_pva_geometry_train_cache \\
        --scenario-split-file outputs/spatial_trackformer_association_head_128d/scenario_split.json \\
        --output-json outputs/freealign_calibration.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from embedding_aware_belt_fusion.alignformer_pva.dataset import (
    AlignFormerPVAPairDataset,
    NoiseSchedule,
)
from embedding_aware_belt_fusion.data.complete_cache import IndexedCompleteFrameDataset
from embedding_aware_belt_fusion.embeddings.all_agent_association import records_for_role
from embedding_aware_belt_fusion.freealign.freealign import (
    EDGE_DISTANCE,
    EDGE_DISTANCE_YAW,
    LMEDS,
    RANSAC,
    FreeAlignConfig,
    freealign_estimate,
    is_fallback,
)

# The grid. Every axis the paper leaves open, plus the two the authors'
# shipped code fixes (included so their own values are measured, not assumed
# best). Smaller than the labmate's own 192-point grid (his search already
# settled the shape of this search; this project's job is to re-run it
# honestly on our own data, not to re-litigate the grid's design).
_GRID = {
    "edge_feature": (EDGE_DISTANCE, EDGE_DISTANCE_YAW),
    "edge_threshold_m": (0.15, 0.3, 0.5, 1.0, 1.5),
    "anchor_limit": (1, 2, 3),
    "epsilon_offset": (0.0, 100.0),
    "epsilon_power": (1.0, 3.0),
    "robust_estimator": (RANSAC, LMEDS),
}
# Probed on the winning configuration only: it is the abstain policy, not a
# matching parameter, and sweeping it inside the grid would let the search
# buy accuracy by declining the hard pairs.
_MIN_NODES = (2, 3, 4)

# The two non-trivial noise points scored, matching evaluation/belt_fusion.py's
# own gate/consensus retuning practice (docs/plan.md, "Phase 1, point 2").
_CALIBRATION_NOISE_POINTS: Tuple[Tuple[float, float], ...] = (
    (1.0, 0.0),
    (2.0, 10.0),
)

# FreeAlign's own headline matching metric (their Table II): the fraction of
# pairs whose transformation error exceeds 3 m. Reported beside the median
# because the MEAN is the wrong summary for this estimator -- a handful of
# coincidental subgraphs produce tens of metres and drag it, while AP responds
# to the bulk of the distribution.
_ERROR_RATE_THRESHOLD_M = 3.0


def _load_validation_pairs(
    cache_dir: Path,
    scenario_split_file: Path,
    index_workers: int,
    max_frames: Optional[int],
    xy_std: float,
    yaw_std_deg: float,
    seed: int,
) -> List[dict]:
    """One (ego, CAV) pair per CAV per validation-role frame, boxes + ground truth.

    Reuses ``AlignFormerPVAPairDataset`` for both the pairing and the noise
    injection (see module docstring); converts each item to the plain dict
    FreeAlign's grid search consumes (``ego_boxes``/``cav_boxes`` with their
    masks, plus ``psi_true``/``t_true``) rather than passing the dataset
    items straight through, so this function's output has no dependency on
    ``alignformer_pva``'s batch schema evolving.
    """
    records = records_for_role(cache_dir, scenario_split_file, "validation", index_workers)
    if max_frames is not None:
        records = records[:max_frames]
    frames = IndexedCompleteFrameDataset(cache_dir, records)

    generator = torch.Generator().manual_seed(seed)
    dataset = AlignFormerPVAPairDataset(
        frames, NoiseSchedule.fixed(xy_std, yaw_std_deg), generator=generator
    )
    pairs = []
    for index in range(len(dataset)):
        item = dataset[index]
        pairs.append({
            "ego_boxes": item["ego_boxes"],
            "cav_boxes": item["cav_boxes"],
            "ego_mask": item["ego_mask"],
            "cav_mask": item["cav_mask"],
            "psi_true": float(item["psi_true"]),
            "t_true": (float(item["t_true"][0]), float(item["t_true"][1])),
        })
    return pairs


def _batch(pair: dict) -> dict:
    return {
        "ego_boxes": pair["ego_boxes"].unsqueeze(0),
        "cav_boxes": pair["cav_boxes"].unsqueeze(0),
        "ego_mask": pair["ego_mask"].unsqueeze(0),
        "cav_mask": pair["cav_mask"].unsqueeze(0),
    }


def _score(pairs: Sequence[dict], config: FreeAlignConfig) -> dict:
    """Translation and yaw error over all pairs, and over answered pairs only."""
    translation, yaw, answered = [], [], []
    for pair in pairs:
        estimate = freealign_estimate(_batch(pair), config)
        psi, t = float(estimate.psi[0]), estimate.t[0]
        translation.append(float(np.hypot(
            float(t[0]) - pair["t_true"][0], float(t[1]) - pair["t_true"][1]
        )))
        difference = psi - pair["psi_true"]
        yaw.append(abs(float(np.degrees(
            np.arctan2(np.sin(difference), np.cos(difference))
        ))))
        answered.append(not bool(is_fallback(estimate)[0]))

    return _summarize(np.array(translation), np.array(yaw), np.array(answered))


def _summarize(translation: np.ndarray, yaw: np.ndarray, answered: np.ndarray) -> dict:
    def over(values: np.ndarray, mask: np.ndarray) -> Optional[float]:
        return (
            float((values[mask] > _ERROR_RATE_THRESHOLD_M).mean())
            if mask.any() else None
        )

    return {
        "translation_mae_m": float(translation.mean()),
        "translation_median_m": float(np.median(translation)),
        "yaw_mae_deg": float(yaw.mean()),
        "yaw_median_deg": float(np.median(yaw)),
        "coverage": float(answered.mean()),
        "error_rate_over_3m": over(translation, np.ones_like(answered, dtype=bool)),
        "answered_translation_mae_m": (
            float(translation[answered].mean()) if answered.any() else None
        ),
        "answered_translation_median_m": (
            float(np.median(translation[answered])) if answered.any() else None
        ),
        "answered_yaw_mae_deg": float(yaw[answered].mean()) if answered.any() else None,
        "answered_error_rate_over_3m": over(translation, answered),
    }


def _mean_translation_mae(pairs_by_point: Dict[Tuple[float, float], Sequence[dict]],
                           config: FreeAlignConfig) -> Tuple[float, Dict[str, dict]]:
    """Score ``config`` at every calibration noise point; return the mean
    translation MAE (the grid's selection criterion) and each point's full
    ``_score`` record (for the result file)."""
    per_point = {}
    for sigma_m, sigma_deg in pairs_by_point:
        key = f"sigma_{sigma_m:g}m_{sigma_deg:g}deg"
        per_point[key] = _score(pairs_by_point[(sigma_m, sigma_deg)], config)
    mean_mae = float(np.mean([row["translation_mae_m"] for row in per_point.values()]))
    return mean_mae, per_point


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("outputs/alignformer_pva_geometry_train_cache"),
    )
    parser.add_argument(
        "--scenario-split-file", type=Path,
        default=Path("outputs/spatial_trackformer_association_head_128d/scenario_split.json"),
    )
    parser.add_argument("--index-workers", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pairs_by_point: Dict[Tuple[float, float], List[dict]] = {}
    for sigma_m, sigma_deg in _CALIBRATION_NOISE_POINTS:
        pairs_by_point[(sigma_m, sigma_deg)] = _load_validation_pairs(
            args.cache_dir, args.scenario_split_file, args.index_workers,
            args.max_frames, sigma_m, sigma_deg, args.seed,
        )
        print(
            f"{len(pairs_by_point[(sigma_m, sigma_deg)])} pairs at sigma = "
            f"{sigma_m:g} m / {sigma_deg:g} deg",
            flush=True,
        )

    names = list(_GRID)
    results = []
    for values in itertools.product(*(_GRID[name] for name in names)):
        config = FreeAlignConfig(**dict(zip(names, values)))
        mean_mae, per_point = _mean_translation_mae(pairs_by_point, config)
        row = {"config": config.to_dict(), "mean_translation_mae_m": mean_mae, "by_point": per_point}
        results.append(row)
        print(
            f"  {config.edge_feature:<13} thr={config.edge_threshold_m:<4} "
            f"gamma={config.anchor_limit} off={config.epsilon_offset:<5} "
            f"p={config.epsilon_power} {config.robust_estimator:<6} "
            f"-> mean {mean_mae:.4f} m",
            flush=True,
        )

    results.sort(key=lambda row: row["mean_translation_mae_m"])
    best = FreeAlignConfig(**{
        name: results[0]["config"][key]
        for name, key in (
            ("edge_feature", "edge_feature"),
            ("edge_threshold_m", "edge_threshold_m"),
            ("anchor_limit", "anchor_limit_gamma"),
            ("epsilon_offset", "epsilon_offset"),
            ("epsilon_power", "epsilon_power"),
            ("robust_estimator", "robust_estimator"),
        )
    })
    abstain = []
    for count in _MIN_NODES:
        mean_mae, per_point = _mean_translation_mae(pairs_by_point, replace(best, min_nodes=count))
        abstain.append({"min_nodes": count, "mean_translation_mae_m": mean_mae, "by_point": per_point})

    payload = {
        "method": "freealign_reimplementation",
        "metric": "freealign_calibration",
        "note": (
            "Parameters of THIS PROJECT'S reimplementation of Lei et al., "
            "FreeAlign, ICRA 2024 (arXiv 2405.02965). Not the authors' code, "
            "and NOT the labmate's own fitted numbers for his detector/data "
            "(embedding-aware-BELT-fusion, inspected read-only for "
            "methodology only). Chosen on our own scenario-disjoint "
            "validation role only, never on test."
        ),
        "cache_dir": str(args.cache_dir),
        "scenario_split_file": str(args.scenario_split_file),
        "role": "validation",
        "calibration_noise_points": [
            {"position_std_m": s, "heading_std_deg": d} for s, d in _CALIBRATION_NOISE_POINTS
        ],
        "pairs_per_point": {
            f"sigma_{s:g}m_{d:g}deg": len(pairs_by_point[(s, d)])
            for s, d in _CALIBRATION_NOISE_POINTS
        },
        "selected": best.to_dict(),
        "selected_score": results[0],
        "grid": results,
        "abstain_threshold_probe": abstain,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nselected: {best}")
    for row in abstain:
        print(f"  min_nodes={row['min_nodes']}  mean {row['mean_translation_mae_m']:.4f} m")
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
