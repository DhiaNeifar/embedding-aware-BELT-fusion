"""Fit :mod:`alignformer_pva.variance`'s inverse-variance model on OUR detector.

``variance.py`` ships with placeholder constants fitted on the labmate's own
detector/validation split (its module docstring says so explicitly, and flags
them as NOT valid for our detector). This script is the "as-yet-unported"
``scripts/fit_correspondence_variance.py`` the labmate's own repo refers to
(``origin/alignformer:scripts/fit_correspondence_variance.py`` in
``embedding-aware-BELT-fusion``, inspected read-only for methodology, never
copied/modified), adapted to our own cache schema and our own detector.

NAMING/LOCATION NOTE: this lives inside the ``alignformer_pva`` package
(``alignformer_pva/calibrate_variance.py``) rather than at
``scripts/fit_correspondence_variance.py`` at the repo root the way the
labmate's own script is laid out. This project already keeps its
train/eval entry points colocated with the package they calibrate/train
(``alignformer_pva/train.py`` is run via
``python -m embedding_aware_belt_fusion.alignformer_pva.train``, not from
``scripts/``), and this script reuses several of that package's own
underscore-prefixed internals (``dataset._project_true``,
``dataset.correspondence_indices``, ``procrustes.heading_orientation``) that
are not meant to be imported from outside the package. Following the
existing convention won over mirroring the labmate's exact path.

WHAT IS MEASURED (mirrors the labmate's own protocol, not his numbers):
measured on the scenario-disjoint VALIDATION role only
(``outputs/spatial_trackformer_association_head_128d/scenario_split.json``,
the same split the geometry-only baseline validated on, per the task's
instruction to keep calibration honestly held out), from the noise-free
complete cache (``outputs/pointpillars_complete_cache_train_v3``,
``noise_disabled: true`` in its manifest), under the TRUE
(``transformation_matrix``) pose projection -- so sigma=0 throughout, and what
is measured is detector disagreement between two agents reporting the SAME
physical object, never pose/localization error. For every (ego, CAV) pair in
every validation-role frame, objects are matched by shared ``gt_ids``
(:func:`alignformer_pva.dataset.correspondence_indices`, reused verbatim, not
reimplemented) and, per matched pair, this records:

* the 2-D centre distance between the two reported centres (metres);
* the angular heading disagreement AFTER folding the 180-degree box-axis
  ambiguity the same way :mod:`alignformer_pva.procrustes` does
  (:func:`alignformer_pva.procrustes.heading_orientation`, reused directly --
  see :func:`_fold_sign` -- not reimplemented, since an unfolded comparison
  would corrupt roughly half the measurements exactly the way it would
  corrupt a naive Kabsch average);
* ``ego_score`` and ``cav_score`` separately (not just their min), since the
  fitted model (``v(s) = sigma_1^2 (s/s_ref)^-2b``, additive across the two
  independent detections) needs both.

FITTING METHOD: the labmate's own script fits five candidate variance forms
per channel by minimizing the Gamma/exponential quasi-deviance of the squared
residual against its modelled mean -- the correct MLE criterion for fitting a
*variance* (as opposed to least-squares on a mean) -- and pre-registers
``score_additive`` (``Var = v(s_ego) + v(s_cav)``) as the selected form because
it is the one implied by the physics: a correspondence's disagreement is the
difference of two independent detections, so its variance is the sum of
theirs. This script keeps that same methodology
(:func:`fit_score_additive_mle`, :func:`_gamma_deviance_objective`) and that
same pre-registered form -- :data:`VARIANCE_MODES`'s ``"split"`` in our own
``variance.py`` is exactly this form, applied separately to the centre and
heading channels as the task requires (one scalar exponent is the wrong shape;
see ``variance.py``'s own docstring) -- rather than reinventing a different
fitting procedure. The five-candidate-form comparison itself is not repeated
here (it is the labmate's own ablation, already settled; re-running it on our
data is out of scope for this task), but a decile table
(:func:`score_decile_table`) is still reported, in the same shape as the one
already printed in ``variance.py``'s own docstring, purely as a visual sanity
check of the fit -- not as a second, competing fitting method.

``score_reference = 0.4`` is kept matching the existing placeholder (the
module's own docstring calls this "purely a parameterization choice"); no
reason to re-derive it found during this task.

Usage::

    python -m embedding_aware_belt_fusion.alignformer_pva.calibrate_variance \\
        --cache-dir outputs/pointpillars_complete_cache_train_v3 \\
        --scenario-split-file outputs/spatial_trackformer_association_head_128d/scenario_split.json \\
        --role validation \\
        --output-json outputs/alignformer_pva_variance_calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import minimize
from torch import Tensor

from embedding_aware_belt_fusion.alignformer_pva.dataset import (
    _agent_has_boxes,
    _project_true,
    correspondence_indices,
)
from embedding_aware_belt_fusion.alignformer_pva.procrustes import heading_orientation
from embedding_aware_belt_fusion.data.complete_cache import IndexedCompleteFrameDataset
from embedding_aware_belt_fusion.embeddings.all_agent_association import (
    ego_index,
    records_for_role,
)

# Index of the yaw field in our [x, y, z, h, w, l, yaw] box layout, matching
# alignformer_pva.dataset.BOX_YAW (each module in this package re-declares it
# locally rather than importing a shared constant; followed here too).
BOX_YAW = 6

# The reference decile table already printed in variance.py's own module
# docstring (the labmate's detector/split) -- carried here verbatim, as
# *data* for the printed sanity-check comparison, never as a fallback value.
LABMATE_REFERENCE_DECILES = (
    {"min_score_range": "0.200-0.284", "n": 6398, "rms_centre_m": 0.532, "rms_yaw_deg": 13.56},
    {"min_score_range": "0.415-0.445", "n": 6396, "rms_centre_m": 0.281, "rms_yaw_deg": 4.23},
    {"min_score_range": "0.560-0.773", "n": 6400, "rms_centre_m": 0.214, "rms_yaw_deg": 3.15},
)


# --------------------------------------------------------------------------
# Correspondence collection: reuses dataset.py / procrustes.py internals.
# --------------------------------------------------------------------------


def _fold_sign(ego_yaw: Tensor, cav_yaw: Tensor) -> Tensor:
    """``(K,)`` of ``+1/-1``, one per matched pair, folding each ``cav_yaw``
    onto its OWN ``ego_yaw``'s half-plane.

    Thin wrapper around :func:`alignformer_pva.procrustes.heading_orientation`,
    which takes ``(B, M)`` references against ``(B, N)`` candidates and returns
    the full ``(B, M, N)`` pairwise matrix. Each of the ``K`` matched pairs here
    must be treated as its OWN batch item with exactly one reference and one
    candidate (``(K, 1)`` / ``(K, 1)`` -> ``(K, 1, 1)``) -- unsqueezing at dim 0
    instead (``(1, K)`` / ``(1, K)`` -> ``(1, K, K)``) would put all K yaws into
    one shared batch item's M/N axes and return the full K x K cross matrix,
    whose diagonal (not row 0) is what corresponds to "pair i with itself".
    Unsqueezing at dim 1 sidesteps that cross-matrix entirely: every pair gets
    its own batch slot, so indexing ``[:, 0, 0]`` is correct for every pair,
    not just pair 0.
    """
    sign = heading_orientation(ego_yaw.unsqueeze(1), cav_yaw.unsqueeze(1))
    return sign[:, 0, 0]


def _wrap_angle(angle: Tensor) -> Tensor:
    """Generic angle wrap to ``(-pi, pi]``, not itself "the fold" -- a trivial
    ``atan2(sin, cos)`` identity every module in this package re-declares
    locally rather than importing (see ``dataset.py``'s own ``_wrap_angle``
    and its comment on this convention)."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def collect_pair_correspondences(
    ego_agent: dict, cav_agent: dict
) -> Optional[Dict[str, np.ndarray]]:
    """Per-matched-object disagreement for one (ego, CAV) pair under the TRUE pose.

    Returns ``None`` when either side has no detections or nothing matched by
    ``gt_ids``; otherwise a dict of equal-length 1-D numpy arrays:
    ``centre_residual_m``, ``yaw_residual_rad`` (post-fold), ``ego_score``,
    ``cav_score``.
    """
    if not _agent_has_boxes(ego_agent) or not _agent_has_boxes(cav_agent):
        return None

    ego_boxes = _project_true(ego_agent)
    cav_boxes = _project_true(cav_agent)
    ego_ids = list(ego_agent["gt_ids"])
    cav_ids = list(cav_agent["gt_ids"])
    ego_match, _ = correspondence_indices(ego_ids, cav_ids)

    matched_rows = (ego_match >= 0).nonzero(as_tuple=True)[0]
    if matched_rows.numel() == 0:
        return None
    partner = ego_match[matched_rows]

    ego_centre = ego_boxes[matched_rows, :2]
    cav_centre = cav_boxes[partner, :2]
    centre_residual = (ego_centre - cav_centre).norm(dim=-1)

    ego_yaw = ego_boxes[matched_rows, BOX_YAW]
    cav_yaw = cav_boxes[partner, BOX_YAW]
    sign = _fold_sign(ego_yaw, cav_yaw)
    folded_direction = sign.unsqueeze(-1) * torch.stack(
        [torch.cos(cav_yaw), torch.sin(cav_yaw)], dim=-1
    )
    folded_yaw = torch.atan2(folded_direction[:, 1], folded_direction[:, 0])
    yaw_residual = _wrap_angle(ego_yaw - folded_yaw).abs()

    ego_scores = ego_agent["scores"].float()[matched_rows]
    cav_scores = cav_agent["scores"].float()[partner]

    return {
        "centre_residual_m": centre_residual.numpy(),
        "yaw_residual_rad": yaw_residual.numpy(),
        "ego_score": ego_scores.numpy(),
        "cav_score": cav_scores.numpy(),
    }


def collect_correspondences(frames) -> Dict[str, np.ndarray]:
    """Concatenate :func:`collect_pair_correspondences` over every (ego, CAV)
    pair in every frame of ``frames`` (anything indexable to a
    ``data/complete_cache.py``-shaped frame dict, e.g. an
    ``IndexedCompleteFrameDataset``).

    Pairing logic -- ego agent located via
    ``embeddings.all_agent_association.ego_index``, paired against every
    OTHER agent in the frame -- mirrors
    ``alignformer_pva.dataset.AlignFormerPVAPairDataset.__init__`` exactly
    (same ego-vs-every-CAV scope the model is trained and evaluated under),
    reused rather than reinvented, but without that class's top-N truncation:
    calibration wants every matched correspondence the detector actually
    produced, not a token-budget-limited subset.
    """
    centre_chunks: List[np.ndarray] = []
    yaw_chunks: List[np.ndarray] = []
    ego_score_chunks: List[np.ndarray] = []
    cav_score_chunks: List[np.ndarray] = []
    pairs_with_matches = 0
    pairs_seen = 0

    for frame_index in range(len(frames)):
        frame = frames[frame_index]
        agents = frame["agents"]
        if not agents:
            continue
        ego_agent_index = ego_index(frame)
        ego_agent = agents[ego_agent_index]
        if not _agent_has_boxes(ego_agent):
            continue
        for agent_position, agent in enumerate(agents):
            if agent_position == ego_agent_index:
                continue
            if not _agent_has_boxes(agent):
                continue
            pairs_seen += 1
            result = collect_pair_correspondences(ego_agent, agent)
            if result is None:
                continue
            pairs_with_matches += 1
            centre_chunks.append(result["centre_residual_m"])
            yaw_chunks.append(result["yaw_residual_rad"])
            ego_score_chunks.append(result["ego_score"])
            cav_score_chunks.append(result["cav_score"])

    if not centre_chunks:
        raise RuntimeError(
            "no matched correspondences found across the given frames -- check "
            "--cache-dir/--scenario-split-file/--role before trusting a fit"
        )

    return {
        "centre_residual_m": np.concatenate(centre_chunks),
        "yaw_residual_rad": np.concatenate(yaw_chunks),
        "ego_score": np.concatenate(ego_score_chunks),
        "cav_score": np.concatenate(cav_score_chunks),
        "pairs_seen": pairs_seen,
        "pairs_with_matches": pairs_with_matches,
    }


# --------------------------------------------------------------------------
# Fitting: Gamma-deviance MLE of the pre-registered "score_additive" form.
# Pure numpy, no cache/dataset dependency, so this half is directly
# unit-testable against synthetic data (see tests/test_alignformer_pva_
# calibrate_variance.py).
# --------------------------------------------------------------------------


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else math.nan


def _gamma_deviance_objective(variance_fn, squared_residual: np.ndarray):
    """``mean(log v + r^2 / v)``: the Gamma/exponential quasi-deviance, up to a
    constant. Minimized by ``v = E[r^2]`` for a constant model, which is what
    makes it the correct criterion for fitting a *variance* rather than a
    mean (same criterion the labmate's own fitting script uses)."""

    def objective(parameters: np.ndarray) -> float:
        variance = variance_fn(parameters)
        if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
            return 1e12
        return float(np.mean(np.log(variance) + squared_residual / variance))

    return objective


def fit_score_additive_mle(
    residual: np.ndarray,
    ego_score: np.ndarray,
    cav_score: np.ndarray,
    score_reference: float,
) -> Tuple[float, float, bool]:
    """Fit ``v(s) = sigma_1^2 * (s / s_ref)^(-2b)``, ``Var = v(ego) + v(cav)``.

    Maximum-likelihood (Gamma deviance) fit of the one pre-registered
    "score_additive" form -- the form implied by the physics (a
    correspondence's disagreement is the difference of two independent
    detections, so its variance is the sum of theirs) and exactly what
    ``variance.py``'s ``mode="split"``/``"scalar"`` models expect per channel.

    Parameters
    ----------
    residual: ``(N,)`` the channel's disagreement (metres for centre, radians
        for yaw) for each matched correspondence.
    ego_score, cav_score: ``(N,)`` the two detections' own confidences.
    score_reference: the confidence ``sigma_1`` is quoted at.

    Returns
    -------
    (sigma_1, exponent, converged)
    """
    if residual.shape != ego_score.shape or residual.shape != cav_score.shape:
        raise ValueError("residual/ego_score/cav_score must have the same shape")
    ego_ratio = ego_score / score_reference
    cav_ratio = cav_score / score_reference
    squared = np.square(residual)
    sigma_guess = max(_rms(residual), 1e-6)

    def variance_fn(params: np.ndarray) -> np.ndarray:
        sigma1, exponent = params
        return (sigma1 * ego_ratio ** (-exponent)) ** 2 + (
            sigma1 * cav_ratio ** (-exponent)
        ) ** 2

    objective = _gamma_deviance_objective(variance_fn, squared)
    result = minimize(
        objective,
        x0=np.array([sigma_guess / math.sqrt(2.0), 1.0]),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-7, "fatol": 1e-10},
    )
    sigma1, exponent = result.x
    # sigma must stay positive; a pathological fit landing on the wrong side
    # of zero is a bug to surface, not to silently mirror.
    return float(abs(sigma1)), float(exponent), bool(result.success)


def score_decile_table(
    min_score: np.ndarray,
    centre_residual: np.ndarray,
    yaw_residual_rad: np.ndarray,
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    """RMS centre/yaw disagreement per ``min(score)`` decile, same shape as the
    table already printed in ``variance.py``'s own module docstring -- a
    visual sanity check of the fit, not a second fitting method."""
    edges = np.percentile(min_score, np.linspace(0, 100, n_bins + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        selection = (min_score >= low) & (min_score < high)
        rows.append(
            {
                "low": float(low),
                "high": float(high),
                "n": int(selection.sum()),
                "rms_centre_m": _rms(centre_residual[selection]),
                "rms_yaw_deg": math.degrees(_rms(yaw_residual_rad[selection])),
            }
        )
    return rows


# --------------------------------------------------------------------------
# End-to-end calibration (cache -> correspondences -> fit -> payload).
# --------------------------------------------------------------------------


def run_calibration(
    cache_dir: Path,
    scenario_split_file: Path,
    role: str,
    index_workers: int,
    max_frames: Optional[int],
    score_reference: float,
    decile_bins: int,
) -> Dict:
    records = records_for_role(cache_dir, scenario_split_file, role, index_workers)
    if max_frames is not None:
        records = records[:max_frames]
    frames = IndexedCompleteFrameDataset(cache_dir, records)

    data = collect_correspondences(frames)
    min_score = np.minimum(data["ego_score"], data["cav_score"])

    sigma_t, exponent_t, converged_t = fit_score_additive_mle(
        data["centre_residual_m"], data["ego_score"], data["cav_score"], score_reference
    )
    sigma_psi, exponent_psi, converged_psi = fit_score_additive_mle(
        data["yaw_residual_rad"], data["ego_score"], data["cav_score"], score_reference
    )

    table = score_decile_table(
        min_score, data["centre_residual_m"], data["yaw_residual_rad"], decile_bins
    )

    return {
        "correspondence_variance": {
            "mode": "split",
            "sigma_translation_m": sigma_t,
            "translation_exponent": exponent_t,
            "sigma_yaw_rad": sigma_psi,
            "sigma_yaw_deg": math.degrees(sigma_psi),
            "yaw_exponent": exponent_psi,
            "score_reference": score_reference,
        },
        "diagnostics": {
            "fit_method": "gamma_deviance_mle_score_additive",
            "fit_converged": {"translation": converged_t, "yaw": converged_psi},
            "n_correspondences": int(data["centre_residual_m"].size),
            "n_pairs_seen": int(data["pairs_seen"]),
            "n_pairs_with_matches": int(data["pairs_with_matches"]),
            "overall_rms_centre_m": _rms(data["centre_residual_m"]),
            "overall_rms_yaw_deg": math.degrees(_rms(data["yaw_residual_rad"])),
            "score_decile_profile": table,
            "labmate_reference_decile_table": list(LABMATE_REFERENCE_DECILES),
        },
        "provenance": {
            "cache_dir": str(cache_dir),
            "scenario_split_file": str(scenario_split_file),
            "role": role,
            "max_frames": max_frames,
            "frame_count": len(frames),
            "index_workers": index_workers,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--scenario-split-file", type=Path, required=True)
    parser.add_argument(
        "--role", choices=("train", "validation"), default="validation",
        help="Scenario-disjoint role to calibrate on; validation (default) keeps "
        "this honestly held out, matching the labmate's own practice.",
    )
    parser.add_argument("--index-workers", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--score-reference", type=float, default=0.4,
        help="Confidence the fitted sigmas are quoted at; 0.4 matches the "
        "existing variance.py placeholder (a parameterization choice, not "
        "re-derived here).",
    )
    parser.add_argument("--decile-bins", type=int, default=10)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _print_report(payload: Dict) -> None:
    diag = payload["diagnostics"]
    block = payload["correspondence_variance"]
    print(
        f"{diag['n_correspondences']} correspondences over "
        f"{diag['n_pairs_with_matches']}/{diag['n_pairs_seen']} (ego, CAV) pairs "
        f"with >=1 match, role={payload['provenance']['role']}, "
        f"frames={payload['provenance']['frame_count']}"
    )
    print(
        f"overall RMS: centre={diag['overall_rms_centre_m']:.4f} m  "
        f"yaw={diag['overall_rms_yaw_deg']:.3f} deg"
    )
    print("min(score) deciles (RMS), ours:")
    for row in diag["score_decile_profile"]:
        print(
            f"  {row['low']:.3f}-{row['high']:.3f}  n={row['n']:6d}  "
            f"centre={row['rms_centre_m']:.4f} m  yaw={row['rms_yaw_deg']:.3f} deg"
        )
    print("labmate's reference deciles (different detector/data -- sanity check only):")
    for row in diag["labmate_reference_decile_table"]:
        print(
            f"  {row['min_score_range']}  n={row['n']:6d}  "
            f"centre={row['rms_centre_m']:.4f} m  yaw={row['rms_yaw_deg']:.3f} deg"
        )
    print(
        "fitted: v_t(s) = ({:.4f} m * (s/{:g})^-{:.4f})^2, "
        "v_psi(s) = ({:.4f} deg * (s/{:g})^-{:.4f})^2  (converged: t={}, psi={})".format(
            block["sigma_translation_m"], block["score_reference"],
            block["translation_exponent"], block["sigma_yaw_deg"],
            block["score_reference"], block["yaw_exponent"],
            diag["fit_converged"]["translation"], diag["fit_converged"]["yaw"],
        )
    )


def main() -> None:
    args = parse_args()
    payload = run_calibration(
        args.cache_dir,
        args.scenario_split_file,
        args.role,
        args.index_workers,
        args.max_frames,
        args.score_reference,
        args.decile_bins,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    _print_report(payload)
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
