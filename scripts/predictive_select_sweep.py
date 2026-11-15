#!/usr/bin/env python3
"""End-to-end one-seed evaluation of cross-fitted predictive correction."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_verify_core  # noqa: E402
import reproduce_opv2v_sweep as base  # noqa: E402

from embedding_aware_belt_fusion.alignformer_ext import (  # noqa: E402
    async_cache, evaluate, fusion, latency, odometry, predictive_select, variants,
)


CONFIG = {
    "opv2v": dict(cache="outputs/propose_cache_opv2v", table="outputs/opv2v_frame_table.pt",
                  geometry="outputs/r140_geometry_test_cache", odometry="outputs/odometry/opv2v_test.npz",
                  checkpoint="models/opv2v/alignformer_boxes_only.pth", config="configs/pointpillar_late_fusion_r140.yaml",
                  eval_range=(140.0, 40.0)),
    "v2xreal": dict(cache="outputs/propose_cache_v2xreal", table="outputs/v2xreal_test_frame_table.pt",
                    geometry="outputs/v2xreal_geometry_test_cache", odometry="outputs/odometry/v2xreal_test.npz",
                    checkpoint="models/v2xreal/alignformer_boxes_only.pth",
                    config="models/v2xreal/late_fusion/pointpillar/config.yaml",
                    eval_range=(102.4, 51.2)),
}


def _score(records, cache, correction, nms, scale=None):
    return evaluate.ap_table(base, evaluate.score_condition(base, records, cache, correction, nms, score_scale=scale))


def _variants(inputs, current):
    cross, cross_diagnostic = predictive_select.decide(inputs, current, mode="crossfit", confidence_z=0.0)
    lcb = current.copy()
    tested = cross_diagnostic["count"] >= 2
    lcb_rejected = tested & ((cross_diagnostic["gain"] - cross_diagnostic["error"]) <= 0.0)
    lcb[lcb_rejected] = 0.0
    lcb_diagnostic = dict(cross_diagnostic)
    lcb_diagnostic["rejected"] = lcb_rejected
    return {
        "current": (current, None),
        "crossfit": (cross, cross_diagnostic),
        "crossfit_lcb": (lcb, lcb_diagnostic),
        "insample": predictive_select.decide(inputs, current, mode="insample", confidence_z=0.0),
    }


def _condition(records, cache, inputs, nms, scale=None):
    methods = _variants(inputs, variants.alignformer_tk(inputs))
    result = {}
    for name, value in methods.items():
        if name == "current":
            decided, diagnostic = value
        else:
            decided, diagnostic = value
        corrected = np.any(np.abs(decided) > 1e-9, axis=1)
        entry = {"ap": _score(records, cache, decided, nms, scale), "coverage": float(corrected.mean())}
        if diagnostic is not None:
            tested = diagnostic["count"] >= 2
            entry.update(tested_fraction=float(tested.mean()), rejected_fraction=float(diagnostic["rejected"].mean()),
                         mean_gain=float(np.nanmean(diagnostic["gain"][tested])) if tested.any() else None)
        result[name] = entry
    return result


def run_position(dataset, cfg, seed, records, model, robust, nms):
    output = {}
    for path in sorted(Path(cfg["cache"]).glob(f"sigma_*_seed_{seed}.npz")):
        with np.load(path) as z:
            cache = {k: z[k] for k in z.files}
        inputs = variants.PairInputs(records, cache, model, robust, base=base)
        key = f"{float(cache['sigma']):g}"
        pose_path = Path(f"outputs/poses/{dataset}/alignformer/sigma_{key}_seed_{seed}.npz")
        with np.load(pose_path) as poses:
            current = poses["correction"].astype(np.float32)
        methods = _variants(inputs, current)
        output[key] = {}
        for name, (decided, diagnostic) in methods.items():
            corrected = np.any(np.abs(decided) > 1e-9, axis=1)
            entry = {"ap": _score(records, cache, decided, nms), "coverage": float(corrected.mean())}
            if diagnostic is not None:
                tested = diagnostic["count"] >= 2
                entry.update(tested_fraction=float(tested.mean()), rejected_fraction=float(diagnostic["rejected"].mean()),
                             mean_gain=float(np.nanmean(diagnostic["gain"][tested])) if tested.any() else None)
            output[key][name] = entry
        print(dataset, "position", key, *(f"{n}={v['ap']['all']['ap_70']:.4f}" for n, v in output[key].items()), flush=True)
    return output


def run_delay(dataset, cfg, seed, records, model, robust, nms):
    scenarios = async_cache.scenario_ids(Path(cfg["geometry"]))
    prediction = fusion.prediction_error_table(base, records, scenarios, range(7))
    output = {}
    for path in sorted(Path(f"outputs/async/{dataset}").glob(f"async_d*_sigma_*_seed_{seed}.npz")):
        if "_comp_" in path.name:
            continue
        parts = path.stem.split("_")
        delay = int(parts[1][1:]); sigma = float(parts[parts.index("sigma") + 1])
        raw = dict(np.load(path))
        grid = latency.latency_grid(records, raw, model, robust, base, list(range(7)))
        _, tau, _ = latency.refine_latency(grid, raw, scenarios)
        moved, inputs = latency.decide_at(records, raw, tau, model, robust, base)
        scale = fusion.staleness_confidence(moved, prediction, tau)
        key = f"d{delay}_s{sigma:g}"
        output[key] = _condition(records, moved, inputs, nms, scale)
        print(dataset, "delay", key, *(f"{n}={v['ap']['all']['ap_70']:.4f}" for n, v in output[key].items()), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(CONFIG), default=sorted(CONFIG))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/diagnostics/predictive_select_seed0.json"))
    args = parser.parse_args()
    result = {"seed": args.seed, "datasets": {}}
    for dataset in args.datasets:
        cfg = CONFIG[dataset]
        base.fusion_core.set_filters(*cfg["eval_range"], -3.0, 1.0, 6.0)
        records = check_verify_core.load_frame_table(Path(cfg["table"]))
        model = base.load_alignformer_pva(Path(cfg["checkpoint"]), torch.device("cpu"))[0]
        robust = base.RobustSolveConfig(mode="huber", iterations=2)
        odometry.ACTIVE = odometry.OdometryTable.for_records(Path(cfg["odometry"]), Path(cfg["geometry"]), records)
        nms = float(base.load_yaml(cfg["config"], None)["postprocess"]["nms_thresh"])
        result["datasets"][dataset] = {
            "position": run_position(dataset, cfg, args.seed, records, model, robust, nms),
            "delay": run_delay(dataset, cfg, args.seed, records, model, robust, nms),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
