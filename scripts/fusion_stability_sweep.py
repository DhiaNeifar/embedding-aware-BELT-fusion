#!/usr/bin/env python3
"""One-seed late-fusion sweeps for the fusion-stability certificate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_verify_core  # noqa: E402
import reproduce_opv2v_sweep as base  # noqa: E402

from embedding_aware_belt_fusion.alignformer_ext import (  # noqa: E402
    async_cache, evaluate, fusion, fusion_stability, latency, odometry, variants,
)


POSITION = {
    "opv2v": dict(cache="outputs/propose_cache_opv2v", table="outputs/opv2v_frame_table.pt",
                  geometry="outputs/r140_geometry_test_cache", odometry="outputs/odometry/opv2v_test.npz",
                  checkpoint="models/opv2v/alignformer_boxes_only.pth", config="configs/pointpillar_late_fusion_r140.yaml",
                  eval_range=(140.0, 40.0)),
    "v2xreal": dict(cache="outputs/propose_cache_v2xreal", table="outputs/v2xreal_test_frame_table.pt",
                    geometry="outputs/v2xreal_geometry_test_cache", odometry="outputs/odometry/v2xreal_test.npz",
                    checkpoint="models/v2xreal/alignformer_boxes_only.pth",
                    config="external/V2X-Real/opencood/logs/point_pillar_late_fusion/config.yaml",
                    eval_range=(102.4, 51.2)),
}


def score(records, cache, correction, nms, scale=None):
    return evaluate.ap_table(base, evaluate.score_condition(base, records, cache, correction, nms, score_scale=scale))


def position(dataset, cfg, seed):
    records = check_verify_core.load_frame_table(Path(cfg["table"]))
    model = base.load_alignformer_pva(Path(cfg["checkpoint"]), torch.device("cpu"))[0]
    robust = base.RobustSolveConfig(mode="huber", iterations=2)
    odometry.ACTIVE = odometry.OdometryTable.for_records(Path(cfg["odometry"]), Path(cfg["geometry"]), records)
    nms = float(base.load_yaml(cfg["config"], None)["postprocess"]["nms_thresh"])
    result = {}
    for path in sorted(Path(cfg["cache"]).glob(f"sigma_*_seed_{seed}.npz")):
        if path.name.startswith("cavcav_"):
            continue
        with np.load(path) as data:
            cache = {key: data[key] for key in data.files}
        inputs = variants.PairInputs(records, cache, model, robust, base=base)
        current = variants.alignformer_tk(inputs)
        certified = fusion_stability.certify(inputs, current)
        sigma = f"{float(cache['sigma']):g}"
        result[sigma] = {
            "current": score(records, cache, current, nms),
            "certified": score(records, cache, certified, nms),
            "coverage_current": float(np.mean(np.any(np.abs(current) > 1e-9, axis=1))),
            "coverage_certified": float(np.mean(np.any(np.abs(certified) > 1e-9, axis=1))),
        }
        print(dataset, "position", sigma, result[sigma]["current"]["all"]["ap_70"],
              result[sigma]["certified"]["all"]["ap_70"])
    return result


def delay(dataset, cfg, seed):
    async_dir = Path(f"outputs/async/{dataset}")
    records = check_verify_core.load_frame_table(Path(cfg["table"]))
    model = base.load_alignformer_pva(Path(cfg["checkpoint"]), torch.device("cpu"))[0]
    robust = base.RobustSolveConfig(mode="huber", iterations=2)
    odometry.ACTIVE = odometry.OdometryTable.for_records(Path(cfg["odometry"]), Path(cfg["geometry"]), records)
    scenarios = async_cache.scenario_ids(Path(cfg["geometry"]))
    table = fusion.prediction_error_table(base, records, scenarios, range(7))
    nms = float(base.load_yaml(cfg["config"], None)["postprocess"]["nms_thresh"])
    result = {}
    for raw_path in sorted(async_dir.glob(f"async_d*_sigma_*_seed_{seed}.npz")):
        if "_comp_" in raw_path.name:
            continue
        parts = raw_path.stem.split("_")
        delay_frames = int(parts[1][1:])
        sigma = float(parts[parts.index("sigma") + 1])
        raw = dict(np.load(raw_path))
        grid = latency.latency_grid(records, raw, model, robust, base, list(range(7)))
        _, tau, _ = latency.refine_latency(grid, raw, scenarios)
        moved, inputs = latency.decide_at(records, raw, tau, model, robust, base)
        current = variants.alignformer_tk(inputs)
        certified = fusion_stability.certify(inputs, current)
        scale = fusion.staleness_confidence(moved, table, tau)
        key = f"d{delay_frames}_s{sigma:g}"
        result[key] = {
            "current": score(records, moved, current, nms, scale),
            "certified": score(records, moved, certified, nms, scale),
            "coverage_current": float(np.mean(np.any(np.abs(current) > 1e-9, axis=1))),
            "coverage_certified": float(np.mean(np.any(np.abs(certified) > 1e-9, axis=1))),
        }
        print(dataset, "delay", key, result[key]["current"]["all"]["ap_70"],
              result[key]["certified"]["all"]["ap_70"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(POSITION), default=sorted(POSITION))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/diagnostics/fusion_stability_seed0.json"))
    args = parser.parse_args()
    output = {"seed": args.seed, "datasets": {}}
    for dataset in args.datasets:
        cfg = POSITION[dataset]
        base.fusion_core.set_filters(*cfg["eval_range"], -3.0, 1.0, 6.0)
        output["datasets"][dataset] = {
            "position": position(dataset, cfg, args.seed),
            "delay": delay(dataset, cfg, args.seed),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
