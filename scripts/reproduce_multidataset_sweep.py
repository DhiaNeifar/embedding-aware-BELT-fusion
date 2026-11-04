#!/usr/bin/env python3
"""Noise sweep on OPV2V, V2X-Real or CooperScene, AP broken out by CAV-count bucket.

A thin layer over ``scripts/reproduce_opv2v_sweep.py``: per (sigma, seed) it
calls that script's own ``run_condition`` -- same noise model (independent
world-frame draws per CAV), same Propose/Verify/Abstain configuration
(``FULL_REFINE_CONFIG``/``FULL_ABSTAIN_CONFIG``/``FULL_AGREEMENT_CONFIG``),
same FreeAlign, same fusion_core late fusion and AP -- so numbers are
comparable across the three datasets by construction. What this adds:

* ``--dataset``: the OpenCOOD dataset (poses + ground truth) comes from
  ``multidataset.sources`` (V2X-Real adapter; CooperScene vehicle-centric
  ego and range), and geometry is attached by (scenario, frame) key instead
  of by shard position.
* CAV-count buckets: each frame's bucket is its number of non-ego agents
  within COM_RANGE that have detections. ``run_condition`` runs once per
  (sigma, seed, bucket) on that bucket's frames; the ``all`` and ``3+`` rows
  are exact merges of the per-bucket TP/FP/score lists (AP is a function of
  the pooled lists), not averages of bucket APs.
* An ``ego_only`` row (ego's own detections alone, noise-independent): the
  bar fusion has to clear at all.

Delay conditions are not offered: bucket subsets break frame adjacency,
which ``run_condition``'s delay indexing assumes. Use the OPV2V script for
delay.

Ground truth is built with a stub LiDAR reader (it depends only on yaml
params); see ``multidataset.sources``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import reproduce_opv2v_sweep as base  # noqa: E402  (also sets up sys.path and torch threads)

from tqdm import tqdm  # noqa: E402

from embedding_aware_belt_fusion.multidataset import sources  # noqa: E402

MAIN_METHODS = ("ego_only", "uncorrected", "oracle", "freealign", "alignformer")
CONTEXT_METHODS = ("alignformer_context", "alignformer_context_adaptive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sources.DATASETS, required=True)
    parser.add_argument("--opencood-config", type=Path, required=True, help="the detector's own OpenCOOD config")
    parser.add_argument(
        "--data-root", type=Path, required=True,
        help=(
            "Dataset root holding <split>/. For cooperscene: the GenCPM layout dir "
            "(e.g. ~/cooperscene_preds/pp/TownCooperScene); a yaml-only staging tree is "
            "built under --stage-root from it."
        ),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--geometry-cache-dir", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, default=Path("outputs/cooperscene_stage"))
    parser.add_argument(
        "--ego-agent", default=None,
        help=f"Ego agent id. cooperscene default {sources.COOPERSCENE_DEFAULT_EGO!r} (agent "
             f"{sources.COOPERSCENE_RSU_ID!r} is the RSU); opv2v/v2xreal default to their own protocol.",
    )
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0])
    parser.add_argument("--seeds", type=int, default=20, help="seeds per nonzero sigma")
    parser.add_argument("--freealign-calibration", type=Path, required=True)
    parser.add_argument("--alignformer-pva-checkpoint", type=Path, required=True)
    parser.add_argument("--alignformer-pva-robust-iterations", type=int, default=2)
    parser.add_argument("--alignformer-context-checkpoint", type=Path, default=None)
    parser.add_argument("--consensus-gate-m", type=float, default=0.5)
    parser.add_argument("--refine-gates-m", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=None, help="smoke tests only")
    parser.add_argument("--frame-table-cache", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--frame-table-only", action="store_true",
        help="Build and save --frame-table-cache, then exit (no sweep). For the C++ variants pipeline.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def cav_count(record) -> int:
    return sum(1 for cav_id in record.cav_boxes if cav_id in record.cav_poses)


def load_frame_table(path: Path):
    """``torch.load`` a frame table, including ones ``reproduce_opv2v_sweep.py`` saved when run as
    ``__main__`` (their records pickle as ``__main__.FrameRecord``)."""
    main_module = sys.modules["__main__"]
    if not hasattr(main_module, "FrameRecord"):
        main_module.FrameRecord = base.FrameRecord
    return torch.load(path, weights_only=False)


def load_or_build_frame_table(args, dataset) -> Tuple[List, Optional[List]]:
    if args.frame_table_cache is not None and args.frame_table_cache.exists():
        print(f"Reusing frame table from {args.frame_table_cache}")
        payload = load_frame_table(args.frame_table_cache)
        if isinstance(payload, list):  # reproduce_opv2v_sweep.py's own format
            return payload[: args.max_frames], None
        if payload["dataset"] != args.dataset or payload["split"] != args.split:
            raise ValueError(
                f"{args.frame_table_cache} is {payload['dataset']}/{payload['split']}, "
                f"not {args.dataset}/{args.split}"
            )
        return payload["records"][: args.max_frames], payload["keys"][: args.max_frames]

    keys = sources.frame_keys(dataset)
    frame_count = len(keys) if args.max_frames is None else min(args.max_frames, len(keys))
    keys = keys[:frame_count]
    with sources.stub_lidar():
        records = base.build_frame_table(dataset, None, frame_count)
    sources.attach_geometry_by_key(records, keys, base.load_geometry_cache(args.geometry_cache_dir))
    if args.frame_table_cache is not None:
        args.frame_table_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"dataset": args.dataset, "split": args.split, "records": records, "keys": keys},
            args.frame_table_cache,
        )
        print(f"Saved frame table to {args.frame_table_cache}")
    return records, keys


def compact(stats: Dict[float, Dict]) -> Dict[float, Dict]:
    """Numpy copies of the TP/FP/score lists: cheap to pickle back from a worker and to merge."""
    return {
        t: {
            "tp": np.asarray(s["tp"], dtype=np.int8),
            "fp": np.asarray(s["fp"], dtype=np.int8),
            "score": np.asarray(s["score"], dtype=np.float32),
            "gt": int(s["gt"]),
        }
        for t, s in stats.items()
    }


def merge(parts: List[Dict[float, Dict]]) -> Dict[float, Dict]:
    return {
        t: {
            "tp": np.concatenate([p[t]["tp"] for p in parts]),
            "fp": np.concatenate([p[t]["fp"] for p in parts]),
            "score": np.concatenate([p[t]["score"] for p in parts]),
            "gt": sum(p[t]["gt"] for p in parts),
        }
        for t in base.THRESHOLDS
    }


def ap_row(stats: Dict[float, Dict]) -> Dict[str, float]:
    return {f"ap_{int(t * 100):02d}": base.calculate_ap(stats[t]) for t in base.THRESHOLDS}


def ego_only_stats(records, nms_thresh: float) -> Dict[float, Dict]:
    stats = base.new_stats()
    for record in records:
        corners = np.ascontiguousarray(
            base.box_utils.boxes_to_corners_3d(record.ego_boxes.numpy(), order=base.box_order()), dtype=np.float32
        )
        scores = np.ascontiguousarray(record.ego_scores.numpy(), dtype=np.float32)
        base.fuse_and_score_accumulate(corners, scores, record.gt_corners, nms_thresh, stats)
    return compact(stats)


# Fork-inherited worker state (see reproduce_opv2v_sweep._WORKER_* for why).
_W: Dict = {}


def run_one(condition: Tuple[float, int, int]):
    torch.set_num_threads(1)
    sigma, seed, bucket = condition
    with torch.no_grad():
        stats = base.run_condition(
            _W["records_by_bucket"][bucket], sigma, seed, _W["nms_thresh"], _W["freealign_config"],
            _W["pva_model"], _W["robust_config"], 0, show_progress=False,
            consensus_gate_m=_W["consensus_gate_m"], pva_context_model=_W["pva_context_model"],
            refine_gates_m=_W["refine_gates_m"],
        )
    return condition, {method: compact(stats[method]) for method in base.METHODS}


def main() -> None:
    args = parse_args()
    ego_id = args.ego_agent
    if args.dataset == "cooperscene":
        ego_id = ego_id or sources.COOPERSCENE_DEFAULT_EGO
        split_dir = sources.stage_cooperscene_split(args.data_root, args.split, args.stage_root)
    else:
        split_dir = args.data_root / args.split

    hypes = base.load_yaml(str(args.opencood_config), None)
    dataset = sources.build_sweep_dataset(args.dataset, hypes, split_dir, ego_id=ego_id)
    nms_thresh = float(dataset.post_processor.params["nms_thresh"])
    import opencood.data_utils.datasets as opencood_datasets

    gt_range, com_range = list(opencood_datasets.GT_RANGE), opencood_datasets.COM_RANGE
    records, _ = load_or_build_frame_table(args, dataset)
    if args.frame_table_only:
        if args.frame_table_cache is None:
            raise ValueError("--frame-table-only needs --frame-table-cache")
        print(f"{len(records)} frames; frame table at {args.frame_table_cache}")
        return
    if args.output is None:
        raise ValueError("--output is required unless --frame-table-only")

    by_bucket: Dict[int, List] = {}
    for record in records:
        by_bucket.setdefault(cav_count(record), []).append(record)
    bucket_sizes = {str(k): len(v) for k, v in sorted(by_bucket.items())}
    print(f"{args.dataset}/{args.split}: {len(records)} frames; frames per CAV-count bucket: {bucket_sizes}")
    if 0 in by_bucket:
        print(f"  note: {len(by_bucket[0])} frames have no CAV in range; they score identically for every method")
    aggregates = {"all": sorted(by_bucket)}
    if any(k >= 3 for k in by_bucket):
        aggregates["3+"] = [k for k in sorted(by_bucket) if k >= 3]

    freealign_config = base.freealign_config_from_calibration(args.freealign_calibration)
    device = torch.device(args.device)
    pva_model, _ = base.load_alignformer_pva(args.alignformer_pva_checkpoint, device)
    pva_context_model = (
        base.load_alignformer_pva_context(args.alignformer_context_checkpoint, device)[0]
        if args.alignformer_context_checkpoint is not None else None
    )
    methods = [m for m in base.METHODS if pva_context_model is not None or m not in CONTEXT_METHODS]

    _W.update(
        records_by_bucket=by_bucket, nms_thresh=nms_thresh, freealign_config=freealign_config,
        pva_model=pva_model, pva_context_model=pva_context_model,
        robust_config=base.RobustSolveConfig(mode="huber", iterations=args.alignformer_pva_robust_iterations),
        consensus_gate_m=args.consensus_gate_m,
        refine_gates_m=tuple(float(x) for x in args.refine_gates_m.split(",")) if args.refine_gates_m else None,
    )

    ego_only = {bucket: ego_only_stats(rs, nms_thresh) for bucket, rs in by_bucket.items()}
    conditions = [
        (sigma, seed, bucket)
        for sigma in args.sigmas
        for seed in ([0] if sigma == 0.0 else range(args.seeds))
        for bucket in sorted(by_bucket)
    ]
    raw: Dict[Tuple[float, int], Dict[int, Dict]] = {}
    started = time.time()
    if args.workers <= 1:
        for condition in tqdm(conditions, desc=f"{args.dataset} sweep (sigma, seed, bucket)"):
            done, stats = run_one(condition)
            raw.setdefault(done[:2], {})[done[2]] = stats
    else:
        import concurrent.futures as cf
        import multiprocessing as mp

        print(f"Running {len(conditions)} (sigma, seed, bucket) conditions across {args.workers} workers...")
        with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
            futures = [pool.submit(run_one, c) for c in conditions]
            for future in tqdm(cf.as_completed(futures), total=len(futures), desc=f"{args.dataset} sweep"):
                done, stats = future.result()
                raw.setdefault(done[:2], {})[done[2]] = stats

    rows = {str(k): [k] for k in sorted(by_bucket)}
    rows.update(aggregates)
    results: Dict[str, Dict] = {}
    for (sigma, seed), per_bucket in sorted(raw.items()):
        label = f"sigma_{sigma:g}"
        for row, members in rows.items():
            entry = results.setdefault(label, {}).setdefault(
                row, {"n_frames": sum(len(by_bucket[k]) for k in members), "per_seed": []}
            )
            ap = {"ego_only": ap_row(merge([ego_only[k] for k in members]))}
            for method in methods:
                ap[method] = ap_row(merge([per_bucket[k][method] for k in members]))
            entry["per_seed"].append({"seed": seed, "ap": ap})
    for per_row in results.values():
        for entry in per_row.values():
            entry["mean"], entry["std"] = {}, {}
            for method in entry["per_seed"][0]["ap"]:
                for key in entry["per_seed"][0]["ap"][method]:
                    values = np.array([s["ap"][method][key] for s in entry["per_seed"]])
                    entry["mean"].setdefault(method, {})[key] = float(values.mean())
                    entry["std"].setdefault(method, {})[key] = float(values.std())

    payload = {
        "meta": {
            "dataset": args.dataset, "split": args.split, "frames": len(records),
            "frames_per_cav_count": bucket_sizes, "ego_agent": ego_id,
            "gt_range": gt_range, "com_range": com_range, "nms_thresh": nms_thresh,
            "opencood_config": str(args.opencood_config), "geometry_cache_dir": str(args.geometry_cache_dir),
            "freealign_calibration": str(args.freealign_calibration),
            "alignformer_pva_checkpoint": str(args.alignformer_pva_checkpoint),
            "alignformer_context_checkpoint": str(args.alignformer_context_checkpoint),
            "refine_gates_m": args.refine_gates_m, "seeds": args.seeds, "sigmas": args.sigmas,
            "max_frames": args.max_frames, "wall_seconds": round(time.time() - started, 1),
        },
        "sigma": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1))
    print(f"Wrote {args.output}")

    shown = [m for m in MAIN_METHODS if m == "ego_only" or m in methods]
    for row in rows:
        print(f"\nAP@0.7 mean over seeds -- {args.dataset}/{args.split}, CAVs={row} "
              f"({results[next(iter(results))][row]['n_frames']} frames)")
        print("sigma   " + "".join(f"{m:>13}" for m in shown) + f"{'AF-FA':>9}")
        for label, per_row in results.items():
            mean = per_row[row]["mean"]
            cells = "".join(f"{mean[m]['ap_70']:>13.4f}" for m in shown)
            delta = mean["alignformer"]["ap_70"] - mean["freealign"]["ap_70"]
            print(f"{label[6:]:<8}{cells}{delta:>+9.4f}")


if __name__ == "__main__":
    main()
