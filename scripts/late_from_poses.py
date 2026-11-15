#!/usr/bin/env python3
"""Late fusion with any plug-in's exported poses (outputs/poses, alignformer_ext/poses.py format).

Each CAV's boxes are placed with the plug-in's corrected pose, then fused by the same NMS and scored by the same AP
(C++ fusion_core) as alignformer_variants_sweep.py, on the same noise draws. This is how CoAlign and RoCo, whose
corrections run in their own code, enter late fusion; FreeAlign and AlignFormer are scored the same way for a
like-for-like table on the same seeds.

    python scripts/late_from_poses.py --dataset opv2v --output outputs/paper/late_opv2v_plugins.json
    python scripts/late_from_poses.py --dataset opv2v --ego-only --output outputs/paper/late_opv2v_ego.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_verify_core  # noqa: E402
import reproduce_opv2v_sweep as base  # noqa: E402

from embedding_aware_belt_fusion.alignformer_ext import evaluate, poses, propose_cache  # noqa: E402

DATASETS = {
    "opv2v": dict(cache="outputs/propose_cache_opv2v", table="outputs/opv2v_frame_table.pt",
                  config="configs/pointpillar_late_fusion_r140.yaml", eval_range=(140.0, 40.0)),
    "v2xreal": dict(cache="outputs/propose_cache_v2xreal", table="outputs/v2xreal_test_frame_table.pt",
                    config="models/v2xreal/late_fusion/pointpillar/config.yaml", eval_range=(102.4, 51.2)),
}
_W = {}


def run(job):
    torch.set_num_threads(1)
    sigma, seed = job
    records = _W["records"]
    with np.load(propose_cache.condition_path(_W["cache"], sigma, seed)) as data:
        cache = {k: data[k] for k in data.files}
    pair_of = {}
    for p in range(len(cache["frame"])):
        f = int(cache["frame"][p])
        pair_of[(f, str(sorted(records[f].cav_boxes)[int(cache["cav_rank"][p])]))] = p
    score = lambda decided, **kw: evaluate.ap_table(base, evaluate.score_condition(base, records, cache, decided, _W["nms"], **kw))
    ap = {"bare": score(None), "perfect": score(None, oracle=True)}
    for plugin in _W["plugins"]:
        with np.load(poses.condition_path(_W["poses"], _W["dataset"], plugin, sigma, seed)) as z:
            decided = np.zeros((len(cache["frame"]), 3), np.float32)
            for f, cav, c in zip(z["frame_index"], z["cav_id"], z["correction"]):
                p = pair_of.get((int(f), str(cav)))
                if p is not None:
                    decided[p] = c
        ap[plugin] = score(decided)
    return sigma, seed, ap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="opv2v")
    parser.add_argument("--plugins", nargs="+", default=["freealign", "coalign", "roco", "alignformer"])
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--poses", type=Path, default=Path("outputs/poses"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ego-only", action="store_true",
                        help="score the ego's own boxes alone (no CAV, so no sigma and no seed): the T3 'Ego only' row")
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    base.fusion_core.set_filters(cfg["eval_range"][0], cfg["eval_range"][1], -3.0, 1.0, 6.0)
    _W.update(dataset=args.dataset, plugins=args.plugins, poses=args.poses, cache=Path(cfg["cache"]),
              records=check_verify_core.load_frame_table(Path(cfg["table"])),
              nms=float(base.load_yaml(cfg["config"], None)["postprocess"]["nms_thresh"]))
    if args.ego_only:
        no_cav = {"frame": np.zeros(0, np.int64), "cav_off": np.zeros(1, np.int64)}
        records = tqdm(_W["records"], desc=f"ego only {args.dataset}")
        ap = evaluate.ap_table(base, evaluate.score_condition(base, records, no_cav, None, _W["nms"]))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"ego": ap}, indent=1))
        print(f"Wrote {args.output}: ego only " + "  ".join(f"{k} {v:.3f}" for k, v in ap["all"].items()))
        return
    jobs = [(sigma, seed) for sigma in args.sigmas for seed in ([0] if sigma == 0.0 else range(args.seeds))]
    results = {}
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in tqdm(cf.as_completed(futures), total=len(futures), desc=f"late fusion from poses {args.dataset}"):
            sigma, seed, ap = future.result()
            results.setdefault(f"{sigma:g}", []).append({"seed": seed, "ap": ap})
    summary = {}
    for sigma, per_seed in results.items():
        per_seed.sort(key=lambda e: e["seed"])
        for row in ["bare", "perfect"] + args.plugins:
            for key in ("ap_30", "ap_50", "ap_70"):
                v = np.array([e["ap"][row]["all"][key] for e in per_seed])
                summary.setdefault(sigma, {}).setdefault(row, {})[key] = {"mean": float(v.mean()), "n": len(v)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"plugins": args.plugins, "summary": summary, "per_seed": results}, indent=1))
    print(f"Wrote {args.output}")
    for sigma in sorted(summary, key=float):
        print(f"sigma {sigma:>4}: " + "  ".join(f"{r} {summary[sigma][r]['ap_70']['mean']:.3f}" for r in ["bare", "perfect"] + args.plugins))


if __name__ == "__main__":
    main()
