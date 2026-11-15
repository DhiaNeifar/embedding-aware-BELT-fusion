#!/usr/bin/env python3
"""CoAlign and RoCo under an unknown delay (Table T6, Figure 1d): the same delayed pairs, the same scoring.

A delayed pair is what scripts/build_async_cache.py stores: at receiver frame t, a CAV's message from frame
g = t - delay, its boxes placed with its noisy pose at g and the ego's pose at t (the receiver does not know the delay).
CoAlign and RoCo then correct the pose exactly as they do synchronously (scripts/export_poses_pgo.py: their own
pose graph, their own stage-1 boxes and uncertainties, their own box_align args), on that delayed input: the ego's
stage-1 boxes at t, the CAV's at g, the CAV's noisy pose at g, the ego's pose at t. Like FreeAlign in T6, they have
no delay handling. The correction is scored like every other row of T6: on the cache's stale boxes, by
evaluate.score_condition (same NMS, same AP, same pairs).

Three steps, each in the environment it needs:

    # 1 the pose-graph inputs of every delayed pair (OPV2V environment: frame table, noise draws)
    python scripts/delay_pgo.py prepare --dataset opv2v
    # 2 each method's correction (its own environment, it needs g2o)
    .../coalign-blackwell/bin/python scripts/delay_pgo.py solve --dataset opv2v --method coalign
    .../roco-blackwell/bin/python    scripts/delay_pgo.py solve --dataset opv2v --method roco
    # 3 AP, written to outputs/paper/delay_<dataset>_pgo.json (OPV2V environment)
    python scripts/delay_pgo.py score --dataset opv2v

Files: outputs/delay_pgo/<dataset>/jobs/<condition>.npz (step 1), outputs/delay_pgo/<dataset>/<method>/<condition>.npz
(step 2: ``correction`` (P, 3) psi, tx, ty per cached pair, in the cache's pair order). Finished files are skipped.
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))

DATASETS = {   # the inputs of the paper's T6 runs (scripts/run_paper.sh revision-5draws) and of late fusion
    "opv2v": dict(async_dir="outputs/async/opv2v", table="outputs/opv2v_frame_table.pt",
                  config="configs/pointpillar_late_fusion_r140.yaml", eval_range=(140.0, 40.0)),
    "v2xreal": dict(async_dir="outputs/async/v2xreal", table="outputs/v2xreal_test_frame_table.pt",
                    config="models/v2xreal/late_fusion/pointpillar/config.yaml", eval_range=(102.4, 51.2)),
}
METHODS = ("coalign", "roco")
OUT = REPO / "outputs" / "delay_pgo"
_W = {}


def conditions(async_dir: Path):
    """(delay, sigma, seed, path) of every raw (not velocity-compensated) delayed cache, as latency_sweep.py reads."""
    out = []
    for path in sorted(Path(async_dir).glob("async_d*_sigma_*_seed_*.npz")):
        if "_comp_" in path.name:
            continue
        parts = path.stem.split("_")
        out.append((int(parts[1][1:]), float(parts[parts.index("sigma") + 1]), int(parts[parts.index("seed") + 1]), path))
    if not out:
        raise SystemExit(f"no delayed caches in {async_dir}")
    return out


def _atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.rename(path)


# ------------------------------------------------------------------------------------------------- 1 prepare
def ego_ids(dataset: str):
    """frame index -> ego id, from the plug-in pose files (alignformer_ext/poses.py format, every frame with a CAV)."""
    path = REPO / "outputs" / "poses" / dataset / "alignformer" / "sigma_0_seed_0.npz"
    with np.load(path) as z:
        return {int(f): str(e) for f, e in zip(z["frame_index"], z["ego_id"])}


def prepare(args) -> None:
    import check_verify_core
    import reproduce_opv2v_sweep as base
    from embedding_aware_belt_fusion.alignformer_ext.propose_cache import noisy_poses

    spec = DATASETS[args.dataset]
    records = check_verify_core.load_frame_table(REPO / spec["table"])
    egos = ego_ids(args.dataset)
    for delay, sigma, seed, path in conditions(REPO / spec["async_dir"]):
        target = OUT / args.dataset / "jobs" / path.name
        if target.exists():
            continue
        with np.load(path) as z:
            frame, rank, past = z["frame"], z["cav_rank"], z["past_frame"]
        noisy = {}
        cav_id, reported, ego_pose, ego_id = [], [], [], []
        for t, r, g in zip(frame.tolist(), rank.tolist(), past.tolist()):
            cav = sorted(records[t].cav_boxes)[r]
            if g not in noisy:                       # the cache's own draw (build_async_condition: noisy_poses at g)
                noisy[g] = noisy_poses(base, records[g], g, sigma, seed)
            cav_id.append(str(cav))
            reported.append(np.asarray(noisy[g][cav], np.float64))
            ego_pose.append(np.asarray(records[t].ego_pose, np.float64))
            ego_id.append(egos[t])
        _atomic_savez(target, delay=np.int64(delay), sigma=np.float64(sigma), seed=np.int64(seed),
                      frame=frame, past_frame=past, cav_rank=rank, cav_id=np.array(cav_id), ego_id=np.array(ego_id),
                      reported_pose=np.array(reported).reshape(-1, 6), ego_pose=np.array(ego_pose).reshape(-1, 6))
        print(f"prepared {target.name}: {len(frame)} delayed pairs", flush=True)


# --------------------------------------------------------------------------------------------------- 2 solve
def correct_frame(order, noisy, corners, uncertainty, align, x1_to_x2):
    """One receiver frame: the method's pose graph over [ego] + CAVs, the ego fixed. Returns (n_cav, 3) psi, tx, ty:
    the correction of each CAV's reported transform, in the ego frame (as export_poses_pgo.py)."""
    refined = noisy.copy()
    if sum(len(c) for c in corners):
        refined[:, [0, 1, 4]] = align(corners, noisy.copy(), uncertainty_list=uncertainty)
    out = np.zeros((len(order) - 1, 3))
    for k in range(1, len(order)):
        transform = np.asarray(x1_to_x2(refined[k].tolist(), refined[0].tolist()))
        reported = np.asarray(x1_to_x2(noisy[k].tolist(), noisy[0].tolist()))
        c = transform @ np.linalg.inv(reported)
        out[k - 1] = (np.arctan2(c[1, 0], c[0, 0]), c[0, 3], c[1, 3])
    return out


def solve_condition(jobs: dict, stage1: dict, align, x1_to_x2) -> np.ndarray:
    """(P, 3) corrections for one delayed condition. A CAV whose message has no stage-1 entry at g, or a frame with
    no ego entry at t, keeps its reported pose (zero correction), as a frame without detections does synchronously."""
    n = len(jobs["frame"])
    correction = np.zeros((n, 3))
    by_frame = {}
    for p, t in enumerate(jobs["frame"].tolist()):
        by_frame.setdefault(t, []).append(p)
    for t, pairs in by_frame.items():
        ego_entry = stage1.get(str(t))
        if ego_entry is None:
            continue
        ego_ids = [str(i) for i in ego_entry["cav_id_list"]]
        ego = str(jobs["ego_id"][pairs[0]])
        if ego not in ego_ids:
            continue
        kept, corners, uncertainty = [], [np.array(ego_entry["pred_corner3d_np_list"][ego_ids.index(ego)], np.float64)], \
            [np.array(ego_entry["uncertainty_np_list"][ego_ids.index(ego)], np.float64)]
        for p in pairs:                               # each CAV's own stage-1 output of the frame its message is from
            entry = stage1.get(str(int(jobs["past_frame"][p])))
            if entry is None:
                continue
            ids = [str(i) for i in entry["cav_id_list"]]
            cav = str(jobs["cav_id"][p])
            if cav not in ids:
                continue
            kept.append(p)
            corners.append(np.array(entry["pred_corner3d_np_list"][ids.index(cav)], np.float64))
            uncertainty.append(np.array(entry["uncertainty_np_list"][ids.index(cav)], np.float64))
        if not kept:
            continue
        order = [ego] + [str(jobs["cav_id"][p]) for p in kept]
        noisy = np.stack([jobs["ego_pose"][kept[0]]] + [jobs["reported_pose"][p] for p in kept]).astype(np.float64)
        correction[kept] = correct_frame(order, noisy, corners, uncertainty, align, x1_to_x2)
    return correction


def _solve_job(path: Path):
    with np.load(path) as z:
        jobs = {k: z[k] for k in z.files}
    correction = solve_condition(jobs, _W["stage1"], _W["align"], _W["x1_to_x2"])
    _atomic_savez(_W["out"] / path.name, correction=correction.astype(np.float32), frame=jobs["frame"],
                  cav_rank=jobs["cav_rank"], delay=jobs["delay"], sigma=jobs["sigma"], seed=jobs["seed"])
    return path.name, float(np.mean(np.abs(correction).max(axis=1) > 1e-9)) if len(correction) else 0.0


def solve(args) -> None:
    import functools

    import yaml
    import export_poses_pgo as pgo                    # the synchronous export: same repos, configs, stage-1 boxes

    method = pgo.METHODS[args.method]
    sys.path.insert(0, str(method["repo"]))
    from opencood.models.sub_modules.box_align_v2 import box_alignment_relative_sample_np
    from opencood.utils.transformation_utils import x1_to_x2

    box_args = yaml.safe_load(pgo.box_align_block(method["config"]))["box_align"]["args"]
    stage1_path = args.stage1 or pgo.STAGE1[args.dataset]
    print(f"{args.method} {args.dataset}: stage-1 boxes {stage1_path}", flush=True)
    out = OUT / args.dataset / args.method
    _W.update(stage1=json.load(open(stage1_path)), out=out, x1_to_x2=x1_to_x2,
              align=functools.partial(box_alignment_relative_sample_np, **box_args))
    todo = [p for p in sorted((OUT / args.dataset / "jobs").glob("*.npz")) if not (out / p.name).exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} conditions to solve", flush=True)
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
        for name, share in pool.map(_solve_job, todo):
            print(f"  {name}: {share * 100:.0f}% of pairs corrected", flush=True)


# --------------------------------------------------------------------------------------------------- 3 score
def _score_job(job):
    import torch
    torch.set_num_threads(1)
    from embedding_aware_belt_fusion.alignformer_ext import evaluate
    delay, sigma, seed, path = job
    with np.load(path) as z:
        raw = {k: z[k] for k in z.files}
    base, records, nms = _W["base"], _W["records"], _W["nms"]
    ap = {}
    for method in METHODS:
        with np.load(OUT / _W["dataset"] / method / path.name) as z:
            if not (np.array_equal(z["frame"], raw["frame"]) and np.array_equal(z["cav_rank"], raw["cav_rank"])):
                raise SystemExit(f"{method} {path.name}: pairs differ from the cache")
            decided = z["correction"].astype(np.float32)
        ap[method] = evaluate.ap_table(base, evaluate.score_condition(base, records, raw, decided, nms))
    if _W["check"]:                                  # same function on the same cache as T6's rows: must reproduce
        ap["bare_check"] = evaluate.ap_table(base, evaluate.score_condition(base, records, raw, None, nms))
    return delay, sigma, seed, ap


def score(args) -> None:
    import check_verify_core
    import reproduce_opv2v_sweep as base
    from embedding_aware_belt_fusion.alignformer_ext import evaluate

    spec = DATASETS[args.dataset]
    base.fusion_core.set_filters(spec["eval_range"][0], spec["eval_range"][1], -3.0, 1.0, 6.0)
    _W.update(base=base, records=check_verify_core.load_frame_table(REPO / spec["table"]), dataset=args.dataset,
              nms=float(base.load_yaml(str(REPO / spec["config"]), None)["postprocess"]["nms_thresh"]), check=True)
    jobs = [c for c in conditions(REPO / spec["async_dir"])
            if all((OUT / args.dataset / m / c[3].name).exists() for m in METHODS)]
    if args.limit:
        jobs = jobs[:args.limit]
    results = {}
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
        for delay, sigma, seed, ap in pool.map(_score_job, jobs):
            results.setdefault(f"d{delay}", {}).setdefault(f"{sigma:g}", []).append({"seed": seed, "ap": ap})
    summary = {}
    for d, by_sigma in results.items():
        for sigma, per_seed in by_sigma.items():
            per_seed.sort(key=lambda e: e["seed"])
            for row in list(METHODS) + ["bare_check"]:
                for bucket in evaluate.BUCKETS:
                    for metric in ("ap_30", "ap_50", "ap_70"):
                        v = np.array([e["ap"][row][bucket][metric] for e in per_seed])
                        summary.setdefault(d, {}).setdefault(sigma, {}).setdefault(bucket, {}).setdefault(row, {})[metric] = {
                            "mean": float(v.mean()), "std": float(v.std()), "n": len(v)}
    output = args.output or REPO / "outputs" / "paper" / f"delay_{args.dataset}_pgo.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "per_seed": results}, indent=1))
    print(f"wrote {output}")
    for d in sorted(summary, key=lambda k: int(k[1:])):
        for sigma in sorted(summary[d], key=float):
            s = summary[d][sigma]["all"]
            print(f"  {int(d[1:]) * 100:>3} ms sigma {sigma}: " + "  ".join(
                f"{r} {s[r]['ap_70']['mean']:.4f} (n={s[r]['ap_70']['n']})" for r in list(METHODS) + ["bare_check"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["prepare", "solve", "score"])
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--method", choices=METHODS, help="solve: the method whose pose graph runs")
    parser.add_argument("--stage1", type=Path, help="solve: stage-1 boxes (default: export_poses_pgo.STAGE1)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--limit", type=int, default=0, help="only the first N conditions (quick check)")
    parser.add_argument("--output", type=Path, default=None, help="score: output json")
    args = parser.parse_args()
    if args.step == "solve" and not args.method:
        parser.error("solve needs --method")
    {"prepare": prepare, "solve": solve, "score": score}[args.step](args)


if __name__ == "__main__":
    main()
