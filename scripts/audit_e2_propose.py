#!/usr/bin/env python3
"""Audit E2: is the learned Propose stage needed? Late fusion, CPU, existing Propose caches.

Variants (alignformer_ext/e2_baselines.py and variants.py), all read only boxes under the reported pose:
    alignformer_c, alignformer_tk                 the method, one frame and pooled
    verify_from_reported, verify_from_reported_tk  Verify from the reported pose, no Propose
    hungarian_ransac, hungarian_ransac_tk          Hungarian on centres (6 m cap) + RANSAC + least squares
    propose_hard                                   Propose's argmax assignment + least squares on centres
Splits: OPV2V test and V2X-Real test at sigma 0, 0.4, 1, 2; V2X-Real validation at 0, 0.4, 1 (its cache has no 2);
seeds 0-4. Also the OPV2V checkpoint evaluated as is on V2X-Real test (alignformer_c and _tk only); its Propose
cache is built first if missing (scripts/build_propose_cache.py, its own progress bar).

Per variant: AP@0.7 and AP@0.5, pose-error median and p90 against the recorded poses, and the share of CAV pairs
answered, all binned by shared objects: the number of ego boxes with a CAV box within 2 m under the TRUE pose
(0, 1, 2, 3-5, 6+). A pair's bin is its own count; a frame's bin (for AP) is the smallest count among its CAVs.
The truth is read only for these bins and the pose error, after every decision is made.

    python scripts/audit_e2_propose.py --workers 16
Output: outputs/paper/audit/e2/<split>.json and outputs/paper/audit/e2_summary.json.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_verify_core  # noqa: E402
import pose_accuracy  # noqa: E402
import reproduce_opv2v_sweep as base  # noqa: E402

from embedding_aware_belt_fusion.alignformer_ext import evaluate, odometry, poses, variants  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "paper" / "audit" / "e2"
# Extra noise levels for the paper's sigma standard (0, 0.5, 1, 1.5, 2): PAPER_SIGMAS="0.5 1.5" runs only those levels and
# writes to files tagged with PAPER_TAG, so the existing results are never overwritten (scripts/run_paper.sh sigma-std).
import os as _os
PAPER_SIGMAS = [float(x) for x in _os.environ.get("PAPER_SIGMAS", "").split()]
PAPER_TAG = _os.environ.get("PAPER_TAG", "")
if PAPER_TAG:
    OUT = OUT.with_name(f"e2_{PAPER_TAG}")

ALL = ["alignformer_c", "alignformer_tk", "verify_from_reported", "verify_from_reported_tk",
       "hungarian_ransac", "hungarian_ransac_tk", "propose_hard"]
V2X = dict(table="outputs/v2xreal_test_frame_table.pt", geometry="outputs/v2xreal_geometry_test_cache",
           odometry="outputs/odometry/v2xreal_test.npz", config="models/v2xreal/late_fusion/pointpillar/config.yaml",
           range=(102.4, 51.2), fa="models/v2xreal/freealign_calibration.json")
SPLITS = {
    "opv2v_test": dict(cache="outputs/propose_cache_opv2v", table="outputs/opv2v_frame_table.pt",
                       geometry="outputs/r140_geometry_test_cache", odometry="outputs/odometry/opv2v_test.npz",
                       config="configs/pointpillar_late_fusion_r140.yaml", range=(140.0, 40.0),
                       checkpoint="models/opv2v/alignformer_boxes_only.pth", sigmas=[0, 0.4, 1, 2], variants=ALL),
    "v2xreal_test": dict(V2X, cache="outputs/propose_cache_v2xreal", checkpoint="models/v2xreal/alignformer_boxes_only.pth",
                         sigmas=[0, 0.4, 1, 2], variants=ALL),
    "v2xreal_val": dict(V2X, cache="outputs/propose_cache_v2xreal_val", table="outputs/v2xreal_val_frame_table.pt",
                        geometry="outputs/v2xreal_geometry_val_cache", odometry="outputs/odometry/v2xreal_val.npz",
                        checkpoint="models/v2xreal/alignformer_boxes_only.pth", sigmas=[0, 0.4, 1], variants=ALL),
    "v2xreal_test_opv2v_checkpoint": dict(V2X, cache="outputs/propose_cache_v2xreal_opv2v_checkpoint",
                                          checkpoint="models/opv2v/alignformer_boxes_only.pth",
                                          sigmas=[0, 0.4, 1, 2], variants=["alignformer_c", "alignformer_tk"]),
}
if PAPER_SIGMAS:   # the tagged run covers only the new levels, on every split that has them in its cache
    for _cfg in SPLITS.values():
        _cfg["sigmas"] = list(PAPER_SIGMAS)
BINS = [("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5), ("6+", 6, 10 ** 6)]
_W = {}


def bin_of(count: int) -> str:
    return next(name for name, lo, hi in BINS if lo <= count <= hi)


def shared_counts(records) -> dict:
    """(frame, cav id) -> ego boxes with a CAV box within 2 m under the true pose (evaluation only)."""
    out = {}
    for f, record in enumerate(records):
        for cav in record.cav_boxes:
            if cav in record.cav_poses:
                truth = base.x1_to_x2(record.cav_poses[cav], record.ego_pose)
                out[(f, str(cav))] = pose_accuracy.shared_objects(record, cav, truth)
    return out


def run(path: Path):
    torch.set_num_threads(1)
    with np.load(path) as data:
        cache = {k: data[k] for k in data.files}
    records, nms, counts = _W["records"], _W["nms"], _W["counts"]
    cav_rank = cache["cav_rank"]
    frame_bin = {}
    for p, f in enumerate(cache["frame"]):
        cav = sorted(records[int(f)].cav_boxes)[int(cav_rank[p])]
        c = counts.get((int(f), str(cav)))
        if c is not None:
            frame_bin[int(f)] = min(frame_bin.get(int(f), 10 ** 6), c)
    frames_in = {name: {f for f, c in frame_bin.items() if bin_of(c) == name} for name, _, _ in BINS}

    def ap(decided=None, **kw):
        table = {"all": evaluate.ap_table(base, evaluate.score_condition(base, records, cache, decided, nms, **kw))["all"]}
        for name, frames in frames_in.items():
            stats = evaluate.score_condition(base, records, cache, decided, nms, frames=frames, **kw)
            table[name] = evaluate.ap_table(base, stats)["all"] if frames else None
        return table

    result = {"bare": ap(), "freealign": ap(freealign=True)}
    if float(cache["sigma"]) == 0.0:
        result["reference"] = ap(oracle=True)
    inputs = variants.PairInputs(records, cache, _W["model"], _W["robust"], base=base)
    for name in _W["variants"]:
        decided = variants.VARIANTS[name](inputs)
        decided = decided[0] if isinstance(decided, tuple) else decided
        entry = {"ap": ap(decided)}
        arrays = poses.build(base, records, odometry.ACTIVE.frames, cache, decided)
        per_bin = {b: {"t": [], "answered": []} for b, _, _ in BINS}
        for f, cav, transform, correction in zip(arrays["frame_index"], arrays["cav_id"], arrays["transform"],
                                                 arrays["correction"]):
            c = counts.get((int(f), str(cav)))
            if c is None:
                continue
            record = records[int(f)]
            residual = np.linalg.inv(transform) @ base.x1_to_x2(record.cav_poses[str(cav)], record.ego_pose)
            per_bin[bin_of(c)]["t"].append(float(np.hypot(residual[0, 3], residual[1, 3])))
            per_bin[bin_of(c)]["answered"].append(bool(np.any(np.abs(correction) > 1e-9)))
        entry["pose"] = {}
        everything = {"t": [], "answered": []}
        for b, values in list(per_bin.items()):
            everything["t"] += values["t"]
            everything["answered"] += values["answered"]
        for b, values in list(per_bin.items()) + [("all", everything)]:
            t = np.array(values["t"])
            entry["pose"][b] = None if not len(t) else {
                "pairs": int(len(t)), "median_m": float(np.median(t)), "p90_m": float(np.percentile(t, 90)),
                "answered": float(np.mean(values["answered"]))}
        result[name] = entry
    return float(cache["sigma"]), int(cache["seed"]), result


def build_checkpoint_cache(workers: int) -> None:
    cfg = SPLITS["v2xreal_test_opv2v_checkpoint"]
    command = [sys.executable, "scripts/build_propose_cache.py", "--frame-table", cfg["table"], "--checkpoint",
               cfg["checkpoint"], "--freealign-calibration", cfg["fa"], "--sigmas", *[f"{x:g}" for x in (PAPER_SIGMAS or [0, 0.4, 1, 2])], "--seeds", "5",
               "--output-dir", cfg["cache"], "--workers", str(workers)]
    print("\n=== Propose cache: OPV2V checkpoint on V2X-Real test", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def summarize(results: dict, names: list) -> dict:
    summary = {}
    for sigma, per_seed in results.items():
        rows = {}
        for name in ["bare", "freealign", "reference"] + names:
            entries = [s[name] for s in per_seed if name in s]
            if not entries:
                continue
            ap_part = entries[0]["ap"] if "ap" in entries[0] else entries[0]
            row = {}
            for b in ap_part:
                values = [(e["ap"] if "ap" in e else e)[b] for e in entries]
                values = [v for v in values if v is not None]
                if values:
                    row[f"ap70_{b}"] = float(np.mean([v["ap_70"] for v in values]))
                    row[f"ap50_{b}"] = float(np.mean([v["ap_50"] for v in values]))
            if "pose" in entries[0]:
                for b in entries[0]["pose"]:
                    values = [e["pose"][b] for e in entries if e["pose"][b] is not None]
                    if values:
                        for key in ("median_m", "p90_m", "answered"):
                            row[f"{key}_{b}"] = float(np.mean([v[key] for v in values]))
                        row[f"pairs_{b}"] = int(np.mean([v["pairs"] for v in values]))
            rows[name] = row
        summary[sigma] = {"seeds": len(per_seed), "rows": rows}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    everything = {}
    for split in args.splits:
        cfg = SPLITS[split]
        output = OUT / f"{split}.json"
        if output.exists():
            print(f"exists, skipping: {output.name}")
            everything[split] = json.loads(output.read_text())["summary"]
            continue
        if split == "v2xreal_test_opv2v_checkpoint" and (PAPER_SIGMAS or not any((ROOT / cfg["cache"]).glob("sigma_*.npz"))):
            build_checkpoint_cache(args.workers)
        paths = [p for p in sorted((ROOT / cfg["cache"]).glob("sigma_*_seed_*.npz"))
                 if float(p.stem.split("_")[1]) in cfg["sigmas"] and int(p.stem.split("_")[3]) < args.seeds]
        base.fusion_core.set_filters(cfg["range"][0], cfg["range"][1], -3.0, 1.0, 6.0)
        records = check_verify_core.load_frame_table(ROOT / cfg["table"])
        odometry.ACTIVE = odometry.OdometryTable.for_records(ROOT / cfg["odometry"], ROOT / cfg["geometry"], records)
        _W.update(records=records, counts=shared_counts(records), variants=cfg["variants"],
                  model=base.load_alignformer_pva(ROOT / cfg["checkpoint"], torch.device("cpu"))[0],
                  robust=base.RobustSolveConfig(mode="huber", iterations=2),
                  nms=float(base.load_yaml(str(ROOT / cfg["config"]), None)["postprocess"]["nms_thresh"]))
        results = {}
        with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
            futures = [pool.submit(run, path) for path in paths]
            for future in tqdm(cf.as_completed(futures), total=len(futures), desc=f"E2 {split} (conditions)"):
                sigma, seed, result = future.result()
                results.setdefault(f"{sigma:g}", []).append({"seed": seed, **result})
        summary = summarize(results, cfg["variants"])
        output.write_text(json.dumps({"split": split, "summary": summary, "per_seed": results}, indent=1))
        everything[split] = summary
        print(f"wrote {output}")
    (OUT.parent / (f"e2_summary_{PAPER_TAG}.json" if PAPER_TAG else "e2_summary.json")).write_text(json.dumps(everything, indent=1))
    for split, summary in everything.items():
        print(f"\n{split}: AP@0.7 all frames | pose p90 m | answered share")
        for sigma in sorted(summary, key=float):
            rows = summary[sigma]["rows"]
            print(f"  sigma {sigma}: " + "  ".join(
                f"{n}: {r.get('ap70_all', float('nan')):.3f}|{r.get('p90_m_all', float('nan')):.2f}|"
                f"{r.get('answered_all', float('nan')):.2f}" for n, r in rows.items() if n not in ("bare",)))
    print(f"\nWrote {OUT.parent / 'e2_summary.json'}")


if __name__ == "__main__":
    main()
