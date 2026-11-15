#!/usr/bin/env python3
"""Audit E0: paired bootstrap confidence intervals and the decision rule's branch frequencies. CPU, minutes.

1. Late fusion, AlignFormer minus FreeAlign, CoAlign and RoCo, AP@0.7 and AP@0.5, sigma 0, 0.4, 1, 2, both datasets.
   Every method is scored per frame from the exported poses (the late_*_plugins_20seeds.json runs), seeds 0-4; one
   bootstrap resample draws frames with replacement, the SAME frames for every method and seed (paired), and each
   method's AP is its mean over seeds on that resample. 95 % interval from 1000 resamples.
2. V2X-Real sigma 0, one draw: AlignFormer (alignformer_tk) minus the recorded (reference) pose, on test (0.416 vs
   0.407) and on validation (0.415 vs 0.427), 2000 resamples.
3. Branch frequencies of the per-frame decision rule (alignformer_c), per sigma and dataset, shares of attempted pairs:
   Verify engaged and the fits agree within 1 m; engaged, disagree, consensus >= 4; engaged and rejected; not engaged
   and the Wald test passes; not engaged and it fails. The branches are recomputed from Verify's outputs and checked
   against the decisions the C++ rule actually made.
Intermediate-fusion networks: per-frame detections were not saved by those runs, so they get no interval here.

    python scripts/audit_e0_bootstrap.py --workers 16
Output: outputs/paper/audit/e0_bootstrap.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_verify_core  # noqa: E402
import reproduce_opv2v_sweep as base  # noqa: E402

from embedding_aware_belt_fusion.alignformer_ext import evaluate, odometry, poses, propose_cache, variants, verify_cpp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "paper" / "audit" / "e0_bootstrap.json"
# Extra noise levels for the paper's sigma standard (0, 0.5, 1, 1.5, 2): PAPER_SIGMAS="0.5 1.5" runs only those levels and
# writes to files tagged with PAPER_TAG, so the existing results are never overwritten (scripts/run_paper.sh sigma-std).
import os as _os
PAPER_SIGMAS = [float(x) for x in _os.environ.get("PAPER_SIGMAS", "").split()]
PAPER_TAG = _os.environ.get("PAPER_TAG", "")

PLUGINS = ["freealign", "coalign", "roco", "alignformer"]
DATASETS = {
    "opv2v": dict(cache="outputs/propose_cache_opv2v", table="outputs/opv2v_frame_table.pt",
                  config="configs/pointpillar_late_fusion_r140.yaml", range=(140.0, 40.0),
                  checkpoint="models/opv2v/alignformer_boxes_only.pth", geometry="outputs/r140_geometry_test_cache",
                  odometry="outputs/odometry/opv2v_test.npz"),
    "v2xreal": dict(cache="outputs/propose_cache_v2xreal", table="outputs/v2xreal_test_frame_table.pt",
                    config="models/v2xreal/late_fusion/pointpillar/config.yaml", range=(102.4, 51.2),
                    checkpoint="models/v2xreal/alignformer_boxes_only.pth", geometry="outputs/v2xreal_geometry_test_cache",
                    odometry="outputs/odometry/v2xreal_test.npz"),
    "v2xreal_val": dict(cache="outputs/propose_cache_v2xreal_val", table="outputs/v2xreal_val_frame_table.pt",
                        config="models/v2xreal/late_fusion/pointpillar/config.yaml", range=(102.4, 51.2),
                        checkpoint="models/v2xreal/alignformer_boxes_only.pth",
                        geometry="outputs/v2xreal_geometry_val_cache", odometry="outputs/odometry/v2xreal_val.npz"),
}
SIGMAS = PAPER_SIGMAS or [0.0, 0.4, 1.0, 2.0]
if PAPER_TAG:
    OUT = OUT.with_name(f"e0_bootstrap_{PAPER_TAG}.json")
THRESHOLDS = (0.5, 0.7)
_W = {}


# ------------------------------------------------------------------------------------------------- per-frame stats
def frame_arrays(per_frame: dict, n_frames: int) -> dict:
    """threshold -> (tp, fp, score concatenated over frames, per-frame offsets, per-frame gt counts)."""
    out = {}
    for t in THRESHOLDS:
        tp, fp, score, off, gt = [], [], [], [0], []
        for f in range(n_frames):
            s = per_frame.get(f)
            if s is None:
                off.append(off[-1])
                gt.append(0)
                continue
            tp += list(s[t]["tp"])
            fp += list(s[t]["fp"])
            score += list(s[t]["score"])
            off.append(off[-1] + len(s[t]["score"]))
            gt.append(s[t]["gt"])
        out[t] = (np.asarray(tp, np.float32), np.asarray(fp, np.float32), np.asarray(score, np.float32),
                  np.asarray(off, np.int64), np.asarray(gt, np.int64))
    return out


def ap_of(arrays, frames: np.ndarray) -> float:
    tp, fp, score, off, gt = arrays
    counts = np.bincount(frames, minlength=len(gt))
    repeat = np.repeat(counts, np.diff(off))                  # each detection repeated as often as its frame is drawn
    keep = repeat > 0
    stat = {"tp": np.repeat(tp[keep], repeat[keep]), "fp": np.repeat(fp[keep], repeat[keep]),
            "score": np.repeat(score[keep], repeat[keep]), "gt": int((counts * gt).sum())}
    return base.calculate_ap(stat)


def pair_index(records, cache):
    pair_of = {}
    for p in range(len(cache["frame"])):
        f = int(cache["frame"][p])
        pair_of[(f, sorted(records[f].cav_boxes)[int(cache["cav_rank"][p])])] = p
    return pair_of


def plugin_condition(job):
    """Per-frame stats of every plug-in at one (sigma, seed), from the exported poses, like late_from_poses.py."""
    torch.set_num_threads(1)
    dataset, sigma, seed = job
    records, nms = _W["records"], _W["nms"]
    with np.load(propose_cache.condition_path(ROOT / DATASETS[dataset]["cache"], sigma, seed)) as data:
        cache = {k: data[k] for k in data.files}
    pair_of = pair_index(records, cache)
    out = {}
    for plugin in PLUGINS:
        with np.load(ROOT / "outputs" / "poses" / dataset / plugin / f"sigma_{sigma:g}_seed_{seed}.npz") as z:
            decided = np.zeros((len(cache["frame"]), 3), np.float32)
            for f, cav, c in zip(z["frame_index"], z["cav_id"], z["correction"]):
                p = pair_of.get((int(f), str(cav)))
                if p is not None:
                    decided[p] = c
        per_frame = {}
        evaluate.score_condition(base, records, cache, decided, nms, per_frame=per_frame)
        out[plugin] = frame_arrays(per_frame, len(records))
    return dataset, sigma, seed, out


def bootstrap_plugins(dataset: str, sigma: float, per_seed: list, resamples: int, seed: int = 0) -> dict:
    n = len(per_seed[0]["alignformer"][0.7][4])
    rng = np.random.default_rng(seed)
    draws = {t: {other: [] for other in PLUGINS[:-1]} for t in THRESHOLDS}
    point = {t: {p: float(np.mean([ap_of(s[p][t], np.arange(n)) for s in per_seed])) for p in PLUGINS} for t in THRESHOLDS}
    for _ in range(resamples):
        frames = rng.integers(0, n, n)
        for t in THRESHOLDS:
            ours = np.mean([ap_of(s["alignformer"][t], frames) for s in per_seed])
            for other in PLUGINS[:-1]:
                draws[t][other].append(ours - np.mean([ap_of(s[other][t], frames) for s in per_seed]))
    result = {}
    for t in THRESHOLDS:
        for other in PLUGINS[:-1]:
            d = np.array(draws[t][other])
            result[f"ap{int(t * 100)}_alignformer_minus_{other}"] = {
                "difference": point[t]["alignformer"] - point[t][other],
                "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    result["point"] = {f"ap{int(t * 100)}": point[t] for t in THRESHOLDS}
    result["seeds"] = len(per_seed)
    return result


# ------------------------------------------------------------------------------- sigma 0 against the reference pose
def reference_difference(dataset: str, resamples: int) -> dict:
    cfg = DATASETS[dataset]
    base.fusion_core.set_filters(cfg["range"][0], cfg["range"][1], -3.0, 1.0, 6.0)
    records = check_verify_core.load_frame_table(ROOT / cfg["table"])
    odometry.ACTIVE = odometry.OdometryTable.for_records(ROOT / cfg["odometry"], ROOT / cfg["geometry"], records)
    nms = float(base.load_yaml(str(ROOT / cfg["config"]), None)["postprocess"]["nms_thresh"])
    with np.load(propose_cache.condition_path(ROOT / cfg["cache"], 0.0, 0)) as data:
        cache = {k: data[k] for k in data.files}
    model = base.load_alignformer_pva(ROOT / cfg["checkpoint"], torch.device("cpu"))[0]
    inputs = variants.PairInputs(records, cache, model, base.RobustSolveConfig(mode="huber", iterations=2), base=base)
    ours, ref = {}, {}
    evaluate.score_condition(base, records, cache, variants.alignformer_tk(inputs), nms, per_frame=ours)
    evaluate.score_condition(base, records, cache, None, nms, oracle=True, per_frame=ref)
    a, b = frame_arrays(ours, len(records)), frame_arrays(ref, len(records))
    n, rng = len(records), np.random.default_rng(1)
    out = {}
    for t in THRESHOLDS:
        draws = []
        for _ in tqdm(range(resamples), desc=f"E0 {dataset} sigma 0 bootstrap AP@{t}", leave=False):
            frames = rng.integers(0, n, n)
            draws.append(ap_of(a[t], frames) - ap_of(b[t], frames))
        full = np.arange(n)
        out[f"ap{int(t * 100)}"] = {"alignformer": ap_of(a[t], full), "reference": ap_of(b[t], full),
                                    "difference": ap_of(a[t], full) - ap_of(b[t], full),
                                    "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}
    return out


# --------------------------------------------------------------------------------------------- decision branches
def branch_condition(job):
    torch.set_num_threads(1)
    dataset, sigma, seed = job
    with np.load(propose_cache.condition_path(ROOT / DATASETS[dataset]["cache"], sigma, seed)) as data:
        cache = {k: data[k] for k in data.files}
    inputs = variants.PairInputs(_W["records"], cache, _W["model"], _W["robust"], base=base)
    a = inputs.attempted
    settings = replace(verify_cpp.VerifySettings(), heading_weight=0.0)          # alignformer_c's settings
    decided, refined, engaged, consensus = verify_cpp.verify(
        inputs.ego_rows, inputs.cav_rows, cache["propose"][a], cache["statistic"][a], cache["dof"][a],
        inputs.model.variance_model, inputs.model.heading_lambda, settings)[:4]
    refined, engaged, consensus = np.asarray(refined), np.asarray(engaged, bool), np.asarray(consensus)
    propose = cache["propose"][a]
    gap = np.abs(np.arctan2(np.sin(refined[:, 0] - propose[:, 0]), np.cos(refined[:, 0] - propose[:, 0])))
    delta = np.hypot(refined[:, 1] - propose[:, 1], refined[:, 2] - propose[:, 2]) + inputs.model.heading_lambda * gap
    wald = verify_cpp.wald_answered(cache["statistic"][a], cache["dof"][a], settings.abstain_level)
    agree = engaged & (delta <= settings.agree_tolerance_m)
    floor = engaged & ~agree & (consensus >= settings.consensus_floor)
    branches = {"engaged_agree": agree, "engaged_consensus": floor, "engaged_rejected": engaged & ~agree & ~floor,
                "not_engaged_wald_pass": ~engaged & wald, "not_engaged_wald_fail": ~engaged & ~wald}
    applied = np.any(np.asarray(decided) != 0, axis=1)
    expected = agree | floor | (~engaged & wald)
    return dataset, sigma, seed, {k: int(v.sum()) for k, v in branches.items()}, int(len(a)), float(np.mean(applied != expected))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    report = {"plugins": {}, "reference_sigma0": {}, "branches": {}}

    for dataset in ("opv2v", "v2xreal"):
        cfg = DATASETS[dataset]
        base.fusion_core.set_filters(cfg["range"][0], cfg["range"][1], -3.0, 1.0, 6.0)
        records = check_verify_core.load_frame_table(ROOT / cfg["table"])
        _W.update(records=records, nms=float(base.load_yaml(str(ROOT / cfg["config"]), None)["postprocess"]["nms_thresh"]),
                  model=base.load_alignformer_pva(ROOT / cfg["checkpoint"], torch.device("cpu"))[0],
                  robust=base.RobustSolveConfig(mode="huber", iterations=2))
        jobs = [(dataset, s, k) for s in SIGMAS for k in ([0] if s == 0.0 else range(args.seeds))]
        per_sigma = {}
        with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
            futures = [pool.submit(plugin_condition, j) for j in jobs]
            for future in tqdm(cf.as_completed(futures), total=len(futures), desc=f"E0 {dataset} per-frame stats"):
                _, sigma, _, out = future.result()
                per_sigma.setdefault(f"{sigma:g}", []).append(out)
        with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
            futures = {pool.submit(bootstrap_plugins, dataset, float(s), per_seed, args.resamples): s
                       for s, per_seed in per_sigma.items()}
            for future in tqdm(cf.as_completed(futures), total=len(futures), desc=f"E0 {dataset} bootstrap"):
                report["plugins"].setdefault(dataset, {})[futures[future]] = future.result()
        branches = {}
        with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
            for future in tqdm(cf.as_completed([pool.submit(branch_condition, j) for j in jobs]), total=len(jobs),
                               desc=f"E0 {dataset} decision branches"):
                _, sigma, _, counts, attempted, mismatch = future.result()
                entry = branches.setdefault(f"{sigma:g}", {"attempted": 0, "mismatch": [], **{k: 0 for k in counts}})
                entry["attempted"] += attempted
                entry["mismatch"].append(mismatch)
                for k, v in counts.items():
                    entry[k] += v
        report["branches"][dataset] = {
            s: {**{k: e[k] / max(e["attempted"], 1) for k in e if k not in ("attempted", "mismatch")},
                "attempted_pairs": e["attempted"], "rule_mismatch": float(np.mean(e["mismatch"]))}
            for s, e in branches.items()}

    for dataset in ("v2xreal", "v2xreal_val"):
        report["reference_sigma0"][dataset] = reference_difference(dataset, 2 * args.resamples)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print("\nLate fusion, AlignFormer minus each plug-in, AP@0.7 [95 % CI over frames]")
    for dataset, rows in report["plugins"].items():
        for sigma in sorted(rows, key=float):
            r = rows[sigma]
            print(f"  {dataset:8s} sigma {sigma:>3}: " + "  ".join(
                f"{o}: {r[f'ap70_alignformer_minus_{o}']['difference']:+.3f} "
                f"[{r[f'ap70_alignformer_minus_{o}']['ci95'][0]:+.3f}, {r[f'ap70_alignformer_minus_{o}']['ci95'][1]:+.3f}]"
                for o in PLUGINS[:-1]))
    print("\nV2X-Real sigma 0, AlignFormer minus the recorded pose, AP@0.7 [95 % CI]")
    for dataset, r in report["reference_sigma0"].items():
        x = r["ap70"]
        print(f"  {dataset:12s} {x['alignformer']:.3f} - {x['reference']:.3f} = {x['difference']:+.3f} "
              f"[{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]")
    print("\nDecision branches, share of attempted pairs (rule_mismatch should be 0)")
    for dataset, rows in report["branches"].items():
        for sigma in sorted(rows, key=float):
            r = rows[sigma]
            print(f"  {dataset:8s} sigma {sigma:>3}: " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
