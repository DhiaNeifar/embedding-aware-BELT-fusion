#!/usr/bin/env python3
"""Audit E5: does the decision rule protect poses that are already right? OPV2V late fusion, CPU, existing caches.

OPV2V's poses are exact, so the pose error of every CAV pair is known. At sigma 0, 0.2, 0.4 and 1 (seeds 0-4), per
attempted pair and for five rules:
    full        the one-frame rule as deployed (alignformer_c)
    wald_only   Verify's fit if it engaged, else the proposal, applied only when the Wald test passes
    agreement   the agreement rule only: an engaged fit if it agrees within 1 m or has consensus >= 4, else keep the
                reported pose; when Verify does not engage, the proposal is always applied
    none        no decision: Verify's fit if it engaged, else the proposal
    pooled      the full method over time (alignformer_tk)
    harmful correction  the pose error after the rule is more than 5 cm larger than before it
    false abstention    the rule keeps the reported pose although Verify's fit would cut the error by more than 20 cm
both as shares of attempted pairs, binned by shared objects (0, 1, 2, 3-5, 6+; ego boxes with a CAV box within 2 m
under the true pose). Also the Wald test's false-fire rate at sigma 0 (any correction is wrong there) against its
nominal 0.2, and the NEES of the pooled estimate against its GLS covariance (3 if calibrated) with its 95 % coverage.
Truth is read only to score, after every decision.

    python scripts/audit_e5_decision.py --workers 16
Output: outputs/paper/audit/e5_decision.json
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
from scipy.stats import chi2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_e2_propose as e2  # noqa: E402
import check_verify_core  # noqa: E402
import reproduce_opv2v_sweep as base  # noqa: E402

from embedding_aware_belt_fusion.alignformer_ext import evaluate, odometry, propose_cache, temporal, variants, verify_cpp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "paper" / "audit" / "e5_decision.json"
# Extra noise levels for the paper's sigma standard (0, 0.5, 1, 1.5, 2): PAPER_SIGMAS="0.5 1.5" runs only those levels and
# writes to files tagged with PAPER_TAG, so the existing results are never overwritten (scripts/run_paper.sh sigma-std).
import os as _os
PAPER_SIGMAS = [float(x) for x in _os.environ.get("PAPER_SIGMAS", "").split()]
PAPER_TAG = _os.environ.get("PAPER_TAG", "")

CFG = e2.SPLITS["opv2v_test"]
SIGMAS = PAPER_SIGMAS or [0.0, 0.2, 0.4, 1.0]
if PAPER_TAG:
    OUT = OUT.with_name(f"e5_decision_{PAPER_TAG}.json")
RULES = ["full", "wald_only", "agreement", "none", "pooled"]
HARM_M, GAIN_M = 0.05, 0.20
COVERAGE_95 = float(chi2.ppf(0.95, 3))
_W = {}


def se2(psi, tx, ty) -> np.ndarray:
    m = np.eye(4)
    c, s = np.cos(psi), np.sin(psi)
    m[0, 0], m[0, 1], m[1, 0], m[1, 1], m[0, 3], m[1, 3] = c, -s, s, c, tx, ty
    return m


def condition(job):
    torch.set_num_threads(1)
    sigma, seed = job
    records, counts = _W["records"], _W["counts"]
    with np.load(propose_cache.condition_path(ROOT / CFG["cache"], sigma, seed)) as data:
        cache = {k: data[k] for k in data.files}
    inputs = variants.PairInputs(records, cache, _W["model"], _W["robust"], base=base)
    a = inputs.attempted
    settings = replace(verify_cpp.VerifySettings(), heading_weight=0.0)
    decided, refined, engaged, consensus = verify_cpp.verify(
        inputs.ego_rows, inputs.cav_rows, cache["propose"][a], cache["statistic"][a], cache["dof"][a],
        inputs.model.variance_model, inputs.model.heading_lambda, settings)[:4]
    decided, refined = np.asarray(decided, np.float64), np.asarray(refined, np.float64)
    engaged, consensus = np.asarray(engaged, bool), np.asarray(consensus)
    propose = cache["propose"][a].astype(np.float64)
    wald = verify_cpp.wald_answered(cache["statistic"][a], cache["dof"][a], settings.abstain_level)
    gap = np.abs(np.arctan2(np.sin(refined[:, 0] - propose[:, 0]), np.cos(refined[:, 0] - propose[:, 0])))
    delta = np.hypot(refined[:, 1] - propose[:, 1], refined[:, 2] - propose[:, 2]) + inputs.model.heading_lambda * gap
    accept = engaged & ((delta <= settings.agree_tolerance_m) | (consensus >= settings.consensus_floor))
    candidate = np.where(engaged[:, None], refined, propose)
    zero = np.zeros_like(candidate)
    temporal.DIAGNOSTICS = []
    pooled = variants.alignformer_tk(inputs)[a]
    diagnostics, temporal.DIAGNOSTICS = temporal.DIAGNOSTICS, None
    rules = {"full": decided, "wald_only": np.where(wald[:, None], candidate, zero),
             "agreement": np.where(engaged[:, None], np.where(accept[:, None], refined, zero), propose),
             "none": candidate, "pooled": pooled}

    # pose error before and after, per attempted pair (truth read only here)
    before, truth, bins, reported = [], [], [], []
    for p in a:
        f = int(cache["frame"][p])
        record = records[f]
        cav = sorted(record.cav_boxes)[int(cache["cav_rank"][p])]
        noisy = propose_cache.noisy_poses(base, record, f, sigma, seed)[cav]
        t_rep = base.x1_to_x2(noisy, record.ego_pose)
        t_true = base.x1_to_x2(record.cav_poses[cav], record.ego_pose)
        reported.append(t_rep), truth.append(t_true)
        residual = np.linalg.inv(t_rep) @ t_true
        before.append(np.hypot(residual[0, 3], residual[1, 3]))
        bins.append(e2.bin_of(counts.get((f, str(cav)), 0)))
    before, bins = np.array(before), np.array(bins)

    def error(corrections):
        out = np.empty(len(a))
        for i, (c, t_rep, t_true) in enumerate(zip(corrections, reported, truth)):
            residual = np.linalg.inv(se2(*c) @ t_rep) @ t_true
            out[i] = np.hypot(residual[0, 3], residual[1, 3])
        return out

    verify_error = error(np.where(engaged[:, None], refined, zero))
    rows = {}
    for name, corrections in rules.items():
        after = error(corrections)
        kept = ~np.any(np.abs(corrections) > 1e-9, axis=1)
        harmful = after - before > HARM_M
        false_abstention = kept & engaged & (before - verify_error > GAIN_M)
        rows[name] = {b: {"pairs": int((bins == b).sum()), "harmful": int((harmful & (bins == b)).sum()),
                          "false_abstention": int((false_abstention & (bins == b)).sum())}
                      for b in [x[0] for x in e2.BINS]}
    # NEES of the pooled estimate against its covariance
    nees, nees_applied = [], []
    for p, x, cov, applied in diagnostics:
        f = int(cache["frame"][p])
        record = records[f]
        cav = sorted(record.cav_boxes)[int(cache["cav_rank"][p])]
        t = base.x1_to_x2(record.cav_poses[cav], record.ego_pose)
        e = x - np.array([t[0, 3], t[1, 3], np.arctan2(t[1, 0], t[0, 0])])
        e[2] = np.arctan2(np.sin(e[2]), np.cos(e[2]))
        value = float(e @ np.linalg.solve(cov, e))
        nees.append(value)
        if applied:
            nees_applied.append(value)
    return sigma, seed, {"rules": rows, "wald_pass": int(wald.sum()), "wald_pass_not_engaged": int((wald & ~engaged).sum()),
                         "not_engaged": int((~engaged).sum()), "attempted": int(len(a)),
                         "nees": nees, "nees_applied": nees_applied}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    base.fusion_core.set_filters(CFG["range"][0], CFG["range"][1], -3.0, 1.0, 6.0)
    records = check_verify_core.load_frame_table(ROOT / CFG["table"])
    odometry.ACTIVE = odometry.OdometryTable.for_records(ROOT / CFG["odometry"], ROOT / CFG["geometry"], records)
    _W.update(records=records, counts=e2.shared_counts(records),
              model=base.load_alignformer_pva(ROOT / CFG["checkpoint"], torch.device("cpu"))[0],
              robust=base.RobustSolveConfig(mode="huber", iterations=2))
    jobs = [(s, k) for s in SIGMAS for k in ([0] if s == 0.0 else range(args.seeds))]
    merged = {}
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
        futures = [pool.submit(condition, j) for j in jobs]
        for future in tqdm(cf.as_completed(futures), total=len(futures), desc="E5 OPV2V (conditions)"):
            sigma, _, out = future.result()
            entry = merged.setdefault(f"{sigma:g}", {"rules": {}, "wald_pass": 0, "wald_pass_not_engaged": 0,
                                                     "not_engaged": 0, "attempted": 0, "nees": [], "nees_applied": []})
            for key in ("wald_pass", "wald_pass_not_engaged", "not_engaged", "attempted"):
                entry[key] += out[key]
            entry["nees"] += out["nees"]
            entry["nees_applied"] += out["nees_applied"]
            for rule, per_bin in out["rules"].items():
                for b, v in per_bin.items():
                    slot = entry["rules"].setdefault(rule, {}).setdefault(b, {"pairs": 0, "harmful": 0, "false_abstention": 0})
                    for k in slot:
                        slot[k] += v[k]
    report = {}
    for sigma, e in merged.items():
        rules = {}
        for rule, per_bin in e["rules"].items():
            total = {k: sum(v[k] for v in per_bin.values()) for k in ("pairs", "harmful", "false_abstention")}
            rules[rule] = {b: {"pairs": v["pairs"], "harmful": v["harmful"] / max(v["pairs"], 1),
                               "false_abstention": v["false_abstention"] / max(v["pairs"], 1)}
                           for b, v in list(per_bin.items()) + [("all", total)]}
        nees, applied = np.array(e["nees"]), np.array(e["nees_applied"])
        report[sigma] = {
            "rules": rules, "attempted_pairs": e["attempted"],
            "wald_fire_rate": e["wald_pass"] / max(e["attempted"], 1),
            "wald_fire_rate_not_engaged": e["wald_pass_not_engaged"] / max(e["not_engaged"], 1),
            "nees_mean": float(nees.mean()) if len(nees) else None,
            "nees_median": float(np.median(nees)) if len(nees) else None,
            "coverage95": float(np.mean(nees <= COVERAGE_95)) if len(nees) else None,
            "nees_mean_applied": float(applied.mean()) if len(applied) else None,
            "coverage95_applied": float(np.mean(applied <= COVERAGE_95)) if len(applied) else None,
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print("\nHarmful-correction rate / false-abstention rate, all pairs (OPV2V)")
    for sigma in sorted(report, key=float):
        r = report[sigma]
        print(f"  sigma {sigma:>3}: " + "  ".join(f"{rule} {r['rules'][rule]['all']['harmful']:.3f}/"
                                              f"{r['rules'][rule]['all']['false_abstention']:.3f}" for rule in RULES))
    print("\nFull rule, harmful-correction rate by shared objects")
    for sigma in sorted(report, key=float):
        r = report[sigma]["rules"]["full"]
        print(f"  sigma {sigma:>3}: " + "  ".join(f"{b}: {v['harmful']:.3f} (n {v['pairs']})" for b, v in r.items()))
    print("\nWald test and pooled-estimate calibration")
    for sigma in sorted(report, key=float):
        r = report[sigma]
        print(f"  sigma {sigma:>3}: Wald fires {r['wald_fire_rate']:.3f} of pairs, {r['wald_fire_rate_not_engaged']:.3f} "
              f"when Verify is not engaged (nominal 0.2 at sigma 0) | NEES mean {r['nees_mean']:.2f} median "
              f"{r['nees_median']:.2f} (3 if calibrated), 95 % coverage {r['coverage95']:.3f}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
