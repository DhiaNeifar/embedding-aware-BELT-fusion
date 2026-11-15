#!/usr/bin/env python3
"""Audit E3: how much does pooling over time depend on the odometry it is fed? Late fusion, CPU, existing caches.

The full method (alignformer_tk) keeps assuming LOAM accuracy; only the odometry fed to its carry step changes:

    OPV2V     exact recorded motion (the paper's setting), KISS-ICP, and exact motion + 4 injected errors
    V2X-Real  KISS-ICP (the paper's setting), recorded-pose motion, and KISS-ICP + 4 injected errors

Injected errors (alignformer_ext/odometry.PerturbedOdometry): white noise at LOAM level, AR(1) error (0.95, same
stationary level), a 1 % scale bias, and a yaw drift of 0.01 deg/m. sigma 0, 1, 2; seeds 0-4. Each configuration is
one run of scripts/alignformer_variants_sweep.py (its own progress bar) into outputs/paper/audit/e3/; then this script
prints and saves the summary (outputs/paper/audit/e3_summary.json). Configurations already on disk are skipped.

    python scripts/audit_e3_odometry.py --workers 12
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "paper" / "audit" / "e3"
# Extra noise levels for the paper's sigma standard (0, 0.5, 1, 1.5, 2): PAPER_SIGMAS="0.5 1.5" runs only those levels and
# writes to files tagged with PAPER_TAG, so the existing results are never overwritten (scripts/run_paper.sh sigma-std).
import os as _os
PAPER_SIGMAS = [float(x) for x in _os.environ.get("PAPER_SIGMAS", "").split()]
PAPER_TAG = _os.environ.get("PAPER_TAG", "")
if PAPER_TAG:
    OUT = OUT.with_name(f"e3_{PAPER_TAG}")

DATASETS = {
    "opv2v": dict(cache="outputs/propose_cache_opv2v", table="outputs/opv2v_frame_table.pt",
                  checkpoint="models/opv2v/alignformer_boxes_only.pth", config="configs/pointpillar_late_fusion_r140.yaml",
                  range=["140", "40"], geometry="outputs/r140_geometry_test_cache",
                  base_odometry=("exact", "outputs/odometry/opv2v_test.npz"),
                  other_odometry=("kissicp", "outputs/odometry/opv2v_test_kissicp.npz")),
    "v2xreal": dict(cache="outputs/propose_cache_v2xreal", table="outputs/v2xreal_test_frame_table.pt",
                    checkpoint="models/v2xreal/alignformer_boxes_only.pth",
                    config="models/v2xreal/late_fusion/pointpillar/config.yaml",
                    range=["102.4", "51.2"], geometry="outputs/v2xreal_geometry_test_cache",
                    base_odometry=("kissicp", "outputs/odometry/v2xreal_test.npz"),
                    other_odometry=("recorded", "outputs/odometry/v2xreal_test_recorded.npz")),
}
ERRORS = ["white", "ar1", "scale", "yawdrift"]
SIGMAS = [f"{s:g}" for s in PAPER_SIGMAS] or ["0", "1", "2"]


def configurations():
    for dataset, cfg in DATASETS.items():
        base_name, base_path = cfg["base_odometry"]
        other_name, other_path = cfg["other_odometry"]
        yield dataset, base_name, base_path, "none"
        yield dataset, other_name, other_path, "none"
        for error in ERRORS:
            yield dataset, f"{base_name}+{error}", base_path, error


def run(dataset, name, odometry_path, error, workers):
    cfg = DATASETS[dataset]
    output = OUT / f"{dataset}_{name.replace('+', '_')}.json"
    if output.exists():
        print(f"exists, skipping: {output.name}")
        return output
    command = [sys.executable, "scripts/alignformer_variants_sweep.py", "--cache-dir", cfg["cache"],
               "--frame-table", cfg["table"], "--checkpoint", cfg["checkpoint"], "--opencood-config", cfg["config"],
               "--eval-range", *cfg["range"], "--odometry", odometry_path, "--geometry-cache", cfg["geometry"],
               "--odometry-error", error, "--pose-error", "--sigmas", *SIGMAS, "--max-seeds", "5",
               "--variants", "alignformer_c", "alignformer_tk", "--workers", str(workers), "--output", str(output)]
    print(f"\n=== E3 {dataset} odometry {name}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    missing = [p for p in ("outputs/odometry/v2xreal_test_recorded.npz", "outputs/odometry/opv2v_test_kissicp.npz")
               if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(f"missing odometry tables: {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for dataset, name, path, error in configurations():
        data = json.loads(run(dataset, name, path, error, args.workers).read_text())
        rows = {}
        for sigma in SIGMAS:
            s = data["summary"][sigma]
            c, tk = s["all"]["alignformer_c"]["ap_70"]["mean"], s["all"]["alignformer_tk"]["ap_70"]["mean"]
            rows[sigma] = {"ap70_one_frame": c, "ap70_pooled": tk, "pooling_gain": tk - c,
                           "p90_m_one_frame": s["pose_error"]["alignformer_c"]["p90_m"],
                           "p90_m_pooled": s["pose_error"]["alignformer_tk"]["p90_m"],
                           "seeds": s["all"]["alignformer_tk"]["ap_70"]["n"]}
        summary.setdefault(dataset, {})[name] = rows
    (OUT.parent / (f"e3_summary_{PAPER_TAG}.json" if PAPER_TAG else "e3_summary.json")).write_text(json.dumps(summary, indent=1))
    print("\nE3 summary, AP@0.7 (one frame -> pooled, gain) and pose-error p90 in m (one frame -> pooled)")
    for dataset, configs in summary.items():
        print(f"\n{dataset}")
        for name, rows in configs.items():
            cells = "   ".join(f"s{sigma}: {r['ap70_one_frame']:.3f}->{r['ap70_pooled']:.3f} ({r['pooling_gain']:+.3f}) "
                               f"p90 {r['p90_m_one_frame']:.2f}->{r['p90_m_pooled']:.2f}" for sigma, r in rows.items())
            print(f"  {name:18s} {cells}")
    print(f"\nWrote {OUT.parent / 'e3_summary.json'}")


if __name__ == "__main__":
    main()
