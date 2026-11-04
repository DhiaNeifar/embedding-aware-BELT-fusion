#!/usr/bin/env python3
"""Normalize and accumulate noise-sweep results from all three evaluation
pipelines (our own evaluate-belt-fusion, CoAlign's inference.py, RoCo's
inference.py) into one master table.

Each of the three pipelines reports results in its own format and knows
nothing about the sweep this run belongs to, so every call here is told
explicitly which sweep dimension/value the source file came from. Rows are
appended to outputs/noise_sweep_results/summary.csv; the raw source file is
also copied alongside so a row can always be traced back to exactly what
produced it.

Usage (library):
    from sweep_results import record_pointpillars_pipeline_run, record_coalign_or_roco_run

    record_pointpillars_pipeline_run(
        "outputs/alignformer_clean_test_mAP/metrics.json",
        sweep_dimension="localization_std_m", sweep_value=0.0,
    )
    record_coalign_or_roco_run(
        "coalign", "Zayd Model",
        "models/Zayd Model_clean_test/eval_intermediateclean_test_no_noise_epoch25.yaml",
        sweep_dimension="localization_std_m", sweep_value=0.0,
        association_json="outputs/noise_sweep_results/coalign/assoc_posstd0.0.json",
    )

Usage (CLI), for CoAlign/RoCo eval.yaml + an optional association diagnostics
JSON, since those pipelines don't have a Python caller to invoke the library
form directly:
    python scripts/sweep_results.py coalign "Zayd Model" \
        --eval-yaml path/to/eval_xxx.yaml \
        --sweep-dimension localization_std_m --sweep-value 0.6 \
        [--association-json path/to/assoc.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "outputs" / "noise_sweep_results"
SUMMARY_CSV = RESULTS_ROOT / "summary.csv"

FIELDNAMES = [
    "timestamp",
    "pipeline",
    "method",
    "sweep_dimension",
    "sweep_value",
    "frames",
    "ap_30",
    "ap_50",
    "ap_70",
    "assoc_precision",
    "assoc_recall",
    "assoc_f1",
    "assoc_candidate_pairs",
    "source_file",
]


def _append_row(row: dict) -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    is_new = not SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


def _copy_source(path: Path, pipeline: str, sweep_dimension: str, sweep_value) -> Path:
    dest_dir = RESULTS_ROOT / pipeline
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{sweep_dimension}={sweep_value}_{path.name}"
    if path.resolve() != dest.resolve():
        shutil.copy2(path, dest)
    return dest


def record_pointpillars_pipeline_run(
    metrics_json: str | Path,
    *,
    sweep_dimension: str,
    sweep_value,
) -> list[dict]:
    """Our own evaluate-belt-fusion run: one file, four methods at once
    (naive_late, ego_only, belt_geometry, simple_trackformer==AlignFormer)."""
    metrics_json = Path(metrics_json)
    with metrics_json.open() as f:
        data = json.load(f)
    saved_path = _copy_source(metrics_json, "pointpillars_pipeline", sweep_dimension, sweep_value)

    method_keys = {
        "naive_late": "naive_late_fusion",
        "ego_only": "pointpillars_ego_only",
        "belt_geometry": "belt",
        "simple_trackformer": "alignformer",
    }
    assoc_keys = {
        "belt_geometry": "belt_association",
    }
    timestamp = datetime.now(timezone.utc).isoformat()
    rows = []
    for raw_key, method_name in method_keys.items():
        ap = data.get(raw_key)
        if ap is None:
            continue
        assoc = None
        if raw_key in assoc_keys:
            assoc = data.get(assoc_keys[raw_key])
        elif raw_key == "simple_trackformer":
            assoc = (data.get("component_diagnostics") or {}).get("pair_head")
        row = {
            "timestamp": timestamp,
            "pipeline": "pointpillars_pipeline",
            "method": method_name,
            "sweep_dimension": sweep_dimension,
            "sweep_value": sweep_value,
            "frames": data.get("frames"),
            "ap_30": ap.get("ap_30"),
            "ap_50": ap.get("ap_50"),
            "ap_70": ap.get("ap_70"),
            "assoc_precision": assoc.get("precision") if assoc else "",
            "assoc_recall": assoc.get("recall") if assoc else "",
            "assoc_f1": assoc.get("f1") if assoc else "",
            "assoc_candidate_pairs": assoc.get("candidate_pairs") if assoc else "",
            "source_file": str(saved_path.relative_to(REPO_ROOT)),
        }
        _append_row(row)
        rows.append(row)
    return rows


def record_coalign_or_roco_run(
    pipeline: str,
    method: str,
    eval_yaml: str | Path,
    *,
    sweep_dimension: str,
    sweep_value,
    association_json: str | Path | None = None,
    frames: int | None = None,
) -> dict:
    """A single CoAlign or RoCo inference.py run (one method, one eval.yaml),
    optionally paired with a separately-run association-diagnostics JSON
    (see tools/evaluate_association_diagnostics.py in each submodule)."""
    assert pipeline in ("coalign", "roco"), pipeline
    eval_yaml = Path(eval_yaml)
    with eval_yaml.open() as f:
        ap = yaml.safe_load(f)
    saved_eval = _copy_source(eval_yaml, pipeline, sweep_dimension, sweep_value)

    assoc = None
    if association_json is not None:
        association_json = Path(association_json)
        text = association_json.read_text()
        # Some dataset constructors print a stray "Dataset dir: ..." line to
        # stdout before the diagnostics script's own JSON output; skip to the
        # first '{' instead of assuming the file is pure JSON from byte 0.
        assoc = json.loads(text[text.index("{"):])
        _copy_source(association_json, pipeline, sweep_dimension, sweep_value)
        frames = frames if frames is not None else assoc.get("frames_evaluated")

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline": pipeline,
        "method": method,
        "sweep_dimension": sweep_dimension,
        "sweep_value": sweep_value,
        "frames": frames,
        "ap_30": ap.get("ap30", ap.get("ap_30")),
        "ap_50": ap.get("ap_50"),
        "ap_70": ap.get("ap_70"),
        "assoc_precision": assoc.get("precision") if assoc else "",
        "assoc_recall": assoc.get("recall") if assoc else "",
        "assoc_f1": assoc.get("f1") if assoc else "",
        "assoc_candidate_pairs": assoc.get("candidate_pairs") if assoc else "",
        "source_file": str(saved_eval.relative_to(REPO_ROOT)),
    }
    _append_row(row)
    return row


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline", choices=("coalign", "roco"))
    parser.add_argument("method", help='e.g. "Zayd Model", "RoCo"')
    parser.add_argument("--eval-yaml", required=True)
    parser.add_argument("--sweep-dimension", required=True,
                        choices=("localization_std_m", "heading_std_deg", "comm_delay_ms"))
    parser.add_argument("--sweep-value", required=True, type=float)
    parser.add_argument("--association-json")
    parser.add_argument("--frames", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    opt = _parse_args()
    row = record_coalign_or_roco_run(
        opt.pipeline, opt.method, opt.eval_yaml,
        sweep_dimension=opt.sweep_dimension, sweep_value=opt.sweep_value,
        association_json=opt.association_json, frames=opt.frames,
    )
    print(json.dumps(row, indent=2))
