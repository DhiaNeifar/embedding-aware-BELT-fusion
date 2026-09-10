#!/usr/bin/env python3
"""Summarize validation AP@0.7 for a TrackFormer gate calibration grid."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    pattern = re.compile(r"d(?P<distance>\d+)_s(?P<similarity>\d+p\d+)")
    for path in sorted(args.metrics_root.glob("d*_s*/metrics.json")):
        match = pattern.fullmatch(path.parent.name)
        if match is None:
            continue
        data = json.loads(path.read_text())
        rows.append(
            {
                "distance_m": int(match["distance"]),
                "minimum_cosine": float(match["similarity"].replace("p", ".")),
                "ap_70": float(data["belt_trackformer"]["ap_70"]),
                "ap_50": float(data["belt_trackformer"]["ap_50"]),
                "groups_per_frame": float(data["mean_trackformer_multi_agent_groups_per_frame"]),
            }
        )
    if not rows:
        raise RuntimeError(f"No calibration metrics found under {args.metrics_root}")
    rows.sort(key=lambda row: row["ap_70"], reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"best": rows[0], "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
