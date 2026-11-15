#!/usr/bin/env python3
"""One table of the host runs in a folder (scripts/run_host.py outputs): AP per network and pose condition, and the
time a full run would take."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--metric", default="ap_70", choices=["ap_30", "ap_50", "ap_70"])
    parser.add_argument("--full-frames", type=int, default=2170)
    parser.add_argument("--full-conditions", type=int, default=50, help="conditions per frame in the full T1 run")
    args = parser.parse_args()
    runs = [json.loads(p.read_text()) for p in sorted(args.folder.glob("*.json"))]
    if not runs:
        raise SystemExit(f"no results in {args.folder}")
    columns = list(runs[0]["results"])
    print(f"{args.metric}, {runs[0]['frames']} frames per network (a dry run: the values only show the pipeline works)")
    print(f"{'network':12s}" + "".join(f"{c.replace('/sigma_', ' s').replace('/seed_', '/'):>18s}" for c in columns)
          + f"{'missing':>9s}{'full run (h)':>14s}")
    for run in runs:
        r = run["results"]
        missing = sum(v["cavs_without_pose"] for v in r.values())
        hours = run["seconds_per_frame_condition"] * args.full_frames * args.full_conditions / 3600
        print(f"{run['host']:12s}" + "".join(f"{r[c][args.metric]:>18.3f}" if c in r else f"{'-':>18s}" for c in columns)
              + f"{missing:>9d}{hours:>14.1f}")


if __name__ == "__main__":
    main()
