#!/usr/bin/env python3
"""Swap re-run plug-in conditions into the existing host files (scripts/run_host.py outputs).

Used when only some plug-ins change, e.g. CoAlign and RoCo after their OPV2V stage-1 detector was retrained: the
networks are re-run with ``--plugins coalign roco`` into a separate folder, and this script replaces those
plug-ins' rows in the main files. The rows both runs share (``true`` and ``bare``) must be IDENTICAL, which checks
that the two runs saw the same frames, poses and network. Each main file is copied to ``--backup`` before it changes.

    python scripts/merge_host_plugins.py --new outputs/paper/hosts_stage1 --main outputs/paper/hosts \
        --backup outputs/paper/hosts_labmate_stage1 --plugins coalign roco
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--plugins", nargs="+", required=True)
    args = parser.parse_args()

    news = sorted(args.new.glob("*.json"))
    if not news:
        raise SystemExit(f"no results in {args.new}")
    args.backup.mkdir(parents=True, exist_ok=True)
    problems, merged = [], 0
    for new_path in news:
        main_path = args.main / new_path.name
        if not main_path.exists():
            problems.append(f"{main_path} missing")
            continue
        new, main = json.loads(new_path.read_text()), json.loads(main_path.read_text())
        shared = [k for k in new["results"] if not k.startswith(tuple(p + "/" for p in args.plugins))]
        # Compare what both runs report: the APs and the CAVs without a pose. Counters added to run_host.py later
        # (e.g. cavs_placed_out_of_range) are missing from older files and are not a difference.
        def core(row):
            return {key: v for key, v in (row or {}).items() if key.startswith("ap_") or key == "cavs_without_pose"}

        differ = [k for k in shared if core(new["results"][k]) != core(main["results"].get(k))]
        if differ:
            problems.append(f"{new_path.name}: rows that should be identical differ: {differ}")
            continue
        replaced = [k for k in new["results"] if k not in shared]
        if not (args.backup / main_path.name).exists():
            shutil.copy2(main_path, args.backup / main_path.name)
        for k in replaced:
            main["results"][k] = new["results"][k]
        main.setdefault("replaced_from", []).append(str(new_path))
        main_path.write_text(json.dumps(main, indent=1))
        merged += 1
    print(f"merged {merged} of {len(news)} files; originals kept in {args.backup}")
    if problems:
        for p in problems:
            print("NOT MERGED:", p)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
