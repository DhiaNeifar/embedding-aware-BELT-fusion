#!/usr/bin/env python3
"""Reproduce each V2X-Real network's own reported AP with jigsawcomm-v2x's own evaluator, on our copy of the data.

For every network: copy its config into a scratch model folder (only the data path and the modality file are
changed), link its checkpoint, run jigsawcomm-v2x's opencood/tools/inference_baselines.py, and compare the AP it
writes with the network's eval_baseline_epoch*.yaml from training. A match means code, weights and data agree, so
any later difference comes from our protocol, not from the setup. Small gaps can come from the frames Chenyi's
cleaned copy quarantined.

    cd external/jigsawcomm-v2x && \
      /mnt/external/backup/dhia/conda_envs/coalign-blackwell/bin/python ../../scripts/check_v2xreal_clean_ap.py \
      attfuse fcooper v2vam v2xvit where2comm
"""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CODE = REPO / "external" / "jigsawcomm-v2x"
DATA = Path("/mnt/external/workspace/public/dataset/v2x-real")
HOSTS = REPO / "models" / "v2xreal" / "intermediate_fusion"
WORK = REPO / "outputs" / "check_v2xreal_clean_ap"


def write_assignment(path: Path) -> Path:
    """jigsawcomm's modality file: every agent is a LiDAR agent (m1)."""
    table = {}
    for split in ("train", "val", "test"):
        for scenario in sorted(os.listdir(DATA / split)):
            folder = DATA / split / scenario
            if folder.is_dir():
                table[scenario] = {c: "m1" for c in os.listdir(folder) if (folder / c).is_dir()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table))
    return path


def prepare(name: str, assignment: Path) -> Path:
    source = HOSTS / name
    folder = WORK / name
    folder.mkdir(parents=True, exist_ok=True)
    text = (source / "config.yaml").read_text()
    text = text.replace("/workspace/nvme/chenyiw/datasets/v2x-real-clean", str(DATA))
    text = text.replace("/home/chenyi/datasets/v2x-real-clean", str(DATA))
    text = text.replace("opencood/logs/heter_modality_assign/v2xreal_lidaronly.json", str(assignment))
    (folder / "config.yaml").write_text(text)
    for checkpoint in source.glob("net_epoch*.pth"):
        link = folder / checkpoint.name
        if not link.exists():
            link.symlink_to(checkpoint)
    return folder


def reported(name: str) -> dict:
    path = sorted((HOSTS / name).glob("eval_baseline_epoch*.yaml"))[-1]
    data = yaml.safe_load(path.read_text())
    return {"ap_50": data.get("ap_50"), "ap_70": data.get("ap_70")}


def main() -> None:
    names = sys.argv[1:] or ["attfuse", "fcooper", "v2vam", "v2xvit", "where2comm"]
    assignment = write_assignment(WORK / "v2xreal_lidaronly.json")
    rows = []
    for name in names:
        folder = prepare(name, assignment)
        print(f"=== {name}: jigsawcomm inference_baselines.py on our data copy", flush=True)
        run = subprocess.run([sys.executable, "opencood/tools/inference_baselines.py", "--model_dir", str(folder)],
                             cwd=CODE, env={**os.environ, "PYTHONPATH": str(CODE)})
        produced = sorted(folder.glob("eval_baseline_epoch*.yaml"))
        if run.returncode != 0 or not produced:
            rows.append((name, reported(name), None))
            continue
        ours = yaml.safe_load(produced[-1].read_text())
        rows.append((name, reported(name), {"ap_50": ours.get("ap_50"), "ap_70": ours.get("ap_70")}))
    print("\nnetwork      reported AP@0.5/0.7   reproduced AP@0.5/0.7   difference@0.7")
    for name, rep, ours in rows:
        if ours is None:
            print(f"{name:12s} {rep['ap_50']:.3f} / {rep['ap_70']:.3f}        FAILED (see the output above)")
        else:
            print(f"{name:12s} {rep['ap_50']:.3f} / {rep['ap_70']:.3f}        {ours['ap_50']:.3f} / {ours['ap_70']:.3f}"
                  f"           {ours['ap_70'] - rep['ap_70']:+.3f}")


if __name__ == "__main__":
    main()
