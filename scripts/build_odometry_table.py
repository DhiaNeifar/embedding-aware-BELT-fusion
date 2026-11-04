#!/usr/bin/env python3
"""LiDAR odometry for every agent of a dataset split: KISS-ICP on the agent's own scans, nothing else.

Each agent's trajectory is estimated from its scans alone, starting at identity at its first scan. No recorded pose
is read, so the plug-in's odometry carries no ground truth and no simulated noise. Runs in its own environment,
because this needs only numpy and kiss-icp:

    outputs/kiss_icp_env/bin/python scripts/build_odometry_table.py \\
        --split-dir /mnt/external/workspace/public/dataset/opv2v/test --output outputs/odometry/opv2v_test.npz
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from tqdm import tqdm


def read_scan(path: Path) -> np.ndarray:
    """(N, 3) float64 points, non-finite rows dropped."""
    if path.suffix == ".bin":
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    else:
        with open(path, "r") as handle:
            for line in handle:
                if line.startswith("DATA"):
                    if "ascii" not in line:
                        raise ValueError(f"{path}: only ascii .pcd is supported")
                    break
            points = np.loadtxt(handle, dtype=np.float32, usecols=(0, 1, 2))
    return points[np.isfinite(points).all(axis=1)].astype(np.float64)


def agent_scans(agent_dir: Path) -> List[Path]:
    scans = [p for p in agent_dir.iterdir() if p.suffix in (".pcd", ".bin")]
    return sorted(scans, key=lambda p: p.stem)


def run_agent(job: Tuple[str, str, str, float, int]):
    """(scenario, agent, frame ids, (F, 4, 4) poses of the agent in its own first-scan frame)."""
    from kiss_icp.config import load_config
    from kiss_icp.kiss_icp import KissICP

    agent_dir, scenario, agent, max_range, limit = job
    config = load_config(None)
    config.data.max_range = max_range
    config.data.deskew = False
    config.mapping.voxel_size = max_range / 100.0
    config.registration.max_num_threads = 1
    icp = KissICP(config=config)
    scans = agent_scans(Path(agent_dir))
    if limit:
        scans = scans[:limit]
    poses = np.empty((len(scans), 4, 4))
    for i, scan in enumerate(scans):
        points = read_scan(scan)
        icp.register_frame(points, np.zeros(len(points)))
        poses[i] = icp.last_pose
    return scenario, agent, [p.stem for p in scans], poses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-range", type=float, default=100.0)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--scenarios", type=int, default=0, help="first N scenarios only (a check, not a table)")
    parser.add_argument("--frames", type=int, default=0, help="first N scans per agent only (a check, not a table)")
    args = parser.parse_args()

    jobs = []
    scenarios = sorted(p for p in args.split_dir.iterdir() if p.is_dir())
    for scenario in scenarios[: args.scenarios or None]:
        for agent_dir in sorted(p for p in scenario.iterdir() if p.is_dir()):
            if agent_scans(agent_dir):
                jobs.append((str(agent_dir), scenario.name, agent_dir.name, args.max_range, args.frames))

    rows, frames, poses = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_agent, job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"odometry {args.split_dir.name}"):
            scenario, agent, ids, trajectory = future.result()
            rows += [(scenario, agent)] * len(ids)
            frames += ids
            poses.append(trajectory)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, scenario=np.array([r[0] for r in rows]), agent=np.array([r[1] for r in rows]),
                        frame=np.array(frames), pose=np.concatenate(poses).astype(np.float64),
                        max_range=np.float64(args.max_range))
    print(f"{len(frames)} scans, {len(jobs)} agent trajectories -> {args.output}")


if __name__ == "__main__":
    main()
