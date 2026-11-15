#!/usr/bin/env python3
"""Odometry table from a frame table's recorded poses (audit E3): every agent's recorded LiDAR pose per frame.

The same table format as scripts/build_odometry_table.py (scenario, agent, frame, 4x4 pose), so the temporal layer
reads it unchanged through alignformer_ext/odometry.py. Only relative motion between an agent's own frames is ever
used. On V2X-Real this gives recorded-pose odometry, to compare with KISS-ICP on the same frames.

    python scripts/build_odometry_from_frame_table.py --frame-table outputs/v2xreal_test_frame_table.pt \\
        --geometry-cache outputs/v2xreal_geometry_test_cache --output outputs/odometry/v2xreal_test_recorded.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_verify_core  # noqa: E402
import reproduce_opv2v_sweep  # noqa: E402,F401  (puts OpenCOOD on the path)
from opencood.utils.transformation_utils import x_to_world  # noqa: E402

from embedding_aware_belt_fusion.alignformer_ext import odometry  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame-table", type=Path, required=True)
    parser.add_argument("--geometry-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = check_verify_core.load_frame_table(args.frame_table)
    frames = odometry._frame_index(args.geometry_cache, records)
    poses = {}
    for (scenario, frame_id, ego), record in tqdm(list(zip(frames, records)), desc="recorded poses"):
        poses[(scenario, str(ego), frame_id)] = x_to_world(list(record.ego_pose))
        for cav_id, pose in record.cav_poses.items():
            poses.setdefault((scenario, str(cav_id), frame_id), x_to_world(list(pose)))
    keys = sorted(poses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, scenario=np.array([k[0] for k in keys]), agent=np.array([k[1] for k in keys]),
                        frame=np.array([k[2] for k in keys]), pose=np.stack([poses[k] for k in keys]).astype(np.float64),
                        max_range=np.float64(0.0), source=np.array("recorded-frame-table"))
    print(f"{len(keys)} agent poses from {len(records)} frames -> {args.output}")


if __name__ == "__main__":
    main()
