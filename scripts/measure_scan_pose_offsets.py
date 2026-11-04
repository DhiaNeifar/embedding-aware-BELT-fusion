"""Measure CooperScene's per-(recording, agent) scan-vs-pose offsets from sigma-0 GICP registrations.

    python scripts/register_cooperscene_scans_gicp.py 0 0 outputs/calibration/cooperscene_gicp_sigma0.npz
    python scripts/measure_scan_pose_offsets.py

Writes the JSON the sweep's ``--calibration`` reads (alignformer_ext/calibration.py).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_verify_core  # noqa: E402

from embedding_aware_belt_fusion.alignformer_ext import calibration  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame-table", type=Path, default=Path("outputs/cooperscene_test_frame_table.pt"))
    parser.add_argument("--cache", type=Path, default=Path("outputs/propose_cache_cooperscene/sigma_0_seed_0.npz"))
    parser.add_argument("--registration", type=Path, default=Path("outputs/calibration/cooperscene_gicp_sigma0.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/calibration/cooperscene_scan_offsets.json"))
    args = parser.parse_args()
    records = check_verify_core.load_frame_table(args.frame_table)
    with np.load(args.cache) as data:
        cache = {k: data[k] for k in data.files}
    if float(cache["sigma"]) != 0.0:
        raise ValueError("offsets are measured from the true poses: use a sigma-0 cache")
    with np.load(args.registration) as data:
        registration = {k: data[k] for k in data.files}
    offsets = calibration.measure_offsets(check_verify_core.base, records, cache, registration)
    calibration.save_offsets(offsets, args.output)
    for (recording, agent), o in sorted(offsets.items()):
        print(f"recording {recording} agent {agent}: {np.hypot(o[0], o[1]):.2f} m, {np.degrees(o[2]):+.2f} deg")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
