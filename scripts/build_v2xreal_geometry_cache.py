#!/usr/bin/env python3
"""V2X-Real geometry cache: ``extract_proposals --cache-mode geometry_only`` through Chenyi's adapter.

``extract_proposals`` resolves ``opencood.data_utils.datasets.build_dataset``
at call time, so routing it through ``multidataset.v2xreal.build_dataset``
(``.bin`` LiDAR, vehicle-class filter, vehicle-centric ego, V2X-Real
GT/COM range) is enough; ``extract_proposals`` itself is not modified. Every
argument after ``--`` is passed to it verbatim.

Usage (GPU, detector inference once over the split):
    python scripts/build_v2xreal_geometry_cache.py -- \\
        --cache-mode geometry_only \\
        --opencood-config $CHENYI/outputs/v2xreal/point_pillar_late_fusion/config.yaml \\
        --detector-dir   $CHENYI/outputs/v2xreal/point_pillar_late_fusion \\
        --data-root /mnt/external/workspace/public/dataset/v2x-real --split test \\
        --output-dir outputs/v2xreal_geometry_test_cache --workers 8
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "external" / "OpenCOOD"))

import opencood.data_utils.datasets as opencood_datasets  # noqa: E402

from embedding_aware_belt_fusion.data import extract_proposals  # noqa: E402
from embedding_aware_belt_fusion.multidataset import v2xreal  # noqa: E402


def main() -> None:
    argv = sys.argv[1:]
    if argv[:1] == ["--"]:
        argv = argv[1:]
    stock_build = opencood_datasets.build_dataset

    def build(hypes, visualize=False, train=True):
        if hypes["fusion"]["core_method"] != v2xreal.CORE_METHOD:
            raise ValueError(
                f"expected a {v2xreal.CORE_METHOD} detector config, got {hypes['fusion']['core_method']!r}"
            )
        v2xreal.install_shims(hypes)
        return v2xreal._dataset_class()(params=hypes, visualize=visualize, train=train)

    opencood_datasets.build_dataset = build
    sys.argv = [extract_proposals.__file__] + argv
    try:
        extract_proposals.main()
    finally:
        opencood_datasets.build_dataset = stock_build


if __name__ == "__main__":
    main()
