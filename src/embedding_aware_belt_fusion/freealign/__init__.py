"""The ICRA 2024 FreeAlign baseline, reimplemented on our own detector/data.

This package is the competitor, not part of our own architecture
(``alignformer_pva/``), and it lives at this separate top-level location
specifically so that distinction stays easy to navigate and so the
comparison between the two stays honest: nothing in here imports
``alignformer_pva`` for its actual pose-fitting logic (see
``freealign.py``'s own module docstring, "Deviations from the paper", for
the one place the two methods' code could have been shared -- the
unweighted least-squares SE(2) core -- and why it is written twice instead).

Reference: Lei, Ni, Han, Tang, Wang, Feng, Chen and Wang, "Robust
Collaborative Perception without External Localization and Clock Devices",
ICRA 2024 (arXiv 2405.02965v2). This is a reimplementation, read and ported
read-only (via ``git show``) from a labmate's own careful reimplementation at
``embedding-aware-BELT-fusion`` (branch ``origin/alignformer``,
``src/embedding_aware_belt_fusion/alignformer/freealign.py``), adapted to
this repo's own box layout (``[x, y, z, h, w, l, yaw]``, ``BOX_YAW = 6`` --
confirmed identical to the source repo's own convention) and its own
calibration (``calibrate_freealign.py`` fits FreeAlign's free parameters on
OUR detector/validation split, never on the labmate's numbers or on test).
Nothing from the paper authors' own public repository
(``github.com/MediaBrain-SJTU/FreeAlign``) is vendored or executed anywhere
in this package.
"""

from embedding_aware_belt_fusion.freealign.freealign import (
    DEFAULT_ANCHOR_LIMIT,
    DEFAULT_EDGE_THRESHOLD_M,
    DEFAULT_EPSILON_OFFSET,
    DEFAULT_EPSILON_POWER,
    DEFAULT_INLIER_THRESHOLD_M,
    DEFAULT_MIN_NODES,
    DEFAULT_RANSAC_ITERATIONS,
    DEFAULT_YAW_WEIGHT,
    EDGE_DISTANCE,
    EDGE_DISTANCE_YAW,
    LMEDS,
    RANSAC,
    FreeAlignConfig,
    FreeAlignMatch,
    PoseEstimate,
    edge_features,
    freealign_estimate,
    is_fallback,
    mass_common_subgraph,
    robust_se2,
    se2_from_points,
)

__all__ = [
    "DEFAULT_ANCHOR_LIMIT",
    "DEFAULT_EDGE_THRESHOLD_M",
    "DEFAULT_EPSILON_OFFSET",
    "DEFAULT_EPSILON_POWER",
    "DEFAULT_INLIER_THRESHOLD_M",
    "DEFAULT_MIN_NODES",
    "DEFAULT_RANSAC_ITERATIONS",
    "DEFAULT_YAW_WEIGHT",
    "EDGE_DISTANCE",
    "EDGE_DISTANCE_YAW",
    "LMEDS",
    "RANSAC",
    "FreeAlignConfig",
    "FreeAlignMatch",
    "PoseEstimate",
    "edge_features",
    "freealign_estimate",
    "is_fallback",
    "mass_common_subgraph",
    "robust_se2",
    "se2_from_points",
]
