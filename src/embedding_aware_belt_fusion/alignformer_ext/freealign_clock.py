"""FreeAlign with its authors' clock-deviation estimation (Lei et al., ICRA 2024, Section IV-C), for delayed messages.

The FreeAlign baseline used throughout this project (``freealign/freealign.py``, Chenyi's reimplementation) leaves
the clock-deviation half of the paper out, because those experiments were synchronous. Under communication delay
the authors' method searches a buffer of the ego's PAST object graphs for the one the received graph matches best:
MASS (their common-subgraph search, unchanged here) is run between the received boxes and the ego's boxes of each
of the last ``buffer`` frames, and the frame whose common subgraph has the smallest discrepancy (MASS's own
selection criterion) gives the clock deviation. The relative pose is then fitted on that subgraph (their robust
SE(2), unchanged) against the ego's boxes OF THAT FRAME, and moved into the ego's current frame with the ego's own
motion. FreeAlign has no motion model, so the delayed boxes are not predicted forward: the estimated clock
deviation corrects the pose, not the objects' motion.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from embedding_aware_belt_fusion.alignformer_ext.se2 import compose
from embedding_aware_belt_fusion.freealign.freealign import edge_features, mass_common_subgraph, robust_se2


def _pose2d(matrix) -> np.ndarray:
    m = np.asarray(matrix, np.float64)
    return np.array([m[0, 3], m[1, 3], np.arctan2(m[1, 0], m[0, 0])])


def freealign_with_clock_search(base, records, cache, config, scenarios: List[str], buffer: int = 7):
    """(decided (P, 3) psi, tx, ty in the ego's current frame, estimated deviation in frames (P,), -1 = discarded)."""
    n = len(cache["frame"])
    decided = np.zeros((n, 3), np.float32)
    deviation = np.full(n, -1, np.int64)
    ego_features = {}
    with torch.no_grad():
        for p in range(n):
            t = int(cache["frame"][p])
            cav = torch.from_numpy(cache["cav_boxes"][cache["cav_off"][p]:cache["cav_off"][p + 1]])
            if len(cav) == 0:
                continue
            cav_features = edge_features(cav, config)
            best = None
            for k in range(buffer):
                g = t - k
                if g < 0 or scenarios[g] != scenarios[t]:
                    break
                ego = records[g].ego_boxes
                if len(ego) == 0:
                    continue
                if g not in ego_features:
                    ego_features[g] = edge_features(ego, config)
                match = mass_common_subgraph(ego_features[g], cav_features, config)
                if len(match) < config.min_nodes:
                    continue
                if best is None or match.epsilon < best[1].epsilon:
                    best = (k, match, g)
            if best is None:
                continue
            k, match, g = best
            ego = records[g].ego_boxes
            psi, shift, _ = robust_se2(ego[match.ego_indices, :2], cav[match.cav_indices, :2], config)
            fit = np.array([float(shift[0]), float(shift[1]), float(psi)])   # current-frame placement -> ego frame g
            moved = compose(_pose2d(base.x1_to_x2(records[g].ego_pose, records[t].ego_pose)), fit)
            decided[p] = (moved[2], moved[0], moved[1])
            deviation[p] = k
    return decided, deviation
