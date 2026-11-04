"""Alignment-aware detection confidence: a CAV box counts only if it is detected AND aligned.

AP ranks boxes by confidence, and late fusion's NMS keeps the highest-confidence box of each object. A CAV box's
detector score says nothing about how well the box was moved into the ego frame, so a confidently detected but
misaligned box ranks high (a false positive at IoU 0.7) and can win NMS over the ego's own, correctly placed box.
Each CAV box's score is multiplied by the probability that its alignment error stays within the shift that
still clears the IoU threshold:

* alignment uncertainty of the CAV's pose, estimated without ground truth (``temporal.temporal_decisions``):
  for an applied window estimate, the window's own robust spread NOT divided by its length (measurement errors
  persist from frame to frame, so the window mean does not average them down; the GLS posterior is 5-7x
  over-confident), plus its odometry covariance; for a CAV left uncorrected, the reported poses' error level;
* propagated to each box: a yaw error rotates the box about the CAV, so the lever arm is box - CAV position;
* tolerance from the box itself: two equal boxes shifted sideways by ``d`` have IoU ``(w - d) / (w + d)``, so
  IoU 0.7 is lost at ``d = w (1 - 0.7) / (1 + 0.7)`` (0.33 m for a 1.9 m-wide car);
* probability under an isotropic 2-D Gaussian: ``1 - exp(-d^2 / (2 s^2))``, ``s^2`` = half the box covariance trace.

No box moves; only scores (hence ranking and NMS's choice) change. Ego boxes are untouched.
"""

from __future__ import annotations

import numpy as np
import torch

from embedding_aware_belt_fusion.alignformer_ext import evaluate
from embedding_aware_belt_fusion.alignformer_ext.joint import compose

IOU_THRESHOLD = 0.7
BOX_W, BOX_L = 4, 5   # (x, y, z, h, w, l, yaw)


def alignment_confidence(inputs, decided: np.ndarray, uncertainty: dict, kind: str = "window") -> np.ndarray:
    """Per cached CAV box (flat, aligned with ``cache["cav_boxes"]``): P(alignment error within the IoU tolerance)."""
    from embedding_aware_belt_fusion.alignformer_ext import temporal
    from embedding_aware_belt_fusion.alignformer_ext.propose_cache import noisy_poses

    base, records, cache = inputs.base, inputs.records, inputs.cache
    sigma, seed = float(cache["sigma"]), int(cache["seed"])
    off = cache["cav_off"]
    out = np.ones(len(cache["cav_boxes"]), np.float32)
    shrink = (1.0 - IOU_THRESHOLD) / (1.0 + IOU_THRESHOLD)
    reported_cache = {}
    for p in range(len(cache["frame"])):
        if off[p + 1] == off[p]:
            continue
        f = int(cache["frame"][p])
        record = records[f]
        cav_id = sorted(record.cav_boxes)[int(cache["cav_rank"][p])]
        if f not in reported_cache:
            reported_cache = {f: noisy_poses(base, record, f, sigma, seed)}
        n = temporal._pose2d(base.x1_to_x2(reported_cache[f][cav_id], record.ego_pose))
        boxes = torch.from_numpy(cache["cav_boxes"][off[p]:off[p + 1]])
        if cache["attempted"][p] and np.any(decided[p] != 0):
            boxes = evaluate.apply_correction(boxes, *map(float, decided[p]))
            pose = compose(np.array([decided[p][1], decided[p][2], decided[p][0]], np.float64), n)
            cov = uncertainty.get(kind, {}).get(p, uncertainty["reported"])
        else:
            pose, cov = n, uncertainty["reported"]
        b = boxes.numpy().astype(np.float64)
        lx, ly = b[:, 0] - pose[0], b[:, 1] - pose[1]
        var = (cov[0, 0] + cov[1, 1] + (lx ** 2 + ly ** 2) * cov[2, 2] - 2 * ly * cov[0, 2] + 2 * lx * cov[1, 2]) / 2.0
        tol = shrink * np.minimum(b[:, BOX_W], b[:, BOX_L])
        out[off[p]:off[p + 1]] = 1.0 - np.exp(-tol ** 2 / (2.0 * np.maximum(var, 1e-12)))
    return out
