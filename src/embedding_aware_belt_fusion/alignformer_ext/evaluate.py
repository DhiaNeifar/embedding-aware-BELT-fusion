"""Fuse and score one condition from cached Propose output + per-pair decided corrections.

Same fusion and AP as ``reproduce_opv2v_sweep.run_condition`` (its own
``fuse_and_score_accumulate`` / ``calculate_ap`` over fusion_core). Corrections
are applied exactly as ``integration.localization.apply_se2_correction`` does,
and only to pairs whose decided correction is non-zero (``accepted``), as
``inference.correct_simple_detections`` does. Frames are split into CAV-count
buckets (CAVs in range with detections): ``all``, ``lt3``, ``ge3``, and ``multi`` (>= 2 CAVs).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

BUCKETS = ("all", "lt3", "ge3", "multi")   # multi: >= 2 CAVs (>= 3 agents)


def cav_count(record) -> int:
    return sum(1 for cav_id in record.cav_boxes if cav_id in record.cav_poses)


def apply_correction(boxes: torch.Tensor, psi: float, tx: float, ty: float) -> torch.Tensor:
    """``apply_se2_correction`` on an (N, 7) box tensor."""
    psi_t = torch.tensor(psi, dtype=torch.float32)
    cos, sin = torch.cos(psi_t), torch.sin(psi_t)
    rotation = torch.stack([torch.stack([cos, -sin]), torch.stack([sin, cos])])
    out = boxes.clone()
    out[:, :2] = boxes[:, :2] @ rotation.T + torch.tensor([tx, ty], dtype=torch.float32)
    yaw = boxes[:, 6] + psi_t
    out[:, 6] = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    return out


def score_condition(base, records, cache: Dict[str, np.ndarray], decided: Optional[np.ndarray],
                    nms_thresh: float, oracle: bool = False, freealign: bool = False,
                    score_scale: Optional[np.ndarray] = None) -> Dict[str, Dict]:
    """Per-bucket TP/FP stats. ``decided`` (P, 3) psi, tx, ty per cached pair (zero = keep the noisy boxes);
    ``None`` scores the uncorrected boxes; ``oracle=True`` the true-pose boxes; ``freealign=True`` the cached
    FreeAlign-corrected boxes. ``score_scale`` (flat, one per cached CAV box) multiplies the CAV boxes' scores."""
    stats = {bucket: base.new_stats() for bucket in BUCKETS}
    pairs_by_frame: Dict[int, list] = {}
    for p, frame in enumerate(cache["frame"]):
        pairs_by_frame.setdefault(int(frame), []).append(p)
    cav_off = cache["cav_off"]
    for frame_index, record in enumerate(records):
        boxes = [record.ego_boxes]
        scores = [record.ego_scores]
        cav_ids = sorted(record.cav_boxes)
        for p in pairs_by_frame.get(frame_index, []):
            cav_id = cav_ids[int(cache["cav_rank"][p])]
            if oracle and "oracle_boxes" in cache:     # asynchronous cache: the stale boxes at their true pose
                cav = torch.from_numpy(cache["oracle_boxes"][cav_off[p]:cav_off[p + 1]])
            elif oracle:
                transform = torch.as_tensor(base.x1_to_x2(record.cav_poses[cav_id], record.ego_pose), dtype=torch.float32)
                cav = base.transform_boxes_to_ego(record.cav_boxes[cav_id], transform)
            elif freealign:
                cav = torch.from_numpy(cache["freealign_boxes"][cav_off[p]:cav_off[p + 1]])
            else:
                cav = torch.from_numpy(cache["cav_boxes"][cav_off[p]:cav_off[p + 1]])
                if decided is not None and cache["attempted"][p] and np.any(decided[p] != 0.0):
                    cav = apply_correction(cav, *map(float, decided[p]))
            boxes.append(cav)
            # an asynchronous cache carries the scores of the (stale) message itself
            cav_scores = (torch.from_numpy(cache["cav_scores"][cav_off[p]:cav_off[p + 1]]) if "cav_scores" in cache
                          else record.cav_scores[cav_id])
            if score_scale is None:
                scores.append(cav_scores)
            else:
                scale = torch.from_numpy(score_scale[cav_off[p]:cav_off[p + 1]]).to(cav_scores.dtype)
                scores.append(cav_scores * scale)
        corners = np.ascontiguousarray(
            base.box_utils.boxes_to_corners_3d(torch.cat(boxes).numpy(), order=base.box_order()), dtype=np.float32
        )
        scores_np = np.ascontiguousarray(torch.cat(scores).numpy(), dtype=np.float32)
        bucket = "ge3" if cav_count(record) >= 3 else "lt3"
        for name in ("all", bucket) + (("multi",) if cav_count(record) >= 2 else ()):
            base.fuse_and_score_accumulate(corners, scores_np, record.gt_corners, nms_thresh, stats[name])
    return stats


def ap_table(base, stats: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    return {
        bucket: {f"ap_{int(t * 100):02d}": base.calculate_ap(s[t]) for t in base.THRESHOLDS}
        for bucket, s in stats.items()
    }
