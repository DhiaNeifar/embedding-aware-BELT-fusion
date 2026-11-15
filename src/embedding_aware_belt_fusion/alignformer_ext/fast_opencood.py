"""OpenCOOD's rotated NMS and AP matching with the box overlaps computed in C++ (fusion_core), same results.

Profiling the intermediate-fusion runner (scripts/run_host.py, AttFuse, 150 frames x 6 conditions) put ~57% of the
time in shapely: ``common_utils.compute_iou`` builds a polygon intersection and union per box pair in Python (and
computes the union twice, once for a debug print). The network's forward pass was ~10%.

These two functions keep OpenCOOD's algorithms line for line (same sort calls, same top-1000 cut, same ``>``/``<``
comparisons, same greedy matching and pop order) and change only how a BEV IoU is computed: exact convex-polygon
clipping in double precision cast to float32 (``fusion_core.iou_matrix``), where OpenCOOD uses shapely in double
precision cast to float32. ``scripts/check_fast_opencood.py`` checks that every decision is identical.

``install()`` replaces ``box_utils.nms_rotated`` and ``eval_utils.caluclate_tp_fp`` in the running process only;
the files in external/OpenCOOD are not edited.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_BUILD = Path(__file__).resolve().parents[1] / "fusion_core" / "build"


def _core():
    if str(_BUILD) not in sys.path:
        sys.path.insert(0, str(_BUILD))
    import fusion_core
    return fusion_core


def _corners(boxes) -> np.ndarray:
    """(N, 8, 3) float32 corners for fusion_core, from OpenCOOD's (N, 8, 3) or (N, 4, 2) tensors or arrays."""
    if hasattr(boxes, "detach"):
        boxes = boxes.detach().cpu().numpy()
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim == 3 and boxes.shape[1:] == (4, 2):
        out = np.zeros((len(boxes), 8, 3), np.float32)
        out[:, :4, :2] = boxes
        return out
    return np.ascontiguousarray(boxes.reshape(len(boxes), 8, 3))


def nms_rotated(boxes, scores, threshold):
    """box_utils.nms_rotated, line for line."""
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int32)
    corners = _corners(boxes)
    scores = scores.cpu().detach().numpy()
    top = 1000
    ixs = scores.argsort()[::-1][:top]
    iou_all = _core().iou_matrix(corners[ixs], corners[ixs])        # rows/columns in ixs order
    position = {int(ix): k for k, ix in enumerate(ixs)}
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        rest = np.fromiter((position[int(j)] for j in ixs[1:]), dtype=np.int64, count=len(ixs) - 1)
        iou = iou_all[position[int(i)], rest]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh):
    """eval_utils.caluclate_tp_fp, line for line."""
    fp = []
    tp = []
    gt = gt_boxes.shape[0]
    if det_boxes is not None:
        det_corners = _corners(det_boxes)
        gt_corners = _corners(gt_boxes)
        det_score = det_score.detach().cpu().numpy() if hasattr(det_score, "detach") else np.asarray(det_score)

        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend]
        iou_all = _core().iou_matrix(det_corners, gt_corners) if len(gt_corners) else None
        remaining = list(range(len(gt_corners)))               # the gt_polygon_list, as indices

        for i in range(score_order_descend.shape[0]):
            if len(remaining):
                ious = iou_all[score_order_descend[i], remaining]
            if len(remaining) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue
            fp.append(0)
            tp.append(1)
            remaining.pop(int(np.argmax(ious)))

        result_stat[iou_thresh]["score"] += det_score.tolist()

    result_stat[iou_thresh]["fp"] += fp
    result_stat[iou_thresh]["tp"] += tp
    result_stat[iou_thresh]["gt"] += gt


def install() -> None:
    """Use the C++ overlaps in OpenCOOD's NMS and AP matching, in this process."""
    from opencood.utils import box_utils, eval_utils

    box_utils.nms_rotated = nms_rotated
    eval_utils.caluclate_tp_fp = caluclate_tp_fp


__all__ = ["caluclate_tp_fp", "install", "nms_rotated"]
