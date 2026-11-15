#!/usr/bin/env python3
"""Check that alignformer_ext.fast_opencood gives OpenCOOD's exact NMS and AP decisions, on a real host run.

Runs scripts/run_host.py unchanged, with OpenCOOD's own nms_rotated and caluclate_tp_fp wrapped: every call is made
both ways on the same inputs, the kept boxes and the tp / fp / score lists are compared, and OpenCOOD's own result is
the one passed on. Any difference is counted and the script exits non-zero.

    PYTHONPATH=src python scripts/check_fast_opencood.py --dataset opv2v --host attfuse --sigmas 1 --seeds 1 \
        --max-frames 150 --workers 8 --output /tmp/check.json
"""

from __future__ import annotations

import copy
import runpy
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "external" / "OpenCOOD"))
sys.path.insert(0, str(REPO / "src"))

from opencood.utils import box_utils, eval_utils  # noqa: E402

from embedding_aware_belt_fusion.alignformer_ext import fast_opencood  # noqa: E402

COUNT = {"nms_calls": 0, "nms_diff": 0, "tpfp_calls": 0, "tpfp_diff": 0, "time_opencood": 0.0, "time_fast": 0.0}
original_nms, original_tpfp = box_utils.nms_rotated, eval_utils.caluclate_tp_fp


def nms_both(boxes, scores, threshold):
    t = time.time(); want = original_nms(boxes, scores, threshold); COUNT["time_opencood"] += time.time() - t
    t = time.time(); got = fast_opencood.nms_rotated(boxes, scores, threshold); COUNT["time_fast"] += time.time() - t
    COUNT["nms_calls"] += 1
    if not np.array_equal(np.asarray(want), np.asarray(got)):
        COUNT["nms_diff"] += 1
    return want


def tpfp_both(det_boxes, det_score, gt_boxes, result_stat, iou_thresh):
    mine = copy.deepcopy(result_stat)
    t = time.time(); original_tpfp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh)
    COUNT["time_opencood"] += time.time() - t
    t = time.time(); fast_opencood.caluclate_tp_fp(det_boxes, det_score, gt_boxes, mine, iou_thresh)
    COUNT["time_fast"] += time.time() - t
    COUNT["tpfp_calls"] += 1
    if mine[iou_thresh] != result_stat[iou_thresh]:
        COUNT["tpfp_diff"] += 1


box_utils.nms_rotated = nms_both
eval_utils.caluclate_tp_fp = tpfp_both

if __name__ == "__main__":
    sys.argv = [str(REPO / "scripts" / "run_host.py")] + sys.argv[1:] + ["--opencood-iou"]   # keep these wrappers
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        print(f"NMS: {COUNT['nms_calls']} calls, {COUNT['nms_diff']} different | AP matching: "
              f"{COUNT['tpfp_calls']} calls, {COUNT['tpfp_diff']} different | time OpenCOOD "
              f"{COUNT['time_opencood']:.1f} s, C++ {COUNT['time_fast']:.1f} s")
    if COUNT["nms_diff"] or COUNT["tpfp_diff"]:
        raise SystemExit("fast_opencood differs from OpenCOOD")
    print("identical")
