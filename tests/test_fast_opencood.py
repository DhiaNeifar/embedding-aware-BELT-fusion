"""alignformer_ext.fast_opencood must make OpenCOOD's exact NMS and AP-matching decisions (shapely vs C++ overlaps)."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "external" / "OpenCOOD"))
torch = pytest.importorskip("torch")
box_utils = pytest.importorskip("opencood.utils.box_utils")
eval_utils = pytest.importorskip("opencood.utils.eval_utils")

from embedding_aware_belt_fusion.alignformer_ext import fast_opencood  # noqa: E402


def _boxes(rng, n, spread):
    """(n, 8, 3) corners of cars scattered over ``spread`` metres, many overlapping."""
    centers = np.c_[rng.uniform(-spread, spread, (n, 2)), rng.uniform(-1, 0, n), rng.uniform(1.4, 1.7, n),
                    rng.uniform(1.6, 2.1, n), rng.uniform(3.8, 4.8, n), rng.uniform(-np.pi, np.pi, n)]
    return box_utils.boxes_to_corners_3d(torch.from_numpy(centers).float(), order="hwl")


@pytest.mark.parametrize("seed", range(20))
def test_same_decisions_as_opencood(seed):
    rng = np.random.default_rng(seed)
    det = _boxes(rng, int(rng.integers(0, 60)), 12.0)
    gt = _boxes(rng, int(rng.integers(0, 30)), 12.0)
    if len(det) and len(gt):                 # some detections sitting almost on a ground-truth box
        k = min(len(det), len(gt)) // 2
        det[:k] = gt[:k] + torch.from_numpy(rng.normal(0, 0.15, (k, 1, 3))).float()
    scores = torch.from_numpy(rng.uniform(0, 1, len(det))).float()

    for threshold in (0.15, 0.5):
        want = box_utils.nms_rotated(det, scores, threshold)
        got = fast_opencood.nms_rotated(det, scores, threshold)
        assert np.array_equal(want, got)

    for iou in (0.3, 0.5, 0.7):
        a = {iou: {"tp": [], "fp": [], "gt": 0, "score": []}}
        b = copy.deepcopy(a)
        eval_utils.caluclate_tp_fp(det if len(det) else None, scores, gt, a, iou)
        fast_opencood.caluclate_tp_fp(det if len(det) else None, scores, gt, b, iou)
        assert a == b
