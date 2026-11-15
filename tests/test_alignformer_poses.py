"""alignformer_ext.poses: the exported transform places boxes exactly where the correction moves them."""

import numpy as np
import torch

from embedding_aware_belt_fusion.alignformer_ext import poses
from embedding_aware_belt_fusion.alignformer_ext.evaluate import apply_correction


def test_correction_matrix_matches_box_correction():
    boxes = torch.tensor([[3.0, -2.0, 0.0, 4.0, 2.0, 1.5, 0.3], [-10.0, 5.0, 0.0, 4.0, 2.0, 1.5, -1.2]])
    psi, tx, ty = 0.05, 0.8, -0.4
    moved = apply_correction(boxes, psi, tx, ty)
    m = poses.correction_matrix(psi, tx, ty)
    centres = np.c_[boxes[:, :2].numpy(), np.zeros(2), np.ones(2)] @ m.T
    assert np.allclose(centres[:, :2], moved[:, :2].numpy(), atol=1e-5)


def test_freealign_correction_is_recovered_from_its_moved_boxes():
    rng = np.random.default_rng(0)
    noisy = np.c_[rng.uniform(-30, 30, (6, 2)), np.zeros((6, 4)), rng.uniform(-3, 3, 6)].astype(np.float32)
    moved = apply_correction(torch.from_numpy(noisy), 0.1, -1.5, 0.7).numpy()
    cache = {"frame": np.array([0]), "cav_off": np.array([0, 6]), "cav_boxes": noisy, "freealign_boxes": moved}
    assert np.allclose(poses.freealign_corrections(cache)[0], [0.1, -1.5, 0.7], atol=1e-4)
