import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external" / "OpenCOOD"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from opencood.utils import box_utils  # noqa: E402

from embedding_aware_belt_fusion.multidataset import sources  # noqa: E402

OPV2V_ROOT = Path("/mnt/external/workspace/public/dataset/opv2v")
OPV2V_FRAME_TABLE = REPO_ROOT / "outputs" / "opv2v_frame_table.pt"
OPV2V_CONFIG = REPO_ROOT / "configs" / "pointpillar_late_fusion_r140.yaml"


def _database():
    return OrderedDict(
        {
            0: OrderedDict(
                (cav, OrderedDict([("000000", {"yaml": f"/d/s/{cav}/000000.yaml"}), ("ego", cav == "0")]))
                for cav in ("0", "1", "2", "3")
            )
        }
    )


def test_set_ego_moves_ego_first_and_flags_only_it():
    reordered = sources.set_ego(_database(), "1")
    agents = reordered[0]
    assert list(agents) == ["1", "0", "2", "3"]
    assert [agents[c]["ego"] for c in agents] == [True, False, False, False]
    # Timestamps stay ahead of the ego flag, as OpenCOOD's own indexing expects.
    assert list(agents["1"]) == ["000000", "ego"]


def test_set_ego_refuses_missing_agent():
    with pytest.raises(KeyError):
        sources.set_ego(_database(), "7")


def test_corner_to_center_round_trip_keeps_heading_not_just_mod_pi():
    rng = np.random.default_rng(0)
    boxes = np.column_stack(
        [
            rng.uniform(-50, 50, (64, 2)),
            rng.uniform(-2, 0, 64),
            rng.uniform(1.4, 2.0, 64),  # h
            rng.uniform(1.6, 2.2, 64),  # w
            rng.uniform(3.5, 5.0, 64),  # l
            rng.uniform(-np.pi, np.pi, 64),
        ]
    ).astype(np.float32)
    corners = box_utils.boxes_to_corners_3d(boxes, order="hwl")
    back = box_utils.corner_to_center(corners, order="hwl")
    np.testing.assert_allclose(back[:, :6], boxes[:, :6], atol=1e-4)
    heading_error = np.angle(np.exp(1j * (back[:, 6] - boxes[:, 6])))
    np.testing.assert_allclose(heading_error, 0.0, atol=1e-4)


@pytest.mark.skipif(
    not (OPV2V_FRAME_TABLE.exists() and OPV2V_ROOT.exists() and OPV2V_CONFIG.exists()),
    reason="needs the OPV2V test split and the saved real-LiDAR frame table",
)
def test_stub_lidar_ground_truth_matches_real_lidar_frame_table():
    import reproduce_multidataset_sweep as sweep
    import reproduce_opv2v_sweep as base

    saved = sweep.load_frame_table(OPV2V_FRAME_TABLE)
    hypes = base.load_yaml(str(OPV2V_CONFIG), None)
    dataset = sources.build_sweep_dataset("opv2v", hypes, OPV2V_ROOT / "test")
    frames = [0, 1, 2, 500, 1000, len(saved) - 1]
    with sources.stub_lidar():
        for index in frames:
            built = base.build_frame_table(_Subset(dataset, index), None, 1)[0]
            reference = saved[index]
            assert built.ego_pose == reference.ego_pose
            assert built.cav_poses == reference.cav_poses
            np.testing.assert_array_equal(
                _sorted_corners(built.gt_corners), _sorted_corners(reference.gt_corners)
            )


class _Subset:
    """One dataset index presented as index 0, for build_frame_table(..., 1)."""

    def __init__(self, dataset, index):
        self._dataset, self._index = dataset, index

    def retrieve_base_data(self, _index):
        return self._dataset.retrieve_base_data(self._index)

    def __getattr__(self, name):
        return getattr(self._dataset, name)


def _sorted_corners(corners):
    # generate_gt_bbx dedups through set(object_ids); compare as a set of boxes.
    flat = np.asarray(corners).reshape(len(corners), -1)
    return flat[np.lexsort(flat.T[::-1])]
