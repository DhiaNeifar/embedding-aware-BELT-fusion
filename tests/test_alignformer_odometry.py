"""alignformer_ext.odometry: a table step means the same as x1_to_x2 between the two poses, and gaps are reported."""

import numpy as np

from embedding_aware_belt_fusion.alignformer_ext import odometry, temporal


def _pose(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0, x], [s, c, 0.0, y], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def _table(tmp_path):
    poses = np.stack([_pose(0.0, 0.0, 0.0), _pose(1.0, 0.0, 0.1), _pose(9.0, 9.0, 0.0), _pose(3.0, 1.0, -0.2)])
    path = tmp_path / "odo.npz"
    np.savez(path, scenario=np.array(["s", "s", "s", "s"]), agent=np.array(["1", "1", "2", "2"]),
             frame=np.array(["a", "b", "a", "b"]), pose=poses)
    return odometry.OdometryTable(path, [("s", "a", "1"), ("s", "b", "1")]), poses


def test_step_is_source_in_target_frame(tmp_path):
    table, poses = _table(tmp_path)
    m = np.linalg.inv(poses[1]) @ poses[0]
    assert np.allclose(table.between(0, 1, odometry.EGO), [m[0, 3], m[1, 3], np.arctan2(m[1, 0], m[0, 0])])
    assert np.allclose(table.between(0, 0, "2"), np.zeros(3))


def test_missing_scan_or_other_scenario_gives_none(tmp_path):
    table, _ = _table(tmp_path)
    assert table.between(0, 1, "3") is None
    table.frames.append(("other", "a", "1"))
    assert table.between(0, 2, odometry.EGO) is None


def test_measured_step_uses_fixed_accuracy_not_simulated_noise(tmp_path):
    table, _ = _table(tmp_path)
    settings = temporal.TemporalSettings(odometry_sd=(0.0, 0.0), odometry_rel=(0.01, 0.0))
    step, cov = temporal._measured_step(table, 0, 1, odometry.EGO, settings)
    assert np.allclose(step, table.between(0, 1, odometry.EGO))
    assert np.isclose(cov[0, 0], (0.01 * np.hypot(step[0], step[1])) ** 2)
