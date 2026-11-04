"""alignformer_ext.latency: the joint pose + latency fit, its significance test and its observability measure."""

import numpy as np
from scipy.stats import f as f_dist

from embedding_aware_belt_fusion.alignformer_ext import latency


def _scene(tau_true, same_velocity=False, seed=0):
    rng = np.random.default_rng(seed)
    cav = rng.uniform(-40, 40, (14, 2))
    vel = np.zeros((14, 2))
    if same_velocity:
        vel[:] = [1.0, 0.0]
    else:
        vel[:5] = 0.0                                                     # parked cars
        vel[5:] = rng.normal(0, 1.0, (9, 2))                              # traffic in several directions
    psi, t = 0.02, np.array([0.8, -0.5])
    c, s = np.cos(psi), np.sin(psi)
    ego = (cav + vel * tau_true) @ np.array([[c, -s], [s, c]]).T + t + rng.normal(0, 0.05, (14, 2))
    return ego, cav, vel, psi, t


def _significant(fit):
    dof = 2 * fit["pairs"] - 4
    return (fit["rss0"] - fit["rss1"]) / (fit["rss1"] / dof) > f_dist.ppf(latency.SIGNIFICANCE, 1, dof)


def test_joint_fit_recovers_pose_and_latency():
    ego, cav, vel, psi, t = _scene(3.0)
    fit = latency.joint_fit(ego, cav, vel, tau0=1.0)
    assert abs(fit["tau"] - 3.0) < 0.1
    assert abs(fit["psi"] - psi) < 2e-3 and np.linalg.norm(fit["t"] - t) < 0.1
    assert _significant(fit)


def test_no_delay_is_not_significant():
    ego, cav, vel, _, _ = _scene(0.0, seed=3)
    assert not _significant(latency.joint_fit(ego, cav, vel, tau0=2.0))


def test_latency_is_unobservable_when_everything_moves_alike():
    ego, cav, vel, _, _ = _scene(3.0, same_velocity=True)
    assert latency.joint_fit(ego, cav, vel, tau0=1.0)["info"] < 1e-9
    ego, cav, vel, _, _ = _scene(3.0)
    assert latency.joint_fit(ego, cav, vel, tau0=1.0)["info"] > 1.0
