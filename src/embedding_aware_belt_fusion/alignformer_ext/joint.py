"""Joint multi-CAV estimation: every CAV's correction solved together from ego<->CAV AND CAV<->CAV evidence.

Pairwise AlignFormer corrects each CAV only against ego. With two or more
CAVs a frame also holds CAV<->CAV evidence: CAV A and CAV B see common
objects, and aligning B onto A gives ``T_AB`` with ``X_B = X_A o T_AB``
(``X`` = the correction that maps a CAV's noisy boxes onto the truth, in the
ego frame). That is new evidence no pairwise method uses:

* a CAV the ego cannot align with (too few shared objects) can still be
  corrected through a CAV that can (a chain ego -> A -> B);
* a CAV with a direct estimate gets a second, independent one through every
  other CAV, and the solve averages them by their uncertainties.

Each edge is decided by the ``alignformer_cw`` rule (centres-only Verify,
applied only where the Wald test also accepts); an edge that rule abstains
on carries no evidence. Edge covariances are Verify's own fit covariances
(``coupling.verify_fit_stats``), CAV<->CAV ones moved into the ego frame by
the SE(2) adjoint of A's noisy pose. The corrections of all CAVs connected to
ego through accepted edges are solved by Gauss-Newton on SE(2) (numeric
Jacobian; at most a handful of CAVs per frame). A CAV with no accepted path
to ego stays uncorrected.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from embedding_aware_belt_fusion.alignformer_ext import coupling, verify_cpp

_RIDGE = 1e-6
_GN_ITERATIONS = 6


def _wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def compose(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``X o Y`` for (tx, ty, psi) with ``X(p) = R(psi) p + t``."""
    c, s = np.cos(x[2]), np.sin(x[2])
    return np.array([c * y[0] - s * y[1] + x[0], s * y[0] + c * y[1] + x[1], _wrap(x[2] + y[2])])


def inverse(x: np.ndarray) -> np.ndarray:
    c, s = np.cos(x[2]), np.sin(x[2])
    return np.array([-(c * x[0] + s * x[1]), -(-s * x[0] + c * x[1]), -x[2]])


def adjoint(n: np.ndarray) -> np.ndarray:
    """Maps a small (tx, ty, psi) correction in a frame with pose ``n`` to the outer frame (N T' N^-1)."""
    c, s = np.cos(n[2]), np.sin(n[2])
    return np.array([[c, -s, n[1]], [s, c, -n[0]], [0.0, 0.0, 1.0]])


def _pose_xytheta(decided_psi_t: np.ndarray) -> np.ndarray:
    """verify_core's (psi, tx, ty) -> (tx, ty, psi)."""
    return np.array([decided_psi_t[1], decided_psi_t[2], decided_psi_t[0]], np.float64)


def _edges(rows_e, rows_c, propose, statistic, dof, model, weight_ref, plans) -> Tuple[np.ndarray, List[Optional[np.ndarray]]]:
    """cw-decided pose (tx, ty, psi) per pair (zeros = abstained) and Verify's fit covariance (None if unavailable)."""
    settings = replace(verify_cpp.VerifySettings(), heading_weight=0.0, gated=True)
    decided, refined, engaged, _, inliers, off = verify_cpp.verify(
        rows_e, rows_c, propose, statistic, dof, model.variance_model, model.heading_lambda, settings
    )
    poses = np.array([_pose_xytheta(d) for d in decided]).reshape(-1, 3)
    covariances = []
    for k in range(len(decided)):
        stats = coupling.verify_fit_stats(rows_e[k], rows_c[k], refined[k], inliers[off[k]:off[k + 1]], weight_ref, plans[k])
        accepted = np.any(decided[k] != 0)
        # The fit covariance belongs to Verify's pose; it describes the decided pose only when that is the one applied.
        covariances.append(stats.covariance if (accepted and engaged[k] and stats is not None) else None)
    return poses, covariances


def solve_frame(direct: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]],
                links: List[Tuple[str, str, np.ndarray, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Joint corrections for the CAVs reachable from ego through accepted edges.

    ``direct[cav] = (pose, cov)`` for accepted ego<->CAV edges (cov None: kept fixed at its pose);
    ``links = [(a, b, T_ab, cov), ...]`` accepted CAV<->CAV edges in the ego frame (X_b = X_a o T_ab).
    """
    reach = set(direct)
    changed = True
    while changed:
        changed = False
        for a, b, _, _ in links:
            for u, v in ((a, b), (b, a)):
                if u in reach and v not in reach:
                    reach.add(v)
                    changed = True
    fixed = {c: pose for c, (pose, cov) in direct.items() if cov is None}
    free = sorted(c for c in reach if c not in fixed)
    if not free:
        return dict(fixed)
    index = {c: i for i, c in enumerate(free)}
    # Initialise from direct estimates, then along links.
    x0: Dict[str, np.ndarray] = {c: pose for c, (pose, _) in direct.items()}
    for _ in range(len(reach)):
        for a, b, t_ab, _ in links:
            if a in x0 and b not in x0:
                x0[b] = compose(x0[a], t_ab)
            elif b in x0 and a not in x0:
                x0[a] = compose(x0[b], inverse(t_ab))
    v = np.concatenate([x0[c] for c in free])

    def value(c: str, vec: np.ndarray) -> np.ndarray:
        return fixed[c] if c in fixed else vec[3 * index[c]:3 * index[c] + 3]

    terms = [(c, None, pose, np.linalg.cholesky(np.linalg.inv(cov + _RIDGE * np.eye(3))))
             for c, (pose, cov) in direct.items() if cov is not None]
    terms += [(b, a, t_ab, np.linalg.cholesky(np.linalg.inv(cov + _RIDGE * np.eye(3))))
              for a, b, t_ab, cov in links if a in reach and b in reach]

    def residual(vec: np.ndarray) -> np.ndarray:
        out = []
        for target, source, meas, whiten in terms:
            predicted = meas if source is None else compose(value(source, vec), meas)
            r = value(target, vec) - predicted
            r[2] = _wrap(r[2])
            out.append(whiten.T @ r)
        return np.concatenate(out)

    for _ in range(_GN_ITERATIONS):
        r = residual(v)
        jac = np.empty((len(r), len(v)))
        for j in range(len(v)):
            step = np.zeros_like(v)
            step[j] = 1e-6
            jac[:, j] = (residual(v + step) - r) / 1e-6
        delta = np.linalg.lstsq(jac, -r, rcond=None)[0]
        v = v + delta
        if np.abs(delta).max() < 1e-7:
            break
    out = dict(fixed)
    for c in free:
        out[c] = value(c, v)
    return out


def joint_decisions(inputs, cavcav: Dict[str, np.ndarray], records, fill: bool = True) -> np.ndarray:
    """(P, 3) psi, tx, ty per cached ego<->CAV pair from the joint solve. ``fill=False`` only refines CAVs whose
    own ego<->CAV edge was accepted (CAV<->CAV evidence averages in, but never overrides an abstention)."""
    model, cache, a = inputs.model, inputs.cache, inputs.attempted
    weight_ref = 2.0 * model.variance_model.sigma_translation_m ** 2
    direct_pose, direct_cov = _edges(
        inputs.ego_rows, inputs.cav_rows, cache["propose"][a], cache["statistic"][a], cache["dof"][a],
        model, weight_ref, [inputs.plan(p) for p in a],
    )
    by_frame: Dict[int, Dict] = {}
    for k, p in enumerate(a):
        frame = int(cache["frame"][p])
        cav_id = sorted(records[frame].cav_boxes)[int(cache["cav_rank"][p])]
        by_frame.setdefault(frame, {"direct": {}, "links": [], "pair": {}})["pair"][cav_id] = p
        if np.any(direct_pose[k] != 0):
            by_frame[frame]["direct"][cav_id] = (direct_pose[k], direct_cov[k])

    if len(cavcav["frame"]):
        sizes = cavcav["plan_shape"].prod(axis=1)
        plan_off = np.concatenate([[0], np.cumsum(sizes)])
        rows_a, rows_b, plans = [], [], []
        for q in range(len(cavcav["frame"])):
            record = records[int(cavcav["frame"][q])]
            ca, cb = str(cavcav["cav_a"][q]), str(cavcav["cav_b"][q])
            rows_a.append(verify_cpp.box_rows(record.cav_boxes[ca], record.cav_scores[ca], model.variance_model))
            moved = torch.from_numpy(cavcav["b_boxes"][cavcav["b_off"][q]:cavcav["b_off"][q + 1]])
            rows_b.append(verify_cpp.box_rows(moved, record.cav_scores[cb], model.variance_model))
            plans.append(cavcav["plan"][plan_off[q]:plan_off[q + 1]].reshape(cavcav["plan_shape"][q]))
        link_pose, link_cov = _edges(rows_a, rows_b, cavcav["propose"], cavcav["statistic"], cavcav["dof"],
                                     model, weight_ref, plans)
        for q in range(len(cavcav["frame"])):
            if not np.any(link_pose[q] != 0) or link_cov[q] is None:
                continue
            frame = int(cavcav["frame"][q])
            if frame not in by_frame:
                continue
            n = np.asarray(cavcav["a_to_ego"][q], np.float64)
            t_ab = compose(compose(n, link_pose[q]), inverse(n))   # A-frame correction -> ego frame
            ad = adjoint(n)
            by_frame[frame]["links"].append((str(cavcav["cav_a"][q]), str(cavcav["cav_b"][q]), t_ab, ad @ link_cov[q] @ ad.T))

    out = np.zeros((len(cache["frame"]), 3), np.float32)
    for frame, entry in by_frame.items():
        solved = solve_frame(entry["direct"], entry["links"])
        for cav_id, pose in solved.items():
            if not fill and cav_id not in entry["direct"]:
                continue
            if cav_id in entry["pair"]:
                out[entry["pair"][cav_id]] = (pose[2], pose[0], pose[1])
    return out
