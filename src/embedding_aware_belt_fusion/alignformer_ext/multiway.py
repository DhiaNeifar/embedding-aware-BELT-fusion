"""Joint matching of ALL agents' boxes in a frame: a multi-marginal Sinkhorn over ego + every CAV at once.

Pairwise AlignFormer matches ego<->CAV_k for each k separately: a log-Sinkhorn with dustbins (SuperGlue) over the
learned scores s_0k gives a soft assignment P_0k, and Propose solves CAV k's pose from it. With K agents, the same
object may be seen by the ego, CAV A and CAV B at once, and the K(K-1)/2 pairwise matchings say more together than
apart: if ego box i matches B's box l and B's box l matches A's box j, then ego box i should match A's box j, even
when s_0A(i, j) alone is weak. Pairwise plans do not have to agree with each other; true matchings must
(cycle consistency).

Here the matching is one joint distribution over TUPLES: one index per agent, each either a detection or that
agent's dustbin ("not seen by this agent"). A tuple is one object hypothesis. Its log-potential is the sum of the
pairwise scores of every agent pair in it,

    log K(i_0, ..., i_{K-1}) = sum_{k < l} phi_kl(i_k, i_l),

and the plan is the entropic multi-marginal optimal transport with the dustbin marginals of SuperGlue generalized
to K agents: every real detection of agent k has mass 1, agent k's dustbin has mass S - n_k (S = sum_k n_k, the
largest possible number of distinct objects), so every marginal has total mass S. It is solved by iterative
scaling (multi-marginal Sinkhorn): for each agent in turn, rescale the tensor so its marginal is right.

* The pairwise marginals of a joint distribution are mutually consistent by construction: P_0A, P_0B and P_AB are
  read off the same tuples, so the ego<->A matching now uses B's evidence and vice versa.
* phi_kl is the cached pairwise log-plan. log P_kl = s_kl + u_k + v_l (Sinkhorn only rescales rows and columns), and
  the multi-marginal scaling absorbs every per-detection term, so this equals using the network's raw scores and
  learned dustbin score: no retraining, and no new parameter.
* With two agents (K = 2) the iteration's fixed point is the pairwise plan itself: the method reduces exactly to
  AlignFormer's own Sinkhorn.

The ego<->CAV marginals then replace the pairwise plans, Propose's pose and Wald statistic are re-solved from them by
the model's own estimator (uncertainty.propose), and Verify / Abstain / the temporal layer run unchanged.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import numpy as np
from scipy.special import logsumexp


def multimarginal_sinkhorn(potentials: Dict[Tuple[int, int], np.ndarray], sizes: List[int],
                           iterations: int = 100, tolerance: float = 1e-4) -> np.ndarray:
    """Log joint plan over tuples, shape (n_0 + 1, ..., n_{K-1} + 1), last index of every axis = dustbin.

    ``potentials[(k, l)]`` (k < l) is an (n_k + 1, n_l + 1) log-potential (dustbins last); a missing pair adds
    nothing. Marginal of agent k: 1 per detection, S - n_k on its dustbin, S = sum(sizes).
    """
    shape = tuple(n + 1 for n in sizes)
    k_agents = len(sizes)
    log_plan = np.zeros(shape, np.float64)
    for (k, l), phi in potentials.items():
        view = [1] * k_agents
        view[k], view[l] = shape[k], shape[l]
        log_plan += phi.astype(np.float64).reshape(view)
    total = float(sum(sizes))
    targets = []
    for n in sizes:
        target = np.zeros(n + 1)
        target[n] = np.log(total - n) if total > n else -np.inf
        targets.append(target)
    for _ in range(iterations):
        worst = 0.0
        for k in range(k_agents):
            others = tuple(a for a in range(k_agents) if a != k)
            marginal = logsumexp(log_plan, axis=others)
            delta = targets[k] - marginal
            delta[~np.isfinite(delta)] = 0.0
            view = [1] * k_agents
            view[k] = shape[k]
            log_plan += delta.reshape(view)
            worst = max(worst, float(np.max(np.abs(np.expm1(-delta[:sizes[k]])))) if sizes[k] else 0.0)
        if worst < tolerance:
            break
    return log_plan


def pairwise_marginal(log_plan: np.ndarray, k: int, l: int) -> np.ndarray:
    """(n_k + 1, n_l + 1) log pairwise plan of agents k < l read off the joint plan."""
    others = tuple(a for a in range(log_plan.ndim) if a not in (k, l))
    return logsumexp(log_plan, axis=others) if others else log_plan


def candidate_tuples(sizes: List[int], allowed: Dict[Tuple[int, int], np.ndarray]) -> np.ndarray:
    """(T, K) tuples whose real members are pairwise plausible (``allowed[(k, l)][i, j]``; a pair with no entry
    allows everything); index n_k = agent k's dustbin. Contains every singleton and the all-dustbin tuple."""
    partial = [()]
    for k, n in enumerate(sizes):
        grown = []
        for prefix in partial:
            mask = np.ones(n, bool)
            for l, i in enumerate(prefix):
                if i < sizes[l] and (l, k) in allowed:
                    mask &= allowed[(l, k)][i]
            grown.append(prefix + (n,))
            grown.extend(prefix + (int(j),) for j in np.flatnonzero(mask))
        partial = grown
    return np.asarray(partial, np.int64).reshape(-1, len(sizes))


def _grouped_logsumexp(values: np.ndarray, order: np.ndarray, starts: np.ndarray, groups: np.ndarray,
                       present: np.ndarray, size: int) -> np.ndarray:
    peak = np.full(size, -np.inf)
    peak[present] = np.maximum.reduceat(values[order], starts)
    total = np.bincount(groups, weights=np.exp(values - peak[groups]), minlength=size)
    with np.errstate(divide="ignore"):
        return np.log(total) + peak


def sparse_multimarginal_sinkhorn(potentials: Dict[Tuple[int, int], np.ndarray], sizes: List[int],
                                  threshold: float = 1e-3, iterations: int = 200, tolerance: float = 1e-4,
                                  max_tuples: int = 2_000_000, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """``multimarginal_sinkhorn`` restricted to tuples whose real members are pairwise plausible (pairwise plan
    probability > ``threshold``; every other tuple gets probability 0). Returns the tuples (T, K) and their log mass.
    ``threshold = 0`` keeps every tuple and gives the dense result. ``scale`` multiplies the potentials (tempering);
    plausibility is always judged on the unscaled pairwise plans, so tempering does not enlarge the tuple set."""
    while True:
        allowed = {key: np.exp(phi[:-1, :-1]) > threshold for key, phi in potentials.items()}
        tuples = candidate_tuples(sizes, allowed)
        if len(tuples) <= max_tuples or threshold >= 0.5:
            break
        threshold *= 10.0
    log_mass = np.zeros(len(tuples), np.float64)
    for (k, l), phi in potentials.items():
        log_mass += scale * phi.astype(np.float64)[tuples[:, k], tuples[:, l]]
    total = float(sum(sizes))
    groups = []
    for k, n in enumerate(sizes):
        order = np.argsort(tuples[:, k], kind="stable")
        present, starts = np.unique(tuples[order, k], return_index=True)
        target = np.zeros(n + 1)
        target[n] = np.log(total - n) if total > n else -np.inf
        groups.append((order, starts, tuples[:, k], present, n + 1, target))
    for _ in range(iterations):
        worst = 0.0
        for k, (order, starts, index, present, size, target) in enumerate(groups):
            marginal = _grouped_logsumexp(log_mass, order, starts, index, present, size)
            delta = target - marginal
            delta[~np.isfinite(delta)] = 0.0
            log_mass += delta[index]
            if sizes[k]:
                worst = max(worst, float(np.max(np.abs(np.expm1(-delta[:sizes[k]])))))
        if worst < tolerance:
            break
    return tuples, log_mass


def sparse_pairwise_marginal(tuples: np.ndarray, log_mass: np.ndarray, sizes: List[int], k: int, l: int,
                             floor: float = -1e4) -> np.ndarray:
    """(n_k + 1, n_l + 1) log pairwise plan of agents k < l from a sparse joint plan (absent pairs -> ``floor``)."""
    rows, cols = sizes[k] + 1, sizes[l] + 1
    key = tuples[:, k] * cols + tuples[:, l]
    order = np.argsort(key, kind="stable")
    present, starts = np.unique(key[order], return_index=True)
    out = _grouped_logsumexp(log_mass, order, starts, key, present, rows * cols)
    out[~np.isfinite(out)] = floor
    return out.reshape(rows, cols)


def joint_plans(inputs, frames: dict = None, temper: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Cache-layout flat plan array with every ego<->CAV plan of a multi-CAV frame replaced by its joint-plan
    marginal, and the boolean mask of the pairs that changed. ``frames`` (a dict), when given, receives
    frame -> (agents [(pair, cav id)], sizes, tuples, log mass) of every jointly matched frame. ``temper`` scales
    every pairwise potential by 1 / (K - 1): each agent then contributes one pairwise term's worth of evidence per
    tuple instead of K - 1 (no double counting of the pairwise posteriors); K = 2 is unchanged."""
    cache, cavcav, records = inputs.cache, inputs.cavcav, inputs.records
    plan = cache["plan"].copy()
    changed = np.zeros(len(cache["frame"]), bool)
    by_frame: Dict[int, List[int]] = {}
    for p in inputs.attempted:
        by_frame.setdefault(int(cache["frame"][p]), []).append(int(p))
    cc_off = np.concatenate([[0], np.cumsum(cavcav["plan_shape"].prod(axis=1))])
    links: Dict[Tuple[int, str, str], int] = {}
    for q in range(len(cavcav["frame"])):
        links[(int(cavcav["frame"][q]), str(cavcav["cav_a"][q]), str(cavcav["cav_b"][q]))] = q
    for frame, pairs in by_frame.items():
        record = records[frame]
        ids = sorted(record.cav_boxes)
        agents = [(p, ids[int(cache["cav_rank"][p])]) for p in pairs if cache["plan_shape"][p][1] > 1]
        if len(agents) < 2:
            continue
        sizes = [int(cache["plan_shape"][agents[0][0]][0]) - 1] + [int(cache["plan_shape"][p][1]) - 1 for p, _ in agents]
        potentials = {(0, k + 1): inputs.plan(p) for k, (p, _) in enumerate(agents)}
        for k, (_, a) in enumerate(agents):
            for l, (_, b) in enumerate(agents):
                q = links.get((frame, a, b)) if k < l else None
                if q is None:
                    continue
                shape = tuple(cavcav["plan_shape"][q])
                if shape != (sizes[k + 1] + 1, sizes[l + 1] + 1):
                    continue
                potentials[(k + 1, l + 1)] = cavcav["plan"][cc_off[q]:cc_off[q + 1]].reshape(shape)
        if len(potentials) == len(agents):          # no CAV<->CAV evidence: the joint plan is the pairwise one
            continue
        scale = 1.0 / (len(sizes) - 1) if temper else 1.0
        tuples, log_mass = sparse_multimarginal_sinkhorn(potentials, sizes, scale=scale)
        for k, (p, _) in enumerate(agents):
            marginal = sparse_pairwise_marginal(tuples, log_mass, sizes, 0, k + 1)
            plan[inputs.plan_off[p]:inputs.plan_off[p + 1]] = marginal.astype(np.float32).ravel()
            changed[p] = True
        if frames is not None:
            frames[frame] = (agents, sizes, tuples, log_mass)
    return plan, changed


def with_joint_plans(inputs, temper: bool = False):
    """A copy of ``inputs`` whose cache carries the joint-plan marginals and the Propose output re-solved from them."""
    import torch

    from embedding_aware_belt_fusion.alignformer_ext import uncertainty

    if inputs.cavcav is None:
        raise FileNotFoundError("joint matching needs the CAV<->CAV cache: build_propose_cache.py --cavcav")
    frames: dict = {}
    plan, changed = joint_plans(inputs, frames, temper=temper)
    cache = dict(inputs.cache)
    cache["plan"] = plan
    cache["propose"] = cache["propose"].copy()
    cache["statistic"] = cache["statistic"].copy()
    cache["dof"] = cache["dof"].copy()
    out = copy.copy(inputs)
    out.cache = cache
    index = {int(p): i for i, p in enumerate(inputs.attempted)}
    with torch.no_grad():
        for p in np.flatnonzero(changed):
            psi, tx, ty, statistic, dof = uncertainty.propose(out.plan(p), inputs.batches[index[p]], inputs.model,
                                                              inputs.robust_config, spread=False)
            cache["propose"][p] = (psi, tx, ty)
            cache["statistic"][p], cache["dof"][p] = statistic, dof
    out.joint_changed = changed
    out.joint_frames = frames
    return out


def with_joint_registration(inputs, rounds: int = 3, relay_mass: float = None):
    """``with_joint_plans`` plus a joint pose solve (generalized Procrustes with shared virtual partners).

    Each CAV k is fitted not only to the ego's boxes (weights: the joint plan's ego<->k marginal) but to every
    other CAV l's boxes placed with l's current correction (weights: the k<->l marginal). An object seen by k and l
    but not by the ego still constrains k once l is registered, so a CAV with little direct overlap with the ego is
    registered through the others. Block-coordinate rounds (each CAV in turn, the others fixed) with the model's own
    estimator (virtual partners, heading points, variance weighting, Huber IRLS, Wald statistic); ego is fixed.

    With ``relay_mass`` (relay mode), only a CAV whose own matched mass with the ego is below ``relay_mass`` is
    re-solved, and only through CAVs whose own mass with the ego is at least ``relay_mass`` (anchored by the ego
    themselves): other agents are used where a CAV's own evidence cannot fix its pose, and only through agents
    that do not need help in turn. Anchors are not changed, so one round is exact.
    """
    import torch

    from embedding_aware_belt_fusion.alignformer_ext import evaluate, uncertainty
    from embedding_aware_belt_fusion.alignformer_pva.inference import build_pairwise_batch

    out = with_joint_plans(inputs)
    cache, records = out.cache, inputs.records
    with torch.no_grad():
        for frame, (agents, sizes, tuples, log_mass) in out.joint_frames.items():
            record = records[frame]
            boxes = [torch.from_numpy(cache["cav_boxes"][cache["cav_off"][p]:cache["cav_off"][p + 1]]) for p, _ in agents]
            scores = [record.cav_scores[cav_id] for _, cav_id in agents]
            weights = {}
            for k in range(len(agents)):
                weights[("ego", k)] = np.exp(sparse_pairwise_marginal(tuples, log_mass, sizes, 0, k + 1))[:-1, :-1]
                for l in range(len(agents)):
                    if l != k:
                        a, b = min(k, l), max(k, l)
                        m = np.exp(sparse_pairwise_marginal(tuples, log_mass, sizes, a + 1, b + 1))[:-1, :-1]
                        weights[(l, k)] = m if l < k else m.T        # rows: l's boxes, columns: k's boxes
            direct = [float(weights[("ego", k)].sum()) for k in range(len(agents))]
            anchored = [relay_mass is None or m >= relay_mass for m in direct]
            for _ in range(1 if relay_mass is not None else rounds):
                for k, (p, _) in enumerate(agents):
                    if relay_mass is not None and anchored[k]:
                        continue
                    target_boxes, target_scores, rows = [record.ego_boxes], [record.ego_scores], [weights[("ego", k)]]
                    for l, (q, _) in enumerate(agents):
                        if l == k or not anchored[l]:
                            continue
                        psi, tx, ty = map(float, cache["propose"][q])
                        target_boxes.append(evaluate.apply_correction(boxes[l], psi, tx, ty))
                        target_scores.append(scores[l])
                        rows.append(weights[(l, k)])
                    w = np.concatenate(rows, axis=0)
                    plan = np.full((w.shape[0] + 1, w.shape[1] + 1), -30.0, np.float32)
                    plan[:-1, :-1] = np.log(np.clip(w, 1e-30, None))
                    batch = build_pairwise_batch(
                        {"boxes": torch.cat(target_boxes), "scores": torch.cat(target_scores)},
                        {"boxes": boxes[k], "scores": scores[k]},
                    )
                    psi, tx, ty, statistic, dof = uncertainty.propose(plan, batch, inputs.model, inputs.robust_config,
                                                                      spread=False)
                    cache["propose"][p] = (psi, tx, ty)
                    cache["statistic"][p], cache["dof"][p] = statistic, dof
    return out
