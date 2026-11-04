"""FreeAlign, REIMPLEMENTED: the training-free salient-object-graph competitor.

**This is a reimplementation, not the authors' code.** Nothing from
``github.com/MediaBrain-SJTU/FreeAlign`` is vendored or executed here. The
reference is Lei, Ni, Han, Tang, Wang, Feng, Chen and Wang, "Robust
Collaborative Perception without External Localization and Clock Devices",
ICRA 2024 (arXiv 2405.02965v2); section numbers below are theirs.

This module is itself ported, read-only (via ``git show``), from a labmate's
own careful reimplementation at ``embedding-aware-BELT-fusion`` (branch
``origin/alignformer``,
``src/embedding_aware_belt_fusion/alignformer/freealign.py``) -- his own
explanation of why a port rather than the authors' pipeline, and of the
scope/deviations below, is preserved almost verbatim because it is a correct
account of this port's design too (the algorithm did not change crossing the
repo boundary, only the box layout and the fitted numbers did -- see "What is
NOT ported" below for the one deliberate difference from his module).

Why a port rather than their pipeline (his assessment, from the public repo
over HTTP): it is built on **CoAlign's** OpenCOOD fork with a different
dataset, postprocessor and AP implementation, needs CoAlign's uncertainty
detector (``use_uncertainty: true``, absent here), a ``stage1_boxes.json``
precalc pass, source-built ``g2opy``, and Baidu-Drive checkpoints. Running it
end to end would mean training a detector from scratch and would *still* not be
apples-to-apples. Porting the algorithm onto our detections, our sweep and our
evaluator isolates the alignment algorithm -- the thing under test -- from
detector and evaluator differences.

**Scope, stated so the comparison is not overclaimed.** FreeAlign estimates
relative pose *and clock deviation*; the temporal half (their Section IV-C
"clock deviation estimation", which runs MASS against a historical buffer) is
out of scope here because this project's sweep perturbs pose only and its
frames are synchronous. Their problem is strictly larger than ours.

What is ported, and from where:

1. **Salient-object graph** (Section IV-A). One node per detected box, fully
   connected, edge tensor ``W in R^(n x n x k)``. In the paper ``W`` is
   produced by **EdgeGAT** from the relative-distance matrix ``R`` and trained
   with a contrastive loss (their eq. 2). **The learned edge feature is the one
   part deliberately not ported**: their own Table VI ablation prices it on
   OPV2V at 0.029 deg / 0.283 m / 0.78% error rate for anchor-based matching
   *without* GNN features against 0.017 / 0.266 / 0.56% with them, and their
   released configs set ``gnn: false``, so this is also the path their code
   ships. Absent the GNN, "edge matching is determined by the relative distance
   between two nodes" (their Section V-C), which is :data:`EDGE_DISTANCE`.
   Their shipped ``freealign/graph/greedy_match.py`` actually builds a
   *2-channel* edge -- relative distance **and relative yaw** -- so
   :data:`EDGE_DISTANCE_YAW` is offered as the second, strictly more informed
   variant and both are measured.
2. **MASS**, Multi-Anchor based Subgraph Searching (Section IV-B), all four
   steps: ``n x m`` anchor initialization, anchor-list expansion by edge
   agreement below a threshold up to ``gamma``, incremental subgraph growth,
   and selection of the subgraph with minimal discrepancy
   ``eps = (1/r^p) sum_e eps_e``.
3. **Robust relative pose** (Section IV-C): RANSAC or LMedS over the two
   matched point sets, *not* this project's weighted Kabsch
   (``alignformer_pva.procrustes.weighted_se2_kabsch``). A non-robust
   least-squares core inside the robust loop is expected and is what their
   ``optimize.py`` does with SVD; :func:`se2_from_points` is that core and is
   written here rather than imported from ``alignformer_pva`` so that NONE of
   its policy -- the ``MIN_MATCH_MASS`` gate, heading virtual points,
   inverse-variance weighting, the Verify/Abstain stages -- can leak into the
   competitor. ``tests/test_freealign.py`` pins that it agrees with
   :func:`alignformer_pva.procrustes.weighted_se2_kabsch` at uniform weights,
   which is the honest statement of how much is shared: the closed form, none
   of the policy, and that comparison lives ONLY in the test, never in this
   module's own imports.
4. **The abstain rule.** "If a collaborative message fails to identify a common
   subgraph, whose number of nodes should exceed a predetermined minimum
   threshold, ... FreeAlign will discard this collaborative perception message
   to ensure safety." Here a discarded message is the identity correction
   (zero confidence), so the CAV's boxes are fused uncorrected -- the same
   fallback shape ``alignformer_pva``'s own ``MIN_MATCH_MASS``/abstain
   fallback produces, which is what makes the two coverage figures
   comparable.

**What is NOT ported, beyond the paper's own scope.** The labmate's own
module docstring states his ``FreeAlignConfig`` defaults are "the winning
cell of a 192-point grid searched on [his] scenario-disjoint validation slice"
on V2X-Real, with his own measured numbers quoted inline (down to
"0.0994 m against 0.1004 m" for the robust-estimator choice). Those are HIS
fitted calibration constants, on his detector and his dataset -- reusing them
here would be exactly the mistake ``alignformer_pva/calibrate_variance.py``
was built to avoid for the variance model. This module therefore keeps only
the constants that are attributable to the PAPER or to the AUTHORS' OWN
SHIPPED CODE (``edge_threshold_m=0.5`` from their ``get_best_match(max_error)``,
``min_nodes=3`` from their ``greedy_find_matching``, ``epsilon_offset=100.0``
from their own accumulator seed, ``edge_feature=EDGE_DISTANCE`` from their
shipped ``gnn: false``) as bare dataclass defaults, and changes
``robust_estimator``'s bare default to :data:`RANSAC` (the paper states no
preference between the two -- "offers either") rather than inheriting the
labmate's LMedS pick. The real, reportable FreeAlign row never relies on
these bare defaults: ``calibrate_freealign.py`` in this same package selects
every open parameter on OUR OWN scenario-disjoint validation role, and
``freealign_calibration.py`` feeds that selection back into the evaluator
whole, exactly the way ``alignformer_pva/calibrate_variance.py`` feeds its own
fit into ``train.py``.

**Deviations from the paper, in full.**

- No EdgeGAT (priced above), and no clock-deviation estimation (out of scope).
- Relative yaw in :data:`EDGE_DISTANCE_YAW` is wrapped to ``(-pi, pi]``; the
  shipped code subtracts raw yaws, which is not invariant across the branch cut.
- The paper does not state ``gamma``, the edge threshold, ``p``, or the minimum
  node count. The defaults here are the authors' shipped code's where it has
  one and are otherwise left at a neutral placeholder -- never fit on our data
  as a module-level constant, only via ``calibrate_freealign.py``'s own grid,
  which never looks at test.
- RANSAC enumerates *all* minimal (2-correspondence) samples rather than
  drawing random ones, capped at ``ransac_iterations`` by even subsampling.
  At subgraph sizes seen here that is exhaustive, hence strictly stronger than
  random sampling, and it is deterministic, which random sampling is not.
- Ties in the greedy steps are broken by smallest discrepancy, then by index.
  The paper specifies no order.

The estimate is returned as a :class:`PoseEstimate` in the CAV-to-ego
direction, the same convention the box-geometry side of ``alignformer_pva``
emits, so the evaluation harness can apply it through the very same
``apply_se2_correction`` (``integration/localization.py``) every other
alignment method uses. :class:`PoseEstimate` is defined fresh in THIS module
(not imported from ``alignformer_pva.model``) even though its field names
echo that class, precisely so this package imports nothing from
``alignformer_pva`` at all -- see the package ``__init__.py`` docstring.
Because every edge feature is invariant to the observer's pose, handing this
the CAV boxes already projected by the *noisy* pose does not change the
matching at all -- it only re-parameterizes the answer from an absolute
relative pose into the residual correction our pipeline consumes. That is
pinned by ``test_matching_is_unchanged_by_the_frame_the_cav_boxes_arrive_in``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import torch
from torch import Tensor

# Index of the yaw field in our own [x, y, z, h, w, l, yaw] box layout.
# Confirmed to match the source repo's convention (its boxes.py, BOX_YAW = 6)
# -- the same confirmation alignformer_pva/model.py's own comment records.
# Re-declared locally rather than imported from alignformer_pva, by design:
# see this package's __init__.py docstring on keeping the two packages
# independent.
BOX_YAW = 6

# Edge-feature variants. EDGE_DISTANCE is the paper's training-free path;
# EDGE_DISTANCE_YAW is what their shipped greedy_match.py actually builds.
EDGE_DISTANCE = "distance"
EDGE_DISTANCE_YAW = "distance_yaw"
_EDGE_FEATURES = (EDGE_DISTANCE, EDGE_DISTANCE_YAW)

RANSAC = "ransac"
LMEDS = "lmeds"
_ESTIMATORS = (RANSAC, LMEDS)

# Their get_best_match(max_error=0.5): the edge-discrepancy threshold in metres.
# From the authors' own shipped code, not a data-dependent fit.
DEFAULT_EDGE_THRESHOLD_M = 0.5
# gamma, the anchor-list size. The paper names it but gives no value; 3 anchors
# is the smallest count that pins a subgraph rigidly in the plane -- a
# structural fact, not a fit. calibrate_freealign.py still grids over it on
# our own validation role, for the same reason edge_threshold_m is also
# re-measured even though it, too, has an authors'-code value.
DEFAULT_ANCHOR_LIMIT = 3
# Their greedy_find_matching(min_nodes=3), and the count below which a distance
# graph cannot determine SE(2) at all: 1 node has no edge, 2 nodes have one
# scalar and fix neither rotation nor the reflection. From the authors' own
# shipped code.
DEFAULT_MIN_NODES = 3
# eps = (offset + sum_e eps_e) / r^p, their Section IV-B step iv. The paper's
# literal formula is eps = (1/r^p) sum_e eps_e, i.e. offset = 0, and p is "a
# tunable hyperparameter"; their own get_best_match seeds the accumulator at
# 100 and divides by the match count -- an AUTHORS'-CODE constant, kept here
# for the same reason: with offset = 0 a THREE-node coincidence can score a
# smaller eps than an eleven-node true subgraph and win, which is precisely
# how a naive port turns into a strawman. p = 1 is the paper's own literal
# formula (no preference stated). Both are still re-measured by
# calibrate_freealign.py on our own validation role rather than trusted here.
DEFAULT_EPSILON_POWER = 1.0
DEFAULT_EPSILON_OFFSET = 100.0
# Residual below which a correspondence is a RANSAC inlier, in metres. An
# implementation default, not stated by the paper or measured per-dataset.
DEFAULT_INLIER_THRESHOLD_M = 1.0
DEFAULT_RANSAC_ITERATIONS = 512
# k in their distance_raw[..., 0] + k * distance_raw[..., 1], the relative
# weight of the yaw channel (radians against metres) in EDGE_DISTANCE_YAW.
DEFAULT_YAW_WEIGHT = 1.0

_INFINITY = float("inf")


@dataclass(frozen=True)
class FreeAlignConfig:
    """Every tunable of the port, in one immutable place.

    The bare defaults below are deliberately NOT a calibration result: they
    are either the paper's own stated choice, the authors' shipped-code
    constant where one exists (``edge_threshold_m``, ``min_nodes``,
    ``epsilon_offset``, ``edge_feature``), or (for ``robust_estimator``, where
    the paper states no preference, and ``anchor_limit``/``epsilon_power``,
    which the paper leaves fully open) a neutral placeholder. None of them
    was fit on this project's own detector or data as a hardcoded constant.

    The REAL, reportable configuration comes from ``calibrate_freealign.py``
    in this same package, which grids every one of these over the
    scenario-disjoint OPV2V validation role (never test, and never the
    labmate's own numbers) and writes a ``selected`` block;
    ``freealign_calibration.py`` reads that block back into a
    :class:`FreeAlignConfig` for the evaluator. This mirrors
    ``alignformer_pva/calibrate_variance.py`` feeding its own fit into
    ``train.py`` rather than trusting ``variance.py``'s placeholder constants.
    """

    edge_feature: str = EDGE_DISTANCE
    edge_threshold_m: float = DEFAULT_EDGE_THRESHOLD_M
    anchor_limit: int = DEFAULT_ANCHOR_LIMIT
    min_nodes: int = DEFAULT_MIN_NODES
    epsilon_power: float = DEFAULT_EPSILON_POWER
    epsilon_offset: float = DEFAULT_EPSILON_OFFSET
    # The paper offers either RANSAC or LMedS with no stated preference; kept
    # at RANSAC here rather than the labmate's own validation-selected LMedS
    # (see the module docstring's "What is NOT ported" section).
    robust_estimator: str = RANSAC
    inlier_threshold_m: float = DEFAULT_INLIER_THRESHOLD_M
    ransac_iterations: int = DEFAULT_RANSAC_ITERATIONS
    yaw_weight: float = DEFAULT_YAW_WEIGHT

    def __post_init__(self) -> None:
        if self.edge_feature not in _EDGE_FEATURES:
            raise ValueError(
                f"edge_feature must be one of {_EDGE_FEATURES}, got {self.edge_feature!r}"
            )
        if self.robust_estimator not in _ESTIMATORS:
            raise ValueError(
                f"robust_estimator must be one of {_ESTIMATORS}, "
                f"got {self.robust_estimator!r}"
            )
        if self.anchor_limit < 1:
            raise ValueError(f"anchor_limit must be >= 1, got {self.anchor_limit}")
        if self.min_nodes < 2:
            raise ValueError(
                f"min_nodes must be >= 2 (one node carries no edge), got {self.min_nodes}"
            )

    def to_dict(self) -> dict:
        """JSON-safe record of the configuration, for the result file."""
        return {
            "reimplementation_of": "Lei et al., FreeAlign, ICRA 2024, arXiv 2405.02965",
            "authors_code_used": False,
            "gnn_edge_features": False,
            "edge_feature": self.edge_feature,
            "edge_threshold_m": self.edge_threshold_m,
            "anchor_limit_gamma": self.anchor_limit,
            "min_nodes": self.min_nodes,
            "epsilon_power": self.epsilon_power,
            "epsilon_offset": self.epsilon_offset,
            "robust_estimator": self.robust_estimator,
            "inlier_threshold_m": self.inlier_threshold_m,
            "ransac_iterations": self.ransac_iterations,
            "yaw_weight": self.yaw_weight,
        }


@dataclass(frozen=True)
class FreeAlignMatch:
    """The common subgraph MASS selected, and its discrepancy ``eps``.

    ``ego_indices`` and ``cav_indices`` are parallel ``(r,)`` long tensors; an
    empty pair of them means no subgraph met ``min_nodes`` and the message is
    to be discarded.
    """

    ego_indices: Tensor
    cav_indices: Tensor
    epsilon: float

    def __len__(self) -> int:
        return int(self.ego_indices.shape[0])


@dataclass(frozen=True)
class PoseEstimate:
    """A CAV's estimated SE(2) correction, with the evidence behind it.

    Field names deliberately match ``alignformer_pva.model.PoseEstimate``'s
    ``psi``/``t``/``confidence`` (so the evaluation harness's existing
    ``apply_se2_correction``/"accepted" glue reads this the same way) but
    this is an independent class: FreeAlign has no Verify/Abstain-stage
    statistic (``offset_statistic``/``offset_dof``), so those fields simply
    do not exist here rather than being carried as ``None`` placeholders.
    """

    psi: Tensor
    t: Tensor
    confidence: Tensor


def is_fallback(estimate: PoseEstimate) -> Tensor:
    """``(B,)`` bool: which samples FreeAlign discarded (no matched subgraph).

    FreeAlign's abstain rule produces the identity correction with zero
    confidence for a discarded message (see :func:`freealign_estimate`), so
    zero confidence IS the fallback indicator -- there is no separate flag to
    track. Matches the shape of ``alignformer_pva``'s own fallback check
    (``torch.any(psi != 0) or torch.any(t != 0)`` in ``inference.py``), stated
    as a named predicate here because the tests and the calibration script
    both need it repeatedly.
    """
    return estimate.confidence <= 0.0


def edge_features(boxes: Tensor, config: FreeAlignConfig) -> Tensor:
    """``(n, n, k)`` pose-invariant edge features of one agent's salient-object graph.

    Nodes are the agent's detected boxes; the graph is fully connected
    (Section IV-A). Channel 0 is the relative distance ``R`` -- which in the
    training-free path *is* the edge feature -- and, under
    :data:`EDGE_DISTANCE_YAW`, channel 1 is the wrapped relative yaw the
    authors' ``greedy_match.py`` adds.
    """
    centres = boxes[:, :2]
    separation = centres.unsqueeze(1) - centres.unsqueeze(0)
    distance = separation.norm(dim=-1, p=2).unsqueeze(-1)
    if config.edge_feature == EDGE_DISTANCE:
        return distance

    yaws = boxes[:, BOX_YAW]
    difference = yaws.unsqueeze(1) - yaws.unsqueeze(0)
    wrapped = torch.atan2(torch.sin(difference), torch.cos(difference))
    return torch.cat([distance, wrapped.unsqueeze(-1)], dim=-1)


def _edge_discrepancy(
    ego_edges: Tensor, cav_edges: Tensor, config: FreeAlignConfig
) -> Tensor:
    """``(n, n, m, m)`` of ``eps_(p,u),(q,v) = |W_i(p, u) - W_j(q, v)|``.

    Their Section IV-B quantity, channel-summed with the yaw channel weighted
    by ``yaw_weight`` exactly as their ``distance_raw[..., 0] + k *
    distance_raw[..., 1]`` does.
    """
    weights = [1.0, config.yaw_weight][: ego_edges.shape[-1]]
    total = None
    for channel, weight in enumerate(weights):
        term = (
            ego_edges[:, :, channel].unsqueeze(-1).unsqueeze(-1)
            - cav_edges[:, :, channel].unsqueeze(0).unsqueeze(0)
        ).abs()
        total = term * weight if total is None else total + term * weight
    return total


def _greedy_pick(cost: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """One one-to-one greedy step over ``(S, n, m)`` costs, in place on ``cost``.

    Returns ``(u, v, taken)``: the lowest-cost candidate per seed and whether it
    was finite. The chosen row and column are struck out so the next step
    cannot reuse either node, which is what keeps a subgraph an injection.
    """
    seeds, _, columns = cost.shape
    flat = cost.reshape(seeds, -1)
    value, index = flat.min(dim=1)
    u, v = index // columns, index % columns
    taken = torch.isfinite(value)

    rows = torch.arange(seeds, device=cost.device)
    cost[rows, u, :] = _INFINITY
    cost[rows, :, v] = _INFINITY
    return u, v, taken


def mass_common_subgraph(
    ego_edges: Tensor, cav_edges: Tensor, config: FreeAlignConfig
) -> FreeAlignMatch:
    """Multi-Anchor based Subgraph Searching (their Section IV-B), all four steps.

    Every step below is run for all ``n x m`` seed anchors at once rather than
    in a Python loop over them; the semantics are the paper's, the batching is
    an implementation detail.
    """
    n, m = ego_edges.shape[0], cav_edges.shape[0]
    device = ego_edges.device
    empty = torch.zeros(0, dtype=torch.long, device=device)
    if n == 0 or m == 0 or min(n, m) < config.min_nodes:
        return FreeAlignMatch(empty, empty, _INFINITY)

    threshold = config.edge_threshold_m
    discrepancy = _edge_discrepancy(ego_edges, cav_edges, config)
    # (p, u, q, v) -> (p, q, u, v): one (n, m) candidate map per seed anchor.
    seeded = discrepancy.permute(0, 2, 1, 3).reshape(n * m, n, m)
    seeds = n * m
    seed_rows = torch.arange(seeds, device=device)
    seed_p, seed_q = seed_rows // m, seed_rows % m

    # i. Initialization -- the n x m potential anchor pairs are the rows of
    # `seeded`. ii. Anchor list expansion -- add the pairs whose edge
    # discrepancy against the anchors so far is below the threshold, until
    # `anchor_limit` (gamma) anchors are held.
    available = torch.where(seeded < threshold, seeded, torch.full_like(seeded, _INFINITY))
    available[seed_rows, seed_p, :] = _INFINITY
    available[seed_rows, :, seed_q] = _INFINITY

    anchor_constraint = seeded.clone()  # the seed anchor's own constraint map
    for _ in range(config.anchor_limit - 1):
        u, v, taken = _greedy_pick(available)
        if not bool(taken.any()):
            break
        added = seeded.index_select(0, u * m + v)
        anchor_constraint = torch.maximum(
            anchor_constraint,
            torch.where(
                taken.reshape(-1, 1, 1), added, torch.full_like(added, -_INFINITY)
            ),
        )

    # iii. Subgraph search -- incrementally add the node pairs that agree with
    # EVERY anchor, until none is left that meets the criterion.
    cost = torch.where(
        anchor_constraint < threshold,
        anchor_constraint,
        torch.full_like(anchor_constraint, _INFINITY),
    )
    chosen_u, chosen_v, chosen_ok = [], [], []
    for _ in range(min(n, m)):
        u, v, taken = _greedy_pick(cost)
        if not bool(taken.any()):
            break
        chosen_u.append(u)
        chosen_v.append(v)
        chosen_ok.append(taken)
    if not chosen_u:
        return FreeAlignMatch(empty, empty, _INFINITY)

    members_u = torch.stack(chosen_u)  # (R, S)
    members_v = torch.stack(chosen_v)
    members_ok = torch.stack(chosen_ok)
    size = members_ok.sum(dim=0)  # r per seed

    # iv. Selection -- eps = (offset + sum_e eps_e) / r^p over the subgraph's
    # own edges; keep the one with minimal eps among those meeting min_nodes.
    # See DEFAULT_EPSILON_OFFSET: the constant decides whether a three-node
    # coincidence can outscore the true eleven-node subgraph.
    rounds = members_u.shape[0]
    flat_discrepancy = discrepancy.reshape(-1)
    left, right = torch.triu_indices(rounds, rounds, offset=1, device=device)
    if left.numel() == 0:
        total = torch.zeros(seeds, device=device, dtype=discrepancy.dtype)
    else:
        index = (
            (members_u[left] * n + members_u[right]) * m + members_v[left]
        ) * m + members_v[right]
        present = members_ok[left] & members_ok[right]
        total = (flat_discrepancy[index] * present).sum(dim=0)

    epsilon = (total + config.epsilon_offset) / (
        size.clamp_min(1).to(total.dtype) ** config.epsilon_power
    )
    epsilon = torch.where(
        size >= config.min_nodes, epsilon, torch.full_like(epsilon, _INFINITY)
    )
    best = int(epsilon.argmin())
    if not torch.isfinite(epsilon[best]):
        return FreeAlignMatch(empty, empty, _INFINITY)

    keep = members_ok[:, best]
    return FreeAlignMatch(
        ego_indices=members_u[keep, best],
        cav_indices=members_v[keep, best],
        epsilon=float(epsilon[best]),
    )


def se2_from_points(p: Tensor, q: Tensor) -> Tuple[Tensor, Tensor]:
    """Unweighted closed-form SE(2) taking ``q`` onto ``p``; the robust loop's core.

    Minimizes ``sum_n |R(psi) q_n + t - p_n|^2``. Written here rather than
    imported from ``alignformer_pva.procrustes`` so that none of
    AlignFormer-PVA's policy -- the ``MIN_MATCH_MASS`` gate, heading virtual
    points, the inverse-variance weighting -- can reach the competitor. It is
    nevertheless the same standard closed form, which is what FreeAlign's
    ``optimize.py`` computes by SVD, and ``tests/test_freealign.py`` pins the
    agreement at uniform weights.
    """
    if p.shape != q.shape or p.dim() != 2 or p.shape[-1] != 2:
        raise ValueError(f"expected matching (N, 2) point sets, got {tuple(p.shape)} "
                         f"and {tuple(q.shape)}")
    p_bar, q_bar = p.mean(dim=0), q.mean(dim=0)
    dp, dq = p - p_bar, q - q_bar
    cross = (dq[:, 0] * dp[:, 1] - dq[:, 1] * dp[:, 0]).sum()
    dot = (dq[:, 0] * dp[:, 0] + dq[:, 1] * dp[:, 1]).sum()
    psi = torch.atan2(cross, dot)
    cos, sin = torch.cos(psi), torch.sin(psi)
    rotated = torch.stack(
        [cos * q_bar[0] - sin * q_bar[1], sin * q_bar[0] + cos * q_bar[1]]
    )
    return psi, p_bar - rotated


def _se2_from_point_pairs(p: Tensor, q: Tensor) -> Tuple[Tensor, Tensor]:
    """:func:`se2_from_points` over a leading sample dimension: ``(S, N, 2)`` in."""
    p_bar, q_bar = p.mean(dim=1), q.mean(dim=1)
    dp, dq = p - p_bar.unsqueeze(1), q - q_bar.unsqueeze(1)
    cross = (dq[..., 0] * dp[..., 1] - dq[..., 1] * dp[..., 0]).sum(dim=1)
    dot = (dq[..., 0] * dp[..., 0] + dq[..., 1] * dp[..., 1]).sum(dim=1)
    psi = torch.atan2(cross, dot)
    cos, sin = torch.cos(psi), torch.sin(psi)
    rotated = torch.stack(
        [cos * q_bar[:, 0] - sin * q_bar[:, 1], sin * q_bar[:, 0] + cos * q_bar[:, 1]],
        dim=-1,
    )
    return psi, p_bar - rotated


def _residuals_batch(p: Tensor, q: Tensor, psi: Tensor, t: Tensor) -> Tensor:
    """``(S, N)`` per-point residual of ``S`` candidate transforms on one point set."""
    cos, sin = torch.cos(psi).unsqueeze(-1), torch.sin(psi).unsqueeze(-1)
    x, y = q[:, 0].unsqueeze(0), q[:, 1].unsqueeze(0)
    moved = torch.stack([cos * x - sin * y, sin * x + cos * y], dim=-1)
    return (moved + t.unsqueeze(1) - p.unsqueeze(0)).norm(dim=-1)


def _minimal_samples(count: int, limit: int, device) -> Tensor:
    """``(S, 2)`` minimal samples: every pair of correspondences, evenly capped.

    Two correspondences determine an SE(2) exactly, so this is the minimal
    sample set RANSAC and LMedS draw from. Enumerating rather than drawing at
    random makes the port deterministic and, at the subgraph sizes seen here,
    exhaustive -- strictly stronger than sampling.
    """
    first, second = torch.triu_indices(count, count, offset=1, device=device)
    pairs = torch.stack([first, second], dim=1)
    if pairs.shape[0] <= limit:
        return pairs
    stride = torch.linspace(0, pairs.shape[0] - 1, limit, device=device).long()
    return pairs[stride]


def robust_se2(
    p: Tensor, q: Tensor, config: FreeAlignConfig
) -> Tuple[Tensor, Tensor, Tensor]:
    """RANSAC (or LMedS) SE(2) over the matched point sets; their Section IV-C.

    Returns ``(psi, t, inliers)``. The consensus set is refit by ordinary least
    squares, which is the usual and expected shape of a RANSAC estimator.
    """
    count = p.shape[0]
    if count < 2:
        psi, t = se2_from_points(p, q) if count == 1 else (
            torch.zeros((), device=p.device), torch.zeros(2, device=p.device)
        )
        return psi, t, torch.ones(count, dtype=torch.bool, device=p.device)

    samples = _minimal_samples(count, config.ransac_iterations, p.device)
    # Every minimal sample is fitted and scored at once: a Python loop over
    # samples would dominate the whole sweep's runtime and change no answer.
    psi, t = _se2_from_point_pairs(p[samples], q[samples])
    residual = _residuals_batch(p, q, psi, t)
    inliers = residual < config.inlier_threshold_m
    if config.robust_estimator == RANSAC:
        # Inlier count first, total inlier residual as the tie-break.
        counts = inliers.sum(dim=1)
        contenders = counts == counts.max()
        tie_break = torch.where(
            contenders,
            (residual * inliers).sum(dim=1),
            torch.full_like(counts, _INFINITY, dtype=residual.dtype),
        )
    else:
        tie_break = residual.median(dim=1).values
    chosen = int(torch.nonzero(tie_break == tie_break.min())[0])
    best_inliers = inliers[chosen]

    if int(best_inliers.sum()) < 2:
        best_inliers = torch.ones(count, dtype=torch.bool, device=p.device)
    psi, t = se2_from_points(p[best_inliers], q[best_inliers])
    return psi, t, best_inliers


def freealign_estimate(
    batch: Mapping[str, Tensor], config: FreeAlignConfig
) -> PoseEstimate:
    """The CAV-to-ego correction FreeAlign would emit for each pair in ``batch``.

    ``batch`` is the same object-set layout ``alignformer_pva``'s model
    consumes: ``ego_boxes``/``cav_boxes`` ``(B, N, 7)`` and
    ``ego_mask``/``cav_mask`` ``(B, N)`` bool; only those four keys are read,
    because a boxes-only method has no use for scores or embeddings.
    ``confidence`` carries the matched node count, and a discarded message
    (fewer than ``min_nodes`` matched nodes) comes back as the identity
    correction with zero confidence, which is what :func:`is_fallback` reads
    and what makes the coverage of the two methods directly comparable.
    """
    ego_boxes, cav_boxes = batch["ego_boxes"], batch["cav_boxes"]
    size, device, dtype = ego_boxes.shape[0], ego_boxes.device, ego_boxes.dtype
    psi = torch.zeros(size, device=device, dtype=dtype)
    translation = torch.zeros(size, 2, device=device, dtype=dtype)
    confidence = torch.zeros(size, device=device, dtype=dtype)

    for index in range(size):
        ego = ego_boxes[index][batch["ego_mask"][index]]
        cav = cav_boxes[index][batch["cav_mask"][index]]
        if ego.shape[0] == 0 or cav.shape[0] == 0:
            continue

        match = mass_common_subgraph(
            edge_features(ego, config), edge_features(cav, config), config
        )
        if len(match) < config.min_nodes:
            continue  # the message is DISCARDED: fuse the CAV uncorrected

        estimated_psi, estimated_t, _ = robust_se2(
            ego[match.ego_indices, :2], cav[match.cav_indices, :2], config
        )
        psi[index] = estimated_psi
        translation[index] = estimated_t
        confidence[index] = float(len(match))

    return PoseEstimate(psi=psi, t=translation, confidence=confidence)
