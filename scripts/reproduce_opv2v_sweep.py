#!/usr/bin/env python3
"""Standalone OPV2V noise/delay sweep: uncorrected, oracle, FreeAlign, AlignFormer.

Does not import or modify evaluation/belt_fusion.py. Detector inference runs
only once, reused from the existing outputs/r140_geometry_test_cache (pure
per-agent local boxes/scores, no noise baked in). Everything else --
pose-noise injection, correction (FreeAlign / AlignFormer), late fusion, and
AP -- runs fresh per (sigma, seed) through the compiled fusion_core
extension, so a full sweep needs no further GPU/detector time.

Noise model matches Chenyi's own (embedding_aware_belt_fusion.alignformer.
coloca.geometry.perturb_pose_2d / alignformer.draws.sweep_noisy_poses):
independent zero-mean Gaussian noise added to each CAV's own ABSOLUTE WORLD
pose (x, y in metres, yaw in degrees, sigma_yaw_deg = sigma_xy_m), then the
ego-relative transform is recomputed via x1_to_x2(noisy_world_pose,
ego_world_pose) -- perturbing in world frame and re-deriving the relative
transform is what makes the noise's effective direction in the ego frame
correctly depend on the ego's own orientation, which perturbing the relative
transform directly would get wrong. The RNG stream itself is ours, not his,
so individual seeds are not bit-identical to his -- this reproduces his
protocol and noise *model*, not his specific draws.

Ground-truth and per-frame agent selection are verified byte-identical to
Chenyi's own evaluation harness (opencood.data_utils.datasets.LateFusionDataset
.get_item_single_car + COM_RANGE filtering + generate_gt_bbx), confirmed by
direct comparison over 50 frames before this script was written.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Every call below processes one frame's handful of boxes at a time; PyTorch's
# default intra-op thread pool (one per core) pays synchronization overhead on
# every tiny op with no parallel work to show for it. Measured: 1063ms/frame
# -> 2.9ms/frame for FreeAlign, 1467ms/frame -> 12.5ms/frame for AlignFormer,
# an ~100-370x speedup, with 24 default threads on this machine. Must be set
# before any tensor op runs, so this import-time call is deliberate.
torch.set_num_threads(1)
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "external" / "OpenCOOD"))
sys.path.insert(0, str(REPO_ROOT / "src" / "embedding_aware_belt_fusion" / "fusion_core" / "build"))

import fusion_core  # noqa: E402
from opencood.hypes_yaml.yaml_utils import load_yaml  # noqa: E402
from opencood.data_utils.datasets import build_dataset  # noqa: E402
import opencood.data_utils.datasets as ocd  # noqa: E402
from opencood.utils.transformation_utils import x1_to_x2  # noqa: E402
from opencood.utils import box_utils, common_utils  # noqa: E402

from embedding_aware_belt_fusion.embeddings.geometry import transform_boxes_to_ego  # noqa: E402
from embedding_aware_belt_fusion.freealign.freealign import FreeAlignConfig  # noqa: E402
from embedding_aware_belt_fusion.freealign.freealign_calibration import (  # noqa: E402
    freealign_config_from_calibration,
)
from embedding_aware_belt_fusion.freealign.inference import (  # noqa: E402
    correct_simple_detections as freealign_correct,
)
from embedding_aware_belt_fusion.alignformer_pva.train import load_alignformer_pva  # noqa: E402
from embedding_aware_belt_fusion.alignformer_pva.inference import (  # noqa: E402
    correct_simple_detections as pva_correct,
)
from embedding_aware_belt_fusion.alignformer_pva.refine import (  # noqa: E402
    AgreementConfig,
    ICP_RANSAC,
    RefineConfig,
)
from embedding_aware_belt_fusion.alignformer_pva.abstain import AbstentionConfig  # noqa: E402
from embedding_aware_belt_fusion.alignformer_pva import abstain as pva_abstain  # noqa: E402
from embedding_aware_belt_fusion.alignformer_pva.refine import icp_refine, agree  # noqa: E402
from embedding_aware_belt_fusion.alignformer_pva.inference import build_pairwise_batch  # noqa: E402
from embedding_aware_belt_fusion.alignformer_pva.model_context import (  # noqa: E402
    load_alignformer_pva_context,
)

# Chenyi's own tuned full-method parameters (selected jointly on V2X-Real AND
# OPV2V validation -- see sec/4_experiments.tex's ablation discussion), not
# bare defaults. Propose alone trails FreeAlign by -0.0246 AP in his own
# ablation; it takes Verify (RANSAC re-solve), the agreement rule, the
# consensus floor and the weighted refit together to reach +0.0097 ahead of
# it. The bare RefineConfig()/AbstentionConfig() this script used before only
# ran Propose (+ robust IRLS), which is why AlignFormer lost to FreeAlign in
# the first 20-seed sweep -- not a real result, a missing configuration.
FULL_REFINE_CONFIG = RefineConfig(
    mode=ICP_RANSAC, gates_m=(6.0, 3.0, 1.5, 0.75, 0.5), weighted=True
)
FULL_ABSTAIN_CONFIG = AbstentionConfig(mode="abstain", level=0.2)
FULL_AGREEMENT_CONFIG = AgreementConfig(tolerance_m=1.0, consensus_floor=4)
from embedding_aware_belt_fusion.alignformer_pva.robust import RobustSolveConfig  # noqa: E402

POSE_X, POSE_Y, POSE_YAW = 0, 1, 4
YAW_STD_PER_XY_STD = 1.0  # degrees per metre, matching Chenyi's dataset.py constant
THRESHOLDS = (0.3, 0.5, 0.7)
METHODS = (
    "uncorrected", "oracle", "freealign", "alignformer", "alignformer_consensus",
    "alignformer_context", "alignformer_adaptive", "alignformer_context_adaptive",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opencood-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--geometry-cache-dir", type=Path, required=True)
    parser.add_argument(
        "--sigmas", type=float, nargs="+",
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0],
    )
    parser.add_argument("--seeds", type=int, default=20, help="seeds per nonzero sigma")
    parser.add_argument("--delays", type=int, nargs="*", default=[], help="frame delays (0=skip)")
    parser.add_argument("--freealign-calibration", type=Path, required=True)
    parser.add_argument("--alignformer-pva-checkpoint", type=Path, required=True)
    parser.add_argument("--alignformer-pva-robust-iterations", type=int, default=2)
    parser.add_argument(
        "--alignformer-context-checkpoint", type=Path, default=None,
        help=(
            "Optional AlignFormerBWithContext checkpoint (alignformer_pva."
            "train_context, pipeline tag alignformer_pva_context_v1). When "
            "given, adds an 'alignformer_context' row that corrects each CAV "
            "using the SAME Propose/Verify/Abstain stages plus CAV-CAV "
            "context cross-attention. Omitted (default): the row is reported "
            "as uncorrected boxes, and nothing else changes."
        ),
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--min-cavs", type=int, default=None,
        help="Keep only frames with at least this many CAVs (not counting ego), after loading the frame table.",
    )
    parser.add_argument(
        "--max-cavs", type=int, default=None,
        help="Keep only frames with at most this many CAVs (not counting ego), after loading the frame table.",
    )
    parser.add_argument(
        "--consensus-gate-m", type=float, default=0.5,
        help=(
            "Max centre distance (metres) for two CAVs' corrected boxes to be "
            "treated as the same physical object in cav_cav_consensus. Looser "
            "gates measurably pull in different-but-nearby vehicles (tested: "
            "2.0m -> 0.865 AP, 1.0m -> 0.905, 0.5m -> 0.940, 0.25m -> 0.951 on "
            "a small sample, all still at or below plain alignformer's 0.953)."
        ),
    )
    parser.add_argument(
        "--refine-gates-m", type=str, default=None,
        help=(
            "Comma-separated Verify-stage gate schedule (metres), overriding "
            "FULL_REFINE_CONFIG's default (6,3,1.5,0.75,0.5) for alignformer "
            "AND alignformer_context. Diagnostic: tests whether a model with "
            "a more precise Propose-stage estimate (e.g. alignformer_context) "
            "can safely use a tighter gate than the default tolerates."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--frame-table-cache", type=Path, default=None,
        help="Reuse/save the one-time pose+ground-truth+geometry pass here (expensive, ~30min for 2170 frames).",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Run this many (sigma, seed)/(delay, seed) conditions in parallel "
            "worker processes. Conditions are fully independent (read-only "
            "shared frame table/model via fork), so this parallelizes almost "
            "linearly up to the core count. 1 (default) keeps the original "
            "sequential behavior with the per-frame progress bar; >1 shows one "
            "progress bar over completed conditions instead (concurrent "
            "per-frame bars from multiple processes would garble the terminal)."
        ),
    )
    return parser.parse_args()


def perturb_pose(pose: Sequence[float], xy_std: float, yaw_std_deg: float, rng: np.random.Generator) -> List[float]:
    noisy = list(map(float, pose))
    noisy[POSE_X] += float(rng.normal(0.0, xy_std))
    noisy[POSE_Y] += float(rng.normal(0.0, xy_std))
    noisy[POSE_YAW] += float(rng.normal(0.0, yaw_std_deg))
    return noisy


def sweep_rng(seed: int, sigma: float, frame: int, agent: int) -> np.random.Generator:
    return np.random.default_rng([seed, int(round(sigma * 1000)), frame, agent])


class FrameRecord:
    __slots__ = ("ego_pose", "cav_poses", "gt_corners", "ego_boxes", "ego_scores", "cav_boxes", "cav_scores")

    def __init__(self):
        self.ego_pose: Optional[List[float]] = None
        self.cav_poses: Dict[str, List[float]] = {}
        self.gt_corners: Optional[np.ndarray] = None
        self.ego_boxes: Optional[torch.Tensor] = None
        self.ego_scores: Optional[torch.Tensor] = None
        self.cav_boxes: Dict[str, torch.Tensor] = {}
        self.cav_scores: Dict[str, torch.Tensor] = {}


def load_geometry_cache(cache_dir: Path) -> Dict[Tuple[str, str], Dict]:
    """scenario_id/frame_id -> {agent_id: {boxes, scores, transformation_matrix}}."""
    shards = sorted(cache_dir.glob("*.pt"))
    index: Dict[Tuple[str, str], Dict] = {}
    for shard in shards:
        frames = torch.load(shard, weights_only=False)
        for frame in frames:
            key = (frame["scenario_id"], frame["frame_id"])
            index[key] = {agent["agent_id"]: agent for agent in frame["agents"]}
    return index


def build_frame_table(
    dataset, geometry_index: Dict[Tuple[str, str], Dict], frame_count: int
) -> List[FrameRecord]:
    """One-time pass: world poses, ground truth, and cached local boxes per frame.

    Reimplements Chenyi's own alignformer.evaluate._build_test_frame
    (COM_RANGE filter + get_item_single_car), verified byte-identical to
    stock LateFusionDataset.__getitem__'s own ground truth over 50 frames.
    """
    records: List[FrameRecord] = []
    for index in tqdm(range(frame_count), desc="building frame table (poses + ground truth)"):
        base_data_dict = dataset.retrieve_base_data(index)
        ego_id, ego_pose = None, None
        for cav_id, cav_content in base_data_dict.items():
            if cav_content["ego"]:
                ego_id, ego_pose = cav_id, cav_content["params"]["lidar_pose"]
                break
        if ego_id is None:
            raise RuntimeError(f"frame {index} has no ego vehicle")

        frame_for_gt = {}
        record = FrameRecord()
        record.ego_pose = list(ego_pose)
        for cav_id, selected_cav_base in base_data_dict.items():
            cav_pose = selected_cav_base["params"]["lidar_pose"]
            distance = math.hypot(cav_pose[0] - ego_pose[0], cav_pose[1] - ego_pose[1])
            if distance > ocd.COM_RANGE:
                continue
            processed = dataset.get_item_single_car(selected_cav_base)
            processed["transformation_matrix"] = x1_to_x2(cav_pose, ego_pose)
            key = "ego" if cav_id == ego_id else cav_id
            frame_for_gt[key] = processed
            if key != "ego":
                record.cav_poses[key] = list(cav_pose)
        batch = dataset.collate_batch_test([frame_for_gt])
        gt_corners = dataset.post_processor.generate_gt_bbx(batch)
        record.gt_corners = common_utils.torch_tensor_to_numpy(gt_corners).astype(np.float32)
        records.append(record)
    return records


def attach_geometry(
    records: List[FrameRecord], geometry_index: Dict[Tuple[str, str], Dict], geometry_order: List[Tuple[str, str]]
) -> None:
    for record, key in zip(records, geometry_order):
        agents = geometry_index[key]
        ego_entry = None
        for agent_id, agent in agents.items():
            if torch.allclose(agent["transformation_matrix"], torch.eye(4), atol=1e-4):
                ego_entry = agent
            else:
                record.cav_boxes[agent_id] = agent["boxes"]
                record.cav_scores[agent_id] = agent["scores"]
        if ego_entry is None:
            raise RuntimeError(f"frame {key} has no identity-transform (ego) agent in the geometry cache")
        record.ego_boxes = ego_entry["boxes"]
        record.ego_scores = ego_entry["scores"]


def geometry_frame_order(cache_dir: Path) -> List[Tuple[str, str]]:
    order = []
    for shard in sorted(cache_dir.glob("*.pt")):
        frames = torch.load(shard, weights_only=False)
        for frame in frames:
            order.append((frame["scenario_id"], frame["frame_id"]))
    return order


def fuse_and_score_accumulate(
    corners: np.ndarray, scores: np.ndarray, gt: np.ndarray, nms_thresh: float, stats: Dict[float, Dict]
) -> None:
    thresholds_arr = np.asarray(THRESHOLDS, dtype=np.float64)
    if corners.shape[0] == 0:
        sc, tp = np.zeros(0, np.float32), np.zeros((len(THRESHOLDS), 0), np.int8)
    else:
        sc, tp = fusion_core.fuse_and_score(corners, scores, gt, nms_thresh, thresholds_arr)
    for i, threshold in enumerate(THRESHOLDS):
        bucket = stats[threshold]
        tp_i = tp[i].astype(int)
        bucket["tp"].extend(tp_i.tolist())
        bucket["fp"].extend((1 - tp_i).tolist())
        bucket["score"].extend(sc.tolist())
        bucket["gt"] += int(gt.shape[0])


def calculate_ap(stat: Dict) -> float:
    order = np.argsort(-np.asarray(stat["score"]))
    fp = np.asarray(stat["fp"])[order]
    tp = np.asarray(stat["tp"])[order]
    fp_cum = np.cumsum(fp)
    tp_cum = np.cumsum(tp)
    recall = tp_cum / max(stat["gt"], 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    mrec = [0.0] + recall.tolist() + [1.0]
    mpre = [0.0] + precision.tolist() + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def new_stats() -> Dict[float, Dict]:
    return {t: {"tp": [], "fp": [], "score": [], "gt": 0} for t in THRESHOLDS}


def box_order() -> str:
    return "hwl"


# Tight gate schedule for pairs Propose is already confident about (see
# select_adaptive_gates); FULL_REFINE_CONFIG.gates_m is the "wide" schedule
# used otherwise, same as the non-adaptive methods.
TIGHT_GATES_M = (1.0, 0.5, 0.25, 0.1)
# Loose significance level (0.5, not Abstain's own 0.2): this never abstains
# anything -- it is reused purely as a confidence PROBE, converting Propose's
# own Wald statistic/dof into "is this statistic small enough, for pairs with
# this much evidence, to trust a tight search" via the SAME
# abstain.abstention_threshold machinery Abstain itself uses, rather than a
# second, unrelated confidence notion.
_GATE_SELECT_PROBE = AbstentionConfig(mode="abstain", level=0.5)


def select_adaptive_gates(offset_statistic: torch.Tensor, offset_dof: torch.Tensor) -> Tuple[float, ...]:
    """Tight gates when Propose's own statistic says this pair is confident, else wide.

    This is "Propose's uncertainty propagating to Verify": instead of every
    pair using the same fixed gate schedule regardless of how well-supported
    its Propose-stage estimate was (measured, in isolation, to hurt both a
    weaker and a stronger Propose model equally -- see the plan's log), a
    pair Propose is confident about gets to use a tight, more precise search;
    one Propose is unsure about keeps the wide, safe default.
    """
    threshold = pva_abstain.abstention_threshold(offset_dof, _GATE_SELECT_PROBE)
    confident = bool((offset_statistic <= threshold).all())
    return TIGHT_GATES_M if confident else FULL_REFINE_CONFIG.gates_m


def correct_with_context(
    model,
    ego_detection: Dict[str, torch.Tensor],
    cav_detection: Dict[str, torch.Tensor],
    context_boxes: Optional[torch.Tensor],
    context_scores: Optional[torch.Tensor],
    refine_config: RefineConfig,
    abstain_config: AbstentionConfig,
    agreement_config: Optional[AgreementConfig],
    robust_config: RobustSolveConfig,
    adaptive_gates: bool = False,
) -> Optional[torch.Tensor]:
    """``alignformer_pva.inference.estimate_correction``, generalized to (a)
    an ``AlignFormerBWithContext`` model, given this CAV's already-known
    CAV-CAV context (the other CAVs' own noisy boxes in the same frame) --
    ``context_boxes=None`` skips the context fields entirely, which an
    ordinary ``AlignFormerB`` simply never looks for, so this same function
    also serves the plain model -- and (b) an adaptive Verify gate schedule
    (``select_adaptive_gates``) when ``adaptive_gates=True``, instead of the
    fixed ``refine_config`` passed in.

    inference.py's ``estimate_correction``/``build_pairwise_batch`` have no
    hook for either of these (confirmed by reading both before writing this
    function), so this mirrors ``estimate_correction``'s body exactly -- same
    Verify (icp_refine) / Abstain (abstain.decide + agree) sequence, same
    correction-dict shape -- with ``build_pairwise_batch``'s output
    conditionally extended by the context fields, and the gate schedule
    conditionally chosen from Propose's own statistic, instead of modifying
    inference.py to add parameters every OTHER (non-context, non-adaptive)
    call site would have to ignore.

    Returns the corrected CAV boxes directly (not a ``(correction,
    diagnostics)`` pair like ``estimate_correction``): this script only ever
    needs the boxes to fuse, and returns ``cav_detection["boxes"]`` unchanged
    when Abstain rejects the correction, matching every other method's
    not-corrected fallback in this script.
    """
    batch = build_pairwise_batch(ego_detection, cav_detection)
    if batch["ego_boxes"].shape[1] == 0 or batch["cav_boxes"].shape[1] == 0:
        return cav_detection["boxes"]
    if context_boxes is not None:
        k = context_boxes.shape[0]
        batch["context_boxes"] = context_boxes.unsqueeze(0)
        batch["context_scores"] = context_scores.unsqueeze(0)
        batch["context_embeddings"] = context_boxes.new_zeros((1, k, 0))
        batch["context_mask"] = torch.ones(1, k, dtype=torch.bool)

    estimate = model(batch, robust=robust_config, statistic=True)
    if adaptive_gates:
        gates = select_adaptive_gates(estimate.offset_statistic, estimate.offset_dof)
        refine_config = RefineConfig(mode=refine_config.mode, gates_m=gates, weighted=refine_config.weighted)
    refined = icp_refine(
        estimate, batch, refine_config,
        heading_lambda=model.heading_lambda,
        variance_model=model.variance_model if refine_config.weighted else None,
    )
    if agreement_config is not None:
        fallback = pva_abstain.decide(estimate, abstain_config)
        decided = agree(estimate, refined, fallback, agreement_config, heading_lambda=model.heading_lambda)
        accepted = bool(torch.any(decided.psi != 0.0) or torch.any(decided.t != 0.0))
    else:
        decided = pva_abstain.decide(refined, abstain_config)
        accepted = (
            bool(torch.any(decided.psi != 0.0) or torch.any(decided.t != 0.0))
            if abstain_config.enabled else True
        )
    if not accepted:
        return cav_detection["boxes"]

    psi = decided.psi[0]
    cosine, sine = torch.cos(psi), torch.sin(psi)
    rotation = torch.stack([torch.stack([cosine, -sine]), torch.stack([sine, cosine])])
    correction = {"rotation": rotation, "translation": decided.t[0], "yaw_correction": psi}
    boxes = cav_detection["boxes"].clone()
    boxes[:, :2] = boxes[:, :2] @ rotation.T + correction["translation"]
    boxes[:, 6] = torch.atan2(
        torch.sin(boxes[:, 6] + correction["yaw_correction"]), torch.cos(boxes[:, 6] + correction["yaw_correction"])
    )
    return boxes


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def cav_cav_consensus(
    corrected_boxes: Dict[str, torch.Tensor],
    wald_statistic: Dict[str, Optional[float]],
    gate_m: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """Nudge mutually-nearest CAV-CAV box clusters toward a precision-weighted
    consensus position, after each CAV's own (unmodified) PVA correction.

    Two CAVs independently correcting against ego never see each other; if
    several happen to see the same physical object, their corrected positions
    for it are a free, independent cross-check no single one alone has. Each
    CAV's own Wald statistic (lower = more confident fit, already computed by
    the existing Propose/Verify/Abstain pipeline) weights how far it moves --
    the more confident correction moves less. Pure post-hoc position
    adjustment; Propose/Verify/Abstain themselves are untouched.

    With 3+ CAVs a box commonly has a mutual-nearest match against EVERY
    other CAV, not just one -- an earlier, sequential pairwise version of this
    function treated each pair as if it were the only one, overwriting a
    box's position once per pair it took part in, so the last pair processed
    silently won. This groups all boxes that belong to the same physical
    object (via connected components over every pairwise mutual-NN match)
    into one cluster and computes a single weighted-average consensus per
    cluster, however many CAVs see it.
    """
    cav_ids = list(corrected_boxes.keys())
    adjusted = {cav_id: corrected_boxes[cav_id].clone() for cav_id in cav_ids}
    if len(cav_ids) < 2:
        return adjusted
    precision = {
        cav_id: 1.0 / (1.0 + wald_statistic[cav_id]) if wald_statistic[cav_id] is not None else 1.0
        for cav_id in cav_ids
    }

    nodes: List[Tuple[str, int]] = [
        (cav_id, box_index)
        for cav_id in cav_ids
        for box_index in range(corrected_boxes[cav_id].shape[0])
    ]
    node_position = {node: index for index, node in enumerate(nodes)}
    uf = _UnionFind(len(nodes))

    for i in range(len(cav_ids)):
        for j in range(i + 1, len(cav_ids)):
            a, b = cav_ids[i], cav_ids[j]
            pa, pb = corrected_boxes[a][:, :2], corrected_boxes[b][:, :2]
            if pa.shape[0] == 0 or pb.shape[0] == 0:
                continue
            dists = torch.cdist(pa, pb)
            a_to_b = dists.argmin(dim=1)
            b_to_a = dists.argmin(dim=0)
            for ia in range(pa.shape[0]):
                jb = int(a_to_b[ia])
                if int(b_to_a[jb]) != ia or dists[ia, jb] > gate_m:
                    continue  # not a mutual nearest neighbor, or too far to be the same object
                uf.union(node_position[(a, ia)], node_position[(b, jb)])

    clusters: Dict[int, List[Tuple[str, int]]] = {}
    for node in nodes:
        root = uf.find(node_position[node])
        clusters.setdefault(root, []).append(node)

    for members in clusters.values():
        if len(members) < 2:
            continue
        total_weight = sum(precision[cav_id] for cav_id, _ in members)
        consensus = sum(
            precision[cav_id] * corrected_boxes[cav_id][box_index, :2] for cav_id, box_index in members
        ) / total_weight
        for cav_id, box_index in members:
            adjusted[cav_id][box_index, :2] = consensus
    return adjusted


def run_condition(
    records: List[FrameRecord],
    sigma: float,
    seed: int,
    nms_thresh: float,
    freealign_config: FreeAlignConfig,
    pva_model,
    robust_config: RobustSolveConfig,
    delay: int,
    label: str = "",
    show_progress: bool = True,
    consensus_gate_m: float = 0.5,
    pva_context_model=None,
    refine_gates_m: Optional[Tuple[float, ...]] = None,
) -> Dict[str, Dict[float, Dict]]:
    stats = {method: new_stats() for method in METHODS}
    refine_config = (
        RefineConfig(mode=ICP_RANSAC, gates_m=refine_gates_m, weighted=True)
        if refine_gates_m is not None else FULL_REFINE_CONFIG
    )
    abstain_config = FULL_ABSTAIN_CONFIG
    agreement_config = FULL_AGREEMENT_CONFIG

    frame_iter = (
        tqdm(records, desc=f"  frames ({label})", leave=False) if show_progress else records
    )
    for frame_index, record in enumerate(frame_iter):
        source_index = max(frame_index - delay, 0)
        source = records[source_index]
        gt = record.gt_corners
        ego_boxes, ego_scores = record.ego_boxes, record.ego_scores

        per_method_boxes = {method: [ego_boxes] for method in METHODS}
        per_method_scores = {method: [ego_scores] for method in METHODS}
        alignformer_per_cav_boxes: Dict[str, torch.Tensor] = {}
        alignformer_per_cav_statistic: Dict[str, Optional[float]] = {}

        # Sorted, stable order so each CAV gets the same agent index (and
        # therefore the same noise draw) on every run, matching Chenyi's own
        # sweep_noisy_poses(..., enumerate(cav_keys)) convention: every CAV
        # in a multi-CAV frame gets an INDEPENDENT draw, not a shared one.
        cav_ids = sorted(source.cav_boxes)

        # Pre-pass: every CAV's own noisy-pose-projected boxes, computed once
        # up front so alignformer_context can use OTHER CAVs' noisy boxes as
        # context (what ego would actually have received from them) without
        # redrawing their noise or reordering the main per-CAV loop below.
        noisy_boxes_by_cav: Dict[str, torch.Tensor] = {}
        for agent_index, cav_id in enumerate(cav_ids):
            if cav_id not in source.cav_poses:
                continue
            if sigma == 0.0:
                transform = torch.as_tensor(
                    x1_to_x2(source.cav_poses[cav_id], source.ego_pose), dtype=torch.float32
                )
            else:
                rng = sweep_rng(seed, sigma, frame_index, agent_index)
                noisy_pose = perturb_pose(
                    source.cav_poses[cav_id], sigma, sigma * YAW_STD_PER_XY_STD, rng
                )
                transform = torch.as_tensor(
                    x1_to_x2(noisy_pose, source.ego_pose), dtype=torch.float32
                )
            noisy_boxes_by_cav[cav_id] = transform_boxes_to_ego(source.cav_boxes[cav_id], transform)

        for agent_index, cav_id in enumerate(cav_ids):
            true_cav_boxes = source.cav_boxes[cav_id]
            cav_scores = source.cav_scores[cav_id]
            if cav_id not in source.cav_poses:
                continue
            true_cav_pose = source.cav_poses[cav_id]
            ego_pose = source.ego_pose

            true_transform = torch.as_tensor(x1_to_x2(true_cav_pose, ego_pose), dtype=torch.float32)
            per_method_boxes["oracle"].append(transform_boxes_to_ego(true_cav_boxes, true_transform))
            per_method_scores["oracle"].append(cav_scores)

            noisy_cav_boxes_ego = noisy_boxes_by_cav[cav_id]
            per_method_boxes["uncorrected"].append(noisy_cav_boxes_ego)
            per_method_scores["uncorrected"].append(cav_scores)

            simple = {
                "ego": {"boxes": ego_boxes, "scores": ego_scores},
                cav_id: {"boxes": noisy_cav_boxes_ego, "scores": cav_scores},
            }
            free_corrected, _ = freealign_correct(simple, ["ego", cav_id], freealign_config)
            per_method_boxes["freealign"].append(free_corrected[cav_id]["boxes"])
            per_method_scores["freealign"].append(cav_scores)

            pva_corrected, pva_diag = pva_correct(
                pva_model, simple, ["ego", cav_id],
                refine_config=refine_config, abstain_config=abstain_config,
                robust_config=robust_config, agreement_config=agreement_config,
            )
            per_method_boxes["alignformer"].append(pva_corrected[cav_id]["boxes"])
            per_method_scores["alignformer"].append(cav_scores)
            alignformer_per_cav_boxes[cav_id] = pva_corrected[cav_id]["boxes"]
            alignformer_per_cav_statistic[cav_id] = pva_diag[cav_id].get("statistic")

            context_boxes = torch.cat(
                [noisy_boxes_by_cav[other] for other in cav_ids if other != cav_id],
                dim=0,
            ) if len(cav_ids) > 1 else noisy_cav_boxes_ego.new_zeros((0, 7))
            context_scores = torch.cat(
                [source.cav_scores[other] for other in cav_ids if other != cav_id],
                dim=0,
            ) if len(cav_ids) > 1 else cav_scores.new_zeros((0,))
            ego_simple = {"boxes": ego_boxes, "scores": ego_scores}
            cav_simple = {"boxes": noisy_cav_boxes_ego, "scores": cav_scores}

            if pva_context_model is not None:
                context_boxes_result = correct_with_context(
                    pva_context_model, ego_simple, cav_simple, context_boxes, context_scores,
                    refine_config, abstain_config, agreement_config, robust_config,
                )
                context_adaptive_result = correct_with_context(
                    pva_context_model, ego_simple, cav_simple, context_boxes, context_scores,
                    refine_config, abstain_config, agreement_config, robust_config,
                    adaptive_gates=True,
                )
            else:
                context_boxes_result = noisy_cav_boxes_ego
                context_adaptive_result = noisy_cav_boxes_ego
            per_method_boxes["alignformer_context"].append(context_boxes_result)
            per_method_scores["alignformer_context"].append(cav_scores)
            per_method_boxes["alignformer_context_adaptive"].append(context_adaptive_result)
            per_method_scores["alignformer_context_adaptive"].append(cav_scores)

            adaptive_result = correct_with_context(
                pva_model, ego_simple, cav_simple, None, None,
                refine_config, abstain_config, agreement_config, robust_config,
                adaptive_gates=True,
            )
            per_method_boxes["alignformer_adaptive"].append(adaptive_result)
            per_method_scores["alignformer_adaptive"].append(cav_scores)

        consensus_boxes = cav_cav_consensus(
            alignformer_per_cav_boxes, alignformer_per_cav_statistic, gate_m=consensus_gate_m
        )
        for cav_id, boxes in consensus_boxes.items():
            per_method_boxes["alignformer_consensus"].append(boxes)
            per_method_scores["alignformer_consensus"].append(source.cav_scores[cav_id])

        for method in METHODS:
            boxes = torch.cat(per_method_boxes[method], dim=0)
            scores = torch.cat(per_method_scores[method], dim=0)
            corners = np.ascontiguousarray(
                box_utils.boxes_to_corners_3d(boxes.detach().numpy(), order=box_order()), dtype=np.float32
            )
            scores_np = np.ascontiguousarray(scores.detach().numpy(), dtype=np.float32)
            fuse_and_score_accumulate(corners, scores_np, gt, nms_thresh, stats[method])
    return stats


# Module-level globals for worker processes. Set once in the parent (main)
# before the ProcessPoolExecutor forks, so every worker inherits them via
# copy-on-write memory instead of having them pickled per task -- `records`
# and `pva_model` are both large/non-trivially-picklable, and re-sending them
# per condition would dwarf the actual (now ~15-20ms/frame) compute.
_WORKER_RECORDS: Optional[List[FrameRecord]] = None
_WORKER_NMS_THRESH: Optional[float] = None
_WORKER_FREEALIGN_CONFIG: Optional[FreeAlignConfig] = None
_WORKER_PVA_MODEL = None
_WORKER_ROBUST_CONFIG: Optional[RobustSolveConfig] = None
_WORKER_CONSENSUS_GATE_M: float = 0.5
_WORKER_PVA_CONTEXT_MODEL = None
_WORKER_REFINE_GATES_M: Optional[Tuple[float, ...]] = None


def _run_one_condition(condition: Tuple[str, float, int, int]) -> Tuple[Tuple[str, float, int, int], Dict]:
    """Picklable per-condition worker body; reads the module-level globals a
    forked worker already has in memory (see _WORKER_* above), not its args."""
    torch.set_num_threads(1)  # fork() should inherit this, but cheap to re-assert
    kind, sigma, seed, delay = condition
    with torch.no_grad():
        stats = run_condition(
            _WORKER_RECORDS, sigma, seed, _WORKER_NMS_THRESH, _WORKER_FREEALIGN_CONFIG,
            _WORKER_PVA_MODEL, _WORKER_ROBUST_CONFIG, delay,
            show_progress=False, consensus_gate_m=_WORKER_CONSENSUS_GATE_M,
            pva_context_model=_WORKER_PVA_CONTEXT_MODEL,
            refine_gates_m=_WORKER_REFINE_GATES_M,
        )
    ap = {
        method: {f"ap_{int(t * 100):02d}": calculate_ap(stats[method][t]) for t in THRESHOLDS}
        for method in METHODS
    }
    return condition, ap


def main() -> None:
    args = parse_args()
    hypes = load_yaml(str(args.opencood_config), None)
    hypes["validate_dir"] = str((args.data_root / args.split).resolve())
    dataset = build_dataset(hypes, visualize=False, train=False)
    nms_thresh = float(dataset.post_processor.params["nms_thresh"])

    frame_count = len(dataset) if args.max_frames is None else min(args.max_frames, len(dataset))

    if args.frame_table_cache is not None and args.frame_table_cache.exists():
        print(f"Reusing frame table from {args.frame_table_cache}")
        records = torch.load(args.frame_table_cache, weights_only=False)
    else:
        geometry_order = geometry_frame_order(args.geometry_cache_dir)[:frame_count]
        geometry_index = load_geometry_cache(args.geometry_cache_dir)
        records = build_frame_table(dataset, geometry_index, frame_count)
        attach_geometry(records, geometry_index, geometry_order)
        if args.frame_table_cache is not None:
            args.frame_table_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(records, args.frame_table_cache)
            print(f"Saved frame table to {args.frame_table_cache}")

    if args.min_cavs is not None or args.max_cavs is not None:
        before = len(records)
        lo = args.min_cavs if args.min_cavs is not None else 0
        hi = args.max_cavs if args.max_cavs is not None else float("inf")
        records = [r for r in records if lo <= len(r.cav_boxes) <= hi]
        print(f"CAV-count filter [{lo}, {hi}]: kept {len(records)}/{before} frames")

    freealign_config = freealign_config_from_calibration(args.freealign_calibration)
    device = torch.device(args.device)
    pva_model, _ = load_alignformer_pva(args.alignformer_pva_checkpoint, device)
    robust_config = RobustSolveConfig(mode="huber", iterations=args.alignformer_pva_robust_iterations)
    pva_context_model = (
        load_alignformer_pva_context(args.alignformer_context_checkpoint, device)[0]
        if args.alignformer_context_checkpoint is not None else None
    )
    refine_gates_m = (
        tuple(float(x) for x in args.refine_gates_m.split(",")) if args.refine_gates_m else None
    )

    results: Dict[str, Dict] = {"sigma": {}, "delay": {}}
    conditions: List[Tuple[str, float, int, int]] = []
    for sigma in args.sigmas:
        seeds = [0] if sigma == 0.0 else list(range(args.seeds))
        for seed in seeds:
            conditions.append(("sigma", sigma, seed, 0))
    for delay in args.delays:
        seeds = list(range(args.seeds))
        for seed in seeds:
            conditions.append(("delay", 0.0, seed, delay))

    def record_result(condition: Tuple[str, float, int, int], ap: Dict) -> None:
        kind, sigma, seed, delay = condition
        bucket = results[kind]
        label = f"sigma_{sigma:g}" if kind == "sigma" else f"delay_{delay}"
        bucket.setdefault(label, {"per_seed": []})
        bucket[label]["per_seed"].append({"seed": seed, "ap": ap})

    if args.workers <= 1:
        for condition in tqdm(conditions, desc="sweep (sigma/seed/delay)"):
            kind, sigma, seed, delay = condition
            label = f"sigma={sigma:g} seed={seed}" if kind == "sigma" else f"delay={delay} seed={seed}"
            with torch.no_grad():
                stats = run_condition(
                    records, sigma, seed, nms_thresh, freealign_config, pva_model, robust_config, delay,
                    label=label, consensus_gate_m=args.consensus_gate_m,
                    pva_context_model=pva_context_model, refine_gates_m=refine_gates_m,
                )
            ap = {
                method: {f"ap_{int(t * 100):02d}": calculate_ap(stats[method][t]) for t in THRESHOLDS}
                for method in METHODS
            }
            record_result(condition, ap)
    else:
        import concurrent.futures as cf
        import multiprocessing as mp

        global _WORKER_RECORDS, _WORKER_NMS_THRESH, _WORKER_FREEALIGN_CONFIG
        global _WORKER_PVA_MODEL, _WORKER_ROBUST_CONFIG, _WORKER_CONSENSUS_GATE_M
        global _WORKER_PVA_CONTEXT_MODEL, _WORKER_REFINE_GATES_M
        _WORKER_RECORDS = records
        _WORKER_NMS_THRESH = nms_thresh
        _WORKER_FREEALIGN_CONFIG = freealign_config
        _WORKER_PVA_MODEL = pva_model
        _WORKER_ROBUST_CONFIG = robust_config
        _WORKER_CONSENSUS_GATE_M = args.consensus_gate_m
        _WORKER_PVA_CONTEXT_MODEL = pva_context_model
        _WORKER_REFINE_GATES_M = refine_gates_m

        print(f"Running {len(conditions)} conditions across {args.workers} worker processes...")
        # Explicit "fork": workers inherit the globals just set above via
        # copy-on-write memory, with no pickling of records/pva_model at all.
        # "spawn" would re-import this whole module per worker (slow, and
        # without the globals set in this parent process).
        with cf.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=mp.get_context("fork")
        ) as pool:
            futures = [pool.submit(_run_one_condition, condition) for condition in conditions]
            for future in tqdm(cf.as_completed(futures), total=len(futures), desc="sweep (conditions completed)"):
                condition, ap = future.result()
                record_result(condition, ap)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
