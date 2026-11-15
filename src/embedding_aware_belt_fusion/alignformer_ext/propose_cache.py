"""Cache AlignFormer's Propose output per (sigma, seed) so Verify/Abstain variants never re-run the network.

For one sweep condition this reproduces ``scripts/reproduce_opv2v_sweep.py``'s
``run_condition`` noise draws exactly (same ``sweep_rng`` / ``perturb_pose`` /
``x1_to_x2`` / ``transform_boxes_to_ego``, same sorted-CAV agent indices),
runs the UNMODIFIED model's ``forward(batch, robust, statistic=True)`` per
ego/CAV pair, and stores what every downstream stage reads: the noisy CAV
boxes, Propose's ``(psi, t)``, its Wald statistic and degrees of freedom, and
the full Sinkhorn plan (``log_assignment``, dustbin row/column included) for
the variants that use Propose's associations. With a FreeAlign config it also
stores FreeAlign's corrected boxes for the same draws (``run_condition``'s own
call), so the baseline is scored on identical noise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from embedding_aware_belt_fusion.alignformer_pva.inference import build_pairwise_batch
from embedding_aware_belt_fusion.freealign.inference import correct_simple_detections as freealign_correct


def noisy_cav_boxes(base, record, frame_index: int, sigma: float, seed: int, noise=None) -> Dict[str, torch.Tensor]:
    """``run_condition``'s noisy ego-frame CAV boxes for one frame, keyed by CAV id. ``noise`` (a
    ``noise_kinds.PoseNoise``) replaces the benchmark draw with another noise kind (T5); None = the benchmark."""
    out = {}
    for agent_index, cav_id in enumerate(sorted(record.cav_boxes)):
        if cav_id not in record.cav_poses:
            continue
        if noise is not None:
            pose = noise.perturb(record.cav_poses[cav_id], frame_index, cav_id, base.POSE_X, base.POSE_Y, base.POSE_YAW)
        elif sigma == 0.0:
            pose = record.cav_poses[cav_id]
        else:
            rng = base.sweep_rng(seed, sigma, frame_index, agent_index)
            pose = base.perturb_pose(record.cav_poses[cav_id], sigma, sigma * base.YAW_STD_PER_XY_STD, rng)
        transform = torch.as_tensor(base.x1_to_x2(pose, record.ego_pose), dtype=torch.float32)
        out[cav_id] = base.transform_boxes_to_ego(record.cav_boxes[cav_id], transform)
    return out


def build_condition(base, records: List, model, robust_config, sigma: float, seed: int,
                    freealign_config=None, noise=None) -> Dict[str, np.ndarray]:
    """Propose over every (frame, CAV) pair of one condition, as flat arrays with offsets."""
    frame, cav_rank, attempted, freealign_boxes = [], [], [], []
    cav_boxes, cav_count = [], []
    propose, statistic, dof = [], [], []
    plans, plan_shape = [], []
    with torch.no_grad():
        for frame_index, record in enumerate(records):
            noisy = noisy_cav_boxes(base, record, frame_index, sigma, seed, noise)
            for rank, cav_id in enumerate(sorted(record.cav_boxes)):
                if cav_id not in noisy:
                    continue
                boxes = noisy[cav_id]
                frame.append(frame_index)
                cav_rank.append(rank)
                cav_boxes.append(boxes.numpy().astype(np.float32))
                cav_count.append(int(boxes.shape[0]))
                if freealign_config is not None:
                    simple = {
                        "ego": {"boxes": record.ego_boxes, "scores": record.ego_scores},
                        cav_id: {"boxes": boxes, "scores": record.cav_scores[cav_id]},
                    }
                    corrected, _ = freealign_correct(simple, ["ego", cav_id], freealign_config)
                    freealign_boxes.append(corrected[cav_id]["boxes"].numpy().astype(np.float32))
                ok = record.ego_boxes.shape[0] > 0 and boxes.shape[0] > 0
                attempted.append(ok)
                if not ok:
                    propose.append((0.0, 0.0, 0.0)); statistic.append(0.0); dof.append(1.0)
                    plan_shape.append((0, 0))
                    continue
                batch = build_pairwise_batch(
                    {"boxes": record.ego_boxes, "scores": record.ego_scores},
                    {"boxes": boxes, "scores": record.cav_scores[cav_id]},
                )
                estimate = model(batch, robust=robust_config, statistic=True)
                propose.append((float(estimate.psi[0]), float(estimate.t[0, 0]), float(estimate.t[0, 1])))
                statistic.append(float(estimate.offset_statistic[0]))
                dof.append(float(estimate.offset_dof[0]))
                plan = estimate.log_assignment[0].numpy().astype(np.float32)
                plans.append(plan.reshape(-1))
                plan_shape.append(plan.shape)
    out = {
        "sigma": np.float64(sigma), "seed": np.int64(seed),
        "noise_kind": np.str_(noise.kind if noise is not None else "benchmark"),
        "frame": np.asarray(frame, np.int64), "cav_rank": np.asarray(cav_rank, np.int64),
        "attempted": np.asarray(attempted, bool),
        "cav_boxes": np.concatenate(cav_boxes) if cav_boxes else np.zeros((0, 7), np.float32),
        "cav_off": np.concatenate([[0], np.cumsum(cav_count)]).astype(np.int64),
        "propose": np.asarray(propose, np.float32).reshape(-1, 3),
        "statistic": np.asarray(statistic, np.float32), "dof": np.asarray(dof, np.float32),
        "plan": np.concatenate(plans) if plans else np.zeros(0, np.float32),
        "plan_shape": np.asarray(plan_shape, np.int64).reshape(-1, 2),
    }
    if freealign_config is not None:
        out["freealign_boxes"] = np.concatenate(freealign_boxes) if freealign_boxes else np.zeros((0, 7), np.float32)
    return out


def condition_path(cache_dir: Path, sigma: float, seed: int) -> Path:
    return Path(cache_dir) / f"sigma_{sigma:g}_seed_{seed}.npz"


def noise_of(cache: Dict[str, np.ndarray]):
    """The ``noise_kinds.PoseNoise`` a cache was built with, or None for the benchmark draws. Its frame keys come from
    the active odometry table (the sweep's --odometry), which every non-benchmark cache needs."""
    kind = str(cache["noise_kind"]) if "noise_kind" in cache else "benchmark"
    if kind == "benchmark":
        return None
    from embedding_aware_belt_fusion.alignformer_ext import noise_kinds, odometry

    table = odometry.active()
    if table is None:
        raise RuntimeError(f"a '{kind}' cache needs the odometry table's frame keys (pass --odometry)")
    return noise_kinds.PoseNoise(kind, float(cache["sigma"]), int(cache["seed"]), table.frames)


def noisy_poses(base, record, frame_index: int, sigma: float, seed: int, noise=None) -> Dict[str, list]:
    """The same per-CAV noisy world poses ``noisy_cav_boxes`` projects with (identical draws; ``noise`` as there)."""
    out = {}
    for agent_index, cav_id in enumerate(sorted(record.cav_boxes)):
        if cav_id not in record.cav_poses:
            continue
        if noise is not None:
            out[cav_id] = noise.perturb(record.cav_poses[cav_id], frame_index, cav_id, base.POSE_X, base.POSE_Y,
                                        base.POSE_YAW)
        elif sigma == 0.0:
            out[cav_id] = list(record.cav_poses[cav_id])
        else:
            rng = base.sweep_rng(seed, sigma, frame_index, agent_index)
            out[cav_id] = base.perturb_pose(record.cav_poses[cav_id], sigma, sigma * base.YAW_STD_PER_XY_STD, rng)
    return out
