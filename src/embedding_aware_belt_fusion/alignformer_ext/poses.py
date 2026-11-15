"""Plug-in output as a file: one corrected CAV-to-ego transform per (frame, CAV), for any host to use.

Every plug-in is reduced to the same thing: for each frame and CAV, the 4x4 transform a host should place that CAV's
data with. Late fusion places boxes with it; intermediate fusion projects points or warps features with it. A file
holds one plug-in at one (dataset, sigma, seed):

* ``scenario``, ``frame_id``, ``ego_id``, ``cav_id``: which frame and agents (strings, as in the dataset);
* ``frame_index``: the row of the frame table;
* ``reported_pose`` (N, 6): the CAV's noisy reported world pose, the same draw for every plug-in and host;
* ``ego_pose`` (N, 6): the ego's own pose (the reference frame; the benchmark treats it as exact);
* ``correction`` (N, 3): psi, tx, ty applied in the ego frame, zero where the plug-in keeps the reported pose;
* ``transform`` (N, 4, 4): the corrected CAV-to-ego transform, ``C(correction) @ x1_to_x2(reported_pose, ego_pose)``.

No ground truth is stored: a host's true-pose condition reads the dataset itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from embedding_aware_belt_fusion.alignformer_ext.propose_cache import noise_of, noisy_poses


def condition_path(root: Path, dataset: str, plugin: str, sigma: float, seed: int) -> Path:
    return Path(root) / dataset / plugin / f"sigma_{sigma:g}_seed_{seed}.npz"


def correction_matrix(psi: float, tx: float, ty: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    m = np.eye(4)
    m[:2, :2] = [[c, -s], [s, c]]
    m[:2, 3] = [tx, ty]
    return m


def freealign_corrections(cache: Dict[str, np.ndarray]) -> np.ndarray:
    """(P, 3) psi, tx, ty of FreeAlign's rigid correction per cached pair, refitted from its cached corrected boxes
    (FreeAlign moves a CAV's boxes rigidly); zero where it left them unchanged (it abstained)."""
    from embedding_aware_belt_fusion.alignformer_pva.procrustes import weighted_se2_kabsch

    out = np.zeros((len(cache["frame"]), 3), np.float32)
    for p in range(len(cache["frame"])):
        rows = slice(cache["cav_off"][p], cache["cav_off"][p + 1])
        noisy, moved = cache["cav_boxes"][rows], cache["freealign_boxes"][rows]
        if not len(noisy) or np.abs(moved[:, :2] - noisy[:, :2]).max() < 1e-5:
            continue
        if len(noisy) == 1:
            psi = float(np.arctan2(np.sin(moved[0, 6] - noisy[0, 6]), np.cos(moved[0, 6] - noisy[0, 6])))
            c, s = np.cos(psi), np.sin(psi)
            t = moved[0, :2] - np.array([c * noisy[0, 0] - s * noisy[0, 1], s * noisy[0, 0] + c * noisy[0, 1]])
            out[p] = (psi, t[0], t[1])
            continue
        psi, t = weighted_se2_kabsch(torch.from_numpy(moved[None, :, :2]), torch.from_numpy(noisy[None, :, :2]),
                                     torch.ones(1, len(noisy)))
        out[p] = (float(psi), float(t[0, 0]), float(t[0, 1]))
    return out


def build(base, records: List, frames: List[Tuple[str, str, str]], cache: Dict[str, np.ndarray],
          decided: np.ndarray) -> Dict[str, np.ndarray]:
    """One file's arrays from a propose cache and a plug-in's decisions over its pairs ((P, 3) psi, tx, ty).

    Every CAV the frame table holds gets a row; a CAV with no cached pair keeps its reported pose."""
    sigma, seed = float(cache["sigma"]), int(cache["seed"])
    noise = noise_of(cache)
    pair_of = {}
    for p in range(len(cache["frame"])):
        f = int(cache["frame"][p])
        pair_of[(f, sorted(records[f].cav_boxes)[int(cache["cav_rank"][p])])] = p
    out = {k: [] for k in ("scenario", "frame_id", "ego_id", "cav_id", "frame_index", "reported_pose", "ego_pose",
                           "correction", "transform")}
    for f, record in enumerate(records):
        scenario, frame_id, ego_id = frames[f]
        for cav_id, pose in sorted(noisy_poses(base, record, f, sigma, seed, noise).items()):
            p = pair_of.get((f, cav_id))
            correction = np.asarray(decided[p], np.float64) if p is not None else np.zeros(3)
            placed = np.asarray(base.x1_to_x2(pose, record.ego_pose), np.float64)
            out["scenario"].append(scenario)
            out["frame_id"].append(frame_id)
            out["ego_id"].append(ego_id)
            out["cav_id"].append(str(cav_id))
            out["frame_index"].append(f)
            out["reported_pose"].append(np.asarray(pose, np.float64))
            out["ego_pose"].append(np.asarray(record.ego_pose, np.float64))
            out["correction"].append(correction)
            out["transform"].append(correction_matrix(*correction) @ placed)
    arrays = {k: np.array(v) for k, v in out.items()}
    arrays.update(sigma=np.float64(sigma), seed=np.int64(seed))
    return arrays


def check_against_cache(base, records: List, cache: Dict[str, np.ndarray], decided: np.ndarray,
                        arrays: Dict[str, np.ndarray], pairs: int = 200) -> float:
    """Largest box-centre difference (m) between a CAV's boxes placed with the exported transform and the corrected
    boxes late fusion scored from the cache, over the first ``pairs`` corrected pairs."""
    from embedding_aware_belt_fusion.alignformer_ext.evaluate import apply_correction

    row = {(int(f), c): i for i, (f, c) in enumerate(zip(arrays["frame_index"], arrays["cav_id"]))}
    worst, seen = 0.0, 0
    for p in range(len(cache["frame"])):
        if seen >= pairs:
            break
        f = int(cache["frame"][p])
        cav_id = sorted(records[f].cav_boxes)[int(cache["cav_rank"][p])]
        local = records[f].cav_boxes[cav_id]
        if not len(local):
            continue
        rows = slice(cache["cav_off"][p], cache["cav_off"][p + 1])
        expected = apply_correction(torch.from_numpy(cache["cav_boxes"][rows]), *map(float, decided[p]))
        placed = base.transform_boxes_to_ego(local, torch.as_tensor(arrays["transform"][row[(f, str(cav_id))]],
                                                                   dtype=torch.float32))
        worst = max(worst, float((placed[:, :2] - expected[:, :2]).abs().max()))
        seen += 1
    return worst
