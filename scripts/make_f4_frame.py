#!/usr/bin/env python3
"""F4: one V2X-Real frame at sigma = 1, late fusion, before and after AlignFormer. From saved files only.

Frame chosen by a fixed rule, not by eye: per frame, the gain is how much closer the CAV's detections get to the
nearest ground-truth box centre once AlignFormer's pose is used (mean over the CAV's detections, frames with a
corrected CAV only); the frame at the median gain is drawn. Inputs: the frame table and the exported poses
(outputs/poses/v2xreal/alignformer/sigma_1_seed_0.npz). Output: AlignFormer-2027/figures/f4_frame.pdf and the frame's key.

    python scripts/make_f4_frame.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_verify_core  # noqa: E402
import reproduce_opv2v_sweep as base  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POSES = ROOT / "outputs/poses/v2xreal/alignformer/sigma_1_seed_0.npz"
TABLE = ROOT / "outputs/v2xreal_test_frame_table.pt"
OUT = ROOT / "AlignFormer-2027/figures/f4_frame.pdf"


def place(boxes, transform) -> np.ndarray:
    return base.transform_boxes_to_ego(boxes, torch.as_tensor(transform, dtype=torch.float32)).numpy()


def corners(boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0, 8, 3))
    return base.box_utils.boxes_to_corners_3d(boxes, order=base.box_order())


def mean_gt_distance(boxes: np.ndarray, gt_centres: np.ndarray) -> float:
    d = np.linalg.norm(boxes[:, None, :2] - gt_centres[None, :, :2], axis=-1)
    return float(d.min(axis=1).mean())


def main() -> None:
    records = check_verify_core.load_frame_table(TABLE)
    z = np.load(POSES)
    rows = []
    for i in range(len(z["frame_index"])):
        if not np.any(z["correction"][i] != 0):
            continue
        f, cav = int(z["frame_index"][i]), str(z["cav_id"][i])
        record = records[f]
        if record.gt_corners is None or len(record.gt_corners) == 0 or len(record.cav_boxes[cav]) == 0:
            continue
        gt_centres = record.gt_corners.mean(axis=1)
        before = place(record.cav_boxes[cav], base.x1_to_x2(list(z["reported_pose"][i]), list(z["ego_pose"][i])))
        after = place(record.cav_boxes[cav], z["transform"][i])
        rows.append((mean_gt_distance(before, gt_centres) - mean_gt_distance(after, gt_centres), i))
    rows.sort()
    gain, i = rows[len(rows) // 2]
    f, cav = int(z["frame_index"][i]), str(z["cav_id"][i])
    record = records[f]
    key = {"scenario": str(z["scenario"][i]), "frame_id": str(z["frame_id"][i]), "cav_id": cav,
           "median_gain_m": gain, "frames_ranked": len(rows)}
    print("F4 frame:", key)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    before = place(record.cav_boxes[cav], base.x1_to_x2(list(z["reported_pose"][i]), list(z["ego_pose"][i])))
    after = place(record.cav_boxes[cav], z["transform"][i])
    reference = place(record.cav_boxes[cav], base.x1_to_x2(record.cav_poses[cav], record.ego_pose))
    ego = record.ego_boxes.numpy()
    colour = {"Raw pose": "#e5383b", "AlignFormer": "#2f7fe0", "Reference pose": "#5c6470"}   # as Fig. 1
    fig, axes = plt.subplots(1, 3, figsize=(3.4, 1.45), sharex=True, sharey=True)
    for ax, cav_boxes, title in ((axes[0], before, "Raw pose"), (axes[1], after, "AlignFormer"),
                                 (axes[2], reference, "Reference pose")):
        for c in record.gt_corners:
            ax.fill(c[:4, 0], c[:4, 1], facecolor="#e4e8ee", edgecolor="#9ba7b6", linewidth=0.5, zorder=1)
        for c in corners(ego):
            ax.plot(np.r_[c[:4, 0], c[0, 0]], np.r_[c[:4, 1], c[0, 1]], color="#17202B", linewidth=0.8, zorder=2)
        for c in corners(cav_boxes):
            ax.plot(np.r_[c[:4, 0], c[0, 0]], np.r_[c[:4, 1], c[0, 1]],
                    color=colour[title], linewidth=1.0, zorder=3)
        ax.plot(0, 0, marker="^", color="#17202B", markersize=3)
        ax.set_title(title, fontsize=7.5, color="#17202B", pad=3)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6, colors="#3b4552", length=2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    # zoom on the CAV's detections (the correction is under a metre at the median frame)
    cav_pts = np.concatenate([before[:, :2], after[:, :2]])
    centre, half = cav_pts.mean(axis=0), max(np.ptp(cav_pts, axis=0).max() / 2 + 4, 8)
    lo, hi = centre - half, centre + half
    axes[0].set_xlim(lo[0], hi[0])
    axes[0].set_ylim(lo[1], hi[1])
    axes[0].set_xlabel("x (m)", fontsize=6.5, labelpad=1)
    axes[1].set_xlabel("x (m)", fontsize=6.5, labelpad=1)
    axes[2].set_xlabel("x (m)", fontsize=6.5, labelpad=1)
    axes[0].set_ylabel("y (m)", fontsize=6.5, labelpad=1)
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.2, top=0.86, wspace=0.1)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
    (ROOT / "outputs/paper/f4_frame.json").write_text(json.dumps(key, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
