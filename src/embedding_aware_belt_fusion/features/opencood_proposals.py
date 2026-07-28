"""Decode OpenCOOD PointPillar proposals while preserving BEV feature indices."""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor


def proposal_roi_cell_indices(
    boxes: Tensor,
    *,
    height: int,
    width: int,
    lidar_range,
):
    """Return every row-major BEV cell covered by each rotated box footprint."""
    x_min, y_min, _, x_max, y_max, _ = lidar_range
    x = torch.linspace(
        x_min + (x_max - x_min) / (2 * width),
        x_max - (x_max - x_min) / (2 * width),
        width,
        device=boxes.device,
        dtype=boxes.dtype,
    )
    y = torch.linspace(
        y_min + (y_max - y_min) / (2 * height),
        y_max - (y_max - y_min) / (2 * height),
        height,
        device=boxes.device,
        dtype=boxes.dtype,
    )
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    results = []
    for box in boxes:
        dx, dy = grid_x - box[0], grid_y - box[1]
        cosine, sine = torch.cos(box[6]), torch.sin(box[6])
        along = dx * cosine + dy * sine
        across = -dx * sine + dy * cosine
        covered = (along.abs() <= box[5] / 2) & (across.abs() <= box[4] / 2)
        indices = torch.nonzero(covered.flatten()).squeeze(1)
        if indices.numel() == 0:
            column = ((box[0] - x_min) / (x_max - x_min) * width).long()
            row = ((box[1] - y_min) / (y_max - y_min) * height).long()
            column = column.clamp(0, width - 1)
            row = row.clamp(0, height - 1)
            indices = (row * width + column).reshape(1)
        results.append(indices.cpu())
    return results


def pointpillar_forward_with_features(detector, cav_content) -> Dict[str, Tensor]:
    processed = cav_content["processed_lidar"]
    batch = {
        "voxel_features": processed["voxel_features"],
        "voxel_coords": processed["voxel_coords"],
        "voxel_num_points": processed["voxel_num_points"],
    }
    batch = detector.pillar_vfe(batch)
    batch = detector.scatter(batch)
    batch = detector.backbone(batch)
    features = batch["spatial_features_2d"]
    return {
        "psm": detector.cls_head(features),
        "rm": detector.reg_head(features),
        "spatial_features_2d": features,
    }


def decode_local_proposals(
    output: Mapping[str, Tensor],
    cav_content: Mapping[str, Tensor],
    postprocessor,
) -> Dict[str, Tensor]:
    """Decode, locally filter, and NMS proposals while retaining anchor indices."""
    from opencood.utils import box_utils

    probability = torch.sigmoid(output["psm"].permute(0, 2, 3, 1)).reshape(-1)
    decoded = postprocessor.delta_to_boxes3d(
        output["rm"], cav_content["anchor_box"]
    )[0]
    selected = torch.nonzero(
        probability > postprocessor.params["target_args"]["score_threshold"]
    ).squeeze(1)
    if selected.numel() == 0:
        feature_dim = output["spatial_features_2d"].shape[1]
        return {
            "boxes": decoded.new_empty((0, 7)),
            "corners": decoded.new_empty((0, 8, 3)),
            "scores": probability.new_empty((0,)),
            "anchor_indices": selected,
            "features": output["spatial_features_2d"].new_empty((0, feature_dim)),
        }

    boxes = decoded[selected]
    scores = probability[selected]
    corners = box_utils.boxes_to_corners_3d(
        boxes, order=postprocessor.params["order"]
    )
    valid = box_utils.remove_large_pred_bbx(corners) & box_utils.remove_bbx_abnormal_z(
        corners
    )
    boxes, scores, corners, selected = (
        boxes[valid],
        scores[valid],
        corners[valid],
        selected[valid],
    )
    keep = box_utils.nms_rotated(corners, scores, postprocessor.params["nms_thresh"])
    keep = torch.as_tensor(keep, dtype=torch.long, device=boxes.device)
    boxes, scores, corners, selected = (
        boxes[keep],
        scores[keep],
        corners[keep],
        selected[keep],
    )
    in_range = box_utils.get_mask_for_boxes_within_range_torch(corners)
    boxes, scores, corners, selected = (
        boxes[in_range],
        scores[in_range],
        corners[in_range],
        selected[in_range],
    )

    _, _, height, width = output["spatial_features_2d"].shape
    anchor_count = output["psm"].shape[1]
    cell_indices = torch.div(selected, anchor_count, rounding_mode="floor")
    rows = torch.div(cell_indices, width, rounding_mode="floor")
    columns = cell_indices % width
    features = output["spatial_features_2d"][0, :, rows, columns].transpose(0, 1)
    return {
        "boxes": boxes,
        "corners": corners,
        "scores": scores,
        "anchor_indices": selected,
        "features": features,
    }


def assign_proposals_to_ground_truth(
    proposal_corners: Tensor,
    gt_boxes: Tensor,
    gt_ids,
    *,
    order: str,
    minimum_iou: float,
) -> Dict[str, object]:
    """One-to-one rotated-BEV IoU assignment of predictions to local GT."""
    from opencood.utils import box_utils, common_utils

    proposal_count = proposal_corners.shape[0]
    assigned_ids = [None] * proposal_count
    assigned_ious = torch.zeros(proposal_count, dtype=torch.float32)
    assigned_indices = torch.full((proposal_count,), -1, dtype=torch.long)
    if proposal_count == 0 or gt_boxes.shape[0] == 0:
        return {
            "gt_ids": assigned_ids,
            "gt_ious": assigned_ious,
            "gt_indices": assigned_indices,
        }

    gt_corners = box_utils.boxes_to_corners_3d(gt_boxes, order=order)
    proposal_polygons = common_utils.convert_format(
        proposal_corners[:, :4, :2].detach().cpu().numpy()
    )
    gt_polygons = common_utils.convert_format(
        gt_corners[:, :4, :2].detach().cpu().numpy()
    )
    iou = np.zeros((proposal_count, len(gt_polygons)), dtype=np.float32)
    for proposal_index, polygon in enumerate(proposal_polygons):
        iou[proposal_index] = common_utils.compute_iou(polygon, gt_polygons)
    rows, columns = linear_sum_assignment(1.0 - iou)
    for row, column in zip(rows, columns):
        if iou[row, column] >= minimum_iou:
            assigned_ids[row] = str(gt_ids[column])
            assigned_ious[row] = float(iou[row, column])
            assigned_indices[row] = int(column)
    return {
        "gt_ids": assigned_ids,
        "gt_ious": assigned_ious,
        "gt_indices": assigned_indices,
    }
