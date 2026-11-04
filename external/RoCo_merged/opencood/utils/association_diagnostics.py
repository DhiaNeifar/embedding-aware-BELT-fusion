# -*- coding: utf-8 -*-
"""Ground-truth association precision/recall/F1 for CoAlign's agent-object
pose-graph box alignment (box_align_v2.box_alignment_relative_sample_np).

CoAlign's own pipeline never checks whether its internal box clustering
(cluster_dict, built while estimating pose corrections) actually groups boxes
belonging to the same real-world object. This module adds that check,
evaluation-only, using OPV2V's labelled vehicle IDs -- mirroring the
precision/recall/F1 diagnostic already computed for the other methods in
this study (see embedding_aware_belt_fusion.evaluation.component_diagnostics).

Nothing here is used by the deployed box-alignment/pose-correction path.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment

from opencood.utils import box_utils, common_utils
from opencood.utils.transformation_utils import pose_to_tfm, x1_to_x2


def world_objects_unfiltered(object_dict, order):
    """Like box_utils.project_world_objects with an identity lidar_pose, but
    without its range mask -- that mask is defined relative to an ego sensor
    and drops everything when applied in raw (often far-from-origin) world
    coordinates. Diagnostics need every labelled object, not just the ones
    near world-origin."""
    output = {}
    identity_pose = [0, 0, 0, 0, 0, 0]
    for object_id, object_content in object_dict.items():
        location = object_content["location"]
        rotation = object_content["angle"]
        center = object_content.get("center", [0, 0, 0])
        extent = object_content["extent"]
        object_pose = [
            location[0] + center[0],
            location[1] + center[1],
            location[2] + center[2],
            rotation[0], rotation[1], rotation[2],
        ]
        object2lidar = x1_to_x2(object_pose, identity_pose)
        bbx = box_utils.create_bbx(extent).T
        bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]
        bbx_world = np.dot(object2lidar, bbx).T
        bbx_world = np.expand_dims(bbx_world[:, :3], 0)
        bbx_world = box_utils.corner_to_center(bbx_world, order=order)
        output[object_id] = bbx_world
    return output


def assign_predictions_to_ground_truth(
    pred_corners_world,
    gt_boxes_world,
    gt_ids,
    *,
    order: str,
    minimum_iou: float = 0.5,
):
    """One-to-one rotated-BEV IoU assignment of world-frame predicted corners
    to world-frame ground-truth boxes. Returns one GT id (or None) per
    prediction, in prediction order."""
    count = len(pred_corners_world)
    assigned_ids = [None] * count
    if count == 0 or len(gt_boxes_world) == 0:
        return assigned_ids

    gt_corners = box_utils.boxes_to_corners_3d(
        np.asarray(gt_boxes_world), order=order
    )
    pred_polygons = common_utils.convert_format(
        np.asarray(pred_corners_world)[:, :4, :2]
    )
    gt_polygons = common_utils.convert_format(gt_corners[:, :4, :2])

    iou = np.zeros((count, len(gt_polygons)), dtype=np.float32)
    for index, polygon in enumerate(pred_polygons):
        iou[index] = common_utils.compute_iou(polygon, gt_polygons)

    rows, columns = linear_sum_assignment(1.0 - iou)
    for row, column in zip(rows, columns):
        if iou[row, column] >= minimum_iou:
            assigned_ids[row] = str(gt_ids[column])
    return assigned_ids


def empty_counts():
    return {
        "candidate_pairs": 0,
        "positive_pairs": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
    }


def merge_counts(total, update):
    for key, value in update.items():
        total[key] = total.get(key, 0) + value


def summarize_counts(counts):
    tp = int(counts.get("true_positive", 0))
    fp = int(counts.get("false_positive", 0))
    fn = int(counts.get("false_negative", 0))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "candidate_pairs": int(counts.get("candidate_pairs", 0)),
        "positive_pairs": int(counts.get("positive_pairs", 0)),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-8),
    }


def cluster_association_counts(
    pred_corners_list,
    noisy_lidar_pose,
    box_object_ids,
    cluster_dict,
    box_idx_to_agent,
    *,
    order: str,
    candidate_distance: float,
):
    """Precision/recall/F1 of CoAlign's box_align_v2 clustering against
    ground-truth object IDs.

    Args:
        pred_corners_list: per-agent predicted box corners, same input given
            to box_alignment_relative_sample_np (local/ego coordinates).
        noisy_lidar_pose: [N_cav, 6] poses used for that same call.
        box_object_ids: flat list (len == sum of boxes across agents, same
            flattening order as box_idx_to_agent) of assigned GT object id
            or None per box, from assign_predictions_to_ground_truth.
        cluster_dict, box_idx_to_agent: box_alignment_relative_sample_np's
            own internal clustering output (call it with return_clusters=True).
        candidate_distance: world-frame center distance (meters) under which
            two boxes from different agents are considered a candidate pair
            -- mirrors the `thres` box_alignment_relative_sample_np itself
            used to seed clusters.
    """
    nonempty_indices = [
        idx for idx, corners in enumerate(pred_corners_list) if len(corners) != 0
    ]
    if not nonempty_indices:
        return empty_counts()

    lidar_pose_tfm = pose_to_tfm(noisy_lidar_pose)
    world_centers = []
    for agent_index in nonempty_indices:
        world_corners = box_utils.project_box3d(
            pred_corners_list[agent_index], lidar_pose_tfm[agent_index]
        )
        world_centers.append(np.mean(world_corners, axis=1))
    world_centers = np.concatenate(world_centers, axis=0)

    # box flat index -> which cluster (same-object group) it belongs to, if any
    cluster_of = {}
    for cluster_id, cluster in cluster_dict.items():
        for box_idx in cluster["box_idx"]:
            cluster_of[box_idx] = cluster_id

    flat_count = len(box_idx_to_agent)
    assert flat_count == len(world_centers) == len(box_object_ids)

    output = empty_counts()
    for first, second in combinations(range(flat_count), 2):
        if box_idx_to_agent[first] == box_idx_to_agent[second]:
            continue
        first_id, second_id = box_object_ids[first], box_object_ids[second]
        if first_id is None or second_id is None:
            continue
        distance = float(np.linalg.norm(world_centers[first] - world_centers[second]))
        if distance > candidate_distance:
            continue
        same = first_id == second_id
        clustered_together = (
            first in cluster_of
            and second in cluster_of
            and cluster_of[first] == cluster_of[second]
        )
        output["candidate_pairs"] += 1
        output["positive_pairs"] += int(same)
        if same and clustered_together:
            output["true_positive"] += 1
        elif (not same) and clustered_together:
            output["false_positive"] += 1
        elif same and not clustered_together:
            output["false_negative"] += 1
    return output
