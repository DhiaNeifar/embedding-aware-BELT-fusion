# -*- coding: utf-8 -*-
"""Ground-truth association precision/recall/F1 for CoAlign's box_align_v2
pose-graph clustering, evaluation-only (see utils/association_diagnostics.py
for why and how).

Reuses the exact same dataset/box-align call the real training/inference
pipeline makes (via a transparent monkeypatch around
box_alignment_relative_sample_np that also asks for the internal clusters),
so the diagnostic is guaranteed consistent with what the deployed pipeline
actually does -- this script changes no deployed behavior, only observes it.
"""

import argparse
import json

import numpy as np
from tqdm import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.models.sub_modules import box_align_v2
from opencood.utils import box_utils
from opencood.utils.association_diagnostics import (
    assign_predictions_to_ground_truth,
    cluster_association_counts,
    empty_counts,
    merge_counts,
    summarize_counts,
    world_objects_unfiltered,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypes_yaml", required=True,
                        help="Same config used for inference.py / training "
                             "(must have box_align enabled).")
    parser.add_argument("--candidate-distance", type=float, default=1.5,
                        help="World-frame meters; mirrors box_align_v2's own "
                             "`thres` used to seed clusters.")
    parser.add_argument("--minimum-iou", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main():
    opt = parse_args()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, None)
    if "box_align" not in hypes:
        raise ValueError("This config has no box_align section -- nothing to diagnose.")

    dataset = build_dataset(hypes, visualize=False, train=False)
    order = hypes["postprocess"]["order"]

    captured = {}

    real_fn = box_align_v2.box_alignment_relative_sample_np

    def capturing_wrapper(pred_corners_list, noisy_lidar_pose, **kwargs):
        kwargs["return_clusters"] = True
        refined_pose, cluster_dict, box_idx_to_agent = real_fn(
            pred_corners_list, noisy_lidar_pose, **kwargs
        )
        captured["pred_corners_list"] = pred_corners_list
        captured["noisy_lidar_pose"] = noisy_lidar_pose
        captured["cluster_dict"] = cluster_dict
        captured["box_idx_to_agent"] = box_idx_to_agent
        return refined_pose

    box_align_v2.box_alignment_relative_sample_np = capturing_wrapper

    totals = empty_counts()
    frame_count = len(dataset) if opt.max_frames is None else min(opt.max_frames, len(dataset))

    frames_with_clusters = 0
    for idx in tqdm(range(frame_count), desc="association diagnostics"):
        captured.clear()
        _ = dataset[idx]
        if "cluster_dict" not in captured:
            continue
        frames_with_clusters += 1

        scenario_database_index = next(
            position for position, end in enumerate(dataset.len_record) if idx < end
        )
        scenario = dataset.scenario_database[scenario_database_index]
        timestamp_index = idx if scenario_database_index == 0 else idx - dataset.len_record[scenario_database_index - 1]
        timestamp_key = dataset.return_timestamp_key(scenario, timestamp_index)

        merged_vehicles = {}
        for cav_content in scenario.values():
            params = yaml_utils.load_yaml(cav_content[timestamp_key]["yaml"], None)
            merged_vehicles.update(params["vehicles"])

        gt_dict = world_objects_unfiltered(merged_vehicles, order)
        gt_ids = list(gt_dict.keys())
        gt_boxes_world = np.array([gt_dict[object_id][0, :] for object_id in gt_ids]) if gt_ids else np.zeros((0, 7))

        pred_corners_list = captured["pred_corners_list"]
        noisy_lidar_pose = captured["noisy_lidar_pose"]
        from opencood.utils.transformation_utils import pose_to_tfm
        lidar_pose_tfm = pose_to_tfm(noisy_lidar_pose)

        box_object_ids = []
        for agent_index, corners in enumerate(pred_corners_list):
            if len(corners) == 0:
                continue
            world_corners = box_utils.project_box3d(corners, lidar_pose_tfm[agent_index])
            box_object_ids.extend(
                assign_predictions_to_ground_truth(
                    world_corners, gt_boxes_world, gt_ids,
                    order=order, minimum_iou=opt.minimum_iou,
                )
            )

        counts = cluster_association_counts(
            pred_corners_list,
            noisy_lidar_pose,
            box_object_ids,
            captured["cluster_dict"],
            captured["box_idx_to_agent"],
            order=order,
            candidate_distance=opt.candidate_distance,
        )
        merge_counts(totals, counts)

    box_align_v2.box_alignment_relative_sample_np = real_fn

    summary = summarize_counts(totals)
    summary["frames_evaluated"] = frame_count
    summary["frames_with_box_align_applied"] = frames_with_clusters
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
