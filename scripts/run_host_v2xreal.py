#!/usr/bin/env python3
"""Run one unchanged V2X-Real intermediate-fusion network under every pose condition (Table 1, V2X-Real).

The V2X-Real networks (AttFuse, F-Cooper, V2VAM, V2X-ViT, Where2comm) were trained with jigsawcomm-v2x
(external/jigsawcomm-v2x, branch baselines-package) and load exactly into it (scripts/check_v2xreal_networks.py);
scripts/check_v2xreal_clean_ap.py reproduces their own reported AP with its own evaluator. That code runs the network,
reads the LiDAR (its per-cloud intensity normalisation, as in training) and scores; this script only chooses each
CAV's pose, like scripts/run_host.py does for OPV2V:

* ``true``: the recorded pose (upper bound);
* ``bare``: the noisy reported pose, no correction;
* one per plug-in: the plug-in's corrected CAV-to-ego transform from ``outputs/poses/v2xreal/<plugin>/``;
* ``ego`` (--ego-only): the ego's data alone, on the same ground truth.

The network places a CAV's features with ``params['lidar_pose']`` and the ground truth with
``params['lidar_pose_clean']``, so a condition sets the CAV's ``lidar_pose`` to the pose whose transform to the ego
is the condition's transform, and leaves the ground truth alone.

Our V2X-Real protocol (multidataset/v2xreal.py), the same as late fusion's: ground truth of the vehicle classes
(Car, PoliceCar, LongVehicle), every scenario, the corrupt frame removed, vehicle-centric agents with the lowest
vehicle id as ego, and the CAV set fixed by TRUE distance at 70 m (with the noisy pose the set would change with
the condition). Each frame's LiDAR is read once and the point-shuffling seed is fixed per frame.

Box overlaps in NMS and AP matching use the C++ code (alignformer_ext/fast_opencood.py, identical decisions).

    python scripts/run_host_v2xreal.py --host attfuse --sigmas 1 --seeds 1 --max-frames 50 --output /tmp/attfuse.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import types
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
CODE = REPO / "external" / "jigsawcomm-v2x"     # branch baselines-package: the code the weights were trained with
DATA = Path("/mnt/external/workspace/public/dataset/v2x-real")
HOSTS = REPO / "models" / "v2xreal" / "intermediate_fusion"
OUR_COMM_RANGE = 70.0          # multidataset.v2xreal.V2XREAL_COM_RANGE: the CAVs our pose files cover
IOUS = (0.3, 0.5, 0.7)

# camera / .pcd-only dependencies, never used by these LiDAR networks on V2X-Real (.bin)
sys.modules.setdefault("efficientnet_pytorch", types.SimpleNamespace(EfficientNet=None))
sys.modules.setdefault("pypcd", types.SimpleNamespace(pypcd=None))
sys.path.insert(0, str(CODE))
sys.path.insert(1, str(REPO / "src"))

from opencood.data_utils.datasets import build_dataset  # noqa: E402
import opencood.data_utils.datasets.basedataset.v2xreal_basedataset as qbase  # noqa: E402
from opencood.hypes_yaml.yaml_utils import load_yaml  # noqa: E402
from opencood.tools import inference_utils, train_utils  # noqa: E402
from opencood.utils import box_utils, eval_utils  # noqa: E402
from opencood.utils.transformation_utils import tfm_to_pose, x1_to_x2, x_to_world  # noqa: E402

from embedding_aware_belt_fusion.alignformer_ext import fast_opencood  # noqa: E402
from embedding_aware_belt_fusion.multidataset import v2xreal as rules  # noqa: E402


# --------------------------------------------------------------------------------------------------------------
# Our V2X-Real protocol inside jigsawcomm-v2x
# --------------------------------------------------------------------------------------------------------------

_reinitialize = qbase.V2XREALBaseDataset.reinitialize


def _reinitialize_all_scenarios(self):
    if self.scenario_folders:
        root = os.path.dirname(self.scenario_folders[0])
        self.scenario_folders = sorted(os.path.join(root, x) for x in os.listdir(root)
                                       if os.path.isdir(os.path.join(root, x)))
    _reinitialize(self)
    database = rules._drop_corrupt_timestamps(self.scenario_database)
    self.scenario_database = rules._vehicle_centric(database)     # lowest vehicle id is the ego, as in late fusion
    self.len_record = rules._len_record(self.scenario_database)


qbase.V2XREALBaseDataset.reinitialize = _reinitialize_all_scenarios

# The dataset's noise step copies lidar_pose into lidar_pose_clean, which would move the ground truth with the
# condition's pose. Each frame here already carries lidar_pose_clean = the recorded pose and lidar_pose = the
# condition's pose (exactly what its noise path produces), so that step is skipped.
import opencood.data_utils.datasets.intermediate_heter_fusion_dataset as qheter  # noqa: E402
qheter.add_noise_data_dict = lambda data_dict, noise_setting: data_dict


def install_fast_eval():
    box_utils.nms_rotated = fast_opencood.nms_rotated
    eval_utils.caluclate_tp_fp = fast_opencood.caluclate_tp_fp


def host_hypes(host: str, assignment: Path) -> dict:
    """The network's own config, pointed at our V2X-Real copy and our protocol. Ranges, anchors, model: unchanged."""
    text = (HOSTS / host / "config.yaml").read_text()
    text = text.replace("/workspace/nvme/chenyiw/datasets/v2x-real-clean", str(DATA))
    text = text.replace("/home/chenyi/datasets/v2x-real-clean", str(DATA))
    text = text.replace("opencood/logs/heter_modality_assign/v2xreal_lidaronly.json", str(assignment))
    path = assignment.parent / f"{host}_config.yaml"
    path.write_text(text)
    hypes = load_yaml(str(path), None)
    hypes["object_classes"] = sorted(rules.VEHICLE_TYPES)   # our ground truth, the same as late fusion's
    hypes["comm_range"] = 1e5           # the CAV set is fixed by true distance in FrameConditions instead
    hypes["validate_dir"] = hypes["test_dir"]
    hypes["noise_setting"] = OrderedDict(add_noise=False)
    return hypes


def write_assignment(path: Path) -> Path:
    """The modality file: every agent is a LiDAR agent (m1)."""
    table = {}
    for split in ("train", "val", "test"):
        for scenario in sorted(os.listdir(DATA / split)):
            folder = DATA / split / scenario
            if folder.is_dir():
                table[scenario] = {c: "m1" for c in os.listdir(folder) if (folder / c).is_dir()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table))
    return path


def frame_key(data, idx: int):
    scenario_index = next(i for i, n in enumerate(data.len_record) if idx < n)
    database = data.scenario_database[scenario_index]
    local = idx if scenario_index == 0 else idx - data.len_record[scenario_index - 1]
    return rules._scenario_name(database), data.return_timestamp_key(database, local), next(iter(database))


# --------------------------------------------------------------------------------------------------------------
# Pose conditions
# --------------------------------------------------------------------------------------------------------------

def load_poses(root: Path, plugins, sigmas, seeds, first_seed: int = 0):
    """{(kind, sigma, seed): {(scenario, frame_id, cav_id): 4x4}}; bare stores the reported 6-dof pose."""
    table = {}
    for sigma in sigmas:
        for seed in ([0] if sigma == 0.0 else range(first_seed, seeds)):
            reference = None
            for plugin in plugins:
                with np.load(root / "v2xreal" / plugin / f"sigma_{sigma:g}_seed_{seed}.npz") as z:
                    keys = list(zip(z["scenario"].tolist(), z["frame_id"].tolist(), z["cav_id"].tolist()))
                    table[(plugin, sigma, seed)] = dict(zip(keys, z["transform"]))
                    reported = dict(zip(keys, z["reported_pose"]))
                if reference is None:
                    reference = reported
                elif any(not np.array_equal(reference[k], reported[k]) for k in keys):
                    raise ValueError(f"{plugin} sigma {sigma} seed {seed}: reported poses differ between plug-ins")
            table[("bare", sigma, seed)] = {"reported": reference}
    return table


class FrameConditions(torch.utils.data.IterableDataset):
    """Each worker takes every num_workers-th frame, reads it once, and yields it ONE CONDITION AT A TIME as
    (frame index, condition index, collated batch, in-range CAVs without a pose), so memory stays a few conditions
    per worker (all 40 at once filled 61 GB of RAM). main() merges the results back in frame order."""

    def __init__(self, data, frames, conditions, table, comm_range=OUR_COMM_RANGE):
        self.data, self.frames, self.conditions, self.table = data, frames, conditions, table
        self.comm_range = comm_range
        self.original = data.retrieve_base_data

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        mine = self.frames if info is None else self.frames[info.id::info.num_workers]
        for idx in mine:
            yield from self.frame(int(idx))

    def frame(self, idx):
        data = self.data
        base = self.original(idx)
        scenario, timestamp, _ = frame_key(data, idx)
        ego_id = next(c for c, v in base.items() if v["ego"])
        ego_pose = base[ego_id]["params"]["lidar_pose"]          # recorded pose (no noise in the base data)
        ego_world = x_to_world(ego_pose)
        # the CAV set: within our 70 m by TRUE distance, the same for every condition
        keep = [c for c, v in base.items() if v["ego"] or math.hypot(
            v["params"]["lidar_pose"][0] - ego_pose[0], v["params"]["lidar_pose"][1] - ego_pose[1])
            <= self.comm_range]
        for c, condition in enumerate(self.conditions):
            kind = condition[0]
            if kind == "ego":
                yield idx, c, self.ego_only(idx, base, keep), 0
                continue
            frame, lacking = OrderedDict(), 0
            for cav_id in keep:
                content = base[cav_id]
                entry = dict(content)
                params = dict(content["params"])
                params["lidar_pose_clean"] = list(content["params"]["lidar_pose"])
                params["lidar_pose"] = list(content["params"]["lidar_pose"])
                if kind != "true" and not content["ego"]:
                    key = (scenario, timestamp, str(cav_id))
                    if kind == "bare":
                        reported = self.table[condition]["reported"].get(key)
                        transform = None if reported is None else x1_to_x2(list(reported), ego_pose)
                    else:
                        transform = self.table[condition].get(key)
                    if transform is None:
                        lacking += 1                       # would keep its TRUE pose: counted, must stay 0
                    else:
                        params["lidar_pose"] = tfm_to_pose(ego_world @ np.asarray(transform, np.float64))
                entry["params"] = params
                frame[cav_id] = entry
            data.retrieve_base_data = lambda *_a, _f=frame, **_k: copy.deepcopy(_f)
            np.random.seed(idx)
            batch = data.collate_batch_test([data[idx]])
            data.retrieve_base_data = self.original
            yield idx, c, batch, lacking


    def ego_only(self, idx, base, keep):
        """The ego's data alone through the network, scored on the SAME ground truth as every other condition:
        the input batch has only the ego, the ground-truth boxes come from the full frame (same CAV set) at true poses."""
        data = self.data
        full_frame = OrderedDict()
        for cav_id in keep:
            entry = dict(base[cav_id])
            params = dict(base[cav_id]["params"])
            params["lidar_pose_clean"] = list(params["lidar_pose"])
            entry["params"] = params
            full_frame[cav_id] = entry
        batches = []
        for frame in (full_frame, OrderedDict((k, v) for k, v in full_frame.items() if v["ego"])):
            data.retrieve_base_data = lambda *_a, _f=frame, **_k: copy.deepcopy(_f)
            np.random.seed(idx)
            batches.append(data.collate_batch_test([data[idx]]))
            data.retrieve_base_data = self.original
        full, ego = batches
        for key in ("object_bbx_center", "object_bbx_mask", "object_ids"):
            if key in full["ego"]:
                ego["ego"][key] = full["ego"][key]
        return ego


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True, help="folder under models/v2xreal/intermediate_fusion/")
    parser.add_argument("--plugins", nargs="+", default=["freealign", "coalign", "roco", "alignformer"])
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0])
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--first-seed", type=int, default=0,
                        help="first noise draw: draws first-seed .. seeds-1 (draw 0 can come from an earlier run)")
    parser.add_argument("--max-frames", type=int, default=0, help="evenly spaced subset of frames (0 = all)")
    parser.add_argument("--poses", type=Path, default=REPO / "outputs" / "poses")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True,
                        help="a .json path; with '{sigma}' in it, one file per sigma from ONE pass over the frames")
    parser.add_argument("--opencood-iou", action="store_true", help="the code's own shapely overlaps (slow)")
    parser.add_argument("--ego-only", action="store_true",
                        help="only the ego-only condition: the ego's data alone, scored on the full frame's ground truth")
    parser.add_argument("--comm-range", type=float, default=OUR_COMM_RANGE,
                        help="CAVs kept, by true distance (default: our 70 m; the network's config says 120)")
    args = parser.parse_args()
    if not args.opencood_iou:
        install_fast_eval()

    work = REPO / "outputs" / "paper" / "hosts_v2xreal" / "_config"
    hypes = host_hypes(args.host, write_assignment(work / "v2xreal_lidaronly.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_utils.create_model(hypes)
    checkpoint = sorted((HOSTS / args.host).glob("net_epoch*.pth"))[-1]
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    data = build_dataset(hypes, visualize=False, train=False)

    if args.ego_only:
        table, conditions = {}, [("ego", None, None)]
    else:
        table = load_poses(args.poses, args.plugins, args.sigmas, args.seeds, args.first_seed)
        conditions = [("true", None, None)] + [k for k in table if not (k[0] == "bare" and k[1] == 0.0)]
    frames = np.arange(len(data))
    if args.max_frames:
        frames = np.unique(np.linspace(0, len(data) - 1, args.max_frames).round().astype(int))

    per_frame = {c: {} for c in conditions}      # condition -> frame -> that frame's tp / fp / score / gt
    missing = {c: 0 for c in conditions}
    loader = torch.utils.data.DataLoader(
        FrameConditions(data, frames, conditions, table, args.comm_range), batch_size=None,
        num_workers=args.workers, prefetch_factor=2 if args.workers else None)
    start = time.time()
    for idx, c, batch, lacking in tqdm(loader, total=len(frames) * len(conditions), unit="cond",
                                       desc=f"v2xreal {args.host} ({len(frames)} frames x {len(conditions)} conditions)"):
        condition = conditions[c]
        missing[condition] += lacking
        if batch is None:
            continue
        batch = train_utils.to_device(batch, device)
        with torch.no_grad():
            result = inference_utils.inference_intermediate_fusion(batch, model, data)
        mine = {iou: {"tp": [], "fp": [], "gt": 0, "score": []} for iou in IOUS}
        for iou in IOUS:
            eval_utils.caluclate_tp_fp(result["pred_box_tensor"], result["pred_score"], result["gt_box_tensor"],
                                       mine, iou)
        per_frame[condition][idx] = mine
    seconds = time.time() - start
    stats = {c: {iou: {"tp": [], "fp": [], "gt": 0, "score": []} for iou in IOUS} for c in conditions}
    for c in conditions:                          # frame order, as one sequential pass would give
        for idx in sorted(per_frame[c]):
            for iou in IOUS:
                for key in ("tp", "fp", "score"):
                    stats[c][iou][key] += per_frame[c][idx][iou][key]
                stats[c][iou]["gt"] += per_frame[c][idx][iou]["gt"]

    def name(c):
        return c[0] if c[0] in ("true", "ego") else f"{c[0]}/sigma_{c[1]:g}/seed_{c[2]}"

    results = {}
    for c in conditions:
        results[name(c)] = {f"ap_{int(iou * 100)}": float(eval_utils.calculate_ap(stats[c], iou)[0]) for iou in IOUS}
        results[name(c)]["cavs_without_pose"] = missing[c]
        results[name(c)]["gt_objects"] = int(stats[c][IOUS[0]]["gt"])
    header = {"dataset": "v2xreal", "host": args.host, "checkpoint": str(checkpoint), "frames": len(frames),
              "code": "jigsawcomm-v2x baselines-package (external/jigsawcomm-v2x), our V2X-Real protocol"}
    per = seconds / max(len(frames) * len(conditions), 1)
    if "{sigma}" in str(args.output):
        for sigma in args.sigmas:
            mine = [c for c in conditions if c[0] == "true" or c[1] == sigma]
            path = Path(str(args.output).replace("{sigma}", f"{sigma:g}"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({**header, "seconds": per * len(frames) * len(mine),
                                        "seconds_per_frame_condition": per, "one_pass_sigmas": args.sigmas,
                                        "results": {name(c): results[name(c)] for c in mine}}, indent=1))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({**header, "seconds": seconds, "seconds_per_frame_condition": per,
                                           "results": results}, indent=1))
    print(f"{args.host}: {len(frames)} frames x {len(conditions)} conditions in {seconds:.0f} s -> {args.output}")
    for c in conditions:
        r = results[name(c)]
        print(f"  {name(c):32s} AP@0.7 {r['ap_70']:.3f}  AP@0.5 {r['ap_50']:.3f}  CAVs without a pose {r['cavs_without_pose']}")


if __name__ == "__main__":
    main()
