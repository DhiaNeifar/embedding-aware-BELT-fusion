#!/usr/bin/env python3
"""Run one unchanged intermediate-fusion network (a "host") under every pose condition (Table T1).

Conditions, all on the same frames, scored by OpenCOOD's own evaluator (its NMS and AP matching, with the box
overlaps computed in C++ by alignformer_ext.fast_opencood: identical decisions, checked by
scripts/check_fast_opencood.py; --opencood-iou uses OpenCOOD's shapely overlaps instead):

* ``true``: the true pose (upper bound);
* ``bare``: the CAV's noisy reported pose, as the network would get it with no correction;
* one per plug-in: the plug-in's corrected pose, read from ``outputs/poses/<dataset>/<plugin>/`` (scripts/export_poses.py).

The network and its weights never change; the only thing that differs between conditions is each CAV's
``params['transformation_matrix']``, the one place OpenCOOD's intermediate fusion uses a CAV's pose (it projects the
points before voxelization, or builds the pairwise feature transforms from it). Ground truth is built from the true
poses as OpenCOOD always does. OpenCOOD's own noise and delay simulation (``wild_setting``) is switched off. Each
frame's LiDAR is read once and the point-shuffling seed is fixed per frame, so every condition sees identical input.

    python scripts/run_host.py --dataset opv2v --host attfuse --max-frames 20 --output outputs/paper/dryrun/attfuse.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "external" / "OpenCOOD"))
sys.path.insert(0, str(REPO / "src"))

from opencood.data_utils.datasets import build_dataset  # noqa: E402
from opencood.hypes_yaml.yaml_utils import load_yaml  # noqa: E402
from opencood.tools import inference_utils, train_utils  # noqa: E402
from opencood.utils import eval_utils  # noqa: E402
from opencood.utils.transformation_utils import x1_to_x2  # noqa: E402

from embedding_aware_belt_fusion.alignformer_ext import fast_opencood, poses  # noqa: E402

SPLITS = {"opv2v": Path("/mnt/external/workspace/public/dataset/opv2v/test")}
HOSTS = {"opv2v": REPO / "models" / "opv2v" / "intermediate_fusion"}
IOUS = (0.3, 0.5, 0.7)


def COM_RANGE() -> float:
    import opencood.data_utils.datasets as opencood_datasets

    return opencood_datasets.COM_RANGE

# point_pillar_intermediate_V2VAM imports two names this OpenCOOD fork lacks and never uses them (Chenyi's
# alignformer/baselines.py); stand-ins that raise make the model loadable without inventing behaviour.
_V2VAM_DEAD_IMPORTS = (("opencood.models.sub_modules.noise", ("data_dropout", "data_dropout_uniform", "transmission_with_noise")),
                       ("opencood.models.fuse_modules.self_attn", ("regroup",)))


def _shim_v2vam():
    import importlib
    import types

    def unavailable(*_args, **_kwargs):
        raise NotImplementedError("stub for a dead import of point_pillar_intermediate_V2VAM")

    for module_name, symbols in _V2VAM_DEAD_IMPORTS:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            module = types.ModuleType(module_name)
            sys.modules[module_name] = module
        for symbol in symbols:
            if not hasattr(module, symbol):
                setattr(module, symbol, unavailable)


def load_host(dataset: str, host: str, device):
    directory = HOSTS[dataset] / host
    hypes = load_yaml(str(directory / "config.yaml"), None)
    hypes.pop("wild_setting", None)
    hypes["root_dir"] = hypes["validate_dir"] = str(SPLITS[dataset])
    data = build_dataset(hypes, visualize=False, train=False)
    _shim_v2vam()
    model = train_utils.create_model(hypes)
    checkpoints = sorted(directory.glob("net_epoch*.pth"), key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
    checkpoint = checkpoints[-1] if checkpoints else directory / "latest.pth"
    state = torch.load(str(checkpoint), map_location="cpu")
    state = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state)
    return data, model.to(device).eval(), checkpoint


def frame_key(data, idx: int):
    """(scenario folder, timestamp) of dataset frame ``idx``, as OpenCOOD's retrieve_base_data resolves it."""
    scenario_index = next(i for i, n in enumerate(data.len_record) if idx < n)
    database = data.scenario_database[scenario_index]
    timestamp_index = idx if scenario_index == 0 else idx - data.len_record[scenario_index - 1]
    timestamp = data.return_timestamp_key(database, timestamp_index)
    any_cav = next(iter(database.values()))
    return Path(any_cav[timestamp]["yaml"]).parent.parent.name, timestamp


def load_poses(root: Path, dataset: str, plugins, sigmas, seeds, first_seed: int = 0):
    """{(kind, sigma, seed): {(scenario, frame_id, cav_id): 4x4}} for ``bare`` and every plug-in."""
    table = {}
    for sigma in sigmas:
        for seed in ([0] if sigma == 0.0 else range(first_seed, seeds)):
            reference = None
            for plugin in plugins:
                path = poses.condition_path(root, dataset, plugin, sigma, seed)
                with np.load(path) as z:
                    keys = list(zip(z["scenario"].tolist(), z["frame_id"].tolist(), z["cav_id"].tolist()))
                    table[(plugin, sigma, seed)] = dict(zip(keys, z["transform"]))
                    reported = dict(zip(keys, z["reported_pose"]))
                if reference is None:
                    reference = reported
                elif any(not np.array_equal(reference[k], reported[k]) for k in keys):
                    raise ValueError(f"{path}: reported poses differ between plug-ins (noise draws not shared)")
            table[("bare", sigma, seed)] = {"reported": reference}
    return table


def _keep_one_point(entry, lidar_range) -> bool:
    """If a CAV placed with its transform keeps no LiDAR point inside the ego's detection range (a wildly wrong
    correction), give it one zero-intensity point at the edge of the range instead of none. It then contributes
    nothing, but OpenCOOD keeps one feature map per agent: with none, a last agent's map is dropped and the network
    gets fewer maps than agents. The CAV stays in the frame, so the ground truth is unchanged. True if applied."""
    points, transform = entry["lidar_np"], np.asarray(entry["params"]["transformation_matrix"], np.float64)
    placed = np.c_[points[:, :3], np.ones(len(points))] @ transform.T
    r = lidar_range
    inside = ((placed[:, 0] > r[0]) & (placed[:, 0] < r[3]) & (placed[:, 1] > r[1]) & (placed[:, 1] < r[4])
              & (placed[:, 2] > r[2]) & (placed[:, 2] < r[5]))
    if inside.any():
        return False
    corner = np.array([r[3] - 0.1, r[4] - 0.1, 0.5 * (r[2] + r[5]), 1.0])
    local = np.linalg.inv(transform) @ corner
    point = np.zeros((1, points.shape[1]), points.dtype)
    point[0, :3] = local[:3]
    entry["lidar_np"] = np.concatenate([points, point])
    return True


class FrameConditions(torch.utils.data.IterableDataset):
    """Each worker takes every num_workers-th frame, reads it once, and yields it ONE CONDITION AT A TIME as
    (frame index, condition index, collated batch, in-range CAVs without a pose, CAVs given one point). Memory is
    then a few conditions per worker, not every condition of a frame (40 per frame filled 61 GB of RAM with two
    runs of 10 workers). The CPU preprocessing (projection, voxelization, ground truth) runs in the workers while
    the GPU runs the network; main() merges the results back in frame order."""

    def __init__(self, data, frames, conditions, table):
        self.data, self.frames, self.conditions, self.table = data, frames, conditions, table
        self.lidar_range = np.asarray(data.params["preprocess"]["cav_lidar_range"], np.float64)
        self.original = data.retrieve_base_data

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        mine = self.frames if info is None else self.frames[info.id::info.num_workers]
        for idx in mine:
            yield from self.frame(int(idx))

    def frame(self, idx):
        data = self.data
        base_data = self.original(idx)
        scenario, timestamp = frame_key(data, idx)
        ego_pose = next(c["params"]["lidar_pose"] for c in base_data.values() if c["ego"])
        for c, condition in enumerate(self.conditions):
            kind = condition[0]
            if kind == "ego":
                yield idx, c, self.ego_only(idx, base_data), 0, 0
                continue
            frame, lacking, emptied = OrderedDict(), 0, 0
            for cav_id, content in base_data.items():
                entry = dict(content)
                if kind != "true" and not content["ego"]:
                    key = (scenario, timestamp, str(cav_id))
                    if kind == "bare":
                        reported = self.table[condition]["reported"].get(key)
                        transform = None if reported is None else x1_to_x2(list(reported), ego_pose)
                    else:
                        transform = self.table[condition].get(key)
                    if transform is None:
                        # a CAV in range with no pose row would keep its TRUE transform: count it, it must stay 0
                        cav_pose = content["params"]["lidar_pose"]
                        if np.hypot(cav_pose[0] - ego_pose[0], cav_pose[1] - ego_pose[1]) <= COM_RANGE():
                            lacking += 1
                    else:
                        params = dict(content["params"])
                        params["transformation_matrix"] = np.asarray(transform, np.float64)
                        entry["params"] = params
                if not content["ego"] and _keep_one_point(entry, self.lidar_range):
                    emptied += 1
                frame[cav_id] = entry
            data.retrieve_base_data = lambda *_a, _f=frame, **_k: _f
            np.random.seed(idx)
            batch = data.collate_batch_test([data[idx]])
            data.retrieve_base_data = self.original
            yield idx, c, batch, lacking, emptied


    def ego_only(self, idx, base_data):
        """The ego's data alone through the network, scored on the SAME ground truth as every other condition:
        the input batch has only the ego, the ground-truth boxes come from the full frame at true poses."""
        data = self.data
        batches = []
        for frame in (OrderedDict(base_data), OrderedDict((k, v) for k, v in base_data.items() if v["ego"])):
            data.retrieve_base_data = lambda *_a, _f=frame, **_k: _f
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
    parser.add_argument("--dataset", choices=sorted(SPLITS), default="opv2v")
    parser.add_argument("--host", required=True, help="folder under models/<dataset>/intermediate_fusion/")
    parser.add_argument("--plugins", nargs="+", default=["freealign", "coalign", "roco", "alignformer"])
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--first-seed", type=int, default=0,
                        help="first noise draw: draws first-seed .. seeds-1 (draw 0 can come from an earlier run)")
    parser.add_argument("--max-frames", type=int, default=0, help="evenly spaced subset of frames (0 = all)")
    parser.add_argument("--poses", type=Path, default=REPO / "outputs" / "poses")
    parser.add_argument("--workers", type=int, default=12, help="CPU processes preparing frames")
    parser.add_argument("--output", type=Path, required=True,
                        help="a .json path; with '{sigma}' in it, one file per sigma (each with the true-pose row), "
                             "all from ONE pass over the frames")
    parser.add_argument("--ego-only", action="store_true",
                        help="only the ego-only condition: the ego's data alone, scored on the full frame's ground truth")
    parser.add_argument("--opencood-iou", action="store_true",
                        help="box overlaps with OpenCOOD's shapely code (slow) instead of the identical C++ ones")
    args = parser.parse_args()
    if not args.opencood_iou:
        fast_opencood.install()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, model, checkpoint = load_host(args.dataset, args.host, device)
    if args.ego_only:
        table, conditions = {}, [("ego", None, None)]
    else:
        table = load_poses(args.poses, args.dataset, args.plugins, args.sigmas, args.seeds, args.first_seed)
        conditions = [("true", None, None)] + [(kind, sigma, seed) for (kind, sigma, seed) in table
                                              if not (kind == "bare" and sigma == 0.0)]
    frames = np.arange(len(data))
    if args.max_frames:
        frames = np.unique(np.linspace(0, len(data) - 1, args.max_frames).round().astype(int))

    per_frame = {c: {} for c in conditions}      # condition -> frame -> that frame's tp / fp / score / gt
    missing = {c: 0 for c in conditions}
    empty = {c: 0 for c in conditions}
    loader = torch.utils.data.DataLoader(
        FrameConditions(data, frames, conditions, table), batch_size=None,
        num_workers=args.workers, prefetch_factor=2 if args.workers else None, persistent_workers=False)
    start = time.time()
    for idx, c, batch, lacking, emptied in tqdm(loader, total=len(frames) * len(conditions), unit="cond",
                                                desc=f"{args.dataset} {args.host} ({len(frames)} frames x {len(conditions)} conditions)"):
        condition = conditions[c]
        missing[condition] += lacking
        empty[condition] += emptied
        batch = train_utils.to_device(batch, device)
        with torch.no_grad():
            pred_box, pred_score, gt_box = inference_utils.inference_intermediate_fusion(batch, model, data)
        mine = {iou: {"tp": [], "fp": [], "gt": 0, "score": []} for iou in IOUS}
        for iou in IOUS:
            eval_utils.caluclate_tp_fp(pred_box, pred_score, gt_box, mine, iou)
        per_frame[condition][idx] = mine
    seconds = time.time() - start
    # frames arrive from several workers in no fixed order; OpenCOOD's AP depends on the order, so merge by frame
    stats = {c: {iou: {"tp": [], "fp": [], "gt": 0, "score": []} for iou in IOUS} for c in conditions}
    for c in conditions:
        for idx in sorted(per_frame[c]):
            for iou in IOUS:
                for key in ("tp", "fp", "score"):
                    stats[c][iou][key] += per_frame[c][idx][iou][key]
                stats[c][iou]["gt"] += per_frame[c][idx][iou]["gt"]

    def name(c):
        return c[0] if c[0] in ("true", "ego") else f"{c[0]}/sigma_{c[1]:g}/seed_{c[2]}"

    results = {}
    for c in conditions:
        results[name(c)] = {f"ap_{int(iou * 100)}": float(eval_utils.calculate_ap(stats[c], iou, False)[0]) for iou in IOUS}
        results[name(c)]["cavs_without_pose"] = missing[c]
        results[name(c)]["cavs_placed_out_of_range"] = empty[c]
        results[name(c)]["gt_objects"] = int(stats[c][IOUS[0]]["gt"])
    header = {"dataset": args.dataset, "host": args.host, "checkpoint": str(checkpoint), "frames": len(frames)}
    if "{sigma}" in str(args.output):
        # One pass, one file per sigma: the same files a run per sigma writes, each with the true-pose row.
        # "seconds" is this sigma's share of the pass (its conditions out of all of them).
        per_frame_condition = seconds / max(len(frames) * len(conditions), 1)
        for sigma in args.sigmas:
            mine = [c for c in conditions if c[0] == "true" or c[1] == sigma]
            path = Path(str(args.output).replace("{sigma}", f"{sigma:g}"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({**header, "seconds": per_frame_condition * len(frames) * len(mine),
                                        "seconds_per_frame_condition": per_frame_condition, "one_pass_sigmas": args.sigmas,
                                        "results": {name(c): results[name(c)] for c in mine}}, indent=1))
        print(f"{args.host}: {len(frames)} frames x {len(conditions)} conditions in {seconds:.0f} s -> "
              f"{len(args.sigmas)} files {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({**header, "seconds": seconds,
                                           "seconds_per_frame_condition": seconds / max(len(frames) * len(conditions), 1),
                                           "results": results}, indent=1))
        print(f"{args.host}: {len(frames)} frames x {len(conditions)} conditions in {seconds:.0f} s -> {args.output}")
    for c in conditions:
        r = results[name(c)]
        print(f"  {name(c):32s} AP@0.7 {r['ap_70']:.3f}  AP@0.5 {r['ap_50']:.3f}  CAVs without a pose {r['cavs_without_pose']}")


if __name__ == "__main__":
    main()
