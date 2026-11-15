#!/usr/bin/env python3
"""CoAlign / RoCo stage 1 on V2X-Real, run with CoAlign's own code (not edited).

CoAlign and RoCo correct poses with a pose graph over each agent's own boxes and their uncertainty, which come from
a stage-1 single-agent detector. That detector is learned, so V2X-Real needs its own. Three commands:

  check   The CoAlign-side V2X-Real dataset lists the same frames, ego and CAVs as our V2X-Real pose files
          (frame by frame). Run it first; it reads no LiDAR.
  train   CoAlign's own opencood/tools/train.py, unchanged, with fusion.dataset: v2xreal registered.
  boxes   stage1_boxes.json for one split, as CoAlign's opencood/tools/pose_graph_pre_calc.py writes it (same model
          call, same post_process_stage1, same keys), for the one split we need instead of all three.

Run in the CoAlign environment, from the repository root:

  ENV=/mnt/external/backup/dhia/conda_envs
  PYTHONPATH=src $ENV/coalign-blackwell/bin/python scripts/coalign_v2xreal.py check
  PYTHONPATH=src $ENV/coalign-blackwell/bin/python scripts/coalign_v2xreal.py train
  PYTHONPATH=src $ENV/coalign-blackwell/bin/python scripts/coalign_v2xreal.py boxes --stage1-dir <log dir> --split test
"""

from __future__ import annotations

import argparse
import os
import copy
import json
import runpy
import sys
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COALIGN = REPO / "external" / "CoAlign"
# Per dataset: the stage-1 config (CoAlign's recipe, the dataset's data settings), where its boxes go, and one of our
# pose files to check frame-by-frame parity against. OPV2V's go to a new folder, next to (not over) the boxes that
# came with the labmate's CoAlign bundle in models/opv2v/intermediate_fusion/coalign_boxalign/.
DATASETS = {
    "v2xreal": dict(yaml=REPO / "models/v2xreal/intermediate_fusion/coalign_boxalign/stage1_pointpillar_uncertainty.yaml",
                    out=REPO / "models/v2xreal/intermediate_fusion/coalign_boxalign",
                    poses=REPO / "outputs/poses/v2xreal/alignformer/sigma_0_seed_0.npz"),
    "opv2v": dict(yaml=REPO / "models/opv2v/intermediate_fusion/coalign_stage1/stage1_pointpillar_uncertainty.yaml",
                  out=REPO / "models/opv2v/intermediate_fusion/coalign_stage1",
                  poses=REPO / "outputs/poses/opv2v/alignformer/sigma_0_seed_0.npz"),
}
STAGE1_YAML = DATASETS["v2xreal"]["yaml"]
SPLIT_DIR = {"train": "root_dir", "val": "validate_dir", "test": "test_dir"}

sys.path.insert(0, str(COALIGN))


def _install():
    from embedding_aware_belt_fusion.multidataset import coalign_v2xreal
    coalign_v2xreal.install()


def _hypes(path: Path):
    from opencood.hypes_yaml import yaml_utils
    return yaml_utils.load_yaml(str(path), None)


def frame_key(dataset, idx: int):
    """(scenario folder, timestamp, agent ids in order) of index ``idx`` of any CoAlign base dataset."""
    from embedding_aware_belt_fusion.multidataset import v2xreal as rules
    scenario_index = next(i for i, end in enumerate(dataset.len_record) if idx < end)
    database = dataset.scenario_database[scenario_index]
    local = idx if scenario_index == 0 else idx - dataset.len_record[scenario_index - 1]
    return rules._scenario_name(database), dataset.return_timestamp_key(database, local), tuple(database.keys())


def box_dataset_hypes(stage1: dict, split: str) -> dict:
    """The stage-1 config turned into CoAlign's precalc dataset: intermediate fusion (every agent of a frame in one
    item, each in its own frame, proj_first false), no noise, every agent kept (comm_range 1e5, as CoAlign's own
    OPV2V precalc.yaml, including its max_cav 7: OPV2V scenes have up to 7 vehicles), the requested split."""
    hypes = copy.deepcopy(stage1)
    hypes["fusion"] = {"core_method": "intermediate", "dataset": stage1["fusion"]["dataset"], "args": {"proj_first": False}}
    hypes.setdefault("train_params", {})["max_cav"] = 7
    hypes["comm_range"] = 100000.0
    hypes["noise_setting"] = OrderedDict(add_noise=False)
    hypes["validate_dir"] = stage1[SPLIT_DIR[split]]
    return hypes


def cmd_check(args) -> None:
    import numpy as np
    from opencood.data_utils.datasets import build_dataset

    cfg = DATASETS[args.dataset]
    dataset = build_dataset(box_dataset_hypes(_hypes(cfg["yaml"]), "test"), visualize=False, train=False)
    with np.load(args.poses or cfg["poses"]) as z:
        rows = {k: z[k] for k in ("frame_index", "scenario", "frame_id", "ego_id", "cav_id")}
    frames = np.unique(rows["frame_index"])
    print(f"CoAlign-side {args.dataset} test: {len(dataset)} frames; our pose file: {len(frames)} frames with a CAV, "
          f"indices up to {frames.max()}")
    bad = []
    for f in frames:
        scenario, timestamp, agents = frame_key(dataset, int(f))
        sel = rows["frame_index"] == f
        want = (str(rows["scenario"][sel][0]), str(rows["frame_id"][sel][0]), str(rows["ego_id"][sel][0]))
        got = (scenario, timestamp, agents[0])
        missing = set(map(str, rows["cav_id"][sel])) - set(agents)
        if want != got or missing:
            bad.append((int(f), want, got, sorted(missing)))
    if bad:
        for b in bad[:10]:
            print("MISMATCH frame", b)
        raise SystemExit(f"{len(bad)} of {len(frames)} frames differ")
    print(f"OK: all {len(frames)} frames have the same scenario, timestamp and ego, and every CAV is present")


def cmd_train(args) -> None:
    if args.data_root:
        # Same files, faster disk: point train / val at a copy of V2X-Real. The log folder's config.yaml records it.
        from opencood.hypes_yaml import yaml_utils
        load_yaml, root = yaml_utils.load_yaml, Path(args.data_root)
        original = Path(_hypes(args.hypes_yaml)["root_dir"]).parent
        for split in ("train", "val"):        # refuse a partial copy (e.g. one still being written)
            count = lambda d: sum(len(files) for _, _, files in os.walk(d))
            want, have = count(original / split), count(root / split)
            if have != want:
                raise SystemExit(f"{root / split} has {have} of {want} files: the copy is not complete. "
                                 "Wait for 'run_paper.sh v2xreal-copy' to finish, or run it again.")
        print(f"training on the copy in {root} (train and val complete)")

        def load_with_root(file, opt=None):
            hypes = load_yaml(file, opt)
            if isinstance(hypes, dict) and "root_dir" in hypes:
                hypes["root_dir"], hypes["validate_dir"] = str(root / "train"), str(root / "val")
            return hypes

        yaml_utils.load_yaml = load_with_root
    if args.workers:
        # train.py hard-codes num_workers=4 for its loaders. More workers only load the same batches faster;
        # the recipe (batch size, epochs, learning rate) is unchanged.
        from torch.utils.data import DataLoader
        init = DataLoader.__init__

        def init_with_workers(self, *a, **k):
            k["num_workers"] = args.workers
            init(self, *a, **k)

        DataLoader.__init__ = init_with_workers
    argv = ["train.py", "--hypes_yaml", str(args.hypes_yaml)]
    if args.model_dir:
        argv += ["--model_dir", str(args.model_dir)]
    sys.argv = argv
    runpy.run_path(str(COALIGN / "opencood/tools/train.py"), run_name="__main__")


def cmd_boxes(args) -> None:
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from opencood.data_utils.datasets import build_dataset
    from opencood.data_utils.post_processor.uncertainty_voxel_postprocessor import UncertaintyVoxelPostprocessor
    from opencood.tools import train_utils

    stage1_dir = Path(args.stage1_dir)
    cfg = DATASETS[args.dataset]
    stage1 = _hypes(cfg["yaml"])
    checkpoint = args.checkpoint or max(stage1_dir.glob("net_epoch_bestval_at*.pth"), key=lambda p: p.stat().st_mtime)
    dataset = build_dataset(box_dataset_hypes(stage1, args.split), visualize=False, train=False)
    items = dataset if not args.max_frames else torch.utils.data.Subset(dataset, range(args.max_frames))
    loader = DataLoader(items, batch_size=1, num_workers=args.workers, collate_fn=dataset.collate_batch_test,
                        shuffle=False, pin_memory=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_utils.create_model(stage1)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=False)
    model.to(device).eval()
    post = UncertaintyVoxelPostprocessor(stage1["postprocess"], train=False)
    anchor_box = torch.from_numpy(post.generate_anchor_box())

    boxes, empty = OrderedDict(), 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"stage-1 boxes, {args.dataset} {args.split}"):
            if batch is None:
                continue
            batch = train_utils.to_device(batch, device)
            corners, _, uncertainty = post.post_process_stage1(model(batch["ego"]), anchor_box)
            if corners is None:
                empty += 1
                continue
            idx = batch["ego"]["sample_idx"]
            scenario, timestamp, _ = frame_key(dataset, idx)
            boxes[idx] = OrderedDict(
                pred_corner3d_np_list=[x.cpu().numpy().tolist() for x in corners],
                uncertainty_np_list=[x.cpu().numpy().tolist() for x in uncertainty],
                lidar_pose_clean_np=batch["ego"]["lidar_pose_clean"].cpu().numpy().tolist(),
                cav_id_list=batch["ego"]["cav_id_list"],
                scenario=scenario, timestamp=timestamp)          # two extra keys, for provenance only
    out = Path(args.out or cfg["out"] / f"stage1_boxes_{args.split}.json")
    meta = dict(checkpoint=str(checkpoint), config=str(cfg["yaml"]), split=args.split, frames=len(dataset),
                frames_written=len(boxes), frames_without_boxes=empty)
    with open(out, "w") as f:
        json.dump(boxes, f, sort_keys=True)
    with open(out.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{len(boxes)} of {len(dataset)} frames written to {out} ({empty} with no box from any agent)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("--dataset", choices=sorted(DATASETS), default="v2xreal")
    p.add_argument("--poses", type=Path, help="default: the dataset's sigma 0 AlignFormer pose file")
    p = sub.add_parser("train")
    p.add_argument("--hypes-yaml", dest="hypes_yaml", type=Path, default=STAGE1_YAML)
    p.add_argument("--model-dir", dest="model_dir", type=Path, help="resume from this log folder")
    p.add_argument("--data-root", dest="data_root", type=Path, help="a copy of v2x-real with train/ and val/")
    p.add_argument("--workers", type=int, help="data-loading workers (train.py's own default: 4)")
    p = sub.add_parser("boxes")
    p.add_argument("--dataset", choices=sorted(DATASETS), default="v2xreal")
    p.add_argument("--stage1-dir", dest="stage1_dir", required=True, help="the training log folder")
    p.add_argument("--checkpoint", type=Path, help="default: the folder's latest net_epoch_bestval_at*.pth")
    p.add_argument("--split", choices=sorted(SPLIT_DIR), default="test")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path)
    p.add_argument("--max-frames", dest="max_frames", type=int, help="first N frames only (a quick test)")
    args = parser.parse_args()
    _install()
    {"check": cmd_check, "train": cmd_train, "boxes": cmd_boxes}[args.cmd](args)


if __name__ == "__main__":
    main()
