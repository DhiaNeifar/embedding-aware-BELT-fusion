"""Paper Fig. 1 data: uncorrected late-fusion AP on OPV2V under position-only and heading-only pose noise.
Evaluation only: reads true poses to perturb them, as the benchmark simulator does. One draw per level."""
import sys, json, concurrent.futures as cf, multiprocessing as mp
from pathlib import Path
import numpy as np, torch
from tqdm import tqdm
ROOT = Path("/mnt/external/workspace/dhia/AlignFormer"); sys.path.insert(0, str(ROOT / "scripts"))
import check_verify_core, reproduce_opv2v_sweep as base
from embedding_aware_belt_fusion.alignformer_ext import evaluate
base.fusion_core.set_filters(140.0, 40.0, -3.0, 1.0, 6.0)
REC = check_verify_core.load_frame_table(ROOT / "outputs/opv2v_frame_table.pt")
NMS = float(base.load_yaml(str(ROOT / "configs/pointpillar_late_fusion_r140.yaml"), None)["postprocess"]["nms_thresh"])
def run(cond):
    kind, level = cond
    frame, rank, boxes, count = [], [], [], []
    for f, r in enumerate(REC):
        for k, cav in enumerate(sorted(r.cav_boxes)):
            if cav not in r.cav_poses: continue
            rng = np.random.default_rng([f, k, int(level * 1000), 7 if kind == "position" else 11])
            pose = list(map(float, r.cav_poses[cav]))
            if kind == "position":
                pose[base.POSE_X] += rng.normal(0, level); pose[base.POSE_Y] += rng.normal(0, level)
            else:
                pose[base.POSE_YAW] += rng.normal(0, level)
            T = torch.as_tensor(base.x1_to_x2(pose, r.ego_pose), dtype=torch.float32)
            b = base.transform_boxes_to_ego(r.cav_boxes[cav], T).numpy().astype(np.float32)
            frame.append(f); rank.append(k); boxes.append(b); count.append(len(b))
    cache = {"frame": np.array(frame), "cav_rank": np.array(rank), "cav_boxes": np.concatenate(boxes),
             "cav_off": np.concatenate([[0], np.cumsum(count)]), "attempted": np.ones(len(frame), bool)}
    return kind, level, evaluate.ap_table(base, evaluate.score_condition(base, REC, cache, None, NMS))["all"]
conds = [("position", s) for s in (0.25, 0.5, 1.0, 2.0)] + [("heading", s) for s in (0.5, 1.0, 2.0, 4.0)]
out = []
with cf.ProcessPoolExecutor(8, mp_context=mp.get_context("fork")) as pool:
    for res in tqdm(pool.map(run, conds), total=len(conds), desc="error types"):
        out.append({"kind": res[0], "level": res[1], **res[2]}); print(out[-1], flush=True)
json.dump({"dataset": "opv2v", "fusion": "late", "seed": "one draw per level", "results": out},
          open(ROOT / "outputs/paper/audit/intro_error_types.json", "w"), indent=1)
