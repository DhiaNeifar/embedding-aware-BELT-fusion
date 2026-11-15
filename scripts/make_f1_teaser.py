#!/usr/bin/env python3
"""F1 (teaser): where earlier correctors fail and what AlignFormer recovers. From saved files only, no model run.

(a) One OPV2V test frame at sigma = 2 (one noise draw, seed 0), drawn with OpenCOOD's BEV canvas
    (external/CoAlign/opencood/visualization/simple_plot3d/canvas_bev.py, the OpenCOOD look: black background, white
    LiDAR points, green ground truth). Four panels, one per corrector (FreeAlign, CoAlign, RoCo, AlignFormer). Each
    panel shows the ego LiDAR, the CAV LiDAR placed with that corrector's transform, the ground truth in green, the
    CAV's detected boxes placed with the raw pose in red and placed with the corrector in its colour.
    Frame rule, not chosen by eye: see frame_panels() (all vehicles detected, raw pose visible in every panel,
    AlignFormer with the lowest box error; the median margin among the pairs that pass).
(b) Late fusion, AP@0.7 against sigma (all 8 levels), OPV2V and V2X-Real, mean of the five paired noise draws behind
    Tab. 1. AlignFormer is the full method (alignformer = alignformer_tk).
(c) Intermediate fusion with V2VAM, the newest network run on both datasets, AP@0.7 against sigma, from the five-draw
    host runs behind Tab. 3 (make_paper_exhibits.host_ap). Uses every sigma that has all five draws.
    Ego alone: outputs/paper/hosts_ego/<ds>_v2vam_ego.json.

    /mnt/external/backup/dhia/conda_envs/opencood-blackwell/bin/python scripts/make_f1_teaser.py
Output: AlignFormer-2027/figures/f1_teaser.pdf and outputs/paper/f1_teaser.json (the frame key and plotted values).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_paper_exhibits as mpe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = ROOT / "AlignFormer-2027/figures/f1_teaser.pdf"
OUT_JSON = ROOT / "outputs/paper/f1_teaser.json"
OPV2V_TEST = Path("/mnt/external/workspace/public/dataset/opv2v/test")
SIGMA_FRAME = "2"
MIN_CAV_BOXES = 6
MIN_GAIN_M = 0.0   # 0.5 gives the frames where AlignFormer is visibly better than every baseline (3 of 3,445)
HALF_WINDOW_M = (25.0, 17.0)   # half width (x) and half height (y) of the crop, metres
PX_PER_M = 12

# one colour per method across the whole figure; ground truth green only in (a)
HEX = {"bare": "#e5383b", "freealign": "#f28e2b", "coalign": "#a274d6", "roco": "#d4a800",
       "alignformer": "#2f7fe0", "gt": "#2ecc40", "perfect": "#17202B", "ego": "#6b7480"}
LABEL = {"bare": "Raw pose", "freealign": "FreeAlign", "coalign": "CoAlign", "roco": "RoCo",
         "alignformer": "AlignFormer"}


def rgb(h: str) -> tuple:
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


# ------------------------------------------------------------------------------------------------- (b) late fusion
def late_curves(dataset: str) -> dict:
    plugins = json.loads((mpe.OUT / f"late_{dataset}_plugins_20seeds.json").read_text())
    variants = json.loads((mpe.OUT / f"late_{dataset}.json").read_text())
    ego = json.loads((mpe.OUT / f"late_{dataset}_ego.json").read_text())["ego"]["all"]["ap_70"]
    curves = {m: [] for m in ["perfect", "bare", "freealign", "coalign", "roco", "one_frame", "alignformer"]}
    for sigma in mpe.SIGMAS:
        seeds = [e for e in plugins["per_seed"][sigma] if e["seed"] < mpe.N_DRAWS]
        for m in ["perfect", "bare", "freealign", "coalign", "roco", "alignformer"]:
            curves[m].append(float(np.mean([e["ap"][m]["all"]["ap_70"] for e in seeds])))
        keep = {e["seed"] for e in seeds}
        one = [e["ap"]["alignformer_c"]["all"]["ap_70"] for e in variants["per_seed"][sigma] if e["seed"] in keep]
        assert len(one) == len(keep), (dataset, sigma)
        curves["one_frame"].append(float(np.mean(one)))
    curves["ego"] = ego
    return curves


# ------------------------------------------------------------------------------------------ (c) intermediate fusion
def five_draws(dataset: str, net: str, sigma: str) -> bool:
    """True when every corrector has all N_DRAWS noise draws at this sigma (sigma 0 has no noise)."""
    if sigma == "0":
        return True
    results = {}
    for folder in (mpe.HOST_DIR[dataset], mpe.HOST_DIR_5[dataset]):
        path = folder / f"{dataset}_{net}_sigma_{sigma}.json"
        if path.exists():
            results.update(json.loads(path.read_text())["results"])
    return all(f"{m}/sigma_{float(sigma):g}/seed_{d}" in results
               for m in ["bare"] + mpe.PLUGINS for d in range(mpe.N_DRAWS))


def host_curves(dataset: str) -> dict:
    nets = ["v2vam"]   # the newest network run on both datasets (V2VAM, IEEE T-IV 2023); not a mean over networks
    sigmas = [s for s in mpe.SIGMAS if all(five_draws(dataset, n, s) for n in nets)]   # only five-draw levels
    curves = {m: [] for m in ["perfect", "bare"] + mpe.PLUGINS}
    for s in sigmas:
        rows = [mpe.host_ap(dataset, n, s) for n in nets]
        for m in curves:
            curves[m].append(float(np.mean([r[m] for r in rows])))
    egos = [json.loads((mpe.OUT / "hosts_ego" / f"{dataset}_{n}_ego.json").read_text())["results"]["ego"]["ap_70"]
            for n in nets]
    curves["ego"] = float(np.mean(egos))
    curves["sigmas"] = [float(s) for s in sigmas]
    curves["networks"] = len(nets)
    return curves


# --------------------------------------------------------------------------------------------- (a) LiDAR frame
def read_pcd(path: Path) -> np.ndarray:
    import open3d as o3d
    return np.asarray(o3d.io.read_point_cloud(str(path)).points)


def frame_panels() -> dict:
    import torch
    import check_verify_core
    import reproduce_opv2v_sweep as base
    import importlib.util
    spec = importlib.util.spec_from_file_location(   # OpenCOOD's BEV canvas (CoAlign fork), loaded from its file
        "canvas_bev", ROOT / "external/CoAlign/opencood/visualization/simple_plot3d/canvas_bev.py")
    canvas_bev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canvas_bev)

    records = check_verify_core.load_frame_table(ROOT / "outputs/opv2v_frame_table.pt")
    placed, err, ids = {}, {}, {}
    for method in ["freealign", "coalign", "roco", "alignformer"]:
        with np.load(ROOT / f"outputs/poses/opv2v/{method}/sigma_{SIGMA_FRAME}_seed_0.npz") as z:
            for f, cav, rep, ego, tr, sc, fid, eid in zip(z["frame_index"], z["cav_id"], z["reported_pose"],
                                                        z["ego_pose"], z["transform"], z["scenario"], z["frame_id"],
                                                        z["ego_id"]):
                key = (int(f), str(cav))
                record = records[key[0]]
                truth = np.asarray(base.x1_to_x2(record.cav_poses[key[1]], record.ego_pose))
                raw = np.asarray(base.x1_to_x2(list(rep), list(ego)))
                placed.setdefault(key, {"truth": truth, "bare": raw})[method] = np.asarray(tr)
                ids[key] = (str(sc), str(fid), str(eid))
                for m, t in (("bare", raw), (method, np.asarray(tr))):
                    err.setdefault(key, {})[m] = float(np.linalg.norm((np.linalg.inv(t) @ truth)[:2, 3]))
    def corners_of(rec, cav, transform):
        b = base.transform_boxes_to_ego(rec.cav_boxes[cav], torch.as_tensor(transform, dtype=torch.float32))
        c = base.box_utils.boxes_to_corners_3d(b, order=base.box_order())
        return c.numpy() if hasattr(c, "numpy") else np.asarray(c)

    def window_of(true_c):
        near = [(np.linalg.norm(true_c - c, axis=1) < 12).sum() for c in true_c]
        return (float(v) for v in true_c[int(np.argmax(near))])

    # Frame rule (authors' criteria of 2026-11-12, applied to every pair, not chosen by eye):
    #   1. the CAV's detector finds every vehicle in the window: each ground-truth box in it has a CAV detection,
    #      placed with the true pose, within 1.5 m, and the window holds at least four vehicles;
    #   2. the raw pose is visible in every panel: each baseline moves the CAV's boxes by 0.5 m or more on average;
    #   3. AlignFormer has the lowest mean box error, by at least MIN_GAIN_M over the best baseline.
    # Among the pairs that pass, the one at the median margin over the best baseline is drawn.
    candidates = []
    for k in err:
        if len(err[k]) != 5:
            continue
        rec = records[k[0]]
        if rec.gt_corners is None or len(rec.gt_corners) < 4 or len(rec.cav_boxes[k[1]]) < 4:
            continue
        true_c = corners_of(rec, k[1], placed[k]["truth"])[:, :4, :2].mean(axis=1)
        cx, cy = window_of(true_c)
        gt_c = np.asarray(rec.gt_corners)[:, :4, :2].mean(axis=1)
        inside = (np.abs(gt_c[:, 0] - cx) < HALF_WINDOW_M[0] - 1) & (np.abs(gt_c[:, 1] - cy) < HALF_WINDOW_M[1] - 1)
        if inside.sum() < 4 or any(np.linalg.norm(true_c - g, axis=1).min() > 1.5 for g in gt_c[inside]):
            continue
        box = {m: corners_of(rec, k[1], placed[k][m])[:, :4, :2].mean(axis=1) for m in placed[k]}
        if min(np.linalg.norm(box[m] - box["bare"], axis=1).mean() for m in ("freealign", "coalign", "roco")) < 0.5:
            continue
        box_err = {m: float(np.linalg.norm(box[m] - true_c, axis=1).mean()) for m in
                   ("bare", "freealign", "coalign", "roco", "alignformer")}
        margin = min(box_err[m] for m in ("freealign", "coalign", "roco")) - box_err["alignformer"]
        if margin > MIN_GAIN_M:
            candidates.append((margin, k, box_err))
    candidates.sort(key=lambda c: c[0])
    _, key, box_err = candidates[len(candidates) // 2]
    record = records[key[0]]
    scenario, frame_id, ego_id = ids[key]

    def corners(transform):
        return corners_of(record, key[1], transform)

    ego_pts = read_pcd(OPV2V_TEST / scenario / ego_id / f"{frame_id}.pcd")
    cav_pts = read_pcd(OPV2V_TEST / scenario / key[1] / f"{frame_id}.pcd")
    gt = np.asarray(record.gt_corners)
    cx, cy = window_of(corners(placed[key]["truth"])[:, :4, :2].mean(axis=1))
    xr = (cx - HALF_WINDOW_M[0], cx + HALF_WINDOW_M[0])
    yr = (cy - HALF_WINDOW_M[1], cy + HALF_WINDOW_M[1])
    shape = (int(2 * HALF_WINDOW_M[1] * PX_PER_M), int(2 * HALF_WINDOW_M[0] * PX_PER_M))   # rows (y), columns (x)

    images = {}
    raw_boxes = corners(placed[key]["bare"])
    for m in ["freealign", "coalign", "roco", "alignformer"]:
        canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=shape, canvas_x_range=xr, canvas_y_range=yr,
                                                     left_hand=True)
        t = placed[key][m]
        cav_ego = (t[:3, :3] @ cav_pts.T).T + t[:3, 3]
        for pts, colour in ((ego_pts, (235, 235, 235)), (cav_ego, (150, 150, 150))):
            xy, valid = canvas.get_canvas_coords(pts)
            canvas.draw_canvas_points(xy[valid], colors=colour)
        canvas.draw_boxes(raw_boxes, colors=rgb(HEX["bare"]), box_line_thickness=3)
        canvas.draw_boxes(corners(t), colors=rgb(HEX[m]), box_line_thickness=2)
        canvas.draw_boxes(gt, colors=rgb(HEX["gt"]), box_line_thickness=2)
        images[m] = canvas.get_canvas().copy()
    meta = {"scenario": scenario, "frame_id": frame_id, "ego_id": ego_id, "cav_id": key[1],
            "rule": "all vehicles in view detected by the CAV, every baseline moves the boxes >= 0.5 m from raw, "
                    f"AlignFormer lowest mean box error by > {MIN_GAIN_M} m; median margin among the pairs that pass",
            "pairs_passing": len(candidates), "pairs_total": len(err),
            "error_m": box_err, "origin_error_m": {m: err[key][m] for m in err[key]},
            "window_m": [list(xr), list(yr)]}
    return {"images": images, "meta": meta}


# ------------------------------------------------------------------------------------------------------- figure
def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 7, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5})
    ink, grid = HEX["perfect"], "#e3e6ea"

    frame = frame_panels()
    late = {ds: late_curves(ds) for ds in ["opv2v", "v2xreal"]}
    hosts = {ds: host_curves(ds) for ds in ["opv2v", "v2xreal"]}

    fig = plt.figure(figsize=(7.0, 3.5))
    gs_a = GridSpec(1, 4, figure=fig, left=0.005, right=0.995, bottom=0.555, top=0.925, wspace=0.03)
    gs_b = GridSpec(1, 4, figure=fig, left=0.055, right=0.995, bottom=0.095, top=0.375, wspace=0.34)

    # (a) LiDAR panels: the coloured titles name the corrector, the header names green and red
    for j, m in enumerate(["freealign", "coalign", "roco", "alignformer"]):
        ax = fig.add_subplot(gs_a[0, j])
        ax.imshow(frame["images"][m], interpolation="lanczos")
        ax.set_xticks([]), ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title(LABEL[m], fontsize=7.6, pad=2, color=HEX[m] if m != "roco" else "#b08c00", fontweight="bold")
        e_raw, e_m = frame["meta"]["error_m"]["bare"], frame["meta"]["error_m"][m]
        ax.text(0.98, 0.03, f"box error: raw {e_raw:.2f} m  $\\rightarrow$  {e_m:.2f} m", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=6.6, color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.6))
    fig.text(0.005, 0.965, "(a) One OPV2V frame at $\\sigma=2$, ground truth in green, raw pose in red",
             fontsize=7.6, fontweight="bold", ha="left")

    x8 = [float(s_) for s_ in mpe.SIGMAS]

    def style(ax, ylabel=False):
        ax.grid(axis="y", color=grid, lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=2, pad=1.5)
        ax.set_xlabel("pose noise $\\sigma$ (m, $^\\circ$)", labelpad=1)
        if ylabel:
            ax.set_ylabel("AP$_{0.7}$", labelpad=1)

    def curves(ax, c, x, ms):
        ax.axhspan(0, c["ego"], color="#f6e9ea", zorder=0)
        ax.axhline(c["ego"], color=HEX["ego"], lw=0.9, ls=(0, (1, 1.2)), zorder=1)
        ax.axhline(c["perfect"][0], color=ink, lw=0.9, ls=(0, (4, 2)), zorder=1)
        for m in ["bare", "coalign", "roco", "freealign"]:
            ax.plot(x, c[m], color=HEX[m], lw=1.1, marker="o", ms=ms, zorder=2)
        ax.plot(x, c["alignformer"], color=HEX["alignformer"], lw=1.8, marker="o", ms=ms + 0.6, zorder=4)

    # (b) late fusion
    for j, ds in enumerate(["opv2v", "v2xreal"]):
        ax = fig.add_subplot(gs_b[0, j])
        curves(ax, late[ds], x8, 2.0)
        ax.set_ylim(0.1 if ds == "opv2v" else 0.12, 0.95 if ds == "opv2v" else 0.445)
        ax.set_xlim(-0.05, 2.05)
        ax.set_xticks([0, 0.5, 1, 1.5, 2])
        ax.set_xticklabels(["0", ".5", "1", "1.5", "2"])
        ax.set_title(f"({'b' if ds == 'opv2v' else 'c'}) Late fusion, {'OPV2V' if ds == 'opv2v' else 'V2X-Real'}",
                     fontsize=7.2, pad=2)
        style(ax, ylabel=(j == 0))

    # (c) intermediate fusion with AttFuse
    for j, ds in enumerate(["opv2v", "v2xreal"]):
        ax = fig.add_subplot(gs_b[0, 2 + j])
        c = hosts[ds]
        curves(ax, c, c["sigmas"], 2.4)
        lo = min(min(c["bare"]), c["ego"]) - 0.05
        ax.set_ylim(max(0.0, lo), c["perfect"][0] + 0.04)
        ax.set_xlim(-0.05, 2.05)
        ax.set_xticks([0, 0.5, 1, 1.5, 2])   # same axis as (b); five-draw points exist at 0, 1 and 2 only
        ax.set_xticklabels(["0", ".5", "1", "1.5", "2"])
        ax.set_title(f"({'d' if ds == 'opv2v' else 'e'}) V2VAM, {'OPV2V' if ds == 'opv2v' else 'V2X-Real'}",
                     fontsize=7.2, pad=2)
        style(ax)

    handles = [Line2D([], [], color=HEX["bare"], lw=1.2, marker="o", ms=3, label="Raw pose"),
               Line2D([], [], color=HEX["freealign"], lw=1.2, marker="o", ms=3, label="FreeAlign"),
               Line2D([], [], color=HEX["coalign"], lw=1.2, marker="o", ms=3, label="CoAlign"),
               Line2D([], [], color=HEX["roco"], lw=1.2, marker="o", ms=3, label="RoCo"),
               Line2D([], [], color=HEX["alignformer"], lw=1.8, marker="o", ms=3.4, label="AlignFormer"),
               Line2D([], [], color=HEX["perfect"], lw=1.0, ls=(0, (4, 2)), label="Reference pose"),
               Line2D([], [], color=HEX["ego"], lw=1.0, ls=(0, (1, 1.2)), label="Ego alone")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.465), ncol=7, frameon=False,
               fontsize=7.2, handlelength=2.4, columnspacing=1.6)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.01, dpi=300)

    record = {"frame": frame["meta"], "late": late, "hosts": hosts}
    OUT_JSON.write_text(json.dumps(record, indent=1))
    print("frame", frame["meta"])
    for ds in hosts:
        print("hosts", ds, {m: [round(v, 3) for v in hosts[ds][m]] if isinstance(hosts[ds][m], list)
                            else hosts[ds][m] for m in hosts[ds]})
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
