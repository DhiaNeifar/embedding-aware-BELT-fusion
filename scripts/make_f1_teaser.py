#!/usr/bin/env python3
"""Figure 1: RobustAlign pose and delay robustness.

Four original LiDAR views form panel a. A two-row, three-column grid shows late
fusion under pose noise, V2VAM under pose noise, and late fusion under delay.
Rows are OPV2V and V2X-Real. All curves come from generated table bodies and
use their standard five pose-noise levels. Delay uses sigma=0 to isolate delay.

Local rebuild without outputs/paper or raw scans:
    python scripts/make_f1_teaser.py --from-tables --preview AlignFormer-2027/build/f1_preview.png
This reuses the four native LiDAR images in the existing figure PDF and changes
neither the underlying scene nor the generated tables. On the data machine,
omit --from-tables to rebuild the original scene with frame_panels(). The scene
is selected by the median margin over the best baseline among pairs where the
CAV detects all vehicles in view, all baselines move boxes and RobustAlign has
the lowest mean box error. FreeAlign's delay results exclude its clock search.
"""

from __future__ import annotations

import argparse
import io
import re
import struct
import zlib

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_paper_exhibits as mpe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = ROOT / "AlignFormer-2027/figures/f1_teaser.pdf"
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
         "alignformer": "RobustAlign"}


def rgb(h: str) -> tuple:
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


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
    #   3. RobustAlign has the lowest mean box error, by at least MIN_GAIN_M over the best baseline.
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

    def fill_boxes(canvas, boxes, colour, alpha):
        """Blend a translucent fill of the boxes' BEV footprints into the canvas."""
        import cv2
        xy, _ = canvas.get_canvas_coords(boxes[:, :4, :2].reshape(-1, 2))
        polygons = xy.reshape(-1, 4, 2)[:, :, ::-1].astype(np.int32)   # canvas coords are (row, col); cv2 wants (x, y)
        overlay = canvas.canvas.copy()
        cv2.fillPoly(overlay, list(polygons), colour)
        canvas.canvas = cv2.addWeighted(overlay, alpha, canvas.canvas, 1 - alpha, 0)

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
        # Ground truth UNDER the estimates: a translucent fill and a wide band. A correct box then shows as a thin
        # coloured line running inside the green band, so it stays visible instead of being painted over.
        fill_boxes(canvas, gt, rgb(HEX["gt"]), alpha=0.25)
        canvas.draw_boxes(gt, colors=rgb(HEX["gt"]), box_line_thickness=5)
        canvas.draw_boxes(raw_boxes, colors=rgb(HEX["bare"]), box_line_thickness=2)
        canvas.draw_boxes(corners(t), colors=rgb(HEX[m]), box_line_thickness=2)
        images[m] = canvas.get_canvas().copy()
    meta = {"scenario": scenario, "frame_id": frame_id, "ego_id": ego_id, "cav_id": key[1],
            "rule": "all vehicles in view detected by the CAV, every baseline moves the boxes >= 0.5 m from raw, "
                    f"RobustAlign lowest mean box error by > {MIN_GAIN_M} m; median margin among the pairs that pass",
            "pairs_passing": len(candidates), "pairs_total": len(err),
            "error_m": box_err, "origin_error_m": {m: err[key][m] for m in err[key]},
            "window_m": [list(xr), list(yr)], "gt_style": "band"}
    return {"images": images, "meta": meta}


# ------------------------------------------------------------------------------------------------------- figure
# The local paper checkout has generated tables but not outputs/paper or the raw LiDAR scans.
# --from-tables reuses ONLY the four native LiDAR images from an existing teaser PDF.
# All plotted values are read directly from the generated tables, which are never modified here.
METHODS = ["freealign", "coalign", "roco", "alignformer"]
SIGMAS = [0.0, 0.5, 1.0, 1.5, 2.0]
MARKERS = {"bare": "x", "freealign": "s", "coalign": "^", "roco": "D", "alignformer": "o"}


def table_rows(name: str):
    rows = []
    dataset = "opv2v"
    for line in (mpe.TABLES / name).read_text().splitlines():
        if r"\multicolumn" in line and "V2X-Real" in line:
            dataset = "v2xreal"
        if " & " not in line:
            continue
        label, *cells = line.split(" & ")
        values = []
        for cell in cells:
            value = re.search(r"(?<![\w.])(?:\d*\.\d+|\d+)", cell)
            if value is None:
                raise ValueError(f"Missing numeric cell in {name}: {cell}")
            values.append(float(value.group()))
        rows.append((dataset, label, values))
    return rows


def curves_from_tables():
    late = {ds: {} for ds in ["opv2v", "v2xreal"]}
    mapping = {"Reference pose": "perfect", "Ego alone": "ego", "Raw pose": "bare",
               "FreeAlign": "freealign", "CoAlign": "coalign", "RoCo": "roco",
               r"\textbf{\paper}": "alignformer"}
    for _, label, values in table_rows("t3_late_body.tex"):
        if label not in mapping:
            continue
        for ds, vals in zip(late, (values[:5], values[5:])):
            late[ds][mapping[label]] = vals[0] if mapping[label] == "ego" else vals
    hosts = {}
    for ds, label, vals in table_rows("t1_hosts_body.tex"):
        if label != "V2VAM":
            continue
        hosts[ds] = {"perfect": [vals[0]] * 5, "bare": [vals[0]] + vals[5::5],
                     "freealign": vals[1::5], "coalign": vals[2::5], "roco": vals[3::5],
                     "alignformer": vals[4::5]}
    for ds in late:
        assert set(late[ds]) == {"perfect", "ego", "bare", *METHODS}
        assert set(hosts[ds]) == {"perfect", "bare", *METHODS}
        for family in (late[ds], hosts[ds]):
            for key, values in family.items():
                assert len(values) == 5 if isinstance(values, list) else key == "ego"
    return late, hosts


def delay_from_tables():
    rows = {label: values for _, label, values in table_rows("t6_delay_body.tex")}
    delay = {}
    for ds, offset in [("opv2v", 0), ("v2xreal", 8)]:
        delay[ds] = {"bare": rows["Raw pose"][offset:offset + 4],
                     "freealign": rows["FreeAlign"][offset:offset + 4],
                     "alignformer": rows[r"\textbf{+ staleness scores}"][offset:offset + 4]}
    return delay


def reuse_lidar_images(source: Path):
    """Decode the four lossless RGB image streams in a Matplotlib PDF, preserving scene pixels.

    Matplotlib encodes these as Flate streams with PNG row predictors. Rewrapping the
    streams as PNG chunks lets Pillow decode the predictors without a PDF dependency.
    Text, plot coordinates and data values are NOT recovered from the old PDF.
    """
    from PIL import Image
    pdf = source.read_bytes()
    images = []
    # The PDF backend can encode a vertical flip in the drawing matrix instead
    # of the image stream. Honor it so repeated local rebuilds preserve orientation.
    image_heights = {}
    number = rb"[-+]?(?:\d*\.\d+|\d+)"
    matrix = rb"(" + number + rb") (" + number + rb") (" + number + rb") (" + number + rb") "
    matrix += rb"(" + number + rb") (" + number + rb") cm\s*/(I\d+) Do"
    for stream in re.finditer(rb"stream\r?\n", pdf):
        try:
            content = zlib.decompress(pdf[stream.end():pdf.index(b"endstream", stream.end())])
        except (zlib.error, ValueError):
            continue
        for draw in re.finditer(matrix, content):
            image_heights[draw[7]] = float(draw[4])

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)

    pattern = rb"\d+ 0 obj\s*<<[^>]*?/Subtype /Image.*?stream\r?\n"
    for match in re.finditer(pattern, pdf, re.S):
        info = match.group()
        if b"/DeviceRGB" not in info or b"/FlateDecode" not in info or b"/Predictor 10" not in info:
            raise ValueError("Expected native RGB LiDAR images with PNG row predictors")
        width = int(re.search(rb"/Width (\d+)", info)[1])
        height = int(re.search(rb"/Height (\d+)", info)[1])
        length_ref = int(re.search(rb"/Length (\d+) 0 R", info)[1])
        length = int(re.search(rb"(?m)^" + str(length_ref).encode() + rb" 0 obj\s*(\d+)", pdf)[1])
        compressed = pdf[match.end():match.end() + length]
        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
        object_id = re.match(rb"(\d+) 0 obj", info)[1]
        resource = re.search(rb"/(I\d+) " + object_id + rb" 0 R", pdf)
        if resource is None or resource[1] not in image_heights:
            raise ValueError("Cannot establish the native LiDAR image orientation")
        with Image.open(io.BytesIO(png)) as image:
            pixels = np.array(image)
            if image_heights[resource[1]] < 0:
                pixels = pixels[::-1].copy()
            images.append(pixels)
    if len(images) != 4:
        raise ValueError(f"Expected four LiDAR views, found {len(images)}")
    return {"images": dict(zip(METHODS, images)), "meta": {"reused_from": str(source)}}


def render_figure(frame, late, hosts, delay, output: Path, preview: Path | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 7,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5})
    # CVPR \textwidth is 6.875 in; the figure is drawn at that width so fonts print at their nominal size.
    fig = plt.figure(figsize=(6.875, 4.6))

    # (a) four LiDAR views edge to edge, method name inside each view, one row title above.
    fig.text(0.5, 0.995, r"(a) Pose correction on one OPV2V frame at $\sigma=2$ m and 2$^\circ$",
             ha="center", va="top", fontsize=7.6, fontweight="bold")
    # Zoom on the boxes: crop all four views to the same window around every coloured (box) pixel, plus a margin of
    # scene context. Same window in every view, so the views stay directly comparable.
    stack = np.stack([frame["images"][m][..., :3].astype(int) for m in METHODS])
    coloured = (stack.max(-1) - stack.min(-1) > 60).any(0)
    edge = 12   # ignore boxes cut by the window border, e.g. the ego vehicle at the edge of the view
    coloured[:edge], coloured[-edge:], coloured[:, :edge], coloured[:, -edge:] = False, False, False, False
    rows, cols = np.nonzero(coloured)
    margin = 6 * PX_PER_M
    r0, r1 = max(rows.min() - margin, 0), min(rows.max() + margin, coloured.shape[0])
    c0, c1 = max(cols.min() - margin, 0), min(cols.max() + margin, coloured.shape[1])
    views = {m: frame["images"][m][r0:r1, c0:c1] for m in METHODS}
    img_left, img_right, img_gap, img_top = 0.004, 0.996, 0.006, 0.955
    img_w = (img_right - img_left - 3 * img_gap) / 4
    img = views[METHODS[0]]
    img_h = min(img_w * 6.875 / 4.6 * img.shape[0] / img.shape[1], 0.27)
    for j, method in enumerate(METHODS):
        ax = fig.add_axes([img_left + j * (img_w + img_gap), img_top - img_h, img_w, img_h])
        ax.imshow(views[method], interpolation="none", aspect="auto")
        ax.set_axis_off()
        ax.text(0.025, 0.955, LABEL[method], transform=ax.transAxes, ha="left", va="top", fontsize=7.4,
                fontweight="bold", color=HEX[method],
                bbox=dict(boxstyle="round,pad=0.25", facecolor="black", edgecolor="none", alpha=0.75))

    # One legend for the whole figure: a method's colour marks its corrected boxes in (a) and its curve in (b) to (d);
    # red is the raw pose in both, so "before correction" needs no separate entry.
    # Panels drawn by frame_panels() show ground truth as a filled band under the estimates; panels reused from an
    # older PDF still show it as a thin outline drawn on top. The legend key follows whichever style is in the image.
    if frame["meta"].get("gt_style") == "band":
        gt_key = Patch(facecolor=matplotlib.colors.to_rgba(HEX["gt"], 0.25), edgecolor=HEX["gt"], lw=2.0,
                       label="Ground truth")
    else:
        gt_key = Patch(facecolor="none", edgecolor=HEX["gt"], lw=1.1, label="Ground truth")
    handles = [gt_key]
    handles += [Line2D([], [], color=HEX[m], lw=1.1, marker=MARKERS[m], ms=2.8,
                       label="Raw pose" if m == "bare" else LABEL[m])
                for m in ["bare", "freealign", "coalign", "roco", "alignformer"]]
    handles += [Line2D([], [], color=HEX["perfect"], lw=0.9, ls=(0, (4, 2)), label="Noise-free"),
                Patch(facecolor="#f6e9ea", edgecolor=HEX["ego"], lw=0.6, ls=(0, (1, 1.2)),
                      label="Worse than ego alone")]
    legend_y = img_top - img_h - 0.032
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, legend_y), ncol=8, frameon=False,
               fontsize=6.6, handlelength=1.6, handletextpad=0.5, columnspacing=1.1)

    columns = [("(b) Late fusion under pose noise", late), ("(c) V2VAM under pose noise", hosts),
               ("(d) Late fusion under delay", delay)]
    grid = fig.add_gridspec(2, 3, left=0.072, right=0.992, bottom=0.075, top=legend_y - 0.06,
                            wspace=0.17, hspace=0.14)
    for col, (title, family) in enumerate(columns):
        for row, ds in enumerate(["opv2v", "v2xreal"]):
            ax = fig.add_subplot(grid[row, col])
            if row == 0:
                ax.set_title(title, fontsize=7.4, pad=3)
            values = family[ds]
            x = [0, 100, 200, 400] if col == 2 else SIGMAS
            ax.set_ylim(0, 0.95 if ds == "opv2v" else 0.50)
            ax.set_yticks([0, .3, .6, .9] if ds == "opv2v" else [0, .2, .4])
            ax.set_yticklabels(["0", ".3", ".6", ".9"] if ds == "opv2v" else ["0", ".2", ".4"])
            if col != 1:
                ego = late[ds]["ego"]
                ax.axhspan(0, ego, color="#f6e9ea", zorder=0)
                ax.axhline(ego, color=HEX["ego"], lw=.85, ls=(0, (1, 1.2)), zorder=1)
            reference = hosts[ds]["perfect"][0] if col == 1 else late[ds]["perfect"][0]
            ax.axhline(reference, color=HEX["perfect"], lw=.85, ls=(0, (4, 2)), zorder=1)
            methods = ["bare", "freealign", "alignformer"] if col == 2 else ["bare", *METHODS]
            for method in methods:
                ax.plot(x, values[method], color=HEX[method], marker=MARKERS[method], ms=2.6,
                        lw=1.5 if method == "alignformer" else 1.0,
                        zorder=4 if method == "alignformer" else 2, clip_on=False)
            ax.grid(axis="y", color="#e3e6ea", lw=.45)
            ax.tick_params(length=2, pad=1.5, labelsize=6.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if col == 0:
                ax.set_ylabel(("OPV2V" if row == 0 else "V2X-Real") + "\nAP$_{0.7}$", labelpad=2, fontsize=7,
                              linespacing=1.3)
            if col == 2:
                ax.set_xlim(-12, 412)
                ax.set_xticks([0, 100, 200, 400])
                if row == 1:
                    ax.set_xlabel("Delay (ms)", labelpad=1.5, fontsize=7)
            else:
                ax.set_xlim(-0.06, 2.06)
                ax.set_xticks(SIGMAS)
                ax.set_xticklabels(["0", ".5", "1", "1.5", "2"])
                if row == 1:
                    ax.set_xlabel(r"Pose noise $\sigma$ (m and $^\circ$)", labelpad=1.5, fontsize=7)
            if row == 0:
                ax.tick_params(labelbottom=False)
    fig.savefig(output, dpi=300, pad_inches=0, metadata={
        "Title": "RobustAlign: pose and delay robustness",
        "Subject": "Native LiDAR views and AP0.7 from generated late-fusion, V2VAM and delay tables; delay sigma=0",
        "Creator": "scripts/make_f1_teaser.py"})
    if preview:
        fig.savefig(preview, dpi=190, pad_inches=0)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RobustAlign teaser from generated tables and original LiDAR views")
    parser.add_argument("--from-tables", action="store_true",
                        help="Reuse native LiDAR images in --scene-pdf and plot the generated table values")
    parser.add_argument("--scene-pdf", type=Path, default=OUT_PDF)
    parser.add_argument("--output", type=Path, default=OUT_PDF)
    parser.add_argument("--preview", type=Path, help="Optional build-directory preview PNG")
    args = parser.parse_args()
    if args.from_tables:
        frame = reuse_lidar_images(args.scene_pdf)
    else:
        frame = frame_panels()
    # Both paths use the same generated source-of-truth tables and standard noise levels.
    late, hosts = curves_from_tables()
    delay = delay_from_tables()
    render_figure(frame, late, hosts, delay, args.output, args.preview)
    print("Wrote", args.output)
    print("Curve sources: t3_late_body.tex, t1_hosts_body.tex V2VAM rows, t6_delay_body.tex sigma=0")
    print("FreeAlign delay results use the evaluation protocol without its clock-offset search.")


if __name__ == "__main__":
    main()
