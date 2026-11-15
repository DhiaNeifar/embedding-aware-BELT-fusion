#!/usr/bin/env python3
"""Paper Fig. 1: each kind of error alone (position, heading, delay) drops late fusion below the ego vehicle alone.

Data: outputs/paper/audit/intro_error_types.json (scripts/intro_error_types.py) and outputs/paper/delay_opv2v.json
(no correction, sigma 0, the 'bare' row of Table 6). The sketches in each panel are illustrations, not data.
Output: AlignFormer-2027/figures/f1_errors.pdf
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.transforms as mtransforms
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "AlignFormer-2027" / "figures" / "f1_errors.pdf"
REF, EGO = 0.882, 0.674                     # late fusion AP0.7 on OPV2V: accurate pose, ego vehicle alone
RED, BLUE, GREY = "#d1495b", "#2a78d6", "#9a9ca1"

data = json.load(open(ROOT / "outputs/paper/audit/intro_error_types.json"))["results"]
delay = json.load(open(ROOT / "outputs/paper/delay_opv2v.json"))["summary"]
curves = {
    "position": ([0] + [r["level"] for r in data if r["kind"] == "position"],
                 [REF] + [r["ap_70"] for r in data if r["kind"] == "position"]),
    "heading": ([0] + [r["level"] for r in data if r["kind"] == "heading"],
                [REF] + [r["ap_70"] for r in data if r["kind"] == "heading"]),
    "delay": ([0, 100, 200, 400], [delay[d]["0"]["all"]["bare"]["ap_70"]["mean"] for d in ("d0", "d1", "d2", "d4")]),
}
titles = {"position": "Position error", "heading": "Heading error", "delay": "Message delay"}
xlabels = {"position": r"$\sigma$ (m)", "heading": r"$\sigma$ (deg)", "delay": "delay (ms)"}


def box(ax, x, y, yaw=0.0, **kw):
    w, l = 0.9, 2.0
    r = Rectangle((x - l / 2, y - w / 2), l, w, **kw)
    r.set_transform(mtransforms.Affine2D().rotate_deg_around(x, y, yaw) + ax.transData)
    ax.add_patch(r)


def sketch(ax, kind):
    """Ground truth in grey outlines, the cooperating vehicle's detections in red, as each error places them."""
    objects = [(-3.0, 1.6, 0), (0.0, -1.4, 0), (3.2, 1.2, 0), (5.5, -1.0, 0)]
    moving = {0: (2.2, 0.0), 2: (-1.6, 0.0)}           # velocity of the moving objects (for the delay sketch)
    cav = (-6.0, -1.8)
    box(ax, *cav, fill=True, color=BLUE, lw=0)
    for i, (x, y, a) in enumerate(objects):
        box(ax, x, y, a, fill=False, ec=GREY, lw=0.8, ls="--")
        if kind == "position":
            px, py, pa = x + 1.1, y + 0.7, a
        elif kind == "heading":
            th = np.radians(9.0)
            dx, dy = x - cav[0], y - cav[1]
            px, py = cav[0] + np.cos(th) * dx - np.sin(th) * dy, cav[1] + np.sin(th) * dx + np.cos(th) * dy
            pa = a + 9.0
        else:
            vx, vy = moving.get(i, (0.0, 0.0))
            px, py, pa = x - vx, y - vy, a
            if i in moving:
                ax.add_patch(FancyArrowPatch((px, py), (x, y), arrowstyle="-|>", mutation_scale=5, color="k", lw=0.6))
        box(ax, px, py, pa, fill=False, ec=RED, lw=1.0)
    ax.set_xlim(-8.5, 8.0); ax.set_ylim(-4.0, 3.2); ax.set_aspect("equal"); ax.axis("off")


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 7})
    fig = plt.figure(figsize=(6.9, 2.75))
    grid = fig.add_gridspec(2, 3, height_ratios=[0.8, 1.6], hspace=0.08, wspace=0.12)
    plots = []
    for col, kind in enumerate(("position", "heading", "delay")):
        top = fig.add_subplot(grid[0, col])
        sketch(top, kind)
        top.set_title(titles[kind], fontsize=8.5, pad=2)
        ax = fig.add_subplot(grid[1, col], sharey=plots[0] if plots else None)
        plots.append(ax)
        x, y = curves[kind]
        idx = np.arange(len(x))                       # evenly spaced levels
        ax.axhspan(0, EGO, color=RED, alpha=0.07, lw=0)
        ax.axhline(REF, color="k", lw=0.7, ls=(0, (4, 2)))
        ax.axhline(EGO, color=BLUE, lw=0.8, ls=(0, (1, 1.5)))
        ax.plot(idx, y, "-o", color=RED, lw=1.4, ms=3)
        ax.set_xticks(idx)
        ax.set_xticklabels([f"{v:g}" for v in x])
        ax.set_xlabel(xlabels[kind], labelpad=1)
        ax.set_xlim(-0.3, len(x) - 0.7)
        ax.set_ylim(0, 0.95)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2, pad=1.5)
        if col:
            ax.tick_params(labelleft=False)
    plots[0].set_ylabel(r"AP$_{0.7}$, late fusion", labelpad=2)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=GREY, ls="--", lw=0.9, label="true object"),
               Line2D([], [], color=RED, lw=1.1, label="as placed by the receiver"),
               Line2D([], [], color=BLUE, lw=4, label="sender")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=False, fontsize=6.5,
               handlelength=1.6, columnspacing=1.4)
    plots[2].text(1.0, REF + 0.015, "accurate pose", transform=plots[2].get_yaxis_transform(), fontsize=6,
                  va="bottom", ha="right")
    plots[2].text(1.0, EGO + 0.015, "ego vehicle alone", transform=plots[2].get_yaxis_transform(), fontsize=6,
                  va="bottom", ha="right", color=BLUE)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
