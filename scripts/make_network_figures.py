#!/usr/bin/env python3
"""One figure per fusion network, from saved results only (no run).

Intermediate fusion, one figure per network (7 OPV2V, 2 V2X-Real): AP@0.7 against position noise sigma for perfect
pose, ego only, no correction and the four plug-ins. Delay is not studied for intermediate fusion.
Late fusion, one figure per dataset: left, AP@0.7 against sigma (20 draws); right, AP@0.7 against delay alone, sigma = 0
(one draw; CoAlign and RoCo were not run under delay).

Output: outputs/figures/<name>.pdf and .png, and outputs/figures/all_networks.pdf with every figure on one page.
A missing result (an ego-only run still going) leaves that line out; rerun to add it.

    python scripts/make_network_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_paper_exhibits as ex  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
EGO = ROOT / "outputs" / "paper" / "hosts_ego"
X = [float(s) for s in ex.SIGMAS]

STYLE = dict(ex.STYLE)
STYLE["ego"] = ("Ego only", "#5b6470", (0, (1, 1.5)))
ORDER = ["perfect", "ego", "bare", "freealign", "coalign", "roco", "alignformer"]


def _line(ax, method, x, y, marker=True):
    label, color, dash = STYLE[method]
    ax.plot(x, y, color=color, linestyle=dash, label=label, linewidth=1.5 if method == "alignformer" else 1.0,
            marker="o" if marker and method in ex.PLUGINS else None, markersize=2.4,
            zorder=3 if method == "alignformer" else 2)


def _style(ax, xlabel):
    ex._axes_style(ax)
    ax.set_xlabel(xlabel, fontsize=7.5, color="#3b4552", labelpad=1)


def ego_ap(dataset, net):
    path = EGO / f"{dataset}_{net}_ego.json"
    return json.loads(path.read_text())["results"]["ego"]["ap_70"] if path.exists() else None


def intermediate_figure(plt, dataset, net, label):
    rows = [ex.host_ap(dataset, net, s) for s in ex.SIGMAS]
    fig, ax = plt.subplots(figsize=(4.4, 2.2))
    for method in ORDER:
        if method == "ego":
            ego = ego_ap(dataset, net)
            if ego is not None:
                _line(ax, "ego", X, [ego] * len(X), marker=False)
            continue
        _line(ax, method, X, [r[method] for r in rows])
    ax.set_title(f"{label}, {ex.DATASET_NAME[dataset]}, intermediate fusion", fontsize=8, color="#17202B", pad=3)
    ax.set_ylabel(r"AP$_{0.7}$", fontsize=7.5, color="#3b4552", labelpad=1)
    ax.set_xticks([0, 0.4, 1, 1.5, 2])
    _style(ax, r"position noise $\sigma$ (m, deg)")
    ax.legend(fontsize=6.2, frameon=False, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5), handlelength=1.8)
    fig.tight_layout(pad=0.3)
    return fig


def late_figure(plt, dataset):
    rows = ex.late_rows(dataset, "ap_70")
    ego = json.loads((ex.OUT / f"late_{dataset}_ego.json").read_text())["ego"]["all"]["ap_70"]
    delay = ex.delay_data(dataset)
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.6, 2.3))
    for method in ORDER:
        if method == "ego":
            _line(left, "ego", X, [ego] * len(X), marker=False)
        else:
            _line(left, method, X, [rows[s][method] for s in ex.SIGMAS])
    left.set_xticks([0, 0.4, 1, 1.5, 2])
    left.set_ylabel(r"AP$_{0.7}$", fontsize=7.5, color="#3b4552", labelpad=1)
    left.set_title("position noise", fontsize=8, color="#17202B", pad=3)
    _style(left, r"position noise $\sigma$ (m, deg)")
    ms = [0, 100, 200, 400]
    keys = ["d0", "d1", "d2", "d4"]
    # delay alone: no position noise (sigma = 0), so no correction equals perfect pose with the delay left in
    _line(right, "ego", ms, [ego] * len(ms), marker=False)
    _line(right, "bare", ms, [delay[(k, "0")]["bare"] for k in keys])
    _line(right, "freealign", ms, [delay[(k, "0")]["freealign"] for k in keys])
    _line(right, "alignformer", ms, [delay[(k, "0")]["latency_scores"] for k in keys])
    right.set_xticks(ms)
    right.set_title("delay", fontsize=8, color="#17202B", pad=3)
    _style(right, "delay (ms)")
    handles, labels = left.get_legend_handles_labels()
    h2, l2 = right.get_legend_handles_labels()
    for h, l in zip(h2, l2):
        if l not in labels:
            handles.append(h), labels.append(l)
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=6.2, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.suptitle(f"{ex.DATASET_NAME[dataset]}, late fusion", fontsize=8.5, color="#17202B", x=0.5, y=1.2)
    fig.tight_layout(pad=0.3)
    return fig


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    OUT.mkdir(parents=True, exist_ok=True)
    figures = []
    for dataset in ex.NETWORKS:
        figures.append((f"late_{dataset}", late_figure(plt, dataset)))
    for dataset, nets in ex.NETWORKS.items():
        for net, label in nets:
            figures.append((f"intermediate_{dataset}_{net}", intermediate_figure(plt, dataset, net, label)))
    with PdfPages(OUT / "all_networks.pdf") as pdf:
        for name, fig in figures:
            fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
            fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)
    print(f"wrote {len(figures)} figures to {OUT} and {OUT / 'all_networks.pdf'}")


if __name__ == "__main__":
    main()
