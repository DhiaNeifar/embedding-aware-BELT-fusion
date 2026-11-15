"""Paper exhibits from the result files: T1, T3, T6 (table bodies) and F3, F5 (PDF figures).

Reads only outputs/paper/ (no run). Writes the table bodies to AlignFormer-2027/tables/gen/ (the hand-written tables/t*.tex
\\input them, so captions and notes stay in the paper) and the figures to AlignFormer-2027/figures/.

    T1  intermediate fusion: outputs/paper/hosts{,_v2xreal}/ (draw 0) merged with hosts_5draws/, hosts_v2xreal_5draws/
        (draws 1-4), mean over the five draws
    T3  late fusion: outputs/paper/late_<ds>_plugins_20seeds.json (seeds 0-19); "one frame" = alignformer_c, same seeds,
        from outputs/paper/late_<ds>.json
    T6  delay, late fusion: outputs/paper/delay_<ds>.json, draws 0-4 at sigma 1 (the full method over time, as in T3)
    F3  AP@0.7 against sigma, AttFuse on each dataset (the one network run on both)
    F5  AP@0.7 against delay at sigma = 1, both datasets
    F6  late fusion, AP@0.7 against sigma, both datasets (T3 as curves)
    T7  ablation: outputs/paper/ablation_<ds>.json (sigma 1) and ablation_<ds>_s0s2.json (sigma 0, 2). The 200 ms delay
        block was removed on 2026-11-12: it repeated three cells of T6.
    T3 is written as two bodies: t3_late_body.tex (AP@0.7) and ts_late_ap50_body.tex (AP@0.5, supplement).
    T5, F8  noise kinds: outputs/paper/noise_<ds>_<kind>.json (drift, bias, mixed at sigma 1; beyond = sigma 5),
            benchmark row = late_<ds>.json at sigma 1, same seeds 0-4
    T4, F7  pose accuracy at sigma = 1: outputs/paper/pose_accuracy_<ds>.json (scripts/pose_accuracy.py)

Usage: python scripts/make_paper_exhibits.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "paper"
PAPER = ROOT / "AlignFormer-2027"          # the paper (was CVPR/, the Nov 9 draft)
TABLES = PAPER / "tables" / "gen"
FIGURES = PAPER / "figures"

SIGMAS = ["0", "0.2", "0.4", "0.6", "0.8", "1", "1.5", "2"]
STD_SIGMAS = ["0", "0.5", "1", "1.5", "2"]   # the paper's sigma standard (authors, 2026-11-12 night), every table
TABLE_SIGMAS = STD_SIGMAS
T1_SIGMAS = STD_SIGMAS
TAG = "s05s15"   # files written by scripts/run_paper.sh sigma-std (0.5 and 1.5 where missing); merged below
PEND = r"\pending"   # a cell whose five draws are not all on disk yet


def merged(*names):
    """Merge result files of one kind: per_seed and summary are dicts keyed by sigma; later files add sigmas."""
    out = None
    for name in names:
        path = OUT / name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        if out is None:
            out = data
            continue
        for key in ("per_seed", "summary"):
            if key in data:
                out.setdefault(key, {}).update(data[key])
    return out

PLUGINS = ["freealign", "coalign", "roco", "alignformer"]
NETWORKS = {
    "opv2v": [("attfuse", "AttFuse"), ("coalign", "CoAlign fusion"), ("cobevt", "CoBEVT"), ("fcooper", "F-Cooper"),
              ("v2vam", "V2VAM"), ("v2xvit", "V2X-ViT"), ("where2comm", "Where2comm")],
    "v2xreal": [("attfuse", "AttFuse"), ("fcooper", "F-Cooper"), ("v2vam", "V2VAM"), ("v2xvit", "V2X-ViT"),
                ("where2comm", "Where2comm")],   # all five run 2026-11-09 (scripts/run_host_v2xreal.py)
}
N_DRAWS = 5   # every noisy result in the paper is the mean over the first five paired noise draws (seeds 0-4)
HOST_DIR_5 = {"opv2v": OUT / "hosts_5draws", "v2xreal": OUT / "hosts_v2xreal_5draws"}   # revision-5draws
HOST_DIR = {"opv2v": OUT / "hosts", "v2xreal": OUT / "hosts_v2xreal"}                    # seed 0 only (fallback)
DATASET_NAME = {"opv2v": "OPV2V", "v2xreal": "V2X-Real"}
DELAYS = ["d0", "d1", "d2", "d4"]          # 0, 100, 200, 400 ms
RAW = "Raw pose"   # the reported pose used as is, with no corrector (was "No correction"); short form "Raw"

# the same colours as the results page; perfect pose dashed, no correction grey
STYLE = {
    "perfect": ("Reference pose", "#17202B", (0, (4, 2))),
    "bare": (RAW, "#9a9ca1", "-"),
    "freealign": ("FreeAlign", "#eb6834", "-"),
    "coalign": ("CoAlign", "#1baf7a", "-"),
    "roco": ("RoCo", "#eda100", "-"),
    "alignformer": ("RobustAlign", "#2a78d6", "-"),
}


def fmt(x: float) -> str:
    return f"{x:.3f}".lstrip("0") if x < 1 else f"{x:.3f}"


def bold_best(values: dict, keys: list) -> dict:
    """Format values; the best of ``keys`` in bold (ties all bold)."""
    best = max(values[k] for k in keys)
    return {k: (r"\best{" + fmt(v) + "}" if k in keys and round(v, 3) == round(best, 3) else fmt(v))
            for k, v in values.items()}


# ---------------------------------------------------------------------------------------------------------------- T1
def host_ap(dataset: str, net: str, sigma: str, metric: str = "ap_70", strict: bool = True) -> dict:
    # draw 0 from the seed-0 run, draws 1..N_DRAWS-1 from revision-5draws (run with --first-seed 1); same keys.
    # strict: a method enters only with all N_DRAWS draws (tables); otherwise whatever draws exist (unused figures).
    results = {}
    for folder in (HOST_DIR[dataset], HOST_DIR_5[dataset]):
        path = folder / f"{dataset}_{net}_sigma_{sigma}.json"
        if path.exists():
            results.update(json.loads(path.read_text())["results"])
    if "true" not in results:
        return {}
    row = {"perfect": results["true"][metric]}
    for method in ["bare"] + PLUGINS:
        values = [results[k][metric] for k in (f"{method}/sigma_{float(sigma):g}/seed_{d}" for d in range(N_DRAWS))
                  if k in results]
        if values and (len(values) == N_DRAWS or not strict or sigma == "0"):
            row[method] = float(np.mean(values))
    if sigma == "0":
        row["bare"] = row["perfect"]
    return row


def table_t1() -> str:
    lines = []
    for dataset, nets in NETWORKS.items():
        lines.append(rf"\multicolumn{{{1 + 5 * len(T1_SIGMAS)}}}{{@{{}}l}}{{\textit{{{DATASET_NAME[dataset]}}}}} \\")
        for net, label in nets:
            cells = []
            for sigma in T1_SIGMAS:
                row = host_ap(dataset, net, sigma)
                present = [m for m in PLUGINS if m in row]
                text = bold_best(row, present) if len(present) == len(PLUGINS) else \
                    {m: (fmt(v) if m in ("perfect", "bare") else PEND) for m, v in row.items()}
                first = ["perfect"] if sigma == "0" else ["bare"]
                cells += [text.get(m, PEND) for m in first + PLUGINS]
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        if dataset == "opv2v":
            lines.append(r"\midrule")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------- T3
def late_rows(dataset: str, metric: str) -> dict:
    plugins = merged(f"late_{dataset}_plugins_20seeds.json", f"late_{dataset}_plugins_{TAG}.json")
    variants = merged(f"late_{dataset}.json", f"ablation_{dataset}_{TAG}.json")
    rows = {}
    for sigma in sorted(set(SIGMAS) | set(STD_SIGMAS), key=float):
        if sigma not in plugins["per_seed"] or sigma not in variants["per_seed"]:
            continue
        seeds = sorted(entry["seed"] for entry in plugins["per_seed"][sigma] if entry["seed"] < N_DRAWS)
        if sigma != "0" and len(seeds) < N_DRAWS:
            continue
        row = {m: float(np.mean([e["ap"][m]["all"][metric] for e in plugins["per_seed"][sigma] if e["seed"] in seeds]))
               for m in ["perfect", "bare"] + PLUGINS}
        one_frame = [e["ap"]["alignformer_c"]["all"][metric] for e in variants["per_seed"][sigma] if e["seed"] in seeds]
        assert len(one_frame) == len(seeds), (dataset, sigma)
        row["one_frame"] = float(np.mean(one_frame))
        # paper and code agree: the plug-in file's AlignFormer is alignformer_tk of the variants file, same seeds
        tk = np.mean([e["ap"]["alignformer_tk"]["all"][metric] for e in variants["per_seed"][sigma] if e["seed"] in seeds])
        if abs(tk - row["alignformer"]) > 1e-6:   # was an assert; a mismatch is reported, never silently used
            print(f"WARNING {dataset} sigma {sigma}: alignformer_tk {tk:.4f} != plug-in file {row['alignformer']:.4f}")
        row["n"] = len(seeds)
        rows[sigma] = row
    return rows


def table_t3(metrics=(("ap_70", "AP$_{0.7}$"),)) -> str:
    order = [("perfect", "Reference pose"), ("ego", "Ego alone"), ("bare", RAW), ("freealign", "FreeAlign"),
             ("coalign", "CoAlign"), ("roco", "RoCo"), ("rule", None), ("one_frame", r"\textbf{\core}, one frame"),
             ("alignformer", r"\textbf{\paper}")]   # \core = name of the per-frame estimator (preamble.tex)
    compared = PLUGINS + ["one_frame"]
    lines = []
    for metric, title in metrics:
        data = {ds: late_rows(ds, metric) for ds in NETWORKS}
        if len(metrics) > 1:
            lines.append(rf"\multicolumn{{9}}{{@{{}}l}}{{\textit{{{title}}}}} \\")
        text = {(ds, s): bold_best({k: v for k, v in data[ds][s].items() if k != "n"}, compared)
                for ds in data for s in TABLE_SIGMAS if s in data[ds]}
        # underline the best single-frame corrector (FreeAlign, CoAlign, RoCo, PRA one frame) in each column
        single = ["freealign", "coalign", "roco", "one_frame"]
        for (ds, s_), cells in text.items():
            vals = {k: round(data[ds][s_][k], 3) for k in single}
            top = max(vals.values())
            for k in single:
                if vals[k] == top:
                    cells[k] = r"\underline{" + cells[k] + "}"
        for key, label in order:
            if key == "rule":
                lines.append(r"\midrule")
                continue
            if key == "ego":   # one value per dataset (no CAV: no sigma, no seed); pending until it is run
                cells = []
                for ds in data:
                    ego = OUT / f"late_{ds}_ego.json"
                    value = fmt(json.loads(ego.read_text())["ego"]["all"][metric]) if ego.exists() else r"\pending"
                    cells += [value] * len(TABLE_SIGMAS)
            else:
                cells = [text[(ds, s)][key] if (ds, s) in text else PEND for ds in data for s in TABLE_SIGMAS]
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        if len(metrics) > 1 and metric == "ap_70":
            lines.append(r"\midrule")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------- T6
def delay_data(dataset: str) -> dict:
    data = json.loads((OUT / f"delay_{dataset}.json").read_text())
    rows = {}
    for d in DELAYS:
        for sigma in ("0", "1"):
            row = {m: v["ap_70"]["mean"] for m, v in data["summary"][d][sigma]["all"].items()}
            row["latency_mae"] = float(np.mean([e["latency"]["mae_ms"] for e in data["per_seed"][d][sigma]]))
            row["n"] = len(data["per_seed"][d][sigma])
            rows[(d, sigma)] = row
    return rows


def table_t6() -> str:
    order = [("bare", RAW), ("freealign", "FreeAlign"), ("ignored", r"\paper, delay ignored"),
             ("latency", r"+ delay estimate"), ("latency_scores", r"\textbf{+ staleness scores}")]
    compared = ["freealign", "ignored", "latency", "latency_scores"]
    data = {ds: delay_data(ds) for ds in NETWORKS}
    cols = [(ds, sigma, d) for ds in NETWORKS for sigma in ("0", "1") for d in DELAYS]
    text = {c: bold_best({k: data[c[0]][(c[2], c[1])][k] for k in [m for m, _ in order]}, compared) for c in cols}
    lines = []
    for key, label in order:
        lines.append(f"{label} & " + " & ".join(text[c][key] for c in cols) + r" \\")
    told = [data[c[0]][(c[2], c[1])].get("told_scores") for c in cols]
    if all(v is not None for v in told):
        lines.append(r"\midrule")
        lines.append(r"\paper, delay given (oracle) & " + " & ".join(fmt(v) for v in told) + r" \\")
    lines.append(r"\midrule")
    lines.append("Delay error in ms & " + " & ".join(f"{data[c[0]][(c[2], c[1])]['latency_mae']:.0f}" for c in cols)
                 + r" \\")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------- supplement: pooled baselines
CORE = r"\core"   # macro for the per-frame estimator (Propose, Refine, Validate), defined in preamble.tex
POOLED_ROWS = [("freealign", "freealign_tk", "FreeAlign"), ("coalign_frame", "coalign_tk", "CoAlign"),
               ("roco_frame", "roco_tk", "RoCo"), ("alignformer_c", "alignformer_tk", CORE)]


def table_pooled() -> str:
    """Assessment EXP-003: each plug-in alone and through AlignFormer's pooling, outputs/paper/pooled_<ds>.json
    (scripts/alignformer_variants_sweep.py, seeds 0-4). AP@0.7, one frame / pooled."""
    data = {ds: merged(f"pooled_{ds}.json", f"pooled_{ds}_{TAG}.json")["summary"] for ds in NETWORKS}
    sigmas = STD_SIGMAS
    ap = lambda ds, s, row: data[ds][s]["all"][row]["ap_70"]["mean"]
    lines = []
    keys = [k for f, p_, _ in POOLED_ROWS for k in (f, p_)]
    best = {(ds, s): max(round(ap(ds, s, k), 3) for k in keys) for ds in NETWORKS for s in sigmas if s in data[ds]}
    cell = lambda ds, s, k: PEND if s not in data[ds] else (r"\best{" + fmt(ap(ds, s, k)) + "}") \
        if round(ap(ds, s, k), 3) == best[(ds, s)] else fmt(ap(ds, s, k))
    for frame_key, pooled_key, label in POOLED_ROWS:
        for key, suffix in ((frame_key, ", one frame"), (pooled_key, ", pooled")):
            lines.append(f"{label}{suffix} & " + " & ".join(cell(ds, s, key) for ds in NETWORKS for s in sigmas)
                         + r" \\")
        if label != CORE:
            lines.append(r"\midrule")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- supplement: audits E2, E5 + E0, E3
def _audit(name):
    base, extra = OUT / "audit" / f"{name}.json", OUT / "audit" / f"{name}_{TAG}.json"
    data = json.loads(base.read_text()) if base.exists() else {}
    if extra.exists():
        new = json.loads(extra.read_text())
        for k, v in new.items():
            if isinstance(v, dict) and isinstance(data.get(k), dict):
                for k2, v2 in v.items():
                    if isinstance(v2, dict) and isinstance(data[k].get(k2), dict) and k in ("branches",):
                        data[k][k2].update(v2)
                    else:
                        data[k][k2] = v2 if k2 not in data[k] or k not in ("branches",) else data[k][k2]
            elif k not in data:
                data[k] = v
    return data


def _bold_rows(rows, lower=False):
    """rows: list of (label, [value or None]); bold the best value of each column (ties included)."""
    ncol = max(len(v) for _, v in rows)
    best = []
    for j in range(ncol):
        col = [round(v[j], 3) for _, v in rows if j < len(v) and v[j] not in (None, "NA")]
        best.append((min if lower else max)(col) if len(col) > 1 else None)
    out = []
    for label, values in rows:
        cells = ["" if v == "NA" else PEND if v is None else
                 (r"\best{" + fmt(v) + "}" if best[j] is not None and round(v, 3) == best[j] else fmt(v))
                 for j, v in enumerate(values)]
        out.append(f"{label} & " + " & ".join(cells) + r" \\")
    return out


def table_e2() -> str:
    e2 = {}
    for name in ("e2_summary.json", f"e2_summary_{TAG}.json"):
        path = OUT / "audit" / name
        if path.exists():
            for split, by_sigma in json.loads(path.read_text()).items():
                e2.setdefault(split, {}).update(by_sigma)
    get = lambda split, s, row: (e2.get(split, {}).get(s, {}).get("rows", {}).get(row, {}).get("ap70_all"))
    cols = lambda row, split=None: [get(split or ds_split, s, row) for ds_split in ("opv2v_test", "v2xreal_test")
                                    for s in STD_SIGMAS] if split is None else \
        ["NA"] * len(STD_SIGMAS) + [get(split, s, row) for s in STD_SIGMAS]
    lines = [rf"\multicolumn{{{1 + 2 * len(STD_SIGMAS)}}}{{@{{}}l}}{{\textit{{One frame}}}} \\"]
    lines += _bold_rows([("Hard Propose", cols("propose_hard")), ("Refine from reported", cols("verify_from_reported")),
                         ("Hungarian + RANSAC", cols("hungarian_ransac")), (r"\core, one frame", cols("alignformer_c"))])
    lines += [r"\midrule", rf"\multicolumn{{{1 + 2 * len(STD_SIGMAS)}}}{{@{{}}l}}{{\textit{{Pooled over time}}}} \\"]
    lines += _bold_rows([("Refine from reported", cols("verify_from_reported_tk")),
                         ("Hungarian + RANSAC", cols("hungarian_ransac_tk")), (r"\paper", cols("alignformer_tk")),
                         (r"\paper, OPV2V ckpt.", cols("alignformer_tk", "v2xreal_test_opv2v_checkpoint"))])
    return "\n".join(lines) + "\n"


def table_decision() -> tuple:
    e0 = _audit("e0_bootstrap").get("branches", {})
    names = [("engaged_agree", "Engaged, fits agree"), ("engaged_consensus", r"Engaged, $\kappa\ge\kappa_{\min}$"),
             ("engaged_rejected", "Engaged, rejected"), ("not_engaged_wald_pass", "Not engaged, $W$ passes"),
             ("not_engaged_wald_fail", "Not engaged, $W$ fails")]
    top = []
    for key, label in names:
        vals = [e0.get(ds, {}).get(s, {}).get(key) for ds in NETWORKS for s in STD_SIGMAS]
        top.append(f"{label} & " + " & ".join(PEND if v is None else fmt(v) for v in vals) + r" \\")
    e5 = {}
    for name in ("e5_decision.json", f"e5_decision_{TAG}.json"):
        path = OUT / "audit" / name
        if path.exists():
            e5.update(json.loads(path.read_text()))
    harm = lambda rule, s: e5.get(s, {}).get("rules", {}).get(rule, {}).get("all", {}).get("harmful")
    bottom = _bold_rows([(label, [harm(rule, s) for s in STD_SIGMAS]) for rule, label in
                         (("none", "No Validate"), ("agreement", "Agreement only"), ("wald_only", "$W$ only"),
                          ("full", "Full one-frame Validate"), ("pooled", r"\paper, pooled"))], lower=True)
    return "\n".join(top) + "\n", "\n".join(bottom) + "\n"


def table_odometry() -> str:
    e3 = {}
    for name in ("e3_summary.json", f"e3_summary_{TAG}.json"):
        path = OUT / "audit" / name
        if path.exists():
            for ds, configs in json.loads(path.read_text()).items():
                for cfg, rows in configs.items():
                    e3.setdefault(ds, {}).setdefault(cfg, {}).update(rows)
    val = lambda ds, cfg, s, key="ap70_pooled": e3.get(ds, {}).get(cfg, {}).get(s, {}).get(key)
    blocks = {"opv2v": [("one frame, no odometry", "exact", "ap70_one_frame"), ("exact simulator motion", "exact", None),
                        ("exact + white, LOAM level", "exact+white", None),
                        ("exact + AR(1), coefficient 0.95", "exact+ar1", None),
                        (r"exact + 1\% scale bias", "exact+scale", None),
                        (r"exact + 0.01$^\circ$/m yaw drift", "exact+yawdrift", None), ("KISS-ICP", "kissicp", None)],
              "v2xreal": [("one frame, no odometry", "kissicp", "ap70_one_frame"), ("KISS-ICP", "kissicp", None),
                          ("recorded-pose motion", "recorded", None), ("KISS-ICP + AR(1)", "kissicp+ar1", None),
                          (r"KISS-ICP + 1\% scale bias", "kissicp+scale", None),
                          ("KISS-ICP + yaw drift", "kissicp+yawdrift", None)]}
    lines = []
    for ds, rows in blocks.items():
        body = _bold_rows([(label, [val(ds, cfg, s, key or "ap70_pooled") for s in STD_SIGMAS])
                           for label, cfg, key in rows])
        body = [(DATASET_NAME[ds] if i == 0 else "") + " & " + line for i, line in enumerate(body)]
        lines += body + ([r"\midrule"] if ds == "opv2v" else [])
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------------------ figures
def _axes_style(ax):
    ax.grid(axis="y", color="#e4e8ee", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9ba7b6")
    ax.tick_params(colors="#3b4552", labelsize=7, length=2)


def figure_f3(plt) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.55), sharey=False)
    x = [float(s) for s in SIGMAS]
    for ax, dataset in zip(axes, NETWORKS):
        rows = [host_ap(dataset, "attfuse", s, strict=False) for s in SIGMAS]
        for method in ["perfect", "bare"] + PLUGINS:
            label, color, dash = STYLE[method]
            ax.plot(x, [r[method] for r in rows], color=color, linestyle=dash, linewidth=1.4 if method == "alignformer"
                    else 1.0, marker="o" if method in PLUGINS else None, markersize=2.2, label=label,
                    zorder=3 if method == "alignformer" else 2)
        ax.set_title(f"AttFuse, {DATASET_NAME[dataset]}", fontsize=7.5, color="#17202B", pad=3)
        ax.set_xlabel(r"$\sigma$ (m, deg)", fontsize=7, color="#3b4552", labelpad=1)
        ax.set_xticks([0, 0.4, 1, 1.5, 2])
        ax.set_xticklabels(["0", ".4", "1", "1.5", "2"])
        _axes_style(ax)
    axes[0].set_ylabel(r"AP$_{0.7}$", fontsize=7, color="#3b4552", labelpad=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=6, frameon=False, bbox_to_anchor=(0.5, 1.13),
               handlelength=1.6, columnspacing=1.0)
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.2, top=0.82, wspace=0.3)
    fig.savefig(FIGURES / "f3_curves.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def figure_f5(plt) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.55))
    x = [0, 100, 200, 400]
    lines = [("bare", RAW, "#9a9ca1", "-"), ("freealign", "FreeAlign", "#eb6834", "-"),
             ("ignored", "RobustAlign, delay ignored", "#86b6ef", "-"),
             ("latency_scores", "RobustAlign", "#2a78d6", "-")]
    for ax, dataset in zip(axes, NETWORKS):
        data = delay_data(dataset)
        for key, label, color, dash in lines:
            ax.plot(x, [data[(d, "1")][key] for d in DELAYS], color=color, linestyle=dash, label=label,
                    linewidth=1.4 if key == "latency_scores" else 1.0, marker="o",
                    markersize=2.2, zorder=3 if key == "latency_scores" else 2)
        ax.set_title(f"{DATASET_NAME[dataset]}, " + r"$\sigma=1$", fontsize=7.5, color="#17202B", pad=3)
        ax.set_xlabel("delay (ms)", fontsize=7, color="#3b4552", labelpad=1)
        ax.set_xticks(x)
        _axes_style(ax)
    axes[0].set_ylabel(r"AP$_{0.7}$", fontsize=7, color="#3b4552", labelpad=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=6, frameon=False, bbox_to_anchor=(0.5, 1.13),
               handlelength=1.6, columnspacing=1.0)
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.2, top=0.82, wspace=0.3)
    fig.savefig(FIGURES / "f5_latency.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def figure_f6(plt) -> None:
    """T3 as curves: late fusion, AP@0.7 against sigma, all 8 levels, both datasets (same seeds as T3)."""
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.55))
    x = [float(s) for s in SIGMAS]
    for ax, dataset in zip(axes, NETWORKS):
        rows = late_rows(dataset, "ap_70")
        for method in ["perfect", "bare"] + PLUGINS:
            label, color, dash = STYLE[method]
            ax.plot(x, [rows[s][method] for s in SIGMAS], color=color, linestyle=dash,
                    linewidth=1.4 if method == "alignformer" else 1.0, marker="o" if method in PLUGINS else None,
                    markersize=2.2, label=label, zorder=3 if method == "alignformer" else 2)
        ax.set_title(f"Late fusion, {DATASET_NAME[dataset]}", fontsize=7.5, color="#17202B", pad=3)
        ax.set_xlabel(r"$\sigma$ (m, deg)", fontsize=7, color="#3b4552", labelpad=1)
        ax.set_xticks([0, 0.4, 1, 1.5, 2])
        ax.set_xticklabels(["0", ".4", "1", "1.5", "2"])
        _axes_style(ax)
    axes[0].set_ylabel(r"AP$_{0.7}$", fontsize=7, color="#3b4552", labelpad=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=6, frameon=False, bbox_to_anchor=(0.5, 1.13),
               handlelength=1.6, columnspacing=1.0)
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.2, top=0.82, wspace=0.3)
    fig.savefig(FIGURES / "f6_late.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------ T4, F7
def pose_data(dataset: str):
    path = OUT / f"pose_accuracy_{dataset}.json"
    return json.loads(path.read_text())["summary"] if path.exists() else None


def table_t4() -> str:
    order = [("bare", RAW), ("freealign", "FreeAlign"), ("coalign", "CoAlign"), ("roco", "RoCo"),
             ("alignformer", r"\textbf{\paper}")]
    lines = []
    for dataset in NETWORKS:
        data = pose_data(dataset)
        for i, (key, label) in enumerate(order):
            head = DATASET_NAME[dataset] if i == 0 else ""
            if data is None:
                cells = [r"\pending"] * 5
            else:
                row = data[key]
                best_t = min(data[m]["median_m"] for m in PLUGINS)
                best_p = min(data[m]["p90_m"] for m in PLUGINS)
                best_y = min(data[m]["median_deg"] for m in PLUGINS)
                mark = lambda v, best: (r"\best{%.2f}" % v) if key in PLUGINS and round(v, 2) == round(best, 2) else "%.2f" % v
                answered = "--" if key == "bare" else f"{100 * row['answered']:.0f}"
                sparse = "--" if key == "bare" else f"{100 * row['sparse_answered']:.0f}"
                cells = [mark(row["median_m"], best_t), mark(row["p90_m"], best_p), mark(row["median_deg"], best_y),
                         answered, sparse]
            lines.append(f"{head} & {label} & " + " & ".join(cells) + r" \\")
        if dataset == "opv2v":
            lines.append(r"\midrule")
    return "\n".join(lines) + "\n"


def figure_f7(plt) -> None:
    """Share of pairs with translation error below x, every plug-in, sigma = 1, both datasets."""
    if any(pose_data(ds) is None for ds in NETWORKS):
        return
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.55))
    share = np.arange(101) / 100.0
    for ax, dataset in zip(axes, NETWORKS):
        data = pose_data(dataset)
        for method in ["bare"] + PLUGINS:
            label, color, dash = STYLE[method]
            ax.plot(data[method]["percentiles_m"], share, color=color, linestyle=dash, label=label,
                    linewidth=1.4 if method == "alignformer" else 1.0, zorder=3 if method == "alignformer" else 2)
        ax.set_xscale("log")
        ax.set_xlim(0.02, 5)
        ax.set_title(f"{DATASET_NAME[dataset]}, " + r"$\sigma=1$", fontsize=7.5, color="#17202B", pad=3)
        ax.set_xlabel("translation error (m)", fontsize=7, color="#3b4552", labelpad=1)
        _axes_style(ax)
    axes[0].set_ylabel("share of pairs", fontsize=7, color="#3b4552", labelpad=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=6, frameon=False, bbox_to_anchor=(0.5, 1.13),
               handlelength=1.6, columnspacing=1.0)
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.2, top=0.82, wspace=0.3)
    fig.savefig(FIGURES / "f7_pose.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------- T7
def table_t7() -> str:
    """Rows: one stage added at a time. Columns: sigma 0 / 1 / 2 per dataset (AP@0.7). sigma 1 from
    ablation_<ds>.json, sigma 0 and 2 from ablation_<ds>_s0s2.json; the delay stages at 200 ms from T6 (sigma 1)."""
    stages = [("propose_only", "Propose"), ("propose_verify", "+ Refine"), ("alignformer_c", "+ Validate"),
              ("alignformer_pool", "+ pooling"), ("alignformer_tk", r"+ window Validate")]
    delay = [("ignored", r"\paper, delay ignored"), ("latency", "+ delay estimate"),
             ("latency_scores", "+ staleness scores")]
    sigmas = STD_SIGMAS
    summary = {}
    for ds in NETWORKS:
        summary[ds] = {}
        for name in (f"ablation_{ds}.json", f"ablation_{ds}_s0s2.json", f"ablation_{ds}_{TAG}.json"):
            path = OUT / name
            if path.exists():   # mean over the first N_DRAWS noise draws, as every other table
                for s, entries in json.loads(path.read_text())["per_seed"].items():
                    kept = [e for e in entries if e["seed"] < N_DRAWS]
                    if s != "0" and len(kept) < N_DRAWS:
                        continue
                    summary[ds][s] = {k: float(np.mean([e["ap"][k]["all"]["ap_70"] for e in kept]))
                                      for k in kept[0]["ap"]}
    # harmful corrections, OPV2V sigma 1: outputs/paper/audit/e5_decision.json (rules none / full / pooled, all pairs)
    rules = json.loads((OUT / "audit" / "e5_decision.json").read_text())["1"]["rules"]
    harmful = {"propose_verify": rules["none"]["all"]["harmful"], "alignformer_c": rules["full"]["all"]["harmful"],
               "alignformer_tk": rules["pooled"]["all"]["harmful"]}
    lines = []
    for key, label in stages:
        cells = []
        for ds in NETWORKS:
            for s in sigmas:
                row = summary[ds].get(s, {})
                cells.append(fmt(row[key]) if key in row else r"\pending")
        cells.append(f"{100 * harmful[key]:.1f}" if key in harmful else "--")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def table_cost() -> str:
    """Cost block, moved from T7 to the supplement for length (2026-11-12). Columns span both datasets."""
    # cost: scripts/measure_cost.py, one CPU core, ms per frame (frames with at least one CAV), sigma 1
    cost = {ds: json.loads((OUT / f"cost_{ds}.json").read_text()) if (OUT / f"cost_{ds}.json").exists() else None
            for ds in NETWORKS}
    lines = []
    rows = [("Propose", lambda c: c["ms_per_frame"]["propose"]),
            ("Refine + Validate", lambda c: c["ms_per_frame"]["verify_abstain"]),
            ("Pooling + window Validate", lambda c: c["ms_per_frame"]["pooling"]),
            (r"\paper, total", lambda c: c["ms_per_frame"]["propose"] + c["ms_per_frame"]["full_method"]),
            ("FreeAlign, reference", lambda c: c["ms_per_frame"]["freealign"])]
    for label, get in rows:
        cells = []
        for ds in NETWORKS:
            cells.append(f"{get(cost[ds]):.1f}" if cost[ds] else r"\pending")   # one plain column per dataset
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    cells = [(f"{cost[ds]['bytes_per_cav_message']:.0f} B, {cost[ds]['detections_per_cav_message']:.1f} boxes"
              if cost[ds] else r"\pending") for ds in NETWORKS]
    lines.append("Plug-in input per CAV message & " + " & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"

# ------------------------------------------------------------------------------------------------------------ T5, F8
NOISE_ROWS = [("benchmark", "Benchmark", "1"), ("drift", "Slow drift", "1"), ("bias", "Constant bias", "1"),
              ("mixed", "Mixed quality", "1"), ("beyond", r"Beyond training, $\sigma=5$", "5")]
NOISE_METHODS = [("bare", "Raw"), ("freealign", "F"), ("alignformer_tk", "A")]


def noise_row(dataset: str, kind: str, sigma: str):
    """{method: AP@0.7 mean over seeds 0-4} or None when not run yet."""
    path = OUT / (f"late_{dataset}.json" if kind == "benchmark" else f"noise_{dataset}_{kind}.json")
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    per_seed = [e for e in data["per_seed"].get(sigma, []) if e["seed"] < 5]
    if not per_seed:
        return None
    return {m: float(np.mean([e["ap"][m]["all"]["ap_70"] for e in per_seed])) for m, _ in NOISE_METHODS}


def table_t5() -> str:
    lines = []
    for kind, label, sigma in NOISE_ROWS:
        cells = []
        for ds in NETWORKS:
            row = noise_row(ds, kind, sigma)
            if row is None:
                cells += [r"\pending"] * 3
            else:
                text = bold_best(row, ["freealign", "alignformer_tk"])
                cells += [text[m] for m, _ in NOISE_METHODS]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def figure_f8(plt) -> None:
    """T5 as grouped bars: AP@0.7 per noise kind, no correction / FreeAlign / AlignFormer, both datasets."""
    rows = {ds: [noise_row(ds, k, s) for k, _, s in NOISE_ROWS] for ds in NETWORKS}
    if any(r is None for ds in rows for r in rows[ds]):
        return
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.55))
    names = ["Bench.", "Drift", "Bias", "Mixed", r"$\sigma$=5"]
    x = np.arange(len(names))
    width = 0.26
    colors = {"bare": (RAW, "#9a9ca1"), "freealign": ("FreeAlign", "#eb6834"),
              "alignformer_tk": ("RobustAlign", "#2a78d6")}
    for ax, ds in zip(axes, NETWORKS):
        for i, (m, _) in enumerate(NOISE_METHODS):
            label, color = colors[m]
            ax.bar(x + (i - 1) * width, [r[m] for r in rows[ds]], width * 0.92, color=color, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=5.8)
        ax.set_title(DATASET_NAME[ds], fontsize=7.5, color="#17202B", pad=3)
        _axes_style(ax)
    axes[0].set_ylabel(r"AP$_{0.7}$", fontsize=7, color="#3b4552", labelpad=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=6, frameon=False, bbox_to_anchor=(0.5, 1.06),
               handlelength=1.2, columnspacing=1.0)
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.16, top=0.84, wspace=0.3)
    fig.savefig(FIGURES / "f8_noise.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, body in (("t1_hosts_body.tex", table_t1()), ("t3_late_body.tex", table_t3()),
                       ("t6_delay_body.tex", table_t6()), ("t4_pose_body.tex", table_t4()),
                       ("t7_ablation_body.tex", table_t7()), ("t5_noise_body.tex", table_t5())):
        (TABLES / name).write_text("%% generated by scripts/make_paper_exhibits.py, do not edit\n" + body)
        print("wrote", TABLES / name)
    (TABLES / "ts_late_ap50_body.tex").write_text("%% generated by scripts/make_paper_exhibits.py, do not edit\n"
                                                  + table_t3((("ap_50", "AP$_{0.5}$"),)))
    print("wrote", TABLES / "ts_late_ap50_body.tex")
    (TABLES / "ts_cost_body.tex").write_text("%% generated by scripts/make_paper_exhibits.py, do not edit\n" + table_cost())
    print("wrote", TABLES / "ts_cost_body.tex")
    top, bottom = table_decision()
    for name, body in (("ts_e2_body.tex", table_e2()), ("ts_decision_top_body.tex", top),
                       ("ts_decision_bottom_body.tex", bottom), ("ts_odometry_body.tex", table_odometry())):
        (TABLES / name).write_text("%% generated by scripts/make_paper_exhibits.py, do not edit\n" + body)
        print("wrote", TABLES / name)
    if all((OUT / f"pooled_{ds}.json").exists() for ds in NETWORKS):
        (TABLES / "ts_pooled_body.tex").write_text("%% generated by scripts/make_paper_exhibits.py, do not edit\n"
                                                   + table_pooled())
        print("wrote", TABLES / "ts_pooled_body.tex")
    for figure in (figure_f3, figure_f5, figure_f6, figure_f7, figure_f8):   # not in the paper; never block the tables
        try:
            figure(plt)
        except Exception as error:  # noqa: BLE001
            print(f"{figure.__name__} skipped:", error)
    print("wrote", FIGURES / "f3_curves.pdf", FIGURES / "f5_latency.pdf", FIGURES / "f6_late.pdf")
    try:                       # one figure per network (outputs/figures), not in the paper; never blocks the paper
        import make_network_figures
        make_network_figures.main()
    except Exception as error:  # noqa: BLE001
        print("network figures skipped:", error)


if __name__ == "__main__":
    main()
