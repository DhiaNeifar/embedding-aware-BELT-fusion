"""Plot a controlled PQ-STF low-rate quantization sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


CONFIGURATIONS = (
    ("Raw\n1,024 B", "raw_1024B"),
    ("Fixed\n1 B", "fixed_1B"),
    ("Fixed\n2 B", "fixed_2B"),
    ("Fixed\n4 B", "fixed_4B"),
    ("Adaptive\n1–4 B", "adaptive_1_2_4B"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for label, directory in CONFIGURATIONS:
        path = args.metrics_root / directory / "metrics.json"
        metrics = json.loads(path.read_text())
        message = metrics["association_message"]
        bits = message.get("mean_association_bits_per_source_object")
        if bits is None:
            bits = message["bits_per_source_object"]
        rows.append(
            {
                "label": label.replace("\n", " "),
                "bytes_per_object": float(bits) / 8.0,
                "ap_70": float(metrics["belt_trackformer"]["ap_70"]),
                "fused_boxes_per_frame": float(
                    metrics["mean_trackformer_fused_boxes_per_frame"]
                ),
                "multi_agent_groups_per_frame": float(
                    metrics["mean_trackformer_multi_agent_groups_per_frame"]
                ),
            }
        )

    output_csv = args.metrics_root / "quantization_rate_sweep.csv"
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update(
        {"font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13}
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    x = list(range(len(rows)))
    labels = [label for label, _ in CONFIGURATIONS]
    ap = [100.0 * row["ap_70"] for row in rows]
    axes[0].plot(x, ap, color="#1769aa", marker="o", linewidth=2.3, markersize=7)
    for index, value in enumerate(ap):
        axes[0].annotate(
            f"{value:.3f}%", (index, value), xytext=(0, 9),
            textcoords="offset points", ha="center", fontsize=9,
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("AP@0.7 (%)")
    axes[0].set_title("Detection quality")
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axes[0].grid(alpha=0.25)

    width = 0.36
    fused = [row["fused_boxes_per_frame"] for row in rows]
    groups = [row["multi_agent_groups_per_frame"] for row in rows]
    axes[1].bar([value - width / 2 for value in x], fused, width, color="#1769aa", label="Final boxes / frame")
    axes[1].bar([value + width / 2 for value in x], groups, width, color="#e0702a", label="Multi-agent groups / frame")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Mean count per frame")
    axes[1].set_title("Association behaviour")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("Controlled PQ-STF query-state quantization sweep", fontsize=16)
    output_png = args.metrics_root / "quantization_rate_sweep.png"
    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    print(output_csv)
    print(output_png)


if __name__ == "__main__":
    main()
