"""Plot measured PQ-STF clean-test AP@0.7 against residual compression."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


ROOT = Path(__file__).resolve().parents[1]
MEASUREMENTS = (
    (
        "Raw 256-D state",
        ROOT / "outputs/belt_trackformer_clean_test_d5_s090/metrics.json",
        "fixed",
    ),
    (
        "Residual stage 3",
        ROOT / "outputs/belt_pq_stf_residual_128bit_clean_test_d5_s090/metrics.json",
        "fixed",
    ),
    (
        "Residual stages 1--2",
        ROOT / "outputs/belt_pq_stf_residual_64bit_clean_test_d5_s090/metrics.json",
        "fixed",
    ),
    (
        "Residual stage 1",
        ROOT / "outputs/belt_pq_stf_residual_32bit_clean_test_d5_s090/metrics.json",
        "fixed",
    ),
    (
        "Adaptive 1/2/4-B residual code",
        ROOT / "outputs/belt_pq_stf_adaptive_1_2_4B_clean_test_d5_s090/metrics.json",
        "adaptive",
    ),
)


def main() -> None:
    rows = []
    for label, path, family in MEASUREMENTS:
        metrics = json.loads(path.read_text())
        message = metrics["association_message"]
        bits = float(
            message.get("mean_association_bits_per_source_object")
            or message["bits_per_source_object"]
        )
        rows.append(
            {
                "method": label,
                "family": family,
                "bits_per_object": bits,
                "bytes_per_object": bits / 8,
                "compression_factor": 8192 / bits,
                "ap_70": float(metrics["belt_trackformer"]["ap_70"]),
            }
        )

    output_dir = ROOT / "outputs/pq_stf_compression_tradeoff"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "clean_ap70_vs_compression.csv"
    png_path = output_dir / "clean_ap70_vs_compression.png"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    # Payload sizes span three orders of magnitude.  Treat configurations as
    # evenly spaced experimental categories rather than visually implying a
    # continuous linear byte scale.
    x = list(range(len(rows)))
    y = [100 * row["ap_70"] for row in rows]
    fixed_indices = [index for index, row in enumerate(rows) if row["family"] == "fixed"]
    adaptive_indices = [index for index, row in enumerate(rows) if row["family"] == "adaptive"]
    axis.plot(
        fixed_indices,
        [y[index] for index in fixed_indices],
        color="#1769aa",
        linewidth=2.4,
        marker="o",
        markersize=8,
        label="Fixed-rate coding",
    )
    axis.scatter(
        adaptive_indices,
        [y[index] for index in adaptive_indices],
        color="#e0702a",
        marker="D",
        s=74,
        zorder=3,
        label="Uncertainty-adaptive coding",
    )
    for index, row in enumerate(rows):
        axis.annotate(
            f"{100 * row['ap_70']:.3f}%",
            (index, 100 * row["ap_70"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    axis.set_xticks(
        x,
        [
            (
                "Raw state\n1,024 B"
                if index == 0
                else f"Residual code\n{row['bytes_per_object']:.0f} B "
                f"({row['compression_factor']:.0f}$\\times$)"
                if row["family"] == "fixed"
                else f"Adaptive code\n{row['bytes_per_object']:.2f} B "
                f"({row['compression_factor']:.0f}$\\times$)"
            )
            for index, row in enumerate(rows)
        ],
    )
    axis.set_xlabel("Association-message payload per detected source object")
    axis.set_ylabel("AP@0.7 (%)")
    axis.set_title("PQ-STF compression retains clean-test detection accuracy")
    axis.grid(True, which="both", alpha=0.25)
    axis.set_ylim(76.44, 76.75)
    axis.set_yticks([76.5, 76.6, 76.7])
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axis.legend(loc="upper left", frameon=False, fontsize=10)
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    print(png_path)
    print(csv_path)


if __name__ == "__main__":
    main()
