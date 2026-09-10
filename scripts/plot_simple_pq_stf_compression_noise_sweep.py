"""Plot PQ-STF-only AP@0.7 against localization noise for fixed bit rates."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs/simple_pq_stf_compression_noise_sweep"
CONDITIONS = (
    ("clean", 0.0, 0.0),
    ("p1_h5", 1.0, 5.0),
    ("p2_h10", 2.0, 10.0),
    ("p4_h20", 4.0, 20.0),
    ("p10_h30", 10.0, 30.0),
)
RATES = (
    ("raw", "Raw state (1,024 B)", "simple_pq_stf_raw_clean_test_d5_s090", "#1b5e9e"),
    ("fixed_4B", "Fixed 4 B", "simple_pq_stf_fixed_4B_clean_test_d5_s090", "#00897b"),
    ("fixed_2B", "Fixed 2 B", "simple_pq_stf_fixed_2B_clean_test_d5_s090", "#f57c00"),
    ("fixed_1B", "Fixed 1 B", "simple_pq_stf_fixed_1B_clean_test_d5_s090", "#b71c1c"),
)


def _metrics(condition: str, rate: str, clean_dir: str) -> dict:
    path = (
        ROOT / "outputs" / clean_dir / "metrics.json"
        if condition == "clean"
        else OUTPUT_ROOT / condition / rate / "metrics.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation result: {path}")
    return json.loads(path.read_text())


def main() -> None:
    rows = []
    for condition, position, heading in CONDITIONS:
        for rate, label, clean_dir, _ in RATES:
            metrics = _metrics(condition, rate, clean_dir)
            message = metrics["association_message"]
            rows.append(
                {
                    "condition": condition,
                    "position_std_m": position,
                    "heading_std_deg": heading,
                    "rate": rate,
                    "bits_per_object": message["mean_association_bits_per_source_object"],
                    "ap_70": metrics["simple_trackformer"]["ap_70"],
                    "groups_per_frame": metrics["mean_trackformer_multi_agent_groups_per_frame"],
                    "boxes_per_frame": metrics["mean_trackformer_fused_boxes_per_frame"],
                }
            )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_ROOT / "ap70_by_noise_and_message_rate.csv"
    figure_path = OUTPUT_ROOT / "ap70_by_noise_and_message_rate.png"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update(
        {"font.size": 12, "axes.labelsize": 13, "axes.titlesize": 15}
    )
    figure, axis = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    x = list(range(len(CONDITIONS)))
    for rate, label, _, color in RATES:
        subset = [row for row in rows if row["rate"] == rate]
        axis.plot(
            x,
            [100.0 * row["ap_70"] for row in subset],
            marker="o",
            markersize=7,
            linewidth=2.3,
            color=color,
            label=label,
        )
    axis.set_xticks(
        x,
        ["0 m / 0°", "1 m / 5°", "2 m / 10°", "4 m / 20°", "10 m / 30°"],
    )
    axis.set_xlabel("Localization noise: position standard deviation / heading standard deviation")
    axis.set_ylabel("AP@0.7 (%)")
    axis.set_title("PQ-STF-only fusion under localization noise")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2, loc="best")
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    print(figure_path)
    print(csv_path)


if __name__ == "__main__":
    main()
