#!/usr/bin/env python3
"""Plot clean/noisy AP@0.7 from the concurrent BELT geometry-noise sweeps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "PointPillars naive late": (67, 105, 180),
    "Spatial TrackFormer": (222, 130, 48),
    "Annealed Spatial TrackFormer": (54, 145, 94),
    "Annealed Spatial TrackFormer + codebook": (166, 83, 163),
}
POINTS = (
    ("clean", "0 m / 0°"),
    ("p1_h5", "1 m / 5°"),
    ("p2_h10", "2 m / 10°"),
    ("p4_h20", "4 m / 20°"),
    ("p10_h30", "10 m / 30°"),
)
MODELS = (
    ("naive", "PointPillars naive late", "naive_late"),
    ("spatial_trackformer_clean", "Spatial TrackFormer", "belt_trackformer"),
    (
        "spatial_trackformer_annealed",
        "Annealed Spatial TrackFormer",
        "belt_trackformer",
    ),
    (
        "spatial_trackformer_annealed_codebook",
        "Annealed Spatial TrackFormer + codebook",
        "belt_trackformer",
    ),
)


def _font(size: int, bold: bool = False):
    name = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    return ImageFont.truetype(name, size)


def _ap70(path: Path, key: str) -> float:
    data = json.loads(path.read_text())
    return float(data[key]["ap_70"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=Path("outputs/belt_geometry_noise_sweep"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    series = {}
    table = []
    for directory, label, metric_key in MODELS:
        values = []
        for tag, noise_label in POINTS:
            value = _ap70(args.metrics_root / directory / tag / "metrics.json", metric_key)
            values.append(value)
            table.append(
                {
                    "model": label,
                    "noise": noise_label,
                    "ap_70": f"{value:.6f}",
                }
            )
        series[label] = values

    width, height = 2200, 1400
    left, top, right, bottom = 285, 330, 2075, 1115
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, axis_font = _font(43, True), _font(30, True)
    tick_font, legend_font = _font(25), _font(25, True)

    draw.text(
        (115, 80),
        "Cooperative Detection Under Geometry-Only Localization Noise",
        font=title_font,
        fill=(28, 35, 45),
    )
    draw.rectangle((left, top, right, bottom), outline=(95, 105, 117), width=3)
    y_min, y_max = 0.0, 0.85
    for value in (0.0, 0.2, 0.4, 0.6, 0.8):
        y = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
        draw.line((left, y, right, y), fill=(226, 231, 237), width=2)
        label = f"{value:.1f}"
        label_width = draw.textbbox((0, 0), label, font=tick_font)[2]
        draw.text((left - label_width - 18, y - 14), label, font=tick_font, fill=(77, 86, 99))

    def project(index: int, value: float):
        x = left + index / (len(POINTS) - 1) * (right - left)
        y = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
        return x, y

    for index, (_, label) in enumerate(POINTS):
        x, _ = project(index, 0.0)
        draw.line((x, top, x, bottom), fill=(239, 242, 246), width=1)
        box = draw.multiline_textbbox((0, 0), label, font=tick_font, align="center")
        draw.multiline_text(
            (x - (box[2] - box[0]) / 2, bottom + 20),
            label,
            font=tick_font,
            fill=(77, 86, 99),
            align="center",
        )

    # Two compact columns keep the legend clear of the y-axis label.
    legend_x, legend_y = 305, 190
    for offset, (label, values) in enumerate(series.items()):
        color = COLORS[label]
        column, row = divmod(offset, 2)
        x_legend = legend_x + column * 780
        y = legend_y + row * 42
        draw.line((x_legend, y + 12, x_legend + 45, y + 12), fill=color, width=6)
        draw.text((x_legend + 62, y), label, font=legend_font, fill=(28, 35, 45))
        pixels = [project(index, value) for index, value in enumerate(values)]
        draw.line(pixels, fill=color, width=7)
        for x, y in pixels:
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill="white", outline=color, width=5)

    draw.text((left, 265), "Vehicle AP@0.7", font=axis_font, fill=(28, 35, 45))
    x_label = "Position-noise standard deviation / heading-noise standard deviation"
    x_width = draw.textbbox((0, 0), x_label, font=axis_font)[2]
    draw.text(
        ((left + right - x_width) / 2, 1270),
        x_label,
        font=axis_font,
        fill=(28, 35, 45),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, dpi=(300, 300))
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "noise", "ap_70"))
        writer.writeheader()
        writer.writerows(table)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
