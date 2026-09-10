#!/usr/bin/env python3
"""Plot position, heading, and combined geometry-noise association sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TEXT, MUTED, GRID = (25, 32, 41), (79, 89, 101), (222, 227, 233)
COLORS = {
    "Combined position + heading": (35, 100, 210),
    "Position only": (220, 111, 27),
    "Heading only": (39, 135, 82),
}


def _font(size: int, bold: bool = False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def _metrics(path: Path) -> float:
    return float(json.loads(path.read_text())["embedding_top1"])


def _series(root: Path, mode: str) -> list[tuple[int, float]]:
    return [(alpha, _metrics(root / f"geometry_{mode}_alpha{alpha}_eval" /
                             "test_metrics.json"))
            for alpha in (1, 2, 3, 4, 5)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    series = {
        "Combined position + heading": _series(args.outputs_root, "combined"),
        "Position only": _series(args.outputs_root, "position_only"),
        "Heading only": _series(args.outputs_root, "heading_only"),
    }
    clean = _metrics(args.outputs_root /
                     "spatial_trackformer_association_head_128d_test" /
                     "test_metrics.json")
    values = [value for data in series.values() for _, value in data] + [clean]
    y_min, y_max = min(values) - 0.0012, max(values) + 0.0012

    image = Image.new("RGB", (2400, 1600), "white")
    draw = ImageDraw.Draw(image)
    title_font, subtitle_font = _font(46, True), _font(27)
    axis_font, tick_font, legend_font = _font(31, True), _font(25), _font(25)
    left, top, right, bottom = 300, 310, 2220, 1225

    draw.text((120, 75), "Cross-Agent Association Under Geometry-Only Localization Noise",
              font=title_font, fill=TEXT)
    draw.text((120, 137),
              "Frozen Spatial TrackFormer embedding on the official OPV2V test split; communication delay = 0 ms",
              font=subtitle_font, fill=MUTED)
    draw.rectangle((left, top, right, bottom), outline=(93, 105, 119), width=3)

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = bottom - fraction * (bottom - top)
        value = y_min + fraction * (y_max - y_min)
        draw.line((left, y, right, y), fill=GRID, width=2)
        label = f"{value:.3f}"
        width = draw.textbbox((0, 0), label, font=tick_font)[2]
        draw.text((left - width - 25, y - 14), label, font=tick_font, fill=MUTED)

    def project(alpha: int, value: float) -> tuple[float, float]:
        x = left + (alpha - 1) / 4 * (right - left)
        y = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
        return x, y

    for alpha in (1, 2, 3, 4, 5):
        x, _ = project(alpha, y_min)
        draw.line((x, top, x, bottom), fill=(238, 241, 245), width=1)
        label = str(alpha)
        width = draw.textbbox((0, 0), label, font=tick_font)[2]
        draw.text((x - width / 2, bottom + 22), label, font=tick_font, fill=MUTED)

    baseline_y = project(1, clean)[1]
    draw.line((left, baseline_y, right, baseline_y), fill=(116, 126, 140), width=3)
    baseline_label = f"Clean reference: {clean:.4f}"
    draw.rounded_rectangle((right - 310, baseline_y - 49, right - 15, baseline_y - 13),
                           radius=8, fill="white", outline=(207, 214, 222), width=1)
    draw.text((right - 296, baseline_y - 47), baseline_label, font=legend_font, fill=MUTED)

    for index, (name, data) in enumerate(series.items()):
        color = COLORS[name]
        pixels = [project(alpha, value) for alpha, value in data]
        draw.line(pixels, fill=color, width=6)
        for x, y in pixels:
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill="white", outline=color, width=5)
        # Keep the legend away from the alpha=1 points in the upper-left.
        legend_y = bottom - 168 + index * 42
        draw.line((left + 22, legend_y + 14, left + 67, legend_y + 14), fill=color, width=6)
        draw.text((left + 82, legend_y), name, font=legend_font, fill=TEXT)

    x_label = "Noise severity multiplier α  (α = 1: 0.2 m position and/or 0.2° heading error)"
    x_width = draw.textbbox((0, 0), x_label, font=axis_font)[2]
    draw.text(((left + right - x_width) / 2, bottom + 74), x_label, font=axis_font, fill=TEXT)
    draw.text((120, top - 61), "Cross-agent Top-1 association accuracy", font=axis_font, fill=TEXT)
    draw.text((left, 1385),
              "The vertical range is expanded to make the small geometry-only differences visible. Higher is better.",
              font=subtitle_font, fill=MUTED)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, dpi=(300, 300))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
