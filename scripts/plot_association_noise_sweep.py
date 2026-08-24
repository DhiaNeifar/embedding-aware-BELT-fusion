#!/usr/bin/env python3
"""Render a publication-ready association-robustness noise sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
NAVY = (25, 81, 137)
GRID = (222, 227, 233)
TEXT = (25, 32, 41)
MUTED = (79, 89, 101)


def parse_point(value: str) -> tuple[float, Path]:
    try:
        severity, path = value.split("=", 1)
        return float(severity), Path(path)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "points must be written as SEVERITY=PATH_TO_test_metrics.json"
        ) from error


def _font(size: int, bold: bool = False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def _centered(draw: ImageDraw.ImageDraw, box, text: str, font, fill) -> None:
    left, top, right, _ = box
    width = draw.textbbox((0, 0), text, font=font)[2]
    draw.text(((left + right - width) / 2, top), text, font=font, fill=fill)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", action="append", required=True, type=parse_point)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    points = []
    for severity, path in args.point:
        metrics = json.loads(path.read_text())
        points.append((severity, float(metrics["embedding_top1"])))
    points.sort()

    image = Image.new("RGB", (2400, 1600), "white")
    draw = ImageDraw.Draw(image)
    title_font, subtitle_font = _font(46, True), _font(27)
    tick_font, axis_font, annotation_font = _font(25), _font(31, True), _font(23, True)
    left, top, right, bottom = 295, 285, 2230, 1245

    draw.text((120, 78), "Cross-Agent Association Robustness Under Localization Noise",
              font=title_font, fill=TEXT)
    draw.text((120, 140),
              "Frozen PointPillars detector and Spatial TrackFormer association embedder; official OPV2V test split",
              font=subtitle_font, fill=MUTED)

    draw.rectangle((left, top, right, bottom), outline=(93, 105, 119), width=3)
    for fraction in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = bottom - fraction * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=2)
        label = f"{fraction:.1f}"
        width = draw.textbbox((0, 0), label, font=tick_font)[2]
        draw.text((left - 32 - width, y - 15), label, font=tick_font, fill=MUTED)

    severities = [value for value, _ in points]
    x_min, x_max = min(severities), max(severities)
    span = max(x_max - x_min, 1.0)
    x_min -= span * 0.04
    x_max += span * 0.04

    def project(severity: float, accuracy: float) -> tuple[float, float]:
        x = left + (severity - x_min) / (x_max - x_min) * (right - left)
        y = bottom - accuracy * (bottom - top)
        return x, y

    pixel_points = [project(severity, accuracy) for severity, accuracy in points]
    for severity, _ in points:
        x, _ = project(severity, 0.0)
        draw.line((x, top, x, bottom), fill=(238, 241, 245), width=1)
        label = f"{severity:g}"
        width = draw.textbbox((0, 0), label, font=tick_font)[2]
        draw.text((x - width / 2, bottom + 22), label, font=tick_font, fill=MUTED)

    draw.line(pixel_points, fill=NAVY, width=7)
    for (_, accuracy), (x, y) in zip(points, pixel_points):
        draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill="white", outline=NAVY, width=6)
        label = f"{accuracy:.4f}"
        label_width = draw.textbbox((0, 0), label, font=annotation_font)[2]
        draw.rounded_rectangle((x - label_width / 2 - 10, y - 55, x + label_width / 2 + 10, y - 19),
                               radius=8, fill="white", outline=(207, 214, 222), width=1)
        draw.text((x - label_width / 2, y - 53), label, font=annotation_font, fill=TEXT)

    _centered(draw, (left, bottom + 76, right, bottom + 76),
              "Noise severity multiplier (α = 1 corresponds to 0.2 m, 0.2°, and 100 ms)", axis_font, TEXT)
    draw.text((120, top - 64), "Cross-agent Top-1 association accuracy", font=axis_font, fill=TEXT)
    draw.text((left, 1395),
              "Higher is better. Each point uses the same 2,170 held-out OPV2V test frames and noise seed 20.",
              font=subtitle_font, fill=MUTED)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, dpi=(300, 300))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
