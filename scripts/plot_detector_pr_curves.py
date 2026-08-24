#!/usr/bin/env python3
"""Render publication-ready globally score-sorted OpenCOOD PR curves."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import yaml


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TEXT, MUTED, GRID = (25, 32, 41), (79, 89, 101), (222, 227, 233)
COLORS = ((33, 132, 80), (36, 104, 194), (221, 116, 35))


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _font(size: int, bold: bool = False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def _sample(points: list[tuple[float, float]], maximum: int = 5000):
    if len(points) <= maximum:
        return points
    stride = max(1, len(points) // maximum)
    return points[::stride] + [points[-1]]


def _panel(draw, rect, iou: str, models, fonts) -> None:
    left, top, right, bottom = rect
    title_font, tick_font, legend_font = fonts
    draw.rectangle(rect, outline=(93, 105, 119), width=3)
    draw.text((left, top - 78), f"IoU threshold = {float(iou) / 100:.1f}",
              font=title_font, fill=TEXT)
    draw.text((left, top - 36), "Precision", font=tick_font, fill=MUTED)

    for fraction in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        x = left + fraction * (right - left)
        y = bottom - fraction * (bottom - top)
        draw.line((x, top, x, bottom), fill=GRID, width=2)
        draw.line((left, y, right, y), fill=GRID, width=2)
        label = f"{fraction:.1f}"
        width = draw.textbbox((0, 0), label, font=tick_font)[2]
        draw.text((x - width / 2, bottom + 17), label, font=tick_font, fill=MUTED)
        draw.text((left - width - 19, y - 13), label, font=tick_font, fill=MUTED)

    for index, (name, values, color) in enumerate(models):
        points = _sample(list(zip(values[f"mrec_{iou}"], values[f"mpre_{iou}"])))
        pixels = [(left + recall * (right - left), bottom - precision * (bottom - top))
                  for recall, precision in points]
        draw.line(pixels, fill=color, width=5)
        ap_key = "ap_50" if iou == "50" else "ap_70"
        y = top + 19 + index * 42
        draw.line((left + 19, y + 14, left + 63, y + 14), fill=color, width=5)
        draw.text((left + 76, y), f"{name}  (AP = {values[ap_key]:.3f})",
                  font=legend_font, fill=TEXT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointpillars", type=Path, required=True)
    parser.add_argument("--swformer-best", type=Path, required=True)
    parser.add_argument("--swformer-last", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    models = [
        ("PointPillars (epoch 30)", _load(args.pointpillars), COLORS[0]),
        ("SWFormer (epoch 112, best)", _load(args.swformer_best), COLORS[1]),
        ("SWFormer (epoch 119, last)", _load(args.swformer_last), COLORS[2]),
    ]
    image = Image.new("RGB", (2600, 1500), "white")
    draw = ImageDraw.Draw(image)
    title_font, subtitle_font = _font(48, True), _font(28)
    panel_title_font, tick_font, legend_font = _font(34, True), _font(25), _font(24)
    axis_font = _font(32, True)

    draw.text((130, 75), "Precision–Recall Performance on the Official OPV2V Test Split",
              font=title_font, fill=TEXT)
    draw.text((130, 140),
              "Late fusion with global confidence sorting; area under each curve is Average Precision (AP)",
              font=subtitle_font, fill=MUTED)

    _panel(draw, (250, 330, 1190, 1160), "50", models,
           (panel_title_font, tick_font, legend_font))
    _panel(draw, (1510, 330, 2450, 1160), "70", models,
           (panel_title_font, tick_font, legend_font))

    draw.text((610, 1250), "Recall", font=axis_font, fill=TEXT)
    draw.text((1870, 1250), "Recall", font=axis_font, fill=TEXT)
    draw.text((250, 1360), "Upper-right is better: high precision retained as recall increases.",
              font=subtitle_font, fill=MUTED)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, dpi=(300, 300))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
