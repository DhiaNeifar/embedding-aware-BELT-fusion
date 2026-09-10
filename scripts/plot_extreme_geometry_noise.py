#!/usr/bin/env python3
"""Plot association robustness under large position and heading errors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TEXT, MUTED, GRID = (25, 32, 41), (79, 89, 101), (222, 227, 233)
BLUE, ORANGE, GREEN, BASELINE = (35, 100, 210), (220, 111, 27), (39, 135, 82), (116, 126, 140)


def _font(size: int, bold: bool = False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def _top1(path: Path) -> float:
    return float(json.loads(path.read_text())["embedding_top1"])


def _series(root: Path, entries: list[tuple[str, str]]) -> list[tuple[str, float]]:
    return [(label, _top1(root / directory / "test_metrics.json"))
            for label, directory in entries]


def _dashed_line(draw, points, color, width, dash=30, gap=18):
    """Draw a dashed polyline while preserving the line direction."""
    for start, end in zip(points, points[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        offset = 0.0
        while offset < length:
            segment_end = min(offset + dash, length)
            draw.line(
                (
                    start[0] + dx * offset / length,
                    start[1] + dy * offset / length,
                    start[0] + dx * segment_end / length,
                    start[1] + dy * segment_end / length,
                ),
                fill=color,
                width=width,
            )
            offset += dash + gap


def _plot(draw, rect, series, labels, fonts):
    left, top, right, bottom = rect
    panel_font, tick_font = fonts
    draw.rectangle(rect, outline=(93, 105, 119), width=3)

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = bottom - fraction * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=2)
        label = f"{fraction:.2f}"
        width = draw.textbbox((0, 0), label, font=tick_font)[2]
        draw.text((left - width - 16, y - 13), label, font=tick_font, fill=MUTED)

    count = len(labels)
    def project(index: int, value: float) -> tuple[float, float]:
        x = left + index / (count - 1) * (right - left)
        y = bottom - value * (bottom - top)
        return x, y

    for index, label in enumerate(labels):
        x, _ = project(index, 0.0)
        draw.line((x, top, x, bottom), fill=(238, 241, 245), width=1)
        label_width = draw.textbbox((0, 0), label, font=tick_font)[2]
        draw.multiline_text((x - label_width / 2, bottom + 14), label,
                            font=tick_font, fill=MUTED, align="center",
                            spacing=1)

    for _, data, color, dashed in series:
        pixels = [project(index, value) for index, (_, value) in enumerate(data)]
        if dashed:
            _dashed_line(draw, pixels, color, width=6)
        else:
            draw.line(pixels, fill=color, width=6)
        for x, y in pixels:
            draw.ellipse((x - 10, y - 10, x + 10, y + 10),
                         fill="white", outline=color, width=5)



def _legend(draw, tick_font):
    """Draw the compact color/style key above the plotting area."""
    entries = (
        (195, "Position only", BLUE, False),
        (500, "Heading only", ORANGE, False),
        (805, "Position + heading", GREEN, False),
        (1215, "Solid: clean", TEXT, False),
        (1485, "Dashed: annealed", TEXT, True),
    )
    for x, label, color, dashed in entries:
        if dashed:
            _dashed_line(draw, [(x, 185), (x + 42, 185)], color, width=5, dash=14, gap=8)
        else:
            draw.line((x, 185, x + 42, 185), fill=color, width=5)
        draw.text((x + 58, 170), label, font=tick_font, fill=TEXT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compare-curriculum", action="store_true",
        help="Overlay dashed curves from the final geometry curriculum model.",
    )
    args = parser.parse_args()
    root = args.outputs_root
    clean = _top1(root / "spatial_trackformer_association_head_128d_test" /
                  "test_metrics.json")

    position = _series(root, [
        ("α=0\n0", "spatial_trackformer_association_head_128d_test"),
        ("α=1\n1", "geometry_position_1m_eval"),
        ("α=2\n2", "geometry_position_2m_eval"),
        ("α=3\n4", "geometry_position_4m_eval"),
        ("α=4\n10", "geometry_position_10m_eval"),
    ])
    heading = _series(root, [
        ("α=0\n0", "spatial_trackformer_association_head_128d_test"),
        ("α=1\n5", "geometry_heading_5deg_eval"),
        ("α=2\n10", "geometry_heading_10deg_eval"),
        ("α=3\n20", "geometry_heading_20deg_eval"),
        ("α=4\n30", "geometry_heading_30deg_eval"),
    ])
    combined = _series(root, [
        ("α=0\n0 / 0", "spatial_trackformer_association_head_128d_test"),
        ("α=1\n1 / 5", "geometry_combined_1m_5deg_eval"),
        ("α=2\n2 / 10", "geometry_combined_2m_10deg_eval"),
        ("α=3\n4 / 20", "geometry_combined_4m_20deg_eval"),
        ("α=4\n10 / 30", "geometry_combined_10m_30deg_eval"),
    ])

    curriculum = None
    if args.compare_curriculum:
        curriculum_clean = "curriculum_strong_clean_eval"
        curriculum = (
            _series(root, [
                ("α=0", curriculum_clean),
                ("α=1", "curriculum_strong_position_1m_eval"),
                ("α=2", "curriculum_strong_position_2m_eval"),
                ("α=3", "curriculum_strong_position_4m_eval"),
                ("α=4", "curriculum_strong_position_10m_eval"),
            ]),
            _series(root, [
                ("α=0", curriculum_clean),
                ("α=1", "curriculum_strong_heading_5deg_eval"),
                ("α=2", "curriculum_strong_heading_10deg_eval"),
                ("α=3", "curriculum_strong_heading_20deg_eval"),
                ("α=4", "curriculum_strong_heading_30deg_eval"),
            ]),
            _series(root, [
                ("α=0", curriculum_clean),
                ("α=1", "curriculum_strong_combined_1m_5deg_eval"),
                ("α=2", "curriculum_strong_combined_2m_10deg_eval"),
                ("α=3", "curriculum_strong_combined_4m_20deg_eval"),
                ("α=4", "curriculum_strong_combined_10m_30deg_eval"),
            ]),
        )

    image = Image.new("RGB", (1900, 1450), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(32, True)
    panel_font, tick_font = _font(29, True), _font(23)
    axis_font = _font(31, True)
    title = (
        "Clean vs. Annealed Embedding Under Extreme Geometry-Only Localization Noise"
        if curriculum else
        "Cross-Agent Association Under Extreme Geometry-Only Localization Noise"
    )
    draw.text((120, 72), title,
              font=title_font, fill=TEXT)
    if curriculum:
        _legend(draw, tick_font)

    labels = ["α=0\n0 m / 0°", "α=1\n1 m / 5°", "α=2\n2 m / 10°",
              "α=3\n4 m / 20°", "α=4\n10 m / 30°"]
    series = [
        ("Position only", position, BLUE, False),
        ("Heading only", heading, ORANGE, False),
        ("Position + heading", combined, GREEN, False),
    ]
    if curriculum:
        series.extend([
            ("Position only", curriculum[0], BLUE, True),
            ("Heading only", curriculum[1], ORANGE, True),
            ("Position + heading", curriculum[2], GREEN, True),
        ])
    _plot(draw, (195, 305, 1770, 1120), series, labels, (panel_font, tick_font))

    draw.text((195, 240), "Cross-agent Top-1 association accuracy", font=axis_font, fill=TEXT)
    draw.text((490, 1260), "Noise level α  (position standard deviation / heading standard deviation)",
              font=axis_font, fill=TEXT)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, dpi=(300, 300))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
