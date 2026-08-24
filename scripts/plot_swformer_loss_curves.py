#!/usr/bin/env python3
"""Render publication-ready SWFormer training and validation loss curves."""

from __future__ import annotations

import argparse
import struct
from collections import deque
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TEXT, MUTED, GRID = (25, 32, 41), (79, 89, 101), (222, 227, 233)
RAW, TRAIN, VALID = (148, 163, 184), (36, 104, 194), (205, 79, 59)


def _varint(data: bytes, position: int) -> tuple[int, int]:
    value = shift = 0
    while position < len(data):
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, position
        shift += 7
    raise ValueError("truncated protobuf varint")


def _fields(data: bytes):
    position = 0
    while position < len(data):
        key, position = _varint(data, position)
        field_number, wire_type = key >> 3, key & 0x07
        if wire_type == 0:
            value, position = _varint(data, position)
        elif wire_type == 1:
            value = data[position:position + 8]
            position += 8
        elif wire_type == 2:
            length, position = _varint(data, position)
            value = data[position:position + length]
            position += length
        elif wire_type == 5:
            value = data[position:position + 4]
            position += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")
        yield field_number, wire_type, value


def _scalar_values(event_payload: bytes):
    for field, wire_type, summary in _fields(event_payload):
        if field != 5 or wire_type != 2:
            continue
        for summary_field, summary_wire, value_message in _fields(summary):
            if summary_field != 1 or summary_wire != 2:
                continue
            tag = value = None
            for value_field, value_wire, raw_value in _fields(value_message):
                if value_field == 1 and value_wire == 2:
                    tag = raw_value.decode("utf-8")
                elif value_field == 2 and value_wire == 5:
                    value = struct.unpack("<f", raw_value)[0]
            if tag is not None and value is not None:
                yield tag, value


def load_scalars(log_dir: Path) -> dict[str, dict[int, float]]:
    scalars: dict[str, dict[int, float]] = {}
    for event_path in sorted(log_dir.glob("events.out.tfevents.*")):
        with event_path.open("rb") as stream:
            while True:
                header = stream.read(12)
                if len(header) < 12:
                    break
                length = struct.unpack("<Q", header[:8])[0]
                payload = stream.read(length)
                stream.read(4)
                if len(payload) != length:
                    break
                event_step = next((int(raw) for field, wire, raw in _fields(payload)
                                   if field == 2 and wire == 0), None)
                if event_step is None:
                    continue
                for tag, value in _scalar_values(payload):
                    scalars.setdefault(tag, {})[event_step] = value
    return scalars


def moving_average(points: list[tuple[float, float]], window: int):
    history: deque[float] = deque()
    total = 0.0
    result = []
    for x_value, y_value in points:
        history.append(y_value)
        total += y_value
        if len(history) > window:
            total -= history.popleft()
        if len(history) == window:
            result.append((x_value, total / window))
    return result


def _font(size: int, bold: bool = False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def draw_curve(draw, rect, points, color, title, fonts) -> None:
    left, top, right, bottom = rect
    panel_font, tick_font = fonts
    draw.rectangle(rect, outline=(93, 105, 119), width=3)
    draw.text((left, top - 50), title, font=panel_font, fill=TEXT)
    if not points:
        draw.text((left + 20, top + 20), "No scalar values found", font=tick_font, fill=color)
        return

    x_min, x_max = points[0][0], points[-1][0]
    values = [value for _, value in points]
    y_min, y_max = min(values), max(values)
    margin = max((y_max - y_min) * 0.08, max(abs(y_max), 1.0) * 0.02)
    y_min -= margin
    y_max += margin

    def project(point):
        x_value, y_value = point
        x_fraction = 0.5 if x_max == x_min else (x_value - x_min) / (x_max - x_min)
        y_fraction = 0.5 if y_max == y_min else (y_value - y_min) / (y_max - y_min)
        return left + x_fraction * (right - left), bottom - y_fraction * (bottom - top)

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + fraction * (right - left)
        y = bottom - fraction * (bottom - top)
        draw.line((x, top, x, bottom), fill=(239, 242, 246), width=1)
        draw.line((left, y, right, y), fill=GRID, width=2)
        x_value = x_min + fraction * (x_max - x_min)
        y_value = y_min + fraction * (y_max - y_min)
        x_label, y_label = f"{x_value:.0f}", f"{y_value:.1f}"
        x_width = draw.textbbox((0, 0), x_label, font=tick_font)[2]
        y_width = draw.textbbox((0, 0), y_label, font=tick_font)[2]
        draw.text((x - x_width / 2, bottom + 15), x_label, font=tick_font, fill=MUTED)
        draw.text((left - y_width - 18, y - 13), y_label, font=tick_font, fill=MUTED)

    pixels = [project(point) for point in points]
    if len(pixels) > 1:
        draw.line(pixels, fill=color, width=4)
    else:
        x, y = pixels[0]
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps-per-epoch", type=int, default=798,
                        help="Optimizer steps per epoch (798 for this saved batch-size-8 OPV2V run)")
    parser.add_argument("--smoothing-window", type=int, default=25)
    args = parser.parse_args()

    scalars = load_scalars(args.log_dir)
    if "SWFormer/total_loss" not in scalars or "Validate_Loss" not in scalars:
        raise RuntimeError(f"Expected SWFormer/total_loss and Validate_Loss in {args.log_dir}")
    raw_train = sorted((step / args.steps_per_epoch, value)
                       for step, value in scalars["SWFormer/total_loss"].items())
    smooth_train = moving_average(raw_train, args.smoothing_window)
    validation = sorted((float(epoch), value) for epoch, value in scalars["Validate_Loss"].items())

    image = Image.new("RGB", (2400, 2040), "white")
    draw = ImageDraw.Draw(image)
    title_font, subtitle_font = _font(46, True), _font(27)
    panel_font, tick_font, axis_font = _font(32, True), _font(24), _font(29, True)
    draw.text((135, 76), "SWFormer Optimization History", font=title_font, fill=TEXT)
    draw.text((135, 138),
              "Late-fusion training run; raw mini-batch loss, smoothed training loss, and held-out validation loss",
              font=subtitle_font, fill=MUTED)

    panels = [(300, 315, 2240, 720), (300, 955, 2240, 1360), (300, 1595, 2240, 2000)]
    draw_curve(draw, panels[0], raw_train, RAW, "Raw training total loss (one point per mini-batch)",
               (panel_font, tick_font))
    draw_curve(draw, panels[1], smooth_train, TRAIN,
               f"Training total loss ({args.smoothing_window}-mini-batch moving average)",
               (panel_font, tick_font))
    draw_curve(draw, panels[2], validation, VALID, "Validation total loss (one point per epoch)",
               (panel_font, tick_font))
    for _, top, _, bottom in panels:
        draw.text((300, top - 94), "Total loss", font=axis_font, fill=TEXT)
        draw.text((1110, bottom + 57), "Training epoch", font=axis_font, fill=TEXT)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, dpi=(300, 300))
    print(f"Saved {args.output}")
    print(f"Training points: {len(raw_train)}; validation points: {len(validation)}")


if __name__ == "__main__":
    main()
