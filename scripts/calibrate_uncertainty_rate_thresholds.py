"""Calibrate a three-rate BELT uncertainty policy on held-out validation data.

This selects the two thresholds used by ``--adaptive-rate-thresholds`` without
re-running detector inference.  The policy deliberately allocates its larger
query-state payloads to proposals whose validation uncertainty is more
predictive of an unmatched detection, subject to a requested mean byte budget.
It is a validation-calibrated candidate policy; its end-to-end AP must still
be measured once on the untouched test split.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIELDS = ("class_uncertainty", "position_std_m", "heading_std_deg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-mean-bytes", type=float, default=8.0)
    parser.add_argument(
        "--rate-bytes",
        nargs=3,
        type=float,
        metavar=("LOW", "MEDIUM", "HIGH"),
        default=(4.0, 8.0, 16.0),
        help="Cumulative payload bytes for residual stages 1, 2, and 3.",
    )
    parser.add_argument(
        "--budget-tolerance-bytes",
        type=float,
        default=0.05,
        help="Allowed deviation while maximizing risk targeting.",
    )
    parser.add_argument("--threshold-grid-size", type=int, default=101)
    return parser.parse_args()


def _rank(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.full(len(values), 0.5, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / (len(values) - 1)


def _load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"frame", "source_cav", "matched", *FIELDS}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(
            "CSV must be regenerated with the current uncertainty collector; "
            f"missing columns: {sorted(missing)}"
        )
    for row in rows:
        row["frame"] = int(row["frame"])
        row["matched"] = row["matched"].lower() == "true"
        for field in FIELDS:
            row[field] = float(row[field])
    return rows


def _uncertainty_scores(rows: list[dict]) -> np.ndarray:
    groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["frame"], row["source_cav"])].append(index)
    scores = np.empty(len(rows), dtype=np.float64)
    for indices in groups.values():
        scores[indices] = 0.0
        for field in FIELDS:
            values = np.asarray([rows[index][field] for index in indices])
            scores[indices] += _rank(values) / len(FIELDS)
    return scores


def _rates(scores: np.ndarray, low: float, high: float, rate_bytes) -> np.ndarray:
    first, second, third = rate_bytes
    return np.where(scores < low, first, np.where(scores < high, second, third))


def _select_thresholds(scores, unmatched, args):
    grid = np.linspace(0.0, 1.0, args.threshold_grid_size)
    candidates = []
    for low_index, low in enumerate(grid[:-1]):
        for high in grid[low_index + 1 :]:
            rates = _rates(scores, low, high, args.rate_bytes)
            mean_bytes = float(rates.mean())
            if abs(mean_bytes - args.target_mean_bytes) > args.budget_tolerance_bytes:
                continue
            # Prefer policies that concentrate larger payloads on the
            # validation detections identified as risky by BELT uncertainty.
            utility = float(rates[unmatched].mean() - rates[~unmatched].mean())
            candidates.append((utility, -abs(mean_bytes - args.target_mean_bytes), low, high, rates))
    if not candidates:
        # A discrete rate policy cannot always meet an arbitrary byte target
        # exactly.  Use the closest feasible policy, then maximize risk focus.
        closest = []
        for low_index, low in enumerate(grid[:-1]):
            for high in grid[low_index + 1 :]:
                rates = _rates(scores, low, high, args.rate_bytes)
                mean_bytes = float(rates.mean())
                utility = float(rates[unmatched].mean() - rates[~unmatched].mean())
                closest.append((-abs(mean_bytes - args.target_mean_bytes), utility, low, high, rates))
        _, _, low, high, rates = max(closest, key=lambda item: (item[0], item[1]))
    else:
        _, _, low, high, rates = max(candidates, key=lambda item: (item[0], item[1]))
    return float(low), float(high), rates


def _plot(scores, unmatched, rates, low, high, rate_bytes, output):
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for status, color, label in (
        (False, "#1769aa", "Matched"),
        (True, "#e0702a", "Unmatched"),
    ):
        axes[0].hist(
            scores[unmatched == status], bins=35, density=True, histtype="step",
            linewidth=2.0, color=color, label=label,
        )
    axes[0].axvline(low, color="#555555", linestyle="--", linewidth=1.5)
    axes[0].axvline(high, color="#555555", linestyle="--", linewidth=1.5)
    axes[0].text(low, axes[0].get_ylim()[1] * 0.94, "$\\tau_1$", ha="right", va="top")
    axes[0].text(high, axes[0].get_ylim()[1] * 0.94, "$\\tau_2$", ha="left", va="top")
    axes[0].set_title("Combined source uncertainty score")
    axes[0].set_xlabel("Within-source uncertainty rank")
    axes[0].set_ylabel("Probability density")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.22)

    matched_share = [
        float((rates[~unmatched] == mode).mean()) for mode in rate_bytes
    ]
    unmatched_share = [
        float((rates[unmatched] == mode).mean()) for mode in rate_bytes
    ]
    position = np.arange(len(rate_bytes))
    width = 0.36
    axes[1].bar(position - width / 2, matched_share, width, color="#1769aa", label="Matched")
    axes[1].bar(position + width / 2, unmatched_share, width, color="#e0702a", label="Unmatched")
    axes[1].set_xticks(position, [f"{value:g} B" for value in rate_bytes])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Learned rate allocation")
    axes[1].set_xlabel("Query-state payload per source object")
    axes[1].set_ylabel("Fraction of proposals")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.22)
    figure.suptitle("Validation-calibrated uncertainty-adaptive residual coding", fontsize=15)
    figure.savefig(output, dpi=300, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    rate_bytes = tuple(args.rate_bytes)
    if not rate_bytes[0] < rate_bytes[1] < rate_bytes[2]:
        raise ValueError("--rate-bytes must be strictly increasing")
    if not rate_bytes[0] <= args.target_mean_bytes <= rate_bytes[-1]:
        raise ValueError("--target-mean-bytes must be within --rate-bytes")
    rows = _load_rows(args.input_csv)
    scores = _uncertainty_scores(rows)
    unmatched = ~np.asarray([row["matched"] for row in rows], dtype=bool)
    low, high, rates = _select_thresholds(scores, unmatched, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_csv": str(args.input_csv.resolve()),
        "target_mean_bytes": args.target_mean_bytes,
        "rate_bytes": rate_bytes,
        "achieved_mean_bytes": float(rates.mean()),
        "adaptive_rate_thresholds": [low, high],
        "cli_argument": f"--adaptive-rate-thresholds {low:.4f} {high:.4f}",
        "selection_objective": (
            "Within the requested byte budget, allocate larger query-state "
            "payloads preferentially to validation proposals whose BELT "
            "uncertainty predicts an unmatched detection."
        ),
        "mode_fraction": {
            f"{int(mode)}_bytes": float((rates == mode).mean())
            for mode in rate_bytes
        },
        "mean_bytes_matched": float(rates[~unmatched].mean()),
        "mean_bytes_unmatched": float(rates[unmatched].mean()),
    }
    (args.output_dir / "adaptive_rate_thresholds.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    _plot(
        scores,
        unmatched,
        rates,
        low,
        high,
        rate_bytes,
        args.output_dir / "adaptive_rate_calibration.png",
    )
    print("ADAPTIVE_RATE", json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
