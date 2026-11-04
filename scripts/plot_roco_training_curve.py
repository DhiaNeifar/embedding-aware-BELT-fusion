#!/usr/bin/env python3
"""Plot RoCo's train vs. validation loss per epoch, to assess convergence.

De-duplicates the two tfevents files a resumed run produces: epoch 0 only
exists in the pre-disconnect file, epoch 1 onward is re-logged from scratch
by the resumed run (the authoritative version), so naively merging both
files double-counts epoch 1 and confuses the step numbering.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

BATCH_LEN = 1593
TRAIN_TAGS = ["Regression_loss", "Confidence_loss", "Dir_loss", "Iou_loss"]

# Okabe-Ito colorblind-safe pair.
COLOR_TRAIN = "#0072B2"
COLOR_VAL = "#E69F00"


def _load(path: Path) -> dict:
    ea = event_accumulator.EventAccumulator(str(path))
    ea.Reload()
    return {
        tag: [(e.step, e.value) for e in ea.Scalars(tag)]
        for tag in ea.Tags().get("scalars", [])
    }


def extract(run_dir: Path):
    files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(f"No tfevents files in {run_dir}")
    loaded = [_load(f) for f in files]

    per_epoch_train = {}
    for tag in TRAIN_TAGS:
        series: dict[int, list[float]] = {}
        for file_index, data in enumerate(loaded):
            for step, val in data.get(tag, []):
                epoch = step // BATCH_LEN
                # Only trust epoch 0 from the first (pre-disconnect) file;
                # every later file is an authoritative redo from its resume
                # point onward.
                if file_index == 0 and epoch != 0:
                    continue
                series.setdefault(epoch, []).append(val)
        per_epoch_train[tag] = {e: sum(v) / len(v) for e, v in series.items()}

    epochs = sorted(per_epoch_train[TRAIN_TAGS[0]].keys())
    train_total = {
        e: sum(per_epoch_train[t].get(e, 0.0) for t in TRAIN_TAGS) for e in epochs
    }

    val = {}
    for data in loaded:
        for step, v in data.get("Validate_Loss", []):
            val[step] = v  # later files override duplicates

    return train_total, val


def plot(train_total: dict, val: dict, output: Path, title: str):
    train_epochs = sorted(train_total.keys())
    val_epochs = sorted(val.keys())

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(
        train_epochs, [train_total[e] for e in train_epochs],
        color=COLOR_TRAIN, linewidth=2, label="Train loss",
    )
    ax.plot(
        val_epochs, [val[e] for e in val_epochs],
        color=COLOR_VAL, linewidth=2, marker="o", markersize=5,
        label="Validation loss",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, linewidth=0.5, alpha=0.3)
    ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="RoCo training/validation loss")
    parser.add_argument("--dump-json", type=Path, help="optional raw data dump")
    args = parser.parse_args()

    train_total, val = extract(args.run_dir)
    if args.dump_json:
        args.dump_json.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_json.open("w") as f:
            json.dump({"train_total": train_total, "val": val}, f, indent=2)
    plot(train_total, val, args.output, args.title)


if __name__ == "__main__":
    main()
