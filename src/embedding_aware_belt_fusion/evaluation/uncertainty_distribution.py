"""Collect and plot per-source-proposal BELT uncertainty on OPV2V.

The plot is deliberately limited to source-CAV proposals, because those are
the messages for which an uncertainty-adaptive communication policy would
choose a residual-codebook rate.  Proposals are labelled by a one-to-one
rotated-BEV IoU match to that CAV's local ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from embedding_aware_belt_fusion.features import (
    assign_proposals_to_ground_truth,
    decode_local_proposals,
)
from embedding_aware_belt_fusion.integration.belt_fusion import proposal_uncertainty
from embedding_aware_belt_fusion.integration.opencood_uncertainty import (
    FrozenPointPillarWithUncertainty,
)
from embedding_aware_belt_fusion.integration.train_uncertainty import (
    _load_detector_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opencood-config", type=Path, required=True)
    parser.add_argument("--detector-dir", type=Path, required=True)
    parser.add_argument("--uncertainty-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--scenario-split-file", type=Path)
    parser.add_argument("--scenario-role", choices=("train", "validation"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--minimum-gt-iou", type=float, default=0.5)
    parser.add_argument("--histogram-bins", type=int, default=40)
    parser.add_argument("--position-noise-std", type=float, default=0.0)
    parser.add_argument("--heading-noise-std-deg", type=float, default=0.0)
    parser.add_argument("--time-delay-ms", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=20)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def _configure_hypes(hypes: dict, args: argparse.Namespace) -> None:
    hypes["validate_dir"] = str((args.data_root / args.split).resolve())
    hypes.setdefault("wild_setting", {})
    hypes["wild_setting"].update(
        {
            "seed": args.noise_seed,
            "async": args.time_delay_ms > 0,
            "async_mode": "sim",
            "async_overhead": args.time_delay_ms,
            "loc_err": (
                args.position_noise_std > 0
                or args.heading_noise_std_deg > 0
            ),
            "xyz_std": args.position_noise_std,
            "ryp_std": args.heading_noise_std_deg,
        }
    )


def _scenario_id(dataset, index: int) -> str:
    scenario_index = next(
        position
        for position, end in enumerate(dataset.len_record)
        if index < end
    )
    scenario = dataset.scenario_database[scenario_index]
    timestamps = next(iter(scenario.values()))
    return Path(next(iter(timestamps.values()))["yaml"]).parents[1].name


def _summary(values: np.ndarray) -> dict[str, float | int]:
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _plot_pdf(rows: list[dict[str, float | bool]], output: Path, bins: int, iou: float) -> None:
    panels = (
        ("class_uncertainty", "Classification uncertainty"),
        ("position_std_m", "Position standard deviation (m)"),
        ("heading_std_deg", "Heading standard deviation (degrees)"),
    )
    matched = np.asarray([bool(row["matched"]) for row in rows])
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    colors = {True: "#1769aa", False: "#e0702a"}
    labels = {True: f"Matched (IoU $\\geq$ {iou:g})", False: "Unmatched"}
    for axis, (field, title) in zip(axes, panels):
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        lower, upper = float(values.min()), float(values.max())
        if math.isclose(lower, upper):
            lower -= 0.5
            upper += 0.5
        edges = np.linspace(lower, upper, bins + 1)
        for status in (True, False):
            selected = values[matched == status]
            if len(selected):
                axis.hist(
                    selected,
                    bins=edges,
                    density=True,
                    histtype="step",
                    linewidth=2.0,
                    color=colors[status],
                    label=labels[status],
                )
        axis.set_title(title)
        axis.set_xlabel("Predicted uncertainty")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Probability density")
    axes[0].legend(loc="upper right", frameon=False, fontsize=10)
    figure.suptitle(
        "BELT uncertainty for transmitted source-CAV detections", fontsize=16
    )
    figure.savefig(output, dpi=300, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not 0 < args.minimum_gt_iou <= 1:
        raise ValueError("--minimum-gt-iou must be in (0, 1]")
    device = torch.device(args.device)

    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.tools.train_utils import create_model, to_device

    hypes = load_yaml(str(args.opencood_config), None)
    _configure_hypes(hypes, args)
    dataset = build_dataset(hypes, visualize=False, train=False)
    if (args.scenario_split_file is None) != (args.scenario_role is None):
        raise ValueError(
            "--scenario-split-file and --scenario-role must be used together"
        )
    selected_scenarios = None
    evaluation_dataset = dataset
    if args.scenario_split_file is not None:
        split = json.loads(args.scenario_split_file.read_text())
        selected_scenarios = set(split[f"{args.scenario_role}_scenarios"])
        indices = [
            index
            for index in range(len(dataset))
            if _scenario_id(dataset, index) in selected_scenarios
        ]
        evaluation_dataset = Subset(dataset, indices)
    loader = DataLoader(
        evaluation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=dataset.collate_batch_test,
        pin_memory=args.device == "cuda",
        persistent_workers=args.workers > 0,
    )
    detector = create_model(hypes)
    detector_checkpoint = _load_detector_checkpoint(args.detector_dir, detector)
    model = FrozenPointPillarWithUncertainty(
        detector,
        anchor_count=int(hypes["model"]["args"]["anchor_number"]),
        hidden_channels=args.hidden_channels,
    ).to(device)
    uncertainty_epoch = model.load_uncertainty_checkpoint(
        args.uncertainty_checkpoint, map_location="cpu"
    )
    model.eval()

    rows: list[dict[str, float | bool]] = []
    progress = tqdm(loader, desc="collecting BELT uncertainty")
    with torch.no_grad():
        for frame_index, batch in enumerate(progress):
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
            batch = to_device(batch, device)
            for cav_id, cav_content in batch.items():
                if cav_id == "ego":
                    continue
                output = model(cav_content)
                proposal = decode_local_proposals(
                    output, cav_content, dataset.post_processor
                )
                detection = proposal_uncertainty(
                    output,
                    proposal,
                    cav_content,
                    position_noise_std=args.position_noise_std,
                    heading_noise_std_deg=args.heading_noise_std_deg,
                )
                gt_mask = cav_content["object_bbx_mask"][0] > 0
                assignment = assign_proposals_to_ground_truth(
                    proposal["corners"],
                    cav_content["object_bbx_center"][0][gt_mask],
                    cav_content["object_ids"],
                    order=dataset.post_processor.params["order"],
                    minimum_iou=args.minimum_gt_iou,
                )
                diagonal = detection["covariances"].diagonal(dim1=-2, dim2=-1)
                position_std = diagonal[:, :2].mean(dim=-1).clamp_min(0).sqrt()
                heading_std_deg = torch.rad2deg(
                    diagonal[:, 6].clamp_min(0).sqrt()
                )
                for index in range(len(proposal["boxes"])):
                    rows.append(
                        {
                            "frame": frame_index,
                            "source_cav": str(cav_id),
                            "matched": assignment["gt_indices"][index].item() >= 0,
                            "class_uncertainty": float(
                                detection["class_uncertainty"][index].cpu()
                            ),
                            "position_std_m": float(position_std[index].cpu()),
                            "heading_std_deg": float(heading_std_deg[index].cpu()),
                            "proposal_score": float(proposal["scores"][index].cpu()),
                            "matched_iou": float(assignment["gt_ious"][index]),
                        }
                    )
            progress.set_postfix(proposals=len(rows))

    if not rows:
        raise RuntimeError("No source-CAV proposals were retained")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "source_proposal_uncertainty.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    matched = np.asarray([bool(row["matched"]) for row in rows])
    summary = {
        "split": args.split,
        "frames": int(frame_index + 1),
        "source_proposals": len(rows),
        "matched_proposals": int(matched.sum()),
        "unmatched_proposals": int((~matched).sum()),
        "minimum_gt_iou": args.minimum_gt_iou,
        "scenario_subset": (
            None
            if selected_scenarios is None
            else {
                "file": str(args.scenario_split_file.resolve()),
                "role": args.scenario_role,
                "scenario_count": len(selected_scenarios),
            }
        ),
        "detector_checkpoint": str(detector_checkpoint),
        "uncertainty_checkpoint": str(args.uncertainty_checkpoint.resolve()),
        "uncertainty_epoch": uncertainty_epoch,
        "by_measurement": {
            field: {
                "all": _summary(np.asarray([float(row[field]) for row in rows])),
                "matched": _summary(
                    np.asarray([float(row[field]) for row in rows if row["matched"]])
                ),
                "unmatched": _summary(
                    np.asarray([float(row[field]) for row in rows if not row["matched"]])
                ),
            }
            for field in ("class_uncertainty", "position_std_m", "heading_std_deg")
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot_pdf(
        rows,
        args.output_dir / "source_proposal_uncertainty_pdf.png",
        args.histogram_bins,
        args.minimum_gt_iou,
    )
    print("UNCERTAINTY", json.dumps(summary, sort_keys=True))
    print(args.output_dir / "source_proposal_uncertainty_pdf.png")


if __name__ == "__main__":
    main()
