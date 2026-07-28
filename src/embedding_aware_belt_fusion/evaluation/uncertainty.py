"""Evaluate trained anchor-level uncertainty heads on held-out OPV2V data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from embedding_aware_belt_fusion.integration.opencood_uncertainty import (
    FrozenPointPillarWithUncertainty,
)
from embedding_aware_belt_fusion.integration.train_uncertainty import (
    _load_detector_checkpoint,
)
from embedding_aware_belt_fusion.integration.uncertainty_loss import (
    _anchor_evidence,
    _anchor_regression,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opencood-config", type=Path, required=True)
    parser.add_argument("--detector-dir", type=Path, required=True)
    parser.add_argument("--uncertainty-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)

    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.tools.train_utils import create_model, to_device

    hypes = load_yaml(str(args.opencood_config), None)
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=dataset.collate_batch_train,
        shuffle=False,
        pin_memory=args.device == "cuda",
        drop_last=False,
    )

    detector = create_model(hypes)
    detector_checkpoint = _load_detector_checkpoint(args.detector_dir, detector)
    model = FrozenPointPillarWithUncertainty(
        detector,
        anchor_count=int(hypes["model"]["args"]["anchor_number"]),
        hidden_channels=args.hidden_channels,
    ).to(device)
    trained_epoch = model.load_uncertainty_checkpoint(
        args.uncertainty_checkpoint, map_location="cpu"
    )
    model.eval()

    totals = {
        "positive_count": 0.0,
        "valid_class_count": 0.0,
        "squared_error": 0.0,
        "absolute_error": 0.0,
        "predicted_variance": 0.0,
        "gaussian_nll": 0.0,
        "coverage_1sigma": 0.0,
        "coverage_1_96sigma": 0.0,
        "log_variance": 0.0,
        "log_variance_min": float("inf"),
        "log_variance_max": float("-inf"),
        "lower_saturation": 0.0,
        "upper_saturation": 0.0,
        "correct_class": 0.0,
        "true_positive": 0.0,
        "false_positive": 0.0,
        "false_negative": 0.0,
        "brier": 0.0,
        "positive_foreground_probability": 0.0,
        "negative_foreground_probability": 0.0,
        "class_uncertainty": 0.0,
        "positive_uncertainty": 0.0,
        "negative_uncertainty": 0.0,
        "positive_class_count": 0.0,
        "negative_class_count": 0.0,
    }

    with torch.no_grad():
        progress = tqdm(loader, desc="evaluating uncertainty")
        for batch_index, batch in enumerate(progress):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch = to_device(batch, device)
            output = model(batch["ego"])
            labels = batch["ego"]["label_dict"]

            positive = labels["pos_equal_one"].reshape(output["psm"].shape[0], -1) > 0
            negative = labels["neg_equal_one"].reshape(output["psm"].shape[0], -1) > 0
            valid = positive | negative

            mean = _anchor_regression(output["rm"])
            target = labels["targets"].reshape_as(mean)
            log_variance = _anchor_regression(output["reg_log_var"])
            variance = torch.exp(log_variance)
            sigma = torch.sqrt(variance)
            error = target - mean
            positive_code_mask = positive.unsqueeze(-1).expand_as(error)
            selected_error = error[positive_code_mask]
            selected_log_variance = log_variance[positive_code_mask]
            selected_variance = variance[positive_code_mask]
            selected_sigma = sigma[positive_code_mask]
            code_count = float(selected_error.numel())

            totals["positive_count"] += code_count
            totals["squared_error"] += float(selected_error.square().sum())
            totals["absolute_error"] += float(selected_error.abs().sum())
            totals["predicted_variance"] += float(selected_variance.sum())
            nll = 0.5 * (
                selected_error.square() / selected_variance
                + selected_log_variance
                + math.log(2.0 * math.pi)
            )
            totals["gaussian_nll"] += float(nll.sum())
            totals["coverage_1sigma"] += float(
                (selected_error.abs() <= selected_sigma).sum()
            )
            totals["coverage_1_96sigma"] += float(
                (selected_error.abs() <= 1.96 * selected_sigma).sum()
            )
            totals["log_variance"] += float(selected_log_variance.sum())
            totals["log_variance_min"] = min(
                totals["log_variance_min"], float(selected_log_variance.min())
            )
            totals["log_variance_max"] = max(
                totals["log_variance_max"], float(selected_log_variance.max())
            )
            totals["lower_saturation"] += float(
                (selected_log_variance <= -9.999).sum()
            )
            totals["upper_saturation"] += float(
                (selected_log_variance >= 9.999).sum()
            )

            alpha = _anchor_evidence(output["alpha"])
            probabilities = alpha / alpha.sum(dim=-1, keepdim=True)
            foreground_probability = probabilities[..., 1]
            target_class = positive.to(foreground_probability.dtype)
            class_uncertainty = alpha.shape[-1] / alpha.sum(dim=-1)
            valid_count = float(valid.sum())
            totals["valid_class_count"] += valid_count
            totals["correct_class"] += float(
                ((foreground_probability >= 0.5) == positive)[valid].sum()
            )
            predicted_positive = foreground_probability >= 0.5
            totals["true_positive"] += float((predicted_positive & positive).sum())
            totals["false_positive"] += float((predicted_positive & negative).sum())
            totals["false_negative"] += float((~predicted_positive & positive).sum())
            totals["brier"] += float(
                ((foreground_probability - target_class).square() * valid).sum()
            )
            totals["positive_foreground_probability"] += float(
                (foreground_probability * positive).sum()
            )
            totals["negative_foreground_probability"] += float(
                (foreground_probability * negative).sum()
            )
            totals["class_uncertainty"] += float(
                (class_uncertainty * valid).sum()
            )
            totals["positive_uncertainty"] += float(
                (class_uncertainty * positive).sum()
            )
            totals["negative_uncertainty"] += float(
                (class_uncertainty * negative).sum()
            )
            totals["positive_class_count"] += float(positive.sum())
            totals["negative_class_count"] += float(negative.sum())

    code_count = totals["positive_count"]
    valid_count = totals["valid_class_count"]
    precision_denominator = totals["true_positive"] + totals["false_positive"]
    recall_denominator = totals["true_positive"] + totals["false_negative"]
    report = {
        "trained_epoch": trained_epoch,
        "detector_checkpoint": str(detector_checkpoint.resolve()),
        "uncertainty_checkpoint": str(args.uncertainty_checkpoint.resolve()),
        "evaluated_batches": min(len(loader), args.max_batches or len(loader)),
        "regression_encoded_anchor_space": {
            "code_count": int(code_count),
            "rmse": math.sqrt(totals["squared_error"] / code_count),
            "mae": totals["absolute_error"] / code_count,
            "mean_predicted_variance": totals["predicted_variance"] / code_count,
            "mean_log_variance": totals["log_variance"] / code_count,
            "min_log_variance": totals["log_variance_min"],
            "max_log_variance": totals["log_variance_max"],
            "gaussian_nll_per_code": totals["gaussian_nll"] / code_count,
            "coverage_1sigma": totals["coverage_1sigma"] / code_count,
            "coverage_1_96sigma": totals["coverage_1_96sigma"] / code_count,
            "fraction_at_min_log_variance": totals["lower_saturation"] / code_count,
            "fraction_at_max_log_variance": totals["upper_saturation"] / code_count,
        },
        "classification": {
            "valid_anchor_count": int(valid_count),
            "accuracy_at_0_5": totals["correct_class"] / valid_count,
            "precision_at_0_5": totals["true_positive"]
            / max(1.0, precision_denominator),
            "recall_at_0_5": totals["true_positive"] / max(1.0, recall_denominator),
            "brier_score": totals["brier"] / valid_count,
            "positive_mean_foreground_probability": totals[
                "positive_foreground_probability"
            ]
            / totals["positive_class_count"],
            "negative_mean_foreground_probability": totals[
                "negative_foreground_probability"
            ]
            / totals["negative_class_count"],
            "mean_uncertainty": totals["class_uncertainty"] / valid_count,
            "positive_mean_uncertainty": totals["positive_uncertainty"]
            / totals["positive_class_count"],
            "negative_mean_uncertainty": totals["negative_uncertainty"]
            / totals["negative_class_count"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
