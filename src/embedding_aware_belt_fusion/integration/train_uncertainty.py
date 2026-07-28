"""CLI for training uncertainty heads on a frozen OpenCOOD PointPillar detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .opencood_uncertainty import FrozenPointPillarWithUncertainty
from .uncertainty_loss import AnchorUncertaintyLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opencood-config", type=Path, required=True)
    parser.add_argument("--detector-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--negative-weight", type=float, default=0.25)
    parser.add_argument("--validation-batches", type=int, default=100)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_detector_checkpoint(detector_dir: Path, detector: torch.nn.Module) -> Path:
    candidates = sorted(
        detector_dir.glob("net_epoch*.pth"),
        key=lambda path: int(path.stem.replace("net_epoch", "")),
    )
    latest = detector_dir / "latest.pth"
    checkpoint_path = latest if latest.exists() else candidates[-1] if candidates else None
    if checkpoint_path is None:
        raise FileNotFoundError(f"No PointPillar checkpoint found in {detector_dir}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = detector.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Detector checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    return checkpoint_path


def _make_loader(dataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=dataset.collate_batch_train,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=shuffle,
    )


def _move_to_device(batch, device: torch.device):
    from opencood.tools.train_utils import to_device

    return to_device(batch, device)


def _run_validation(
    model: FrozenPointPillarWithUncertainty,
    criterion: AnchorUncertaintyLoss,
    loader: Iterable,
    device: torch.device,
    epoch: int,
    max_batches: int,
) -> Dict[str, float]:
    model.eval()
    values: Dict[str, list] = {
        "loss": [],
        "regression_loss": [],
        "classification_loss": [],
        "kl_loss": [],
    }
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch = _move_to_device(batch, device)
            losses = criterion(
                model(batch["ego"]), batch["ego"]["label_dict"], epoch=epoch
            )
            for key in values:
                values[key].append(float(losses[key].item()))
    return {key: mean(items) for key, items in values.items() if items}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for practical PointPillar uncertainty training")
    device = torch.device("cuda")

    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.tools.train_utils import create_model

    hypes = load_yaml(str(args.opencood_config), None)
    if hypes["fusion"]["core_method"] != "LateFusionDataset":
        raise ValueError("Uncertainty Plan 0 currently requires LateFusionDataset")

    print("Building OPV2V train and validation datasets")
    train_dataset = build_dataset(hypes, visualize=False, train=True)
    validation_dataset = build_dataset(hypes, visualize=False, train=False)
    train_loader = _make_loader(
        train_dataset, args.batch_size, args.workers, shuffle=True
    )
    validation_loader = _make_loader(
        validation_dataset, args.batch_size, args.workers, shuffle=False
    )

    detector = create_model(hypes)
    detector_checkpoint = _load_detector_checkpoint(args.detector_dir, detector)
    anchor_count = int(hypes["model"]["args"]["anchor_number"])
    model = FrozenPointPillarWithUncertainty(
        detector=detector,
        anchor_count=anchor_count,
        hidden_channels=args.hidden_channels,
        freeze_detector=True,
    ).to(device)
    criterion = AnchorUncertaintyLoss(
        kl_weight=args.kl_weight,
        negative_weight=args.negative_weight,
        anneal_epochs=args.epochs,
    )
    optimizer = torch.optim.AdamW(
        model.uncertainty_heads.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )

    start_epoch = 0
    if args.resume:
        start_epoch = model.load_uncertainty_checkpoint(args.resume)
        resume_payload = torch.load(args.resume, map_location="cpu")
        if "optimizer_state_dict" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "opencood_config": str(args.opencood_config.resolve()),
        "detector_checkpoint": str(detector_checkpoint.resolve()),
        "anchor_count": anchor_count,
        "bev_channels": int(detector.backbone.num_bev_features),
        "arguments": vars(args),
    }
    metadata["arguments"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in metadata["arguments"].items()
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"uncertainty epoch {epoch + 1}/{args.epochs}")
        for batch_index, batch in enumerate(progress):
            if (
                args.max_train_batches is not None
                and batch_index >= args.max_train_batches
            ):
                break
            batch = _move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            losses = criterion(
                model(batch["ego"]), batch["ego"]["label_dict"], epoch=epoch
            )
            losses["loss"].backward()
            optimizer.step()
            progress.set_postfix(
                loss=f"{losses['loss'].item():.4f}",
                reg=f"{losses['regression_loss'].item():.4f}",
                cls=f"{losses['classification_loss'].item():.4f}",
            )
        scheduler.step()

        validation = _run_validation(
            model,
            criterion,
            validation_loader,
            device,
            epoch,
            args.validation_batches,
        )
        print("validation", json.dumps(validation, sort_keys=True))
        checkpoint_path = args.output_dir / f"uncertainty_epoch{epoch + 1}.pth"
        model.save_uncertainty_checkpoint(
            checkpoint_path, epoch=epoch + 1, optimizer=optimizer
        )
        (args.output_dir / f"validation_epoch{epoch + 1}.json").write_text(
            json.dumps(validation, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
