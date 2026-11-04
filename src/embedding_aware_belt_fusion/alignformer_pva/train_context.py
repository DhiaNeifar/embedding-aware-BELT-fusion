"""Train AlignFormerBWithContext (CAV-CAV context cross-attention).

Additive, new entrypoint: does not modify ``train.py``. Reuses, unchanged,
every piece of its training-loop machinery that has no architecture-specific
assumption baked in -- ``parse_args``, ``build_datasets`` (wrapped, not
reimplemented; see ``dataset_frames.py``), ``_variance_model``,
``build_embedding_encoder``, ``_trainable_parameters``, ``_to_device``,
``_forward_and_loss``, ``evaluate``, ``_format``, ``train_one_epoch``,
``_loaders``, ``_checkpoint`` -- confirmed, before writing this file, that
none of them assume ``AlignFormerB`` specifically rather than any model
sharing its ``forward(batch) -> PoseEstimate`` contract. Only the dataset
wrapping, collate function, and model class differ from ``train.py:train``.

Geometry-only (``embed_dim=0``) only, deliberately: the plan this
implements chose the geometric joint-attention fix specifically because
Chenyi's own appearance ablation showed a well-separated appearance signal
*costing* AP when fed into the matcher. ``--embed-dim > 0`` is rejected
outright rather than silently mismatched against a cache/collate path this
class was never built or tested against.
"""

from __future__ import annotations

import torch

from embedding_aware_belt_fusion.alignformer_pva.dataset_frames import (
    AlignFormerPVAFrameDataset,
    collate_frames,
)
from embedding_aware_belt_fusion.alignformer_pva.model_context import (
    AlignFormerBWithContext,
    _PIPELINE_TAG,
)
from embedding_aware_belt_fusion.alignformer_pva.train import (
    _checkpoint,
    _format,
    _loaders,
    _trainable_parameters,
    _variance_model,
    build_datasets,
    build_embedding_encoder,
    evaluate,
    parse_args,
    train_one_epoch,
)

import json
import math


def build_model_context(args, device: torch.device) -> AlignFormerBWithContext:
    model = AlignFormerBWithContext(
        embed_dim=args.embed_dim,
        heading_lambda=args.heading_lambda,
        sinkhorn_iterations=args.sinkhorn_iterations,
        variance_model=_variance_model(args),
        model_dim=args.model_dim,
        layers=args.layers,
        heads=args.heads,
    )
    return model.to(device)


def train(args) -> "Path":  # noqa: F821 - Path only used for the return annotation
    if args.embed_dim > 0:
        raise ValueError(
            "train_context.py only supports --embed-dim 0 (geometry-only); "
            "AlignFormerBWithContext/collate_frames have never been built or "
            "tested against appearance embeddings, and the plan this "
            "implements deliberately avoided that direction (Chenyi's own "
            "appearance ablation cost 0.011 AP when fed into the matcher)"
        )
    if args.workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_train_set, base_validation_set = build_datasets(args)
    train_set = AlignFormerPVAFrameDataset(base_train_set)
    validation_set = AlignFormerPVAFrameDataset(base_validation_set)
    train_loader, validation_loader = _loaders(train_set, validation_set, args, collate_frames)
    print(
        f"alignformer_pva_context: {len(train_set)} train pairs / "
        f"{len(validation_set)} validation pairs (geometry-only + CAV context, embed_dim=0)",
        flush=True,
    )

    model = build_model_context(args, device)
    embedding_encoder = build_embedding_encoder(args, device)  # None at embed_dim=0
    optimizer = torch.optim.AdamW(
        _trainable_parameters(model, embedding_encoder),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate / 100
    )

    (args.output_dir / "config.json").write_text(
        json.dumps({key: str(value) for key, value in vars(args).items()}, indent=2) + "\n"
    )

    history = []
    best = {"position_error_m": math.inf, "epoch": 0}
    best_path = args.output_dir / "best.pth"
    stale = 0

    for epoch in range(args.epochs):
        train_set.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, args,
            epoch=epoch, embedding_encoder=embedding_encoder,
        )
        validation_metrics = evaluate(
            model, validation_loader, device, args, embedding_encoder=embedding_encoder
        )
        scheduler.step()

        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(record)
        print(
            f"[ep{epoch + 1}/{args.epochs}] train {_format(train_metrics)} | "
            f"val {_format(validation_metrics)}",
            flush=True,
        )
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")

        checkpoint = _checkpoint(model, args, epoch + 1, validation_metrics, embedding_encoder)
        checkpoint["pipeline"] = _PIPELINE_TAG  # "alignformer_pva_context_v1", not train.py's tag
        torch.save(checkpoint, args.output_dir / "latest.pth")
        if validation_metrics["position_error_m"] < best["position_error_m"]:
            best = {"position_error_m": validation_metrics["position_error_m"], "epoch": epoch + 1}
            torch.save(checkpoint, best_path)
            stale = 0
            print(f"  -> new best val position error {best['position_error_m']:.4f}m", flush=True)
        else:
            stale += 1
        if args.early_stopping_patience and stale >= args.early_stopping_patience:
            print("Early stopping on validation position error", flush=True)
            break

    print(
        f"\nbest val position error {best['position_error_m']:.4f}m at epoch "
        f"{best['epoch']}; checkpoint {best_path}",
        flush=True,
    )
    return best_path


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
