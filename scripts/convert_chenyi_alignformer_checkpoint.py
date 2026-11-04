#!/usr/bin/env python3
"""Convert Chenyi's stage2_B_matched_boxes_only checkpoint into our
alignformer_pva_v1 format, so --alignformer-pva-checkpoint can load his
actual trained weights instead of a checkpoint we trained ourselves.

Why this conversion is exact, not approximate
----------------------------------------------
Every key of his checkpoint's ``pose.*`` submodule matches ours one-to-one in
name AND shape except ``trunk.input_projection.weight``: his is (256, 137),
ours is (256, 9). The extra 128 columns are his embed_dim (an appearance
embedding). His ``stage2_B_matched_boxes_only`` run trained with
message_content="boxes_only" from stage 1 onward (stage1_checkpoint is
"stage1_zero_embeddings"), which routes embeddings through zero_embeddings()
before they ever reach tokenize() (embedding_aware_belt_fusion.alignformer.
train.embed_batch: "the boxes-only condition is literally the full model
with its embeddings set to zero"). tokenize() concatenates
[geometry(9), embeddings(128)] in that order (confirmed identical between
his trunk.py and ours), so columns 9:137 of his input_projection.weight
multiply an input that was exactly zero through the entire training run and
never affect any forward or backward pass. Slicing [:, :9] therefore
reproduces, bit for bit, the function his checkpoint actually computes --
it is not a reduced or re-fit model.

Usage
-----
    python scripts/convert_chenyi_alignformer_checkpoint.py \\
        --chenyi-checkpoint /path/to/stage2_B_matched_boxes_only/best.pth \\
        --output models/opv2v/alignformer_boxes_only.pth
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embedding_aware_belt_fusion.alignformer_pva.model import AlignFormerB
from embedding_aware_belt_fusion.alignformer_pva.variance import (
    CorrespondenceVarianceModel,
)

_GEOMETRY_FEATURES = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chenyi-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def convert(chenyi_checkpoint: Path) -> dict:
    checkpoint = torch.load(str(chenyi_checkpoint), map_location="cpu", weights_only=False)
    if checkpoint.get("message_content") != "boxes_only":
        raise ValueError(
            "This converter only handles a message_content='boxes_only' "
            f"checkpoint (embeddings provably zero throughout training); got "
            f"{checkpoint.get('message_content')!r}. A boxes+embeddings "
            "checkpoint would need the embedding head converted too, which "
            "this script does not attempt."
        )
    source_state = checkpoint["state_dict"]
    pose_prefix = "pose."
    remapped = {}
    for key, value in source_state.items():
        if not key.startswith(pose_prefix):
            continue  # drop embedding.* entirely: proven unused (see module docstring)
        remapped[key[len(pose_prefix):]] = value

    weight_key = "trunk.input_projection.weight"
    full_weight = remapped[weight_key]
    if full_weight.shape[1] <= _GEOMETRY_FEATURES:
        raise ValueError(
            f"{weight_key} has {full_weight.shape[1]} input columns, expected "
            f"more than {_GEOMETRY_FEATURES} (geometry + embedding); this "
            "checkpoint may already be geometry-only and need no slicing."
        )
    remapped[weight_key] = full_weight[:, :_GEOMETRY_FEATURES].contiguous()

    model_config = checkpoint["config"]["model"]
    model_kwargs = {
        "embed_dim": 0,
        "heading_lambda": float(model_config["heading_lambda"]),
        "sinkhorn_iterations": int(model_config["sinkhorn_iterations"]),
        "model_dim": int(model_config["model_dim"]),
        "layers": int(model_config["layers"]),
        "heads": int(model_config["heads"]),
    }

    variance_config = model_config["correspondence_variance"]
    variance_model_kwargs = {
        "mode": variance_config["mode"],
        "sigma_translation_m": float(variance_config["sigma_translation_m"]),
        "translation_exponent": float(variance_config["translation_exponent"]),
        "sigma_yaw_rad": math.radians(float(variance_config["sigma_yaw_deg"])),
        "yaw_exponent": float(variance_config["yaw_exponent"]),
        "score_reference": float(variance_config["score_reference"]),
    }

    # Build and strictly load: this IS the correctness check. Any shape or
    # name mismatch the analysis above missed raises here, loudly, before
    # anything is written to disk.
    variance_model = CorrespondenceVarianceModel(**variance_model_kwargs)
    model = AlignFormerB(variance_model=variance_model, **model_kwargs)
    missing, unexpected = model.load_state_dict(remapped, strict=True)
    assert not missing and not unexpected  # strict=True already raises; belt and suspenders

    # Smoke test: a tiny synthetic batch must run end to end and produce
    # finite outputs. Does not validate the numbers (needs Chenyi's own
    # validation split for that), only that the converted weights are wired
    # up correctly and the forward pass is numerically sane.
    model.eval()
    with torch.no_grad():
        batch = {
            "ego_boxes": torch.randn(1, 3, 7),
            "ego_scores": torch.rand(1, 3),
            "ego_embeddings": torch.zeros(1, 3, 0),
            "ego_mask": torch.ones(1, 3, dtype=torch.bool),
            "cav_boxes": torch.randn(1, 4, 7),
            "cav_scores": torch.rand(1, 4),
            "cav_embeddings": torch.zeros(1, 4, 0),
            "cav_mask": torch.ones(1, 4, dtype=torch.bool),
        }
        estimate = model(batch)
        assert torch.isfinite(estimate.psi).all()
        assert torch.isfinite(estimate.t).all()

    return {
        "pipeline": "alignformer_pva_v1",
        "state_dict": model.state_dict(),
        "model_kwargs": model_kwargs,
        "variance_model_kwargs": variance_model_kwargs,
        "epoch": checkpoint.get("epoch"),
        "source": {
            "converted_from": str(chenyi_checkpoint),
            "head": checkpoint.get("head"),
            "message_content": checkpoint.get("message_content"),
            "val_metrics": checkpoint.get("val_metrics"),
            "note": (
                "Weights are Chenyi's stage2_B_matched_boxes_only/best.pth "
                "verbatim (pose.* submodule only; embedding.* dropped since "
                "it is provably unused -- see this script's module "
                "docstring). Not retrained or recalibrated by us."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    converted = convert(args.chenyi_checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, args.output)
    print(f"Wrote {args.output}")
    print(f"  model_kwargs: {converted['model_kwargs']}")
    print(f"  variance_model_kwargs: {converted['variance_model_kwargs']}")
    print(f"  source val_metrics: {converted['source']['val_metrics']}")


if __name__ == "__main__":
    main()
