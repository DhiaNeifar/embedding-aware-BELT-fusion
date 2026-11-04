"""Train the Propose-Verify-Abstain pose head (``AlignFormerB``) on our own data.

Single-stage, unlike the source repo's two-stage curriculum: that split exists
to warm-start a learned per-object *embedding* before the pose head trusts it.
By default this backend carries no embeddings at all (``embed_dim=0`` --
the source repo's own ablation found box geometry alone resolves 99.4% of
associations and that appearance cues do not help), so there is no embedding
head to pre-train and nothing stage 2 would warm-start from. One loop jointly
trains the trunk + Sinkhorn matcher (``match_nll``) and the pose solve
(``corner_loss``).

``--embed-dim > 0`` (Phase 1 of the richer-embedding plan) opts into a
richer, context-padded per-object embedding instead
(``embeddings/context_embedding.py:ContextEmbedding``), fed by a
context-padded cache (``data/materialize_context_roi_cache.py``). Still
single-stage: the embedding encoder's parameters join the trunk's in the SAME
optimizer and the SAME ``match_nll``/``corner_loss`` objective, with no
separate warm-start stage of its own, since (unlike the source repo's
appearance embedding) this one has no existing checkpoint to pre-train.

Data comes entirely from our own cache (``data/complete_cache.py`` +
``embeddings/all_agent_association.py:records_for_role``), wrapped by
``alignformer_pva/dataset.py``'s ``AlignFormerPVAPairDataset`` -- not from the
source repo's ``coloca``/``alignformer.cache`` stack, which this codebase does
not have and does not need.

No AMP: the source repo's own training explicitly avoids fp16 because
Sinkhorn's log-domain normalization and the Kabsch solve's ``atan2`` are
numerically sensitive at that precision, and this model is small enough
(a few million parameters, at most 64 tokens per side) that the run is
data-loader bound regardless.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from embedding_aware_belt_fusion.alignformer_pva.dataset import (
    AlignFormerPVAContextPairDataset,
    AlignFormerPVAPairDataset,
    NoiseSchedule,
    collate,
    collate_context,
)
from embedding_aware_belt_fusion.alignformer_pva.losses import corner_loss, match_nll
from embedding_aware_belt_fusion.alignformer_pva.model import (
    DEFAULT_HEADING_LAMBDA,
    AlignFormerB,
)
from embedding_aware_belt_fusion.alignformer_pva.head_match import (
    DEFAULT_SINKHORN_ITERATIONS,
)
from embedding_aware_belt_fusion.alignformer_pva.trunk import (
    DEFAULT_HEADS,
    DEFAULT_LAYERS,
    DEFAULT_MODEL_DIM,
)
from embedding_aware_belt_fusion.alignformer_pva.variance import (
    UNWEIGHTED,
    CorrespondenceVarianceModel,
    variance_model_from_config,
)
from embedding_aware_belt_fusion.data.complete_cache import IndexedCompleteFrameDataset
from embedding_aware_belt_fusion.embeddings.all_agent_association import records_for_role
from embedding_aware_belt_fusion.embeddings.context_embedding import ContextEmbedding

GRAD_CLIP = 5.0
_PROGRESS_INTERVAL = 100
_PIPELINE_TAG = "alignformer_pva_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--validation-cache-dir",
        type=Path,
        help="Optional separately materialized validation cache; defaults to --cache-dir.",
    )
    parser.add_argument("--scenario-split-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index-workers", type=int, default=8)
    parser.add_argument("--max-train-frames", type=int)
    parser.add_argument("--max-validation-frames", type=int)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--match-weight", type=float, default=1.0)
    parser.add_argument("--pose-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--early-stopping-patience", type=int, default=5)

    parser.add_argument("--heading-lambda", type=float, default=DEFAULT_HEADING_LAMBDA)
    parser.add_argument(
        "--sinkhorn-iterations", type=int, default=DEFAULT_SINKHORN_ITERATIONS
    )
    parser.add_argument("--model-dim", type=int, default=DEFAULT_MODEL_DIM)
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)

    parser.add_argument(
        "--xy-noise-std-start", type=float, default=0.0,
        help="Injected relative-pose translation noise std (metres) at epoch 0.",
    )
    parser.add_argument(
        "--xy-noise-std-end", type=float, default=2.0,
        help="Injected relative-pose translation noise std (metres) at the end of the ramp.",
    )
    parser.add_argument("--yaw-noise-std-start-deg", type=float, default=0.0)
    parser.add_argument("--yaw-noise-std-end-deg", type=float, default=10.0)
    parser.add_argument(
        "--noise-ramp-epochs", type=int, default=None,
        help="Epochs to linearly reach the end noise std; defaults to all --epochs.",
    )

    parser.add_argument(
        "--embed-dim", type=int, default=0,
        help=(
            "Per-object embedding width concatenated onto the 9-D geometry "
            "vector the trunk already consumes (alignformer_pva/trunk.py: "
            "tokenize). 0 (the default) is the original geometry-only path, "
            "completely unchanged: --cache-dir is read as "
            "AlignFormerPVAPairDataset always has, with zero-width "
            "embeddings. >0 switches --cache-dir to be read as a "
            "context-padded cache (data/materialize_context_roi_cache.py's "
            "output) via AlignFormerPVAContextPairDataset, and jointly "
            "trains a ContextEmbedding encoder (embeddings/"
            "context_embedding.py) whose output width must equal this flag."
        ),
    )
    parser.add_argument(
        "--embedding-d-model", type=int, default=128,
        help="ContextEmbedding's internal attention width (only used when --embed-dim > 0).",
    )
    parser.add_argument("--embedding-layers", type=int, default=2)
    parser.add_argument("--embedding-heads", type=int, default=4)
    parser.add_argument("--embedding-feedforward-dim", type=int, default=256)
    parser.add_argument("--embedding-dropout", type=float, default=0.1)

    parser.add_argument(
        "--variance-config", type=Path, default=None,
        help=(
            "Optional JSON produced by alignformer_pva/calibrate_variance.py: "
            "a {'correspondence_variance': {sigma_translation_m, "
            "translation_exponent, sigma_yaw_rad/sigma_yaw_deg, yaw_exponent, "
            "score_reference}} block fitted against our own detector's "
            "validation-role correspondence disagreement. When given, builds "
            "a mode='split' CorrespondenceVarianceModel (separate centre/"
            "heading exponents) instead of the default UNWEIGHTED estimator. "
            "Purely additive: omitting this flag reproduces the original "
            "geometry-only, unweighted training path exactly."
        ),
    )

    return parser.parse_args()


def _limit(records, maximum):
    return records if maximum is None else records[:maximum]


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _dataset_class(args: argparse.Namespace) -> type:
    """The geometry-only dataset by default; the context-padded variant when
    ``--embed-dim > 0``. Kept as its own one-line function so ``build_datasets``
    and any future caller agree on exactly one place this choice is made."""
    return AlignFormerPVAContextPairDataset if args.embed_dim > 0 else AlignFormerPVAPairDataset


def build_datasets(
    args: argparse.Namespace,
) -> Tuple[AlignFormerPVAPairDataset, AlignFormerPVAPairDataset]:
    """Build the train and validation pair datasets from our own complete cache.

    ``--embed-dim > 0`` switches both to ``AlignFormerPVAContextPairDataset``
    (:func:`_dataset_class`), which expects ``--cache-dir``/
    ``--validation-cache-dir`` to point at a context-padded cache
    (``data/materialize_context_roi_cache.py``'s output, a strict superset of
    the geometry-only cache's own fields) rather than the plain geometry-only
    one -- everything else below (the scenario-role split, the noise
    schedule, the frame indexing) is identical either way.
    """
    train_records = _limit(
        records_for_role(args.cache_dir, args.scenario_split_file, "train", args.index_workers),
        args.max_train_frames,
    )
    validation_cache_dir = args.validation_cache_dir or args.cache_dir
    validation_records = _limit(
        records_for_role(
            validation_cache_dir, args.scenario_split_file, "validation", args.index_workers
        ),
        args.max_validation_frames,
    )
    train_frames = IndexedCompleteFrameDataset(args.cache_dir, train_records)
    validation_frames = IndexedCompleteFrameDataset(validation_cache_dir, validation_records)

    ramp_epochs = args.noise_ramp_epochs or args.epochs
    train_schedule = NoiseSchedule(
        xy_std_start=args.xy_noise_std_start,
        xy_std_end=args.xy_noise_std_end,
        yaw_std_start_deg=args.yaw_noise_std_start_deg,
        yaw_std_end_deg=args.yaw_noise_std_end_deg,
        ramp_epochs=ramp_epochs,
    )
    # Validation is pinned at the curriculum's MAXIMUM noise, not the current
    # epoch's: a moving evaluation target makes the per-epoch curve
    # uninterpretable and would report the easiest condition at epoch 0.
    validation_schedule = NoiseSchedule.fixed(args.xy_noise_std_end, args.yaw_noise_std_end_deg)

    dataset_class = _dataset_class(args)
    train_set = dataset_class(
        train_frames,
        noise_schedule=train_schedule,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_set = dataset_class(
        validation_frames,
        noise_schedule=validation_schedule,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    return train_set, validation_set


def _variance_model(args: argparse.Namespace) -> CorrespondenceVarianceModel:
    """``UNWEIGHTED`` by default; a calibrated ``mode="split"`` model when
    ``--variance-config`` points at a ``calibrate_variance.py`` JSON output
    (Step 3 of the plan: fit the heteroscedastic noise model against our own
    detector's score-to-error relationship, replacing ``UNWEIGHTED``).

    ``getattr`` with a default so a caller's ``args`` namespace that predates
    this flag (e.g. ``tests/test_alignformer_pva_train.py``'s own
    ``SimpleNamespace`` fixtures) still gets the original UNWEIGHTED behaviour
    instead of an ``AttributeError`` -- the same backward-compatibility
    convention :func:`_checkpoint` already uses for ``embed_dim``.

    Reuses :func:`variance.variance_model_from_config` (forcing
    ``mode="split"``, the two-separate-exponent model the task calls for)
    rather than hand-rolling a second JSON-to-dataclass loader.
    """
    path = getattr(args, "variance_config", None)
    if path is None:
        return UNWEIGHTED
    config = json.loads(Path(path).read_text())
    return variance_model_from_config(config, mode="split")


def build_model(args: argparse.Namespace, device: torch.device) -> AlignFormerB:
    """An ``AlignFormerB``, with :func:`_variance_model` deciding whether its
    Procrustes weighting is the uncalibrated identity estimator or a
    calibrated inverse-variance one.

    ``embed_dim`` defaults to 0 (geometry-only -- the source repo's own
    ablation found box geometry alone resolves 99.4% of associations and that
    appearance cues do not help there), but accepts ``--embed-dim > 0`` to
    train against a richer per-object embedding instead (Phase 1 of the
    richer-embedding plan); see :func:`build_embedding_encoder` for the
    encoder that produces it.
    """
    model = AlignFormerB(
        embed_dim=args.embed_dim,
        heading_lambda=args.heading_lambda,
        sinkhorn_iterations=args.sinkhorn_iterations,
        variance_model=_variance_model(args),
        model_dim=args.model_dim,
        layers=args.layers,
        heads=args.heads,
    )
    return model.to(device)


def build_embedding_encoder(
    args: argparse.Namespace, device: torch.device
) -> Optional[ContextEmbedding]:
    """The context-padded embedding encoder, or ``None`` on the geometry-only path.

    ``input_dim`` is left at :class:`ContextEmbedding`'s own default (384),
    matching every cache this repository currently materializes
    (``feature_dim: 384`` in every ``complete``/``context_roi_training``
    manifest); not exposed as a CLI flag since changing it would require a
    differently-extracted cache anyway, at which point this default should be
    revisited alongside it rather than silently overridden here.
    """
    if args.embed_dim <= 0:
        return None
    encoder = ContextEmbedding(
        d_model=args.embedding_d_model,
        embed_dim=args.embed_dim,
        heads=args.embedding_heads,
        layers=args.embedding_layers,
        feedforward_dim=args.embedding_feedforward_dim,
        dropout=args.embedding_dropout,
    )
    return encoder.to(device)


def _trainable_parameters(model: AlignFormerB, embedding_encoder: Optional[ContextEmbedding]):
    """The full parameter list the optimizer/grad-clipping should cover.

    A plain list (not a generator) so it can be iterated twice per step
    (once by the optimizer, once by ``clip_grad_norm_``) without exhausting
    anything -- both ``build_model``'s ``build_embedding_encoder``'s modules'
    ``.parameters()`` are themselves generators.
    """
    parameters = list(model.parameters())
    if embedding_encoder is not None:
        parameters += list(embedding_encoder.parameters())
    return parameters


def _compute_side_embeddings(
    encoder: ContextEmbedding, batch: Mapping[str, torch.Tensor], prefix: str
) -> torch.Tensor:
    """Run ``encoder`` over one side's ``(B, N, K, ...)`` padded-cell batch.

    ``ContextEmbedding.forward`` expects one flat proposal dimension (it
    processes a per-agent list of proposals, not a batch-of-agents), so the
    leading ``(batch, object)`` dims are flattened into it and restored
    afterwards. A fully object-padded slot (an object index beyond that
    sample's own real count) carries an all-``True`` cell mask
    (:func:`alignformer_pva.dataset._pad_cell_field`), which is exactly
    :class:`ContextEmbedding`'s own fully-masked-row case -- it returns a
    harmless, finite zero vector there rather than raising.
    """
    tokens = batch[f"{prefix}_cell_tokens"]
    batch_size, object_count, cell_count = tokens.shape[:3]
    flat = encoder(
        tokens.reshape(batch_size * object_count, cell_count, -1),
        batch[f"{prefix}_cell_positions"].reshape(batch_size * object_count, cell_count, 2),
        batch[f"{prefix}_cell_footprint"].reshape(batch_size * object_count, cell_count),
        batch[f"{prefix}_cell_mask"].reshape(batch_size * object_count, cell_count),
        batch[f"{prefix}_boxes"].reshape(batch_size * object_count, 7),
        batch[f"{prefix}_scores"].reshape(batch_size * object_count),
    )
    return flat.reshape(batch_size, object_count, -1)


def _with_computed_embeddings(
    embedding_encoder: Optional[ContextEmbedding], batch: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Overwrite the batch's placeholder embeddings with the encoder's real output.

    A no-op that returns ``batch`` unchanged when ``embedding_encoder is
    None`` -- the original geometry-only path, where the zero-width
    placeholders :func:`alignformer_pva.dataset.collate` already produced ARE
    the intended (empty) embeddings, not a placeholder awaiting replacement.
    """
    if embedding_encoder is None:
        return batch
    batch = dict(batch)
    batch["ego_embeddings"] = _compute_side_embeddings(embedding_encoder, batch, "ego")
    batch["cav_embeddings"] = _compute_side_embeddings(embedding_encoder, batch, "cav")
    return batch


def _to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


class _Tally:
    """Accumulates loss and pose/matching diagnostics over a whole split.

    Sums, not per-batch means: the last batch of a split is usually short and
    averaging per-batch means would weight it as heavily as a full one.
    """

    def __init__(self) -> None:
        self.match_loss_sum = 0.0
        self.pose_loss_sum = 0.0
        self.total_loss_sum = 0.0
        self.position_error_sum = 0.0
        self.heading_error_deg_sum = 0.0
        self.pairs = 0
        self.batches = 0
        self.skipped = 0

    def update(
        self,
        match_loss: float,
        pose_loss: float,
        total_loss: float,
        position_error: torch.Tensor,
        heading_error_deg: torch.Tensor,
    ) -> None:
        count = position_error.shape[0]
        self.match_loss_sum += match_loss * count
        self.pose_loss_sum += pose_loss * count
        self.total_loss_sum += total_loss * count
        self.position_error_sum += float(position_error.sum())
        self.heading_error_deg_sum += float(heading_error_deg.sum())
        self.pairs += count
        self.batches += 1

    def compute(self) -> Dict[str, float]:
        def ratio(numerator: float) -> float:
            return numerator / self.pairs if self.pairs else math.nan

        return {
            "match_nll": ratio(self.match_loss_sum),
            "corner_loss_m": ratio(self.pose_loss_sum),
            "total_loss": ratio(self.total_loss_sum),
            "position_error_m": ratio(self.position_error_sum),
            "heading_error_deg": ratio(self.heading_error_deg_sum),
            "pairs": float(self.pairs),
            "skipped_batches": float(self.skipped),
        }


def _forward_and_loss(
    model: AlignFormerB,
    batch: Mapping[str, torch.Tensor],
    args: argparse.Namespace,
    embedding_encoder: Optional[ContextEmbedding] = None,
):
    """Run one batch; returns ``(estimate, match_loss, pose_loss, total_loss)`` or ``None``.

    ``None`` means the batch had an empty object set on one side for every
    sample (``estimate.log_assignment is None``) -- there is no correspondence
    to supervise and nothing to score. This should not occur in practice
    (``AlignFormerPVAPairDataset`` only emits pairs where both agents detected
    at least one object), but the guard is cheap and matches the source
    repo's own training loop.

    ``embedding_encoder``, when given, replaces ``batch``'s zero-width
    ``ego_embeddings``/``cav_embeddings`` placeholders with its real output
    (:func:`_with_computed_embeddings`) before the model ever sees them --
    ``None`` (the default) is the original geometry-only path, unchanged.
    """
    batch = _with_computed_embeddings(embedding_encoder, batch)
    estimate = model(batch)
    if estimate.log_assignment is None:
        return None
    match_loss = match_nll(estimate.log_assignment, batch["ego_match"], batch["cav_match"])
    pose_loss = corner_loss(
        batch["cav_boxes"], estimate.psi, estimate.t,
        batch["psi_true"], batch["t_true"], batch["cav_mask"],
    )
    total_loss = args.match_weight * match_loss + args.pose_weight * pose_loss
    return estimate, match_loss, pose_loss, total_loss


@torch.no_grad()
def evaluate(
    model: AlignFormerB,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    embedding_encoder: Optional[ContextEmbedding] = None,
) -> Dict[str, float]:
    model.eval()
    if embedding_encoder is not None:
        embedding_encoder.eval()
    tally = _Tally()
    for batch in loader:
        batch = _to_device(batch, device)
        result = _forward_and_loss(model, batch, args, embedding_encoder)
        if result is None:
            tally.skipped += 1
            continue
        estimate, match_loss, pose_loss, total_loss = result
        position_error = (estimate.t - batch["t_true"]).norm(dim=-1)
        heading_error_deg = torch.rad2deg(
            _wrap_angle(estimate.psi - batch["psi_true"]).abs()
        )
        tally.update(
            float(match_loss), float(pose_loss), float(total_loss),
            position_error, heading_error_deg,
        )
    return tally.compute()


def _format(metrics: Mapping[str, float]) -> str:
    return (
        f"total={metrics['total_loss']:.4f} match_nll={metrics['match_nll']:.4f} "
        f"corner={metrics['corner_loss_m']:.4f}m pos_err={metrics['position_error_m']:.4f}m "
        f"yaw_err={metrics['heading_error_deg']:.3f}deg n={int(metrics['pairs'])}"
    )


def train_one_epoch(
    model: AlignFormerB,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    *,
    epoch: int,
    embedding_encoder: Optional[ContextEmbedding] = None,
) -> Dict[str, float]:
    model.train()
    if embedding_encoder is not None:
        embedding_encoder.train()
    tally = _Tally()
    started = time.time()
    for step, batch in enumerate(loader):
        batch = _to_device(batch, device)
        result = _forward_and_loss(model, batch, args, embedding_encoder)
        if result is None:
            tally.skipped += 1
            continue
        estimate, match_loss, pose_loss, total_loss = result
        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"non-finite training loss at epoch {epoch + 1} step {step}; check the "
                "NaN guards in trunk.py::_attend_without_degenerate_rows and "
                "model.py's atan2 has_weight guard before weakening anything"
            )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            _trainable_parameters(model, embedding_encoder), GRAD_CLIP
        )
        optimizer.step()

        with torch.no_grad():
            position_error = (estimate.t - batch["t_true"]).norm(dim=-1)
            heading_error_deg = torch.rad2deg(
                _wrap_angle(estimate.psi - batch["psi_true"]).abs()
            )
        tally.update(
            float(match_loss), float(pose_loss), float(total_loss),
            position_error, heading_error_deg,
        )
        if step % _PROGRESS_INTERVAL == 0:
            rate = (step + 1) / (time.time() - started)
            print(
                f"  [ep{epoch + 1}/{args.epochs}] step {step}/{len(loader)} "
                f"{_format(tally.compute())} {rate:.1f} it/s",
                flush=True,
            )
    return tally.compute()


def _loaders(
    train_set: AlignFormerPVAPairDataset,
    validation_set: AlignFormerPVAPairDataset,
    args: argparse.Namespace,
    collate_fn=collate,
) -> Tuple[DataLoader, DataLoader]:
    common = {"batch_size": args.batch_size, "num_workers": args.workers, "collate_fn": collate_fn}
    # Never persistent on the train loader: each worker holds its own dataset
    # copy, so set_epoch on the main process's copy would never reach them and
    # every epoch would silently reuse epoch 0's noise draw.
    train_loader = DataLoader(
        train_set, shuffle=True, drop_last=True, persistent_workers=False, **common
    )
    validation_loader = DataLoader(
        validation_set, shuffle=False, drop_last=False,
        persistent_workers=args.workers > 0, **common,
    )
    return train_loader, validation_loader


def _checkpoint(
    model: AlignFormerB,
    args: argparse.Namespace,
    epoch: int,
    metrics: Mapping[str, float],
    embedding_encoder: Optional[ContextEmbedding] = None,
) -> Dict[str, Any]:
    """Build the saved checkpoint dict.

    ``embed_dim`` is read via ``getattr(args, "embed_dim", 0)`` rather than
    ``args.embed_dim`` directly so a caller's ``args`` namespace that
    predates this flag (e.g. ``tests/test_alignformer_pva_train.py``'s own
    ``SimpleNamespace`` fixtures) still gets the original, correct
    geometry-only value instead of an ``AttributeError``.

    ``embedding_state_dict``/``embedding_kwargs`` are added only when
    ``embedding_encoder`` is given (``--embed-dim > 0``'s training path);
    ``load_alignformer_pva`` (below) never reads these extra keys -- it only
    ever reconstructs ``AlignFormerB`` from ``model_kwargs``, which is also
    exactly what the evaluation harness's ``--alignformer-pva-checkpoint``
    integration already does generically regardless of ``embed_dim``, since
    ``AlignFormerTrunk``'s input width is derived from it either way. The
    extra keys exist purely so THIS script can resume/inspect the embedding
    encoder's own weights later; they are additive and harmless to any
    reader that does not look for them.

    ``variance_model_kwargs`` is the resolved ``model.variance_model``'s own
    five ``CorrespondenceVarianceModel`` fields (whatever :func:`_variance_model`
    actually built -- ``UNWEIGHTED`` when ``--variance-config`` was omitted,
    the calibrated ``mode="split"`` model otherwise), stored unconditionally
    so :func:`load_alignformer_pva` can reconstruct the EXACT variance model
    training used with no separate eval-side flag and no train/eval mismatch
    possible -- the same "checkpoint carries its own reconstruction recipe"
    pattern ``model_kwargs`` already establishes.
    """
    checkpoint = {
        "pipeline": _PIPELINE_TAG,
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "model_kwargs": {
            "embed_dim": getattr(args, "embed_dim", 0),
            "heading_lambda": args.heading_lambda,
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "model_dim": args.model_dim,
            "layers": args.layers,
            "heads": args.heads,
        },
        "variance_model_kwargs": {
            "mode": model.variance_model.mode,
            "sigma_translation_m": model.variance_model.sigma_translation_m,
            "translation_exponent": model.variance_model.translation_exponent,
            "sigma_yaw_rad": model.variance_model.sigma_yaw_rad,
            "yaw_exponent": model.variance_model.yaw_exponent,
            "score_reference": model.variance_model.score_reference,
        },
        "validation_metrics": dict(metrics),
    }
    if embedding_encoder is not None:
        checkpoint["embedding_state_dict"] = embedding_encoder.state_dict()
        checkpoint["embedding_kwargs"] = {
            "input_dim": 384,
            "d_model": getattr(args, "embedding_d_model", 128),
            "embed_dim": getattr(args, "embed_dim", 0),
            "heads": getattr(args, "embedding_heads", 4),
            "layers": getattr(args, "embedding_layers", 2),
            "feedforward_dim": getattr(args, "embedding_feedforward_dim", 256),
            "dropout": getattr(args, "embedding_dropout", 0.1),
        }
    return checkpoint


def load_alignformer_pva(checkpoint_path: Path, device: torch.device) -> Tuple[AlignFormerB, dict]:
    """Rebuild an ``AlignFormerB`` from a checkpoint this script saved and load its weights.

    Reconstructs whichever variance model training actually used from the
    checkpoint's own ``variance_model_kwargs`` (see :func:`_checkpoint`), so a
    calibrated checkpoint evaluates weighted with no separate eval-side flag
    and no risk of a train/eval mismatch. Falls back to ``UNWEIGHTED`` for a
    checkpoint saved before this key existed -- every pre-existing
    geometry-only checkpoint (e.g. ``alignformer_pva_train_v1/best.pth``)
    loads exactly as it always has.
    """
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if checkpoint.get("pipeline") != _PIPELINE_TAG:
        raise ValueError(f"{checkpoint_path} is not an {_PIPELINE_TAG} checkpoint")
    variance_kwargs = checkpoint.get("variance_model_kwargs")
    variance_model = (
        CorrespondenceVarianceModel(**variance_kwargs)
        if variance_kwargs is not None
        else UNWEIGHTED
    )
    model = AlignFormerB(variance_model=variance_model, **checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def load_embedding_encoder(
    checkpoint_path: Path, device: torch.device
) -> Optional[ContextEmbedding]:
    """Rebuild the ``ContextEmbedding`` this script additively saved, if any.

    Returns ``None`` for a checkpoint trained with ``--embed-dim 0`` (no
    ``embedding_state_dict``/``embedding_kwargs`` keys -- see
    :func:`_checkpoint`) -- every geometry-only checkpoint produced before
    this function existed, and every one produced since with the default
    ``--embed-dim 0``, loads exactly as before through
    :func:`load_alignformer_pva` alone; this function is purely additive and
    a caller that never calls it (e.g. an existing test or script) is
    unaffected.

    Reads the checkpoint file a second time rather than extending
    ``load_alignformer_pva``'s return signature: evaluation only loads a
    checkpoint once per run, so the extra disk read is negligible, and
    keeping ``load_alignformer_pva`` itself unchanged avoids touching its
    existing callers/tests at all.
    """
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if checkpoint.get("pipeline") != _PIPELINE_TAG:
        raise ValueError(f"{checkpoint_path} is not an {_PIPELINE_TAG} checkpoint")
    if "embedding_state_dict" not in checkpoint:
        return None
    encoder = ContextEmbedding(**checkpoint["embedding_kwargs"]).to(device)
    encoder.load_state_dict(checkpoint["embedding_state_dict"])
    encoder.eval()
    return encoder


def train(args: argparse.Namespace) -> Path:
    if args.workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_set, validation_set = build_datasets(args)
    collate_fn = collate_context if args.embed_dim > 0 else collate
    train_loader, validation_loader = _loaders(train_set, validation_set, args, collate_fn)
    mode = f"context-padded embedding, embed_dim={args.embed_dim}" if args.embed_dim > 0 else (
        "geometry-only, embed_dim=0"
    )
    print(
        f"alignformer_pva: {len(train_set)} train pairs / {len(validation_set)} "
        f"validation pairs ({mode})",
        flush=True,
    )

    model = build_model(args, device)
    embedding_encoder = build_embedding_encoder(args, device)
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
    # Lower is better: mean translation error on the fixed-max-noise
    # validation set is the most directly interpretable proxy for "does this
    # correction actually help", ahead of corner_loss (mixes rotation+
    # translation in metres but convolves them with object distance) or
    # match_nll alone (says nothing about the solved pose).
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
