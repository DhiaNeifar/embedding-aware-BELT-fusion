import math
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from embedding_aware_belt_fusion.alignformer_pva.dataset import (
    AlignFormerPVAContextPairDataset,
    NoiseSchedule,
    collate_context,
)
from embedding_aware_belt_fusion.alignformer_pva.model import AlignFormerB
from embedding_aware_belt_fusion.alignformer_pva.train import (
    _checkpoint,
    _compute_side_embeddings,
    _forward_and_loss,
    _trainable_parameters,
    _with_computed_embeddings,
    build_embedding_encoder,
    evaluate,
    load_alignformer_pva,
    train_one_epoch,
)
from embedding_aware_belt_fusion.alignformer_pva.variance import UNWEIGHTED
from embedding_aware_belt_fusion.embeddings.context_embedding import ContextEmbedding

INPUT_DIM = 6


def _planar_transform(yaw, tx, ty):
    matrix = torch.eye(4)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    matrix[0, 0], matrix[0, 1] = cosine, -sine
    matrix[1, 0], matrix[1, 1] = sine, cosine
    matrix[0, 3], matrix[1, 3] = tx, ty
    return matrix


def _context_agent(centers, yaws, gt_ids, *, max_cells=4, transformation_matrix=None, seed=0):
    n = len(centers)
    boxes = torch.zeros(n, 7)
    if n:
        boxes[:, 0:2] = torch.tensor(centers, dtype=torch.float32)
        boxes[:, 3] = 1.5
        boxes[:, 4] = 1.5
        boxes[:, 5] = 4.0
        boxes[:, 6] = torch.tensor(yaws, dtype=torch.float32)
    scores = torch.linspace(0.9, 0.5, n) if n else torch.zeros(0)
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randn(n, max_cells, INPUT_DIM, generator=generator)
    positions = torch.randn(n, max_cells, 2, generator=generator)
    footprint_flag = torch.zeros(n, max_cells, dtype=torch.bool)
    footprint_flag[:, :1] = True
    mask = torch.zeros(n, max_cells, dtype=torch.bool)
    if n and max_cells > 2:
        mask[:, -1] = True
    return {
        "boxes": boxes,
        "scores": scores,
        "gt_ids": list(gt_ids),
        "transformation_matrix": (
            transformation_matrix if transformation_matrix is not None else torch.eye(4)
        ),
        "tokens": tokens,
        "positions": positions,
        "footprint_flag": footprint_flag,
        "mask": mask,
    }


def _synthetic_context_frames():
    frame_a = {
        "agents": [
            _context_agent(
                [[1.0, 1.0], [4.0, -2.0], [-3.0, 2.0]], [0.1, 0.4, -0.2], ["o1", "o2", "o3"], seed=1
            ),
            _context_agent(
                [[0.0, 0.0], [2.0, 3.0], [-1.0, -1.0]],
                [0.0, 0.5, -0.3],
                ["o2", "o3", None],
                transformation_matrix=_planar_transform(0.1, 1.0, 0.5),
                seed=2,
            ),
        ]
    }
    frame_b = {
        "agents": [
            _context_agent(
                [[5.0, 0.0], [1.0, 1.0], [-3.0, 2.0]], [0.0, 0.2, 1.0], ["p1", "p2", "p3"], seed=3
            ),
            _context_agent(
                [[0.5, 0.5], [4.5, -0.4]],
                [0.0, 0.9],
                ["p1", "p2"],
                transformation_matrix=_planar_transform(-0.2, -0.5, 1.5),
                seed=4,
            ),
        ]
    }
    return [frame_a, frame_b]


def _loader(generator_seed, xy_std=0.3, yaw_std_deg=5.0):
    dataset = AlignFormerPVAContextPairDataset(
        _synthetic_context_frames(),
        noise_schedule=NoiseSchedule.fixed(xy_std, yaw_std_deg),
        generator=torch.Generator().manual_seed(generator_seed),
    )
    return DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_context)


def _args(embed_dim=5, match_weight=1.0, pose_weight=1.0, epochs=1):
    return SimpleNamespace(
        embed_dim=embed_dim,
        embedding_d_model=16,
        embedding_layers=1,
        embedding_heads=2,
        embedding_feedforward_dim=32,
        embedding_dropout=0.0,
        match_weight=match_weight,
        pose_weight=pose_weight,
        epochs=epochs,
        heading_lambda=2.0,
        sinkhorn_iterations=20,
        model_dim=16,
        layers=1,
        heads=2,
    )


def _model(embed_dim=5):
    torch.manual_seed(0)
    return AlignFormerB(
        embed_dim=embed_dim, layers=1, heads=2, model_dim=16, variance_model=UNWEIGHTED
    )


def _encoder(args):
    torch.manual_seed(1)
    return ContextEmbedding(
        input_dim=INPUT_DIM,
        d_model=args.embedding_d_model,
        embed_dim=args.embed_dim,
        heads=args.embedding_heads,
        layers=args.embedding_layers,
        feedforward_dim=args.embedding_feedforward_dim,
        dropout=args.embedding_dropout,
    )


def test_build_embedding_encoder_returns_none_for_embed_dim_zero():
    args = SimpleNamespace(embed_dim=0)
    assert build_embedding_encoder(args, torch.device("cpu")) is None


def test_compute_side_embeddings_shape_matches_object_batch():
    batch = next(iter(_loader(generator_seed=0)))
    args = _args()
    encoder = _encoder(args)

    ego_embeddings = _compute_side_embeddings(encoder, batch, "ego")

    assert ego_embeddings.shape == (
        batch["ego_boxes"].shape[0],
        batch["ego_boxes"].shape[1],
        args.embed_dim,
    )
    assert torch.isfinite(ego_embeddings).all()


def test_with_computed_embeddings_overwrites_placeholders_and_is_noop_without_encoder():
    batch = next(iter(_loader(generator_seed=0)))
    assert batch["ego_embeddings"].shape[-1] == 0  # the dataset's own placeholder

    args = _args()
    encoder = _encoder(args)
    updated = _with_computed_embeddings(encoder, batch)
    assert updated["ego_embeddings"].shape[-1] == args.embed_dim
    assert updated["cav_embeddings"].shape[-1] == args.embed_dim
    # Original batch dict is untouched (a fresh dict is returned).
    assert batch["ego_embeddings"].shape[-1] == 0

    noop = _with_computed_embeddings(None, batch)
    assert noop is batch


def test_forward_and_loss_runs_with_embedding_encoder():
    args = _args()
    model = _model(args.embed_dim)
    encoder = _encoder(args)
    batch = next(iter(_loader(generator_seed=0)))

    result = _forward_and_loss(model, batch, args, encoder)

    assert result is not None
    estimate, match_loss, pose_loss, total_loss = result
    assert torch.isfinite(match_loss)
    assert torch.isfinite(pose_loss)
    assert torch.isfinite(total_loss)


def test_trainable_parameters_includes_encoder_params_and_optimizer_updates_both():
    args = _args()
    model = _model(args.embed_dim)
    encoder = _encoder(args)
    parameters = _trainable_parameters(model, encoder)
    assert len(parameters) == len(list(model.parameters())) + len(list(encoder.parameters()))

    before_model = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    before_encoder = {name: tensor.clone() for name, tensor in encoder.state_dict().items()}
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)

    metrics = train_one_epoch(
        model, _loader(generator_seed=1), optimizer, torch.device("cpu"), args,
        epoch=0, embedding_encoder=encoder,
    )

    assert metrics["pairs"] == 2
    assert metrics["skipped_batches"] == 0

    model_changed = any(
        not torch.equal(before_model[name], tensor)
        for name, tensor in model.state_dict().items()
        if tensor.dtype.is_floating_point
    )
    encoder_changed = any(
        not torch.equal(before_encoder[name], tensor)
        for name, tensor in encoder.state_dict().items()
        if tensor.dtype.is_floating_point
    )
    assert model_changed, "trunk/head should have been updated"
    assert encoder_changed, "embedding encoder should have been updated too"


def test_evaluate_runs_with_embedding_encoder_in_eval_mode():
    args = _args()
    model = _model(args.embed_dim)
    encoder = _encoder(args)

    metrics = evaluate(model, _loader(generator_seed=2), torch.device("cpu"), args, encoder)

    assert math.isfinite(metrics["total_loss"])
    assert not encoder.training
    assert not model.training


def test_checkpoint_round_trip_with_embedding_encoder_preserves_model_kwargs():
    args = _args()
    model = _model(args.embed_dim)
    encoder = _encoder(args)

    checkpoint = _checkpoint(model, args, epoch=1, metrics={"position_error_m": 0.5}, embedding_encoder=encoder)

    assert checkpoint["model_kwargs"]["embed_dim"] == args.embed_dim
    assert "embedding_state_dict" in checkpoint
    assert checkpoint["embedding_kwargs"]["embed_dim"] == args.embed_dim

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.pth"
        torch.save(checkpoint, path)
        loaded, saved = load_alignformer_pva(path, torch.device("cpu"))

    assert saved["epoch"] == 1
    # load_alignformer_pva reconstructs AlignFormerB generically from
    # model_kwargs alone, confirming the existing eval-harness integration's
    # checkpoint-driven reconstruction already handles a non-zero embed_dim
    # without needing any change there.
    assert loaded.trunk.input_projection.in_features == 9 + args.embed_dim


def test_checkpoint_without_embedding_encoder_matches_original_geometry_only_shape():
    # The original, embed_dim-less call signature (as in
    # tests/test_alignformer_pva_train.py) must still work unchanged.
    model = _model(embed_dim=0)
    checkpoint = _checkpoint(
        model,
        SimpleNamespace(heading_lambda=2.0, sinkhorn_iterations=20, model_dim=16, layers=1, heads=2),
        epoch=1,
        metrics={"position_error_m": 0.5},
    )
    assert checkpoint["model_kwargs"]["embed_dim"] == 0
    assert "embedding_state_dict" not in checkpoint
