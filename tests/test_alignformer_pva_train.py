import json
import math
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from embedding_aware_belt_fusion.alignformer_pva.dataset import (
    AlignFormerPVAPairDataset,
    NoiseSchedule,
    collate,
)
from embedding_aware_belt_fusion.alignformer_pva.model import AlignFormerB
from embedding_aware_belt_fusion.alignformer_pva.train import (
    _checkpoint,
    _forward_and_loss,
    _variance_model,
    evaluate,
    load_alignformer_pva,
    train_one_epoch,
)
from embedding_aware_belt_fusion.alignformer_pva.variance import UNWEIGHTED


def _agent(centers, yaws, gt_ids, transformation_matrix=None):
    n = len(centers)
    boxes = torch.zeros(n, 7)
    if n:
        boxes[:, 0:2] = torch.tensor(centers, dtype=torch.float32)
        boxes[:, 3] = 1.5
        boxes[:, 4] = 1.5
        boxes[:, 5] = 4.0
        boxes[:, 6] = torch.tensor(yaws, dtype=torch.float32)
    scores = torch.linspace(0.9, 0.5, n) if n else torch.zeros(0)
    return {
        "boxes": boxes,
        "scores": scores,
        "gt_ids": list(gt_ids),
        "transformation_matrix": (
            transformation_matrix if transformation_matrix is not None else torch.eye(4)
        ),
    }


def _planar_transform(yaw, tx, ty):
    matrix = torch.eye(4)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    matrix[0, 0], matrix[0, 1] = cosine, -sine
    matrix[1, 0], matrix[1, 1] = sine, cosine
    matrix[0, 3], matrix[1, 3] = tx, ty
    return matrix


def _synthetic_frames():
    frame_a = {
        "agents": [
            _agent([[1.0, 1.0], [4.0, -2.0], [-3.0, 2.0]], [0.1, 0.4, -0.2], ["o1", "o2", "o3"]),
            _agent(
                [[0.0, 0.0], [2.0, 3.0], [-1.0, -1.0]],
                [0.0, 0.5, -0.3],
                ["o2", "o3", None],
                transformation_matrix=_planar_transform(0.1, 1.0, 0.5),
            ),
        ]
    }
    frame_b = {
        "agents": [
            _agent([[5.0, 0.0], [1.0, 1.0], [-3.0, 2.0]], [0.0, 0.2, 1.0], ["p1", "p2", "p3"]),
            _agent(
                [[0.5, 0.5], [4.5, -0.4]],
                [0.0, 0.9],
                ["p1", "p2"],
                transformation_matrix=_planar_transform(-0.2, -0.5, 1.5),
            ),
        ]
    }
    return [frame_a, frame_b]


def _loader(generator_seed, xy_std=0.3, yaw_std_deg=5.0):
    dataset = AlignFormerPVAPairDataset(
        _synthetic_frames(),
        noise_schedule=NoiseSchedule.fixed(xy_std, yaw_std_deg),
        generator=torch.Generator().manual_seed(generator_seed),
    )
    return DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate)


def _args(match_weight=1.0, pose_weight=1.0, epochs=1):
    return SimpleNamespace(match_weight=match_weight, pose_weight=pose_weight, epochs=epochs)


def _model():
    torch.manual_seed(0)
    return AlignFormerB(
        embed_dim=0, layers=1, heads=2, model_dim=16, variance_model=UNWEIGHTED
    )


def test_forward_and_loss_runs_and_gives_finite_scalars():
    model = _model()
    batch = next(iter(_loader(generator_seed=0)))
    result = _forward_and_loss(model, batch, _args())
    assert result is not None
    estimate, match_loss, pose_loss, total_loss = result
    assert torch.isfinite(match_loss)
    assert torch.isfinite(pose_loss)
    assert torch.isfinite(total_loss)
    assert math.isclose(
        float(total_loss), float(match_loss + pose_loss), rel_tol=1e-5
    )


def test_train_one_epoch_updates_weights_and_reports_metrics():
    model = _model()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    metrics = train_one_epoch(model, _loader(generator_seed=1), optimizer, device, _args(), epoch=0)

    assert metrics["pairs"] == 2
    assert metrics["skipped_batches"] == 0
    assert math.isfinite(metrics["total_loss"])
    assert math.isfinite(metrics["position_error_m"])
    assert math.isfinite(metrics["heading_error_deg"])

    after = model.state_dict()
    changed = any(
        not torch.equal(before[name], after[name]) for name in before
        if after[name].dtype.is_floating_point
    )
    assert changed, "optimizer step should have changed at least one trainable parameter"


def test_evaluate_matches_zero_noise_estimate_closely_after_a_few_training_steps():
    # Train briefly on a NO-noise schedule (true_psi = true_t = 0 exactly), so
    # a sane training loop should drive both the match loss and the pose
    # errors down rather than leaving them at their untrained-model values.
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    device = torch.device("cpu")
    args = _args()

    zero_noise_loader = lambda: _loader(generator_seed=2, xy_std=0.0, yaw_std_deg=0.0)
    initial = evaluate(model, zero_noise_loader(), device, args)
    for epoch in range(25):
        train_one_epoch(model, zero_noise_loader(), optimizer, device, args, epoch=epoch)
    final = evaluate(model, zero_noise_loader(), device, args)

    assert final["total_loss"] < initial["total_loss"]
    assert final["position_error_m"] < initial["position_error_m"] + 1e-6


def test_checkpoint_round_trip_preserves_weights_and_forward_output(tmp_path):
    model = _model()
    checkpoint = _checkpoint(model, SimpleNamespace(heading_lambda=2.0, sinkhorn_iterations=20,
                                                      model_dim=16, layers=1, heads=2),
                              epoch=1, metrics={"position_error_m": 0.5})
    path = tmp_path / "checkpoint.pth"
    torch.save(checkpoint, path)

    loaded, saved = load_alignformer_pva(path, torch.device("cpu"))
    assert saved["epoch"] == 1

    batch = next(iter(_loader(generator_seed=3)))
    model.eval()
    with torch.no_grad():
        original_estimate = model(batch)
        loaded_estimate = loaded(batch)
    assert torch.allclose(original_estimate.psi, loaded_estimate.psi)
    assert torch.allclose(original_estimate.t, loaded_estimate.t)


def test_load_alignformer_pva_rejects_a_checkpoint_from_a_different_pipeline(tmp_path):
    path = tmp_path / "bad.pth"
    torch.save({"pipeline": "something_else", "state_dict": {}}, path)
    try:
        load_alignformer_pva(path, torch.device("cpu"))
    except ValueError:
        return
    raise AssertionError("expected a ValueError for a non-alignformer_pva checkpoint")


# ---- --variance-config / _variance_model / checkpoint round-trip ----------


def _variance_config_file(tmp_path, **overrides):
    block = {
        "sigma_translation_m": 0.3,
        "translation_exponent": 1.1,
        "sigma_yaw_deg": 5.0,
        "yaw_exponent": 1.8,
        "score_reference": 0.4,
    }
    block.update(overrides)
    path = tmp_path / "variance_calibration.json"
    path.write_text(json.dumps({"correspondence_variance": block}))
    return path


def test_variance_model_defaults_to_unweighted_without_the_flag():
    # No variance_config attribute at all (an args namespace predating the
    # flag, like other SimpleNamespace fixtures in this file) must still work.
    assert _variance_model(SimpleNamespace()) is UNWEIGHTED
    assert _variance_model(SimpleNamespace(variance_config=None)) is UNWEIGHTED


def test_variance_model_builds_calibrated_split_model_from_json(tmp_path):
    config_path = _variance_config_file(tmp_path)
    model = _variance_model(SimpleNamespace(variance_config=config_path))
    assert model.mode == "split"
    assert math.isclose(model.sigma_translation_m, 0.3)
    assert math.isclose(model.translation_exponent, 1.1)
    assert math.isclose(model.sigma_yaw_rad, math.radians(5.0))
    assert math.isclose(model.yaw_exponent, 1.8)


def test_checkpoint_round_trip_preserves_calibrated_variance_model(tmp_path):
    config_path = _variance_config_file(tmp_path, sigma_translation_m=0.25, yaw_exponent=1.93)
    model = AlignFormerB(
        embed_dim=0, layers=1, heads=2, model_dim=16,
        variance_model=_variance_model(SimpleNamespace(variance_config=config_path)),
    )
    checkpoint = _checkpoint(
        model,
        SimpleNamespace(
            heading_lambda=2.0, sinkhorn_iterations=20, model_dim=16, layers=1, heads=2,
        ),
        epoch=1, metrics={"position_error_m": 0.5},
    )
    path = tmp_path / "checkpoint.pth"
    torch.save(checkpoint, path)

    loaded, _ = load_alignformer_pva(path, torch.device("cpu"))
    assert loaded.variance_model.mode == "split"
    assert math.isclose(loaded.variance_model.sigma_translation_m, 0.25)
    assert math.isclose(loaded.variance_model.yaw_exponent, 1.93)


def test_load_alignformer_pva_falls_back_to_unweighted_for_checkpoints_predating_variance_kwargs(
    tmp_path,
):
    # Simulates a checkpoint saved before variance_model_kwargs existed
    # (e.g. outputs/alignformer_pva_train_v1/best.pth): no such key present.
    model = _model()
    checkpoint = _checkpoint(
        model,
        SimpleNamespace(heading_lambda=2.0, sinkhorn_iterations=20, model_dim=16, layers=1, heads=2),
        epoch=1, metrics={"position_error_m": 0.5},
    )
    del checkpoint["variance_model_kwargs"]
    path = tmp_path / "old_checkpoint.pth"
    torch.save(checkpoint, path)

    loaded, _ = load_alignformer_pva(path, torch.device("cpu"))
    assert loaded.variance_model is UNWEIGHTED
