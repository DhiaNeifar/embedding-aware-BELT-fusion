import math

import torch

from embedding_aware_belt_fusion.alignformer_pva.dataset import (
    AlignFormerPVAPairDataset,
    NoiseSchedule,
    _apply_planar_noise,
    _project_true,
    _true_correction,
    _wrap_angle,
    collate,
    correspondence_indices,
)
from embedding_aware_belt_fusion.alignformer_pva.losses import apply_se2
from embedding_aware_belt_fusion.alignformer_pva.model import AlignFormerB


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


# ---- correspondence_indices -------------------------------------------------


def test_correspondence_indices_shared_unmatched_and_duplicate_ids():
    # "a" is shared but duplicated on the cav side (only the first occurrence
    # counts); "b" exists only on the ego side; "c" is shared; the trailing
    # None on each side can never match anything.
    ego_ids = ["a", "b", None, "c"]
    cav_ids = ["c", "a", "a", None]

    ego_match, cav_match = correspondence_indices(ego_ids, cav_ids)

    assert torch.equal(ego_match, torch.tensor([1, -1, -1, 0]))
    assert torch.equal(cav_match, torch.tensor([3, 0, -1, -1]))


def test_correspondence_indices_empty_inputs():
    ego_match, cav_match = correspondence_indices([], [])
    assert ego_match.shape == (0,)
    assert cav_match.shape == (0,)


# ---- collate -----------------------------------------------------------------


def _sample(ego_n, cav_n, ego_match, cav_match, psi_true, t_true):
    return {
        "ego_boxes": torch.arange(ego_n * 7, dtype=torch.float32).reshape(ego_n, 7),
        "cav_boxes": torch.arange(cav_n * 7, dtype=torch.float32).reshape(cav_n, 7),
        "ego_scores": torch.full((ego_n,), 0.8),
        "cav_scores": torch.full((cav_n,), 0.7),
        "ego_mask": torch.ones(ego_n, dtype=torch.bool),
        "cav_mask": torch.ones(cav_n, dtype=torch.bool),
        "ego_embeddings": torch.zeros(ego_n, 0),
        "cav_embeddings": torch.zeros(cav_n, 0),
        "ego_match": torch.tensor(ego_match, dtype=torch.long),
        "cav_match": torch.tensor(cav_match, dtype=torch.long),
        "psi_true": torch.tensor(psi_true),
        "t_true": torch.tensor(t_true),
    }


def test_collate_pads_to_batch_max_and_builds_masks():
    sample_a = _sample(2, 3, [1, -1], [-1, 0, -1], 0.1, [0.2, 0.3])
    sample_b = _sample(3, 1, [-1, -1, 0], [2], -0.05, [-0.1, 0.0])

    batch = collate([sample_a, sample_b])

    assert batch["ego_boxes"].shape == (2, 3, 7)
    assert batch["cav_boxes"].shape == (2, 3, 7)
    assert batch["ego_scores"].shape == (2, 3)
    assert batch["cav_scores"].shape == (2, 3)
    assert batch["ego_embeddings"].shape == (2, 3, 0)
    assert batch["cav_embeddings"].shape == (2, 3, 0)

    assert torch.equal(
        batch["ego_mask"], torch.tensor([[True, True, False], [True, True, True]])
    )
    assert torch.equal(
        batch["cav_mask"], torch.tensor([[True, True, True], [True, False, False]])
    )

    assert torch.equal(
        batch["ego_match"], torch.tensor([[1, -1, -1], [-1, -1, 0]])
    )
    assert torch.equal(
        batch["cav_match"], torch.tensor([[-1, 0, -1], [2, -1, -1]])
    )

    assert torch.allclose(batch["psi_true"], torch.tensor([0.1, -0.05]))
    assert torch.allclose(batch["t_true"], torch.tensor([[0.2, 0.3], [-0.1, 0.0]]))

    # Padded geometry/score slots are exactly 0, never garbage.
    assert torch.equal(batch["ego_boxes"][0, 2], torch.zeros(7))
    assert torch.equal(batch["cav_boxes"][1, 1], torch.zeros(7))
    assert float(batch["ego_scores"][0, 2]) == 0.0


def test_collate_rejects_empty_batch():
    try:
        collate([])
    except ValueError:
        return
    raise AssertionError("collate([]) should raise ValueError")


# ---- noise injection / correction consistency --------------------------------
#
# The single most important property: applying (psi_true, t_true) to the
# NOISY-projected cav boxes must recover the TRUE-pose-projected cav boxes.
# Deliberately fixed, non-random noise here (not drawn from the dataset's
# generator), per the orchestrator's instructions.


def test_true_correction_undoes_a_known_injected_noise():
    cav_agent = _agent(
        centers=[[3.0, 1.0], [-2.0, 4.5], [0.5, -3.0]],
        yaws=[0.2, -1.1, 2.8],
        gt_ids=["o1", "o2", "o3"],
        transformation_matrix=_planar_transform(yaw=0.3, tx=5.0, ty=-2.0),
    )
    true_boxes = _project_true(cav_agent)

    # A fixed, deliberate perturbation -- not sampled randomly.
    psi_noise = torch.tensor(0.15)
    t_noise = torch.tensor([0.4, -0.2])

    noisy_boxes = _apply_planar_noise(true_boxes, psi_noise, t_noise)
    psi_true, t_true = _true_correction(psi_noise, t_noise)

    recovered_centers = apply_se2(
        noisy_boxes[:, :2].unsqueeze(0), psi_true.unsqueeze(0), t_true.unsqueeze(0)
    ).squeeze(0)
    assert torch.allclose(recovered_centers, true_boxes[:, :2], atol=1e-5)

    recovered_yaw = _wrap_angle(noisy_boxes[:, 6] + psi_true)
    assert torch.allclose(recovered_yaw, _wrap_angle(true_boxes[:, 6]), atol=1e-5)

    # Non-planar fields must be untouched by the noise.
    assert torch.allclose(noisy_boxes[:, 2:6], true_boxes[:, 2:6])


def test_true_correction_is_identity_for_zero_noise():
    psi_true, t_true = _true_correction(torch.tensor(0.0), torch.tensor([0.0, 0.0]))
    assert torch.allclose(psi_true, torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(t_true, torch.tensor([0.0, 0.0]), atol=1e-7)


# ---- AlignFormerPVAPairDataset end-to-end -------------------------------------


def _two_frame_dataset(generator_seed=0, noise_schedule=None):
    frame_a = {
        "agents": [
            _agent([[1.0, 1.0], [4.0, -2.0]], [0.1, 0.4], ["o1", "o2"]),
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
            _agent(
                [[5.0, 0.0], [1.0, 1.0], [-3.0, 2.0]],
                [0.0, 0.2, 1.0],
                ["p1", "p2", "p3"],
            ),
            _agent(
                [[0.5, 0.5]],
                [0.0],
                ["p1"],
                transformation_matrix=_planar_transform(-0.2, -0.5, 1.5),
            ),
        ]
    }
    return AlignFormerPVAPairDataset(
        [frame_a, frame_b],
        noise_schedule=noise_schedule or NoiseSchedule.fixed(0.3, 5.0),
        generator=torch.Generator().manual_seed(generator_seed),
    )


def test_dataset_len_and_getitem_shapes():
    dataset = _two_frame_dataset()
    assert len(dataset) == 2  # one CAV per frame, across 2 frames

    sample = dataset[0]
    ego_n, cav_n = sample["ego_boxes"].shape[0], sample["cav_boxes"].shape[0]
    assert ego_n == 2
    assert cav_n == 3
    assert sample["ego_boxes"].shape == (ego_n, 7)
    assert sample["cav_boxes"].shape == (cav_n, 7)
    assert sample["ego_scores"].shape == (ego_n,)
    assert sample["cav_scores"].shape == (cav_n,)
    assert sample["ego_mask"].shape == (ego_n,)
    assert sample["ego_mask"].dtype == torch.bool
    assert bool(sample["ego_mask"].all())
    assert sample["cav_mask"].shape == (cav_n,)
    assert bool(sample["cav_mask"].all())
    assert sample["ego_embeddings"].shape == (ego_n, 0)
    assert sample["cav_embeddings"].shape == (cav_n, 0)
    assert sample["ego_match"].shape == (ego_n,)
    assert sample["cav_match"].shape == (cav_n,)
    assert sample["psi_true"].shape == ()
    assert sample["t_true"].shape == (2,)

    # ego object 1 ("o2") is shared with cav object 0 ("o2"); the ego's "o1"
    # and the cav's "o3"/None have no match.
    assert int(sample["ego_match"][0]) == -1
    assert int(sample["ego_match"][1]) == 0
    assert int(sample["cav_match"][0]) == 1
    assert int(sample["cav_match"][1]) == -1
    assert int(sample["cav_match"][2]) == -1


def test_dataset_set_epoch_changes_noise_schedule_std():
    schedule = NoiseSchedule(
        xy_std_start=0.0, xy_std_end=2.0, yaw_std_start_deg=0.0, yaw_std_end_deg=20.0,
        ramp_epochs=5,
    )
    dataset = _two_frame_dataset(noise_schedule=schedule)
    assert dataset.noise_schedule.stds(0) == (0.0, 0.0)
    dataset.set_epoch(4)
    xy_std, yaw_std = dataset.noise_schedule.stds(dataset.epoch)
    assert math.isclose(xy_std, 2.0)
    assert math.isclose(yaw_std, math.radians(20.0))


def test_dataset_collate_and_model_forward_run_end_to_end():
    dataset = _two_frame_dataset()
    samples = [dataset[index] for index in range(len(dataset))]
    batch = collate(samples)

    for key in (
        "ego_boxes", "cav_boxes", "ego_scores", "cav_scores", "ego_mask", "cav_mask",
        "ego_embeddings", "cav_embeddings", "ego_match", "cav_match", "psi_true", "t_true",
    ):
        assert key in batch

    torch.manual_seed(0)
    model = AlignFormerB(embed_dim=0, layers=1, heads=2, model_dim=16)
    estimate = model(batch)

    assert estimate.psi.shape == (len(samples),)
    assert estimate.t.shape == (len(samples), 2)
    assert torch.isfinite(estimate.psi).all()
    assert torch.isfinite(estimate.t).all()
