import math

import torch

from embedding_aware_belt_fusion.alignformer_pva.dataset import (
    AlignFormerPVAContextPairDataset,
    NoiseSchedule,
    collate_context,
)


def _planar_transform(yaw, tx, ty):
    matrix = torch.eye(4)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    matrix[0, 0], matrix[0, 1] = cosine, -sine
    matrix[1, 0], matrix[1, 1] = sine, cosine
    matrix[0, 3], matrix[1, 3] = tx, ty
    return matrix


def _context_agent(centers, yaws, gt_ids, *, max_cells=4, input_dim=6, transformation_matrix=None, seed=0):
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
    tokens = torch.randn(n, max_cells, input_dim, generator=generator)
    positions = torch.randn(n, max_cells, 2, generator=generator)
    footprint_flag = torch.zeros(n, max_cells, dtype=torch.bool)
    footprint_flag[:, :1] = True
    mask = torch.zeros(n, max_cells, dtype=torch.bool)
    if n and max_cells > 2:
        mask[:, -1] = True  # last cell slot is padding for every proposal
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


def _two_frame_context_dataset(generator_seed=0, max_objects=64):
    frame_a = {
        "agents": [
            _context_agent([[1.0, 1.0], [4.0, -2.0]], [0.1, 0.4], ["o1", "o2"], seed=1),
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
                [[0.5, 0.5]], [0.0], ["p1"],
                transformation_matrix=_planar_transform(-0.2, -0.5, 1.5), seed=4,
            ),
        ]
    }
    return AlignFormerPVAContextPairDataset(
        [frame_a, frame_b],
        noise_schedule=NoiseSchedule.fixed(0.3, 5.0),
        generator=torch.Generator().manual_seed(generator_seed),
        max_objects=max_objects,
    )


def test_context_dataset_getitem_carries_cell_fields_aligned_with_geometry():
    dataset = _two_frame_context_dataset()
    sample = dataset[0]

    ego_n = sample["ego_boxes"].shape[0]
    cav_n = sample["cav_boxes"].shape[0]
    assert sample["ego_cell_tokens"].shape[0] == ego_n
    assert sample["cav_cell_tokens"].shape[0] == cav_n
    assert sample["ego_cell_tokens"].shape[-1] == 6
    assert sample["ego_cell_positions"].shape[-1] == 2
    assert sample["ego_cell_footprint"].shape == sample["ego_cell_mask"].shape
    # Still carries the zero-width placeholders for compatibility with the
    # shared `collate` helper reused inside `collate_context`.
    assert sample["ego_embeddings"].shape == (ego_n, 0)
    assert sample["cav_embeddings"].shape == (cav_n, 0)


def test_context_dataset_truncation_keeps_cell_fields_in_lockstep_with_geometry():
    # max_objects=1 forces truncation to the single highest-score proposal on
    # each side; the retained cell tensor must be THAT proposal's own cells,
    # not an arbitrary one, since _agent_cell_fields slices by the same order
    # _truncate_top_n used for the boxes/scores/ids.
    dataset = _two_frame_context_dataset(max_objects=1)
    sample = dataset[0]

    assert sample["ego_boxes"].shape[0] == 1
    assert sample["ego_cell_tokens"].shape[0] == 1
    # The ego agent's highest-score proposal is index 0 (scores are
    # descending by construction in _context_agent), so its cell tokens must
    # match the raw agent's own proposal-0 tokens exactly.
    frame = dataset.frames[0]
    ego_agent = frame["agents"][0]
    assert torch.equal(sample["ego_cell_tokens"][0], ego_agent["tokens"][0])


def test_collate_context_pads_object_and_cell_dimensions():
    dataset = _two_frame_context_dataset()
    samples = [dataset[index] for index in range(len(dataset))]

    batch = collate_context(samples)

    ego_object_max = max(sample["ego_boxes"].shape[0] for sample in samples)
    ego_cell_max = max(sample["ego_cell_tokens"].shape[1] for sample in samples)
    cav_object_max = max(sample["cav_boxes"].shape[0] for sample in samples)
    cav_cell_max = max(sample["cav_cell_tokens"].shape[1] for sample in samples)

    assert batch["ego_cell_tokens"].shape == (len(samples), ego_object_max, ego_cell_max, 6)
    assert batch["cav_cell_tokens"].shape == (len(samples), cav_object_max, cav_cell_max, 6)
    assert batch["ego_cell_mask"].dtype == torch.bool
    assert batch["ego_cell_footprint"].dtype == torch.bool

    # Every fully object-padded slot (beyond that sample's own real object
    # count) must be entirely cell-padded too -- an all-True cell mask row.
    for sample_index, sample in enumerate(samples):
        real_count = sample["ego_boxes"].shape[0]
        if real_count < ego_object_max:
            assert bool(batch["ego_cell_mask"][sample_index, real_count:].all())

    # collate() is still reused for the ordinary fields.
    assert batch["ego_boxes"].shape == (len(samples), ego_object_max, 7)
    assert batch["ego_embeddings"].shape == (len(samples), ego_object_max, 0)


def test_collate_context_rejects_empty_batch():
    try:
        collate_context([])
    except ValueError:
        return
    raise AssertionError("collate_context([]) should raise ValueError")
