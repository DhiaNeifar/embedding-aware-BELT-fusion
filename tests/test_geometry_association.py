import math

import torch

from embedding_aware_belt_fusion.embeddings.geometry import (
    normalized_ego_geometry,
    transform_boxes_to_ego,
)
from embedding_aware_belt_fusion.embeddings.train_complete_roi import (
    CompactROIDataset,
    _collate_frames,
)


def _rigid_transform(yaw, translation):
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    matrix = torch.eye(4)
    matrix[:3, 3] = torch.tensor(translation)
    matrix[0, 0] = cosine
    matrix[0, 1] = -sine
    matrix[1, 0] = sine
    matrix[1, 1] = cosine
    return matrix


def test_transform_boxes_to_ego_transforms_center_and_yaw():
    boxes = torch.tensor(
        [[1.0, 2.0, 0.5, 1.5, 2.0, 4.0, math.pi / 4]]
    )
    matrix = _rigid_transform(math.pi / 2, [10.0, -3.0, 1.0])
    transformed = transform_boxes_to_ego(boxes, matrix)
    assert torch.allclose(
        transformed[0, :3],
        torch.tensor([8.0, -2.0, 1.5]),
        atol=1e-6,
    )
    assert torch.allclose(transformed[0, 3:6], boxes[0, 3:6])
    assert torch.allclose(
        transformed[0, 6],
        torch.tensor(3 * math.pi / 4),
        atol=1e-6,
    )


def test_normalized_geometry_contains_ego_box_orientation_and_score():
    boxes = torch.tensor([[7.04, 4.0, 0.4, 2.0, 2.0, 5.0, 0.0]])
    features = normalized_ego_geometry(
        boxes,
        torch.tensor([0.75]),
        torch.eye(4),
    )
    expected = torch.tensor(
        [[0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.0, 1.0, 0.75]]
    )
    assert torch.allclose(features, expected, atol=1e-6)


class _SyntheticCompactCache:
    def __init__(self):
        self.references = [(0, 0)]
        self.manifest = {"bev_shape": [384, 100, 176]}

    def __len__(self):
        return 1

    def __getitem__(self, index):
        del index
        return {
            "agents": [
                {
                    "tokens": torch.ones(2, 3, 384),
                    "positions": torch.zeros(2, 3, 2),
                    "mask": torch.zeros(2, 3, dtype=torch.bool),
                    "boxes": torch.tensor(
                        [
                            [1.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0],
                            [2.0, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0],
                        ]
                    ),
                    "scores": torch.tensor([0.8, 0.9]),
                    "gt_ids": ["car", "car"],
                    "gt_ious": torch.tensor([0.6, 0.9]),
                    "transformation_matrix": _rigid_transform(
                        0.0, [10.0, 0.0, 0.0]
                    ),
                }
            ]
        }


def test_compact_dataset_selects_best_proposal_and_collates_geometry():
    dataset = CompactROIDataset(
        _SyntheticCompactCache(), geometry_aware=True
    )
    frame = dataset[0]
    assert frame["agents"][0]["object_ids"] == ["car"]
    # The higher-IoU proposal at x=2 is transformed to ego x=12.
    assert torch.allclose(
        frame["agents"][0]["geometry"][0, 0],
        torch.tensor(12.0 / 70.4),
    )
    batch = _collate_frames([frame])
    assert batch["geometry"].shape == (1, 9)
