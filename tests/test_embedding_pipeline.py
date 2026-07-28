import json
from pathlib import Path

import torch

from embedding_aware_belt_fusion.data.proposal_cache import (
    CrossAgentProposalDataset,
    collate_proposal_frames,
)
from embedding_aware_belt_fusion.embeddings import (
    CrossAgentSupervisedContrastiveLoss,
    ProposalEmbeddingHead,
)
from embedding_aware_belt_fusion.embeddings.train import _retrieval_metrics


def _frame(scenario, frame_id):
    return {
        "scenario_id": scenario,
        "frame_id": frame_id,
        "agents": [
            {
                "agent_id": "a",
                "features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "gt_ids": ["object-1", "object-2"],
                "gt_ious": torch.tensor([0.8, 0.7]),
            },
            {
                "agent_id": "b",
                "features": torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
                "gt_ids": ["object-1", "object-2"],
                "gt_ious": torch.tensor([0.9, 0.8]),
            },
        ],
    }


def _cache(tmp_path: Path):
    frames = [
        _frame("scenario-a", "1"),
        _frame("scenario-b", "1"),
        _frame("scenario-c", "1"),
    ]
    torch.save(frames, tmp_path / "train_00000.pt")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "feature_dim": 2,
                "shards": [{"path": "train_00000.pt", "frames": len(frames)}],
            }
        )
    )


def test_proposal_cache_uses_scenario_disjoint_splits(tmp_path):
    _cache(tmp_path)
    train = CrossAgentProposalDataset(
        tmp_path, split="train", validation_fraction=1 / 3
    )
    validation = CrossAgentProposalDataset(
        tmp_path, split="validation", validation_fraction=1 / 3
    )
    assert len(train) == 2
    assert len(validation) == 1
    assert train[0]["features"].shape == (4, 2)
    batch = collate_proposal_frames([train[0], train[1]])
    assert batch["features"].shape == (8, 2)
    assert batch["labels"].unique().numel() == 4
    assert batch["groups"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_cross_agent_contrastive_loss_trains_normalized_embeddings(tmp_path):
    _cache(tmp_path)
    dataset = CrossAgentProposalDataset(
        tmp_path, split="train", validation_fraction=1 / 3
    )
    batch = collate_proposal_frames([dataset[0], dataset[1]])
    model = ProposalEmbeddingHead(
        input_dim=2, hidden_dims=(8, 4), embedding_dim=3, dropout=0
    )
    embeddings = model(batch["features"])
    assert torch.allclose(
        embeddings.norm(dim=1), torch.ones(len(embeddings)), atol=1e-6
    )
    loss = CrossAgentSupervisedContrastiveLoss()(
        embeddings, batch["labels"], batch["agents"]
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_retrieval_is_restricted_to_the_same_frame():
    embeddings = torch.tensor([[1.0, 0.0], [0.8, 0.2], [1.0, 0.0]])
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    labels = torch.tensor([0, 0, 1])
    agents = torch.tensor([0, 1, 2])
    groups = torch.tensor([0, 0, 1])
    metrics = _retrieval_metrics(embeddings, labels, agents, groups)
    assert metrics["top1_cross_agent_accuracy"] == 1.0
    assert metrics["random_top1_accuracy"] == 1.0
