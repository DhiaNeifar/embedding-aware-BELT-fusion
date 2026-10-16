"""Datasets over cached same-timestamp, multi-agent proposal records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


def _load_shard(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


class CrossAgentProposalDataset(Dataset):
    """One item is a frame containing positives and natural cross-agent negatives."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        split: str,
        validation_fraction: float = 0.2,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        self.cache_dir = cache_dir
        self.manifest = json.loads((cache_dir / "manifest.json").read_text())
        self.shard_paths = [
            cache_dir / item["path"] for item in self.manifest["shards"]
        ]
        references = []
        scenarios = set()
        for shard_index, path in enumerate(self.shard_paths):
            for frame_index, frame in enumerate(_load_shard(path)):
                if self._has_cross_agent_positive(frame):
                    scenario = frame["scenario_id"]
                    scenarios.add(scenario)
                    references.append((scenario, shard_index, frame_index))
        ordered_scenarios = sorted(scenarios)
        validation_count = max(1, round(len(ordered_scenarios) * validation_fraction))
        validation_scenarios = set(ordered_scenarios[-validation_count:])
        self.references = [
            (shard_index, frame_index)
            for scenario, shard_index, frame_index in references
            if (scenario in validation_scenarios) == (split == "validation")
        ]
        self._cached_shard_index = None
        self._cached_shard = None

    @staticmethod
    def _has_cross_agent_positive(frame: Dict) -> bool:
        observations: Dict[str, set] = {}
        for agent in frame["agents"]:
            for gt_id in agent["gt_ids"]:
                if gt_id is not None:
                    observations.setdefault(gt_id, set()).add(agent["agent_id"])
        return any(len(agent_ids) >= 2 for agent_ids in observations.values())

    def __len__(self) -> int:
        return len(self.references)

    def _frame(self, shard_index: int, frame_index: int):
        if self._cached_shard_index != shard_index:
            self._cached_shard = _load_shard(self.shard_paths[shard_index])
            self._cached_shard_index = shard_index
        return self._cached_shard[frame_index]

    def __getitem__(self, index: int) -> Dict[str, object]:
        shard_index, frame_index = self.references[index]
        frame = self._frame(shard_index, frame_index)
        best: Dict[Tuple[str, str], Tuple[float, torch.Tensor]] = {}
        for agent in frame["agents"]:
            for proposal_index, gt_id in enumerate(agent["gt_ids"]):
                if gt_id is None:
                    continue
                key = (str(gt_id), str(agent["agent_id"]))
                iou = float(agent["gt_ious"][proposal_index])
                if key not in best or iou > best[key][0]:
                    best[key] = (iou, agent["features"][proposal_index].float())

        object_ids = sorted({key[0] for key in best})
        label_by_id = {object_id: label for label, object_id in enumerate(object_ids)}
        agent_ids = sorted({key[1] for key in best})
        agent_by_id = {agent_id: label for label, agent_id in enumerate(agent_ids)}
        ordered = sorted(best)
        return {
            "features": torch.stack([best[key][1] for key in ordered]),
            "labels": torch.tensor(
                [label_by_id[key[0]] for key in ordered], dtype=torch.long
            ),
            "agents": torch.tensor(
                [agent_by_id[key[1]] for key in ordered], dtype=torch.long
            ),
            "scenario_id": frame["scenario_id"],
            "frame_id": frame["frame_id"],
        }


def collate_proposal_frames(frames: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
    features = []
    labels = []
    agents = []
    groups = []
    label_offset = 0
    agent_offset = 0
    for group_index, frame in enumerate(frames):
        features.append(frame["features"])
        labels.append(frame["labels"] + label_offset)
        agents.append(frame["agents"] + agent_offset)
        groups.append(
            torch.full_like(frame["labels"], group_index, dtype=torch.long)
        )
        label_offset += int(frame["labels"].max()) + 1
        agent_offset += int(frame["agents"].max()) + 1
    return {
        "features": torch.cat(features),
        "labels": torch.cat(labels),
        "agents": torch.cat(agents),
        "groups": torch.cat(groups),
    }
