"""Frame dataset for spatial PointPillars memories and proposal queries."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


class TrackFormerProposalDataset(Dataset):
    def __init__(self, cache_dir: Path, *, split: str, validation_fraction=0.2):
        self.cache_dir = cache_dir
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Completed TrackFormer cache not found at {manifest_path}. "
                "Run the schema-v2 spatial-cache extraction before training."
            )
        self.manifest = json.loads(manifest_path.read_text())
        if int(self.manifest.get("schema_version", 1)) < 2:
            raise ValueError(
                "TrackFormer cache schema 2 is required because schema 1 lacks "
                "matched ground-truth boxes; re-extract to a new directory"
            )
        if not self.manifest.get("bev_shape"):
            raise ValueError("Cache has no spatial BEV maps; extract with --bev-height/width")
        self.paths = [cache_dir / item["path"] for item in self.manifest["shards"]]
        references, scenarios = [], set()
        for shard_index, path in enumerate(self.paths):
            for frame_index, frame in enumerate(_load(path)):
                observations = {}
                for agent in frame["agents"]:
                    for object_id in agent["gt_ids"]:
                        if object_id is not None:
                            observations.setdefault(str(object_id), set()).add(
                                str(agent["agent_id"])
                            )
                if any(len(value) > 1 for value in observations.values()):
                    scenario = frame["scenario_id"]
                    scenarios.add(scenario)
                    references.append((scenario, shard_index, frame_index))
        ordered = sorted(scenarios)
        count = max(1, round(len(ordered) * validation_fraction))
        validation = set(ordered[-count:])
        self.references = [
            (shard, frame)
            for scenario, shard, frame in references
            if (scenario in validation) == (split == "validation")
        ]
        self._shard_index = None
        self._shard = None

    def __len__(self):
        return len(self.references)

    def __getitem__(self, index):
        shard_index, frame_index = self.references[index]
        if shard_index != self._shard_index:
            self._shard = _load(self.paths[shard_index])
            self._shard_index = shard_index
        frame = self._shard[frame_index]
        agents, all_ids = [], set()
        for agent in frame["agents"]:
            if "gt_boxes" not in agent:
                raise ValueError(
                    "Spatial cache predates TrackFormer GT-box supervision; "
                    "re-extract it with the current extractor"
                )
            best = {}
            for proposal_index, object_id in enumerate(agent["gt_ids"]):
                if object_id is None:
                    continue
                object_id = str(object_id)
                iou = float(agent["gt_ious"][proposal_index])
                if object_id not in best or iou > best[object_id][0]:
                    best[object_id] = (iou, proposal_index)
            if not best:
                continue
            object_ids = sorted(best)
            indices = torch.tensor([best[item][1] for item in object_ids])
            all_ids.update(object_ids)
            agents.append(
                {
                    "agent_id": str(agent["agent_id"]),
                    "bev": agent["bev_features"].float(),
                    "query_features": agent["features"][indices].float(),
                    "boxes": agent["boxes"][indices].float(),
                    "target_boxes": agent["gt_boxes"][indices].float(),
                    "scores": agent["scores"][indices].float(),
                    "object_ids": object_ids,
                    "false_query_features": agent["features"][
                        torch.tensor(
                            [
                                i
                                for i, value in enumerate(agent["gt_ids"])
                                if value is None
                            ],
                            dtype=torch.long,
                        )
                    ].float(),
                    "false_boxes": agent["boxes"][
                        torch.tensor(
                            [
                                i
                                for i, value in enumerate(agent["gt_ids"])
                                if value is None
                            ],
                            dtype=torch.long,
                        )
                    ].float(),
                    "false_scores": agent["scores"][
                        torch.tensor(
                            [
                                i
                                for i, value in enumerate(agent["gt_ids"])
                                if value is None
                            ],
                            dtype=torch.long,
                        )
                    ].float(),
                }
            )
        label_by_id = {value: index for index, value in enumerate(sorted(all_ids))}
        for agent in agents:
            agent["labels"] = torch.tensor(
                [label_by_id[value] for value in agent.pop("object_ids")],
                dtype=torch.long,
            )
        return {"agents": agents}


def collate_trackformer_frames(frames):
    agents = []
    label_offset = 0
    agent_offset = 0
    for group, frame in enumerate(frames):
        frame_label_count = 0
        for agent_index, agent in enumerate(frame["agents"]):
            count = len(agent["labels"])
            agents.append(
                {
                    **{key: value for key, value in agent.items() if key != "agent_id"},
                    "labels": agent["labels"] + label_offset,
                    "agent_labels": torch.full(
                        (count,), agent_offset + agent_index, dtype=torch.long
                    ),
                    "groups": torch.full((count,), group, dtype=torch.long),
                }
            )
            frame_label_count = max(
                frame_label_count, int(agent["labels"].max()) + 1
            )
        label_offset += frame_label_count
        agent_offset += len(frame["agents"])
    return agents
