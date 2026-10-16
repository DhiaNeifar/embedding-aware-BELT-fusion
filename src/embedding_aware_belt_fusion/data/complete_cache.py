"""Access complete cache records without discarding any vehicle-covered cells."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def proposal_roi_tokens(agent, proposal_index: int):
    """Return all 384-D native BEV cells covered by one proposal."""
    bev = agent["bev_features"]
    indices = agent["proposal_roi_cell_indices"][proposal_index].long()
    flattened = bev.flatten(1).transpose(0, 1)
    return flattened[indices]


def _load_shard(path):
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )


def _index_one_shard(cache_dir, shard_index, shard):
    frames = _load_shard(Path(cache_dir) / shard["path"])
    return [
        {
            "scenario_id": str(frame["scenario_id"]),
            "frame_id": str(frame["frame_id"]),
            "shard_index": shard_index,
            "frame_index": frame_index,
        }
        for frame_index, frame in enumerate(frames)
    ]


def load_or_build_scenario_index(cache_dir: Path, *, workers=8):
    """Return lightweight frame metadata, building it once when necessary."""
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    index_path = cache_dir / "scenario_index.json"
    shard_names = [item["path"] for item in manifest["shards"]]
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        if (
            index.get("cache_schema_version") == manifest.get("schema_version")
            and index.get("shards") == shard_names
            and len(index.get("frames", [])) == manifest.get("frame_count")
        ):
            return index

    indexed = [None] * len(manifest["shards"])
    if workers <= 1:
        progress = tqdm(
            enumerate(manifest["shards"]),
            total=len(manifest["shards"]),
            desc="indexing cache scenarios",
        )
        for shard_index, shard in progress:
            indexed[shard_index] = _index_one_shard(
                cache_dir, shard_index, shard
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _index_one_shard,
                    str(cache_dir),
                    shard_index,
                    shard,
                ): shard_index
                for shard_index, shard in enumerate(manifest["shards"])
            }
            progress = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="indexing cache scenarios",
            )
            for future in progress:
                shard_index = futures[future]
                indexed[shard_index] = future.result()
    frames = [frame for shard in indexed for frame in shard]
    index = {
        "schema_version": 1,
        "cache_schema_version": manifest["schema_version"],
        "shards": shard_names,
        "frames": frames,
    }
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    return index


class CompleteFrameDataset(Dataset):
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.manifest = json.loads((cache_dir / "manifest.json").read_text())
        if self.manifest.get("cache_mode") != "complete":
            raise ValueError("CompleteFrameDataset requires cache_mode=complete")
        self.references = []
        self.paths = []
        for shard_index, item in enumerate(self.manifest["shards"]):
            path = cache_dir / item["path"]
            self.paths.append(path)
            self.references.extend(
                (shard_index, frame_index)
                for frame_index in range(item["frames"])
            )
        self._shard_index = None
        self._shard = None

    def __len__(self):
        return len(self.references)

    def __getitem__(self, index):
        shard_index, frame_index = self.references[index]
        if self._shard_index != shard_index:
            self._shard = _load_shard(self.paths[shard_index])
            self._shard_index = shard_index
        return self._shard[frame_index]


class IndexedCompleteFrameDataset(Dataset):
    """Complete frames selected by a precomputed scenario-aware index."""

    def __init__(self, cache_dir: Path, records):
        self.cache_dir = Path(cache_dir)
        self.manifest = json.loads(
            (self.cache_dir / "manifest.json").read_text()
        )
        self.paths = [
            self.cache_dir / item["path"]
            for item in self.manifest["shards"]
        ]
        self.records = list(records)
        self.references = [
            (int(item["shard_index"]), int(item["frame_index"]))
            for item in self.records
        ]
        self._shard_index = None
        self._shard = None

    def __len__(self):
        return len(self.references)

    def __getitem__(self, index):
        shard_index, frame_index = self.references[index]
        if self._shard_index != shard_index:
            self._shard = _load_shard(self.paths[shard_index])
            self._shard_index = shard_index
        return self._shard[frame_index]
