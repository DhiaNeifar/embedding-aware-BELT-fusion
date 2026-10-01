"""Reusable, provenance-checked frame artifacts for late-fusion evaluation.

The cache boundary deliberately follows PointPillars proposal decoding and ROI
materialisation.  These artifacts are identical for experiments that change
only PQ-STF association, RANSAC, box fusion, or message quantization.  A
TrackFormer checkpoint is *not* part of the cache key because its states are
recomputed from the cached ROI tokens for every experiment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


CACHE_VERSION = 1


def _file_fingerprint(path: Path) -> dict[str, Any]:
    """Return a content fingerprint, not merely a path or modification time."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def build_provenance(
    *,
    opencood_config: Path,
    detector_checkpoint: Path,
    uncertainty_checkpoint: Path,
    data_root: Path,
    split: str,
    noise: dict[str, Any],
    scenario_subset: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the strict identity of detector-side evaluation artifacts."""
    return {
        "cache_version": CACHE_VERSION,
        "opencood_config": _file_fingerprint(opencood_config),
        "detector_checkpoint": _file_fingerprint(detector_checkpoint),
        "uncertainty_checkpoint": _file_fingerprint(uncertainty_checkpoint),
        "data_root": str(data_root.resolve()),
        "split": split,
        "noise": noise,
        "scenario_subset": scenario_subset,
    }


_FINGERPRINT_FIELDS = ("opencood_config", "detector_checkpoint", "uncertainty_checkpoint")


def provenance_matches(saved: dict[str, Any] | None, current: dict[str, Any]) -> bool:
    """Compare two provenance dicts by content, not by file path.

    A file's identity is its (bytes, sha256), never the absolute path it
    happened to be loaded from. Comparing paths too makes every cache break
    the moment the repo is relocated or renamed, even though the checkpoint/
    config content -- the only thing that actually determines what got
    cached -- is unchanged. Everything else (noise, split, data_root, ...)
    is still compared exactly, since those genuinely must match.
    """
    if saved is None:
        return False
    if saved.keys() != current.keys():
        return False
    for key, current_value in current.items():
        saved_value = saved[key]
        if key in _FINGERPRINT_FIELDS:
            if saved_value is None or current_value is None:
                return False
            if (saved_value.get("bytes"), saved_value.get("sha256")) != (
                current_value.get("bytes"),
                current_value.get("sha256"),
            ):
                return False
        elif saved_value != current_value:
            return False
    return True


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return value


def to_device(value, device: torch.device):
    """Recursively move a cached artifact to the evaluator device."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=device.type == "cuda")
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    return value


class EvaluationFrameCache:
    """Append-only cache of detector proposals, uncertainty, and ROI tokens.

    ``read-write`` builds missing artifacts and can replay only after all
    requested frames have been written.  ``read-only`` requires a complete,
    provenance-compatible cache and skips PointPillars inference and dataset
    frame loading entirely.
    """

    def __init__(
        self,
        directory: Path | None,
        *,
        mode: str,
        provenance: dict[str, Any],
        expected_frames: int,
    ):
        self.directory = directory
        self.mode = mode
        self.provenance = provenance
        self.expected_frames = expected_frames
        self.enabled = directory is not None and mode != "off"
        self.replay = False
        if not self.enabled:
            return
        if mode not in {"read-write", "read-only", "refresh"}:
            raise ValueError(f"Unknown frame-cache mode: {mode}")
        assert directory is not None
        self.manifest_path = directory / "manifest.json"
        if mode == "refresh" and directory.exists():
            for artifact in directory.glob("frame_*.pt"):
                artifact.unlink()
            self.manifest_path.unlink(missing_ok=True)
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text())
            if not provenance_matches(manifest.get("provenance"), provenance):
                raise RuntimeError(
                    "Frame cache provenance does not match this experiment. "
                    "Use a separate --frame-cache-dir or --frame-cache-mode refresh."
                )
            self.replay = int(manifest.get("complete_frames", 0)) >= expected_frames
        elif mode == "read-only":
            raise FileNotFoundError(
                f"No complete frame cache at {directory}; read-only replay is impossible"
            )
        else:
            directory.mkdir(parents=True, exist_ok=True)
            self._write_manifest(complete_frames=0)
        if mode == "read-only" and not self.replay:
            raise RuntimeError(
                "Frame cache is incomplete; run once with --frame-cache-mode read-write"
            )

    def _write_manifest(self, *, complete_frames: int):
        assert self.directory is not None
        payload = {
            "format": "pq_stf_detector_roi_frame_cache_v1",
            "provenance": self.provenance,
            "expected_frames": self.expected_frames,
            "complete_frames": complete_frames,
        }
        self.manifest_path.write_text(json.dumps(payload, indent=2) + "\n")

    def artifact_path(self, index: int) -> Path:
        assert self.directory is not None
        return self.directory / f"frame_{index:06d}.pt"

    def load(self, index: int) -> dict[str, Any]:
        path = self.artifact_path(index)
        if not path.exists():
            raise FileNotFoundError(f"Missing cached frame artifact: {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

    def store(self, index: int, artifact: dict[str, Any]):
        if not self.enabled or self.mode == "read-only":
            return
        path = self.artifact_path(index)
        torch.save(_cpu(artifact), path)
        # Frame indices are dense because evaluator traversal is sequential.
        complete = 0
        while self.artifact_path(complete).exists():
            complete += 1
        self._write_manifest(complete_frames=complete)

