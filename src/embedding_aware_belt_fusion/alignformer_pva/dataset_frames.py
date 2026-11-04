"""Frame-grouped wrapper over ``AlignFormerPVAPairDataset``, for CAV-context training.

Additive, new file: does not modify ``dataset.py``. Builds the training data
``model_context.py``'s ``AlignFormerBWithContext`` needs -- every sample still
corrects ONE CAV against ego (same ``ego_*``/``cav_*``/``psi_true``/``t_true``
fields, same shapes, same noise-injection and ground-truth-correspondence
logic), plus a ``context_*`` field set: the OTHER CAVs' own (independently
noised) boxes/scores in the same frame, for the trunk's extra cross-attention
to see.

Deliberately implemented by WRAPPING an existing, already-built
``AlignFormerPVAPairDataset`` rather than reimplementing frame/pair indexing,
noise injection, or ground-truth correspondence matching a second time: this
dataset's ``__len__``/``__getitem__`` indexing scheme is identical to the
pair dataset it wraps (one item per (ego, CAV) pair, same flat index space),
so it drops into ``train.py``'s existing ``_loaders``/``DataLoader`` machinery
with no changes beyond the collate function.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from embedding_aware_belt_fusion.alignformer_pva.dataset import AlignFormerPVAPairDataset, collate

_CONTEXT_FIELDS = ("context_boxes", "context_scores", "context_embeddings", "context_mask")
_CONTEXT_PAD_VALUES = {"context_mask": False}


class AlignFormerPVAFrameDataset(Dataset):
    """Wraps ``pair_dataset``, attaching every other same-frame CAV as context.

    ``__len__``/``__getitem__`` match ``pair_dataset`` exactly (same flat
    pair index), so ``shuffle=True`` in the training ``DataLoader`` still
    shuffles individual CAV-corrections, not whole frames -- unchanged
    training dynamics from ``AlignFormerPVAPairDataset``'s own loader.

    A frame with only one CAV gets a genuine zero-width (``K=0``) context set
    for that CAV's sample (empty tensors, not ``None``), matching exactly
    what ``AlignFormerTrunkWithContext`` treats as its architecturally-exact
    no-op case.

    Known inefficiency, acceptable for a first version: each ``__getitem__``
    call re-invokes ``pair_dataset[sibling_index]`` for every sibling CAV in
    the frame (re-running its own noise injection and truncation), rather
    than caching one frame's siblings across the several training items that
    read from it. ``pair_dataset.__getitem__`` is pure tensor-level CPU work
    (no I/O, no model), so this is a constant-factor slowdown roughly
    proportional to the average CAV count per frame, not a new bottleneck
    class -- worth revisiting only if training throughput turns out to need
    it.
    """

    def __init__(self, pair_dataset: AlignFormerPVAPairDataset) -> None:
        self.pair_dataset = pair_dataset
        self._frame_groups: Dict[int, List[int]] = defaultdict(list)
        for flat_index, (frame_index, _ego_agent_index, _cav_agent_index) in enumerate(
            pair_dataset._pairs
        ):
            self._frame_groups[frame_index].append(flat_index)

    def set_epoch(self, epoch: int) -> None:
        self.pair_dataset.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.pair_dataset)

    def __getitem__(self, flat_index: int) -> dict:
        frame_index, _ego_agent_index, _cav_agent_index = self.pair_dataset._pairs[flat_index]
        sample = dict(self.pair_dataset[flat_index])

        sibling_boxes: List[Tensor] = []
        sibling_scores: List[Tensor] = []
        sibling_embeddings: List[Tensor] = []
        for sibling_index in self._frame_groups[frame_index]:
            if sibling_index == flat_index:
                continue
            sibling = self.pair_dataset[sibling_index]
            sibling_boxes.append(sibling["cav_boxes"])
            sibling_scores.append(sibling["cav_scores"])
            sibling_embeddings.append(sibling["cav_embeddings"])

        cav_boxes = sample["cav_boxes"]
        embed_width = sample["cav_embeddings"].shape[-1]
        if sibling_boxes:
            context_boxes = torch.cat(sibling_boxes, dim=0)
            context_scores = torch.cat(sibling_scores, dim=0)
            context_embeddings = torch.cat(sibling_embeddings, dim=0)
        else:
            context_boxes = cav_boxes.new_zeros((0, 7))
            context_scores = cav_boxes.new_zeros((0,))
            context_embeddings = cav_boxes.new_zeros((0, embed_width))
        sample["context_boxes"] = context_boxes
        sample["context_scores"] = context_scores
        sample["context_embeddings"] = context_embeddings
        sample["context_mask"] = torch.ones(context_boxes.shape[0], dtype=torch.bool)
        return sample


def collate_frames(samples: Sequence[dict]) -> dict:
    """``collate`` (unchanged, reused) plus padding for the new context fields."""
    batch = collate(samples)
    context_count = max(sample["context_boxes"].shape[0] for sample in samples)
    if context_count == 0:
        example = samples[0]
        batch["context_boxes"] = example["context_boxes"].new_zeros((len(samples), 0, 7))
        batch["context_scores"] = example["context_scores"].new_zeros((len(samples), 0))
        batch["context_embeddings"] = example["context_embeddings"].new_zeros(
            (len(samples), 0, example["context_embeddings"].shape[-1])
        )
        batch["context_mask"] = torch.zeros((len(samples), 0), dtype=torch.bool)
        return batch
    for key in _CONTEXT_FIELDS:
        pad_value = _CONTEXT_PAD_VALUES.get(key, 0)
        rows = []
        for sample in samples:
            value = sample[key]
            n = value.shape[0]
            if n < context_count:
                pad = value.new_full((context_count - n,) + tuple(value.shape[1:]), pad_value)
                value = torch.cat([value, pad], dim=0)
            rows.append(value)
        batch[key] = torch.stack(rows, dim=0)
    return batch
