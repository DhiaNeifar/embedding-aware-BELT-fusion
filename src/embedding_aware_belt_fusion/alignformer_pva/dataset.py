"""Training-pair dataset for AlignFormer PVA, built from OUR OWN frame cache.

``AlignFormerB.forward`` (model.py) needs ``ego_boxes``/``cav_boxes`` already
projected into the ego frame, plus -- for training -- ground-truth
correspondence indices and the SE(2) correction that undoes an injected
localization-noise error. This module builds exactly that, entirely from
infrastructure already in this repository:

* frame/agent records come from ``data/complete_cache.py``'s
  ``IndexedCompleteFrameDataset`` (built over
  ``embeddings/all_agent_association.py``'s ``records_for_role``), whose
  per-agent schema (``boxes`` in the agent's own local LiDAR frame, ``scores``,
  ``gt_ids`` -- matched physical object id or ``None`` per detection --, and
  ``transformation_matrix`` -- agent-local LiDAR to ego LiDAR, the TRUE,
  noise-free relative pose) is documented in
  ``data/materialize_roi_cache.py`` (its ``_FIELD_DESCRIPTIONS``-style
  comments) and already consumed the same way by
  ``embeddings/all_agent_association.py`` (``ego_index``,
  ``best_identity_indices``, ``binary_pair_loss_and_metrics``).
* the local-to-ego box projection reuses
  ``embeddings/geometry.py:transform_boxes_to_ego`` verbatim (confirmed
  signature: ``(boxes: (N, 7), transformation_matrix: (4, 4)) -> (N, 7)``,
  already used the same way by ``all_agent_association.py``).

No reusable "perturb a pose/transform by an xy/yaw std" utility exists
anywhere in ``src/`` (confirmed by grepping for the std-dev CLI flags, for
``loc_err``/``xyz_std``/``ryp_std``, and for ``randn``/``normal`` near any
transform/pose code): every place this repo injects localization noise
(``evaluation/belt_fusion.py:_set_noise``,
``data/extract_proposals.py``, ``evaluation/uncertainty_distribution.py``)
does it by forwarding ``xyz_std``/``ryp_std`` into OpenCOOD's own
``wild_setting``, so the actual noise sampling happens deep inside the
vendored OpenCOOD dataset (``external/OpenCOOD``), not as a standalone
function we could import here. ``_sample_relative_pose_noise`` /
``_apply_planar_noise`` below are therefore new, minimal, dependency-free
code, not a port of anything.

``correspondence_indices`` IS a port (verbatim, per the orchestrator's
instructions) of the labmate's dependency-free ``dataset.py`` matching logic.
``NoiseSchedule`` is a new, small reimplementation of the same *idea* as his
-- a linear ramp of noise std across epochs -- rather than a port, since units
and field names needed to match ours (metres / degrees, not his).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from embedding_aware_belt_fusion.alignformer_pva.trunk import MAX_OBJECTS
from embedding_aware_belt_fusion.embeddings.all_agent_association import ego_index
from embedding_aware_belt_fusion.embeddings.geometry import transform_boxes_to_ego

# Index of the yaw field in our own [x, y, z, h, w, l, yaw] box layout. Each
# module in alignformer_pva re-declares this locally rather than importing a
# shared constant (see trunk.py's own comment on this); followed here for
# consistency.
BOX_YAW = 6


def correspondence_indices(ego_ids, cav_ids):
    """Match detections across agents by physical object id.

    Ported verbatim from the labmate's ``dataset.py`` (given to this module
    by the orchestrator): pure, dependency-free index bookkeeping with no
    coupling to his ``coloca`` cache/geometry subsystem, so there is nothing
    here worth reimplementing differently.

    None means a detection matched no ground-truth object, so it can never
    correspond to anything. Duplicate ids keep only the first occurrence, so
    the result is a genuine one-to-one assignment. Returns (ego_match, cav_match),
    each a LongTensor, -1 where unmatched.
    """
    ego_match = torch.full((len(ego_ids),), -1, dtype=torch.long)
    cav_match = torch.full((len(cav_ids),), -1, dtype=torch.long)
    first_cav = {}
    for index, identifier in enumerate(cav_ids):
        if identifier is not None and identifier not in first_cav:
            first_cav[identifier] = index
    claimed = set()
    for row, identifier in enumerate(ego_ids):
        if identifier is None:
            continue
        cav_index = first_cav.get(identifier)
        if cav_index is None or cav_index in claimed:
            continue
        ego_match[row] = cav_index
        cav_match[cav_index] = row
        claimed.add(cav_index)
    return ego_match, cav_match


@dataclass
class NoiseSchedule:
    """Linear ramp of injected relative-pose noise std, from epoch 0 to ``ramp_epochs``.

    A tiny, dependency-free curriculum: epoch 0 samples noise at
    (``xy_std_start``, ``yaw_std_start_deg``); by ``ramp_epochs - 1`` (and
    every epoch after) it has linearly reached (``xy_std_end``,
    ``yaw_std_end_deg``). This is the same idea as the labmate's own
    ``NoiseSchedule`` -- curriculum noise is generic enough that reinventing
    its *shape* would be pointless -- but reimplemented rather than ported,
    since it must speak our field names/units (metres and degrees here).

    Use :meth:`fixed` for a constant std with no ramp at all, which is a
    perfectly reasonable choice too (see ``AlignFormerPVAPairDataset``'s own
    docstring for why a ramp is supported but not mandatory).
    """

    xy_std_start: float
    xy_std_end: float
    yaw_std_start_deg: float
    yaw_std_end_deg: float
    ramp_epochs: int = 1

    @classmethod
    def fixed(cls, xy_std: float, yaw_std_deg: float) -> "NoiseSchedule":
        """A constant-std schedule: ``set_epoch`` becomes a no-op in effect."""
        return cls(xy_std, xy_std, yaw_std_deg, yaw_std_deg, ramp_epochs=1)

    def stds(self, epoch: int) -> Tuple[float, float]:
        """Return ``(xy_std_metres, yaw_std_radians)`` for the given epoch."""
        if self.ramp_epochs <= 1:
            fraction = 1.0
        else:
            fraction = min(1.0, max(0.0, epoch / (self.ramp_epochs - 1)))
        xy_std = self.xy_std_start + fraction * (self.xy_std_end - self.xy_std_start)
        yaw_std_deg = (
            self.yaw_std_start_deg
            + fraction * (self.yaw_std_end_deg - self.yaw_std_start_deg)
        )
        return xy_std, math.radians(yaw_std_deg)


def _agent_has_boxes(agent: Mapping) -> bool:
    boxes = agent.get("boxes")
    return boxes is not None and boxes.shape[0] > 0


def _project_true(agent: Mapping) -> Tensor:
    """The agent's boxes in the ego frame, under the TRUE (noise-free) pose.

    ``agent["transformation_matrix"]`` is the cache's own ground-truth
    agent-local-to-ego-LiDAR transform (materialize_roi_cache.py), so this is
    exactly the "truth" the injected noise below is measured against.
    """
    return transform_boxes_to_ego(
        agent["boxes"].float(), agent["transformation_matrix"].float()
    )


def _truncate_top_n(
    boxes: Tensor, scores: Tensor, ids: Sequence, max_count: int
) -> Tuple[Tensor, Tensor, List, Tensor]:
    """Keep the ``max_count`` highest-score detections.

    Must run BEFORE ``correspondence_indices`` is called on the result: an
    index computed against the full set would point at positions a later
    truncation could remove, silently corrupting ``ego_match``/``cav_match``
    (the same ordering requirement the ported source repo's dataset notes
    about its own truncation step).

    Also returns the selected ``order`` (a ``LongTensor`` into the ORIGINAL,
    untruncated proposal array -- ``torch.arange(count)`` when nothing was
    truncated). ``AlignFormerPVAPairDataset`` itself has no use for this
    (its own boxes/scores/ids are already truncated together, in lockstep,
    inside this one function); ``AlignFormerPVAContextPairDataset`` needs it
    to apply the IDENTICAL selection to a cached agent's raw per-proposal
    cell tensors (``tokens``/``positions``/``footprint_flag``/``mask``),
    which this function has no knowledge of.
    """
    count = boxes.shape[0]
    if count <= max_count:
        return boxes, scores, list(ids), torch.arange(count)
    order = torch.argsort(scores, descending=True)[:max_count]
    truncated_ids = [ids[i] for i in order.tolist()]
    return boxes[order], scores[order], truncated_ids, order


def _rotation_matrix(angle: Tensor) -> Tensor:
    """``R(angle)``, matching ``losses.py:apply_se2``'s own convention."""
    cosine, sine = torch.cos(angle), torch.sin(angle)
    return torch.stack([torch.stack([cosine, -sine]), torch.stack([sine, cosine])])


def _wrap_angle(angle: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _apply_planar_noise(boxes_true_ego: Tensor, psi_noise: Tensor, t_noise: Tensor) -> Tensor:
    """Perturb already-ego-frame boxes by a sampled planar (SE(2)) pose error.

    This directly simulates "the CAV's boxes projected using a noisy relative
    pose": composing the sampled noise with the TRUE relative transform and
    then projecting from the local frame would give the identical result,
    since a planar pose error acts on (x, y, yaw) only and
    ``transform_boxes_to_ego`` is linear in the transform for a fixed set of
    local points -- this is just that same operation applied directly to the
    already-true-projected boxes, without an extra 4x4 matrix composition.
    """
    rotation = _rotation_matrix(psi_noise)
    noisy_centers = boxes_true_ego[:, :2] @ rotation.T + t_noise
    noisy_yaw = _wrap_angle(boxes_true_ego[:, BOX_YAW] + psi_noise)
    return torch.cat(
        [noisy_centers, boxes_true_ego[:, 2:BOX_YAW], noisy_yaw.unsqueeze(-1)],
        dim=-1,
    )


def _true_correction(psi_noise: Tensor, t_noise: Tensor) -> Tuple[Tensor, Tensor]:
    """The SE(2) correction that undoes a sampled noise ``(psi_noise, t_noise)``.

    By construction (see ``_apply_planar_noise``), the noisy boxes are
    ``R(psi_noise) @ true + t_noise``, so recovering ``true`` from the noisy
    boxes needs the inverse rigid motion: rotate by ``-psi_noise`` first,
    which cancels ``t_noise``'s own rotation too.
    """
    psi_true = _wrap_angle(-psi_noise)
    t_true = -(_rotation_matrix(psi_true) @ t_noise)
    return psi_true, t_true


class AlignFormerPVAPairDataset(Dataset):
    """One (ego, CAV) pose-correction training sample per item.

    Parameters
    ----------
    frames:
        Anything supporting ``len()`` and integer indexing to a frame dict
        shaped like ``data/complete_cache.py``'s
        ``IndexedCompleteFrameDataset`` items (and so, in particular, an
        actual ``IndexedCompleteFrameDataset`` built over
        ``embeddings/all_agent_association.py:records_for_role(...)``'s
        output works directly): ``{"agents": [agent, ...]}``, each agent a
        mapping with ``boxes`` ``(n, 7)`` in local LiDAR
        ``[x, y, z, h, w, l, yaw]``, ``scores`` ``(n,)``, ``gt_ids`` (length
        ``n``, each entry a matched physical object id or ``None``), and
        ``transformation_matrix`` ``(4, 4)`` (agent-local LiDAR to ego
        LiDAR, per ``data/materialize_roi_cache.py``). A plain Python list of
        such frame dicts (as the tests here use) works identically, which is
        why the type is left unannotated rather than pinned to
        ``IndexedCompleteFrameDataset``.
    noise_schedule:
        Controls injected relative-pose noise std as a function of the
        current epoch (see :meth:`set_epoch`). Defaults to a fixed,
        non-ramping 0.5 m / 2 degree std if omitted -- a modest but
        non-trivial localization error, in the same range this repo's own
        sweep scripts explore (``--position-noise-std``,
        ``--heading-noise-std-deg``) -- since *some* default is needed for
        the dataset to be directly usable without extra setup. Pass
        ``NoiseSchedule.fixed(xy_std, yaw_std_deg)`` for a different constant
        std, or a ramping ``NoiseSchedule`` for a curriculum.
    generator:
        Optional ``torch.Generator`` for reproducible noise sampling (the
        tests pass one with a fixed seed so the injected noise -- and hence
        the expected ``psi_true``/``t_true`` -- is deterministic).
    max_objects:
        Per-side truncation cap; defaults to ``trunk.MAX_OBJECTS`` so this
        dataset can never hand the model more objects than its own token
        budget, without hardcoding a second copy of that number.

    Design note -- why pairs are indexed eagerly at construction: each item
    here is one (ego, CAV) PAIR, not one frame (a frame can hold a variable
    number of CAVs), so ``__len__`` needs a flat pair count up front. This
    dataset builds that index once in ``__init__`` by reading every frame
    exactly once (through ``frames[i]``) and recording, per usable CAV, which
    frame and which CAV agent index it is -- skipping any CAV with zero
    detections and any frame whose own physical ego agent (found via
    ``all_agent_association.ego_index``, the same identity-transform lookup
    ``source_agent_pairs``/``all_agent_pairs`` already use) has zero
    detections, since neither side can be truncated/matched in that case.
    That is one full pass over the underlying cache before training starts;
    for a cache large enough that this pass is itself a bottleneck, a
    lighter-weight index (e.g. per-agent detection counts recorded in
    ``manifest.json`` at cache-build time) would be worth adding, but nothing
    in our data layer currently exposes that without reading the frame, and
    the frame shards are mmap-loaded (``complete_cache.py:_load_shard``), so
    the pass is cheap relative to actually training on the data.
    """

    def __init__(
        self,
        frames,
        noise_schedule: Optional[NoiseSchedule] = None,
        generator: Optional[torch.Generator] = None,
        max_objects: int = MAX_OBJECTS,
    ) -> None:
        self.frames = frames
        self.noise_schedule = noise_schedule or NoiseSchedule.fixed(0.5, 2.0)
        self.generator = generator
        self.max_objects = max_objects
        self.epoch = 0

        self._pairs: List[Tuple[int, int, int]] = []
        for frame_index in range(len(frames)):
            frame = frames[frame_index]
            agents = frame["agents"]
            if not agents:
                continue
            ego_agent_index = ego_index(frame)
            if not _agent_has_boxes(agents[ego_agent_index]):
                continue
            for agent_position, agent in enumerate(agents):
                if agent_position == ego_agent_index:
                    continue
                if not _agent_has_boxes(agent):
                    continue
                self._pairs.append((frame_index, ego_agent_index, agent_position))

    def set_epoch(self, epoch: int) -> None:
        """Advance the injected-noise std to the value scheduled for ``epoch``."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> dict:
        frame_index, ego_agent_index, cav_agent_index = self._pairs[index]
        frame = self.frames[frame_index]
        ego_agent = frame["agents"][ego_agent_index]
        cav_agent = frame["agents"][cav_agent_index]

        ego_boxes = _project_true(ego_agent)
        ego_scores = ego_agent["scores"].float()
        ego_ids = list(ego_agent["gt_ids"])
        ego_boxes, ego_scores, ego_ids, _ = _truncate_top_n(
            ego_boxes, ego_scores, ego_ids, self.max_objects
        )

        cav_boxes_true = _project_true(cav_agent)
        cav_scores = cav_agent["scores"].float()
        cav_ids = list(cav_agent["gt_ids"])
        cav_boxes_true, cav_scores, cav_ids, _ = _truncate_top_n(
            cav_boxes_true, cav_scores, cav_ids, self.max_objects
        )

        xy_std, yaw_std = self.noise_schedule.stds(self.epoch)
        psi_noise = torch.randn((), generator=self.generator) * yaw_std
        t_noise = torch.randn(2, generator=self.generator) * xy_std
        cav_boxes_noisy = _apply_planar_noise(cav_boxes_true, psi_noise, t_noise)
        psi_true, t_true = _true_correction(psi_noise, t_noise)

        ego_match, cav_match = correspondence_indices(ego_ids, cav_ids)

        ego_count, cav_count = ego_boxes.shape[0], cav_boxes_noisy.shape[0]
        return {
            "ego_boxes": ego_boxes,
            "cav_boxes": cav_boxes_noisy,
            "ego_scores": ego_scores,
            "cav_scores": cav_scores,
            "ego_mask": torch.ones(ego_count, dtype=torch.bool),
            "cav_mask": torch.ones(cav_count, dtype=torch.bool),
            "ego_embeddings": ego_boxes.new_zeros((ego_count, 0)),
            "cav_embeddings": cav_boxes_noisy.new_zeros((cav_count, 0)),
            "ego_match": ego_match,
            "cav_match": cav_match,
            "psi_true": psi_true,
            "t_true": t_true,
        }


# Pad value per per-object field; anything not listed defaults to 0 (boxes,
# scores, and the zero-width embeddings all pad correctly with 0).
_OBJECT_FIELD_PAD_VALUES = {
    "ego_match": -1,
    "cav_match": -1,
    "ego_mask": False,
    "cav_mask": False,
}
_EGO_FIELDS = ("ego_boxes", "ego_scores", "ego_embeddings", "ego_mask", "ego_match")
_CAV_FIELDS = ("cav_boxes", "cav_scores", "cav_embeddings", "cav_mask", "cav_match")


def _pad_field(samples: Sequence[Mapping[str, Tensor]], key: str, count: int) -> Tensor:
    pad_value = _OBJECT_FIELD_PAD_VALUES.get(key, 0)
    rows = []
    for sample in samples:
        value = sample[key]
        n = value.shape[0]
        if n < count:
            pad = value.new_full((count - n,) + tuple(value.shape[1:]), pad_value)
            value = torch.cat([value, pad], dim=0)
        rows.append(value)
    return torch.stack(rows, dim=0)


def collate(samples: Sequence[Mapping[str, Tensor]]) -> dict:
    """Pad a list of per-pair samples (:class:`AlignFormerPVAPairDataset` items)
    to a batch, producing ``ego_mask``/``cav_mask`` and stacking
    ``psi_true``/``t_true``.

    Every per-object field is padded up to the batch's own max object count
    on that side (``ego_boxes``/``ego_scores``/``ego_embeddings``/
    ``ego_mask``/``ego_match`` to the batch-max ego count, the ``cav_*``
    fields likewise), with 0 for geometry/score/embedding fields, ``-1`` for
    the match indices (an out-of-range index, so a padded slot can never be
    read as a real match), and ``False`` for the masks -- the same general
    shape as the labmate's own collate (pad dict + an arange-vs-count mask),
    reimplemented here directly against our field names/padding values
    instead of being ported, since the two schemas are not identical (we
    have no per-object embeddings to pad, for instance).
    """
    if not samples:
        raise ValueError("collate requires at least one sample")
    ego_count = max(sample["ego_boxes"].shape[0] for sample in samples)
    cav_count = max(sample["cav_boxes"].shape[0] for sample in samples)

    batch = {field: _pad_field(samples, field, ego_count) for field in _EGO_FIELDS}
    batch.update({field: _pad_field(samples, field, cav_count) for field in _CAV_FIELDS})
    batch["psi_true"] = torch.stack([sample["psi_true"] for sample in samples])
    batch["t_true"] = torch.stack([sample["t_true"] for sample in samples])
    return batch


# ---- Context-padded embedding variant (Phase 1 of the richer-embedding plan) ----
#
# AlignFormerPVAContextPairDataset reads the SAME frame/agent schema as the
# base class above, PLUS the raw per-proposal padded-cell tensors
# data/materialize_context_roi_cache.py writes onto every agent record
# (``tokens``/``positions``/``footprint_flag``/``mask``, on top of the same
# ``boxes``/``scores``/``gt_ids``/``transformation_matrix`` fields the
# geometry-only cache already has -- the context-padded cache is a strict
# superset, so it can be pointed at by ``--cache-dir`` in place of the
# geometry-only one with no second cache to cross-reference, which would
# otherwise require trusting that two SEPARATELY materialized caches keep an
# identical per-agent proposal order).
#
# This dataset does NOT itself run the embedding encoder
# (``embeddings/context_embedding.py:ContextEmbedding``): a ``Dataset``'s
# ``__getitem__``/``collate_fn`` runs in a plain CPU worker process with no
# model, no device, and (for a DataLoader worker) no shared autograd graph
# with the training process, so it can only ever hand back raw data. Running
# the encoder is ``alignformer_pva/train.py``'s job, as part of the SAME
# forward pass that calls ``AlignFormerB`` -- exactly where joint training
# needs it, since the encoder's parameters must receive gradients from
# ``match_nll``/``corner_loss`` same as the trunk's own parameters do.
_CELL_FIELD_SUFFIX_PAD_VALUES = {
    "mask": True,
    "footprint": False,
}
_EGO_CELL_FIELDS = (
    "ego_cell_tokens",
    "ego_cell_positions",
    "ego_cell_footprint",
    "ego_cell_mask",
)
_CAV_CELL_FIELDS = (
    "cav_cell_tokens",
    "cav_cell_positions",
    "cav_cell_footprint",
    "cav_cell_mask",
)


def _cell_pad_value(key: str) -> object:
    suffix = key.rsplit("_", 1)[-1]
    return _CELL_FIELD_SUFFIX_PAD_VALUES.get(suffix, 0)


def _agent_cell_fields(prefix: str, agent: Mapping, order: Tensor) -> dict:
    """Slice one agent's raw padded-cell tensors by the SAME ``order`` the
    geometry side (``_truncate_top_n``) already selected, and prefix the keys
    ``ego``/``cav`` to match ``collate_context``'s field tables above."""
    return {
        f"{prefix}_cell_tokens": agent["tokens"][order],
        f"{prefix}_cell_positions": agent["positions"][order],
        f"{prefix}_cell_footprint": agent["footprint_flag"][order],
        f"{prefix}_cell_mask": agent["mask"][order],
    }


class AlignFormerPVAContextPairDataset(AlignFormerPVAPairDataset):
    """:class:`AlignFormerPVAPairDataset`, plus raw per-proposal padded-cell tensors.

    Identical noise injection / correspondence / truncation logic to the base
    class (reused directly, not reimplemented, via the same free functions:
    ``_project_true``, ``_truncate_top_n``, ``_apply_planar_noise``,
    ``_true_correction``, ``correspondence_indices``) -- the only addition is
    carrying each selected proposal's padded-cell tensors alongside its
    geometry, so a downstream ``ContextEmbedding`` encoder has everything it
    needs without a second, separately-indexed cache lookup.

    ``ego_embeddings``/``cav_embeddings`` are still emitted as zero-width
    placeholders here, exactly like the base class: this keeps
    :func:`collate` (reused unchanged inside :func:`collate_context`) working
    without a parallel reimplementation, and is harmless since
    ``alignformer_pva/train.py`` overwrites both with the encoder's real
    output before the model ever sees the batch.
    """

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        frame_index, ego_agent_index, cav_agent_index = self._pairs[index]
        frame = self.frames[frame_index]
        ego_agent = frame["agents"][ego_agent_index]
        cav_agent = frame["agents"][cav_agent_index]

        # Recompute the same top-N order super().__getitem__ already used
        # internally (same inputs, same deterministic argsort), since the
        # base class does not expose it. Scores/counts come straight from the
        # cached agent, unaffected by noise or ego-frame projection, so this
        # is exactly the order _truncate_top_n applied, not a new selection.
        _, _, _, ego_order = _truncate_top_n(
            ego_agent["boxes"], ego_agent["scores"].float(),
            list(ego_agent["gt_ids"]), self.max_objects,
        )
        _, _, _, cav_order = _truncate_top_n(
            cav_agent["boxes"], cav_agent["scores"].float(),
            list(cav_agent["gt_ids"]), self.max_objects,
        )

        sample.update(_agent_cell_fields("ego", ego_agent, ego_order))
        sample.update(_agent_cell_fields("cav", cav_agent, cav_order))
        return sample


def _pad_cell_field(
    samples: Sequence[Mapping[str, Tensor]], key: str, object_count: int, cell_count: int
) -> Tensor:
    """Pad one ``(n, k, ...)`` per-sample cell field to ``(object_count, cell_count, ...)``
    before stacking into a ``(B, object_count, cell_count, ...)`` batch tensor.

    Two independent ragged dimensions need padding here (object count AND
    cell count per object), unlike :func:`_pad_field`'s single ragged
    dimension, so cell-dimension padding is applied first and object-
    dimension padding second; the pad value is the same per-side-matched
    convention :func:`_agent_cell_fields`/``materialize_context_roi_cache.py``
    already use (``True`` for ``*_mask`` -- a padded slot is never a real
    cell -- ``False`` for ``*_footprint``, ``0`` for ``tokens``/``positions``).
    A fully object-padded slot therefore comes back with an all-``True`` cell
    mask, which is exactly the fully-masked-row case
    ``ContextEmbedding`` already guards against (a harmless zero embedding),
    so no special case is needed here for it.
    """
    pad_value = _cell_pad_value(key)
    rows = []
    for sample in samples:
        value = sample[key]
        object_n, cell_n = value.shape[0], value.shape[1]
        if cell_n < cell_count:
            pad_cells = value.new_full(
                (object_n, cell_count - cell_n) + tuple(value.shape[2:]), pad_value
            )
            value = torch.cat([value, pad_cells], dim=1)
        if object_n < object_count:
            pad_objects = value.new_full(
                (object_count - object_n, cell_count) + tuple(value.shape[2:]), pad_value
            )
            value = torch.cat([value, pad_objects], dim=0)
        rows.append(value)
    return torch.stack(rows, dim=0)


def collate_context(samples: Sequence[Mapping[str, Tensor]]) -> dict:
    """:func:`collate`, plus padding :class:`AlignFormerPVAContextPairDataset`'s
    raw per-proposal padded-cell tensors to the batch's max object count
    (same as every other per-object field) AND the batch's max cell count per
    side.

    Reuses :func:`collate` unchanged for every field it already knows about
    (including the zero-width ``ego_embeddings``/``cav_embeddings``
    placeholders, which ``alignformer_pva/train.py`` overwrites with the
    encoder's real output before the model forward) rather than
    reimplementing that padding a second time, and only adds the two new
    ``*_cell_*`` field groups.
    """
    if not samples:
        raise ValueError("collate_context requires at least one sample")
    cell_keys = _EGO_CELL_FIELDS + _CAV_CELL_FIELDS
    base_samples = [
        {key: value for key, value in sample.items() if key not in cell_keys}
        for sample in samples
    ]
    batch = collate(base_samples)

    ego_cell_count = max(sample["ego_cell_tokens"].shape[1] for sample in samples)
    cav_cell_count = max(sample["cav_cell_tokens"].shape[1] for sample in samples)
    ego_object_count = batch["ego_boxes"].shape[1]
    cav_object_count = batch["cav_boxes"].shape[1]

    for key in _EGO_CELL_FIELDS:
        batch[key] = _pad_cell_field(samples, key, ego_object_count, ego_cell_count)
    for key in _CAV_CELL_FIELDS:
        batch[key] = _pad_cell_field(samples, key, cav_object_count, cav_cell_count)
    return batch
