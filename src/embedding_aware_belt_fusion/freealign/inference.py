"""Inference-time glue between the evaluation harness and FreeAlign.

Mirrors ``alignformer_pva/inference.py``'s shape closely (same
``simple_detections_by_cav``/``cav_order`` input, same correction dict
``apply_se2_correction`` expects, same ``CorrectionDiagnostics``-style
record) so ``evaluation/belt_fusion.py`` can wire FreeAlign in with the same
pattern -- but FreeAlign is training-free, so there is no model, no
Verify-stage re-solve, and no Wald test: the whole Propose/Abstain step is
one call to :func:`freealign.freealign_estimate`, whose own ``min_nodes``
gate IS the abstain decision.

The one place this module reuses ``alignformer_pva`` code is the
multi-agent grouping step (:func:`associate_simple_by_geometry`) and the
score-weighted fusion (:func:`fuse_groups_score_weighted`): both are generic
post-correction bookkeeping that every corrected-detections method in this
repo shares (``alignformer_pva``'s own row reuses the SAME
``fuse_groups_score_weighted`` from ``integration/belt_fusion.py``), not
pose-fitting logic, so reusing them here does not blur the comparison the
way reusing the Propose/Verify/Abstain code would. :mod:`freealign.freealign`
itself still imports nothing from ``alignformer_pva`` (see that module's own
docstring and ``tests/test_freealign.py``'s
``test_freealign_module_imports_nothing_from_alignformer_pva``).
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import torch
from torch import Tensor

from embedding_aware_belt_fusion.alignformer_pva.inference import (
    associate_simple_by_geometry,
)
from embedding_aware_belt_fusion.freealign.freealign import (
    FreeAlignConfig,
    freealign_estimate,
    is_fallback,
)
from embedding_aware_belt_fusion.integration.belt_fusion import (
    fuse_groups_score_weighted,
)
from embedding_aware_belt_fusion.integration.localization import apply_se2_correction

DEFAULT_ASSOCIATION_DISTANCE_M = 5.0


def build_pairwise_batch(
    ego_detection: Mapping[str, Tensor], cav_detection: Mapping[str, Tensor]
) -> Dict[str, Tensor]:
    """Build one ``freealign_estimate`` batch (size 1) from live detections.

    ``ego_detection``/``cav_detection`` are one CAV's entry in
    ``simple_detections_by_cav``: ``{"boxes": (N, 7), "scores": (N,), ...}``,
    boxes already in the ego frame. FreeAlign reads only geometry
    (``*_boxes``/``*_mask``), so unlike ``alignformer_pva``'s batch builder
    this carries no scores/embeddings fields at all.
    """
    ego_boxes = ego_detection["boxes"]
    cav_boxes = cav_detection["boxes"]
    device, dtype = ego_boxes.device, ego_boxes.dtype
    return {
        "ego_boxes": ego_boxes.unsqueeze(0),
        "cav_boxes": cav_boxes.to(device=device, dtype=dtype).unsqueeze(0),
        "ego_mask": torch.ones(1, ego_boxes.shape[0], dtype=torch.bool, device=device),
        "cav_mask": torch.ones(1, cav_boxes.shape[0], dtype=torch.bool, device=device),
    }


class CorrectionDiagnostics(dict):
    """Lightweight record of one CAV's FreeAlign outcome.

    A plain ``dict`` subclass (rather than a dataclass) so it round-trips
    into ``metrics.json`` without a custom encoder, mirroring
    ``alignformer_pva.inference.CorrectionDiagnostics``: keys are
    ``attempted`` (bool), ``accepted`` (bool), and ``matched_nodes`` (float,
    the size of the common subgraph MASS found -- FreeAlign's own evidence
    quantity, the analogue of AlignFormer-PVA's Wald ``statistic``).
    """


def estimate_correction(
    ego_detection: Mapping[str, Tensor],
    cav_detection: Mapping[str, Tensor],
    config: FreeAlignConfig,
) -> Tuple[dict | None, CorrectionDiagnostics]:
    """Run FreeAlign for one ego/CAV pair.

    Returns ``(correction, diagnostics)``. ``correction`` is ``None`` only
    when there is nothing to estimate from at all (either side detected zero
    objects this frame); otherwise it is always a dict with the same
    ``rotation``/``translation``/``yaw_correction`` keys
    ``apply_se2_correction`` expects, so the caller can apply it
    unconditionally when ``diagnostics["accepted"]`` is true and must leave
    the detection untouched otherwise. ``accepted`` is FreeAlign's own
    abstain rule (:func:`freealign.is_fallback`): below ``config.min_nodes``
    matched objects the message is discarded.
    """
    batch = build_pairwise_batch(ego_detection, cav_detection)
    if batch["ego_boxes"].shape[1] == 0 or batch["cav_boxes"].shape[1] == 0:
        return None, CorrectionDiagnostics(attempted=False, accepted=False, matched_nodes=0.0)

    estimate = freealign_estimate(batch, config)
    accepted = not bool(is_fallback(estimate)[0])
    psi = estimate.psi[0]
    cosine, sine = torch.cos(psi), torch.sin(psi)
    rotation = torch.stack(
        [torch.stack([cosine, -sine]), torch.stack([sine, cosine])]
    )
    correction = {
        "rotation": rotation,
        "translation": estimate.t[0],
        "yaw_correction": psi,
    }
    diagnostics = CorrectionDiagnostics(
        attempted=True,
        accepted=accepted,
        matched_nodes=float(estimate.confidence[0]),
    )
    return correction, diagnostics


def correct_simple_detections(
    simple_detections_by_cav: Mapping[str, Mapping[str, Tensor]],
    cav_order: Sequence[str],
    config: FreeAlignConfig,
) -> Tuple[Dict[str, Mapping[str, Tensor]], Dict[str, CorrectionDiagnostics]]:
    """Apply a per-CAV FreeAlign correction to one frame's simple detections.

    Same contract as ``alignformer_pva.inference.correct_simple_detections``:
    ego's own entry is passed through unchanged, every other CAV is
    corrected against ego when FreeAlign's own abstain rule accepted the
    estimate, applied via the same ``apply_se2_correction`` every alignment
    method in this repo uses. A rejected (or unattempted) CAV keeps its
    original, uncorrected entry.
    """
    if "ego" not in simple_detections_by_cav:
        raise RuntimeError("Late-fusion batch has no ego agent")
    ego_detection = simple_detections_by_cav["ego"]
    corrected = dict(simple_detections_by_cav)
    diagnostics: Dict[str, CorrectionDiagnostics] = {}
    for cav_id in cav_order:
        if cav_id == "ego":
            continue
        cav_detection = simple_detections_by_cav[cav_id]
        correction, stats = estimate_correction(ego_detection, cav_detection, config)
        diagnostics[cav_id] = stats
        if correction is not None and stats["accepted"]:
            corrected[cav_id] = apply_se2_correction(cav_detection, correction)
    return corrected, diagnostics


def fuse_corrected_simple_detections(
    corrected_detections_by_cav: Mapping[str, Mapping[str, Tensor]],
    cav_order: Sequence[str],
    *,
    maximum_distance: float = DEFAULT_ASSOCIATION_DISTANCE_M,
) -> Dict[str, Tensor]:
    """Associate corrected simple detections and fuse by detector-score weighting.

    Thin wrapper over ``alignformer_pva.inference.associate_simple_by_geometry``
    + ``fuse_groups_score_weighted`` -- the SAME grouping/fusion code every
    other corrected-detections row in this repo uses (see module docstring
    for why reusing this, specifically, does not blur the comparison).
    """
    detections = [corrected_detections_by_cav[cav_id] for cav_id in cav_order]
    groups = associate_simple_by_geometry(detections, maximum_distance=maximum_distance)
    return fuse_groups_score_weighted(groups)
