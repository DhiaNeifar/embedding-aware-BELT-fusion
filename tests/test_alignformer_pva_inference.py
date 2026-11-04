import math

import torch

from embedding_aware_belt_fusion.alignformer_pva.abstain import AbstentionConfig
from embedding_aware_belt_fusion.alignformer_pva.inference import (
    associate_simple_by_geometry,
    build_pairwise_batch,
    correct_simple_detections,
    estimate_correction,
    fuse_corrected_simple_detections,
)
from embedding_aware_belt_fusion.alignformer_pva.model import AlignFormerB
from embedding_aware_belt_fusion.alignformer_pva.refine import (
    AgreementConfig,
    ICP,
    ICP_RANSAC,
    RefineConfig,
)
from embedding_aware_belt_fusion.integration.localization import apply_se2_correction


def _rotation(yaw: float) -> torch.Tensor:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return torch.tensor([[cosine, -sine], [sine, cosine]])


def _detection(boxes, scores=None, agent_id=None, oracle_object_ids=None):
    boxes = boxes if torch.is_tensor(boxes) else torch.tensor(boxes, dtype=torch.float32)
    boxes = boxes.reshape(-1, 7).to(torch.float32)
    count = len(boxes)
    detection = {
        "boxes": boxes,
        "scores": (
            torch.tensor(scores, dtype=torch.float32)
            if scores is not None
            else torch.full((count,), 0.8)
        ),
    }
    if agent_id is not None:
        detection["agent_id"] = agent_id
    if oracle_object_ids is not None:
        detection["oracle_object_ids"] = oracle_object_ids
    return detection


# ---- build_pairwise_batch ---------------------------------------------------


def test_build_pairwise_batch_has_correct_shapes_dtype_and_zero_width_embeddings():
    ego = _detection([[0.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0], [5.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]])
    cav = _detection([[0.2, 0.1, 0.0, 1.5, 1.6, 4.0, 0.05]])
    batch = build_pairwise_batch(ego, cav)

    assert batch["ego_boxes"].shape == (1, 2, 7)
    assert batch["cav_boxes"].shape == (1, 1, 7)
    assert batch["ego_scores"].shape == (1, 2)
    assert batch["cav_scores"].shape == (1, 1)
    assert batch["ego_mask"].shape == (1, 2)
    assert batch["cav_mask"].shape == (1, 1)
    assert batch["ego_mask"].dtype == torch.bool
    assert bool(batch["ego_mask"].all())
    assert bool(batch["cav_mask"].all())
    # Geometry-only backend: embeddings are zero-width, not absent.
    assert batch["ego_embeddings"].shape == (1, 2, 0)
    assert batch["cav_embeddings"].shape == (1, 1, 0)


def test_build_pairwise_batch_tolerates_an_empty_side():
    ego = _detection([[0.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]])
    empty_cav = _detection(torch.zeros((0, 7)), scores=[])
    batch = build_pairwise_batch(ego, empty_cav)
    assert batch["cav_boxes"].shape == (1, 0, 7)
    assert batch["cav_mask"].shape == (1, 0)


# ---- estimate_correction: end-to-end Propose -> Verify -> Abstain ----------


def _model():
    torch.manual_seed(0)
    return AlignFormerB(embed_dim=0, layers=1, heads=2, model_dim=16)


def test_estimate_correction_returns_none_when_a_side_is_empty():
    model = _model()
    ego = _detection([[0.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]])
    empty_cav = _detection(torch.zeros((0, 7)), scores=[])
    correction, diagnostics = estimate_correction(
        model,
        ego,
        empty_cav,
        refine_config=RefineConfig(mode=ICP),
        abstain_config=AbstentionConfig(mode="abstain", level=0.05),
    )
    assert correction is None
    assert diagnostics["attempted"] is False
    assert diagnostics["accepted"] is False


def test_estimate_correction_finite_and_shaped_with_an_untrained_model():
    model = _model()
    ego = _detection(
        [[0.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0], [5.0, 1.0, 0.0, 1.5, 1.6, 4.0, 0.3]]
    )
    cav = _detection(
        [[0.3, -0.2, 0.0, 1.5, 1.6, 4.0, 0.05], [5.4, 0.7, 0.0, 1.5, 1.6, 4.0, 0.25]]
    )
    correction, diagnostics = estimate_correction(
        model,
        ego,
        cav,
        refine_config=RefineConfig(mode=ICP_RANSAC, min_pairs=1, gates_m=(5.0, 2.0, 1.0)),
        abstain_config=AbstentionConfig(mode="abstain", level=0.05),
    )
    assert correction is not None
    assert correction["rotation"].shape == (2, 2)
    assert correction["translation"].shape == (2,)
    assert torch.isfinite(correction["rotation"]).all()
    assert torch.isfinite(correction["translation"]).all()
    assert torch.isfinite(correction["yaw_correction"])
    assert diagnostics["attempted"] is True
    assert isinstance(diagnostics["accepted"], bool)
    assert diagnostics["statistic"] is not None
    assert math.isfinite(diagnostics["statistic"])


def test_abstain_disabled_always_accepts_the_verify_stage_correction():
    model = _model()
    ego = _detection([[0.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0], [5.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]])
    cav = _detection([[0.01, 0.01, 0.0, 1.5, 1.6, 4.0, 0.0], [5.01, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]])
    correction, diagnostics = estimate_correction(
        model,
        ego,
        cav,
        refine_config=RefineConfig(mode=ICP, min_pairs=1, gates_m=(5.0, 2.0, 1.0)),
        abstain_config=AbstentionConfig(mode="none"),
    )
    assert correction is not None
    assert diagnostics["accepted"] is True


def test_abstain_rejects_when_level_is_vanishingly_strict():
    # At level -> 0 the threshold -> infinity, so even a real, well-estimated
    # correction must be rejected: the gate must be able to say no.
    model = _model()
    ego = _detection([[0.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0], [5.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]])
    cav = _detection([[0.01, 0.01, 0.0, 1.5, 1.6, 4.0, 0.0], [5.01, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]])
    correction, diagnostics = estimate_correction(
        model,
        ego,
        cav,
        refine_config=RefineConfig(mode=ICP, min_pairs=1, gates_m=(5.0, 2.0, 1.0)),
        abstain_config=AbstentionConfig(mode="abstain", level=1e-12),
    )
    assert correction is not None
    assert diagnostics["accepted"] is False


# ---- estimate_correction: the optional agreement-rule wiring --------------


def test_estimate_correction_without_agreement_config_is_wald_only_by_default():
    # agreement_config defaults to None -- omitting it entirely (every
    # pre-existing call site) must behave exactly like passing it explicitly.
    model = _model()
    ego = _detection([[0.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0], [5.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]])
    cav = _detection([[0.01, 0.01, 0.0, 1.5, 1.6, 4.0, 0.0], [5.01, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]])
    refine_config = RefineConfig(mode=ICP, min_pairs=1, gates_m=(5.0, 2.0, 1.0))
    abstain_config = AbstentionConfig(mode="abstain", level=0.05)

    omitted_correction, omitted_diag = estimate_correction(
        model, ego, cav, refine_config=refine_config, abstain_config=abstain_config
    )
    explicit_correction, explicit_diag = estimate_correction(
        model, ego, cav, refine_config=refine_config, abstain_config=abstain_config,
        agreement_config=None,
    )
    assert omitted_diag["accepted"] == explicit_diag["accepted"]
    assert torch.equal(omitted_correction["translation"], explicit_correction["translation"])
    assert torch.equal(omitted_correction["rotation"], explicit_correction["rotation"])


def test_estimate_correction_agreement_consensus_floor_overrides_strict_wald_rejection():
    # Chenyi's own "agreement with a consensus floor" (sec/3_method.tex,
    # "Abstain"): the Verify-stage exact fit is accepted once its own
    # hard-pair consensus clears the floor, even when the Wald test alone
    # would reject everything at a vanishingly strict level.
    model = _model()
    ego = _detection([[0.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0], [5.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]])
    cav = _detection([[0.01, 0.01, 0.0, 1.5, 1.6, 4.0, 0.0], [5.01, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]])
    refine_config = RefineConfig(mode=ICP, min_pairs=1, gates_m=(5.0, 2.0, 1.0))
    strict_abstain = AbstentionConfig(mode="abstain", level=1e-12)

    _, wald_only_diag = estimate_correction(
        model, ego, cav, refine_config=refine_config, abstain_config=strict_abstain,
    )
    assert wald_only_diag["accepted"] is False

    _, with_agreement_diag = estimate_correction(
        model, ego, cav, refine_config=refine_config, abstain_config=strict_abstain,
        agreement_config=AgreementConfig(tolerance_m=1e-6, consensus_floor=1),
    )
    assert with_agreement_diag["accepted"] is True


# ---- correct_simple_detections: apply-or-leave-uncorrected wiring ----------


def test_correct_simple_detections_leaves_ego_untouched_and_corrects_accepted_cavs():
    model = _model()
    simple_detections_by_cav = {
        "ego": _detection(
            [[0.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0], [5.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]],
            agent_id="ego",
        ),
        "cav1": _detection(
            [[0.01, 0.01, 0.0, 1.5, 1.6, 4.0, 0.0], [5.01, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]],
            agent_id="cav1",
        ),
    }
    corrected, diagnostics = correct_simple_detections(
        model,
        simple_detections_by_cav,
        ["ego", "cav1"],
        refine_config=RefineConfig(mode=ICP, min_pairs=1, gates_m=(5.0, 2.0, 1.0)),
        abstain_config=AbstentionConfig(mode="none"),
    )
    assert corrected["ego"] is simple_detections_by_cav["ego"]
    assert "cav1" in diagnostics
    if diagnostics["cav1"]["accepted"]:
        assert not torch.equal(
            corrected["cav1"]["boxes"], simple_detections_by_cav["cav1"]["boxes"]
        )
    else:
        assert corrected["cav1"] is simple_detections_by_cav["cav1"]


def test_correct_simple_detections_rejects_leave_detection_bit_identical():
    model = _model()
    simple_detections_by_cav = {
        "ego": _detection([[0.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]], agent_id="ego"),
        "cav1": _detection([[10.0, 10.0, 0.0, 1.5, 1.6, 4.0, 0.0]], agent_id="cav1"),
    }
    corrected, diagnostics = correct_simple_detections(
        model,
        simple_detections_by_cav,
        ["ego", "cav1"],
        refine_config=RefineConfig(mode=ICP, min_pairs=1, gates_m=(5.0, 2.0, 1.0)),
        # A level this strict can never be cleared: forces rejection so the
        # "leave detections uncorrected" branch is exercised deterministically.
        abstain_config=AbstentionConfig(mode="abstain", level=1e-12),
    )
    assert diagnostics["cav1"]["accepted"] is False
    assert corrected["cav1"] is simple_detections_by_cav["cav1"]
    assert torch.equal(corrected["cav1"]["boxes"], simple_detections_by_cav["cav1"]["boxes"])


def test_correct_simple_detections_raises_without_an_ego_entry():
    model = _model()
    simple_detections_by_cav = {
        "cav1": _detection([[0.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]], agent_id="cav1"),
    }
    try:
        correct_simple_detections(
            model,
            simple_detections_by_cav,
            ["cav1"],
            refine_config=RefineConfig(mode=ICP),
            abstain_config=AbstentionConfig(mode="none"),
        )
        assert False, "expected a RuntimeError for a batch with no ego agent"
    except RuntimeError:
        pass


# ---- associate_simple_by_geometry / fuse_corrected_simple_detections -------


def test_associate_simple_by_geometry_groups_nearby_boxes_across_agents():
    ego = _detection(
        [[0.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0], [20.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]]
    )
    cav = _detection(
        [[0.2, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0], [20.1, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]]
    )
    groups = associate_simple_by_geometry([ego, cav], maximum_distance=1.0)
    assert sorted(len(group) for group in groups) == [2, 2]


def test_associate_simple_by_geometry_keeps_far_boxes_as_singletons():
    ego = _detection([[0.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]])
    cav = _detection([[50.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]])
    groups = associate_simple_by_geometry([ego, cav], maximum_distance=1.0)
    assert sorted(len(group) for group in groups) == [1, 1]


def test_fuse_corrected_simple_detections_score_weights_matched_boxes():
    ego = _detection([[0.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]], scores=[0.8], agent_id="ego")
    cav = _detection([[2.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]], scores=[0.2], agent_id="cav1")
    fused = fuse_corrected_simple_detections(
        {"ego": ego, "cav1": cav}, ["ego", "cav1"], maximum_distance=5.0
    )
    assert int(fused["member_counts"][0]) == 2
    # Score weights 0.8 and 0.2: x = (0*0.8 + 2*0.2) / (0.8+0.2) = 0.4.
    assert torch.allclose(fused["boxes"][0, 0], torch.tensor(0.4))


# ---- correction-application wiring reuses apply_se2_correction verbatim ----


def test_correction_from_estimate_correction_is_a_valid_apply_se2_correction_input():
    # A correction built by estimate_correction must be structurally usable
    # by the exact same apply_se2_correction the RANSAC path calls.
    model = _model()
    ego = _detection([[0.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0], [5.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]])
    cav = _detection([[0.01, 0.01, 0.0, 1.5, 1.6, 4.0, 0.0], [5.01, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]])
    correction, _ = estimate_correction(
        model,
        ego,
        cav,
        refine_config=RefineConfig(mode=ICP, min_pairs=1, gates_m=(5.0, 2.0, 1.0)),
        abstain_config=AbstentionConfig(mode="none"),
    )
    corrected = apply_se2_correction(cav, correction)
    assert corrected["boxes"].shape == cav["boxes"].shape
    assert torch.isfinite(corrected["boxes"]).all()
