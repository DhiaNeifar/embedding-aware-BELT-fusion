import math

import torch

from embedding_aware_belt_fusion.alignformer_pva.abstain import AbstentionConfig
from embedding_aware_belt_fusion.alignformer_pva.inference import (
    build_pairwise_batch,
    correct_simple_detections,
    estimate_correction,
)
from embedding_aware_belt_fusion.alignformer_pva.model import AlignFormerB
from embedding_aware_belt_fusion.alignformer_pva.refine import ICP, RefineConfig
from embedding_aware_belt_fusion.embeddings.context_embedding import ContextEmbedding

INPUT_DIM = 6


def _detection_with_cells(boxes, scores=None, agent_id=None, max_cells=4, seed=0):
    boxes = boxes if torch.is_tensor(boxes) else torch.tensor(boxes, dtype=torch.float32)
    boxes = boxes.reshape(-1, 7).to(torch.float32)
    count = len(boxes)
    generator = torch.Generator().manual_seed(seed)
    detection = {
        "boxes": boxes,
        "scores": (
            torch.tensor(scores, dtype=torch.float32)
            if scores is not None
            else torch.full((count,), 0.8)
        ),
        "cell_tokens": torch.randn(count, max_cells, INPUT_DIM, generator=generator),
        "cell_positions": torch.randn(count, max_cells, 2, generator=generator),
        "cell_footprint": torch.zeros(count, max_cells, dtype=torch.bool),
        "cell_mask": torch.zeros(count, max_cells, dtype=torch.bool),
    }
    if count:
        detection["cell_footprint"][:, :1] = True
    if agent_id is not None:
        detection["agent_id"] = agent_id
    return detection


def _encoder(embed_dim=5):
    torch.manual_seed(1)
    return ContextEmbedding(
        input_dim=INPUT_DIM, d_model=16, embed_dim=embed_dim, heads=2, layers=1,
        feedforward_dim=32, dropout=0.0,
    )


def _model(embed_dim=0):
    torch.manual_seed(0)
    return AlignFormerB(embed_dim=embed_dim, layers=1, heads=2, model_dim=16)


def test_build_pairwise_batch_uses_given_embeddings_when_provided():
    ego = _detection_with_cells([[0.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]])
    cav = _detection_with_cells([[0.2, 0.1, 0.0, 1.5, 1.6, 4.0, 0.05]])
    ego_embeddings = torch.randn(1, 5)
    cav_embeddings = torch.randn(1, 5)

    batch = build_pairwise_batch(ego, cav, ego_embeddings, cav_embeddings)

    assert batch["ego_embeddings"].shape == (1, 1, 5)
    assert torch.equal(batch["ego_embeddings"][0], ego_embeddings)
    assert torch.equal(batch["cav_embeddings"][0], cav_embeddings)


def test_build_pairwise_batch_none_embeddings_matches_original_zero_width_behavior():
    ego = _detection_with_cells([[0.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]])
    cav = _detection_with_cells([[0.2, 0.1, 0.0, 1.5, 1.6, 4.0, 0.05]])

    batch = build_pairwise_batch(ego, cav)

    assert batch["ego_embeddings"].shape == (1, 1, 0)
    assert batch["cav_embeddings"].shape == (1, 1, 0)


def test_estimate_correction_with_embedding_encoder_runs_finite_end_to_end():
    model = _model(embed_dim=5)
    encoder = _encoder(embed_dim=5)
    ego = _detection_with_cells(
        [[0.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0], [5.0, 1.0, 0.0, 1.5, 1.6, 4.0, 0.3]], seed=1
    )
    cav = _detection_with_cells(
        [[0.3, -0.2, 0.0, 1.5, 1.6, 4.0, 0.05], [5.4, 0.7, 0.0, 1.5, 1.6, 4.0, 0.25]], seed=2
    )

    correction, diagnostics = estimate_correction(
        model, ego, cav,
        refine_config=RefineConfig(mode=ICP, min_pairs=1, gates_m=(5.0, 2.0, 1.0)),
        abstain_config=AbstentionConfig(mode="none"),
        embedding_encoder=encoder,
    )

    assert correction is not None
    assert torch.isfinite(correction["translation"]).all()
    assert diagnostics["attempted"] is True


def test_estimate_correction_without_embedding_encoder_is_unaffected_by_cell_data():
    # Detections carry cell_* fields (as belt_fusion.py's live path always
    # would once embeddings are ever used elsewhere in a run), but
    # embedding_encoder=None must still take the pure geometry-only path --
    # the cell fields must be silently ignored, not read.
    model = _model(embed_dim=0)
    ego = _detection_with_cells([[0.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]])
    cav = _detection_with_cells([[0.2, 0.1, 0.0, 1.5, 1.6, 4.0, 0.05]])

    correction, diagnostics = estimate_correction(
        model, ego, cav,
        refine_config=RefineConfig(mode=ICP, min_pairs=1, gates_m=(5.0, 2.0, 1.0)),
        abstain_config=AbstentionConfig(mode="none"),
    )
    assert correction is not None
    assert diagnostics["attempted"] is True


def test_correct_simple_detections_threads_embedding_encoder_through_to_every_cav():
    model = _model(embed_dim=5)
    encoder = _encoder(embed_dim=5)
    simple_detections_by_cav = {
        "ego": _detection_with_cells(
            [[0.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0], [5.0, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]],
            agent_id="ego", seed=1,
        ),
        "cav1": _detection_with_cells(
            [[0.01, 0.01, 0.0, 1.5, 1.6, 4.0, 0.0], [5.01, 0.0, 0.0, 1.5, 1.6, 4.0, 0.0]],
            agent_id="cav1", seed=2,
        ),
    }
    corrected, diagnostics = correct_simple_detections(
        model, simple_detections_by_cav, ["ego", "cav1"],
        refine_config=RefineConfig(mode=ICP, min_pairs=1, gates_m=(5.0, 2.0, 1.0)),
        abstain_config=AbstentionConfig(mode="none"),
        embedding_encoder=encoder,
    )
    assert corrected["ego"] is simple_detections_by_cav["ego"]
    assert "cav1" in diagnostics
    assert diagnostics["cav1"]["attempted"] is True


def test_embed_detection_raises_a_clear_error_when_cell_fields_are_missing():
    model = _model(embed_dim=5)
    encoder = _encoder(embed_dim=5)
    # Plain geometry-only detections, as naive/ego_only/belt_geometry/the
    # existing geometry-only alignformer_pva row always build -- no cell_*
    # keys at all. Passing an embedding_encoder against these must fail
    # loudly, not silently degrade to something wrong.
    ego = {"boxes": torch.zeros(1, 7), "scores": torch.tensor([0.8])}
    cav = {"boxes": torch.zeros(1, 7), "scores": torch.tensor([0.8])}
    try:
        estimate_correction(
            model, ego, cav,
            refine_config=RefineConfig(mode=ICP),
            abstain_config=AbstentionConfig(mode="none"),
            embedding_encoder=encoder,
        )
        assert False, "expected a KeyError for a detection missing cell_* fields"
    except KeyError:
        pass
