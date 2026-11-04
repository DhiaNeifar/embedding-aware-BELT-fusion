import torch

from embedding_aware_belt_fusion.embeddings.context_embedding import (
    ContextEmbedding,
    proposal_geometry,
)


def _synthetic_agent(proposal_count=3, max_cells=5, input_dim=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randn(proposal_count, max_cells, input_dim, generator=generator)
    positions = torch.randn(proposal_count, max_cells, 2, generator=generator)
    boxes = torch.zeros(proposal_count, 7)
    boxes[:, 0:2] = torch.randn(proposal_count, 2, generator=generator)
    boxes[:, 3:6] = torch.tensor([1.5, 2.0, 4.5])
    boxes[:, 6] = torch.rand(proposal_count, generator=generator) * 6.28 - 3.14
    scores = torch.rand(proposal_count, generator=generator)

    # Proposal 0: all cells valid, first 2 are footprint, rest are context.
    # Proposal 1: only 1 valid cell.
    # Proposal 2 (if present): fully padded except a single fallback cell,
    # which mirrors the materializer's own degenerate-box fallback (always at
    # least one real cell, flagged as footprint).
    mask = torch.ones(proposal_count, max_cells, dtype=torch.bool)
    footprint_flag = torch.zeros(proposal_count, max_cells, dtype=torch.bool)
    mask[0, :] = False
    footprint_flag[0, :2] = True
    mask[1, :1] = False
    if proposal_count > 2:
        mask[2, :1] = False
        footprint_flag[2, :1] = True
    return tokens, positions, footprint_flag, mask, boxes, scores


def test_proposal_geometry_shape_and_values():
    boxes = torch.zeros(2, 7)
    boxes[0] = torch.tensor([1.0, 2.0, 0.0, 1.5, 2.0, 4.0, 0.0])
    boxes[1] = torch.tensor([-1.0, 0.5, 0.0, 1.5, 2.0, 4.0, 3.14159265 / 2])
    scores = torch.tensor([0.9, 0.4])

    geometry = proposal_geometry(boxes, scores)

    assert geometry.shape == (2, 9)
    assert torch.allclose(geometry[0, :6], boxes[0, :6])
    assert torch.allclose(geometry[0, 6], torch.cos(boxes[0, 6]))
    assert torch.allclose(geometry[0, 7], torch.sin(boxes[0, 6]))
    assert torch.allclose(geometry[:, 8], scores)


def test_context_embedding_output_shape_and_normalized():
    tokens, positions, footprint_flag, mask, boxes, scores = _synthetic_agent()
    model = ContextEmbedding(
        input_dim=8, d_model=16, embed_dim=6, heads=2, layers=1, feedforward_dim=32
    )

    output = model(tokens, positions, footprint_flag, mask, boxes, scores)

    assert output.shape == (3, 6)
    assert torch.isfinite(output).all()
    norms = output.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_context_embedding_empty_proposal_set_returns_empty_tensor():
    model = ContextEmbedding(input_dim=8, d_model=16, embed_dim=6, heads=2, layers=1)
    empty = torch.zeros(0, 4, 8)
    output = model(
        empty,
        torch.zeros(0, 4, 2),
        torch.zeros(0, 4, dtype=torch.bool),
        torch.zeros(0, 4, dtype=torch.bool),
        torch.zeros(0, 7),
        torch.zeros(0),
    )
    assert output.shape == (0, 6)


def test_context_embedding_fully_masked_row_does_not_produce_nan_or_poison_gradient():
    # A row with ALL cells padded should never occur by construction (the
    # materializer always keeps at least one real cell per proposal), but the
    # encoder must not propagate NaN into every other row's gradient if it
    # ever did -- mirrors alignformer_pva.trunk's own degenerate-row guard.
    proposal_count, max_cells, input_dim = 3, 4, 8
    torch.manual_seed(0)
    tokens = torch.randn(proposal_count, max_cells, input_dim, requires_grad=False)
    positions = torch.randn(proposal_count, max_cells, 2)
    footprint_flag = torch.zeros(proposal_count, max_cells, dtype=torch.bool)
    mask = torch.ones(proposal_count, max_cells, dtype=torch.bool)
    mask[0] = False  # one fully valid row
    mask[1, 0] = False  # one normal row with a single valid cell
    # mask[2] stays all-True: fully masked row, the pathological case.
    boxes = torch.zeros(proposal_count, 7)
    boxes[:, 3:6] = torch.tensor([1.5, 2.0, 4.5])
    scores = torch.full((proposal_count,), 0.5)

    model = ContextEmbedding(input_dim=input_dim, d_model=16, embed_dim=6, heads=2, layers=1)
    output = model(tokens, positions, footprint_flag, mask, boxes, scores)

    # The guard's job is to keep the forward/backward pass finite everywhere,
    # not to force any particular value out of the degenerate row: after
    # output_norm's (zero-initialized) bias and output_projection's own bias,
    # a zero-weighted pooled vector does not generally map back to exactly
    # zero, and that is fine -- what must never happen is NaN/Inf leaking
    # into this row's output OR poisoning the other (legitimate) rows' own
    # gradients, since the encoder's weights are shared across the batch.
    assert torch.isfinite(output).all()

    output.sum().backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_context_embedding_gradients_flow_to_every_parameter():
    tokens, positions, footprint_flag, mask, boxes, scores = _synthetic_agent()
    model = ContextEmbedding(input_dim=8, d_model=16, embed_dim=6, heads=2, layers=1)

    output = model(tokens, positions, footprint_flag, mask, boxes, scores)
    output.sum().backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name} received no gradient"
