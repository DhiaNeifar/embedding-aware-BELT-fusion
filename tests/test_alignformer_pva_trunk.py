import math

import pytest
import torch

from embedding_aware_belt_fusion.alignformer_pva.trunk import (
    BOX_YAW,
    GEOMETRY_FEATURES,
    AlignFormerTrunk,
    _attend_without_degenerate_rows,
    tokenize,
)


def _make_trunk(embed_dim=0, model_dim=16, layers=2, heads=2):
    torch.manual_seed(0)
    return AlignFormerTrunk(embed_dim=embed_dim, model_dim=model_dim, layers=layers, heads=heads)


# ---- tokenize ----------------------------------------------------------------


def test_tokenize_shape_and_yaw_cos_sin_encoding():
    # boxes: [x, y, z, h, w, l, yaw] -- BOX_YAW == 6 commits to this layout,
    # matching our detector's confirmed (boxes, scores, embeddings) output.
    assert BOX_YAW == 6
    yaws = torch.tensor([0.0, math.pi / 2, math.pi, -math.pi / 2])
    boxes = torch.zeros(1, 4, 7)
    boxes[0, :, :6] = torch.arange(24).reshape(4, 6).float()
    boxes[0, :, BOX_YAW] = yaws
    scores = torch.tensor([[0.9, 0.5, 0.1, 0.7]])
    embeddings = torch.zeros(1, 4, 0)  # our usage: embeddings unused (zero-width)

    tokens = tokenize(boxes, scores, embeddings)

    assert tokens.shape == (1, 4, GEOMETRY_FEATURES)
    # Columns 0:6 pass the first six geometry fields through unchanged.
    assert torch.allclose(tokens[0, :, :6], boxes[0, :, :6])
    # Columns 6/7 are cos(yaw)/sin(yaw) -- continuous across the +/-pi wrap,
    # unlike the raw angle (yaw[2] == pi and yaw[3] == -pi/2 would otherwise
    # look like a large numeric jump from yaw[1] == pi/2).
    assert torch.allclose(tokens[0, :, 6], torch.cos(yaws), atol=1e-6)
    assert torch.allclose(tokens[0, :, 7], torch.sin(yaws), atol=1e-6)
    # Column 8 is the passed-through detector score.
    assert torch.allclose(tokens[0, :, 8], scores[0])


def test_tokenize_appends_embeddings_of_any_width():
    # The signature stays general-purpose even though our own usage passes a
    # zero-width embeddings tensor -- verify a non-zero width still works and
    # is appended untouched after the geometry block.
    boxes = torch.zeros(2, 3, 7)
    scores = torch.rand(2, 3)
    embeddings = torch.randn(2, 3, 5)

    tokens = tokenize(boxes, scores, embeddings)

    assert tokens.shape == (2, 3, GEOMETRY_FEATURES + 5)
    assert torch.allclose(tokens[..., GEOMETRY_FEATURES:], embeddings)


def test_tokenize_zero_width_embeddings_do_not_change_geometry_block():
    boxes = torch.randn(1, 2, 7)
    scores = torch.rand(1, 2)
    embeddings = torch.zeros(1, 2, 0)

    tokens = tokenize(boxes, scores, embeddings)

    assert tokens.shape == (1, 2, GEOMETRY_FEATURES)


# ---- AlignFormerTrunk.forward: shapes and masking -----------------------------


def test_forward_shapes_and_padded_positions_are_exactly_zero():
    trunk = _make_trunk(embed_dim=0, model_dim=16, layers=2, heads=2)

    batch, ego_n, cav_n = 2, 3, 4
    ego_tokens = torch.randn(batch, ego_n, GEOMETRY_FEATURES)
    cav_tokens = torch.randn(batch, cav_n, GEOMETRY_FEATURES)

    # Partial padding, not a wholly-empty set, in either agent's object set.
    ego_mask = torch.tensor([[True, True, False], [True, False, False]])
    cav_mask = torch.tensor([[True, True, True, False], [True, True, False, False]])

    ego_out, cav_out = trunk(ego_tokens, cav_tokens, ego_mask, cav_mask)

    assert ego_out.shape == (batch, ego_n, 16)
    assert cav_out.shape == (batch, cav_n, 16)

    # The trunk's own masking logic -- output_norm(x) * mask -- guarantees
    # padded positions come back as EXACTLY zero, not just numerically small.
    assert torch.all(ego_out[~ego_mask] == 0)
    assert torch.all(cav_out[~cav_mask] == 0)
    # Sanity check against a vacuously-true all-zero assertion: valid
    # (non-padded) positions should not themselves be all zero.
    assert not torch.all(ego_out[ego_mask] == 0)
    assert not torch.all(cav_out[cav_mask] == 0)


def test_forward_runs_end_to_end_with_nonzero_embeddings():
    # Same scenario, but with the signature's general embed_dim != 0 path
    # exercised too (our usage sets embed_dim=0, but the trunk itself must
    # stay general).
    trunk = _make_trunk(embed_dim=3, model_dim=16, layers=2, heads=2)

    batch, ego_n, cav_n = 2, 3, 4
    ego_tokens = torch.randn(batch, ego_n, GEOMETRY_FEATURES + 3)
    cav_tokens = torch.randn(batch, cav_n, GEOMETRY_FEATURES + 3)
    ego_mask = torch.ones(batch, ego_n, dtype=torch.bool)
    cav_mask = torch.ones(batch, cav_n, dtype=torch.bool)

    ego_out, cav_out = trunk(ego_tokens, cav_tokens, ego_mask, cav_mask)

    assert ego_out.shape == (batch, ego_n, 16)
    assert cav_out.shape == (batch, cav_n, 16)
    assert torch.isfinite(ego_out).all()
    assert torch.isfinite(cav_out).all()


# ---- the degenerate wholly-empty-set guard ------------------------------------


def test_attend_without_degenerate_rows_raises_on_zero_width_keys():
    # Direct unit test of the documented precondition on
    # _attend_without_degenerate_rows itself: valid_keys must have a real,
    # non-empty key dimension. There is no key position 0 to unmask
    # otherwise, so the function must refuse rather than silently misbehave.
    attention = torch.nn.MultiheadAttention(8, 2, batch_first=True)
    query = torch.randn(2, 3, 8)
    key = torch.randn(2, 0, 8)
    value = torch.randn(2, 0, 8)
    valid_keys = torch.zeros(2, 0, dtype=torch.bool)

    with pytest.raises(ValueError, match="zero keys"):
        _attend_without_degenerate_rows(attention, query, key, value, valid_keys)


def test_forward_raises_when_one_agent_set_is_wholly_empty():
    # The documented precondition: AlignFormerTrunk itself does not guard
    # against a wholly empty ego/cav set (N == 0) -- the caller must exclude
    # that case first. Passing one anyway must surface as the ValueError
    # from _attend_without_degenerate_rows, not silently produce NaN.
    trunk = _make_trunk(embed_dim=0, model_dim=16, layers=1, heads=2)

    ego_tokens = torch.randn(2, 3, GEOMETRY_FEATURES)
    cav_tokens = torch.randn(2, 0, GEOMETRY_FEATURES)
    ego_mask = torch.ones(2, 3, dtype=torch.bool)
    cav_mask = torch.zeros(2, 0, dtype=torch.bool)

    with pytest.raises(ValueError):
        trunk(ego_tokens, cav_tokens, ego_mask, cav_mask)


def test_backward_is_finite_when_one_batch_item_has_no_valid_objects():
    # This is the real failure mode _attend_without_degenerate_rows exists to
    # prevent (its own docstring, R27): a batch item whose object set is
    # entirely padded (an agent that detected nothing this frame) makes
    # nn.MultiheadAttention compute softmax over an all -inf row -- 0/0 --
    # which is NaN not only in that item's output but, through the shared
    # projection/norm/feed-forward parameters, in every other batch item's
    # gradient too. Confirm the guard actually prevents that, rather than
    # just asserting it does.
    trunk = _make_trunk(embed_dim=0, model_dim=16, layers=2, heads=2)

    batch, ego_n, cav_n = 2, 3, 4
    ego_tokens = torch.randn(batch, ego_n, GEOMETRY_FEATURES, requires_grad=True)
    cav_tokens = torch.randn(batch, cav_n, GEOMETRY_FEATURES, requires_grad=True)

    # Batch item 0's ego set is entirely padded; item 1 is a normal partial
    # padding case, so this also checks the degenerate item does not poison
    # the non-degenerate one's otherwise-unrelated gradient path.
    ego_mask = torch.tensor([[False, False, False], [True, True, False]])
    cav_mask = torch.tensor([[True, True, False, False], [True, True, True, False]])

    ego_out, cav_out = trunk(ego_tokens, cav_tokens, ego_mask, cav_mask)
    loss = ego_out.sum() + cav_out.sum()
    loss.backward()

    assert torch.isfinite(ego_tokens.grad).all()
    assert torch.isfinite(cav_tokens.grad).all()
    for name, param in trunk.named_parameters():
        assert param.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has a non-finite gradient"
