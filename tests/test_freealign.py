"""FreeAlign, REIMPLEMENTED: salient-object graph + MASS + robust SE(2).

These tests pin a *port*, not the authors' code (see
``freealign/freealign.py`` for why the published pipeline cannot be run
here). A port is only worth reporting if it is demonstrably not a strawman,
so what is pinned is first and foremost the method's **good regime**: with
three or more well-spread shared objects and realistic detection noise it
must recover the SE(2) transform cleanly.

Then the properties the comparison rests on:

- the edge feature is **invariant to the viewer's pose**, which is the whole
  reason a distance graph can align agents with no localization prior. The
  matched index pairs must therefore be *identical* whichever frame the CAV
  boxes arrive in -- that is what makes it legitimate to hand FreeAlign the
  same noisily-projected boxes AlignFormer-PVA gets and read its answer as a
  residual correction;
- MASS recovers the correspondence through distractors, and RANSAC survives
  outlier correspondences;
- the **abstain rule**: below ``min_nodes`` shared objects FreeAlign discards
  the collaborative message rather than emit a pose, which here means the
  identity correction and therefore :func:`is_fallback`;
- the least-squares core is the standard unweighted closed form, and agrees
  with this project's own weighted Kabsch (``alignformer_pva.procrustes``) at
  uniform weights -- stated as a test so the report can say exactly how much
  of our solver the port reuses (the test cross-checks the two; neither
  module imports the other).

NEGATIVE CONTROL: ``test_destroying_the_shared_geometry_destroys_the_recovery``
scrambles the CAV scene so no common subgraph exists. Permuting *ids* would be
a no-op here (FreeAlign never reads an id), so the control has to attack the
geometry, which is the only evidence this method has.
"""

import math

import pytest
import torch

from embedding_aware_belt_fusion.freealign.freealign import (
    DEFAULT_MIN_NODES,
    EDGE_DISTANCE,
    EDGE_DISTANCE_YAW,
    FreeAlignConfig,
    edge_features,
    freealign_estimate,
    is_fallback,
    mass_common_subgraph,
    robust_se2,
    se2_from_points,
)


def _boxes(centres, yaws=None) -> torch.Tensor:
    """``(N, 7)`` [x, y, z, h, w, l, yaw] boxes from 2-D centres and optional yaws."""
    centres = torch.as_tensor(centres, dtype=torch.float32).reshape(-1, 2)
    boxes = torch.zeros((centres.shape[0], 7), dtype=torch.float32)
    boxes[:, :2] = centres
    boxes[:, 3:6] = torch.tensor([1.56, 1.85, 4.2])
    if yaws is not None:
        boxes[:, 6] = torch.as_tensor(yaws, dtype=torch.float32)
    return boxes


def _move(boxes: torch.Tensor, psi: float, tx: float, ty: float) -> torch.Tensor:
    """Apply an SE(2) to boxes, the way a pose correction would."""
    cos, sin = math.cos(psi), math.sin(psi)
    rotation = torch.tensor([[cos, -sin], [sin, cos]], dtype=torch.float32)
    moved = boxes.clone()
    moved[:, :2] = boxes[:, :2] @ rotation.T + torch.tensor([tx, ty])
    moved[:, 6] = boxes[:, 6] + psi
    return moved


def _pair(ego: torch.Tensor, cav: torch.Tensor) -> dict:
    return {
        "ego_boxes": ego.unsqueeze(0),
        "cav_boxes": cav.unsqueeze(0),
        "ego_mask": torch.ones((1, ego.shape[0]), dtype=torch.bool),
        "cav_mask": torch.ones((1, cav.shape[0]), dtype=torch.bool),
    }


# A spread-out urban scene: eight vehicles, no two at the same separation, so
# the distance graph is not accidentally degenerate.
_SCENE = [
    (5.0, 2.0), (18.0, -7.0), (31.0, 11.0), (-9.0, 14.0),
    (24.0, 26.0), (-15.0, -21.0), (40.0, -3.0), (12.0, 33.0),
]
_SCENE_YAWS = [0.1, 1.4, -0.7, 2.9, 0.4, -2.2, 1.9, -0.3]


def test_the_edge_feature_is_invariant_to_the_viewers_pose():
    # The claim the whole method rests on: relative distance does not know
    # where the observer is. If this fails, nothing downstream means anything.
    boxes = _boxes(_SCENE, _SCENE_YAWS)
    moved = _move(boxes, psi=0.9, tx=-31.0, ty=17.5)

    config = FreeAlignConfig()
    assert torch.allclose(
        edge_features(boxes, config), edge_features(moved, config), atol=1e-4
    )


def test_the_shipped_codes_distance_and_yaw_edge_feature_is_also_invariant():
    # github.com/MediaBrain-SJTU/FreeAlign's greedy_match.py builds a 2-channel
    # edge (relative distance, relative yaw), not the paper's distance alone.
    # Both channels rotate with the observer, so both are invariant -- which is
    # why the port offers it as a second, strictly-more-informed variant.
    boxes = _boxes(_SCENE, _SCENE_YAWS)
    moved = _move(boxes, psi=-1.7, tx=12.0, ty=4.0)

    config = FreeAlignConfig(edge_feature=EDGE_DISTANCE_YAW)
    assert torch.allclose(
        edge_features(boxes, config), edge_features(moved, config), atol=1e-4
    )


def test_mass_recovers_the_correspondence_through_distractors():
    # Six shared objects, two ego-only and two CAV-only, and the CAV's rows in
    # a different order: the search has to find the common subgraph, not the
    # row alignment.
    ego = _boxes(_SCENE, _SCENE_YAWS)
    order = [4, 0, 6, 2, 1, 3]
    shared = [_SCENE[i] for i in order]
    cav = _move(_boxes(shared + [(-40.0, 40.0), (50.0, 44.0)]), 0.6, 9.0, -13.0)

    match = mass_common_subgraph(
        edge_features(ego, FreeAlignConfig()),
        edge_features(cav, FreeAlignConfig()),
        FreeAlignConfig(),
    )

    recovered = dict(zip(match.ego_indices.tolist(), match.cav_indices.tolist()))
    assert len(recovered) >= DEFAULT_MIN_NODES
    for cav_index, ego_index in enumerate(order):
        if ego_index in recovered:
            assert recovered[ego_index] == cav_index


def test_the_good_regime_recovers_the_pose_cleanly():
    # THE ANTI-STRAWMAN TEST. Five well-spread correspondences and realistic
    # per-detection centre noise: a port far off the injected transform here
    # would be broken, and reporting its losses would be dishonest.
    torch.manual_seed(7)
    ego = _boxes(_SCENE[:5], _SCENE_YAWS[:5])
    jitter = 0.08 * torch.randn(5, 2)
    cav = _move(_boxes([_SCENE[i] for i in range(5)]), 0.35, -22.0, 8.0)
    cav[:, :2] += jitter

    estimate = freealign_estimate(_pair(ego, cav), FreeAlignConfig())

    # Undo the CAV's frame: the estimate must be the inverse of what was applied.
    psi = float(estimate.psi[0])
    assert psi == pytest.approx(-0.35, abs=0.02)
    recovered = _move(cav, psi, float(estimate.t[0, 0]), float(estimate.t[0, 1]))
    assert float((recovered[:, :2] - ego[:, :2]).norm(dim=1).mean()) < 0.15
    assert not bool(is_fallback(estimate)[0])


def test_matching_is_unchanged_by_the_frame_the_cav_boxes_arrive_in():
    # This is what licenses handing FreeAlign the same noisily-projected CAV
    # boxes AlignFormer-PVA gets: the matching cannot see the projection, so
    # only the parameterization of the answer changes (absolute pose ->
    # residual).
    ego = _boxes(_SCENE, _SCENE_YAWS)
    cav = _move(_boxes(_SCENE[:6], _SCENE_YAWS[:6]), 0.2, 3.0, -4.0)
    extra = _move(cav, 1.1, -25.0, 40.0)

    config = FreeAlignConfig()
    direct = freealign_estimate(_pair(ego, cav), config)
    shifted = freealign_estimate(_pair(ego, extra), config)

    # Same correspondences (as a set: the greedy order of discovery is a float
    # tie-break and carries no meaning, the pairing is what the method emits).
    def _pairs(cav_boxes):
        match = mass_common_subgraph(
            edge_features(ego, config), edge_features(cav_boxes, config), config
        )
        return sorted(zip(match.ego_indices.tolist(), match.cav_indices.tolist()))

    assert _pairs(cav) == _pairs(extra)
    # ...and the two corrections land the CAV boxes in the same place.
    a = _move(cav, float(direct.psi[0]), float(direct.t[0, 0]), float(direct.t[0, 1]))
    b = _move(extra, float(shifted.psi[0]), float(shifted.t[0, 0]), float(shifted.t[0, 1]))
    assert torch.allclose(a[:, :2], b[:, :2], atol=1e-3)


@pytest.mark.parametrize("shared", [0, 1, 2])
def test_it_abstains_below_the_minimum_node_count(shared):
    # FreeAlign's own policy: below a predetermined minimum node count the
    # collaborative message is DISCARDED rather than aligned with a pose no
    # distance graph can determine. Here that is the identity correction,
    # comparable to alignformer_pva's own MIN_MATCH_MASS fallback.
    ego = _boxes(_SCENE, _SCENE_YAWS)
    common = [_SCENE[i] for i in range(shared)]
    cav = _move(_boxes(common + [(-60.0, 55.0), (-70.0, 61.0), (-80.0, 52.0)]),
                0.5, 20.0, -10.0)

    estimate = freealign_estimate(_pair(ego, cav), FreeAlignConfig())

    assert bool(is_fallback(estimate)[0])
    assert float(estimate.confidence[0]) < DEFAULT_MIN_NODES


def test_destroying_the_shared_geometry_destroys_the_recovery():
    # NEGATIVE CONTROL. Note what it must attack: FreeAlign never reads an
    # object id, so permuting ids would be a literal no-op here. The only
    # evidence it has is the geometry, so the control scrambles the geometry.
    # The honest arm is asserted in the same test, so this cannot pass by the
    # method being inert.
    ego = _boxes(_SCENE, _SCENE_YAWS)
    honest = _move(_boxes(_SCENE[:6], _SCENE_YAWS[:6]), 0.4, 14.0, -6.0)

    good = freealign_estimate(_pair(ego, honest), FreeAlignConfig())
    landed = _move(honest, float(good.psi[0]), float(good.t[0, 0]), float(good.t[0, 1]))
    assert float((landed[:, :2] - ego[:6, :2]).norm(dim=1).max()) < 0.05

    torch.manual_seed(11)
    scrambled = honest.clone()
    scrambled[:, :2] += 12.0 * torch.randn(6, 2)
    broken = freealign_estimate(_pair(ego, scrambled), FreeAlignConfig())

    if not bool(is_fallback(broken)[0]):
        moved = _move(
            scrambled, float(broken.psi[0]),
            float(broken.t[0, 0]), float(broken.t[0, 1]),
        )
        assert float((moved[:, :2] - ego[:6, :2]).norm(dim=1).mean()) > 1.0


def test_ransac_survives_outlier_correspondences():
    # The reason the paper specifies a robust estimator rather than plain SVD:
    # MASS can admit a coincidental pair, and one 20 m outlier in a
    # least-squares fit moves the whole pose.
    ego = torch.tensor(
        [[0.0, 0.0], [20.0, 1.0], [8.0, 17.0], [-11.0, 9.0], [30.0, -12.0], [4.0, -20.0]]
    )
    cav = ego.clone()
    cav[4] += torch.tensor([19.0, -23.0])  # a wrong correspondence
    cav[5] += torch.tensor([-17.0, 25.0])  # and another

    psi, t = se2_from_points(ego, cav)
    robust_psi, robust_t, inliers = robust_se2(ego, cav, FreeAlignConfig())

    assert float(t.norm()) > 1.0  # plain least squares is dragged off
    assert float(robust_t.norm()) < 0.05 and abs(float(robust_psi)) < 0.01
    assert int(inliers.sum()) == 4


def test_the_least_squares_core_agrees_with_our_weighted_kabsch_at_uniform_weights():
    # Stated as a test so the report can say precisely how much of
    # alignformer_pva's solver the port reuses: NONE of it at runtime (the
    # port has its own unweighted core, so MIN_MATCH_MASS, heading virtual
    # points and the inverse-variance weighting cannot leak into the
    # competitor), but the core is the same standard closed form.
    from embedding_aware_belt_fusion.alignformer_pva.procrustes import weighted_se2_kabsch

    torch.manual_seed(3)
    source = 20.0 * torch.randn(9, 2)
    target = _move(_boxes(source), 0.61, 7.0, -2.5)[:, :2]

    psi, t = se2_from_points(target, source)
    reference_psi, reference_t = weighted_se2_kabsch(
        target.unsqueeze(0), source.unsqueeze(0), torch.ones(1, 9)
    )

    assert float(psi) == pytest.approx(float(reference_psi[0]), abs=1e-5)
    assert torch.allclose(t, reference_t[0], atol=1e-4)


def test_an_empty_object_set_yields_the_identity_correction():
    ego = _boxes(_SCENE)
    empty = torch.zeros((0, 7))

    estimate = freealign_estimate(_pair(ego, empty), FreeAlignConfig())

    assert bool(is_fallback(estimate)[0])
    assert float(estimate.confidence[0]) == 0.0


def test_lmeds_and_ransac_agree_when_every_correspondence_is_clean():
    # The paper offers either; on clean data they must not disagree, so a
    # difference in the sweep would be a difference in outlier handling alone.
    ego = _boxes(_SCENE[:6], _SCENE_YAWS[:6])
    cav = _move(_boxes(_SCENE[:6], _SCENE_YAWS[:6]), -0.8, 5.0, 31.0)

    ransac = freealign_estimate(_pair(ego, cav), FreeAlignConfig(robust_estimator="ransac"))
    lmeds = freealign_estimate(_pair(ego, cav), FreeAlignConfig(robust_estimator="lmeds"))

    assert float(ransac.psi[0]) == pytest.approx(float(lmeds.psi[0]), abs=1e-4)
    assert torch.allclose(ransac.t, lmeds.t, atol=1e-3)


def test_the_configuration_is_immutable_and_names_its_edge_features():
    config = FreeAlignConfig()
    assert config.edge_feature == EDGE_DISTANCE
    with pytest.raises(Exception):
        config.edge_feature = EDGE_DISTANCE_YAW
    with pytest.raises(ValueError):
        FreeAlignConfig(edge_feature="learned_edgegat")


def test_a_batch_of_pairs_is_estimated_independently():
    # The sweep runs one pair at a time, but a silent broadcast across the
    # batch dimension would be the kind of bug that makes a competitor look
    # bad for free.
    ego = _boxes(_SCENE[:6], _SCENE_YAWS[:6])
    first = _move(_boxes(_SCENE[:6]), 0.3, 10.0, -5.0)
    second = _move(_boxes(_SCENE[:6]), -0.5, -4.0, 21.0)

    stacked = {
        "ego_boxes": torch.stack([ego, ego]),
        "cav_boxes": torch.stack([first, second]),
        "ego_mask": torch.ones((2, 6), dtype=torch.bool),
        "cav_mask": torch.ones((2, 6), dtype=torch.bool),
    }
    batched = freealign_estimate(stacked, FreeAlignConfig())
    alone = [
        freealign_estimate(_pair(ego, cav), FreeAlignConfig()) for cav in (first, second)
    ]

    for index, single in enumerate(alone):
        assert float(batched.psi[index]) == pytest.approx(float(single.psi[0]), abs=1e-5)
        assert torch.allclose(batched.t[index], single.t[0], atol=1e-4)


def test_padded_objects_are_never_matched():
    ego = _boxes(_SCENE[:6], _SCENE_YAWS[:6])
    cav = _move(_boxes(_SCENE[:6], _SCENE_YAWS[:6]), 0.25, 6.0, -9.0)
    padded = torch.cat([cav, torch.zeros((3, 7))], dim=0)

    batch = _pair(ego, padded)
    batch["cav_mask"] = torch.tensor([[True] * 6 + [False] * 3])
    estimate = freealign_estimate(batch, FreeAlignConfig())

    assert float(estimate.confidence[0]) == 6.0
    assert float(estimate.psi[0]) == pytest.approx(-0.25, abs=1e-3)


def test_freealign_module_imports_nothing_from_alignformer_pva():
    # The separation the whole package exists to preserve: freealign.py's own
    # pose-fitting/algorithm code must not import alignformer_pva for
    # anything (its docstrings DISCUSS alignformer_pva, by name, to explain
    # the separation -- this checks the actual import statements, not the
    # prose). The test file above imports it deliberately, for the one
    # sanctioned cross-check; this assertion is about the shipped module.
    import ast
    import inspect

    import embedding_aware_belt_fusion.freealign.freealign as module

    tree = ast.parse(inspect.getsource(module))
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    assert not any("alignformer_pva" in name for name in imported_modules)
