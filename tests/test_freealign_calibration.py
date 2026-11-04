"""Feeding a FreeAlign calibration back into the evaluator.

``calibrate_freealign.py`` selects six of ``FreeAlignConfig``'s tunables on
our own scenario-disjoint validation role; the evaluator's two flags
(``--freealign-edge-feature``, ``--freealign-min-nodes``) cover only the pair
a calibration might not move. On a dataset where the other four moved,
scoring FreeAlign at the bare defaults would be strawmanning it.
``--freealign-calibration`` hands the whole ``selected`` block over.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from embedding_aware_belt_fusion.freealign.freealign import (
    DEFAULT_MIN_NODES,
    EDGE_DISTANCE,
    EDGE_DISTANCE_YAW,
    RANSAC,
    FreeAlignConfig,
)
from embedding_aware_belt_fusion.freealign.freealign_calibration import (
    freealign_config_from_args,
    freealign_config_from_calibration,
)


def _write_calibration(path: Path, config: FreeAlignConfig) -> Path:
    path.write_text(json.dumps({"metric": "freealign_calibration", "selected": config.to_dict()}))
    return path


def _args(**overrides) -> argparse.Namespace:
    fields = {
        "freealign": True,
        "freealign_calibration": None,
        "freealign_edge_feature": EDGE_DISTANCE,
        "freealign_min_nodes": DEFAULT_MIN_NODES,
    }
    return argparse.Namespace(**{**fields, **overrides})


def test_a_calibration_round_trips_through_its_selected_block(tmp_path):
    selected = FreeAlignConfig(
        edge_feature=EDGE_DISTANCE_YAW, edge_threshold_m=1.5, anchor_limit=5,
        epsilon_power=2.0, epsilon_offset=10.0, robust_estimator=RANSAC,
    )
    path = _write_calibration(tmp_path / "calibration_result.json", selected)

    assert freealign_config_from_calibration(path) == selected


def test_the_provenance_keys_of_the_selected_block_are_not_constructor_fields(tmp_path):
    path = _write_calibration(tmp_path / "c.json", FreeAlignConfig())
    assert "reimplementation_of" in json.loads(path.read_text())["selected"]

    assert freealign_config_from_calibration(path) == FreeAlignConfig()


def test_a_file_without_a_selected_block_is_refused(tmp_path):
    path = tmp_path / "not_a_calibration.json"
    path.write_text(json.dumps({"metric": "noisy_ap"}))

    with pytest.raises(ValueError, match="selected"):
        freealign_config_from_calibration(path)


def test_no_freealign_flag_means_no_config():
    assert freealign_config_from_args(_args(freealign=False)) is None


def test_the_two_flags_alone_build_the_config_the_evaluator_always_built():
    config = freealign_config_from_args(_args(freealign_edge_feature=EDGE_DISTANCE_YAW, freealign_min_nodes=4))

    assert config == FreeAlignConfig(edge_feature=EDGE_DISTANCE_YAW, min_nodes=4)


def test_a_calibration_file_defines_every_parameter(tmp_path):
    selected = FreeAlignConfig(edge_threshold_m=1.0, robust_estimator=RANSAC)
    path = _write_calibration(tmp_path / "c.json", selected)

    assert freealign_config_from_args(_args(freealign_calibration=path)) == selected


def test_a_calibration_file_and_a_moved_flag_together_are_refused(tmp_path):
    path = _write_calibration(tmp_path / "c.json", FreeAlignConfig())

    with pytest.raises(ValueError, match="freealign-calibration"):
        freealign_config_from_args(_args(freealign_calibration=path, freealign_min_nodes=4))


def test_a_calibration_file_without_the_freealign_flag_is_refused(tmp_path):
    """Silently ignoring it would leave a result file claiming a calibrated row it never scored."""
    path = _write_calibration(tmp_path / "c.json", FreeAlignConfig())

    with pytest.raises(ValueError, match="--freealign"):
        freealign_config_from_args(_args(freealign=False, freealign_calibration=path))
