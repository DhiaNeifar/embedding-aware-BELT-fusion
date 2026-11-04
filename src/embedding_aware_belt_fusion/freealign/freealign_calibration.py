"""Feed a FreeAlign calibration back into the evaluator.

``calibrate_freealign.py`` selects :class:`FreeAlignConfig`'s tunables on our
own scenario-disjoint OPV2V validation role. ``--freealign-calibration`` hands
that whole ``selected`` block to ``evaluation/belt_fusion.py`` so the
FreeAlign row of a sweep is the calibrated one by construction, never the
bare (paper/authors'-code) defaults.

Ported, with the box/field names unchanged, from the labmate's own
``alignformer/freealign_calibration.py`` (inspected read-only via
``git show``) -- its two flags (edge feature, min_nodes) and the
whole-block-override contract are a faithful reuse of his design, adapted
only to import from this package's own :mod:`freealign` module instead of
his.

``freealign.py`` itself is frozen for the fair comparison; this module only
reads what its ``to_dict`` wrote.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Union

from embedding_aware_belt_fusion.freealign.freealign import (
    DEFAULT_MIN_NODES,
    EDGE_DISTANCE,
    FreeAlignConfig,
)

SELECTED_KEY = "selected"
# ``to_dict`` name -> constructor name, for the one key that differs.
_FIELD_NAMES = {"anchor_limit_gamma": "anchor_limit"}
# Provenance stamped into every record; not parameters.
_PROVENANCE_KEYS = frozenset({"reimplementation_of", "authors_code_used", "gnn_edge_features"})


def freealign_config_from_calibration(path: Union[Path, str]) -> FreeAlignConfig:
    """The configuration a calibration result selected."""
    payload = json.loads(Path(path).read_text())
    if SELECTED_KEY not in payload:
        raise ValueError(f"{path} has no '{SELECTED_KEY}' block; is it a calibrate_freealign.py result?")
    fields = {
        _FIELD_NAMES.get(key, key): value
        for key, value in payload[SELECTED_KEY].items()
        if key not in _PROVENANCE_KEYS
    }
    return FreeAlignConfig(**fields)


def freealign_config_from_args(args: argparse.Namespace) -> Optional[FreeAlignConfig]:
    """``None`` without ``--freealign``; the calibration's config, or the two flags'."""
    calibration = getattr(args, "freealign_calibration", None)
    if not args.freealign:
        if calibration is not None:
            raise ValueError("--freealign-calibration needs --freealign; without it no FreeAlign row is scored")
        return None
    if calibration is None:
        return FreeAlignConfig(edge_feature=args.freealign_edge_feature, min_nodes=args.freealign_min_nodes)
    moved = args.freealign_edge_feature != EDGE_DISTANCE or args.freealign_min_nodes != DEFAULT_MIN_NODES
    if moved:
        raise ValueError(
            "--freealign-calibration sets every FreeAlign parameter; drop "
            "--freealign-edge-feature and --freealign-min-nodes"
        )
    return freealign_config_from_calibration(calibration)


__all__ = ["SELECTED_KEY", "freealign_config_from_args", "freealign_config_from_calibration"]
