"""Detector feature extraction adapters."""

from .opencood_proposals import (
    assign_proposals_to_ground_truth,
    decode_local_proposals,
    pointpillar_forward_with_features,
    proposal_roi_cell_indices,
)

__all__ = [
    "assign_proposals_to_ground_truth",
    "decode_local_proposals",
    "pointpillar_forward_with_features",
    "proposal_roi_cell_indices",
]
