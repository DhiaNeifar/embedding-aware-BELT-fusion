"""AlignFormerB extended with CAV-CAV context cross-attention.

Additive, new file: does not subclass or modify ``model.py``'s
``AlignFormerB``. See ``trunk_context.py``'s module docstring for why this
extension exists and what it does.

Reuses, unchanged, every piece of Propose/Verify/Abstain's actual math:
``reduce_correspondence``, ``solve_pose`` (the weighted Kabsch solve, the
Wald statistic), and ``log_sinkhorn`` (via the inherited
``_soft_correspondence``) are all imported from ``model.py``/``head_match.py``
and called exactly as ``AlignFormerB`` calls them -- confirmed, before writing
this file, that none of them assume anything about how many token sets the
trunk that produced their input ever saw. The only override is *which* trunk
encodes the tokens, and that it is additionally handed a context token set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Tuple

import torch
from torch import Tensor

from embedding_aware_belt_fusion.alignformer_pva.model import (
    DEFAULT_HEADING_LAMBDA,
    AlignFormerB,
    PoseEstimate,
    _is_empty,
    _tokens,
    _zero_estimate,
    solve_pose,
)
from embedding_aware_belt_fusion.alignformer_pva.head_match import DEFAULT_SINKHORN_ITERATIONS
from embedding_aware_belt_fusion.alignformer_pva.robust import RobustSolveConfig
from embedding_aware_belt_fusion.alignformer_pva.trunk_context import AlignFormerTrunkWithContext
from embedding_aware_belt_fusion.alignformer_pva.variance import UNWEIGHTED, CorrespondenceVarianceModel

_PIPELINE_TAG = "alignformer_pva_context_v1"


class AlignFormerBWithContext(AlignFormerB):
    """``AlignFormerB`` whose trunk additionally cross-attends to other CAVs."""

    def __init__(
        self,
        embed_dim: int,
        heading_lambda: float = DEFAULT_HEADING_LAMBDA,
        sinkhorn_iterations: int = DEFAULT_SINKHORN_ITERATIONS,
        variance_model: CorrespondenceVarianceModel = UNWEIGHTED,
        **trunk_kwargs,
    ) -> None:
        # AlignFormerB.__init__ -> _Base.__init__ builds self.trunk as a plain
        # AlignFormerTrunk; everything else it sets up (match_projection,
        # dustbin, log_temperature, heading_lambda, sinkhorn_iterations,
        # variance_model) is exactly what this class needs too, so run it
        # unchanged and only replace the trunk afterward, rather than
        # duplicating that setup here.
        super().__init__(
            embed_dim,
            heading_lambda=heading_lambda,
            sinkhorn_iterations=sinkhorn_iterations,
            variance_model=variance_model,
            **trunk_kwargs,
        )
        self.trunk = AlignFormerTrunkWithContext(embed_dim=embed_dim, **trunk_kwargs)

    def _encode_with_context(self, batch: Mapping[str, Tensor]) -> Tuple[Tensor, Tensor]:
        return self.trunk(
            _tokens(batch, "ego"),
            _tokens(batch, "cav"),
            batch["ego_mask"],
            batch["cav_mask"],
            _tokens(batch, "context"),
            batch["context_mask"],
        )

    def forward(
        self,
        batch: Mapping[str, Tensor],
        robust: Optional[RobustSolveConfig] = None,
        statistic: bool = False,
    ) -> PoseEstimate:
        if _is_empty(batch):
            return _zero_estimate(batch, statistic)

        ego_mask, cav_mask = batch["ego_mask"], batch["cav_mask"]
        ego, cav = self._encode_with_context(batch)

        correspondence = self._soft_correspondence(batch, ego, cav, ego_mask, cav_mask)
        return solve_pose(
            correspondence, batch, self.heading_lambda, self.variance_model, robust,
            statistic=statistic,
        )


def load_alignformer_pva_context(
    checkpoint_path: Path, device: torch.device
) -> Tuple[AlignFormerBWithContext, dict]:
    """Rebuild an ``AlignFormerBWithContext`` from a checkpoint this pipeline saved.

    Mirrors ``train.py:load_alignformer_pva`` exactly, except for the class
    being constructed and the pipeline tag being checked -- kept as a
    separate tag (not ``train.py``'s ``"alignformer_pva_v1"``) specifically
    so a checkpoint from one architecture can never be silently handed to the
    other's loader.
    """
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if checkpoint.get("pipeline") != _PIPELINE_TAG:
        raise ValueError(f"{checkpoint_path} is not an {_PIPELINE_TAG} checkpoint")
    variance_kwargs = checkpoint.get("variance_model_kwargs")
    variance_model = (
        CorrespondenceVarianceModel(**variance_kwargs) if variance_kwargs is not None else UNWEIGHTED
    )
    model = AlignFormerBWithContext(variance_model=variance_model, **checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
