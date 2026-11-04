"""Idea 1: the soft match's own spread as an uncertainty term in Propose.

Propose matches each ego object to a Sinkhorn mixture of CAV candidates and
gives that correspondence the mixture-average of the candidates' detector
variances. By the law of total variance the mixture's variance also has a
between-candidate term -- how far the candidates sit from the virtual point
they are averaged into. That term is exactly the "blur": a row whose mass
sits on one box has none, a row split over boxes metres apart has metres
squared. Adding it

* down-weights blurred rows in the weighted Kabsch fit (Propose's averaging), and
* enlarges the disagreement covariance the Wald statistic is taken against,
  so a fit resting on blurred rows has to clear a higher bar (Abstain).

Centre: ``sum_j w_ij |c_j - v_i|^2 / m_i`` (a 2-D total, the same units as
``cav_variance_centre``). Heading: ``-2 ln R_i`` with ``R_i`` the length of
the row's mean folded unit heading (the wrapped-normal variance estimate;
zero when the candidates agree). No tunable.

The pose is solved by the UNMODIFIED ``model.reduce_correspondence`` and
``model.solve_pose``; only the two variance fields of the correspondence
change. With the spread switched off this reproduces the cached Propose
output (checked by ``scripts/check_uncertainty_noop.py``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

import numpy as np
import torch

from embedding_aware_belt_fusion.alignformer_pva.model import reduce_correspondence, solve_pose
from embedding_aware_belt_fusion.alignformer_pva.procrustes import heading_orientation

BOX_YAW = 6
_MIN_MASS = 1e-6  # model._MIN_MASS
_MIN_RESULTANT = 1e-6


def spread_terms(weights: torch.Tensor, batch, correspondence) -> Tuple[torch.Tensor, torch.Tensor]:
    """(centre spread (B, M), heading spread (B, M)) of each ego row's soft match."""
    mass = correspondence.mass
    safe_mass = mass.clamp_min(_MIN_MASS)
    cav_centres = batch["cav_boxes"][..., :2]
    offset = cav_centres.unsqueeze(1) - correspondence.virtual_centres.unsqueeze(2)  # (B, M, N, 2)
    centre = (weights * offset.pow(2).sum(-1)).sum(-1) / safe_mass
    cav_yaws = batch["cav_boxes"][..., BOX_YAW]
    orientation = heading_orientation(batch["ego_boxes"][..., BOX_YAW], cav_yaws)
    direction = torch.stack([torch.cos(cav_yaws), torch.sin(cav_yaws)], dim=-1)
    resultant = ((weights * orientation) @ direction / safe_mass.unsqueeze(-1)).norm(dim=-1)
    heading = -2.0 * torch.log(resultant.clamp(_MIN_RESULTANT, 1.0))
    has_weight = mass > _MIN_MASS
    return torch.where(has_weight, centre, torch.zeros_like(centre)), torch.where(
        has_weight, heading, torch.zeros_like(heading)
    )


def propose(
    plan: np.ndarray, batch, model, robust_config, *, spread: bool = True, with_precision: bool = False
):
    """``(psi, tx, ty, statistic, dof)`` from a cached Sinkhorn plan ``(M+1, N+1)`` (log, dustbin included);
    ``with_precision`` also returns the fit's (3, 3) precision over ``(tx, ty, psi)`` (``offset_precision``)."""
    log_assignment = torch.from_numpy(plan).unsqueeze(0)
    weights = log_assignment[:, :-1, :-1].exp()
    correspondence = reduce_correspondence(weights, batch, model.variance_model, log_assignment=log_assignment)
    if spread:
        centre, heading = spread_terms(weights, batch, correspondence)
        correspondence = replace(
            correspondence,
            cav_variance_centre=correspondence.cav_variance_centre + centre,
            cav_variance_heading=correspondence.cav_variance_heading + heading,
        )
    estimate = solve_pose(
        correspondence, batch, model.heading_lambda, model.variance_model, robust_config, statistic=True
    )
    out = (
        float(estimate.psi[0]), float(estimate.t[0, 0]), float(estimate.t[0, 1]),
        float(estimate.offset_statistic[0]), float(estimate.offset_dof[0]),
    )
    if with_precision:
        return out, estimate.offset_precision[0].double().numpy()
    return out
