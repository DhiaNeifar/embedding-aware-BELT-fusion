"""Training objectives for the Propose-Verify-Abstain pose head.

The pose objective is a **corner loss** rather than a weighted MSE on
``(dx, dy, dpsi)``. That formulation mixes metres and radians under
arbitrary weights, with yaw prone to dominating the loss while still not
being learned well. Measuring displacement of the box corners is in metres
throughout, weights a yaw error by how far away the object is, and optimizes
something close to what IoU and AP actually measure.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Index of the yaw field in our own [x, y, z, h, w, l, yaw] box layout,
# confirmed to match the source repo's convention (its boxes.py, BOX_YAW = 6).
BOX_YAW = 6
_BOX_WIDTH = 4
_BOX_LENGTH = 5


def bev_corners(boxes: Tensor) -> Tensor:
    """Return the 4 BEV corners of each box as ``(B, N, 4, 2)``."""
    if boxes.shape[-1] != 7:
        raise ValueError(f"boxes must end in 7 features, got {tuple(boxes.shape)}")

    half_length = boxes[..., _BOX_LENGTH] / 2
    half_width = boxes[..., _BOX_WIDTH] / 2
    signs = boxes.new_tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])

    along = half_length.unsqueeze(-1) * signs[:, 0]
    across = half_width.unsqueeze(-1) * signs[:, 1]

    cosine = torch.cos(boxes[..., BOX_YAW]).unsqueeze(-1)
    sine = torch.sin(boxes[..., BOX_YAW]).unsqueeze(-1)
    x = boxes[..., 0:1] + along * cosine - across * sine
    y = boxes[..., 1:2] + along * sine + across * cosine
    return torch.stack([x, y], dim=-1)


def apply_se2(points: Tensor, psi: Tensor, t: Tensor) -> Tensor:
    """Apply a per-batch SE(2) transform to ``(B, ..., 2)`` points."""
    extra_dims = points.dim() - 2
    shape = (psi.shape[0],) + (1,) * extra_dims
    cosine = torch.cos(psi).reshape(shape)
    sine = torch.sin(psi).reshape(shape)

    x, y = points[..., 0], points[..., 1]
    rotated = torch.stack([cosine * x - sine * y, sine * x + cosine * y], dim=-1)
    return rotated + t.reshape(shape + (2,))


def corner_displacements(
    boxes: Tensor,
    psi_pred: Tensor,
    t_pred: Tensor,
    psi_true: Tensor,
    t_true: Tensor,
) -> Tensor:
    """Per-object L1 corner displacement, ``(B, N)``, summed over the 4 corners.

    The unreduced quantity behind :func:`corner_loss`. Kept separate so that a
    per-sample or per-subset breakdown is the *same* measurement as the
    training loss rather than a second definition of it.
    """
    corners = bev_corners(boxes)
    predicted = apply_se2(corners, psi_pred, t_pred)
    target = apply_se2(corners, psi_true, t_true)
    return (predicted - target).abs().sum(dim=-1).sum(dim=-1)


def corner_loss(
    boxes: Tensor,
    psi_pred: Tensor,
    t_pred: Tensor,
    psi_true: Tensor,
    t_true: Tensor,
    mask: Tensor,
) -> Tensor:
    """Mean L1 displacement of BEV corners under the predicted vs true correction.

    Parameters
    ----------
    boxes: ``(B, N, 7)`` CAV boxes in the ego frame, before correction.
    psi_pred, psi_true: ``(B,)`` yaw corrections in radians.
    t_pred, t_true: ``(B, 2)`` translation corrections in metres.
    mask: ``(B, N)`` bool, True for real objects.

    Returns
    -------
    Tensor
        Scalar loss in metres, averaged over real objects only.
    """
    per_object = corner_displacements(boxes, psi_pred, t_pred, psi_true, t_true)
    weights = mask.to(per_object.dtype)
    return (per_object * weights).sum() / weights.sum().clamp_min(1.0)


def match_nll(log_assignment: Tensor, ego_match: Tensor, cav_match: Tensor) -> Tensor:
    """Negative log-likelihood of the ground-truth assignment, dustbins included.

    Parameters
    ----------
    log_assignment: ``(B, M + 1, N + 1)`` from :func:`log_sinkhorn`.
    ego_match: ``(B, M)`` matched CAV index per ego object, ``-1`` if unmatched.
    cav_match: ``(B, N)`` matched ego index per CAV object, ``-1`` if unmatched.

    Unmatched objects are supervised onto their dustbin, which is what teaches
    the model that a detection seen by only one agent must not be matched.
    """
    _, rows, columns = log_assignment.shape
    ego_count, cav_count = rows - 1, columns - 1

    ego_target = torch.where(
        ego_match < 0, torch.full_like(ego_match, cav_count), ego_match
    )
    cav_target = torch.where(
        cav_match < 0, torch.full_like(cav_match, ego_count), cav_match
    )

    ego_terms = (
        log_assignment[:, :ego_count, :].gather(2, ego_target.unsqueeze(-1)).squeeze(-1)
    )
    cav_terms = (
        log_assignment[:, :, :cav_count].gather(1, cav_target.unsqueeze(1)).squeeze(1)
    )

    return -(ego_terms.mean() + cav_terms.mean()) / 2
