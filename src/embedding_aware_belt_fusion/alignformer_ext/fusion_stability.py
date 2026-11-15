"""Object-level stability certificate for an already estimated correction."""

from __future__ import annotations

import numpy as np


def _move(points: np.ndarray, theta: np.ndarray) -> np.ndarray:
    yaw, tx, ty = map(float, theta)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return points @ rotation.T + np.array([tx, ty])


def _signature_and_energy(ego: np.ndarray, cav: np.ndarray, theta: np.ndarray, gate: float):
    moved = _move(cav, theta)
    if not len(ego) or not len(moved):
        return (), float((len(ego) + len(moved)) * gate * gate)
    distance = np.linalg.norm(ego[:, None, :] - moved[None, :, :], axis=-1)
    ego_nearest = distance.argmin(axis=1)
    cav_nearest = distance.argmin(axis=0)
    pairs = tuple(
        (row, int(column)) for row, column in enumerate(ego_nearest)
        if cav_nearest[column] == row and distance[row, column] <= gate
    )
    # Symmetric truncated Chamfer energy: unmatched/distant objects cannot
    # dominate, while every association decision inside the gate contributes.
    energy = float(np.minimum(distance.min(axis=1) ** 2, gate * gate).sum())
    energy += float(np.minimum(distance.min(axis=0) ** 2, gate * gate).sum())
    return pairs, energy


def certify(inputs, decided: np.ndarray, *, gate_m: float = 1.0) -> np.ndarray:
    """Keep corrections whose associations are stable from Propose to final.

    A correction is certified when Propose and the final estimator induce the
    same mutual-nearest association set (at least two objects) and both reduce
    symmetric truncated association energy relative to the reported pose.
    This adds no learned score or noise-level-dependent threshold.
    """
    output = decided.copy()
    for local, pair_index in enumerate(inputs.attempted):
        final = decided[pair_index]
        if not np.any(np.abs(final) > 1e-9):
            continue
        batch = inputs.batches[local]
        ego = batch["ego_boxes"][0, :, :2].detach().cpu().numpy().astype(np.float64)
        cav = batch["cav_boxes"][0, :, :2].detach().cpu().numpy().astype(np.float64)
        proposal = inputs.cache["propose"][pair_index]
        identity_pairs, identity_energy = _signature_and_energy(ego, cav, np.zeros(3), gate_m)
        proposal_pairs, proposal_energy = _signature_and_energy(ego, cav, proposal, gate_m)
        final_pairs, final_energy = _signature_and_energy(ego, cav, final, gate_m)
        certified = (
            len(final_pairs) >= 2
            and proposal_pairs == final_pairs
            and max(proposal_energy, final_energy) < identity_energy
        )
        if not certified:
            output[pair_index] = 0.0
    return output


__all__ = ["certify"]
