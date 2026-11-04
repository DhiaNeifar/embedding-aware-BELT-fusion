"""Log-domain Sinkhorn with dustbins, for soft cross-agent correspondence.

Objects seen by only one agent have no counterpart, so the score matrix
carries a dustbin row and column that absorb their mass. Running Sinkhorn in
the log domain keeps it stable for the peaked score matrices a trained model
produces.

Follows Sarlin et al., "SuperGlue: Learning Feature Matching with Graph
Neural Networks", CVPR 2020.
"""

from __future__ import annotations

import torch
from torch import Tensor

DEFAULT_SINKHORN_ITERATIONS = 20


def log_sinkhorn(scores: Tensor, alpha: Tensor, iterations: int) -> Tensor:
    """Normalize ``scores`` into a log-domain soft assignment with dustbins.

    Parameters
    ----------
    scores: ``(B, M, N)`` match scores between ego and CAV objects.
    alpha: scalar tensor, the learnable dustbin score.
    iterations: number of Sinkhorn normalization steps.

    Returns
    -------
    Tensor
        ``(B, M + 1, N + 1)`` log-assignment. The last row and column are the
        dustbins.

        In the non-degenerate case (``rows > 0`` and ``columns > 0``), this is
        a proper Sinkhorn normalization: every *real* row and every *real*
        column sums to 1 after ``.exp()``. The dustbin row/column is a shared
        bucket that can absorb leftover mass from several real rows/columns
        at once, so it does not itself sum to 1 -- the dustbin row sums to
        ``columns`` and the dustbin column sums to ``rows``.

        That row/column-sums-to-1 contract does NOT hold in the degenerate
        ``rows == 0`` or ``columns == 0`` case (an agent that detected
        nothing) -- the guard below returns ``couplings * 0``, i.e. every
        entry is exactly ``log(1) == 0``, so every entry of ``.exp()`` is 1,
        not a per-row/column probability distribution. The semantic content
        is still correct: an all-zero log-assignment says "every real object
        on the other side matches its dustbin with certainty"; it is just not
        phrased as a normalized distribution. ``couplings * 0`` is used
        deliberately instead of the equal-valued ``torch.zeros_like``:
        multiplying by zero keeps the result attached to the autograd graph
        (with a correctly zero-valued gradient into ``scores``/``alpha``),
        whereas ``zeros_like`` would return a disconnected constant and
        ``.backward()`` would then raise on this path.
    """
    if scores.dim() != 3:
        raise ValueError(f"scores must be (B, M, N), got {tuple(scores.shape)}")
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")

    batch, rows, columns = scores.shape
    bin_score = alpha.to(scores)

    couplings = torch.cat(
        [
            torch.cat([scores, bin_score.expand(batch, rows, 1)], dim=2),
            torch.cat(
                [bin_score.expand(batch, 1, columns), bin_score.expand(batch, 1, 1)],
                dim=2,
            ),
        ],
        dim=1,
    )

    if rows == 0 or columns == 0:
        return couplings * 0

    row_count = scores.new_tensor(float(rows))
    column_count = scores.new_tensor(float(columns))

    normalizer = -(row_count + column_count).log()
    log_mu = torch.cat(
        [normalizer.expand(rows), column_count.log().reshape(1) + normalizer]
    ).expand(batch, -1)
    log_nu = torch.cat(
        [normalizer.expand(columns), row_count.log().reshape(1) + normalizer]
    ).expand(batch, -1)

    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iterations):
        u = log_mu - torch.logsumexp(couplings + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(couplings + u.unsqueeze(2), dim=1)

    return couplings + u.unsqueeze(2) + v.unsqueeze(1) - normalizer
