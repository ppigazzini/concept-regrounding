"""Split conformal prediction sets, one per concept.

The nonconformity of a label is one minus the probability the model gives it. On ``n``
calibration scenes the threshold is the ``ceil((n + 1)(1 - alpha))``-th smallest score,
and a label enters a scene's set when its nonconformity is at most the threshold: on a
scene exchangeable with the calibration scenes, the set holds the true label with
probability at least ``1 - alpha`` (Vovk et al. 2005; Angelopoulos and Bates 2021).

The guarantee rests on exchangeability, which a domain shift breaks: thresholds calibrated
on the source say nothing about coverage on the target, and the study measures how much
they lose.
"""

import math

import torch


def nonconformity(probabilities: torch.Tensor, label: int) -> torch.Tensor:
    """Return one minus the probability of ``label``, elementwise.

    Calibration and prediction both score through here, so a label is scored by the same
    floating-point operations on both sides: a test scene identical to a calibration
    scene then reaches the threshold exactly instead of missing it by a rounding.
    """
    given = probabilities if label == 1 else 1.0 - probabilities
    return 1.0 - given


def thresholds(probabilities: torch.Tensor, labels: torch.Tensor, alpha: float) -> torch.Tensor:
    """Return one threshold per concept from calibration probabilities and labels.

    When the rank passes the ``n`` calibration scores, fewer than ``1 / alpha - 1`` scenes,
    the threshold is infinite and every set holds both labels: the largest score in its
    place would give up the guarantee.
    """
    n = probabilities.shape[0]
    scores = torch.where(
        labels.bool(), nonconformity(probabilities, 1), nonconformity(probabilities, 0)
    )
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return torch.full(scores.shape[1:], math.inf, dtype=scores.dtype)
    return scores.sort(dim=0).values[rank - 1]


def prediction_sets(
    probabilities: torch.Tensor, threshold: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (has0, has1): whether each concept's set holds 0 and holds 1."""
    has0 = nonconformity(probabilities, 0) <= threshold
    has1 = nonconformity(probabilities, 1) <= threshold
    return has0, has1
