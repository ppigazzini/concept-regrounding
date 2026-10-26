"""The fixed rule: the conjunction of the three concepts. It is never trained or re-grounded.

On concept bits it is a conjunction; on concept logits its score is the log of the product
of the concept probabilities, the log-probability of the conjunction for independent
concepts, which ranks scenes for the matched-sensitivity measure and carries the gradient
when a model is trained from verdicts alone; on concept prediction sets it is three-valued.
"""

import torch
import torch.nn.functional as F


def verdict(bits: torch.Tensor) -> torch.Tensor:
    """Return the conjunction of each row of concept bits."""
    return bits.bool().all(dim=-1)


def log_score(logits: torch.Tensor) -> torch.Tensor:
    """Return the log of the product of each row's concept probabilities, from logits.

    The sum of the log-sigmoids: a product of float32 probabilities underflows to 0 for
    scenes far from the conjunction and ties them, and a tie at the matched-sensitivity
    threshold counts every tied negative as a false positive.
    """
    return F.logsigmoid(logits).sum(dim=-1)


def three_valued(has0: torch.Tensor, has1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (decided, value) for each row of concept prediction sets.

    ``has0``/``has1`` say whether each concept's set contains 0 and 1. The verdict is true
    when every set is exactly {1}, false when some set is exactly {0}, and undecided
    otherwise -- an empty set included, since it rules out both values.
    """
    only1 = has1 & ~has0
    only0 = has0 & ~has1
    true = only1.all(dim=-1)
    false = only0.any(dim=-1)
    return true | false, true
