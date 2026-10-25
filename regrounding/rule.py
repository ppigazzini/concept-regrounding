"""The fixed rule: the conjunction of the three concepts. It is never trained or re-grounded.

On concept bits it is a conjunction; on concept probabilities its score is their product,
the probability of the conjunction for independent concepts, which ranks scenes for the
matched-sensitivity measure and carries the gradient when a model is trained from
verdicts alone; on concept prediction sets it is three-valued.
"""

import torch


def verdict(bits: torch.Tensor) -> torch.Tensor:
    """Return the conjunction of each row of concept bits."""
    return bits.bool().all(dim=-1)


def score(probabilities: torch.Tensor) -> torch.Tensor:
    """Return the product of each row of concept probabilities."""
    return probabilities.prod(dim=-1)


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
