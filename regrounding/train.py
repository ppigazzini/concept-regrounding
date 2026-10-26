"""Training from concept labels or from verdicts alone, and re-grounding on a few scenes.

With concept labels each output is supervised as the concept it names. With verdicts
alone only the conjunction is supervised, through the product of the concept
probabilities, and the outputs are free to mean whatever makes the conjunction right: the
setting in which reasoning shortcuts arise (Marconato et al. 2023), measured by comparing
the two.

Re-grounding fine-tunes the network on the ``k`` labelled target scenes with the same
supervision it was trained with; the rule is untouched.
"""

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from regrounding import rule
from regrounding.model import ConceptNet

type Mode = Literal["concepts", "verdict"]
type Scope = Literal["readout", "head", "all"]

#: The largest log-probability of the conjunction the loss takes, so that the log of its
#: complement stays finite; in float64 it is reached only when every logit is above 690.
_LOG_ALMOST_ONE = -1e-300


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Return ``log(1 - exp(x))`` for ``x < 0``, accurate at both ends.

    Each branch is evaluated on inputs clamped to its own side, so the branch
    `torch.where` discards holds no infinity whose zero weight would make the
    gradient NaN.
    """
    half = -math.log(2.0)
    near = torch.log(-torch.expm1(x.clamp(min=half, max=_LOG_ALMOST_ONE)))
    far = torch.log1p(-torch.exp(x.clamp(max=half)))
    return torch.where(x > half, near, far)


def loss(logits: torch.Tensor, concepts: torch.Tensor, mode: Mode) -> torch.Tensor:
    """Return the training loss of a batch under the given supervision.

    Under verdict supervision the loss is the cross-entropy of the rule's log-score, in
    float64. A product taken in probability space and clamped away from 0 and 1 has no
    gradient past the clamp, so a positive scene whose product falls below it is never
    pulled back; in float32 a negative scene with every logit above 75 has a log-score
    that rounds to 0 and no gradient either.
    """
    if mode == "concepts":
        return F.binary_cross_entropy_with_logits(logits, concepts.float())
    log_p = rule.log_score(logits.double())
    target = rule.verdict(concepts).double()
    return -(target * log_p + (1.0 - target) * _log1mexp(log_p)).mean()


def fit(
    model: nn.Module,
    images: torch.Tensor,
    concepts: torch.Tensor,
    *,
    mode: Mode,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    generator: torch.Generator,
) -> None:
    """Train ``model`` on the scenes for ``epochs`` passes, in a seeded order."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    n = images.shape[0]
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            loss(model(images[batch]), concepts[batch], mode).backward()
            optimizer.step()


def reground(
    model: ConceptNet,
    images: torch.Tensor,
    concepts: torch.Tensor,
    *,
    mode: Mode,
    steps: int,
    learning_rate: float,
    scope: Scope,
) -> None:
    """Fine-tune ``model`` on a handful of labelled target scenes, the whole batch per step.

    ``scope`` is what is fine-tuned: ``readout``, the last layer, 64 weights and a bias
    per concept; ``head``, both dense layers, which read the spatial feature map and hold
    131,331 weights; ``all``, every weight, the 14,336 convolution weights included.

    The layers outside the scope stop requiring gradients for the steps, so the backward
    pass does not run through them; the weights it updates come out the same.
    """
    if images.shape[0] == 0 or steps == 0:
        return
    layers = {"readout": model.head[-1], "head": model.head, "all": model}
    trained = list(layers[scope].parameters())
    inside = {id(p) for p in trained}
    frozen = [p for p in model.parameters() if id(p) not in inside]
    for p in frozen:
        p.requires_grad_(False)  # noqa: FBT003 - torch's own positional flag
    optimizer = torch.optim.Adam(trained, lr=learning_rate)
    model.train()
    try:
        for _ in range(steps):
            optimizer.zero_grad()
            loss(model(images), concepts, mode).backward()
            optimizer.step()
    finally:
        for p in frozen:
            p.requires_grad_(True)  # noqa: FBT003


@torch.no_grad()
def logits(model: nn.Module, images: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    """Return the concept logits of every scene, in evaluation mode."""
    model.eval()
    return torch.cat([model(images[i : i + batch_size]) for i in range(0, len(images), batch_size)])
