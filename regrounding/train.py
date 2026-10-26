"""Training from concept labels or from verdicts, and re-grounding on a few scenes.

Three supervisions. With concept labels each output is supervised as the concept it
names. With verdicts through the rule only the conjunction is supervised, through the
rule's score of the concept outputs, and the outputs are free to mean whatever makes the
conjunction right: the setting in which reasoning shortcuts arise (Marconato et al. 2023).
With verdicts directly a single output is the verdict, with no concepts and no rule: the
baseline that says what the fixed rule adds.

Re-grounding fine-tunes the network on the ``k`` labelled target scenes; the rule is
untouched.

No step calls an operation torch computes on the CPU through MKL's vector maths
(``sqrt``, ``exp``, ``log``, ``tanh``, ``erf`` and the trigonometric functions, listed in
ATen's ``cpu/vml.h``): MKL picks that kernel by CPU vendor, whatever its conditional
numerical reproducibility mode, and an AMD and an Intel processor round the results
differently. Adam runs as torch's fused CPU kernel, the verdict loss takes its logarithm
and exponential from ATen's own loops over glibc, and the converged fit's line search
takes Python's square root.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast, override

import torch
import torch.nn.functional as F
from torch import nn
from torch.overrides import TorchFunctionMode

from regrounding import rule
from regrounding.model import ConceptNet

type Mode = Literal["concepts", "verdict", "direct"]
type Scope = Literal["first", "readout", "head", "all"]
type Fit = Literal["steps", "converged"]

#: L-BFGS iterations a converged fit may take; on 64 scenes it stops at its tolerance first.
_MAX_ITER = 1000

#: The largest log-probability of the conjunction the loss takes, so that the log of its
#: complement stays finite; in float64 it is reached only when every logit is above 690.
_LOG_ALMOST_ONE = -1e-300


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Return ``log(1 - exp(x))`` for ``x < 0``, accurate at both ends.

    Each branch is evaluated on inputs clamped to its own side, so the branch
    `torch.where` discards holds no infinity whose zero weight would make the
    gradient NaN. The logarithm is ``xlogy`` with a factor of 1 and the exponential is
    ``sigmoid(x) / sigmoid(-x)``, which keeps both off MKL's vector maths.
    """
    half = -math.log(2.0)
    near = torch.special.xlogy(1.0, -torch.expm1(x.clamp(min=half, max=_LOG_ALMOST_ONE)))
    far_x = x.clamp(max=half)
    far = torch.log1p(-torch.sigmoid(far_x) / torch.sigmoid(-far_x))
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
        return F.binary_cross_entropy_with_logits(logits, concepts.to(logits.dtype))
    if mode == "direct":
        target = rule.verdict(concepts).to(logits.dtype)
        return F.binary_cross_entropy_with_logits(logits[:, 0], target)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, fused=True)
    model.train()
    n = images.shape[0]
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            loss(model(images[batch]), concepts[batch], mode).backward()
            optimizer.step()


@dataclass(frozen=True, slots=True)
class Regrounding:
    """How a model is fine-tuned on the ``k`` target scenes.

    ``scope`` is what is fine-tuned: ``first``, the first convolution, 448 weights;
    ``readout``, the last layer, 64 weights and a bias per concept; ``head``, both dense
    layers, which read the spatial feature map and hold 131,331 weights; ``all``, every
    weight. ``steps`` full-batch Adam steps at ``learning_rate``, or, with ``fit`` set to
    ``converged``, the readout fitted to convergence with an L2 pull of weight ``anchor``
    toward its source weights (L2-SP, Li et al. 2018): a fit that ends where the scenes and
    the pull balance, whatever the number of steps.
    """

    scope: Scope
    steps: int
    learning_rate: float
    fit: Fit = "steps"
    anchor: float = 0.0


def reground(
    model: ConceptNet,
    images: torch.Tensor,
    concepts: torch.Tensor,
    *,
    mode: Mode,
    how: Regrounding,
) -> None:
    """Fine-tune ``model`` on a handful of labelled scenes, the whole batch per step.

    The layers outside the scope stop requiring gradients for the steps, so the backward
    pass does not run through them; the weights it updates come out the same.
    """
    if images.shape[0] == 0:
        return
    if how.fit == "converged":
        _refit_readout(model, images, concepts, mode, how.anchor)
        return
    layers = {
        "first": model.features[0],
        "readout": model.head[-1],
        "head": model.head,
        "all": model,
    }
    trained = list(layers[how.scope].parameters())
    inside = {id(p) for p in trained}
    frozen = [p for p in model.parameters() if id(p) not in inside]
    for p in frozen:
        p.requires_grad_(False)  # noqa: FBT003 - torch's own positional flag
    optimizer = torch.optim.Adam(trained, lr=how.learning_rate, fused=True)
    model.train()
    try:
        for _ in range(how.steps):
            optimizer.zero_grad()
            loss(model(images), concepts, mode).backward()
            optimizer.step()
    finally:
        for p in frozen:
            p.requires_grad_(True)  # noqa: FBT003


class _PythonSqrt(TorchFunctionMode):
    """Take the square root of a 0-dim tensor with Python's ``math.sqrt``.

    The strong-Wolfe line search of ``torch.optim.LBFGS`` takes the square root of a
    0-dim tensor, which torch computes through MKL's vector maths; ``math.sqrt`` is the
    processor's correctly rounded instruction.
    """

    @override
    def __torch_function__(
        self,
        func: Callable[..., object],
        types: Sequence[type],
        args: Sequence[object] = (),
        kwargs: Mapping[str, object] | None = None,
    ) -> object:
        x = args[0] if args else None
        if func is torch.Tensor.sqrt and isinstance(x, torch.Tensor) and x.dim() == 0:
            return torch.tensor(math.sqrt(x.item()), dtype=x.dtype)
        return func(*args, **(kwargs or {}))


def _refit_readout(
    model: ConceptNet, images: torch.Tensor, concepts: torch.Tensor, mode: Mode, anchor: float
) -> None:
    """Fit the readout on frozen features to convergence, pulled toward its source weights.

    In float64 with L-BFGS: under concept supervision the objective is a logistic
    regression per concept plus a quadratic pull, strictly convex, so its minimum does not
    depend on the optimiser's path.
    """
    readout = cast("nn.Linear", model.head[-1])
    with torch.no_grad():
        features = model.head[:-1](model.features(images)).double()
    w0, b0 = readout.weight.detach().double(), readout.bias.detach().double()
    w, b = w0.clone().requires_grad_(True), b0.clone().requires_grad_(True)  # noqa: FBT003
    optimizer = torch.optim.LBFGS(
        [w, b],
        max_iter=_MAX_ITER,
        tolerance_grad=1e-10,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def objective() -> torch.Tensor:
        optimizer.zero_grad()
        pull = (w - w0).pow(2).sum() + (b - b0).pow(2).sum()
        value = loss(features @ w.T + b, concepts, mode) + 0.5 * anchor * pull
        value.backward()
        return value

    with _PythonSqrt():
        optimizer.step(objective)
    with torch.no_grad():
        readout.weight.copy_(w)
        readout.bias.copy_(b)


@torch.no_grad()
def logits(model: nn.Module, images: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    """Return the concept logits of every scene, in evaluation mode."""
    model.eval()
    return torch.cat([model(images[i : i + batch_size]) for i in range(0, len(images), batch_size)])
