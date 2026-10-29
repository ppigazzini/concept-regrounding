"""The concept extractor: a small convolutional network with one output per concept.

Two pooling stages bring a 32-pixel scene to 8 by 8, and the features stay laid out in
space up to a dense layer: the ``left`` concept is a relation between two positions, and
a global average would discard where things are.
"""

import math

import torch
from torch import nn


class ConceptNet(nn.Module):
    """Map an image to one logit per concept.

    The weights are drawn from the generator passed in, within torch's default bounds of one
    over the square root of the fan-in: the layers' own initialization draws from torch's
    global generator, which a count must not depend on.
    """

    def __init__(self, generator: torch.Generator, concepts: int = 3, size: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )
        side = size // 4
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * side * side, 64),
            nn.ReLU(),
            nn.Linear(64, concepts),
        )
        with torch.no_grad():
            for layer in self.modules():
                if isinstance(layer, nn.Conv2d | nn.Linear):
                    bound = 1.0 / math.sqrt(layer.weight[0].numel())
                    layer.weight.uniform_(-bound, bound, generator=generator)
                    if layer.bias is not None:
                        layer.bias.uniform_(-bound, bound, generator=generator)

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        """Return the concept logits, shape (batch, concepts)."""
        return self.head(self.features(images))
