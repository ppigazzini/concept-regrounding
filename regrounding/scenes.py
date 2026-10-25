"""Synthetic scenes in two styles, with the three concepts and the verdict they carry.

A scene holds two objects. Object A, coloured, carries the concepts; object B, grey, is the
reference for the spatial one and a distractor for the others.

| concept | true when |
|---|---|
| ``round`` | A is a circle rather than a square |
| ``tone`` | A has tone 1 rather than tone 0 |
| ``left`` | A's centre lies at least 2 pixels left of B's |

The verdict is the conjunction of the three concepts. The abstraction -- the concepts and the
rule -- is the same in both styles; what changes is how a concept looks:

- **source**: tone 1 is red and tone 0 blue, A's radius is 4 to 6, the edges are crisp;
- **target**: tone 1 is teal and tone 0 yellow, so a colour cue learned on the source reads
  inverted; A's radius is 3 to 7, and the image is blurred.

What tells A from B is the same in both styles: A is coloured and B is grey, at a level
drawn per scene, so brightness never tells them apart. The objects never touch, so
neither hides the other, and both styles place them over the same range.

Each concept is drawn independently with probability one half, so a verdict is true in
one scene out of eight. Every draw comes from the `numpy.random.Generator` passed in.
"""

import math
from dataclasses import dataclass

import numpy as np

CONCEPTS = ("round", "tone", "left")

type Colour = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Style:
    """How a domain draws the same abstraction."""

    name: str
    tone: tuple[Colour, Colour]  # tone 0, tone 1
    radius: tuple[int, int]
    blur: bool


SOURCE = Style("source", ((0.15, 0.25, 0.9), (0.9, 0.15, 0.1)), (4, 6), blur=False)
TARGET = Style("target", ((0.9, 0.8, 0.2), (0.15, 0.8, 0.8)), (3, 7), blur=True)

#: The range B's grey level is drawn from. It holds the luminance of every tone of both
#: styles, 0.28 to 0.78: at one fixed grey, A would be the darker object on the source and
#: the brighter one on the target, and a model that finds A by brightness loses it.
GREY = (0.2, 0.9)
#: The distance from an object's centre to the edge: the largest radius a style draws, 7,
#: plus 2. One margin for both styles, so where the objects lie is not part of the shift.
MARGIN = 9
#: B's radius, in both styles.
_RADIUS_B = (3, 5)
#: The smallest distance between the two shapes, in pixels: more than the diagonal of a
#: pixel, so at least one background pixel separates them once they are rasterised.
_GAP = 2
#: Placements tried before a scene too small to hold both objects apart is refused.
_TRIES = 10_000


def _extent(r: int, *, round_: bool) -> float:
    """Return the radius of the circle around a shape: a square's half-side is 0.85 r."""
    return r if round_ else 0.85 * math.sqrt(2.0) * r


def _mask(size: int, cx: float, cy: float, r: float, *, round_: bool) -> np.ndarray:
    ys, xs = np.mgrid[0:size, 0:size]
    if round_:
        return (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
    side = 0.85 * r
    return (np.abs(xs - cx) <= side) & (np.abs(ys - cy) <= side)


def _blur(image: np.ndarray) -> np.ndarray:
    """Average each pixel with its eight neighbours, edges padded by repetition."""
    padded = np.pad(image, ((0, 0), (1, 1), (1, 1)), mode="edge")
    size = image.shape[1]
    shifts = [padded[:, dy : dy + size, dx : dx + size] for dy in range(3) for dx in range(3)]
    return np.mean(shifts, axis=0)


def render(rng: np.random.Generator, style: Style, concepts: np.ndarray, size: int) -> np.ndarray:
    """Draw one scene with the given concepts, as a float32 array of shape (3, size, size)."""
    round_, tone, left = (bool(c) for c in concepts)
    low, high = style.radius
    ra, rb = int(rng.integers(low, high + 1)), int(rng.integers(_RADIUS_B[0], _RADIUS_B[1] + 1))
    round_b = bool(rng.integers(2))
    apart = _extent(ra, round_=round_) + _extent(rb, round_=round_b) + _GAP
    for _ in range(_TRIES):
        xa, xb, ya, yb = rng.uniform(MARGIN, size - MARGIN, size=4)
        relation = (xa + 2 < xb) if left else (xa > xb + 2)
        if relation and math.hypot(xa - xb, ya - yb) >= apart:
            break
    else:
        msg = f"no two objects {apart:.1f} pixels apart fit a {size}-pixel scene"
        raise ValueError(msg)
    image = np.full((3, size, size), 0.05, dtype=np.float64)
    image[:, _mask(size, xb, yb, rb, round_=round_b)] = rng.uniform(*GREY)
    image[:, _mask(size, xa, ya, ra, round_=round_)] = np.array(style.tone[int(tone)])[:, None]
    if style.blur:
        image = _blur(image)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def dataset(
    rng: np.random.Generator, style: Style, n: int, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` scenes: images (n, 3, size, size) float32 and concepts (n, 3) uint8."""
    concepts = rng.integers(0, 2, size=(n, len(CONCEPTS))).astype(np.uint8)
    images = np.stack([render(rng, style, c, size) for c in concepts])
    return images, concepts


def verdicts(concepts: np.ndarray) -> np.ndarray:
    """Return the rule's verdict for each row of concepts: their conjunction."""
    return concepts.all(axis=-1).astype(np.uint8)
