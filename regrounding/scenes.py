"""Synthetic scenes in two styles, with the three concepts and the verdict they carry.

A scene holds two objects. Object A, colored, carries the concepts; object B, gray, is the
reference for the spatial one and a distractor for the others.

| concept | true when |
|---|---|
| ``round`` | A is a circle rather than a square |
| ``tone`` | A has tone 1 rather than tone 0 |
| ``left`` | A's center lies at least 2 pixels left of B's |

The verdict is the conjunction of the three concepts. The abstraction -- the concepts and the
rule -- is the same in both styles; what changes is how a concept looks:

- **source**: tone 1 is a red and tone 0 a blue, each drawn from a family of hues, A's
  radius is 4 to 6, and the edges are crisp;
- **target**: tone 1 is teal and tone 0 yellow, hues outside both source families and
  nearer the other tone's, so a color cue learned on the source reads inverted; A's
  radius is 3 to 7, and the image is blurred.

A source tone spans 60 degrees of hue, so that no one color goes with a shape or a
position in the source scenes.

What tells A from B is the same in both styles: A is colored and B is gray, at a level
drawn per scene, so brightness never tells them apart. The objects never touch, so
neither hides the other; their spacing does not depend on A's shape, and both styles
place them over the same range.

Each concept is drawn independently with probability one half, so a verdict is true in
one scene out of eight. Every draw comes from the `numpy.random.Generator` passed in.
"""

import colorsys
import math
from dataclasses import dataclass

import numpy as np

CONCEPTS = ("round", "tone", "left")

type Hues = tuple[float, float]

#: The saturation and value of every tone; a tone is a range of hues, in degrees.
SATURATION, VALUE = 0.8, 0.9


@dataclass(frozen=True, slots=True)
class Style:
    """How a domain draws the same abstraction."""

    name: str
    hues: tuple[Hues, Hues]  # tone 0, tone 1
    radius: tuple[int, int]
    blur: bool


SOURCE = Style("source", ((200.0, 260.0), (-30.0, 30.0)), (4, 6), blur=False)
TARGET = Style("target", ((50.0, 50.0), (180.0, 180.0)), (3, 7), blur=True)
#: The target with one of its three changes alone and the source's look otherwise, so a
#: study on each says which change moves which count.
TARGETS = {
    "target": TARGET,
    "palette": Style("palette", TARGET.hues, SOURCE.radius, blur=False),
    "size": Style("size", SOURCE.hues, TARGET.radius, blur=False),
    "blur": Style("blur", SOURCE.hues, SOURCE.radius, blur=True),
}


def color(hue: float) -> tuple[float, float, float]:
    """Return the RGB color of a hue in degrees, at the tones' saturation and value."""
    return colorsys.hsv_to_rgb((hue / 360.0) % 1.0, SATURATION, VALUE)


#: The range B's gray level is drawn from. It holds the luminance of every tone of both
#: styles, 0.23 to 0.76, so brightness alone never tells A from B; color does.
GRAY = (0.2, 0.9)
#: The distance from an object's center to the edge: the largest radius a style draws, 7,
#: plus 2. One margin for both styles, so where the objects lie is not part of the shift.
MARGIN = 9
#: B's radius, in both styles.
_RADIUS_B = (3, 5)
#: The smallest distance between the two shapes, in pixels: more than the diagonal of a
#: pixel, so at least one background pixel separates them once they are rasterized.
_GAP = 2
#: The radius of the circle around either shape, per unit of its radius: a square's
#: half-side is 0.85 r, so its corner lies 0.85 times the square root of 2, 1.2 r, out.
_EXTENT = 0.85 * math.sqrt(2.0)
#: Placements tried before a scene too small to hold both objects apart is refused.
_TRIES = 10_000


def _mask(size: int, cx: float, cy: float, r: float, *, round_: bool) -> np.ndarray:
    ys, xs = np.mgrid[0:size, 0:size]
    if round_:
        return (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
    side = 0.85 * r
    return (np.abs(xs - cx) <= side) & (np.abs(ys - cy) <= side)


def _blur(image: np.ndarray) -> np.ndarray:
    """Average each pixel with its eight neighbors, edges padded by repetition."""
    padded = np.pad(image, ((0, 0), (1, 1), (1, 1)), mode="edge")
    size = image.shape[1]
    shifts = [padded[:, dy : dy + size, dx : dx + size] for dy in range(3) for dx in range(3)]
    return np.mean(shifts, axis=0)


def place(
    rng: np.random.Generator, ra: int, rb: int, *, left: bool, size: int
) -> tuple[float, float, float, float]:
    """Draw the centers of A and B, ``(xa, xb, ya, yb)``: the relation holds, the shapes apart.

    The spacing depends on the radii alone, every shape counted at a square's extent: kept
    nearer B as a circle than as a square, A's position would carry ``round``.
    """
    apart = _EXTENT * (ra + rb) + _GAP
    for _ in range(_TRIES):
        xa, xb, ya, yb = rng.uniform(MARGIN, size - MARGIN, size=4)
        relation = (xa + 2 < xb) if left else (xa > xb + 2)
        if relation and math.hypot(xa - xb, ya - yb) >= apart:
            return float(xa), float(xb), float(ya), float(yb)
    msg = f"no two objects {apart:.1f} pixels apart fit a {size}-pixel scene"
    raise ValueError(msg)


def render(rng: np.random.Generator, style: Style, concepts: np.ndarray, size: int) -> np.ndarray:
    """Draw one scene with the given concepts, as a float32 array of shape (3, size, size)."""
    round_, tone, left = (bool(c) for c in concepts)
    low, high = style.radius
    ra, rb = int(rng.integers(low, high + 1)), int(rng.integers(_RADIUS_B[0], _RADIUS_B[1] + 1))
    round_b = bool(rng.integers(2))
    xa, xb, ya, yb = place(rng, ra, rb, left=left, size=size)
    image = np.full((3, size, size), 0.05, dtype=np.float64)
    image[:, _mask(size, xb, yb, rb, round_=round_b)] = rng.uniform(*GRAY)
    hue = rng.uniform(*style.hues[int(tone)])
    image[:, _mask(size, xa, ya, ra, round_=round_)] = np.array(color(hue))[:, None]
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


def paired(
    rng: np.random.Generator, style: Style, n: int, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` scenes in complementary pairs, the first a positive and its complement.

    Scene ``2 i + 1`` carries the negation of scene ``2 i``'s concepts, so the first ``k``
    scenes hold both values of every concept at every even ``k``, and a positive from
    ``k = 1``. Drawn independently, the first 2 scenes leave some concept with one value with
    probability 0.88, and the first 12 hold no positive with probability 0.20. Each concept
    is still true in half the scenes.
    """
    halves = rng.integers(0, 2, size=((n + 1) // 2, len(CONCEPTS))).astype(np.uint8)
    halves[0] = 1
    concepts = np.empty((2 * len(halves), len(CONCEPTS)), dtype=np.uint8)
    concepts[0::2], concepts[1::2] = halves, 1 - halves
    concepts = concepts[:n]
    images = np.stack([render(rng, style, c, size) for c in concepts])
    return images, concepts


def verdicts(concepts: np.ndarray) -> np.ndarray:
    """Return the rule's verdict for each row of concepts: their conjunction."""
    return concepts.all(axis=-1).astype(np.uint8)
