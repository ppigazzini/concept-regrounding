import colorsys

import numpy as np
import pytest

from regrounding import scenes


def _draw(style, concepts, seed=0, size=32):
    return scenes.render(np.random.default_rng(seed), style, np.array(concepts), size)


def _objects(image):
    """Return the masks of object A, the coloured pixels, and of object B, the grey ones."""
    spread = image.max(axis=0) - image.min(axis=0)
    a = spread > scenes.SATURATION * scenes.VALUE - 1e-6
    b = (spread == 0) & (image[0] >= scenes.GREY[0] - 1e-6)
    return a, b


def _hue(image, mask):
    r, g, b = image[:, mask][:, 0]
    return 360.0 * colorsys.rgb_to_hsv(float(r), float(g), float(b))[0]


def _distance(hue, hues):
    """Return how many degrees ``hue`` lies outside the range ``hues``, 0 inside it."""
    low, high = hues
    offset = (hue - low) % 360.0
    return 0.0 if offset <= high - low else min(offset - (high - low), 360.0 - offset)


def _luminance(colour):
    return 0.2126 * colour[0] + 0.7152 * colour[1] + 0.0722 * colour[2]


class TestRendering:
    def test_a_scene_is_a_float_image_in_the_unit_range(self):
        image = _draw(scenes.SOURCE, [1, 0, 1])
        assert image.shape == (3, 32, 32)
        assert image.dtype == np.float32
        assert 0.0 <= image.min() <= image.max() <= 1.0

    def test_the_same_generator_draws_the_same_scene(self):
        assert np.array_equal(
            _draw(scenes.TARGET, [0, 1, 0], 7), _draw(scenes.TARGET, [0, 1, 0], 7)
        )

    @pytest.mark.parametrize("tone", [0, 1])
    def test_object_a_carries_a_hue_of_its_style_s_tone(self, tone):
        for seed in range(20):
            image = _draw(scenes.SOURCE, [1, tone, 1], seed)
            a, _ = _objects(image)
            assert _distance(_hue(image, a), scenes.SOURCE.hues[tone]) < 0.5

    @pytest.mark.parametrize("left", [0, 1])
    def test_the_relation_holds_in_the_drawn_positions(self, left):
        for seed in range(20):
            a, b = _objects(_draw(scenes.SOURCE, [1, 0, left], seed))
            assert (np.nonzero(a)[1].mean() < np.nonzero(b)[1].mean()) == bool(left)

    def test_the_two_objects_never_touch(self):
        for seed in range(200):
            a, b = _objects(_draw(scenes.SOURCE, [seed % 2, 1, 1], seed))
            padded = np.pad(a, 1)
            grown = np.zeros_like(a)
            for dy in range(3):
                for dx in range(3):
                    grown |= padded[dy : dy + 32, dx : dx + 32]
            assert a.any()
            assert b.any()
            assert not (grown & b).any()

    def test_a_scene_too_small_to_hold_both_objects_apart_is_refused(self):
        with pytest.raises(ValueError, match="fit a 20-pixel scene"):
            _draw(scenes.TARGET, [1, 1, 1], size=20)


class TestStyles:
    @pytest.mark.parametrize("tone", [0, 1])
    def test_a_target_tone_lies_outside_the_source_families_nearer_the_other_tone(self, tone):
        hue = scenes.TARGET.hues[tone][0]
        own, other = scenes.SOURCE.hues[tone], scenes.SOURCE.hues[1 - tone]
        assert 0.0 < _distance(hue, other) < _distance(hue, own)

    @pytest.mark.parametrize("style", [scenes.SOURCE, scenes.TARGET])
    def test_brightness_never_tells_object_a_from_object_b(self, style):
        low, high = scenes.GREY
        for hues in style.hues:
            for hue in np.linspace(*hues, 61):
                assert low < _luminance(scenes.colour(hue)) < high

    @pytest.mark.parametrize("style", [scenes.SOURCE, scenes.TARGET])
    def test_both_styles_place_the_objects_over_the_same_range(self, style):
        assert style.radius[1] + 2 <= scenes.MARGIN


class TestDatasets:
    def test_a_dataset_balances_each_concept_and_the_verdict_is_their_conjunction(self):
        images, concepts = scenes.dataset(np.random.default_rng(1), scenes.SOURCE, 800, 32)
        assert images.shape == (800, 3, 32, 32)
        assert np.all(np.abs(concepts.mean(axis=0) - 0.5) < 0.06)
        assert np.array_equal(scenes.verdicts(concepts), concepts.all(axis=1))

    @pytest.mark.parametrize("style", [scenes.SOURCE, scenes.TARGET])
    def test_every_style_draws_at_the_smallest_image_size_the_settings_admit(self, style):
        images, _ = scenes.dataset(np.random.default_rng(2), style, 500, 32)
        assert images.shape == (500, 3, 32, 32)
