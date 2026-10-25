import pytest

from regrounding.config import Experiment


class TestSettings:
    def test_the_defaults_are_a_valid_experiment(self):
        assert Experiment().as_record()["shots"] == [0, 1, 2, 4, 6, 12]

    @pytest.mark.parametrize(
        ("changes", "named"),
        [
            ({"epochs": 0}, "epochs"),
            ({"learning_rate": 1.5}, "learning_rate"),
            ({"alpha": 0.7}, "alpha"),
            ({"test": True}, "test"),
            ({"shots": (0, -1)}, "shots"),
            ({"regrounding_scope": "everything"}, "regrounding_scope"),
            ({"image_size": 20}, "image_size"),
        ],
    )
    def test_a_setting_outside_its_declared_range_is_refused_with_its_name(self, changes, named):
        with pytest.raises(ValueError, match=named):
            Experiment(**changes)

    def test_the_pool_must_hold_the_largest_shot_count(self):
        with pytest.raises(ValueError, match="exceeds pool"):
            Experiment(shots=(0, 12), pool=8)
