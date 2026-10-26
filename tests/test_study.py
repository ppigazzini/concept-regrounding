import json
from typing import cast

import pytest

from regrounding.config import Experiment
from regrounding.study import Row, Study, StudyError, differences, dumps, first_shots, loads


def _row(**changes) -> Row:
    row: dict = {
        "mode": "concepts",
        "seed": 0,
        "shots": 0,
        "domain": "target",
        "scenes": 100,
        "concept_correct": [90, 80, 95],
        "permuted_correct": 265,
        "verdict_correct": 90,
        "positives": 12,
        "negatives": 88,
        "false_positives": 5,
        "covered": [91, 92, 93],
        "decided": 70,
        "decided_correct": 66,
    }
    return cast("Row", row | changes)


def _study(*rows: Row) -> Study:
    return Study(
        format=1, device="cpu", settings=Experiment().as_record(), rows=list(rows) or [_row()]
    )


class TestStudyFile:
    def test_a_written_study_reads_back_equal(self):
        assert loads(dumps(_study())) == _study()

    @pytest.mark.parametrize(
        ("edit", "named"),
        [
            (lambda r: r.__setitem__("verdict_correct", True), "rows[0].verdict_correct"),
            (lambda r: r.__setitem__("covered", [1, 2]), "rows[0].covered"),
            (lambda r: r.__setitem__("extra", 1), "rows[0].extra"),
            (lambda r: r.__setitem__("domain", "elsewhere"), "rows[0].domain"),
        ],
    )
    def test_a_malformed_row_is_refused_with_its_path(self, edit, named):
        payload = json.loads(dumps(_study()))
        edit(payload["rows"][0])
        with pytest.raises(StudyError, match=named.replace("[", r"\[").replace("]", r"\]")):
            loads(json.dumps(payload))

    @pytest.mark.parametrize(
        ("changes", "named"),
        [
            ({"positives": 13}, "positives"),
            ({"permuted_correct": 200}, "identity"),
            ({"permuted_correct": 301, "concept_correct": [100, 100, 100]}, "exceeds"),
            ({"decided_correct": 71}, "part exceeds"),
        ],
    )
    def test_counts_that_contradict_each_other_are_refused(self, changes, named):
        with pytest.raises(StudyError, match=named):
            dumps(_study(_row(**changes)))


class TestCheck:
    def test_a_tolerance_accepts_a_small_move_and_names_a_large_one(self):
        moved = _study(_row(verdict_correct=91))
        assert differences(_study(), moved, tolerance=0.02) == []
        assert differences(_study(), moved) == ["rows[0].verdict_correct: 90 != 91"]

    @pytest.mark.parametrize("key", ["seed", "shots"])
    def test_a_tolerance_never_accepts_a_row_of_another_seed_or_shot_count(self, key):
        moved = _study(_row(**{key: 1}))
        assert differences(_study(), moved, tolerance=0.05) == [f"rows[0].{key}: 0 != 1"]

    def test_a_tolerance_never_accepts_a_row_of_other_scenes(self):
        moved = _study(_row(positives=13, negatives=87))
        assert differences(_study(), moved, tolerance=0.05) == [
            "rows[0].positives: 12 != 13",
            "rows[0].negatives: 88 != 87",
        ]


class TestFirstShots:
    def _study(self):
        rows = [
            _row(seed=seed, shots=k, domain=domain, concept_correct=correct, permuted_correct=280)
            for seed, k, domain, correct in [
                (0, 0, "source", [100, 100, 100]),
                (0, 0, "target", [95, 10, 92]),
                (0, 4, "target", [91, 91, 95]),
                (1, 0, "target", [50, 50, 50]),
                (1, 4, "target", [89, 90, 90]),
            ]
        ]
        settings = Experiment(seeds=(0, 1), shots=(0, 4)).as_record()
        return Study(format=1, device="cpu", settings=settings, rows=rows)

    def test_a_seed_s_entry_is_its_smallest_target_shot_count_at_the_share(self):
        first = first_shots(self._study(), "concepts", 0.9)
        assert first["round"] == [0, None]
        assert first["tone"] == [4, 4]
        assert first["left"] == [0, 4]

    def test_all_three_needs_every_concept_at_one_shot_count(self):
        assert first_shots(self._study(), "concepts", 0.9)["all three"] == [4, None]
