import json
from typing import cast

import pytest

from regrounding.config import Experiment
from regrounding.study import (
    DirectRow,
    Row,
    Study,
    StudyError,
    differences,
    dumps,
    first_shots,
    identity,
    largest_move,
    loads,
    matched,
    restrict,
    signed,
)


def _row(**changes) -> Row:
    row: dict = {
        "mode": "concepts",
        "seed": 0,
        "shots": 0,
        "domain": "target",
        "trained": "regrounded",
        "scenes": 100,
        "agreement": [[90, 50, 50], [50, 80, 50], [50, 50, 95]],
        "verdict_agreement": [60, 60, 60],
        "verdict_correct": 90,
        "true_positives": 8,
        "positives": 12,
        "negatives": 88,
        "false_positives": 5,
        "covered": [91, 92, 93],
        "empty": [2, 3, 1],
        "full": [10, 12, 9],
        "decided": 70,
        "decided_correct": 66,
    }
    return cast("Row", row | changes)


def _direct(**changes) -> DirectRow:
    row: dict = {
        "seed": 0,
        "shots": 0,
        "domain": "target",
        "trained": "regrounded",
        "scenes": 100,
        "verdict_correct": 85,
        "true_positives": 4,
        "positives": 12,
        "negatives": 88,
        "false_positives": 20,
    }
    return cast("DirectRow", row | changes)


def _study(*rows: Row, direct: tuple[DirectRow, ...] = ()) -> Study:
    return Study(
        format=2,
        device="cpu",
        settings=Experiment().as_record(),
        rows=list(rows) or [_row()],
        direct=list(direct) or [_direct()],
    )


class TestStudyFile:
    def test_a_written_study_reads_back_equal(self):
        assert loads(dumps(_study())) == _study()

    @pytest.mark.parametrize(
        ("edit", "named"),
        [
            (lambda r: r.__setitem__("verdict_correct", True), "rows[0].verdict_correct"),
            (lambda r: r.__setitem__("covered", [1, 2]), "rows[0].covered"),
            (lambda r: r.__setitem__("agreement", [[1, 2, 3]]), "rows[0].agreement"),
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
            ({"agreement": [[101, 0, 0], [0, 0, 0], [0, 0, 0]]}, "exceeds"),
            ({"decided_correct": 71}, "part exceeds"),
            ({"true_positives": 13}, "part exceeds"),
            ({"verdict_correct": 95, "true_positives": 6}, "part exceeds"),
            ({"full": [92, 12, 9]}, "both empty and covering"),
            ({"empty": [10, 3, 1]}, "both empty and covering"),
        ],
    )
    def test_counts_that_contradict_each_other_are_refused(self, changes, named):
        with pytest.raises(StudyError, match=named):
            dumps(_study(_row(**changes)))

    @pytest.mark.parametrize(
        ("changes", "named"),
        [
            ({"positives": 13}, r"direct\[0\]: positives"),
            ({"false_positives": 89}, r"direct\[0\]: a part exceeds"),
            ({"verdict_correct": 101}, r"direct\[0\]: a count exceeds"),
        ],
    )
    def test_a_direct_row_is_held_to_the_verdict_s_rules(self, changes, named):
        with pytest.raises(StudyError, match=named):
            dumps(_study(direct=(_direct(**changes),)))


class TestMatchings:
    # Output 0 is constant (right on the 50 round scenes of 100), output 1 is tone
    # inverted, output 2 is left: the identity scores 50 + 0 + 100.
    ROW = _row(agreement=[[50, 50, 50], [50, 0, 50], [50, 50, 100]])

    def test_the_identity_reads_the_diagonal(self):
        assert identity(self.ROW) == [50, 0, 100]

    def test_the_best_permutation_can_gain_without_renamed_concepts(self):
        assert matched(self.ROW) == 200

    def test_a_signed_matching_credits_the_inverted_output(self):
        assert signed(self.ROW) == 250


class TestCheck:
    def test_a_tolerance_accepts_a_small_move_and_names_a_large_one(self):
        moved = _study(_row(verdict_correct=91))
        assert differences(_study(), moved, tolerance=0.02) == []
        assert differences(_study(), moved) == ["rows[0].verdict_correct: 90 != 91"]

    def test_a_tolerance_applies_to_every_entry_of_a_matrix(self):
        moved = _study(_row(agreement=[[90, 50, 50], [50, 80, 50], [50, 50, 99]]))
        assert differences(_study(), moved, tolerance=0.05) == []
        assert len(differences(_study(), moved, tolerance=0.03)) == 1

    @pytest.mark.parametrize("key", ["seed", "shots"])
    def test_a_tolerance_never_accepts_a_row_of_another_seed_or_shot_count(self, key):
        moved = _study(_row(**{key: 1}))
        assert differences(_study(), moved, tolerance=0.05) == [f"rows[0].{key}: 0 != 1"]

    def test_the_largest_move_is_over_every_count_of_every_row(self):
        moved = _study(
            _row(agreement=[[90, 50, 50], [50, 80, 50], [50, 50, 88]]),
            direct=(_direct(false_positives=23),),
        )
        assert largest_move(_study(), moved) == 7
        assert largest_move(_study(), _study()) == 0

    def test_a_direct_row_is_compared_like_a_row(self):
        moved = _study(direct=(_direct(false_positives=25),))
        assert differences(_study(), moved) == ["direct[0].false_positives: 20 != 25"]
        assert differences(_study(), moved, tolerance=0.05) == []

    def test_a_tolerance_never_accepts_a_row_of_other_scenes(self):
        moved = _study(_row(positives=13, negatives=87))
        assert differences(_study(), moved, tolerance=0.05) == [
            "rows[0].positives: 12 != 13",
            "rows[0].negatives: 88 != 87",
        ]

    def test_a_slice_keeps_the_first_seeds_and_their_rows(self):
        study = Study(
            format=2,
            device="cpu",
            settings=Experiment(seeds=(3, 7, 9)).as_record(),
            rows=[_row(seed=3), _row(seed=7), _row(seed=9)],
            direct=[_direct(seed=3), _direct(seed=9)],
        )
        sliced = restrict(study, 2)
        assert sliced["settings"]["seeds"] == [3, 7]
        assert [r["seed"] for r in sliced["rows"]] == [3, 7]
        assert [r["seed"] for r in sliced["direct"]] == [3]
        assert sliced["settings"] | {"seeds": [3, 7, 9]} == study["settings"]


def _diagonal(round_, tone, left):
    return [[round_, 50, 50], [50, tone, 50], [50, 50, left]]


class TestFirstShots:
    def _study(self):
        rows = [
            _row(seed=seed, shots=k, domain=domain, trained=trained, agreement=_diagonal(*correct))
            for seed, k, domain, trained, correct in [
                (0, 0, "source", "source", [100, 100, 100]),
                (0, 0, "target", "regrounded", [95, 10, 92]),
                (0, 4, "target", "regrounded", [91, 91, 95]),
                (0, 3000, "target", "ceiling", [100, 100, 100]),
                (1, 0, "target", "regrounded", [50, 50, 50]),
                (1, 4, "target", "regrounded", [89, 90, 90]),
                (1, 3000, "target", "scratch", [100, 100, 100]),
            ]
        ]
        settings = Experiment(seeds=(0, 1), shots=(0, 4)).as_record()
        return Study(format=2, device="cpu", settings=settings, rows=rows, direct=[])

    def test_a_seed_s_entry_is_its_smallest_target_shot_count_at_the_share(self):
        first = first_shots(self._study(), "concepts", 0.9)
        assert first["round"] == [0, None]
        assert first["tone"] == [4, 4]
        assert first["left"] == [0, 4]

    def test_rows_trained_on_the_target_training_scenes_are_not_a_shot_count(self):
        assert first_shots(self._study(), "concepts", 0.99)["round"] == [None, None]

    def test_all_three_needs_every_concept_at_one_shot_count(self):
        assert first_shots(self._study(), "concepts", 0.9)["all three"] == [4, None]
