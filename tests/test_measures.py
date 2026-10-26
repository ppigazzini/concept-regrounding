import torch

from regrounding.experiment import Key, _false_positives_at, measure, measure_direct


def _measure(logits, concepts, threshold=0.5):
    return measure(
        torch.tensor(logits),
        torch.tensor(concepts),
        torch.full((3,), threshold),
        0.9,
        mode="concepts",
        key=Key(0, 0, "regrounded"),
    )


# Four scenes: the outputs read round, tone inverted, and a constant "yes" for left.
LOGITS = ((4.0, -4.0, 4.0), (-4.0, 4.0, 4.0), (4.0, 4.0, 4.0), (-4.0, -4.0, 4.0))
CONCEPTS = ((1, 1, 1), (0, 0, 1), (1, 0, 0), (0, 1, 0))


class TestMeasure:
    def test_agreement_counts_each_output_against_each_concept(self):
        row = _measure(LOGITS, CONCEPTS)
        assert row["agreement"] == [[4, 2, 2], [2, 0, 2], [2, 2, 2]]

    def test_each_output_is_counted_against_the_verdict(self):
        row = _measure(LOGITS, CONCEPTS)
        assert row["positives"] == 1
        assert row["verdict_agreement"] == [3, 1, 1]

    def test_true_positives_are_the_verdicts_right_on_positive_scenes(self):
        row = _measure(LOGITS, CONCEPTS)
        assert row["verdict_correct"] == 2
        assert row["true_positives"] == 0

    def test_a_set_is_empty_full_or_a_singleton(self):
        confident = _measure(LOGITS, CONCEPTS, threshold=0.5)
        assert confident["empty"] == [0, 0, 0]
        assert confident["full"] == [0, 0, 0]
        assert confident["covered"] == [4, 0, 2]
        everything = _measure(LOGITS, CONCEPTS, threshold=1.0)
        assert everything["full"] == [4, 4, 4]
        assert everything["covered"] == [4, 4, 4]
        nothing = _measure(LOGITS, CONCEPTS, threshold=-1.0)
        assert nothing["empty"] == [4, 4, 4]
        assert nothing["covered"] == [0, 0, 0]


class TestMatchedSensitivity:
    def test_the_threshold_is_the_highest_that_reaches_the_sensitivity(self):
        scores = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
        truth = torch.tensor([True, False, True, False, True, False])
        # 90 % of 3 positives needs all 3, so the threshold is the lowest positive, 1.0.
        assert _false_positives_at(scores, truth, 0.9) == 2

    def test_a_tie_at_the_threshold_counts_every_tied_negative(self):
        scores = torch.tensor([1.0, 1.0, 1.0, 0.0])
        truth = torch.tensor([True, False, False, False])
        assert _false_positives_at(scores, truth, 0.9) == 2

    def test_no_positive_scene_counts_no_false_positive(self):
        assert _false_positives_at(torch.tensor([1.0]), torch.tensor([False]), 0.9) == 0


class TestMeasureDirect:
    def test_the_single_output_is_the_verdict(self):
        row = measure_direct(
            torch.tensor([[2.0], [-3.0], [3.0], [-1.0]]),
            torch.tensor(CONCEPTS),
            0.9,
            key=Key(0, 4, "regrounded", draw=2),
        )
        assert (row["shots"], row["draw"], row["domain"]) == (4, 2, "target")
        assert (row["positives"], row["negatives"]) == (1, 3)
        assert row["true_positives"] == 1
        assert row["verdict_correct"] == 3
        assert row["false_positives"] == 1
