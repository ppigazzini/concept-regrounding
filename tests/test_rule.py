import torch

from regrounding import rule


class TestRule:
    def test_the_verdict_is_the_conjunction_of_the_concepts(self):
        bits = torch.tensor([[1, 1, 1], [1, 0, 1], [0, 0, 0]])
        assert rule.verdict(bits).tolist() == [True, False, False]

    def test_the_score_is_the_product_of_the_probabilities(self):
        assert torch.allclose(rule.score(torch.tensor([[0.5, 0.5, 0.8]])), torch.tensor([0.2]))


class TestThreeValued:
    def _sets(self, *rows):
        has0 = torch.tensor([[0 in s for s in r] for r in rows])
        has1 = torch.tensor([[1 in s for s in r] for r in rows])
        return has0, has1

    def test_every_set_exactly_one_is_a_decided_true(self):
        decided, value = rule.three_valued(*self._sets(({1}, {1}, {1})))
        assert decided.tolist() == [True]
        assert value.tolist() == [True]

    def test_one_set_exactly_zero_is_a_decided_false_whatever_the_others(self):
        decided, value = rule.three_valued(*self._sets(({0}, {0, 1}, set())))
        assert decided.tolist() == [True]
        assert value.tolist() == [False]

    def test_an_ambiguous_or_empty_set_without_a_zero_leaves_it_undecided(self):
        decided, _ = rule.three_valued(*self._sets(({1}, {0, 1}, {1}), ({1}, set(), {1})))
        assert decided.tolist() == [False, False]
