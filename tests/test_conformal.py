import math

import torch

from regrounding import conformal


class TestConformalSets:
    def test_sets_cover_at_least_one_minus_alpha_on_exchangeable_scenes(self):
        generator = torch.Generator().manual_seed(0)
        n, alpha = 4000, 0.1
        labels = torch.randint(0, 2, (2 * n, 3), generator=generator)
        noise = torch.rand((2 * n, 3), generator=generator)
        probs = (
            torch.where(labels.bool(), 0.5 + 0.5 * noise, 0.5 - 0.5 * noise) * 0.9 + 0.05 * noise
        )
        q = conformal.thresholds(probs[:n], labels[:n], alpha)
        has0, has1 = conformal.prediction_sets(probs[n:], q)
        covered = torch.where(labels[n:].bool(), has1, has0).float().mean(dim=0)
        assert torch.all(covered >= 1 - alpha - 0.02)

    def test_a_perfect_model_gets_singleton_sets(self):
        probs = torch.tensor([[0.99, 0.01, 0.99]] * 50)
        labels = torch.tensor([[1, 0, 1]] * 50)
        has0, has1 = conformal.prediction_sets(probs, conformal.thresholds(probs, labels, 0.1))
        assert has1[:, 0].all()
        assert not has0[:, 0].any()
        assert has0[:, 1].all()
        assert not has1[:, 1].any()

    def test_too_few_calibration_scenes_give_full_sets(self):
        probs = torch.tensor([[0.9, 0.1, 0.6]] * 5)
        labels = torch.tensor([[1, 0, 1]] * 5)
        threshold = conformal.thresholds(probs, labels, 0.1)
        assert threshold.tolist() == [[math.inf] * 3] * 2
        has0, has1 = conformal.prediction_sets(probs, threshold)
        assert has0.all()
        assert has1.all()

    def test_coverage_holds_from_the_smallest_calibration_size(self):
        generator = torch.Generator().manual_seed(1)
        covered = trials = 0
        for n in (5, 9, 12):
            for _ in range(400):
                probs = torch.rand((n + 1, 1), generator=generator)
                labels = (torch.rand((n + 1, 1), generator=generator) < probs).long()
                q = conformal.thresholds(probs[:n], labels[:n], 0.1)
                has0, has1 = conformal.prediction_sets(probs[n:], q)
                covered += int(torch.where(labels[n:].bool(), has1, has0).item())
                trials += 1
        assert covered >= 0.9 * trials - 0.02 * trials


class TestConditionalSets:
    def _scenes(self, n, generator):
        """Scenes where the model is sure of label 1 and unsure of label 0."""
        labels = torch.randint(0, 2, (n, 1), generator=generator)
        noise = torch.rand((n, 1), generator=generator)
        probs = torch.where(labels.bool(), 0.9 + 0.1 * noise, 0.2 + 0.7 * noise)
        return probs, labels

    def _coverage(self, conditional):
        generator = torch.Generator().manual_seed(3)
        calibration, test = self._scenes(4000, generator), self._scenes(4000, generator)
        q = conformal.thresholds(*calibration, 0.1, conditional=conditional)
        has0, has1 = conformal.prediction_sets(test[0], q)
        covered = torch.where(test[1].bool(), has1, has0)
        return [float(covered[test[1] == v].float().mean()) for v in (0, 1)]

    def test_marginal_sets_can_leave_one_value_short_of_the_level(self):
        zero, one = self._coverage(conditional=False)
        assert zero < 0.85
        assert one > 0.95

    def test_conditional_sets_cover_each_value_at_the_level(self):
        assert all(c >= 0.88 for c in self._coverage(conditional=True))

    def test_a_value_no_calibration_scene_carries_gets_an_infinite_threshold(self):
        probs = torch.tensor([[0.9], [0.8], [0.95]])
        q = conformal.thresholds(probs, torch.ones((3, 1), dtype=torch.long), 0.1, conditional=True)
        assert q[0].tolist() == [math.inf]
