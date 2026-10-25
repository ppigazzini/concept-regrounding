"""One seed and one supervision mode: train on the source, re-ground on k target scenes, measure.

Every scene for a seed is drawn from one generator, in a fixed order -- source training,
source calibration, source test, target pool, target test -- so both supervision modes see
the same scenes, and the ``k`` scenes for re-grounding are the first ``k`` of the target
pool, nested across ``k``.

For each model the measures are counts on a test set: correct concepts, per concept and
under the best matching of outputs to concepts; correct verdicts; false positive
verdicts at the threshold that first reaches the configured sensitivity; how often the
conformal set holds the true value, per concept; and how many three-valued verdicts are
decided, and decided right.
"""

import copy
import itertools
from typing import Literal

import numpy as np
import torch

from regrounding import conformal, rule, scenes
from regrounding.config import Experiment
from regrounding.execution import apply_runtime, seed_everything
from regrounding.model import ConceptNet
from regrounding.study import Row
from regrounding.train import Mode, fit, probabilities, reground

#: A concept is read as present when its probability reaches one half.
_HALF = 0.5


def _tensors(
    data: tuple[np.ndarray, np.ndarray], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    images, concepts = data
    return torch.from_numpy(images).to(device), torch.from_numpy(concepts).to(device)


def _false_positives_at(scores: torch.Tensor, truth: torch.Tensor, sensitivity: float) -> int:
    """Count negatives scored at or above the highest threshold reaching ``sensitivity``."""
    positives = scores[truth]
    if positives.numel() == 0:
        return 0
    ordered = positives.sort(descending=True).values
    needed = max(1, int(np.ceil(sensitivity * positives.numel())))
    threshold = ordered[needed - 1]
    return int((scores[~truth] >= threshold).sum())


def measure(
    probs: torch.Tensor,
    concepts: torch.Tensor,
    threshold: torch.Tensor,
    sensitivity: float,
    *,
    mode: Mode,
    seed: int,
    shots: int,
    domain: Literal["source", "target"],
) -> Row:
    """Return one row of counts for one model's probabilities on one test set."""
    bits = probs >= _HALF
    truth = concepts.bool()
    matches = [[int((bits[:, i] == truth[:, j]).sum()) for j in range(3)] for i in range(3)]
    best = max(sum(matches[i][p[i]] for i in range(3)) for p in itertools.permutations(range(3)))
    verdict_truth = rule.verdict(truth)
    has0, has1 = conformal.prediction_sets(probs, threshold)
    covered = torch.where(truth, has1, has0).sum(dim=0)
    decided, value = rule.three_valued(has0, has1)
    return Row(
        mode=mode,
        seed=seed,
        shots=shots,
        domain=domain,
        scenes=int(concepts.shape[0]),
        concept_correct=[matches[i][i] for i in range(3)],
        permuted_correct=best,
        verdict_correct=int((rule.verdict(bits) == verdict_truth).sum()),
        positives=int(verdict_truth.sum()),
        negatives=int((~verdict_truth).sum()),
        false_positives=_false_positives_at(rule.score(probs), verdict_truth, sensitivity),
        covered=[int(c) for c in covered],
        decided=int(decided.sum()),
        decided_correct=int((decided & (value == verdict_truth)).sum()),
    )


def run(config: Experiment, seed: int, mode: Mode, device: torch.device) -> list[Row]:
    """Train, re-ground for every shot count, and return one row per model and test set."""
    apply_runtime(device)
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    size = config.image_size
    train = _tensors(scenes.dataset(rng, scenes.SOURCE, config.source_train, size), device)
    calib = _tensors(scenes.dataset(rng, scenes.SOURCE, config.calibration, size), device)
    source_test = _tensors(scenes.dataset(rng, scenes.SOURCE, config.test, size), device)
    pool = _tensors(scenes.dataset(rng, scenes.TARGET, config.pool, size), device)
    target_test = _tensors(scenes.dataset(rng, scenes.TARGET, config.test, size), device)

    generator = torch.Generator().manual_seed(seed)
    model = ConceptNet(generator, size=size).to(device)
    fit(
        model,
        *train,
        mode=mode,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        generator=generator,
    )
    rows: list[Row] = []
    threshold = conformal.thresholds(probabilities(model, calib[0]), calib[1], config.alpha)
    rows.append(
        measure(
            probabilities(model, source_test[0]),
            source_test[1],
            threshold,
            config.sensitivity,
            mode=mode,
            seed=seed,
            shots=0,
            domain="source",
        )
    )
    for k in config.shots:
        adapted = copy.deepcopy(model)
        reground(
            adapted,
            pool[0][:k],
            pool[1][:k],
            mode=mode,
            steps=config.regrounding_steps,
            learning_rate=config.regrounding_rate,
            scope=config.regrounding_scope,
        )
        k_threshold = conformal.thresholds(probabilities(adapted, calib[0]), calib[1], config.alpha)
        rows.append(
            measure(
                probabilities(adapted, target_test[0]),
                target_test[1],
                k_threshold,
                config.sensitivity,
                mode=mode,
                seed=seed,
                shots=k,
                domain="target",
            )
        )
    return rows
