"""One seed and one supervision mode: train on the source, re-ground on k target scenes, measure.

Every scene for a seed is drawn from one generator, in a fixed order -- source training,
source calibration, source test, target pool, target test, target training -- so both
supervision modes see the same scenes, and the ``k`` scenes for re-grounding are the first
``k`` of the target pool, drawn in complementary pairs and nested across ``k``.

Two rows bound what re-grounding can reach in the target style. ``ceiling`` fine-tunes
every weight of the source model on the target training scenes, and ``scratch`` trains a
new model on them, both with the source's own training settings, so the two differ only in
where they start.

For each model the measures are counts on a test set: how often each output equals each
concept and the verdict; correct verdicts and true positives; false positive verdicts at
the threshold that first reaches the configured sensitivity; how often the conformal set
holds the true value, is empty and is full, per concept; and how many three-valued verdicts
are decided, and decided right.
"""

import copy
from typing import Literal

import numpy as np
import torch

from regrounding import conformal, rule, scenes
from regrounding.config import Experiment
from regrounding.execution import apply_runtime, seed_everything
from regrounding.model import ConceptNet
from regrounding.study import Row
from regrounding.train import Mode, fit, logits, reground

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
    scores: torch.Tensor,
    concepts: torch.Tensor,
    threshold: torch.Tensor,
    sensitivity: float,
    *,
    mode: Mode,
    seed: int,
    shots: int,
    trained: Literal["source", "regrounded", "ceiling", "scratch"],
) -> Row:
    """Return one row of counts for one model's concept logits on one test set.

    The source model is measured on source scenes and every other model on target scenes.
    """
    probs = torch.sigmoid(scores)
    bits = probs >= _HALF
    truth = concepts.bool()
    verdict_truth = rule.verdict(truth)
    predicted = rule.verdict(bits)
    has0, has1 = conformal.prediction_sets(probs, threshold)
    decided, value = rule.three_valued(has0, has1)
    return Row(
        mode=mode,
        seed=seed,
        shots=shots,
        domain="source" if trained == "source" else "target",
        trained=trained,
        scenes=int(concepts.shape[0]),
        agreement=[[int((bits[:, i] == truth[:, j]).sum()) for j in range(3)] for i in range(3)],
        verdict_agreement=[int((bits[:, i] == verdict_truth).sum()) for i in range(3)],
        verdict_correct=int((predicted == verdict_truth).sum()),
        true_positives=int((predicted & verdict_truth).sum()),
        positives=int(verdict_truth.sum()),
        negatives=int((~verdict_truth).sum()),
        false_positives=_false_positives_at(rule.log_score(scores), verdict_truth, sensitivity),
        covered=[int(c) for c in torch.where(truth, has1, has0).sum(dim=0)],
        empty=[int(c) for c in (~has0 & ~has1).sum(dim=0)],
        full=[int(c) for c in (has0 & has1).sum(dim=0)],
        decided=int(decided.sum()),
        decided_correct=int((decided & (value == verdict_truth)).sum()),
    )


def run(config: Experiment, seed: int, mode: Mode, device: torch.device) -> list[Row]:
    """Train, re-ground for every shot count, and return one row per model and test set."""
    apply_runtime(device)
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    size = config.image_size
    source = _tensors(scenes.dataset(rng, scenes.SOURCE, config.source_train, size), device)
    calib = _tensors(scenes.dataset(rng, scenes.SOURCE, config.calibration, size), device)
    source_test = _tensors(scenes.dataset(rng, scenes.SOURCE, config.test, size), device)
    pool = _tensors(scenes.paired(rng, scenes.TARGET, config.pool, size), device)
    target_test = _tensors(scenes.dataset(rng, scenes.TARGET, config.test, size), device)
    target_train = _tensors(scenes.dataset(rng, scenes.TARGET, config.target_train, size), device)

    generator = torch.Generator().manual_seed(seed)

    def train(m: ConceptNet, data: tuple[torch.Tensor, torch.Tensor]) -> ConceptNet:
        fit(
            m,
            *data,
            mode=mode,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            generator=generator,
        )
        return m

    def evaluate(
        m: ConceptNet,
        test: tuple[torch.Tensor, torch.Tensor],
        k: int,
        trained: Literal["source", "regrounded", "ceiling", "scratch"],
    ) -> Row:
        threshold = conformal.thresholds(torch.sigmoid(logits(m, calib[0])), calib[1], config.alpha)
        return measure(
            logits(m, test[0]),
            test[1],
            threshold,
            config.sensitivity,
            mode=mode,
            seed=seed,
            shots=k,
            trained=trained,
        )

    model = train(ConceptNet(generator, size=size).to(device), source)
    rows = [evaluate(model, source_test, 0, "source")]
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
        rows.append(evaluate(adapted, target_test, k, "regrounded"))
    many = config.target_train
    rows.append(evaluate(train(copy.deepcopy(model), target_train), target_test, many, "ceiling"))
    scratch = train(ConceptNet(generator, size=size).to(device), target_train)
    rows.append(evaluate(scratch, target_test, many, "scratch"))
    return rows
