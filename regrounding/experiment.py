"""One seed and one supervision mode: train on the source, re-ground on k target scenes, measure.

Every scene for a seed is drawn from one generator, in a fixed order -- source training,
source calibration, source test, target pool, target test, target training -- so every
supervision mode sees the same scenes, and the ``k`` scenes for re-grounding are the first
``k`` of the target pool, drawn in complementary pairs and nested across ``k``.

A mode names the supervision of the source training and of the target fine-tuning, in
`SUPERVISION`. ``mixed`` learns the concepts from their labels on the source and is
re-grounded from target verdicts through the fixed rule; ``direct`` is a network with one
output, the verdict, trained without concepts or rule, which `run_direct` measures as the
baseline at an equal number of target verdicts.

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
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from regrounding import conformal, rule, scenes
from regrounding.config import Experiment
from regrounding.execution import apply_runtime, seed_everything
from regrounding.model import ConceptNet
from regrounding.study import DirectRow, Row
from regrounding.train import Mode, fit, logits, reground

type Run = Literal["concepts", "verdict", "mixed"]
type Trained = Literal["source", "regrounded", "ceiling", "scratch"]

#: Per mode, the supervision of the source training and of every target fine-tuning.
SUPERVISION: dict[str, tuple[Mode, Mode]] = {
    "concepts": ("concepts", "concepts"),
    "verdict": ("verdict", "verdict"),
    "mixed": ("concepts", "verdict"),
    "direct": ("direct", "direct"),
}

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
    mode: Run,
    seed: int,
    shots: int,
    trained: Trained,
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


def measure_direct(
    scores: torch.Tensor,
    concepts: torch.Tensor,
    sensitivity: float,
    *,
    seed: int,
    shots: int,
    trained: Trained,
) -> DirectRow:
    """Return one row of verdict counts for a direct network's logits on one test set."""
    score = scores[:, 0]
    predicted = torch.sigmoid(score) >= _HALF
    truth = rule.verdict(concepts.bool())
    return DirectRow(
        seed=seed,
        shots=shots,
        domain="source" if trained == "source" else "target",
        trained=trained,
        scenes=int(concepts.shape[0]),
        verdict_correct=int((predicted == truth).sum()),
        true_positives=int((predicted & truth).sum()),
        positives=int(truth.sum()),
        negatives=int((~truth).sum()),
        false_positives=_false_positives_at(score, truth, sensitivity),
    )


@dataclass(frozen=True, slots=True)
class _Scenes:
    source: tuple[torch.Tensor, torch.Tensor]
    calibration: tuple[torch.Tensor, torch.Tensor]
    source_test: tuple[torch.Tensor, torch.Tensor]
    pool: tuple[torch.Tensor, torch.Tensor]
    target_test: tuple[torch.Tensor, torch.Tensor]
    target_train: tuple[torch.Tensor, torch.Tensor]


def _draw(config: Experiment, seed: int, device: torch.device) -> _Scenes:
    """Draw every scene of a seed from one generator, in the order the fields are listed."""
    rng = np.random.default_rng(seed)
    size = config.image_size

    def draw(style: scenes.Style, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        return _tensors(scenes.dataset(rng, style, n, size), device)

    source = draw(scenes.SOURCE, config.source_train)
    calibration = draw(scenes.SOURCE, config.calibration)
    source_test = draw(scenes.SOURCE, config.test)
    pool = _tensors(scenes.paired(rng, scenes.TARGET, config.pool, size), device)
    target_test = draw(scenes.TARGET, config.test)
    target_train = draw(scenes.TARGET, config.target_train)
    return _Scenes(source, calibration, source_test, pool, target_test, target_train)


def _models(
    config: Experiment, seed: int, mode: str, outputs: int, data: _Scenes, device: torch.device
) -> Iterator[tuple[ConceptNet, int, Trained]]:
    """Yield every model of one seed and mode with its shot count and how it was trained.

    One torch generator, seeded by the seed, draws every initial weight and every batch
    order in a fixed sequence: the source model, then the ceiling, then the model from
    scratch. Re-grounding takes the whole batch and draws nothing.
    """
    generator = torch.Generator().manual_seed(seed)
    source_mode, target_mode = SUPERVISION[mode]

    def train(m: ConceptNet, scenes_: tuple[torch.Tensor, torch.Tensor], how: Mode) -> ConceptNet:
        fit(
            m,
            *scenes_,
            mode=how,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            generator=generator,
        )
        return m

    size = config.image_size
    model = train(ConceptNet(generator, outputs, size).to(device), data.source, source_mode)
    yield model, 0, "source"
    for k in config.shots:
        adapted = copy.deepcopy(model)
        reground(
            adapted,
            data.pool[0][:k],
            data.pool[1][:k],
            mode=target_mode,
            steps=config.regrounding_steps,
            learning_rate=config.regrounding_rate,
            scope=config.regrounding_scope,
        )
        yield adapted, k, "regrounded"
    many = config.target_train
    yield train(copy.deepcopy(model), data.target_train, target_mode), many, "ceiling"
    scratch = ConceptNet(generator, outputs, size).to(device)
    yield train(scratch, data.target_train, target_mode), many, "scratch"


def run(config: Experiment, seed: int, mode: Run, device: torch.device) -> list[Row]:
    """Train, re-ground for every shot count, and return one row per model and test set."""
    apply_runtime(device)
    seed_everything(seed)
    data = _draw(config, seed, device)
    rows: list[Row] = []
    for model, k, trained in _models(config, seed, mode, 3, data, device):
        test = data.source_test if trained == "source" else data.target_test
        calibration = torch.sigmoid(logits(model, data.calibration[0]))
        threshold = conformal.thresholds(calibration, data.calibration[1], config.alpha)
        scores = logits(model, test[0])
        rows.append(
            measure(
                scores,
                test[1],
                threshold,
                config.sensitivity,
                mode=mode,
                seed=seed,
                shots=k,
                trained=trained,
            )
        )
    return rows


def run_direct(config: Experiment, seed: int, device: torch.device) -> list[DirectRow]:
    """Train and re-ground a network that outputs the verdict directly, on the same scenes."""
    apply_runtime(device)
    seed_everything(seed)
    data = _draw(config, seed, device)
    rows: list[DirectRow] = []
    for model, k, trained in _models(config, seed, "direct", 1, data, device):
        test = data.source_test if trained == "source" else data.target_test
        scores = logits(model, test[0])
        rows.append(
            measure_direct(scores, test[1], config.sensitivity, seed=seed, shots=k, trained=trained)
        )
    return rows
