# concept-regrounding

Transfers a concept abstraction to a new visual domain with the rule held fixed, and
measures how many labelled target examples re-grounding the concepts takes.

A scene carries three concepts — `round`, `tone`, `left` — and a rule over them, their
conjunction. A network learns the concepts on a source style; a target style draws the same
concepts differently: other colours, so a colour cue learned on the source reads inverted,
other sizes, and blur. The rule is never trained and never changed. Re-grounding fine-tunes
the dense head of the network by default, or its last layer, or every weight, on the first
`k` labelled target scenes, and the study counts, for each `k`, what the re-grounded
concepts get right.

Two supervision modes run side by side on the same scenes: concept labels, and verdicts
alone. A model trained from verdicts alone can get the verdict right through concepts that
mean something else, a reasoning shortcut, and the study counts how far its concepts are
from the named ones.

## Layout

```
regrounding/scenes.py        the two styles, the scenes, the concepts and the verdict
regrounding/rule.py          the fixed rule: Boolean, product score, three-valued
regrounding/model.py         the concept extractor
regrounding/train.py         training from concepts or verdicts; re-grounding on k scenes
regrounding/conformal.py     split conformal prediction sets, one per concept
regrounding/experiment.py    one seed and one mode, measured as counts
regrounding/config.py        the settings, checked against their declared ranges
regrounding/study.py         the study file format, checked with valgebra
regrounding/reproducible.py  the environment pinned before torch is imported
regrounding/execution.py     seeding and the deterministic runtime switches
regrounding/_valgebra.py     the one module that imports valgebra
tools/study.py               the committed command that produces published numbers
```

## Running

```bash
uv sync                                         # one lockfile; dev tools included
uv run python tools/study.py --seeds 1          # one seed, both modes: a printed table
uv run python tools/study.py --json docs/studies/regrounding.json
uv run python tools/study.py --check docs/studies/regrounding.json
uv run pytest -m "not slow"                     # seconds; without it, two tiny trainings
uv run ruff check && uv run ruff format --check && uv run ty check
uv run python tools/check_markdown.py
```

`--check` reruns a published study with the settings it records and fails on any count that
differs.

## Results

Over 5 seeds of 1000 test scenes each (lowest to highest seed in brackets), with concept
supervision and the rule unchanged:

| concept | source | target, `k = 0` | target, `k = 12` |
|---|---:|---:|---:|
| `round` | 99.8 % (99.6 to 100.0) | 55.4 % (49.5 to 60.5) | 58.1 % (53.5 to 61.8) |
| `tone` | 100.0 % (100.0 to 100.0) | 0.2 % (0.0 to 1.0) | 79.0 % (64.6 to 97.1) |
| `left` | 99.5 % (99.4 to 99.7) | 94.6 % (82.3 to 99.2) | 91.8 % (70.5 to 99.3) |

`left` carries over in 4 of 5 seeds; `tone` reads inverted and moves with `k`; `round`
stays near chance. The count is censored at 12: no seed reaches 90 % on `round` or on the
three concepts together within 12 target scenes, one seed of 5 does on `tone`, and the
verdict accuracy at `k = 12`, 88.7 % (84.4 to 94.0), is level with the 87.5 % of a model
that never says yes. Trained from verdicts alone, the model gets 99.9 % (99.6 to 100.0) of
the source verdicts right through outputs that agree with the named concepts on 63.9 %,
62.7 % and 63.7 % of the scenes and each equal the verdict itself on 98.2 % or more, a
reasoning shortcut. `uv run python tools/study.py --check
docs/studies/regrounding.json` reruns the study;
[`docs/04-results.md`](docs/04-results.md) has every count and what is open.

## Limits

Synthetic 32-pixel scenes, three binary concepts and one rule, a conjunction, which no
condition varies: under concept supervision the rule enters only the measures. The conformal
sets are per concept and calibrated on source scenes: their guarantee needs exchangeability,
which the shift to the target breaks, and no correction applies to styles that share no
scene. [`docs/01-method.md`](docs/01-method.md) states each choice.

## Documentation

[`docs/README.md`](docs/README.md) indexes the pages.

## Citing

An open prototype released in October 2026; [`CITATION.cff`](CITATION.cff) has the entry.

## License

Apache-2.0.
