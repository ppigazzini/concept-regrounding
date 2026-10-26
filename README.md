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

Over 20 seeds of 1000 test scenes each (lowest to highest seed in brackets), with concept
supervision and the rule unchanged:

| concept | source | target, `k = 0` | target, `k = 64` | first `k` at 90 %, lower median |
|---|---:|---:|---:|---:|
| `round` | 99.9 % (99.4 to 100.0) | 54.4 % (48.2 to 73.9) | 65.8 % (54.3 to 84.6) | >64, in 0 of 20 seeds |
| `tone` | 100.0 % (100.0 to 100.0) | 1.1 % (0.0 to 18.9) | 99.4 % (97.3 to 100.0) | 16, in 20 of 20 seeds |
| `left` | 99.6 % (99.0 to 99.9) | 97.7 % (91.2 to 99.9) | 97.7 % (94.4 to 99.4) | 0, in 20 of 20 seeds |

`left` carries over; `tone` reads inverted and re-grounds, reaching 90 % within 4 to 64
target scenes, 16 for the lower median seed; `round` does not re-ground within 64, though
fine-tuning every weight on 3000 target scenes brings it to 96.0 % (73.4 to 99.8). The
verdict rises above the 87.5 % of a model that never says yes only from `k = 32`. Trained
from verdicts alone, the model gets 99.9 % (99.7 to 100.0) of the source verdicts right
through outputs that each equal the verdict on 98.5 % or more of the scenes, a reasoning
shortcut that 3000 target verdicts do not repair. `uv run python tools/study.py --check
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
