# concept-regrounding

Transfers a concept abstraction to a new visual domain with the rule held fixed, and
measures how many labeled target examples re-grounding the concepts takes.

A scene carries three concepts — `round`, `tone`, `left` — and a rule over them, their
conjunction. A network learns the concepts on a source style; a target style draws the same
concepts differently: other colors, so a color cue learned on the source reads inverted,
other sizes, and blur. The rule is never trained and never changed. Re-grounding fine-tunes
the dense head of the network by default, or its last layer, or every weight, on the first
`k` labeled target scenes, and the study counts, for each `k`, what the re-grounded
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
tools/vector_math.py         what MKL's vector math, which training avoids, gives on this CPU
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

Over 20 seeds of 1000 test scenes each (lowest to highest seed in parentheses), with concept
supervision and the rule unchanged:

| concept | source | target, `k = 0` | target, `k = 64` | first `k` at 90 %, lower median |
|---|---:|---:|---:|---:|
| `round` | 99.9 % (99.4 to 100.0) | 52.8 % (48.2 to 75.7) | 64.1 % (52.4 to 80.9) | >64, in 0 of 20 seeds |
| `tone` | 100.0 % (100.0 to 100.0) | 0.2 % (0.0 to 3.0) | 99.4 % (97.2 to 100.0) | 16, in 20 of 20 seeds |
| `left` | 99.6 % (98.9 to 99.9) | 97.5 % (92.2 to 99.8) | 97.6 % (93.9 to 99.5) | 0, in 20 of 20 seeds |

`left` carries over; `tone` reads inverted and re-grounds, reaching 90 % within 8 to 64
target scenes, 16 for the lower median seed; `round` does not re-ground within 64, though
fine-tuning every weight on 3000 target scenes brings it to 95.9 % (71.0 to 99.9). The
verdict rises above the 87.5 % of a model that never says yes only from `k = 32`. Trained
from verdicts alone, the model gets 99.8 % (99.5 to 100.0) of the source verdicts right
through outputs that each equal the verdict on 97.9 % or more of the scenes, a reasoning
shortcut that 3000 target verdicts do not repair. Target verdicts passed through the fixed
rule do not re-ground the concepts either, but they rank target scenes with 12.5 % (6.2 to
25.0) false positives at 90 % sensitivity at `k = 64`, against 58.8 % (18.9 to 88.8) for a
network that learns the verdict without concepts or rule. The palette alone sets the
count for `tone`, and the blur alone stops `round`. Fitted to convergence instead of moved
by 150 steps, the readout alone re-grounds `tone` with a lower median of 8 scenes.
`uv run python tools/study.py --check docs/studies/regrounding.json` reruns the study;
[`docs/04-results.md`](docs/04-results.md) has every count, the ten studies and what is
open.

## Limits

Synthetic 32-pixel scenes, three binary concepts and one rule, a conjunction, which no
condition varies: under concept supervision the rule enters only the measures. The conformal
sets are per concept and calibrated on source scenes: their guarantee needs exchangeability,
which the shift to the target breaks, and no correction applies to styles that share no
scene. [`docs/01-method.md`](docs/01-method.md) states each choice.

## Documentation

[`docs/README.md`](docs/README.md) indexes the pages.

## Citing

[doi:10.5281/zenodo.23686213](https://doi.org/10.5281/zenodo.23686213) cites every
version; version 0.0.1 is
[doi:10.5281/zenodo.23686214](https://doi.org/10.5281/zenodo.23686214).
[`CITATION.cff`](CITATION.cff) holds the record.

## License

Apache-2.0.
