# concept-regrounding

Transfers a concept abstraction to a new visual domain with the rule held fixed, and
measures how few labelled target examples re-grounding the concepts takes.

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

No study is published yet; [`docs/04-results.md`](docs/04-results.md) states what is open
and what would close it.

## Limits

Synthetic 32-pixel scenes, three binary concepts and one rule, a conjunction. The conformal
sets are per concept and calibrated on source scenes: their guarantee needs exchangeability,
which the shift to the target breaks, and the study measures the loss rather than
correcting it. [`docs/01-method.md`](docs/01-method.md) states each choice.

## Documentation

[`docs/README.md`](docs/README.md) indexes the pages.

## Citing

An open prototype released in October 2026; [`CITATION.cff`](CITATION.cff) has the entry.

## License

Apache-2.0.
