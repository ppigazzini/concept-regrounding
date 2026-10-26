# Using the package

## Install

```bash
uv sync
```

Python 3.14 or later, as `requires-python` in `pyproject.toml` says. The runtime
dependencies are torch and numpy (the model and the scenes), and valgebra with
annotated-types and typing-extensions (the settings and the study file checks). PyPI's torch
build carries CUDA; the published studies do not use it.

## Run the study

```bash
uv run python tools/study.py
```

Trains one model per seed and supervision mode, concepts, verdict, mixed and direct, on
source scenes, re-grounds a copy of it on the first `k` labelled target scenes for every `k`
in the settings, and prints, per mode and `k`, the seed mean of each measure with the
per-seed range in brackets, then each seed's first `k` at 90 % and at 95 %. Every
(seed, mode) job runs in its own process with one thread, one job per core; `--jobs`
changes the count. `--seeds N` runs the first `N` seeds only.

The settings are the defaults of `Experiment` in `regrounding/config.py`. A setting outside
its declared range is refused when the object is built, with the field named:

```python
from regrounding.config import Experiment

Experiment(alpha=0.7)  # ValueError: Experiment: at alpha: expected < 0.5, got 0.7 [less_than]
```

## Write and check a study file

```bash
uv run python tools/study.py --json docs/studies/regrounding.json
uv run python tools/study.py --check docs/studies/regrounding.json
```

A study file holds the settings and one row of counts per model and test set; ratios are
computed where they are printed. `--check` reads the settings back from the file, reruns,
and prints one line per count that differs. On the machine that wrote the file every count
reproduces exactly. Whether another machine does is untested; `regrounding/reproducible.py`
and `regrounding/execution.py` remove the kernel choices that depend on the CPU, and
`docs/01-method.md` states each. `--check --seeds N` reruns the first `N` seeds of the file
and compares their rows, and `--tolerance`, a fraction of a row's scenes below 1, accepts a
count that moves by up to that many scenes.

```bash
uv run python tools/study.py --check docs/studies/regrounding.json --seeds 1 --native-kernels
```

reruns the first seed on ATen's widest kernels, AVX2 on the machine that wrote the study,
instead of its portable ones: 300 of the 704 fields of that seed's rows differ, the largest
by 804 of 1000 scenes. A kernel that rounds differently sends the training elsewhere, so a
tolerance does not stand in for equal arithmetic, and `--json` refuses the flag.

## Explore on a GPU

```bash
uv run python tools/study.py --device cuda --seeds 1
```

cuDNN determinism on and TF32 off; whether two runs on one GPU give equal counts is
untested, and they are not the CPU counts: a study file records its device, and `--check`
reruns on the device it records.

## Test

```bash
uv run pytest -m "not slow"   # scenes, rule, conformal sets, settings, study format
uv run pytest                 # adds two tiny end-to-end trainings
```
