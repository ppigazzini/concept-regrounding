"""Run the experiment for every seed and supervision mode; print, write or check the study.

    uv run python tools/study.py                      # every seed, both modes, all cores
    uv run python tools/study.py --json docs/studies/regrounding.json
    uv run python tools/study.py --check docs/studies/regrounding.json
    uv run python tools/study.py --device cuda        # explore on a GPU; not for a study

Each (seed, mode) job trains one model and re-grounds it for every shot count, in its own
worker process with one thread, under the reproducible profile; the jobs run side by side,
one per core. ``--check`` reruns with the settings the file records; ``--tolerance`` is the
fraction of a row's scenes a count may move by on a machine other than the one that
produced the study.
"""

import argparse
import os
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal

from regrounding import reproducible

reproducible.pin()

from regrounding.config import Experiment  # noqa: E402 - after the environment is pinned
from regrounding.execution import resolve  # noqa: E402
from regrounding.experiment import run  # noqa: E402
from regrounding.study import (  # noqa: E402
    Row,
    Study,
    differences,
    dumps,
    first_shots,
    identity,
    loads,
    matched,
    restrict,
    signed,
)
from regrounding.train import Mode  # noqa: E402

MODES: tuple[Mode, ...] = ("concepts", "verdict")
#: Per row, what each column reads off the counts, as a share of its denominator.
OUTPUTS: tuple[tuple[str, Callable[[Row], float]], ...] = (
    ("round %", lambda r: identity(r)[0] / r["scenes"]),
    ("tone %", lambda r: identity(r)[1] / r["scenes"]),
    ("left %", lambda r: identity(r)[2] / r["scenes"]),
    ("matched %", lambda r: matched(r) / (3 * r["scenes"])),
    ("signed %", lambda r: signed(r) / (3 * r["scenes"])),
    ("round = verdict %", lambda r: r["verdict_agreement"][0] / r["scenes"]),
    ("tone = verdict %", lambda r: r["verdict_agreement"][1] / r["scenes"]),
    ("left = verdict %", lambda r: r["verdict_agreement"][2] / r["scenes"]),
    ("verdict %", lambda r: r["verdict_correct"] / r["scenes"]),
    ("always no %", lambda r: r["negatives"] / r["scenes"]),
    ("true positives %", lambda r: r["true_positives"] / max(1, r["positives"])),
    ("FP at sens. %", lambda r: r["false_positives"] / max(1, r["negatives"])),
)


def _per_concept(key: Literal["covered", "empty", "full"], i: int) -> Callable[[Row], float]:
    return lambda r: r[key][i] / r["scenes"]


SETS: tuple[tuple[str, Callable[[Row], float]], ...] = (
    *(
        (f"{key} {name} %", _per_concept(key, i))
        for key in ("covered", "empty", "full")
        for i, name in enumerate(("round", "tone", "left"))
    ),
    ("decided %", lambda r: r["decided"] / r["scenes"]),
    ("decided right %", lambda r: r["decided_correct"] / r["scenes"]),
)
#: The shares of the target test scenes at which the first `k` of each seed is printed.
SHARES = (0.9, 0.95)


def job(task: tuple[Experiment, int, Mode, str]) -> list[Row]:
    """Run one seed and one mode; a module-level function so a worker can run it."""
    config, seed, mode, device = task
    return run(config, seed, mode, resolve(device))


def measure(config: Experiment, device: str, jobs: int) -> Study:
    """Run every job and return the study, rows in a fixed order."""
    tasks = [(config, seed, mode, device) for mode in MODES for seed in config.seeds]
    if jobs > 1 and device == "cpu":
        with ProcessPoolExecutor(max_workers=jobs, initializer=reproducible.hide_gpus) as pool:
            done = list(pool.map(job, tasks))
    else:
        done = [job(t) for t in tasks]
    rows = [row for rows in done for row in rows]
    return Study(
        format=2,
        device="cuda" if device == "cuda" else "cpu",
        settings=config.as_record(),
        rows=rows,
    )


def _pct(values: list[float]) -> str:
    mean = sum(values) / len(values)
    return f"{100 * mean:.1f} ({100 * min(values):.1f} to {100 * max(values):.1f})"


def _first(study: Study, mode: str) -> list[str]:
    """Render each seed's first `k` at every share in `SHARES`, censored past the last `k`."""
    last = max(study["settings"]["shots"])
    seeds = len(study["settings"]["seeds"])
    head = ["| target concept |"]
    rule = ["|---|"]
    for share in SHARES:
        head.append(f" first `k` at {100 * share:.0f} %, per seed | seeds reaching it |")
        rule.append("---|---:|")
    per_share = [first_shots(study, mode, share) for share in SHARES]
    lines = ["".join(head), "".join(rule)]
    for name in per_share[0]:
        cells = []
        for first in per_share:
            ks = first[name]
            cells.append(" ".join(f">{last}" if k is None else str(k) for k in ks))
            cells.append(f"{sum(k is not None for k in ks)} of {seeds}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def _table(
    study: Study, mode: str, columns: tuple[tuple[str, Callable[[Row], float]], ...]
) -> list[str]:
    """Render one table: per test set and shot count, each column's seed mean and range."""
    lines = [
        "| test | shots | " + " | ".join(name for name, _ in columns) + " |",
        "|---|---:|" + "---:|" * len(columns),
    ]
    for target, shots in sorted({(r["domain"] != "source", r["shots"]) for r in study["rows"]}):
        rows = [
            r
            for r in study["rows"]
            if r["mode"] == mode and (r["domain"] != "source") == target and r["shots"] == shots
        ]
        if rows:
            cells = [_pct([read(r) for r in rows]) for _, read in columns]
            domain = "target" if target else "source"
            lines.append(f"| {domain} | {shots} | " + " | ".join(cells) + " |")
    return lines


def table(study: Study) -> str:
    """Render, per mode, the concepts and verdicts, the first `k`, and the conformal sets.

    "always no" is the share of negatives, the verdict accuracy of a model that never says
    yes, against which the verdict column reads; "signed" matches outputs to concepts and
    may flip one; "round = verdict" is how often the `round` output equals the verdict.
    """
    lines: list[str] = []
    for mode in MODES:
        lines += [f"## {mode}", "", *_table(study, mode, OUTPUTS), ""]
        lines += [*_first(study, mode), "", *_table(study, mode, SETS), ""]
    return "\n".join(lines)


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        msg = f"expected a count of at least 1, got {value}"
        raise argparse.ArgumentTypeError(msg)
    return value


def _fraction(text: str) -> float:
    value = float(text)
    if not 0.0 <= value < 1.0:
        msg = f"expected a fraction of a row's scenes in [0, 1), got {value}"
        raise argparse.ArgumentTypeError(msg)
    return value


def main(argv: list[str]) -> int:
    """Measure, then print, write or check."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--jobs", type=int, default=os.process_cpu_count() or 1)
    parser.add_argument("--seeds", type=_positive, help="the first N seeds only, also of a check")
    out = parser.add_mutually_exclusive_group()
    out.add_argument("--json", type=Path, help="write the study file")
    out.add_argument("--check", type=Path, help="rerun a study file and compare")
    parser.add_argument("--tolerance", type=_fraction, default=0.0)
    args = parser.parse_args(argv)

    if args.check:
        expected = loads(args.check.read_text(encoding="utf-8"))
        if args.seeds is not None:
            expected = restrict(expected, args.seeds)
        settings = dict(expected["settings"])
        settings["shots"], settings["seeds"] = tuple(settings["shots"]), tuple(settings["seeds"])
        actual = measure(Experiment(**settings), expected["device"], args.jobs)
        print(table(actual))
        diff = differences(expected, actual, args.tolerance)
        for line in diff:
            print(line, file=sys.stderr)
        print(f"{len(diff)} difference(s) against {args.check}")
        return 1 if diff else 0

    config = Experiment()
    if args.seeds is not None:
        config = config.updated(seeds=config.seeds[: args.seeds])
    study = measure(config, args.device, args.jobs)
    print(table(study))
    if args.json:
        args.json.write_text(dumps(study), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
