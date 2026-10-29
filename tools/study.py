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
import math
import os
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Literal, cast

from regrounding import reproducible

#: Read before torch is imported, here and in every worker, which is handed these arguments.
NATIVE_KERNELS = "--native-kernels" in sys.argv[1:]
reproducible.pin(native_kernels=NATIVE_KERNELS)

from regrounding.config import Experiment  # noqa: E402 - after the environment is pinned
from regrounding.execution import resolve  # noqa: E402
from regrounding.experiment import Run, run, run_direct  # noqa: E402
from regrounding.study import (  # noqa: E402
    DirectRow,
    Row,
    Study,
    differences,
    dumps,
    first_shots,
    identity,
    largest_move,
    loads,
    matched,
    restrict,
    signed,
)

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
VERDICTS: tuple[tuple[str, Callable[[DirectRow], float]], ...] = (
    ("verdict %", lambda r: r["verdict_correct"] / r["scenes"]),
    ("always no %", lambda r: r["negatives"] / r["scenes"]),
    ("true positives %", lambda r: r["true_positives"] / max(1, r["positives"])),
    ("FP at sens. %", lambda r: r["false_positives"] / max(1, r["negatives"])),
)
#: The shares of the target test scenes at which the first `k` of each seed is printed.
SHARES = (0.9, 0.95)


def job(task: tuple[Experiment, int, Run | None, str]) -> list[Row] | list[DirectRow]:
    """Run one seed and one mode, `None` for the direct network; a worker can run it."""
    config, seed, mode, device = task
    if mode is None:
        return run_direct(config, seed, resolve(device))
    return run(config, seed, mode, resolve(device))


def measure(config: Experiment, device: str, jobs: int) -> Study:
    """Run every job of the settings' modes and return the study, rows in a fixed order."""
    concept_modes: list[Run] = [m for m in config.modes if m != "direct"]
    modes: list[Run | None] = [*concept_modes, *([None] if "direct" in config.modes else [])]
    tasks = [(config, seed, mode, device) for mode in modes for seed in config.seeds]
    if jobs > 1 and device == "cpu":
        with ProcessPoolExecutor(max_workers=jobs, initializer=reproducible.hide_gpus) as pool:
            done = list(pool.map(job, tasks))
    else:
        done = [job(t) for t in tasks]
    split = len(concept_modes) * len(config.seeds)
    return Study(
        format=2,
        device="cuda" if device == "cuda" else "cpu",
        settings=config.as_record(),
        rows=[cast("Row", row) for rows in done[:split] for row in rows],
        direct=[cast("DirectRow", row) for rows in done[split:] for row in rows],
    )


def _pct(values: list[float]) -> str:
    mean = sum(values) / len(values)
    return f"{100 * mean:.1f} ({100 * min(values):.1f} to {100 * max(values):.1f})"


def _median(ks: list[int | None], last: int) -> str:
    """Return the lower median of the seeds' first `k`, an actual shot count or censored."""
    ordered = sorted(ks, key=lambda k: math.inf if k is None else k)
    k = ordered[(len(ordered) - 1) // 2]
    return f">{last}" if k is None else str(k)


def _first(study: Study, mode: str) -> list[str]:
    """Render the first `k` at every share in `SHARES`, censored past the last `k`.

    With one draw per seed the entries are the seeds'; with more, an entry is the lowest
    to highest first `k` over a seed's draws, and the counts run over every draw.
    """
    last = max(study["settings"]["shots"])
    seeds = len(study["settings"]["seeds"])
    draws = study["settings"]["draws"]
    entries = "per seed" if draws == 1 else f"per seed, lowest..highest of {draws} draws"
    head = ["| target concept |"]
    rule = ["|---|"]
    for share in SHARES:
        head.append(f" first `k` at {100 * share:.0f} %, {entries} | reaching it | lower median |")
        rule.append("---|---:|---:|")
    per_share = [first_shots(study, mode, share) for share in SHARES]
    lines = ["".join(head), "".join(rule)]

    def shown(k: int | None) -> str:
        return f">{last}" if k is None else str(k)

    for name in per_share[0]:
        cells = []
        for first in per_share:
            ks = first[name]
            by_seed = [ks[i : i + draws] for i in range(0, len(ks), draws)]
            if draws == 1:
                cells.append(" ".join(shown(k) for (k,) in by_seed))
            else:
                ordered = [
                    sorted(group, key=lambda k: math.inf if k is None else k) for group in by_seed
                ]
                spans = (
                    shown(g[0]) if g[0] == g[-1] else f"{shown(g[0])}..{shown(g[-1])}"
                    for g in ordered
                )
                cells.append(" ".join(spans))
            cells.append(f"{sum(k is not None for k in ks)} of {seeds * draws}")
            cells.append(_median(ks, last))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


#: How each kind of row is ordered and labeled in a table.
_ORDER = {"source": 0, "regrounded": 1, "ceiling": 2, "scratch": 3}
_LABELS = {
    "source": "source",
    "regrounded": "target",
    "ceiling": "target, every weight",
    "scratch": "target, from scratch",
}


def _table(rows: Sequence[Any], columns: Sequence[tuple[str, Callable[[Any], float]]]) -> list[str]:
    """Render one table: per kind of row and shot count, each column's seed mean and range."""
    lines = [
        "| test | shots | " + " | ".join(name for name, _ in columns) + " |",
        "|---|---:|" + "---:|" * len(columns),
    ]
    keys = sorted({(r["trained"], r["shots"]) for r in rows}, key=lambda k: (_ORDER[k[0]], k[1]))
    for trained, shots in keys:
        group = [r for r in rows if r["trained"] == trained and r["shots"] == shots]
        cells = [_pct([read(r) for r in group]) for _, read in columns]
        lines.append(f"| {_LABELS[trained]} | {shots} | " + " | ".join(cells) + " |")
    return lines


def table(study: Study) -> str:
    """Render, per mode, the concepts and verdicts, the first `k`, and the conformal sets.

    "always no" is the share of negatives, the verdict accuracy of a model that never says
    yes, against which the verdict column reads; "signed" matches outputs to concepts and
    may flip one; "round = verdict" is how often the `round` output equals the verdict.
    """
    lines: list[str] = []
    for mode in study["settings"]["modes"]:
        if mode == "direct":
            lines += ["## direct", "", *_table(study["direct"], VERDICTS), ""]
            continue
        rows = [r for r in study["rows"] if r["mode"] == mode]
        lines += [f"## {mode}", "", *_table(rows, OUTPUTS), ""]
        lines += [*_first(study, mode), "", *_table(rows, SETS), ""]
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
    parser.add_argument("--target", choices=["target", "palette", "size", "blur"])
    parser.add_argument("--modes", nargs="+", choices=["concepts", "verdict", "mixed", "direct"])
    parser.add_argument("--scope", choices=["first", "readout", "head", "all"])
    parser.add_argument("--fit", choices=["steps", "converged"])
    parser.add_argument("--rehearsal", type=_positive, help="source scenes in every batch")
    parser.add_argument("--calibrate-on", choices=["source", "target"])
    parser.add_argument("--conditional", action="store_true", help="one threshold per value")
    parser.add_argument("--draws", type=_positive, help="draws of the target pool per seed")
    parser.add_argument(
        "--native-kernels",
        action="store_true",
        help="ATen's widest kernels, to measure with --check how far a kernel moves counts",
    )
    args = parser.parse_args(argv)
    if args.native_kernels and not args.check:
        parser.error("--native-kernels measures against --check; a study is written portably")
    chosen: dict[str, Any] = {
        "target": args.target,
        "modes": tuple(args.modes) if args.modes else None,
        "regrounding_scope": args.scope,
        "regrounding_fit": args.fit,
        "rehearsal": args.rehearsal,
        "calibrate_on": args.calibrate_on,
        "conditional": args.conditional or None,
        "draws": args.draws,
    }
    chosen = {
        key: value for key, value in chosen.items() if value is not None
    }  # checked by Experiment
    if args.check and chosen:
        parser.error("--check reruns the settings the file records")

    if args.check:
        expected = loads(args.check.read_text(encoding="utf-8"))
        if args.seeds is not None:
            expected = restrict(expected, args.seeds)
        settings = dict(expected["settings"])
        for key in ("shots", "seeds", "modes"):
            settings[key] = tuple(settings[key])
        actual = measure(Experiment(**settings), expected["device"], args.jobs)
        print(table(actual))
        diff = differences(expected, actual, args.tolerance)
        for line in diff:
            print(line, file=sys.stderr)
        fields = sum(len(r) for r in (*expected["rows"], *expected["direct"]))
        print(
            f"{len(diff)} difference(s) in {fields} fields against {args.check}; "
            f"the largest move is {largest_move(expected, actual)} scenes"
        )
        return 1 if diff else 0

    config = Experiment(**chosen)
    if args.seeds is not None:
        config = config.updated(seeds=config.seeds[: args.seeds])
    study = measure(config, args.device, args.jobs)
    print(table(study))
    if args.json:
        args.json.write_text(dumps(study), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
