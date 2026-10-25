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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from regrounding import reproducible

reproducible.pin()

from regrounding.config import Experiment  # noqa: E402 - after the environment is pinned
from regrounding.execution import resolve  # noqa: E402
from regrounding.experiment import run  # noqa: E402
from regrounding.study import Row, Study, differences, dumps, loads  # noqa: E402
from regrounding.train import Mode  # noqa: E402

MODES: tuple[Mode, ...] = ("concepts", "verdict")
HEADER = (
    "| test | shots | round % | tone % | left % | matched % | verdict % | FP at sens. % "
    "| covered round % | covered tone % | covered left % | decided % | decided right % |"
)


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
        format=1,
        device="cuda" if device == "cuda" else "cpu",
        settings=config.as_record(),
        rows=rows,
    )


def _pct(values: list[float]) -> str:
    mean = sum(values) / len(values)
    return f"{100 * mean:.1f} ({100 * min(values):.0f}-{100 * max(values):.0f})"


def table(study: Study) -> str:
    """Render, per mode and shot count, the seed mean with the per-seed range."""
    lines: list[str] = []
    for mode in MODES:
        lines += [
            f"## {mode}",
            "",
            HEADER,
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        keys = sorted({(r["domain"] != "source", r["shots"]) for r in study["rows"]})
        for target, shots in keys:
            rows = [
                r
                for r in study["rows"]
                if r["mode"] == mode and (r["domain"] != "source") == target and r["shots"] == shots
            ]
            if not rows:
                continue
            cells = [_pct([r["concept_correct"][i] / r["scenes"] for r in rows]) for i in range(3)]
            cells += [
                _pct([r["permuted_correct"] / (3 * r["scenes"]) for r in rows]),
                _pct([r["verdict_correct"] / r["scenes"] for r in rows]),
                _pct([r["false_positives"] / max(1, r["negatives"]) for r in rows]),
            ]
            cells += [_pct([r["covered"][i] / r["scenes"] for r in rows]) for i in range(3)]
            cells += [
                _pct([r["decided"] / r["scenes"] for r in rows]),
                _pct([r["decided_correct"] / r["scenes"] for r in rows]),
            ]
            domain = "target" if target else "source"
            lines.append(f"| {domain} | {shots} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        msg = f"expected a count of at least 1, got {value}"
        raise argparse.ArgumentTypeError(msg)
    return value


def main(argv: list[str]) -> int:
    """Measure, then print, write or check."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--jobs", type=int, default=os.process_cpu_count() or 1)
    parser.add_argument("--seeds", type=_positive, help="the first N seeds only")
    out = parser.add_mutually_exclusive_group()
    out.add_argument("--json", type=Path, help="write the study file")
    out.add_argument("--check", type=Path, help="rerun a study file and compare")
    parser.add_argument("--tolerance", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.check and args.seeds is not None:
        parser.error("--check reruns the seeds the file records; --seeds does not apply")

    if args.check:
        expected = loads(args.check.read_text(encoding="utf-8"))
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
