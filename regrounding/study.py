"""The study file: the counts one run of the experiment publishes, and the check of them.

Written by ``tools/study.py --json`` and read back by ``tools/study.py --check``, which
reruns the experiment and compares. Its shape is declared once, here, as closed records
checked by valgebra: a key the format does not name is refused, every count is a
non-negative integer that is not a ``bool``, and every failure is reported at once with
its path. The rules between fields of a row are plain Python in `check_rows`.

A row holds counts and nothing derived from them. A `Row` is a model with three concept
outputs and the rule over them; a `DirectRow` is a network that outputs the verdict alone,
and holds the verdict's counts only, since it has no concepts to count. How well the
outputs carry the concepts is the 3 by 3 count of scenes where output ``i`` equals
concept ``j``; the identity, the best permutation and the best signed permutation are
read off it by `identity`, `matched` and `signed` where they are printed.
"""

import itertools
import json
from typing import Annotated, Any, Literal

from annotated_types import Ge, Len
from typing_extensions import TypedDict

from regrounding._valgebra import NOT_BOOL, ValidationError, Validator

type Count = Annotated[int, NOT_BOOL, Ge(0)]
type Triple = Annotated[list[Count], Len(3, 3)]
type Matrix = Annotated[list[Triple], Len(3, 3)]


class Row(TypedDict, closed=True):
    """One model on one test set: counts only."""

    mode: Literal["concepts", "verdict", "mixed"]
    seed: Count
    shots: Count
    domain: Literal["source", "target"]
    trained: Literal["source", "regrounded", "ceiling", "scratch"]
    scenes: Count
    agreement: Matrix
    verdict_agreement: Triple
    verdict_correct: Count
    true_positives: Count
    positives: Count
    negatives: Count
    false_positives: Count
    covered: Triple
    empty: Triple
    full: Triple
    decided: Count
    decided_correct: Count


class DirectRow(TypedDict, closed=True):
    """One network that outputs the verdict directly, on one test set: counts only."""

    seed: Count
    shots: Count
    domain: Literal["source", "target"]
    trained: Literal["source", "regrounded", "ceiling", "scratch"]
    scenes: Count
    verdict_correct: Count
    true_positives: Count
    positives: Count
    negatives: Count
    false_positives: Count


class Study(TypedDict, closed=True):
    """The settings and every row of one run."""

    format: Literal[2]
    device: Literal["cpu", "cuda"]
    settings: dict[str, Any]
    rows: list[Row]
    direct: list[DirectRow]


class StudyError(ValueError):
    """A study file that is malformed or contradicts itself."""


STUDY = Validator(Study)

#: The fields that say which model and which scenes a row is about. The scenes come from
#: NumPy's generator and not from torch's floating point, so these compare exactly at any
#: tolerance: a tolerance on them would accept a row of the wrong seed or shot count.
IDENTITY = frozenset(
    {"mode", "seed", "shots", "domain", "trained", "scenes", "positives", "negatives"}
)

#: The 6 ways to assign the three outputs to the three concepts.
_PERMUTATIONS = tuple(itertools.permutations(range(3)))


def identity(row: Row) -> list[int]:
    """Return, per concept, the scenes where the output named after it equals it."""
    return [row["agreement"][i][i] for i in range(3)]


def matched(row: Row) -> int:
    """Return the scenes right summed over the concepts, under the best of the 6 matchings."""
    a = row["agreement"]
    return max(sum(a[i][p[i]] for i in range(3)) for p in _PERMUTATIONS)


def signed(row: Row) -> int:
    """Return the same under the best of the 48 matchings that may also flip an output.

    A flipped output is right where it is wrong, so it scores the scenes less its
    agreement; an inverted concept, as a colour cue reads after the palette shift, is
    credited here and not by `matched`.
    """
    a, n = row["agreement"], row["scenes"]
    return max(sum(max(a[i][p[i]], n - a[i][p[i]]) for i in range(3)) for p in _PERMUTATIONS)


def _verdict_rules(r: Row | DirectRow) -> list[str]:
    """Return every rule between the verdict's counts that a row breaks, row or direct."""
    out: list[str] = []
    n = r["scenes"]
    if r["verdict_correct"] > n:
        out.append(f"a count exceeds the {n} scenes")
    if r["positives"] + r["negatives"] != n:
        out.append("positives + negatives != scenes")
    if (
        r["false_positives"] > r["negatives"]
        or r["true_positives"] > min(r["positives"], r["verdict_correct"])
        or r["verdict_correct"] - r["true_positives"] > r["negatives"]
    ):
        out.append("a part exceeds its whole")
    return out


def check_rows(study: Study) -> list[str]:
    """Return every rule between two fields that a row breaks."""
    out: list[str] = []
    for i, r in enumerate(study["rows"]):
        n = r["scenes"]
        broken = _verdict_rules(r)
        bounded = [
            *(c for line in r["agreement"] for c in line),
            *r["verdict_agreement"],
            *r["covered"],
            r["decided"],
        ]
        if any(c > n for c in bounded):
            broken.append(f"a count exceeds the {n} scenes")
        if r["decided_correct"] > r["decided"]:
            broken.append("a part exceeds its whole")
        if any(
            full > covered or covered + empty > n
            for full, covered, empty in zip(r["full"], r["covered"], r["empty"], strict=True)
        ):
            broken.append("a set is both empty and covering, or full and not covering")
        out += [f"rows[{i}]: {line}" for line in dict.fromkeys(broken)]
    for i, r in enumerate(study["direct"]):
        out += [f"direct[{i}]: {line}" for line in _verdict_rules(r)]
    return out


def first_shots(study: Study, mode: str, share: float) -> dict[str, list[int | None]]:
    """Return, per concept and for the three together, each seed's smallest target ``k``.

    A seed's entry is the smallest shot count at which the concept is right on at least
    ``share`` of the target test scenes, or `None` when no shot count of the study reaches
    it: the count is censored at the largest ``k``, and `None` says so rather than
    standing in for a number. Seeds are in the order of the study's settings.
    """
    names = ("round", "tone", "left")
    out: dict[str, list[int | None]] = {name: [] for name in (*names, "all three")}
    for seed in study["settings"]["seeds"]:
        rows = sorted(
            (r for r in study["rows"] if r["mode"] == mode and r["seed"] == seed),
            key=lambda r: r["shots"],
        )
        reached = [
            (r["shots"], [c >= share * r["scenes"] for c in identity(r)])
            for r in rows
            if r["trained"] == "regrounded"
        ]
        for i, name in enumerate(names):
            out[name].append(next((k for k, ok in reached if ok[i]), None))
        out["all three"].append(next((k for k, ok in reached if all(ok)), None))
    return out


def restrict(study: Study, seeds: int) -> Study:
    """Return the study's first ``seeds`` seeds: their settings and their rows.

    A rerun of the first seeds of a study is compared against this, so a check can rerun
    a slice of a study that is too long to rerun whole.
    """
    kept = study["settings"]["seeds"][:seeds]
    return Study(
        format=study["format"],
        device=study["device"],
        settings=study["settings"] | {"seeds": kept},
        rows=[r for r in study["rows"] if r["seed"] in kept],
        direct=[r for r in study["direct"] if r["seed"] in kept],
    )


def loads(text: str) -> Study:
    """Parse and check a study, reporting every failure rather than the first."""
    try:
        study = STUDY.load(text)
    except ValidationError as exc:
        raise StudyError(str(exc)) from exc
    broken = check_rows(study)
    if broken:
        raise StudyError("; ".join(broken))
    return study


def dumps(study: Study) -> str:
    """Serialise a study, refusing one this code should never have built."""
    try:
        STUDY.validate(study)
    except ValidationError as exc:
        raise StudyError(str(exc)) from exc
    broken = check_rows(study)
    if broken:
        raise StudyError("; ".join(broken))
    return json.dumps(study, indent=1) + "\n"


def _counts(value: object) -> list[int]:
    """Return a field's counts, flattened: every field outside `IDENTITY` holds counts."""
    if isinstance(value, list):
        return [c for item in value for c in _counts(item)]
    return [value] if isinstance(value, int) else []


def largest_move(expected: Study, actual: Study) -> int:
    """Return the most scenes by which any count of a row moves between two runs."""
    moves = [0]
    for part in ("rows", "direct"):
        for old, new in zip(expected[part], actual[part], strict=False):
            for key, value in old.items():
                if key not in IDENTITY:
                    x, y = _counts(value), _counts(new.get(key))
                    moves += [abs(p - q) for p, q in zip(x, y, strict=False)]
    return max(moves)


def differences(expected: Study, actual: Study, tolerance: float = 0.0) -> list[str]:
    """Return one line per setting or count that differs by more than the tolerance.

    ``tolerance`` is a fraction of a row's scenes, and it never applies to the fields in
    `IDENTITY`. A run on the machine that produced the study reproduces every count
    exactly.
    """
    out = (
        [f"settings: {expected['settings']} != {actual['settings']}"]
        if (expected["settings"] != actual["settings"])
        else []
    )
    for part in ("rows", "direct"):
        olds, news = expected[part], actual[part]
        if len(olds) != len(news):
            out.append(f"{part}: {len(olds)} != {len(news)}")
            continue
        for i, (old, new) in enumerate(zip(olds, news, strict=True)):
            allowed = tolerance * old["scenes"]
            for key, a in old.items():
                b = new.get(key)
                if key in IDENTITY:
                    bad = a != b
                else:
                    x, y = _counts(a), _counts(b)
                    bad = len(x) != len(y) or any(
                        abs(p - q) > allowed for p, q in zip(x, y, strict=True)
                    )
                if bad:
                    out.append(f"{part}[{i}].{key}: {a} != {b}")
    return out
