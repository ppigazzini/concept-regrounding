"""The study file: the counts one run of the experiment publishes, and the check of them.

Written by ``tools/study.py --json`` and read back by ``tools/study.py --check``, which
reruns the experiment and compares. Its shape is declared once, here, as closed records
checked by valgebra: a key the format does not name is refused, every count is a
non-negative integer that is not a ``bool``, and every failure is reported at once with
its path. The rules between fields of a row are plain Python in `check_rows`.
"""

import json
from typing import Annotated, Any, Literal

from annotated_types import Ge, Len
from typing_extensions import TypedDict

from regrounding._valgebra import NOT_BOOL, ValidationError, Validator

type Count = Annotated[int, NOT_BOOL, Ge(0)]
type Triple = Annotated[list[Count], Len(3, 3)]


class Row(TypedDict, closed=True):
    """One model on one test set: counts only."""

    mode: Literal["concepts", "verdict"]
    seed: Count
    shots: Count
    domain: Literal["source", "target"]
    scenes: Count
    concept_correct: Triple
    permuted_correct: Count
    verdict_correct: Count
    positives: Count
    negatives: Count
    false_positives: Count
    covered: Triple
    decided: Count
    decided_correct: Count


class Study(TypedDict, closed=True):
    """The settings and every row of one run."""

    format: Literal[1]
    device: Literal["cpu", "cuda"]
    settings: dict[str, Any]
    rows: list[Row]


class StudyError(ValueError):
    """A study file that is malformed or contradicts itself."""


STUDY = Validator(Study)

#: The fields that say which model and which scenes a row is about. The scenes come from
#: NumPy's generator and not from torch's floating point, so these compare exactly at any
#: tolerance: a tolerance on them would accept a row of the wrong seed or shot count.
IDENTITY = frozenset({"mode", "seed", "shots", "domain", "scenes", "positives", "negatives"})


def check_rows(study: Study) -> list[str]:
    """Return every rule between two fields that a row breaks."""
    out: list[str] = []
    for i, r in enumerate(study["rows"]):
        n = r["scenes"]
        bounded = [*r["concept_correct"], *r["covered"], r["verdict_correct"], r["decided"]]
        if any(c > n for c in bounded) or r["permuted_correct"] > 3 * n:
            out.append(f"rows[{i}]: a count exceeds the {n} scenes")
        if r["positives"] + r["negatives"] != n:
            out.append(f"rows[{i}]: positives + negatives != scenes")
        if r["false_positives"] > r["negatives"] or r["decided_correct"] > r["decided"]:
            out.append(f"rows[{i}]: a part exceeds its whole")
        if r["permuted_correct"] < sum(r["concept_correct"]):
            out.append(f"rows[{i}]: the best matching is worse than the identity")
    return out


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


def differences(expected: Study, actual: Study, tolerance: float = 0.0) -> list[str]:
    """Return one line per setting or count that differs by more than the tolerance.

    ``tolerance`` is a fraction of a row's scenes. A run on the machine that produced the
    study reproduces every count exactly; another machine's floating point can move a
    count by a few scenes, and a nonzero tolerance says how many are accepted. It never
    applies to the fields in `IDENTITY`.
    """
    out = (
        [f"settings: {expected['settings']} != {actual['settings']}"]
        if (expected["settings"] != actual["settings"])
        else []
    )
    if len(expected["rows"]) != len(actual["rows"]):
        return [*out, f"rows: {len(expected['rows'])} != {len(actual['rows'])}"]
    for i, (old, new) in enumerate(zip(expected["rows"], actual["rows"], strict=True)):
        for key in old:
            allowed = 0.0 if key in IDENTITY else tolerance * old["scenes"]
            a, b = old[key], new[key]
            if isinstance(a, list) and isinstance(b, list):
                bad = any(abs(x - y) > allowed for x, y in zip(a, b, strict=True))
            elif isinstance(a, int) and isinstance(b, int):
                bad = abs(a - b) > allowed
            else:
                bad = a != b
            if bad:
                out.append(f"rows[{i}].{key}: {a} != {b}")
    return out
