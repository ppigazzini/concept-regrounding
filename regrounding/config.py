"""The settings of one experiment, checked against the ranges their annotations declare.

A field's range is written once, as an ``Annotated`` refinement, and valgebra checks the
constructed object against the class's own annotations in ``__post_init__``: a setting a
caller assigns in code is refused as surely as a typo on the command line, with the field
named. The one rule between two fields -- re-grounding draws its examples from a pool
that must hold the largest shot count -- is plain Python.
"""

from dataclasses import dataclass, fields, replace
from typing import Annotated, Any, Literal

from annotated_types import Ge, Gt, Le, Lt

from regrounding._valgebra import NOT_BOOL, Validator, refuse

type Count = Annotated[int, NOT_BOOL, Ge(1)]
type Shots = Annotated[int, NOT_BOOL, Ge(0), Le(64)]
type Seed = Annotated[int, NOT_BOOL, Ge(0)]
type Rate = Annotated[float, Gt(0.0), Lt(1.0)]


@dataclass(frozen=True, slots=True)
class Experiment:
    """What one study runs: data sizes, training, re-grounding, and evaluation."""

    #: The smallest image that holds both objects of the largest radius apart.
    image_size: Annotated[int, NOT_BOOL, Ge(32), Le(128)] = 32
    source_train: Count = 3000
    calibration: Count = 500
    test: Count = 1000
    pool: Count = 64
    target_train: Count = 3000
    epochs: Annotated[int, NOT_BOOL, Ge(1), Le(200)] = 15
    batch_size: Annotated[int, NOT_BOOL, Ge(1), Le(4096)] = 64
    learning_rate: Rate = 2e-3
    regrounding_steps: Annotated[int, NOT_BOOL, Ge(0), Le(10_000)] = 150
    regrounding_rate: Rate = 1e-3
    regrounding_scope: Literal["readout", "head", "all"] = "head"
    #: Which target the scenes shift to: the target, or one of its three changes alone.
    target: Literal["target", "palette", "size", "blur"] = "target"
    modes: tuple[Literal["concepts", "verdict", "mixed", "direct"], ...] = (
        "concepts",
        "verdict",
        "mixed",
        "direct",
    )
    shots: tuple[Shots, ...] = (0, 1, 2, 4, 8, 16, 32, 64)
    #: Seeds no pilot run used: the scene design was fixed on runs scored on the target
    #: test scenes of seeds 0 to 4 and 100 to 104.
    seeds: tuple[Seed, ...] = tuple(range(10, 30))
    alpha: Annotated[float, Gt(0.0), Lt(0.5)] = 0.1
    sensitivity: Rate = 0.9

    def __post_init__(self) -> None:
        refuse(_SCHEMA, self, "Experiment")
        if max(self.shots, default=0) > self.pool:
            msg = f"Experiment: the largest shot count {max(self.shots)} exceeds pool {self.pool}"
            raise ValueError(msg)
        if not self.modes or len(set(self.modes)) != len(self.modes):
            msg = f"Experiment: modes must name each mode at most once, got {self.modes}"
            raise ValueError(msg)

    def updated(self, **changes: object) -> Experiment:
        """Return a copy with ``changes`` applied, checked like a new one."""
        return replace(self, **changes)

    def as_record(self) -> dict[str, Any]:
        """Return the settings as a JSON-ready mapping, tuples as lists."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = list(value) if isinstance(value, tuple) else value
        return out


_SCHEMA = Validator(Experiment)
