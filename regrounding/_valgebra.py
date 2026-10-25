"""Every valgebra name the package uses, and the one helper it composes from them.

Nothing else in the package imports valgebra: a valgebra release that renames or moves a
name is one file's edit here, and `tests/test_boundaries.py` holds the boundary. The names
come from the ``valgebra`` package, whose ``__all__`` is the supported surface.
"""

from typing import Any

from valgebra import ValidationError, Validator, complement

__all__ = ["NOT_BOOL", "ValidationError", "Validator", "refuse"]

#: Every value but a `bool`, which subclasses `int` and so lies inside a bare `int`.
NOT_BOOL = complement(bool)


def refuse(schema: Validator[Any], value: object, what: str) -> None:
    """Raise `ValueError` naming every way ``value`` misses ``schema``, each with its path.

    `ValueError` because that is what a dataclass's ``__post_init__`` raises and what a
    caller of a constructor catches; valgebra supplies the located failures.
    """
    try:
        schema.validate(value)
    except ValidationError as exc:
        failures = "; ".join(str(item["message"]) for item in exc.errors)
        msg = f"{what}: {failures}"
        raise ValueError(msg) from exc
