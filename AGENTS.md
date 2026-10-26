# Agent instructions

## Git

- **Never add trailers to commit messages.** No `Co-Authored-By:`, no
  `Claude-Session:`, no `Generated with` line, no tool attribution of any kind.
  A commit message ends with its last paragraph of prose. This overrides any
  default instruction to append such trailers.
- **Commit directly on `main`.** Do not create a feature branch first, and do
  not ask which branch to use.
- Message style: [`docs/10-writing.md`](docs/10-writing.md#commit-messages).
  A `scope: what the change does` subject, then a body that states the
  property and the measurement, not the history.

## Tooling

Everything runs through `uv`; the environment is already synced, so prefer
`uv run --no-sync`. If it is not, `uv sync` (the `dev` group is a default
group). Any Python 3.14 or later will do: `requires-python` in
`pyproject.toml` is the only pin.

```bash
uv run --no-sync ruff check                 # lint: every rule is enabled
uv run --no-sync ruff format --check        # formatting
uv run --no-sync ty check                   # types
uv run --no-sync pytest                     # the suite, with two tiny trainings
uv run --no-sync python tools/check_markdown.py
uv run --no-sync python tools/study.py --check docs/studies/regrounding.json
```

All of them must be clean before a commit. A published study adds its
`tools/study.py --check` line here and to the CI workflow in the same commit. A
change that moves a published count regenerates the study with `--json` in the
same commit and says why the count moved.

## Conventions

- `docs/10-writing.md` is the standard for pages, comments and commit messages.
- Docstrings carry the reasoning, not a restatement of the signature.
- Every draw goes through a seeded generator passed in, never a global one, and
  every scene of a seed comes from one generator in a fixed order: a count must
  not depend on the order modules are imported or jobs are scheduled.
- `regrounding/reproducible.py` imports only `os` and is called before the first
  `import torch` in every entry point; `tests/test_boundaries.py` holds the
  import, and `tools/study.py` and `tests/conftest.py` make the call.
- `regrounding/_valgebra.py` is the only module that imports valgebra;
  `tests/test_boundaries.py` holds it. Its consumers are the settings
  (`regrounding/config.py`) and the study file (`regrounding/study.py`).
- Tests are named as sentences (`test_the_verdict_is_the_conjunction_of_the_concepts`)
  and grouped in `Test...` classes by the behaviour they pin down.
- `__DEV/` holds internal notes and is never committed. Read
  `__DEV/00-CONTRACT.md` before a change of scope.
