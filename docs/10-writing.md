# Technical writing

Three surfaces, one standard: state what is true, with the evidence, in as few words as
carry it.

## The standard

**Authoritative.** Write the fact, not the journey to it. "The ladder starts at 640" — not
"we tried starting at 160 and then realised".

**No history.** Not in pages, not in comments, not in commit messages. A text describes
what is, not what was or how it came to be known. Where a past defect is load-bearing,
state the property that prevents it: "every scene of a seed comes from one generator in a
fixed order, so drawing the target pool before the source test set would change every source
count" — never "we drew them in another order and then discovered".

**No meta.** Do not describe the text, the process, or who did what. No "this section
covers", no "as mentioned above", no agent, tool or author names, no counts of how long
something went unnoticed.

**No marketing.** No "powerful", "robust", "seamless", "comprehensive", "state of the art".
A property either has a number or is not claimed.

**Lean.** Every sentence carries a fact, a number, or a consequence. Cut anything that
would survive being deleted.

## Claims

A claim needs evidence in the same sentence or the next one.

- **Measured** — give the number and the command: the verdict accuracy at `k = 4` over 5
  seeds, with the command that reproduces it, `tools/study.py --check
  docs/studies/regrounding.json`.
- **Inferred** — say so first, not in a trailing caveat. "The mechanism is X" and "X is
  consistent with the data" are different claims.
- **Untested** — say the gap and what would close it. A gap described as a gap is
  information; a gap left implicit is a defect.

Never write a comparative result from one seed without the words "seed N, one sample" in
the same sentence as the number.

## Numbers

Every published number needs a committed command. Prose is not a study: if a number cannot
be grepped out of a command's output, it is not a result.

Quote a rate with its denominator and its spread: the seed mean to one decimal, the lowest
and highest seed, the seed count and the scenes per seed — "(lowest to highest over 5
seeds, 1000 test scenes each)" — never the mean alone.

A study records counts; a ratio is computed where it is printed, to one decimal place of a
percentage. The counts are the record, and they compare exactly.

## Pages

Structure by what the reader is doing, not by how the code is organised.

State the limit of what a page covers. A page that silently omits a case reads as though
the case does not exist.

Link with relative paths; `tools/check_markdown.py` fails the build on a dead one.

### Markdown that GitHub renders

The renderer is strict and the failures are silent. The checker catches these; the rules
are here so they are not rediscovered.

- Never split a token across the math boundary. `${}^{238}\mathrm{U}$` or plain `U-238`,
  never `$^{238}$U`.
- No bare sub- or superscript spans.
- No spacing macros in inline `$...$`. They render literally.
- Keep a numeric range in one span. `$\sim10^4\text{ to }10^5$`, or plain text.
- Do not glue an opening `$` to a preceding `~` or `-`.
- When wrapping prose, never start a continuation line with `+ `, `- ` or `* `.
- No images, anywhere. A committed plot stops tracking the code that produced it and cannot
  be grepped against a study row.

## Code comments

**The default is no comment.** Code states what it does; a comment is for what the code
cannot state. Most lines need none.

Comment when the reason is not recoverable from the line:

- a value that looks arbitrary and is not
- an ordering that matters
- a library default being overridden
- an invariant a later edit would silently break
- a cheaper-looking alternative that is wrong

**Imperative mood.** "Sort the objects", not "This sorts the objects" or "Sorts the
objects". Same for docstring summaries, which `ruff` enforces.

Name the owner, the invariant and the failure:

```python
# Set before torch is imported. ATen reads ATEN_CPU_CAPABILITY once, at start-up, and a
# wider vector unit sums in a different order, so a count would change with the CPU.
```

That names what is set, who defaults it otherwise, what happens without it, and the size of
the effect. `# sort for determinism` carries none of it.

Do not restate the signature, narrate the obvious, or leave commented-out code.

Docstrings follow the numpy convention. A module docstring states what the module is
responsible for and what it must not do. A function docstring states the contract, not the
implementation.

## Commit messages

A commit states what the change does and what it measures. Same standard as everything
else: fact, no history, no meta.

```
scope: what the change does

The defect, stated as a property of the code. What the change does about it. The
measurement. What remains open.
```

- **Subject**: lowercase after the scope, no trailing period, under 72 characters. Scope is
  the module touched — `scenes`, `rule`, `model`, `train`, `conformal`, `experiment`,
  `config`, `study`, `docs`, `tools`.
- **Body**: wrapped at 88. Give the numbers.
- One logical change per commit. A subject needing "and" is two commits.
- No provenance: not who found it, not how, not what was tried first, not how long it stood.
- No references to uncommitted files, external trackers, or the tooling that produced the
  change.

A defect is described in the present tense as a property:

```
`thresholds` scores a 0 label as `1 - (1 - p)` and `prediction_sets` as `p`, which differ
by a rounding in float32, so a test scene identical to a calibration scene can miss the
threshold and drop its own true label. Both score through `nonconformity`;
`tests/test_conformal.py` pins the singleton sets of a confident model.
```

Not: "this was introduced by the previous fix", "found while investigating X", "had been
wrong for four milestones".

If a cause is not established, state what is established and what is open. Do not narrate
the search.

## What not to write

| avoid | write |
|---|---|
| "significantly faster" | "2.68x, 917.2 ms to 342.2 ms" |
| "should work correctly" | the test that shows it does |
| "we discovered that" | the fact |
| "this was broken for months" | the property that makes it correct |
| "obviously", "simply", "just" | nothing; delete |
| "comprehensive", "robust" | the coverage, as a number |
| "note that", "it is worth mentioning" | the note |
