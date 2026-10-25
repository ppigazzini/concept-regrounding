# Results

No study is published, and no number on these pages is a result. A study is published by
`tools/study.py --json docs/studies/regrounding.json` and checked by the same command with
`--check`; this page then quotes its table.

## Open

The following is inferred from runs with the default settings that are not a published
study, and every statement in it is untested until a study file holds it.

- **The source features do not carry shape and position to the target style.** With
  concept supervision, `round` and `left` fall to chance on the target at `k = 0`, and
  `tone` reads inverted, as the swapped palette intends. Head-only re-grounding on up to 12
  scenes recovers little of either. The inference is that the convolutional features tie
  shape and position to the source colours, which the head cannot undo. Two changes would
  test it: source scenes drawn in varied colours, so the features learn shape apart from
  colour, and `regrounding_scope="all"`.
- **Verdict supervision finds a reasoning shortcut.** Trained from verdicts alone, the
  model gets the verdict right on the source while its outputs agree with the named
  concepts barely above chance. `permuted_correct` against `concept_correct` separates a
  permutation of the concepts from a shortcut that no matching repairs.
- **Conformal coverage holds on the source and does not on the target**, as the loss of
  exchangeability predicts; how much of it re-grounding restores is not measured.

Each item becomes a result when a study file holds the counts, over 5 seeds of 1000 test
scenes each.
