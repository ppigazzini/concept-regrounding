# Results

Every number on this page is a seed mean over 5 seeds of 1000 test scenes each, with the
lowest and highest seed in brackets, from `docs/studies/regrounding.json`;
`uv run python tools/study.py --check docs/studies/regrounding.json` reruns it and compares
every count. The defaults of `Experiment` in `regrounding/config.py` are the settings:
head re-grounding for 150 steps at a rate of 0.001, and conformal sets at `alpha` 0.1.

## Concept supervision

| test | `k` | `round` % | `tone` % | `left` % | false positives at 90 % sensitivity % |
|---|---:|---:|---:|---:|---:|
| source | 0 | 99.8 (100 to 100) | 100.0 (100 to 100) | 99.5 (99 to 100) | 0.0 (0 to 0) |
| target | 0 | 55.4 (50 to 60) | 0.2 (0 to 1) | 94.6 (82 to 99) | 78.5 (62 to 91) |
| target | 1 | 50.0 (47 to 53) | 41.3 (24 to 49) | 88.4 (67 to 99) | 43.3 (9 to 76) |
| target | 2 | 56.9 (50 to 69) | 39.0 (28 to 49) | 86.4 (68 to 99) | 40.0 (9 to 64) |
| target | 4 | 57.6 (50 to 64) | 51.0 (29 to 73) | 87.6 (67 to 99) | 49.5 (16 to 86) |
| target | 6 | 57.6 (52 to 64) | 69.0 (41 to 93) | 90.0 (66 to 98) | 33.2 (13 to 72) |
| target | 12 | 58.1 (54 to 62) | 79.0 (65 to 97) | 91.8 (70 to 99) | 18.5 (12 to 31) |

- **`left` carries over.** 94.6 % (82 to 99) on the target before any re-grounding.
- **`tone` reads inverted and re-grounds.** 0.2 % at `k = 0`, 79.0 % (65 to 97) at
  `k = 12`, with the rule unchanged; false positives at 90 % sensitivity fall from 78.5 %
  to 18.5 % of the negatives.
- **`round` does not re-ground within 12 scenes.** 55.4 % at `k = 0` and 58.1 % at
  `k = 12`. Inferred, not separated by a study: the blur, since `round` is the one concept
  that rests on the outline and only the target blurs it.
- **One scene costs a concept that needed none.** At `k = 1`, `left` falls from 94.6 % to
  88.4 % (67 to 99) while `tone` rises only to 41.3 %.

## Verdict supervision

Trained from verdicts alone, the model gets 99.9 % (100 to 100) of the source verdicts
right while its outputs agree with the named concepts on 64.0 % (63 to 65), 62.9 % (61 to
66) and 63.6 % (62 to 68) of the scenes, and the best matching of outputs to concepts on
63.7 % (62 to 66): the outputs do not carry the concepts under other names. Inferred, not
measured: every output computes the whole conjunction, since an output that equals the
verdict agrees with any one concept on 1/2 + 1/8 = 62.5 % of the scenes. Re-grounding from
up to 12 verdicts leaves every concept between 42.8 % and 56.9 % on the target, and the
false positives at 90 % sensitivity at 79.0 % (34 to 98).

## Conformal sets

On the source, with sets calibrated on 500 source scenes, each concept's set holds the true
value on 89.9 % (87 to 91), 90.7 % (88 to 93) and 90.2 % (89 to 92) of the scenes, where
the guarantee is at least 90 % on average over calibration draws. On the target at `k = 0`
the same thresholds cover `round` on 30.9 %, `tone` on 0.0 % and `left` on 51.8 %: the shift breaks exchangeability and the
guarantee with it. Re-grounding brings `tone` to 95.0 % (76 to 100) at `k = 12`, still
calibrated on source scenes.

## Open

- Whether `round` re-grounds with more than 12 target scenes, or with every weight
  fine-tuned (`regrounding_scope="all"`), is not in a study.
- Which of palette, size and blur moves which count is not separated.
- The conformal sets are not corrected for the shift; weighting the calibration scores by
  the likelihood ratio between the styles (Tibshirani et al. 2019) would be.
