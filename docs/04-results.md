# Results

Every number on this page is a seed mean over 5 seeds of 1000 test scenes each, with the
lowest and highest seed in brackets, from `docs/studies/regrounding.json`;
`uv run python tools/study.py --check docs/studies/regrounding.json` reruns it, prints these
tables and compares every count. The settings are the defaults of `Experiment` in
`regrounding/config.py`: head re-grounding for 150 steps at a rate of 0.001, and conformal
sets at `alpha` 0.1. The source hue families and the scene layout were fixed after pilot
runs scored on target test scenes, seeds 0 to 4 among them, so the study is not a held-out
confirmation of them.

## Concept supervision

| test | `k` | `round` % | `tone` % | `left` % | verdict % | always no % | false positives at 90 % sensitivity % |
|---|---:|---:|---:|---:|---:|---:|---:|
| source | 0 | 99.8 (99.6 to 100.0) | 100.0 (100.0 to 100.0) | 99.5 (99.4 to 99.7) | 99.9 (99.7 to 100.0) | 87.2 (85.9 to 89.0) | 0.0 (0.0 to 0.0) |
| target | 0 | 55.4 (49.5 to 60.5) | 0.2 (0.0 to 1.0) | 94.6 (82.3 to 99.2) | 85.9 (80.5 to 88.2) | 87.5 (86.9 to 88.2) | 78.5 (62.3 to 90.7) |
| target | 1 | 50.0 (46.7 to 53.2) | 41.3 (23.8 to 48.9) | 88.4 (67.3 to 99.2) | 80.9 (63.6 to 87.9) | 87.5 (86.9 to 88.2) | 43.3 (9.4 to 76.2) |
| target | 2 | 56.9 (50.3 to 68.8) | 39.0 (27.8 to 48.9) | 86.4 (67.9 to 99.2) | 84.0 (74.4 to 88.2) | 87.5 (86.9 to 88.2) | 40.0 (8.8 to 64.3) |
| target | 4 | 57.6 (50.5 to 63.9) | 51.0 (29.2 to 73.2) | 87.6 (66.9 to 99.1) | 82.1 (73.5 to 88.2) | 87.5 (86.9 to 88.2) | 49.5 (15.7 to 85.7) |
| target | 6 | 57.6 (52.3 to 63.5) | 69.0 (41.4 to 93.4) | 90.0 (65.8 to 98.3) | 86.5 (76.8 to 93.2) | 87.5 (86.9 to 88.2) | 33.2 (12.8 to 71.7) |
| target | 12 | 58.1 (53.5 to 61.8) | 79.0 (64.6 to 97.1) | 91.8 (70.5 to 99.3) | 88.7 (84.4 to 94.0) | 87.5 (86.9 to 88.2) | 18.5 (11.7 to 31.3) |

The first `k` at which a concept is right on 90 % of the target test scenes, per seed in
the order 0 to 4, `>12` when no `k` of the study reaches it:

| target concept | first `k` at 90 % | seeds reaching it |
|---|---|---:|
| `round` | >12 >12 >12 >12 >12 | 0 of 5 |
| `tone` | >12 >12 >12 6 >12 | 1 of 5 |
| `left` | >12 0 0 0 0 | 4 of 5 |
| all three | >12 >12 >12 >12 >12 | 0 of 5 |

- **The count is censored at 12.** Within 12 target scenes no seed reaches 90 % on `round`
  or on the three concepts together, and one seed of 5 does on `tone`. How many scenes
  re-grounding takes is larger than the study measures for every concept that needs it.
- **`left` carries over** in 4 of 5 seeds: 94.6 % (82.3 to 99.2) before any re-grounding.
- **`tone` reads inverted and moves with `k`.** It is right on 0.2 % (0.0 to 1.0) of the
  scenes at `k = 0`, so the inverted output is right on 99.0 to 100.0 %, and on 79.0 % (64.6
  to 97.1) at `k = 12`.
- **`round` stays near chance:** 55.4 % (49.5 to 60.5) at `k = 0` and 58.1 % (53.5 to
  61.8) at `k = 12`. Inferred, not separated by a study: the blur, since `round` is the one
  concept that rests on the outline and only the target blurs it.
- **The verdict stays at the always-no rate.** The thresholded verdict is right on 88.7 %
  (84.4 to 94.0) of the scenes at `k = 12`, against 87.5 % for a model that never says yes,
  and below that rate at every `k` up to 6. The ranking improves: false positives at 90 %
  sensitivity fall from 78.5 % to 18.5 % of the negatives, where an uninformative score has
  90 %.
- **One scene costs a concept that needed none.** At `k = 1`, `left` falls from 94.6 % to
  88.4 % (67.3 to 99.2) while `tone` rises to 41.3 %.

## Verdict supervision

Trained from verdicts alone, the model gets 99.9 % (99.6 to 100.0) of the source verdicts
right while its outputs agree with the named concepts on 64.0 % (63.1 to 65.1), 62.9 %
(60.9 to 66.3) and 63.6 % (61.8 to 67.6) of the scenes, and under the best matching of
outputs to concepts on 63.7 % (62.4 to 65.8): matching outputs to other concepts recovers
almost nothing. Inferred, not measured: every output computes the whole conjunction, since
an output equal to the verdict agrees with any one concept on 1/2 + 1/8 = 62.5 % of the
scenes. Re-grounding from up to 12 target verdicts leaves every concept between 30.8 % and
72.4 % in every seed, and the false positives at 90 % sensitivity at 79.0 % (33.9 to 98.0)
at `k = 12`. The conformal sets of these models are calibrated on concept labels of source
scenes, which verdict supervision otherwise never sees, so their coverage lies outside the
premise of the mode.

## Conformal sets

On the source, with sets calibrated on 500 source scenes, each concept's set holds the true
value on 89.9 % (86.9 to 91.3), 90.7 % (88.1 to 93.2) and 90.2 % (89.1 to 91.7) of the
scenes, where the guarantee is at least 90 % on average over calibration draws. A set that
holds both values counts as covering, and a set can be empty; the study counts neither, so
coverage is not the share of right singletons. On the target at `k = 0` the same thresholds
cover `round` on 30.9 % (4.7 to 49.5), `tone` on 0.0 % (0.0 to 0.0) and `left` on 51.8 %
(4.6 to 91.8). At `k = 12`, still calibrated on source scenes, `tone` is covered on 95.0 %
(75.8 to 100.0), full sets included.

No coverage bound holds on the target. The two styles share no scene, so the total-variation
distance between them is 1 and the bound of Barber et al. 2023 is vacuous, and weighting the
calibration scores by the likelihood ratio (Tibshirani et al. 2019) needs the target inside
the source's support, which it is not.

## Open

- How many scenes `round` and `tone` take past 12, against the best each concept reaches in
  the target style with ample labels.
- Whether holding the rule fixed contributes: under concept supervision the rule enters only
  the measures, and no condition varies it.
- Whether the count belongs to the scenes or to the optimiser: every `k` gets 150 steps at
  a rate of 0.001.
- Which of palette, size and blur moves which count.
- How many sets are empty or full, and coverage with sets calibrated on target scenes.
