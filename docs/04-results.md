# Results

Every number on this page is a seed mean over 20 seeds of 1000 test scenes each, with the
lowest and highest seed in brackets, from `docs/studies/regrounding.json`;
`uv run python tools/study.py --check docs/studies/regrounding.json` reruns it, prints these
tables and compares every count. The settings are the defaults of `Experiment` in
`regrounding/config.py`: head re-grounding for 150 steps at a rate of 0.001, `k` from 0 to
64 drawn in complementary pairs, and conformal sets at `alpha` 0.1. The concepts, verdict
and mixed rows are models with three concept outputs; the direct rows are a network with
one output, the verdict. The study runs seeds 10
to 29; the source hue families and the scene layout were fixed on pilot runs of other
seeds.

## Concept supervision

| test | `k` | `round` % | `tone` % | `left` % | verdict % | false positives at 90 % sensitivity % |
|---|---:|---:|---:|---:|---:|---:|
| source | 0 | 99.9 (99.4 to 100.0) | 100.0 (100.0 to 100.0) | 99.6 (99.0 to 99.9) | 99.9 (99.6 to 100.0) | 0.0 (0.0 to 0.0) |
| target | 0 | 54.4 (48.2 to 73.9) | 1.1 (0.0 to 18.9) | 97.7 (91.2 to 99.9) | 86.3 (81.2 to 89.2) | 80.0 (61.5 to 90.6) |
| target | 1 | 54.5 (49.8 to 73.1) | 38.7 (20.5 to 51.7) | 90.7 (70.3 to 98.6) | 66.3 (31.4 to 86.0) | 38.0 (10.1 to 97.1) |
| target | 2 | 55.5 (48.8 to 71.5) | 32.0 (17.3 to 53.8) | 93.6 (79.0 to 98.6) | 82.4 (69.5 to 91.7) | 40.5 (13.8 to 82.0) |
| target | 4 | 55.1 (48.6 to 73.5) | 55.7 (36.1 to 90.9) | 93.9 (82.8 to 98.5) | 84.9 (72.2 to 90.6) | 34.2 (10.5 to 85.8) |
| target | 8 | 55.8 (50.0 to 74.3) | 75.2 (51.3 to 95.3) | 95.1 (77.4 to 98.5) | 85.5 (76.7 to 89.4) | 31.4 (11.4 to 83.4) |
| target | 16 | 58.5 (50.0 to 78.6) | 92.7 (78.4 to 100.0) | 95.8 (90.0 to 99.2) | 87.1 (81.4 to 92.5) | 19.0 (10.6 to 49.0) |
| target | 32 | 61.4 (50.7 to 79.3) | 97.6 (85.5 to 100.0) | 96.9 (92.3 to 99.0) | 89.3 (87.0 to 92.2) | 13.9 (8.4 to 25.3) |
| target | 64 | 65.8 (54.3 to 84.6) | 99.4 (97.3 to 100.0) | 97.7 (94.4 to 99.4) | 90.7 (86.6 to 96.1) | 10.2 (3.9 to 17.3) |
| target, every weight | 3000 | 96.0 (73.4 to 99.8) | 100.0 (100.0 to 100.0) | 99.7 (99.0 to 99.9) | 98.5 (89.4 to 100.0) | 0.9 (0.0 to 11.3) |
| target, from scratch | 3000 | 90.1 (78.0 to 98.2) | 100.0 (100.0 to 100.0) | 99.8 (99.3 to 100.0) | 97.4 (91.5 to 99.6) | 1.5 (0.0 to 7.5) |

A model that never says yes gets 87.5 % (85.9 to 89.2) of the target verdicts right. The
first `k` at which a concept is right on 90 % of the target test scenes, per seed in the
order 10 to 29, `>64` when no `k` of the study reaches it:

| target concept | first `k` at 90 % | seeds reaching it | lower median |
|---|---|---:|---:|
| `round` | >64 in every seed | 0 of 20 | >64 |
| `tone` | 8 32 32 16 16 16 32 16 32 16 32 16 16 64 16 32 64 8 4 16 | 20 of 20 | 16 |
| `left` | 0 in every seed | 20 of 20 | 0 |
| all three | >64 in every seed | 0 of 20 | >64 |

- **`tone` re-grounds, and the count is 16.** It reads inverted before re-grounding, 1.1 %
  (0.0 to 18.9), and every seed reaches 90 % within 64 target scenes: the first `k` lies
  between 4 and 64, with a lower median of 16. At 95 % the lower median is 32, in 20 of 20
  seeds.
- **`left` carries over:** 97.7 % (91.2 to 99.9) before any re-grounding, at 90 % in 20 of
  20 seeds at `k = 0`.
- **`round` does not re-ground within 64 scenes, though the target style allows it.** It
  is at 65.8 % (54.3 to 84.6) at `k = 64` and reaches 90 % in no seed, while fine-tuning
  every weight on 3000 target scenes brings it to 96.0 % (73.4 to 99.8) and training from
  scratch on them to 90.1 % (78.0 to 98.2). The blur alone does it, as the next section
  measures.
- **The verdict rises above the always-no rate only from `k = 32`:** 89.3 % (87.0 to 92.2)
  at `k = 32` and 90.7 % (86.6 to 96.1) at `k = 64`, against 87.5 %. The ranking improves
  sooner: false positives at 90 % sensitivity fall from 80.0 % to 19.0 % of the negatives
  at `k = 16` and to 10.2 % at `k = 64`, where an uninformative score has 90 %.
- **One scene costs a concept that needed none.** At `k = 1`, `left` falls from 97.7 % to
  90.7 % (70.3 to 98.6) and the verdict accuracy from 86.3 % to 66.3 % (31.4 to 86.0).

## Which shift

Three more studies change one property of the target at a time, the source's look
otherwise, under concept supervision on the same 20 seeds:
`docs/studies/regrounding-palette.json`, `regrounding-size.json` and
`regrounding-blur.json`, each rechecked by `tools/study.py --check`.

| target | `k` | `round` % | `tone` % | `left` % |
|---|---:|---:|---:|---:|
| palette | 0 | 94.5 (75.5 to 99.9) | 0.3 (0.0 to 5.0) | 99.0 (94.8 to 99.8) |
| palette | 64 | 98.4 (88.5 to 100.0) | 99.7 (97.6 to 100.0) | 98.7 (97.0 to 99.7) |
| size | 0 | 99.3 (96.7 to 100.0) | 100.0 (99.8 to 100.0) | 99.5 (99.0 to 100.0) |
| size | 64 | 98.4 (94.7 to 99.9) | 100.0 (99.8 to 100.0) | 99.2 (98.6 to 99.8) |
| blur | 0 | 55.5 (47.8 to 83.7) | 100.0 (100.0 to 100.0) | 99.6 (98.5 to 100.0) |
| blur | 64 | 71.8 (56.4 to 87.6) | 100.0 (99.9 to 100.0) | 99.3 (98.4 to 99.9) |
| blur, every weight | 3000 | 95.8 (54.6 to 100.0) | 100.0 (100.0 to 100.0) | 99.8 (99.5 to 100.0) |
| blur, from scratch | 3000 | 77.3 (59.4 to 98.9) | 100.0 (100.0 to 100.0) | 99.8 (99.4 to 100.0) |

- **The palette alone sets the count for `tone`.** It inverts `tone`, 0.3 % at `k = 0`, and
  every seed re-grounds it to 90 % with a lower median of 16 scenes, as with all three
  changes; `round` and `left` carry over, so the three concepts together reach 90 % in 19
  of 20 seeds, lower median 16.
- **Size alone moves nothing:** every concept is at 90 % at `k = 0` in 20 of 20 seeds.
- **Blur alone stops `round`:** 55.5 % at `k = 0` and 90 % in no seed by `k = 64`, while
  `tone` and `left` stay at 99 % or more. Fine-tuning every weight of the source model on
  3000 blurred scenes brings `round` to 95.8 %, and a model from scratch on them reaches
  77.3 %: the source's features help, and head re-grounding within 64 scenes does not reach
  them.

## Verdict supervision

Trained from verdicts alone, the model gets 99.9 % (99.7 to 100.0) of the source verdicts
right while its outputs agree with the named concepts on 61.7 % (56.1 to 64.2), 62.8 %
(60.3 to 65.4) and 62.5 % (59.4 to 65.3) of the scenes, and each output equals the verdict
itself on 98.5 % (89.3 to 100.0), 98.6 % (87.1 to 99.9) and 99.3 % (94.9 to 99.9): the
three outputs compute the conjunction, and an output equal to the verdict agrees with any
one concept on 1/2 + 1/8 = 62.5 % of the scenes. No concept reaches 90 % in any seed.
Re-grounding from up to 64 target verdicts leaves every concept between 39.2 % and 74.2 %
in every seed, and 3000 target verdicts do not repair it: fine-tuning every weight leaves
the concepts at 58.8 % (52.3 to 63.8), 64.2 % (53.7 to 77.7) and 63.5 % (60.7 to 71.5)
with the verdict right on 98.3 % (93.8 to 99.7). The conformal sets of these models are
calibrated on concept labels of source scenes, which verdict supervision otherwise never
sees, so their coverage lies outside the premise of the mode.

## The rule's part

Under concept supervision the rule enters only the measures. Two conditions put target
verdicts, one bit per scene, through it or around it. **mixed** starts from the concept
model of the source and re-grounds it from target verdicts through the fixed rule;
**direct** is a network of the same architecture with one output, the verdict, trained on
source verdicts and fine-tuned on the same target verdicts, with no concepts and no rule.

| test | `k` | false positives at 90 % sensitivity, mixed % | the same, direct % | verdict, mixed % | verdict, direct % |
|---|---:|---:|---:|---:|---:|
| source | 0 | 0.0 (0.0 to 0.0) | 1.2 (0.0 to 12.9) | 99.9 (99.6 to 100.0) | 98.7 (88.1 to 100.0) |
| target | 0 | 80.0 (61.5 to 90.6) | 99.3 (98.3 to 99.9) | 86.3 (81.2 to 89.2) | 84.4 (62.9 to 89.2) |
| target | 1 | 38.3 (10.1 to 97.1) | 90.7 (45.8 to 99.9) | 65.9 (31.4 to 84.6) | 20.4 (11.5 to 40.5) |
| target | 4 | 25.7 (10.0 to 76.6) | 97.7 (92.0 to 99.7) | 84.0 (70.9 to 89.1) | 56.3 (13.2 to 86.5) |
| target | 16 | 19.0 (8.8 to 80.9) | 83.6 (33.6 to 99.8) | 88.0 (83.8 to 92.6) | 80.1 (22.8 to 88.0) |
| target | 64 | 11.6 (5.2 to 24.8) | 57.2 (16.8 to 90.4) | 90.8 (86.2 to 95.5) | 86.0 (72.1 to 88.6) |
| target, every weight | 3000 | 0.0 (0.0 to 0.1) | 6.1 (0.0 to 100.0) | 99.2 (96.4 to 99.9) | 97.4 (86.7 to 99.9) |
| target, from scratch | 3000 | 0.7 (0.0 to 11.4) | 9.8 (0.0 to 15.9) | 98.3 (88.2 to 99.9) | 90.7 (85.6 to 98.6) |

- **The fixed rule carries target verdicts to the verdict.** At every `k` the mixed model
  ranks target scenes with fewer false positives than the direct network: 19.0 % against
  83.6 % at `k = 16`, 11.6 % against 57.2 % at `k = 64`. The direct network's verdict is
  below the always-no rate, 87.5 %, at every `k` up to 64.
- **It does not carry them to the concepts.** Re-grounded from verdicts, `tone` reaches
  90 % in 0 of 20 seeds within 64 scenes and stands at 49.7 % (43.5 to 55.3) at `k = 64`,
  against a lower median of 16 scenes with concept labels; `left` falls from 97.7 % to
  83.2 % (59.2 to 98.1). With 3000 target verdicts on every weight the verdict is right on
  99.2 % (96.4 to 99.9) while the concepts are at 64.8 % (49.5 to 74.3), 49.4 % (38.8 to
  64.3) and 67.4 % (48.6 to 93.0), and the `round` output equals the verdict on 91.7 %
  (77.3 to 99.3) of the scenes: a reasoning shortcut reached from grounded concepts, as
  verdict supervision on a conjunction permits (Marconato et al. 2023).

## Conformal sets

Under concept supervision, with sets calibrated on 500 source scenes, each concept's set
holds the true value on 89.6 % (87.3 to 91.9), 90.2 % (87.5 to 92.4) and 90.0 % (85.5 to
91.9) of the source scenes, where the guarantee is at least 90 % on average over
calibration draws. No source set is full and 10.4 % (8.1 to 12.7), 9.8 % (7.6 to 12.5) and
10.0 % (8.1 to 14.4) are empty: every covering set is a right singleton. On the target at
`k = 0` the same thresholds cover `round` on 39.1 % (11.3 to 51.9), `tone` on 0.0 % (0.0 to
0.0) and `left` on 58.6 % (24.7 to 82.1). At `k = 64`, still calibrated on source scenes,
`tone` is covered on 100.0 % (100.0 to 100.0) of the scenes because its set holds both
values on 85.5 % (58.6 to 99.0), and `left`, right on 97.7 %, is covered on 72.8 % (53.0 to
86.1) with 27.1 % (13.9 to 46.8) of its sets empty: the source thresholds fit neither.

No coverage bound holds on the target. The two styles share no scene, so the total-variation
distance between them is 1 and the bound of Barber et al. 2023 is vacuous, and weighting the
calibration scores by the likelihood ratio (Tibshirani et al. 2019) needs the target inside
the source's support, which it is not.

## Open

- A rule other than the conjunction, and a condition that varies the rule.
- Whether the count belongs to the scenes or to the optimiser: every `k` gets 150 steps at
  a rate of 0.001.
- Coverage with sets calibrated on target scenes.
- How much of a seed's range is the draw of its `k` scenes.
