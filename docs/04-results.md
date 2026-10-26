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
| source | 0 | 99.9 (99.4 to 100.0) | 100.0 (100.0 to 100.0) | 99.6 (98.9 to 99.9) | 99.9 (99.7 to 100.0) | 0.0 (0.0 to 0.0) |
| target | 0 | 52.8 (48.2 to 75.7) | 0.2 (0.0 to 3.0) | 97.5 (92.2 to 99.8) | 85.8 (66.6 to 89.2) | 76.6 (52.8 to 91.0) |
| target | 1 | 53.4 (48.7 to 59.1) | 39.5 (20.6 to 50.5) | 90.9 (62.0 to 98.3) | 67.2 (44.2 to 88.7) | 34.8 (8.8 to 98.3) |
| target | 2 | 55.0 (47.7 to 74.4) | 33.1 (20.4 to 47.8) | 93.0 (70.9 to 99.6) | 80.5 (65.3 to 88.3) | 37.6 (11.8 to 88.1) |
| target | 4 | 54.8 (49.9 to 68.7) | 54.7 (37.3 to 86.3) | 93.1 (78.6 to 99.1) | 84.5 (72.9 to 89.6) | 37.3 (11.0 to 86.8) |
| target | 8 | 55.1 (50.5 to 66.5) | 73.4 (50.9 to 95.3) | 95.2 (78.9 to 98.5) | 84.4 (73.7 to 89.7) | 31.2 (11.4 to 88.7) |
| target | 16 | 57.3 (50.0 to 78.7) | 91.7 (76.6 to 100.0) | 95.9 (88.0 to 99.2) | 86.8 (81.7 to 91.5) | 19.7 (10.0 to 58.8) |
| target | 32 | 59.9 (50.2 to 82.0) | 97.6 (85.1 to 100.0) | 96.9 (93.6 to 99.1) | 89.0 (86.3 to 92.4) | 13.7 (7.2 to 28.9) |
| target | 64 | 64.1 (52.4 to 80.9) | 99.4 (97.2 to 100.0) | 97.6 (93.9 to 99.5) | 90.2 (86.7 to 94.9) | 10.6 (5.5 to 17.2) |
| target, every weight | 3000 | 95.9 (71.0 to 99.9) | 100.0 (100.0 to 100.0) | 99.7 (99.2 to 99.9) | 98.6 (88.4 to 100.0) | 1.1 (0.0 to 12.3) |
| target, from scratch | 3000 | 89.7 (76.2 to 98.5) | 100.0 (100.0 to 100.0) | 99.7 (99.2 to 100.0) | 97.1 (90.2 to 99.8) | 1.8 (0.0 to 10.0) |

A model that never says yes gets 87.5 % (85.9 to 89.2) of the target verdicts right. The
first `k` at which a concept is right on 90 % of the target test scenes, per seed in the
order 10 to 29, `>64` when no `k` of the study reaches it:

| target concept | first `k` at 90 % | seeds reaching it | lower median |
|---|---|---:|---:|
| `round` | >64 in every seed | 0 of 20 | >64 |
| `tone` | 8 32 32 16 16 16 32 16 32 16 16 16 8 64 32 32 64 8 8 16 | 20 of 20 | 16 |
| `left` | 0 in every seed | 20 of 20 | 0 |
| all three | >64 in every seed | 0 of 20 | >64 |

- **`tone` re-grounds, and the count is 16.** It reads inverted before re-grounding, 0.2 %
  (0.0 to 3.0), and every seed reaches 90 % within 64 target scenes: the first `k` lies
  between 8 and 64, with a lower median of 16. At 95 % the lower median is 32, in 20 of 20
  seeds.
- **`left` carries over:** 97.5 % (92.2 to 99.8) before any re-grounding, at 90 % in 20 of
  20 seeds at `k = 0`.
- **`round` does not re-ground within 64 scenes, though the target style allows it.** It
  is at 64.1 % (52.4 to 80.9) at `k = 64` and reaches 90 % in no seed, while fine-tuning
  every weight on 3000 target scenes brings it to 95.9 % (71.0 to 99.9) and training from
  scratch on them to 89.7 % (76.2 to 98.5). The blur alone stops it, as
  [Which shift](#which-shift) measures.
- **The verdict rises above the always-no rate only from `k = 32`:** 89.0 % (86.3 to 92.4)
  at `k = 32` and 90.2 % (86.7 to 94.9) at `k = 64`, against 87.5 %. The ranking improves
  sooner: false positives at 90 % sensitivity fall from 76.6 % to 19.7 % of the negatives
  at `k = 16` and to 10.6 % at `k = 64`, where an uninformative score has 90 %.
- **One scene costs a concept that needed none.** At `k = 1`, `left` falls from 97.5 % to
  90.9 % (62.0 to 98.3) and the verdict accuracy from 85.8 % to 67.2 % (44.2 to 88.7).

## Re-grounding procedures

Four more studies re-ground the same source models, under concept supervision on the same
20 seeds, in other ways: the first convolution alone (`regrounding-first.json`), the
readout alone for the same 150 steps (`regrounding-readout.json`), the readout fitted to
convergence under a pull toward its source weights (`regrounding-converged.json`), and the
head with 64 labelled source scenes in every batch (`regrounding-rehearsal.json`). Each
holds the rows of the main study's ceiling and from-scratch models unchanged.

| procedure | `tone` % at `k = 1` | `tone` % at `k = 64` | `left` % at `k = 1` | `left` % at `k = 64` | `tone` at 90 %: seeds, lower median of the first `k` |
|---|---:|---:|---:|---:|---:|
| head, 150 steps | 39.5 (20.6 to 50.5) | 99.4 (97.2 to 100.0) | 90.9 (62.0 to 98.3) | 97.6 (93.9 to 99.5) | 20 of 20, 16 |
| first convolution, 150 steps | 36.8 (12.3 to 53.0) | 58.0 (15.6 to 100.0) | 69.5 (48.2 to 91.2) | 68.0 (35.7 to 96.0) | 8 of 20, >64 |
| readout, 150 steps | 37.8 (16.1 to 51.6) | 13.6 (0.1 to 35.7) | 95.4 (71.3 to 99.8) | 97.6 (93.4 to 99.6) | 0 of 20, >64 |
| readout, converged | 43.0 (25.1 to 53.0) | 100.0 (99.7 to 100.0) | 94.8 (55.5 to 99.8) | 98.0 (95.4 to 99.7) | 20 of 20, 8 |
| head with rehearsal | 6.7 (1.0 to 17.4) | 91.8 (81.5 to 99.4) | 97.0 (93.8 to 99.3) | 97.9 (93.7 to 99.5) | 14 of 20, 64 |

- **The count by steps holds the optimiser's budget.** The readout carries `tone`: fitted
  to convergence it re-grounds it in every seed with a lower median of 8 scenes, half the
  head's 16, while the same readout moved by 150 steps at a rate of 0.001 reaches 90 % in no
  seed and stands at 13.6 % at `k = 64`.
- **The first convolution does not re-ground a shift at the input here.** Fine-tuned alone,
  as surgical fine-tuning favours for input-level shifts (Lee et al. 2023), it brings `tone`
  to 90 % in 8 of 20 seeds by 64 and costs `left`, 68.0 % at `k = 64`.
- **Rehearsal holds `left` and slows `tone`.** With 64 source scenes in every batch `left`
  stays at 97.0 % at `k = 1`, against 90.9 % without, and `tone` reaches 90 % in 14 of 20
  seeds, lower median 64.
- **No procedure re-grounds `round` within 64 scenes.**

## Draws

One more study (`regrounding-draws.json`) re-grounds each seed's source model, under concept
supervision, on 10 draws of the target pool instead of one, the first being the main study's;
its brackets run over the 200 seed and draw pairs.

| target concept | first `k` at 90 %, per seed, lowest..highest of its 10 draws | pairs reaching it | lower median |
|---|---|---:|---:|
| `tone` | 8..64 8..32 16..32 8..32 8..32 8..32 16..32 4..32 8..32 8..32 8..32 8..64 8..32 32..64 16..32 16..32 8..64 8..32 8..32 8..16 | 200 of 200 | 16 |
| `left` | 0 in every draw of every seed | 200 of 200 | 0 |
| `round` | >64 in every draw of every seed | 0 of 200 | >64 |

- **The count holds over draws:** `tone` reaches 90 % in 200 of 200 pairs, lower median
  16, as in the main study's single draw; at `k = 16` it is right on 91.9 % (53.8 to
  100.0) of the scenes.
- **The draw moves the count as much as the seed does.** Within a seed the first `k` spans
  a factor of 4 or more, such as 8 to 32, in 14 of the 20 seeds, and the seeds' ranges
  overlap: which scenes are drawn moves the count as much as how the model trained.

## Which shift

Three more studies change one property of the target at a time, the source's look
otherwise, under concept supervision on the same 20 seeds:
`docs/studies/regrounding-palette.json`, `regrounding-size.json` and
`regrounding-blur.json`, each rechecked by `tools/study.py --check`.

| target | `k` | `round` % | `tone` % | `left` % |
|---|---:|---:|---:|---:|
| palette | 0 | 93.0 (73.4 to 100.0) | 0.0 (0.0 to 0.4) | 98.6 (96.0 to 99.8) |
| palette | 64 | 97.9 (87.4 to 100.0) | 99.7 (97.8 to 100.0) | 98.6 (95.1 to 99.7) |
| size | 0 | 99.4 (98.3 to 100.0) | 100.0 (100.0 to 100.0) | 99.5 (99.1 to 100.0) |
| size | 64 | 98.3 (91.6 to 99.9) | 100.0 (99.7 to 100.0) | 99.2 (98.2 to 99.8) |
| blur | 0 | 53.0 (47.8 to 72.8) | 100.0 (100.0 to 100.0) | 99.5 (98.8 to 100.0) |
| blur | 64 | 71.2 (55.4 to 89.6) | 100.0 (100.0 to 100.0) | 99.2 (97.9 to 99.8) |
| blur, every weight | 3000 | 99.6 (95.7 to 100.0) | 100.0 (100.0 to 100.0) | 99.8 (99.4 to 100.0) |
| blur, from scratch | 3000 | 76.1 (59.8 to 98.8) | 100.0 (100.0 to 100.0) | 99.8 (99.4 to 100.0) |

- **The palette alone sets the count for `tone`.** It inverts `tone`, 0.0 % at `k = 0`, and
  every seed re-grounds it to 90 % with a lower median of 16 scenes, as with all three
  changes; `left` carries over and `round` stands at 93.0 % at `k = 0`, so the three
  concepts together reach 90 % in 19 of 20 seeds, lower median 16.
- **Size alone moves nothing:** every concept is at 90 % at `k = 0` in 20 of 20 seeds.
- **Blur alone stops `round`:** 53.0 % at `k = 0` and 90 % in no seed by `k = 64`, while
  `tone` and `left` stay at 99 % or more. Fine-tuning every weight of the source model on
  3000 blurred scenes brings `round` to 99.6 %, and a model from scratch on them reaches
  76.1 %: the source's features help, and head re-grounding within 64 scenes does not reach
  them.

## Verdict supervision

Trained from verdicts alone, the model gets 99.8 % (99.5 to 100.0) of the source verdicts
right while its outputs agree with the named concepts on 61.5 % (49.8 to 64.2), 62.9 %
(60.2 to 65.4) and 62.5 % (59.8 to 65.3) of the scenes, and each output equals the verdict
itself on 97.9 % (74.4 to 100.0), 98.9 % (90.6 to 100.0) and 99.3 % (95.1 to 99.9): the
three outputs track the conjunction, and an output equal to the verdict agrees with any one
concept on 1/2 + 1/8 = 62.5 % of the scenes. No concept reaches 90 % in any seed.
Re-grounding from up to 64 target verdicts leaves every concept between 39.9 % and 74.3 %
in every seed, and 3000 target verdicts do not repair it: fine-tuning every weight leaves
the concepts at 58.3 % (46.4 to 62.2), 65.4 % (54.5 to 77.2) and 63.9 % (59.5 to 71.9)
with the verdict right on 98.1 % (91.6 to 99.6). The conformal sets of these models are
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
| source | 0 | 0.0 (0.0 to 0.0) | 1.5 (0.0 to 12.0) | 99.9 (99.7 to 100.0) | 98.3 (87.6 to 100.0) |
| target | 0 | 76.6 (52.8 to 91.0) | 99.3 (98.4 to 99.9) | 85.8 (66.6 to 89.2) | 84.4 (63.0 to 89.2) |
| target | 1 | 35.0 (8.8 to 98.3) | 92.1 (59.9 to 99.9) | 67.0 (44.2 to 88.3) | 21.2 (12.0 to 41.1) |
| target | 4 | 28.7 (11.1 to 88.8) | 97.9 (91.6 to 99.5) | 83.2 (63.5 to 89.1) | 56.6 (13.3 to 86.7) |
| target | 16 | 23.6 (8.9 to 91.3) | 82.6 (41.0 to 99.5) | 87.5 (83.6 to 93.8) | 81.8 (49.7 to 88.6) |
| target | 64 | 12.5 (6.2 to 25.0) | 58.8 (18.9 to 88.8) | 90.0 (87.0 to 94.8) | 86.3 (77.4 to 89.1) |
| target, every weight | 3000 | 0.0 (0.0 to 0.6) | 1.9 (0.0 to 13.2) | 99.5 (97.7 to 100.0) | 97.4 (87.7 to 99.8) |
| target, from scratch | 3000 | 0.7 (0.0 to 11.8) | 9.6 (0.0 to 15.7) | 98.2 (88.4 to 99.8) | 90.7 (85.7 to 99.0) |

- **The fixed rule carries target verdicts to the verdict.** At every `k` the mixed model
  ranks target scenes with fewer false positives than the direct network: 23.6 % against
  82.6 % at `k = 16`, 12.5 % against 58.8 % at `k = 64`. The direct network's verdict is
  below the always-no rate, 87.5 %, at every `k` up to 64.
- **It does not carry them to the concepts.** Re-grounded from verdicts, `tone` reaches
  90 % in 0 of 20 seeds within 64 scenes and stands at 48.3 % (30.5 to 53.4) at `k = 64`,
  against a lower median of 16 scenes with concept labels; `left` falls from 97.5 % to
  84.9 % (62.0 to 97.9). With 3000 target verdicts on every weight the verdict is right on
  99.5 % (97.7 to 100.0) while the concepts are at 67.1 % (58.3 to 74.8), 49.5 % (36.6 to
  80.1) and 67.3 % (48.5 to 99.1), and the `round` output equals the verdict on 92.8 %
  (77.9 to 99.7) of the scenes: a reasoning shortcut reached from grounded concepts, as
  verdict supervision on a conjunction permits (Marconato et al. 2023).

## Conformal sets

Under concept supervision, with sets calibrated on 500 source scenes, each concept's set
holds the true value on 89.8 % (87.4 to 92.9), 90.1 % (87.5 to 93.1) and 90.2 % (88.1 to
91.9) of the source scenes, where the guarantee is at least 90 % on average over
calibration draws. No source set is full and 10.1 % (7.1 to 12.6), 9.9 % (6.9 to 12.5) and
9.8 % (8.1 to 11.8) are empty: every covering set is a right singleton. On the target at
`k = 0` the same thresholds cover `round` on 40.6 % (10.0 to 51.9), `tone` on 0.0 % (0.0 to
0.0) and `left` on 58.0 % (28.2 to 86.1). At `k = 64`, still calibrated on source scenes,
`tone` is covered on 100.0 % (100.0 to 100.0) of the scenes because its set holds both
values on 83.5 % (54.7 to 100.0), and `left`, right on 97.6 %, is covered on 70.6 % (33.2 to
94.6) with 29.2 % (4.9 to 66.1) of its sets empty: the source thresholds fit neither.

No coverage bound holds on the target. The two styles share no scene, so the total-variation
distance between them is 1 and the bound of Barber et al. 2023 is vacuous, and weighting the
calibration scores by the likelihood ratio (Tibshirani et al. 2019) needs the target inside
the source's support, which it is not.

Two more studies change the calibration of the target models, under concept supervision on
the same 20 seeds. Calibrated on 500 labelled target scenes
(`regrounding-target-calibrated.json`), every target model's sets cover each concept on
89.3 % to 100.0 % of the scenes at every `k`, seed means, 100.0 % where the sets hold both
values, as exchangeability with the target test scenes guarantees, and the counts say what
valid sets cost: at `k = 0` the `tone` set holds both values on 98.1 % (88.5 to 100.0) of
the scenes, at `k = 64` on 0.0 % (0.0 to 0.0), and the three-valued verdict is decided on
80.3 % (69.1 to 94.0) of the scenes, right on 78.1 % (68.1 to 91.7), against 61.6 % (44.0
to 77.5) and 58.6 % (44.0 to 73.5) at `k = 0`. With one threshold per concept value on the
source calibration scenes (`regrounding-conditional.json`), the target coverage stays short:
`round` at 49.2 % (11.5 to 95.1) at `k = 64`; the verdict is decided on 90.8 % (36.1 to
100.0) and right on 80.8 % (36.1 to 89.0), where one threshold per concept decides 41.8 %
(23.1 to 56.9) and gets 41.5 % (23.1 to 55.5) right.

## Open

- A rule other than the conjunction, and a condition that varies the rule.
- How the converged count depends on the pull toward the source readout, whose weight,
  0.01, is not tuned.
- Conformal sets calibrated on the `k` target scenes themselves, which re-grounding also
  trains on (Fisch et al. 2021 calibrate on few scenes with auxiliary tasks).
