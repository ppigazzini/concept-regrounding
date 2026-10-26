# Method

The question: when only the appearance of the concepts changes and the rule over them does
not, how many labelled scenes from the new appearance does it take to ground the concepts
again?

## Scenes

A scene is a 32-pixel RGB image of two objects. Object A, coloured, carries the concepts;
object B, grey, is the reference for the spatial concept and a distractor for the others.

| concept | true when |
|---|---|
| `round` | A is a circle rather than a square |
| `tone` | A has tone 1 rather than tone 0 |
| `left` | A's centre lies at least 2 pixels left of B's |

Each concept is drawn independently with probability one half, so the verdict, their
conjunction, is true in one scene out of eight. Two styles draw the same concepts:

| style | tone 0 hue | tone 1 hue | radius of A | blur |
|---|---|---|---|---|
| source | blues, 200 to 260 degrees | reds, 330 to 30 degrees | 4 to 6 | no |
| target | yellow, 50 degrees | teal, 180 degrees | 3 to 7 | 3 by 3 mean |

Every tone has saturation 0.8 and value 0.9. A source scene draws its hue from the tone's
family, so that no one colour goes with a shape or a position in the source scenes. Each
target hue
lies 20 degrees outside the other tone's source family and 150 degrees from its own, so a
cue of the form "warm means tone 1" reads inverted after the shift.

What tells A from B is the same in both styles: A is coloured, B is grey, at a level drawn
per scene between 0.2 and 0.9, a range that holds the luminance of every tone, 0.23 to
0.76, so brightness alone does not tell them apart. The two shapes stay at least 2 pixels
apart, so neither hides the other, and their spacing counts either shape at a square's
extent, 1.2 times its radius, so the distance between the objects says nothing about
`round`. Both styles place the centres over the same range, 9 to 23 pixels.
`regrounding/scenes.py` holds both styles; `tests/test_scenes.py` pins each of these
properties on the drawn images.

## Rule

The verdict is the conjunction of the three concepts, in `regrounding/rule.py`. It is
never trained and never re-grounded. On concept logits it scores a scene by the log of the
product of the concept probabilities, the log-probability of the conjunction for independent
concepts, as the sum of their log-sigmoids, which keeps apart scenes whose product
underflows to zero; on concept prediction sets it is three-valued: true when every set is exactly {1}, false when some set is exactly
{0}, undecided otherwise.

## Model

`ConceptNet`: three 3-by-3 convolutions with two pooling stages, then a dense head with one
logit per concept. The features stay laid out in space up to the head because `left` is a
relation between two positions, which a global average would discard.

## Supervision

- **concepts**: each output is trained, with binary cross-entropy, as the concept it names.
- **verdict**: only the conjunction is trained, through the rule's log-score in float64. Nothing ties an output to the concept it is named after, and a model can
  get the verdict right through concepts that mean something else: a reasoning shortcut
  (Marconato et al. 2023). The conjunction is symmetric in its three inputs, so the
  outputs are also scored under the best matching of outputs to concepts.
- **mixed**: the concepts are learnt from their labels on the source, as under **concepts**,
  and every target fine-tuning is from verdicts through the rule, as under **verdict**: the
  one condition in which the fixed rule carries the target supervision to the concepts.
- **direct**: a network of the same architecture with one output, the verdict, trained
  with binary cross-entropy on the verdict alone, with no concepts and no rule, and
  measured on its verdicts only. At an equal number of target verdicts it is the baseline
  against which the mixed mode says what the fixed rule adds.

Every mode sees the same scenes: every scene of a seed is drawn from one generator in a
fixed order — source training, source calibration, source test, target pool, target test,
target training.

## Re-grounding

A copy of the source-trained model is fine-tuned on the first `k` scenes of the target pool,
`k` from 0 to 64, with the target supervision of its mode, the whole batch per step, for a
fixed number of steps. The pool is drawn in complementary pairs, the first a positive and
its complement: scene `2 i + 1` carries the negation of scene `2 i`'s concepts, so the first
`k` scenes hold both values of every concept at every even `k`, and a positive from
`k = 1`. The `k` scenes are nested: the scenes for `k = 4` are among those for `k = 8`.
`k = 0` is the source model unchanged. `regrounding_scope` says what is fine-tuned:

| scope | layers | weights |
|---|---|---|
| `readout` | the last layer, one logit per concept | 195 |
| `head` (default) | both dense layers, which read the spatial feature map | 131,331 |
| `all` | every layer, the convolutions included | 145,667 |

The head is where `left`, a relation between two positions, is read off the feature map, so
it is more than a concept readout and holds nine times the weights of the convolutions.

Two more models bound what re-grounding can reach. Both train on 3000 labelled target
scenes with the source's own training settings: one fine-tunes every weight of the source
model, the other a new model from scratch. They are the best the target style allows with
ample labels, from the source and without it.

## Measures

Every measure is a count on a test set of 1000 scenes, one row per model and test set:

| count | what it counts |
|---|---|
| `agreement` | 3 by 3: scenes where output `i` thresholded at one half equals concept `j` |
| `verdict_agreement` | per output, scenes where it equals the verdict |
| `verdict_correct`, `true_positives` | scenes where the rule on the thresholded outputs gives the true verdict, and how many of them are positive |
| `false_positives` | negatives whose log-score is at or above the highest threshold that reaches 90 % of positives |
| `covered`, `empty`, `full` | per concept, scenes whose conformal set holds the true value, holds neither value, holds both |
| `decided`, `decided_correct` | scenes where the three-valued verdict is decided, and decided right |

`tools/study.py` reads three matchings off `agreement`: the identity, its diagonal; the
best of the 6 permutations of outputs to concepts; and the best of the 48 that may also
flip an output, which credits an output that reads its concept inverted. The best
permutation exceeds the identity whenever some output agrees with another concept more than
with its own, which a constant output can do, so the excess does not by itself show
renamed concepts.
`false_positives` compares models at one sensitivity instead of at one threshold, which a
shift in calibration would move. A model that never says yes gets every negative right, 7
scenes in 8 on average; `tools/study.py` prints that rate per row as "always no", and
`verdict_correct` reads against it.

`tools/study.py` also prints, per seed, the first `k` at which a concept is right on 90 %
and on 95 % of the target test scenes, censored at the largest `k` of the study: the count
the question asks for.

## Conformal sets

Split conformal prediction, per concept: the nonconformity of a label is one minus the
probability the model gives it, and the threshold is the `ceil((n + 1)(1 - alpha))`-th
smallest nonconformity over `n` calibration scenes, with `alpha` 0.1. On a scene
exchangeable with the calibration scenes the set holds the true value with probability at
least 0.9 (Angelopoulos and Bates 2021). When the rank passes `n`, fewer than `1 / alpha - 1`
calibration scenes, the threshold is infinite and every set holds both values.

The calibration scenes are source scenes, for the source model and for every re-grounded
one: `k` target scenes are too few to calibrate on, since at `k = 12` and `alpha` 0.1 the
rank `ceil(13 × 0.9)` is the largest score. They carry concept labels, so under verdict
supervision the sets use labels the training never saw. Target scenes are not exchangeable
with source scenes, so the guarantee does not carry over, and `covered` on the target
measures what is left of it. No correction applies: the styles share no scene, so the
total-variation distance between them is 1 and the bound of Barber et al. 2023 is vacuous,
and likelihood-ratio weighting (Tibshirani et al. 2019) needs the target inside the
source's support. A set that holds both values counts as covering, and the nonconformity
score can leave a set empty; `full` and `empty` count both.

## Reproducibility

A published study runs on the CPU, one job per process and one thread per job. Before
torch is imported, `regrounding/reproducible.py` pins ATen to its portable kernels, MKL to
one code path and every thread pool to one thread; a CPU worker also hides the GPUs, since
torch's optimizers open a CUDA context on every step when one is visible.
`regrounding/execution.py` seeds Python, NumPy and torch, switches off oneDNN and NNPACK,
which pick their convolution kernels and blocking by CPU model, so that a convolution runs
as a matrix product on MKL, and makes torch refuse a nondeterministic operation. The
model's weights and the order of the training scenes come from one seeded generator. Two
runs on one machine give equal studies, which `tests/test_training.py` pins on a tiny
configuration. Whether another machine gives equal counts is untested: the profile removes
the kernel choices that depend on the CPU, and a training run turns any difference left in
the last bit into different counts.

## Limits

- Synthetic scenes, three binary concepts, one rule. A rule other than a conjunction, or
  concepts that are not independent, is not measured.
- Under concept supervision the rule enters only the measures: re-grounding trains each
  concept on its own label, so a target with another rule would give the same concept
  counts. The mixed mode passes the target supervision through the rule and the direct
  network learns the verdict without it; no condition varies the rule.
- Every `k` gets the same 150 steps at a rate of 0.001, so the count holds the optimiser's
  budget as well as the scenes.
- The source hue families and the scene layout were fixed after pilot runs scored on target
  test scenes of seeds 0 to 4 and 100 to 104; the study runs seeds 10 to 29.
- The shift is one fixed pair of styles. How the counts depend on which property shifts —
  palette, size, blur — is not separated.
- Re-grounding is plain fine-tuning. No method built for few-shot transfer is compared
  (Chen et al. 2019 survey the baselines).
- The conformal sets are calibrated on source scenes and are not corrected for the shift.
- The `k` scenes are one nested draw per seed. The per-seed range holds which scenes were
  drawn as well as how the model trained, and the two are not separated.
