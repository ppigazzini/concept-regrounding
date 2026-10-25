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

| style | tone 0 | tone 1 | radius of A | blur |
|---|---|---|---|---|
| source | blue | red | 4 to 6 | no |
| target | yellow | teal | 3 to 7 | 3 by 3 mean |

Tone 1 is the warm colour on the source and the cool one on the target, so a cue of the
form "warm means tone 1" reads inverted after the shift.

What tells A from B is the same in both styles: A is coloured, B is grey. B's grey level is
drawn per scene between 0.2 and 0.9, a range that holds the luminance of every tone, 0.28
on the source to 0.78 on the target. At one fixed grey of 0.6, A would be the darker object
on the source and the brighter one on the target, and a model that finds A by brightness
would lose it after the shift. The two shapes stay at least 2 pixels apart, so neither
hides the other, and both styles place their centres over the same range, 9 to 23 pixels.
`regrounding/scenes.py` holds both styles; `tests/test_scenes.py` pins each of these
properties on the drawn images.

## Rule

The verdict is the conjunction of the three concepts, in `regrounding/rule.py`. It is
never trained and never re-grounded. On concept probabilities it scores a scene by their
product, the probability of the conjunction for independent concepts; on concept prediction
sets it is three-valued: true when every set is exactly {1}, false when some set is exactly
{0}, undecided otherwise.

## Model

`ConceptNet`: three 3-by-3 convolutions with two pooling stages, then a dense head with one
logit per concept. The features stay laid out in space up to the head because `left` is a
relation between two positions, which a global average would discard.

## Supervision

- **concepts**: each output is trained, with binary cross-entropy, as the concept it names.
- **verdict**: only the conjunction is trained, through the product of the three
  probabilities. Nothing ties an output to the concept it is named after, and a model can
  get the verdict right through concepts that mean something else: a reasoning shortcut
  (Marconato et al. 2023). The conjunction is symmetric in its three inputs, so the
  outputs are also scored under the best matching of outputs to concepts.

Both modes see the same scenes: every scene of a seed is drawn from one generator in a fixed
order — source training, source calibration, source test, target pool, target test.

## Re-grounding

A copy of the source-trained model is fine-tuned on the first `k` scenes of the target pool,
with the supervision it was trained with, the whole batch per step, for a fixed number of
steps. The `k` scenes are nested: the scenes for `k = 4` are among those for `k = 6`. `k = 0`
is the source model unchanged. `regrounding_scope` says what is fine-tuned:

| scope | layers | weights |
|---|---|---|
| `readout` | the last layer, one logit per concept | 195 |
| `head` (default) | both dense layers, which read the spatial feature map | 131,331 |
| `all` | every layer, the convolutions included | 145,667 |

The head is where `left`, a relation between two positions, is read off the feature map, so
it is more than a concept readout and holds nine times the weights of the convolutions.

## Measures

Every measure is a count on a test set of 1000 scenes, one row per model and test set:

| count | what it counts |
|---|---|
| `concept_correct` | per concept, scenes where output `i` thresholded at one half equals concept `i` |
| `permuted_correct` | the same summed over the three concepts, under the best of the 6 matchings |
| `verdict_correct` | scenes where the rule on the thresholded outputs gives the true verdict |
| `false_positives` | negatives scored at or above the highest product threshold that reaches 90 % of positives |
| `covered` | per concept, scenes whose conformal set holds the true value |
| `decided`, `decided_correct` | scenes where the three-valued verdict is decided, and decided right |

`permuted_correct` above the sum of `concept_correct` says the outputs carry the concepts
under the wrong names. `false_positives` compares models at one sensitivity instead of at
one threshold, which a shift in calibration would move. `verdict_correct` is bounded below
by the base rate: a model that never says yes gets the 7 negatives in 8 right, 87.5 %.

## Conformal sets

Split conformal prediction, per concept: the nonconformity of a label is one minus the
probability the model gives it, and the threshold is the `ceil((n + 1)(1 - alpha))`-th
smallest nonconformity over `n` calibration scenes, with `alpha` 0.1. On a scene
exchangeable with the calibration scenes the set holds the true value with probability at
least 0.9 (Angelopoulos and Bates 2021).

The calibration scenes are source scenes, for the source model and for every re-grounded
one: `k` target scenes are too few to calibrate on, since at `k = 12` and `alpha` 0.1 the
rank `ceil(13 × 0.9)` is the largest score. Target scenes are not exchangeable with source
scenes, so the guarantee does not carry over, and `covered` on the target measures how much
of it is lost. Weighting the calibration scores by the likelihood ratio between the styles
(Tibshirani et al. 2019) is the known correction; it is not implemented.

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
- The shift is one fixed pair of styles. How the counts depend on which property shifts —
  palette, size, blur — is not separated.
- Re-grounding is plain fine-tuning. No method built for few-shot transfer is compared
  (Chen et al. 2019 survey the baselines).
- The conformal sets are calibrated on source scenes and are not corrected for the shift.
- The `k` scenes are the first `k` of one pool per seed, not stratified. Under verdict
  supervision a scene carries one bit and 1 in 8 is positive: the first `k` hold no
  positive with probability 0.88, 0.77, 0.59, 0.45 and 0.20 for `k` = 1, 2, 4, 6, 12.
  Under concept supervision some concept takes one value in all `k` scenes with
  probability 0.88 at `k = 2`, 0.33 at `k = 4` and 0.09 at `k = 6`. The per-seed range at
  small `k` holds which scenes were drawn as well as how the model trained.
