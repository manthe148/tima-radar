# Validation methodology

The point of this document is that the **per-subtype breakdown is the real signal**,
not the aggregate ROC. Tornado detection is easy to make *look* good with a leaky
split or a pile of trivial negatives; the numbers below are reported so that a
reader can see exactly where the model is strong and where it pays a cost.

## Split

Every box is tagged with its **convective day** (`conv_day`). The train/val
partition is by conv_day (hash-based, ~20% to val), so no storm-day ever appears
on both sides — a storm cannot be memorized in training and "recognized" in
validation. Negatives are additionally mined ≥50 km from any tornado report and
kept off the tornadic couplet.

Dataset: **1031 tornado events across 261 convective days → 4386 boxes.**

| split | tornadic | strong non-tornadic | clear-air | non-rotating precip |
|---|---|---|---|---|
| train | 880 | 1753 | 656 | 418 |
| val   | 151 | 302  | 144 | 82  |

## The four sample types

- **tornadic (positive)** — storm-centered box at a confirmed NCEI tornado report.
- **strong non-tornadic storm (hard negative)** — a ≥45 dBZ cell on the same day,
  ≥50 km from any tornado. The hard case: separate tornadic rotation from ordinary
  strong storms.
- **clear-air** — a storm-free box (often entirely no-data) sampled on an *active
  severe-weather day*. Teaches "no storm → not tornadic."
- **non-rotating precip** — moderate-to-strong reflectivity with weak azimuthal
  shear. Forces the model to key on *rotation*, not raw reflectivity intensity.

## Results — held-out validation, model `mrms_v8_jitter_ca`

| sample type | n | median P(tor) | rate @ 0.5 |
|---|---|---|---|
| tornadic (positive) | 151 | 0.906 | recall **0.901** |
| strong non-tornadic storm | 302 | 0.157 | FAR 0.268 |
| clear-air | 144 | 0.014 | FAR **0.000** |
| non-rotating precip | 82 | 0.019 | FAR **0.000** |

Synthetic probes:

| probe | P(tor) | expected |
|---|---|---|
| empty box (all no-data, i.e. an off-storm click) | **0.093** | low |
| strong centered couplet | **0.948** | high |

Aggregate: ROC-AUC 0.944 · PR-AUC 0.803 · best CSI 0.621 @ 0.75.

## Reading the numbers honestly

- **The aggregate ROC (0.944) is not the headline.** Clear-air and precip
  negatives are trivially separable, so they flatter ROC/PR relative to a
  positive-vs-hard-negative-only set. Use the per-subtype table as the real signal.
- **Recall (0.90) holds** even after adding 1300 negatives, because the training
  loss auto-upweights positives as the negative pool grows
  (`pos_weight = neg/pos ≈ 3.2`).
- **The honest cost is hard-negative FAR (0.27).** Adding clear-air negatives made
  the model marginally more trigger-happy on rotating-but-non-tornadic storms
  (FAR ≈ 0.22 → 0.27, median 0.09 → 0.16). That is an accepted trade for killing
  the storm-free false alarm — empty boxes went from ~0.90 to **0.09**. The lever
  to tighten hard-negative FAR is `pos_weight`, not more data.
- **An earlier model reported ROC 0.985** — but it was exploiting a *centering
  shortcut*: the couplet always sat at box-center, so the model learned position
  instead of structure. Jitter augmentation (±12 px) removed that crutch and the
  honest number fell to 0.919 before the clear-air work. We report the lower,
  real numbers throughout.

## Reproduce

```bash
cd training
python validate_clearair.py --ckpt ../models/mrms_v8_jitter_ca.pt
```

Prints the per-subtype table above plus the empty-box and strong-couplet probes.
