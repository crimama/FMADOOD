# AD2 Proposed Full-Normal Threshold Comparison — Preregistration

Date: 2026-07-13

## Motivation and claim boundary

The existing AD2 proposed result uses a fixed 16-image reference set and is
not data-budget comparable to SuperADD, which uses the full sorted
`train/good` population. This run changes only the normal-data budget and
threshold protocol. It applies SuperADD's stated 100,000-prototype k-NN
score-ranking selection to the proposed method's Flow latents, but does not
claim to reproduce SuperADD's upstream patching, augmentation, or binary
morphology.

## Frozen method

- MVTec AD2 single-image, all eight public objects and full
  `test_public/good,bad`.
- DINOv3 ViT-H+/16, layers `[7,15,23,31]`, dataset-native 672 resolution
  except `sheet_metal=448`.
- Flow 3 epochs, 2 couplings, hidden multiplier 1, static latent bank,
  latent 1-NN, LOO standardization, DVT position mean alpha 1.0, density
  weight 0.25, RGB guided-r8, and fixed line17/angles16 close/fill/erode.
- No TTE expansion, context/register routing, tiling, support transforms, or
  brightness augmentation.

## Full-normal and threshold contract

For each object, sort all `train/good` paths exactly as the dataset adapter
does. Paths with `index % 8 == 0` are threshold-only holdouts; the remaining
7/8 are the sole Flow-training and latent-bank inputs. The sets must be
disjoint and exhaustive.

The SuperADD-style threshold is computed after RGB guided-r8 from every pixel
of every threshold-holdout anomaly map:

`threshold = percentile(calibration pixels, 95) * 1.421`

Test binarization uses strict `score > threshold`, followed by the frozen
proposed morphology. In parallel, the same test maps are evaluated with the
existing TESTpublic raw-best oracle threshold, followed by the same
morphology. No TESTpublic label participates in the SuperADD-style threshold.

After Flow is trained on every patch from the 7/8 prototype images, each
latent's distances to its `k=100` global nearest neighbors are computed. One
global threshold `tau` is the mean of all those distances; each latent's score
is the number of its 100 distances below `tau`. Stable ascending score order
selects at most 100,000 latents. LOO standardization and test 1-NN both use
this frozen compact bank.

## Pre-result amendments

- Run `v1` matched the modulo-8 split but retained every latent. It OOMed on
  the 24 GB workers before producing any metric and was discarded.
- Run `v2` used the repository's older stochastic distance sampler. The user
  supplied the paper's explicit score-ranking definition while feature
  extraction was still running; `v2` produced no metric and was discarded.
- Run `v3` exposed a wiring error: selection was attached to the unused
  map-flow branch rather than the proposed fused FlowTTE branch. A stage-log
  assertion detected the missing 100k event; no v3 metric is retained.
- Run `v4` is the sole result-bearing run. It injects selection directly
  between Flow transformation and ReservoirMemory construction, asserts the
  resulting bank cap, and freezes the chunked, mathematically equivalent
  training-NLL calculation.

## Metrics and gates

- Primary outputs: class-macro `seg_AUROC_0.05`, calibrated raw/morph F1, and
  oracle raw/morph F1.
- Required audit: eight finite rows, exact 7/8+1/8 split, zero split overlap,
  static bank, identical maps for both threshold arms, and complete cleanup.
- Reported SuperADD `0.8393/0.6261` remains contextual because the proposed
  upstream method and morphology are intentionally unchanged.
- Verdict ceiling is `PROMISING_DIAGNOSTIC`; strict SuperADD superiority is
  `BLOCKED_BASELINE` without a same-code comparator.

## Remote allocation

- dsba3 A6000 GPUs 0–3: six objects in four queues.
- dsba5 TITAN RTX GPUs 0–1: one object per GPU after exact data/model/runtime
  synchronization.
- Dense anomaly and calibration maps are deleted only after per-object oracle
  and calibrated metrics are written.
