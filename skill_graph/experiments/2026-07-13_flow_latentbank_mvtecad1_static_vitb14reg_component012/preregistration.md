# MVTec AD1 static Flow-LatentBank component 0/1/2 diagnostic

Date sealed: 2026-07-13

## Motivation and negative evidence

The accepted 4-shot DVT-off static setting is strong on MVTec AD1, while every
nonzero DVT alpha failed the all-five retention gate. This diagnostic therefore
keeps DVT excluded and tests three non-DVT components independently. It is an
ablation/transfer diagnostic, not a new method claim.

## Locked shared contract

- Dataset: classic MVTec AD, all 15 classes and full test split.
- Supports: first four `train/good` paths, seed 0, identity only.
- Encoder: frozen/eval `dinov2_vitb14_reg`, shorter-edge 448, layers
  `[2,5,8,11]`, `layer_norm_mean`.
- Flow: three epochs, two couplings, hidden multiplier 1, LR 2e-4, clamp 1.9,
  tail 0.3/top-k 0.05, lambda-logdet 1e-3.
- Static memory: `expansion_budget=1.0`; DVT mode `none`; no TTE.
- Score: latent 1-NN distance 1.0, density quantile 0.90, top-1% image score.
- Excluded: register conditioning, foreground prior, morphology, calibration,
  support augmentation, resolution changes, and score-field quantile matching.
- Metrics: class-macro i-AUROC, i-AUPRC, p-AUROC, p-AUPRC, p-AUPRO@FPR0.30.

Accepted reference:
`0.972100416682/0.987431867365/0.973747446630/0.581160338468/0.938160366361`.

## Independent arms

0. `density0`: change only `density_weight 0.25 -> 0.0`.
1. `cls_soft_w10`: keep the unconditional Flow and change only memory-distance
   retrieval by adding frozen CLS cosine soft penalty, weight 10. The choice is
   the previously best AD2 component value; no AD1 weight sweep is permitted.
2. `guided_r8`: regenerate the unchanged reference maps, then apply only the
   frozen RGB grayscale guided filter at half-native scale, radius 8 and
   epsilon 0.01. Recompute all five metrics from the refined maps.

The guided identity regeneration must reproduce the accepted reference within
`1e-9`; otherwise arm 2 is invalid. Each arm must retain exact supports and
static `4096 -> 4096` memory.

## Gates

- Retention: no requested macro metric may lose more than 0.10 percentage
  point versus the accepted reference.
- Positive diagnostic: retention passes and either p-AUPRC or p-AUPRO gains at
  least 0.20 percentage point.
- Class no-harm is reported independently; a macro win dominated by one class
  is not promoted.
- No same-condition VisionAD artifact is available, so strict external
  superiority remains `BLOCKED_BASELINE`.

## Execution

- dsba5 GPU 0: density0; GPU 1: CLS soft w10.
- Initially scheduled on dsba3 GPUs 0/1/2/3 after the existing grid-shift run.
  The user redirected this arm to dsba5 after arms 0/1 completed, so it runs as
  two disjoint class shards on dsba5 GPUs 0/1. This is an execution-allocation
  amendment only; the locked method, classes, maps, and metrics are unchanged.
- Chunk aggregation is over the 15 per-class rows, never over chunk means.
- Dense maps are deleted after metric and manifest generation.
