# Static Flow-LatentBank MVTec AD1 DVT alpha sweep

## Negative Evidence Intake

This is a bounded diagnostic of the existing DVT position-mean component, not
a new method claim. AD2 evidence shows a positive mean gradient from alpha 0
to 1 but class-level no-harm failures; alpha above 1 degrades the H+ setting.
The risk under test is passive feature over-correction on low-shift MVTec AD1.

## Motivation

Determine whether the DVT strength that benefits distribution-shifted MVTec
AD2 progressively harms the accepted MVTec AD1 4-shot static baseline.

## Locked design

- Dataset: classic MVTec AD, all 15 categories and complete test split.
- Base: accepted 4-shot static Flow-LatentBank run, first four supports, seed 0.
- Encoder/preprocessing: frozen/eval `dinov2_vitb14_reg`, shorter-edge 448,
  layers `[2,5,8,11]`, `layer_norm_mean`.
- Flow/scoring: 3 epochs, 2 coupling layers, hidden multiplier 1, LR 2e-4,
  clamp 1.9, tail 0.3/top-k 0.05, lambda-logdet 1e-3, latent 1-NN weight
  1.0 plus density weight 0.25, top 1% image aggregation.
- Static memory: `expansion_budget=1.0`; no TTE.
- DVT mode: `position_mean`; alpha grid `{0,0.25,0.5,0.75,1.0}`.
- Excluded: context/register conditioning, foreground prior, morphology, score
  calibration, and support augmentation.
- Metrics: class-macro i-AUROC, i-AUPRC, p-AUROC, p-AUPRC, p-AUPRO@FPR0.30.
- Execution: dsba5 existing container; GPU 0 runs 0/0.5/1.0 and GPU 1 runs
  0.25/0.75 as independent processes.

## Baseline and gates

The exact no-DVT reference is
`flow_latentbank_mvtecad1_all15_shot4_vitb14reg_static_20260712_v1`:
`0.972100/0.987432/0.973747/0.581160/0.938160`.

- Validity gate: alpha 0 must reproduce no-DVT metrics within `1e-9`; every
  alpha must have 15 finite category rows, exact first-four supports, and
  static `4096 -> 4096` memory.
- AD1 retention gate: a fixed alpha is considered safe only if none of the
  five macro metrics loses more than 0.10 percentage point versus no-DVT.
- Mechanism interpretation: monotonic or accelerating degradation toward
  alpha 1 supports low-shift over-correction; neutral/improving results reject
  DVT as the explanation for the earlier AD1 discrepancy.
- Class no-harm is reported separately and is not hidden by macro retention.
- Same-condition SuperAD/VisionAD is unavailable; strict superiority remains
  `BLOCKED_BASELINE` regardless of the best diagnostic alpha.
