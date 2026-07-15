# Static Flow-LatentBank MVTec AD1 DVT-alpha diagnostic

Date: 2026-07-13  
Verdict: `KILL_FIXED_DVT_FOR_CLAIM / CONTINUE_DIAGNOSTIC`

## 1. Negative Evidence Intake

This is a bounded mechanism diagnostic, not a new method claim. Prior MVTec
AD2 runs showed a positive mean gradient from DVT alpha 0 to 1, while alpha
above 1 degraded the H+ setting and class-level no-harm already failed. The
specific risk tested here is that support position-mean subtraction corrects a
real distribution shift in AD2 but over-corrects low-shift MVTec AD1.

## 2. Motivation

Starting from the accepted MVTec AD1 4-shot static Flow-LatentBank result,
change only DVT strength and determine whether it explains the apparent
AD1/AD2 conflict.

## 3. Implementable Design

- Dataset: classic MVTec AD, all 15 classes and complete test split.
- Supports: first four `train/good` images, seed 0, identity transform.
- Encoder: frozen/eval `dinov2_vitb14_reg`, shorter-edge 448; layers
  `[2,5,8,11]` fused by `layer_norm_mean`.
- Flow: three epochs, two coupling layers, hidden multiplier 1, LR 2e-4,
  clamp 1.9, tail 0.3/top-k 0.05, lambda-logdet 1e-3.
- Score: latent 1-NN weight 1.0 plus NF density weight 0.25, density quantile
  0.90, and image top-1% aggregation.
- Memory: fixed `4096 -> 4096`; no TTE.
- DVT: support position-mean artifact subtraction with alpha
  `{0,0.25,0.5,0.75,1.0}`.
- Excluded: context/register conditioning, foreground prior, morphology,
  calibration, and support augmentation.
- Metrics: class-macro i-AUROC, i-AUPRC, p-AUROC, p-AUPRC, and exact
  p-AUPRO@FPR0.30.

## 4. Evaluation Alignment

The five arms use identical data, supports, encoder, preprocessing, features,
flow, memory, scorer, and evaluator. Alpha is the only intended factor. Alpha
0 had to reproduce the accepted no-DVT run within `1e-9`. A nonzero alpha was
pre-registered as AD1-safe only if none of the five macro metrics lost more
than 0.10 percentage point. Class-wise wins and losses were audited separately.

The earlier AD2 alpha sweeps use different backbones, support budgets, and an
AD2 evaluator. They provide directional context only; their absolute values
are not a unified benchmark comparison.

## 5. Code Modification / Creation

- `scripts/run_flow_latentbank_mvtecad1_static_vitb14reg_dvt_alpha_sweep.sh`:
  resumable two-GPU launcher, shared preflight, alpha-isolated outputs,
  aggregate JSON, and dense-map cleanup evidence.
- `tests/test_denoising.py`: exact alpha-zero identity regression test.
- `preregistration.md`: locked design, validity, retention, and mechanism
  interpretation before remote execution.

No method implementation was changed.

## 6. Added Code Evaluation

- Focused DVT/backbone/classic-adapter tests: 28 passed.
- Full local regression suite: 419 passed with five pre-existing PyTorch
  transformer warnings.
- Launcher shell syntax: passed.
- Affected Python compilation: passed.
- Repository whitespace/error check: passed.
- Alpha-zero output reproduced all five accepted reference metrics exactly.

## 7. Remote Execution

- Host/container: dsba5, existing `hun_fsad_tta_012` container.
- GPUs: GPU 0 ran alpha 0, 0.5, and 1.0; GPU 1 ran 0.25 and 0.75.
- Preflight: 15 classes, layers `(2,5,8,11)`, 32x32 feature grid, four
  1024x768 feature tensors, frozen/eval encoder.
- Result root:
  `results/remote_runs/dsba5/flow_latentbank_mvtecad1_shot4_vitb14reg_static_dvt_alpha_20260713_v1`.

The audit passed over 75 class-runs: all metrics were finite; standalone,
seed, manifest, and summary copies agreed; supports were exactly 000--003;
every bank stayed `4096 -> 4096`; logs contained no traceback, OOM, NaN, or
Inf; and no dense anomaly-map directory remains.

## 8. Unified Metrics and Analysis

Values and deltas are class-macro percentage points. Deltas are relative to
alpha 0; bold marks the best value in each metric.

| Alpha | i-AUROC | i-AUPRC | p-AUROC | p-AUPRC | p-AUPRO | Retention |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 0.00 | **97.2100** | **98.7432** | 97.3747 | 58.1160 | 93.8160 | pass |
| 0.25 | 97.0864 (-0.1237) | 98.7126 (-0.0306) | 97.4769 (+0.1022) | 58.2196 (+0.1036) | 93.8906 (+0.0746) | fail |
| 0.50 | 96.7065 (-0.5036) | 98.5412 (-0.2020) | **97.5438 (+0.1690)** | **58.3405 (+0.2245)** | **94.2154 (+0.3994)** | fail |
| 0.75 | 95.7073 (-1.5028) | 97.9266 (-0.8166) | 97.4210 (+0.0462) | 57.9231 (-0.1929) | 93.9917 (+0.1757) | fail |
| 1.00 | 94.5796 (-2.6304) | 97.1261 (-1.6171) | 97.1998 (-0.1750) | 56.9063 (-1.2098) | 93.4684 (-0.3476) | fail |

No nonzero alpha passes the pre-registered all-five retention gate. Image
performance falls progressively, with alpha 1 losing 2.63 i-AUROC and 1.62
i-AUPRC points. Pixel metrics form a shallow optimum at alpha 0.5, but this is
not a broad class-wise improvement: relative to alpha 0, alpha 0.5 improves /
degrades p-AUROC on 5/10 classes, p-AUPRC on 6/9, and p-AUPRO on 5/10.

The alpha-0.5 macro localization gain is strongly concentrated in
`transistor`: +3.87 p-AUROC, +12.30 p-AUPRC, and +11.74 p-AUPRO points.
Meanwhile `cable` loses 3.64 i-AUROC, 3.14 p-AUPRC, and 1.87 p-AUPRO;
`screw` loses 3.09 p-AUPRC and 1.12 p-AUPRO. At alpha 1, `transistor` rises
even more in localization, but `cable` loses 18.46 i-AUROC and 11.78 p-AUPRC,
and `screw` loses 7.19 i-AUROC and 10.78 p-AUPRC. Thus the mean pixel gain at
moderate alpha is dominated by a narrow rescue rather than class-level
no-harm.

Prior AD2 directional evidence moved the other way: in the DINOv3-L sweep,
seg-AUROC@0.05 rose from 0.797743 at alpha 0 to 0.825207 at alpha 1, and the
H+ extension peaked around alpha 1 before declining above it. Because those
runs are not same-backbone comparisons, the present experiment supports DVT
as a contributing factor to the AD1/AD2 tension, but it does not isolate
dataset shift from backbone or support-budget interactions.

## 9. Continuation Assessment and Conclusion

Verdict: `KILL_FIXED_DVT_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.

A single nonzero DVT alpha should not be the universal method setting. Alpha
0 is the only AD1-safe arm; alpha 0.5 offers the best mean localization but
fails image retention and harms most classes. Strong DVT clearly over-corrects
AD1, while prior AD2 evidence indicates that it can help under shift.

This makes a shift-aware gate or adaptive alpha scientifically plausible, but
not yet claim-ready. Before implementing such a selector, the next decisive
experiment should repeat a compact `{0,0.5,1}` comparison on AD1 and AD2 with
the same backbone, preprocessing, support budget, flow, and scoring contract.
That experiment separates dataset-shift behavior from the currently
confounded backbone/protocol difference.

Strict VisionAD/SuperAD superiority remains `BLOCKED_BASELINE` because no
same-condition external baseline is included in this diagnostic.

## Post-Conclusion Storage Cleanup

Only compact metrics, manifests, logs, summaries, and cleanup evidence were
retained. Local and remote audits found zero dense anomaly-map directories.
