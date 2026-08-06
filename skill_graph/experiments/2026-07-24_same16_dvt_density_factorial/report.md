# Same16 DVT × Density Factorial Attribution

Date: 2026-07-24  
Verdict: `MIXED_OR_INCONCLUSIVE_ATTRIBUTION / KILL_FOR_CLAIM`

## Negative Evidence Intake

The completed Same16 Basic-to-A6 comparison removed support-fitted
position-mean DVT and the density penalty together. Historical runs could not
separate those factors because their support, backbone, population, or metric
contracts differed. This diagnostic therefore reuses the exact completed v6
Basic/A6 cells and runs only the two missing cells.

## Motivation

Determine whether Basic's macro advantage over A6 comes from DVT, density, or
a DVT×density interaction without reopening a broad component sweep.

## Implementable Design

The exact 2×2 cells are:

| Arm | DVT | Density weight |
|---|---|---:|
| A6 | off | 0.00 |
| DVT-only | position mean, alpha 1.0 | 0.00 |
| Density-only | off | 0.25 |
| Basic | position mean, alpha 1.0 | 0.25 |

All cells share DINOv2 ViT-L/14 layers `[5,11,17,23]`,
`layer_norm_mean`, the same Flow/fused detector, the same fixed 16 supports,
static memory, and raw postprocessing.

## Evaluation Alignment

The diagnostic uses all eight MVTec AD2 objects and the full
`test_public/good,bad` population. The support manifest SHA-256 is
`7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
Metrics are original-resolution standardized pixel pAUROC@0.05, pooled raw
oracle pixel F1, and connected-component AU-PRO@0.05. Effects use 2,000
deterministic object bootstraps at seed 0.

## Code Modification / Creation

- Added a fixed-container remote runner that executes only DVT-only and
  density-only, verifies the completed v6 source cells, and removes dense maps
  only after aggregation.
- Added a fail-closed factorial aggregator that validates every arm's support,
  configuration, metric schema, and eight-object coverage.
- Added focused aggregator and remote-runner regression tests.

## Added Code Evaluation

The dsba3 preflight passed Python compilation, shell syntax, DINOv2 offline
cache validation, four-GPU idle checks, and 21 focused tests. Aggregation
accepted exactly 16 new metrics and 16 new manifests. No evaluation log
contains an error or traceback.

## Remote Execution

Run `same16_dvt_density_factorial_20260724_v1` completed in fixed container
`hun_fsad_tta_012`.

Remote compact root:
`/var/tmp/fsad_tta/results_remote/same16_dvt_density_factorial_20260724_v1`

Local compact pull:
`results/remote_runs/dsba3/same16_dvt_density_factorial_20260724_v1`

The local and remote summary JSON SHA-256 values both equal
`ba78a63a562d0e957a1e78e9b05310f8d057937429221547b9c8d956f63b6f16`.

## Unified Metrics

| Arm | pAUROC@0.05 | Raw F1 | AU-PRO@0.05 |
|---|---:|---:|---:|
| A6 | 0.759283906 | 0.350215920 | 0.538267291 |
| DVT-only | 0.771822410 | 0.356363126 | 0.530337672 |
| Density-only | 0.760328114 | 0.350909687 | 0.543297936 |
| Basic | 0.772635045 | 0.355675500 | 0.533407769 |
| SuperAD-16 | 0.761857074 | 0.368423110 | 0.534924081 |
| RN-FMLK | 0.745312086 | 0.331950199 | 0.425132995 |

## Factorial Results

| Effect | pAUROC@0.05 | Raw F1 | AU-PRO@0.05 |
|---|---:|---:|---:|
| DVT at density 0 | +0.012538504 | +0.006147207 | -0.007929619 |
| DVT at density 0.25 | +0.012306931 | +0.004765813 | -0.009890167 |
| Density without DVT | +0.001044208 | +0.000693767 | +0.005030645 |
| Density with DVT | +0.000812636 | -0.000687626 | +0.003070097 |
| DVT average main effect | +0.012422717 | +0.005456510 | -0.008909893 |
| Density average main effect | +0.000928422 | +0.000003070 | +0.004050371 |
| Interaction | -0.000231572 | -0.001381394 | -0.001960548 |

DVT's pAUROC main effect exceeds the material `0.005` bound, but its
object-bootstrap 95% interval is `[-0.001024612, 0.033642027]`; it therefore
does not receive factor credit. Its F1 interval
`[-0.011821426, 0.022734445]` also crosses zero. Its AU-PRO effect is negative,
with interval `[-0.022564111, 0.002732938]`.

Density has narrow positive intervals for pAUROC
`[0.000115606, 0.001764395]` and AU-PRO
`[0.001256668, 0.007674860]`, but both effects remain below their frozen
material bounds. Its F1 effect is effectively zero.

The interaction is small on every metric. F1 and AU-PRO intervals exclude
zero, but their magnitudes (`0.001381394` and `0.001960548`) are far below the
`0.010` material bounds, so interaction credit is also rejected.

## Why Basic Degrades Under Other Attempts

The factorial closes the Basic-to-A6 attribution:

- Almost all of Basic's `+0.013351139` pAUROC and `+0.005459580` F1 over A6
  comes from the DVT main effect, not density.
- The effect is not broad. `sheet_metal` contributes
  `+0.084306` pAUROC and `+0.048598` F1 but loses `0.051259` AU-PRO.
  `fruit_jelly` and `wallplugs` also gain ranking/F1, whereas `walnuts` loses
  `0.041365` F1, `can` loses pAUROC/AU-PRO, and `vial` loses `0.017859`
  AU-PRO.
- Density is a weak calibration term: it adds only about `0.00405` AU-PRO on
  average and essentially zero F1. Removing it cannot explain the main Basic
  regression.
- Basic is therefore a fragile macro compromise. DVT can improve pixel
  ordering for a few objects while degrading connected-region coherence or
  threshold behavior for others. This ranking-versus-region explanation is
  an inference from the opposing pAUROC/AU-PRO signs, not a direct causal
  measurement of feature semantics.
- The larger losses in other confirmed variants arise from removing useful
  detector structure in addition to these factors. A6 still beats raw fused
  A7 by `+0.040891` pAUROC, `+0.054212` F1, and `+0.038498` AU-PRO, supporting
  the Flow transform. Replacing the four-layer fusion with L17-only reduces A6
  by `0.036660` pAUROC, `0.059900` F1, and `0.082659` AU-PRO. Those failures
  are not density effects.

The practical conclusion is not that every Basic component is independently
beneficial. The retained Basic score is produced by Flow plus multi-layer
fusion and a heterogeneous DVT ranking effect; density supplies only a small
AU-PRO offset. Added routing, single-layer, or calibration mechanisms tend to
redistribute an already fragile score tail without adding stable defect
evidence, so macro or per-object collapse is common.

## Decision

`MIXED_OR_INCONCLUSIVE_ATTRIBUTION`. DVT, density, and interaction all fail the
frozen attribution-credit gates. Basic remains the retained descriptive arm,
but it still fails the strict same-condition SuperAD claim because it is
`-0.012747610` in F1 and `-0.001516312` in AU-PRO despite
`+0.010777972` pAUROC.

This is diagnostic-only public-shadow evidence. No shot scaling, density/DVT
sweep, method promotion, or private-test claim is authorized.

## Post-Conclusion Storage Cleanup

All anomaly/calibration map directories were removed after validation and
aggregation. The local compact pull contains zero dense-map directories, and
the isolated remote source snapshot was removed. Compact metrics, manifests,
logs, summary artifacts, completion marker, and cleanup evidence were retained.
