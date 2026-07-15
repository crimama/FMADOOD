# FlowTTE Transformer Flow Failure Image Analysis

Date: 2026-07-10

## Purpose

Diagnose whether `normality_mode=transformer_flow` fails because of its flow
architecture rather than unrelated settings. The comparison below keeps the
pipeline fixed and changes only the flow head:

```text
DINOv3-H+/16 layers [7,15,23,31]
DVT position_mean alpha=1.0
fixed SuperAD-16 support JSON
no-TTE, expansion_budget=1.0
context/register disabled
density_weight=0.0
baseline: normality_mode=fused
candidate: normality_mode=transformer_flow
```

## Artifacts

- Baseline metrics/logs:
  `results/remote_runs/dsba3/flowtte_mlp_dw0_maps_all8_20260710_v1`
- Transformer metrics/logs:
  `results/remote_runs/dsba3/flowtte_transformer_dw0_maps_all8_20260710_v1`
- Per-image analysis and panels:
  `results/remote_runs/dsba3/flowtte_transformer_failure_map_analysis_20260710_v1`
- Analysis script:
  `scripts/analyze_flowtte_transformer_failure_maps.py`

Dense anomaly maps were regenerated for analysis and then removed from the
remote and local run roots. Local map count after pullback is `0`.

## Architecture-Only Result

| class | MLP AUROC | TF AUROC | Δ AUROC | MLP F1 | TF F1 | Δ F1 | Δ gap-z bad | Δ top1 GT | Δ comp | worse gap % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| can | 0.5618 | 0.5501 | -0.0117 | 0.0008 | 0.0006 | -0.0003 | -0.156 | -0.000 | +1.82 | 73.3% |
| fabric | 0.9606 | 0.9700 | +0.0094 | 0.6657 | 0.7032 | +0.0375 | -0.146 | +0.000 | -4.73 | 74.4% |
| fruit_jelly | 0.7865 | 0.7199 | -0.0665 | 0.4895 | 0.3008 | -0.1887 | -0.313 | -0.128 | +4.33 | 86.7% |
| rice | 0.9501 | 0.9438 | -0.0063 | 0.7122 | 0.6985 | -0.0137 | -0.300 | -0.002 | -0.25 | 75.6% |
| vial | 0.7503 | 0.7869 | +0.0366 | 0.4344 | 0.4813 | +0.0469 | -0.306 | -0.088 | +2.27 | 82.9% |
| wallplugs | 0.9050 | 0.8912 | -0.0138 | 0.6143 | 0.5797 | -0.0346 | -0.042 | -0.004 | -0.41 | 54.4% |
| walnuts | 0.8908 | 0.8745 | -0.0162 | 0.7267 | 0.6709 | -0.0558 | -0.348 | -0.014 | +0.77 | 84.4% |
| sheet_metal | 0.8951 | 0.8924 | -0.0027 | 0.5475 | 0.5830 | +0.0355 | -0.304 | -0.006 | -2.77 | 86.7% |

Mean:

- MLP dw0: `seg_AUROC_0.05=0.837518`, `seg_F1=0.523904`
- Transformer dw0: `seg_AUROC_0.05=0.828600`, `seg_F1=0.502237`
- Delta: `-0.008918` AUROC, `-0.021666` F1

## What The Image Analysis Shows

`Δ gap-z bad` is the change in normalized score separation between GT anomaly
pixels and non-GT pixels on bad images. A negative value means Transformer Flow
made local anomaly contrast weaker.

Main observations:

1. Transformer Flow reduces continuous anomaly contrast almost everywhere.
   Every class has negative mean `Δ gap-z bad`, and most bad images have worse
   GT-vs-background gap. This includes classes where F1 improves.
2. The strongest structural failure is `fruit_jelly`: `-0.313` gap-z,
   `-0.128` top-1% GT share, `-0.187` per-image IoU, and `+4.33`
   components. This is true across regular, exposure, and shift variants.
3. `vial` and `sheet_metal` improve global F1, but their bad-image contrast
   still worsens (`vial -0.306`, `sheet_metal -0.304`). Representative panels
   show Transformer raises object/background structure together rather than
   isolating the defect.
4. `fabric` is the cleanest positive case. Its map shape is similar to MLP,
   while thresholded components decrease by `-4.73`. This looks like
   smoothing/regularization of the score field, not improved anomaly ranking.
5. `can` remains collapsed in both settings; Transformer adds components and
   area but does not recover meaningful localization.

## Structural Diagnosis

The Transformer conditioner is not simply "more context is better." In this
few-shot latent-bank setup it appears to mix patch tokens into a more globally
consistent field before memory distance. That regularizes some fragmented
classes, but it also suppresses precisely the local contrast that defect
localization needs.

Failure mode:

```text
patch-local MLP flow:
  noisy but preserves sharp local defect contrast

Transformer flow:
  patch-token interaction smooths/co-normalizes the map
  -> weaker GT-vs-non-GT score gap
  -> defect evidence diffuses into object/global structure
  -> class-dependent threshold/F1 behavior
```

This explains why smoke results looked positive on `vial` and `fabric`, while
all-eight performance fell. The gains are not robust structural anomaly
separation; they are mostly score-field regularization effects that can help
fragmented masks but harm true local ranking.

## Verdict

`KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`

Do not present Transformer Flow as a method improvement. Keep it only as
evidence that patch-token interaction must be constrained. The next structural
branch should not be a stronger global Transformer flow. It should preserve
local patch contrast and add context only as a routing/calibration signal.

Recommended next direction:

```text
local patch flow or raw latent score
+ class-agnostic object/foreground prior
+ support-stat score calibration
+ limited neighborhood-aware distance
```

Avoid:

```text
unconstrained all-token Transformer conditioning before memory distance
```
