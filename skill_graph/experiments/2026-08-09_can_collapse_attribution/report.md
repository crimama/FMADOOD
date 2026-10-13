# `can` Collapse Attribution Report

Date: 2026-08-09

Protocol: `can-collapse-attribution-all8-20260809`

Verdict: `INCONCLUSIVE` (all three preregistered hypotheses rejected)

Scope: mechanism attribution diagnostic. `superad_seg_AUROC_0.05=N/A`,
`superad_seg_F1=N/A`, `delta_vs_superad=N/A`, `comparable=false`.

## Negative Evidence Intake

Eleven module-level branches had been closed under an all-eight macro gate. The
per-object decomposition showed the macro deficit is not diffuse: `can` alone
carries `79%` of the public F1 gap to SuperADD. This protocol asked why, under a
frozen three-way attribution rule fixed before execution.

## Implementable Design

Executed as preregistered, with one custody-driven deviation recorded below.

- Block A1 support ladder `S16 ⊂ S64 ⊂ S128` (24 runs, local RTX 4090)
- Block A2 per-layer representation probe, DVT off / Flow identity, layers
  `7,15,23,31,fused` (40 runs, dsba5 GPU 3 immutable snapshot)
- Block A3 ordered stage path `pre_dvt_feature -> post_dvt_feature ->
  flow_latent -> deployed_final` (8 objects, local)
- Parity duplicate: A2 `fused` rerun locally against the dsba5 arm

## Custody Deviation

The preregistration allowed reusing the 2026-08-08 stage-path artifacts where
custody passes. Custody **failed**: those artifacts were produced with
`binary_postprocess=closefill_erode`, while this protocol's contract fixes
`none`. Per the preregistration's own rule ("any stage/object cell whose custody
check fails is rerun rather than reused"), all eight A3 objects were rerun
under the correct contract. No reused cell entered the result.

The preregistration also originally cited support manifest SHA-256
`7e93f15e…`. That hash belongs to the DINOv2-L `same16` line and is a different
image set. The DINOv3-H+ Baseline1-core line — used by the official private
submission and by the `raw`/`position_current`/`position_lowpass` anchors — is
`dinov3_noctx_support_paths.json`, SHA-256 `9f223f7c…`. The preregistration was
corrected before execution and the corrected manifest was used throughout.
Independently verified: hash match, `S16` equals the frozen manifest on all
eight objects, `S16 ⊂ S64 ⊂ S128`, and all 1,664 ladder members lie in
`train/good` with zero test/validation/private intersection.

## Validity

All seven gates passed. Local/dsba5 parity on the duplicated A2 `fused` arm:
custody pass, maximum shared-metric delta well below the `1e-3` bound
(largest observed `~3.6e-8`).

## Results

### Frozen attribution rule, evaluated in the fixed order

| Order | Hypothesis | Statistic | Threshold | Value | Fires |
| --- | --- | --- | --- | ---: | --- |
| 1 | H-B representation deficiency | `can` best-layer pixel AUROC | `< 0.60` | `0.8442` | no |
| 2 | H-C pipeline signal loss | best-layer minus deployed | `>= 0.10` | `+0.0130` | no |
| 3 | H-A support insufficiency | `can` S128 minus S16 | `>= 0.10` | `-0.0034` | no |

Seven of seven control objects reached `best_layer_auroc >= 0.75`, so the H-B
control condition was satisfied while the `can` condition was not.

### Block A1 — support ladder (full-range pixel AUROC)

| object | S16 | S64 | S128 | Δ |
| --- | ---: | ---: | ---: | ---: |
| can | 0.8312 | 0.8293 | 0.8279 | **-0.0034** |
| fabric | 0.9955 | 0.9956 | 0.9955 | +0.0001 |
| fruit_jelly | 0.9257 | 0.9297 | 0.9305 | +0.0047 |
| rice | 0.9817 | 0.9823 | 0.9828 | +0.0011 |
| sheet_metal | 0.9647 | 0.9712 | 0.9736 | +0.0089 |
| vial | 0.9612 | 0.9629 | 0.9646 | +0.0034 |
| wallplugs | 0.9459 | 0.9510 | 0.9519 | +0.0061 |
| walnuts | 0.9529 | 0.9726 | 0.9726 | +0.0197 |

Eight times the normal references leaves `can` unchanged to slightly worse. The
ladder was monotone-clean (no non-monotone flag), so this is not instability.

### Block A2 — per-layer representation probe (DVT off, Flow identity)

| object | layer7 | layer15 | layer23 | layer31 | fused | best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| can | 0.8438 | 0.8442 | 0.8252 | 0.7061 | 0.8280 | 0.8442 |
| fabric | 0.5336 | 0.7214 | 0.8800 | 0.9968 | 0.9726 | 0.9968 |
| fruit_jelly | 0.9058 | 0.9018 | 0.9077 | 0.9100 | 0.9224 | 0.9224 |
| rice | 0.8222 | 0.9412 | 0.9639 | 0.9686 | 0.9726 | 0.9726 |
| sheet_metal | 0.6444 | 0.5407 | 0.7696 | 0.9570 | 0.7894 | 0.9570 |
| vial | 0.9033 | 0.9351 | 0.9418 | 0.9631 | 0.9553 | 0.9631 |
| wallplugs | 0.7643 | 0.9158 | 0.9445 | 0.9111 | 0.9444 | 0.9445 |
| walnuts | 0.9096 | 0.9270 | 0.9667 | 0.8713 | 0.9583 | 0.9667 |

### Block A3 — `can` ordered stage path

| stage | pixel AUROC | pixel AP | pAUROC@0.05 | F1 | comp. recall | normal FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pre_dvt_feature | 0.8276 | 0.000176 | 0.5501 | 0.00062 | 0.0625 | 0.00724 |
| post_dvt_feature | 0.8500 | 0.000285 | 0.5924 | 0.00207 | 0.0833 | 0.00237 |
| flow_latent | 0.8312 | 0.000217 | 0.5736 | 0.00108 | 0.1458 | 0.00707 |
| deployed_final | 0.8312 | 0.000217 | 0.5736 | 0.00108 | 0.1458 | 0.00707 |

No stage loses `can`'s signal. DVT subtraction actually *improves* every `can`
endpoint and cuts normal FPR from `0.00724` to `0.00237`; the Flow stage then
gives part of that back. There is no single loss onset to attribute.

### Macro counterfactual

| metric | all-8 | excluding `can` | Δ |
| --- | ---: | ---: | ---: |
| pAUROC@0.05 | 0.8395 | 0.8775 | +0.0380 |
| F1 | 0.5290 | 0.6044 | +0.0754 |
| AU-PRO@0.05 | 0.5123 | 0.5689 | +0.0566 |

## Analysis

The verdict is `INCONCLUSIVE` under the frozen rule, and that is informative
rather than empty: it rejects representation, support coverage, and pipeline
loss as explanations of the `can` collapse.

An earlier reading of this project — that `can` carries almost no defect signal,
based on pAUROC@0.05 `0.5734` and AP `0.00022` — is corrected by the full-range
measurement. `can`'s deployed full-range pixel AUROC is `0.8312` and its best
layer probe reaches `0.8442`. The signal is present. The collapse is confined to
the low-false-positive operating region and to thresholded outputs.

Three measurements converge on an operating-point account:

1. more normal references do not help and marginally hurt, which is what happens
   when defects lie inside the normal manifold spanned by `train/good` rather
   than outside a poorly covered one;
2. no pipeline stage destroys the signal, and the stage that most reduces normal
   FPR (`post_dvt_feature`) also produces `can`'s best endpoints;
3. `can` and `vial` — the two highest normal FPRs of the eight — show the same
   signature of adequate full-range ranking with low-FPR collapse, from opposite
   directions.

This association between normal FPR and the low-FPR deficit is descriptive. This
protocol did not test whether the normal tail *causes* the deficit.

## Continuation Assessment

- No per-object parameter, layer, resolution, threshold, or support policy is
  selected or licensed by this result. The weak-object rule is retained.
- Representation-side work on `can` (backbone swap, layer reselection,
  resolution) is not supported: the representation already separates `can`
  defects at `0.8442`.
- Support-efficiency work motivated by `can` is not supported: `-0.0034` at 8x.
- The DVT/Flow signal-path branch is not reopened: no stage loses the signal.
- What this licenses is a measurement of the normal-side tail — its structure,
  its identifiability without test access, and whether suppressing it could
  possibly be safe. That is preregistered separately as
  `normal-tail-structure-predictability-all8-20260809`, which is measurement
  only and licenses no intervention by itself.

## Conclusion

`INCONCLUSIVE`. All three preregistered hypotheses are rejected. `can` is not a
no-signal object; it is a low-FPR-operating-point failure, and the same
signature appears in `vial`. The next justified step is to characterise the
normal-side tail, not to modify the representation, the support budget, or the
scoring path.

Primary artifact: `results/can_collapse_attribution_20260809/summary.json`.
