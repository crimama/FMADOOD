# Basic P0/P1/P2 Output-Chain Custody Audit

Date: 2026-07-23  
Status: `DIAGNOSTIC_COMPLETE / PRIMARY_P2_NOT_AVAILABLE`  
Maximum claim: `PUBLIC_SHADOW_CONDITIONAL_OUTPUT_EFFECT`

## Question and scope

This audit closes the previously missing public-shadow P0/P1/P2 diagnostic
without rerunning or tuning the detector. It reaggregates the frozen
DINOv2-L/14 `minus_loo` artifact from the 2026-07-13 Basic component ablation:

- all eight MVTec AD2 public objects;
- exact SuperAD-selected 16 normal references;
- support-manifest SHA-256
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`;
- P0: raw continuous map;
- P1: guided-r8 continuous map;
- P2-oracle: P1 thresholded at the public best threshold, followed by the
  frozen close/fill/erode sequence.

The custody source is
`results/remote_runs/dsba3/flowtte_basic_component_ablation_fixed16_20260713_v1/configs/minus_loo/summary.json`,
SHA-256
`a6e5721d09e87453a1d46c8af9ba3cdd022d236877846d05bc0f07a521505657`.
Its `seg_AUROC` field is the repository's standardized pixel
pAUROC@0.05. `seg_F1_raw` is used for continuous-map threshold diagnostics;
the post-morphology `seg_F1` is reported only for P2-oracle.

The primary preregistered P2 uses a support-fit clean q=0.995 threshold. That
threshold was not produced by the historical run, and the dense maps were
already removed after its cleanup audit. Therefore this report does not
substitute the public best threshold for the missing primary result.

## Results

Values below are percentages. `P1-P0` compares guided and raw continuous maps.
`P2-P1 F1` is the oracle-threshold morphology delta and is not deployable.

| Object | P0 pAUROC | P0 raw F1 | P1 pAUROC | P1 raw F1 | P1-P0 pAUROC | P1-P0 F1 | P2-oracle F1 | P2-P1 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| can | 63.426 | 0.170 | 60.361 | 0.129 | -3.065 | -0.041 | 0.117 | -0.012 |
| fabric | 69.440 | 26.666 | 71.100 | 28.182 | +1.660 | +1.517 | 71.915 | +43.732 |
| fruit_jelly | 75.901 | 36.252 | 77.434 | 41.852 | +1.534 | +5.600 | 44.207 | +2.355 |
| rice | 92.128 | 66.098 | 93.040 | 67.699 | +0.912 | +1.602 | 67.562 | -0.137 |
| sheet_metal | 83.905 | 38.605 | 86.014 | 41.994 | +2.109 | +3.389 | 41.899 | -0.095 |
| vial | 69.553 | 36.073 | 70.641 | 38.084 | +1.088 | +2.011 | 38.187 | +0.102 |
| wallplugs | 75.614 | 10.773 | 78.287 | 14.066 | +2.673 | +3.293 | 14.231 | +0.165 |
| walnuts | 88.142 | 69.890 | 88.672 | 72.853 | +0.531 | +2.963 | 74.742 | +1.889 |
| **Macro** | **77.263** | **35.566** | **78.194** | **38.108** | **+0.930** | **+2.542** | **44.107** | **+6.000** |

P1 improves both continuous metrics on seven of eight objects. P2-oracle
improves F1 on five objects and is nearly neutral on the other three. P2 does
not receive a separate pAUROC value: morphology is applied after thresholding,
while the retained ranking metric remains the P1 continuous-map pAUROC.

## Frozen-gate decision

The output-stage gate requires at least `+0.010` fixed F1, less than `0.005`
pAUROC harm, and no object collapse. The available evidence does not pass it:

1. P1 has positive macro effects, but `can` loses `0.03065` pAUROC, crossing
   the frozen `-0.02` object-collapse boundary.
2. The P2 macro F1 gain is an oracle public-threshold diagnostic, not the
   required support-fit clean-threshold result.
3. The historical artifact cannot be upgraded to primary P2 after dense-map
   cleanup; doing so would require a fresh, preregistered rerun.

The output chain is therefore classified
`CONDITIONAL_PRESENTATION_EFFECT`, not globally safe postprocessing and not
distribution-shift robustness. RGB guidance is helpful on most objects but
unsafe as an unconditional default because of the `can` ranking collapse.
Morphology shows a large aggregate oracle benefit dominated by `fabric`, but
its primary fixed-threshold effect remains `NOT_EVALUATED`.

## Consequence

This audit closes the historical diagnostic gap only. It does not reopen the
2026-07-19 Stage-M1 verdict, authorize Stage M2, or support a superiority
claim. Same-condition SuperAD-16/RN-FMLK and original-resolution AU-PRO@0.05
remain pending in the joint Same16 run. A fresh Basic-only support-fit P2 is
stopped because thresholding cannot repair the failed P1 ranking gate; P2 may
be reconsidered only after a new detector is frozen.
