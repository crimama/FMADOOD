# MVTec AD1 density0 + guided-r8 hybrid 1/2/4/8-shot sweep

Date sealed: 2026-07-13

## Motivation and boundary

The isolated 4-shot diagnostic found that density removal alone passes the
all-five retention/positive gate, while guided-r8 improves p-AUPRC on all 15
classes but loses image AUROC because the refined map also supplied the image
top-1% score. This experiment tests one fixed two-output contract and does not
tune density, radius, epsilon, support selection, or image aggregation.

## Locked method

- Dataset: classic MVTec AD, all 15 classes, complete test split.
- Shots: `{1,2,4,8}`; first `train/good` images, seed 0, identity only.
- Frozen `dinov2_vitb14_reg`, shorter-edge 448, layers `[2,5,8,11]`,
  `layer_norm_mean`.
- Flow: three epochs, two couplings, hidden multiplier 1, LR 2e-4, clamp 1.9,
  tail 0.3/top-k 0.05, lambda-logdet 1e-3.
- Static latent bank: expansion budget 1.0; no TTE or DVT.
- Raw score: latent 1-NN distance 1.0, density weight 0.0, top-1% image score.
- Pixel refinement: grayscale guided filter on the same raw map, half-native
  scale, radius 8, epsilon 0.01, per-image min-max/de-min-max, bilinear return.
- Hybrid output: i-AUROC/i-AUPRC come only from the raw density0 map;
  p-AUROC/p-AUPRC/p-AUPRO come only from the guided-r8 map.
- No GT, label, threshold, class-specific parameter, morphology, calibration,
  context, or register signal enters either output.

## Same-shot controls

The paired controls are the completed standard density-weight-0.25 static
results under the identical contract:

| Shot | i-AUROC | i-AUPRC | p-AUROC | p-AUPRC | p-AUPRO |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.955967725 | 0.978914073 | 0.967922356 | 0.551566232 | 0.926363513 |
| 2 | 0.969709291 | 0.984958077 | 0.972010011 | 0.570467363 | 0.935150986 |
| 4 | 0.972100417 | 0.987431867 | 0.973747447 | 0.581160338 | 0.938160366 |
| 8 | 0.981751436 | 0.991162486 | 0.975455201 | 0.590661421 | 0.942404898 |

Strict external VisionAD/SuperAD comparison remains `BLOCKED_BASELINE`.

## Gates

- Validity: the new 4-shot raw density0 five-metric row must reproduce the
  prior density0 artifact within `1e-9`; hybrid image fields must exactly equal
  raw fields and hybrid pixel fields must exactly equal guided fields.
- Static-memory/support: every class must use exact first-N supports and have
  identical initial/final bank sizes.
- Per-shot retention: no hybrid macro may lose more than 0.10 percentage point
  against its same-shot control.
- Per-shot positive diagnostic: retention passes and p-AUPRC or p-AUPRO gains
  at least 0.20 percentage point.
- Robust continuation: at least three of four shots pass both gates, without a
  catastrophic class loss greater than 5.0 points on a requested metric.
- Class wins/losses and shot trend are reported independently. No cross-shot
  averaging can conceal a failed shot.

## Execution and cleanup

- dsba5 GPU 0: shots 1 then 4; GPU 1: shots 2 then 8.
- Source and refined dense maps are deleted immediately after each shot's
  metrics are written. Compact metrics, manifests, logs, and evidence remain.
