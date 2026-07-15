# AD2 H+ Flow-LatentBank + DVT + Density + RGB Guide + Morphology

Date: 2026-07-13
Status: PREREGISTERED_BEFORE_EXECUTION

## Question and frozen comparison

Test whether RGB guided-r8 map refinement improves the retained AD2 H+
Flow-LatentBank setting when both control and candidate use the same fixed
binary morphology. The matched comparison is generated from one fresh run:

- control: raw continuous map + `closefill_erode`;
- candidate: the same raw map + `guided_r8_eps1e-2` + `closefill_erode`.

Only the RGB-guided refinement differs. No component may be tuned per object.

## Frozen upstream setting

- Dataset: MVTec AD2 single-image, all eight objects, full
  `test_public/good,bad`.
- Support: 16 normal images from the fixed JSON selected by the retained
  SuperAD protocol; identity support views; seed 0.
- Backbone: frozen DINOv3-H+/16 (`dinov3_vith16plus`), layers
  `[7,15,23,31]`.
- Resolution: dataset defaults, 672 for all objects except `sheet_metal=448`;
  resize factor 1, no tiling, no masking, no rotation.
- Feature fusion: `layer_norm_mean`.
- Flow: 3 epochs, 2 coupling layers, hidden multiplier 1, LR `2e-4`, clamp
  1.9, tail weight 0.3, tail top-k 0.05, logdet `1e-3`.
- Static latent bank: expansion budget 1.0; latent 1-NN distance weight 1.0.
- LOO support standardization: existing scorer behavior, unchanged.
- DVT: position-mean denoising, alpha 1.0.
- Density: weight 0.25, quantile 0.90.
- Context/register/flow conditioning: none.

## Frozen downstream setting

- RGB guide: half-scale continuous guided filter, radius 8, epsilon 0.01,
  then bilinear restoration to the native map size.
- Binary morphology for both arms: multi-angle directional close with line
  length 17 and 16 angles, hole fill, then fixed 3x3 erode.
- Morphology threshold: each arm's raw continuous-map best threshold. This is
  evaluator-only diagnostic thresholding; it is not a deployable calibration
  claim.
- Ground truth is not consumed by RGB refinement or morphology parameters.

## Metrics and gates

Primary metrics are mean pixel `seg_AUROC@FPR<=0.05` and mean morphology
`seg_F1`. Raw `seg_F1_raw`, thresholds, and per-object values are retained.

- Validity gate: all eight objects finish with fixed support paths, maps are
  finite and shape matched, and control/candidate manifests preserve the
  frozen settings.
- Positive gate: candidate improves either mean AUROC by at least 0.002 or
  mean morphology F1 by at least 0.005 without degrading the other by more
  than 0.002/0.005 respectively.
- Harm gate: flag any object losing more than 0.03 on either primary metric.
- Strict external parity remains blocked because the reported SuperADD result
  is contextual rather than a same-run baseline.

## Resources

Run on dsba3 GPUs 0,1,2,3, two objects per GPU. Existing jobs have priority;
launch only after all four requested devices are free.
