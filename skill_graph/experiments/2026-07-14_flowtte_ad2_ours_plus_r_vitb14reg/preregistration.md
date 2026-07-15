# Ours+R DINOv2-R ViT-B/14 Preregistration

Date: 2026-07-14  
Status: frozen before remote execution  
Protocol: `fmad-experiment-protocol`

## Question

Measure the missing Table 1 Ours+R row while changing only the frozen image
encoder from the Ours Basic DINOv2 ViT-L/14 encoder to DINOv2-R ViT-B/14.
This is a controlled backbone-scaling measurement, not a new detector branch.

## Frozen contract

- Dataset: MVTec AD 2 single-image, all eight public objects, full
  `test_public/good,bad` (`TESTpub`).
- Support: the exact SuperAD-selected 16 normal reference paths per object,
  seed 0; full `train/good` is only the selection pool.
- Encoder: frozen `dinov2_vitb14_reg`, layers `[2,5,8,11]`.
- Resolution: shorter side 672, except `sheet_metal` at 448.
- Detector: Flow-LatentBank, 3 epochs, 2 coupling layers, hidden multiplier 1,
  LR `2e-4`, clamp 1.9, log-det `1e-3`, static memory, latent 1-NN.
- Components: LOO off, DVT position-mean alpha 1, density weight 0.25,
  RGB guided-r8, and fixed line17/angles16 close--fill--erode morphology.
- Disabled: TTE expansion, context/register routing, tiling, support transforms,
  and brightness augmentation.
- Primary outputs: macro pixel AUROC at max FPR 0.05 and oracle-threshold
  segmentation F1 after the frozen morphology, both reported as percentages.

## Comparability and gate

The result is directly comparable with Ours Basic for the detector, support,
split, resolution, and evaluator. It is contextual against SuperAD because the
backbone differs. Report the row regardless of direction; continue the +R
branch only if it improves Basic by at least 0.3 point on pAUROC or morphology
F1 without catastrophic object-level collapse.

Same-condition SuperAD with DINOv2-R ViT-B/14 is absent, so any superiority
claim against SuperAD is `BLOCKED_BASELINE`. The experiment remains valid as
the preregistered backbone ablation needed for Table 1.
