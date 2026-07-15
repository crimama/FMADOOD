# AD2 Basic Nested Few-Shot Scaling Preregistration

Date: 2026-07-14  
Status: frozen before remote execution

## Contract

- Target: MVTec AD2 single-image, all eight objects, full `test_public`.
- Backbone: frozen DINOv2 ViT-L/14, layers `[5,11,17,23]`.
- Resolution: shorter side 672, except `sheet_metal` at 448.
- Support: nested prefixes K=1,2,4,8 of the exact SuperAD-selected P16 JSON;
  seed 0. The completed K=16 no-LOO Basic result is the anchor.
- Detector and postprocessing: Basic default unchanged: static Flow-LatentBank,
  3 epochs, 2 coupling layers, width 1, LR 2e-4, clamp 1.9, logdet 1e-3,
  LOO off, DVT position-mean alpha 1, density weight 0.25, guided-r8, and
  line17/angles16 close--fill--erode.
- Metrics: macro pAUROC at max FPR 0.05 and oracle-threshold morphology F1.
- This is a support-budget scaling ablation. Only K=16 is paper-aligned with
  SuperAD-16; smaller K rows are not direct SuperAD superiority claims.

The queue starts only after the active Ours+R run terminates and executes
1,2,4,8 shots sequentially, with four object shards in parallel per shot.

