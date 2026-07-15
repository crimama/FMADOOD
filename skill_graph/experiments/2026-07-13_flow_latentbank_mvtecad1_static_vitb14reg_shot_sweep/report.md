# Static Flow-LatentBank MVTec AD1 ViT-B/14-register shot sweep

Date: 2026-07-13  
Verdict: `ACCEPT_MEASUREMENT / BLOCKED_BASELINE`

## Negative Evidence Intake

This experiment extends an accepted baseline rather than introducing a new
method. The only intended factor is support count. TTE and all additional
context, denoising, calibration, foreground, and morphology components remain
disabled.

## Motivation

Measure the same static Flow-LatentBank recipe at 1, 2, and 8 shots and combine
it with the already completed same-condition 4-shot anchor. This directly tests
support-budget scaling under a deterministic first-support policy.

## Implementable Design

- Dataset: classic MVTec AD, all 15 categories and full test split.
- Supports: first N `train/good` images, N in `{1,2,8}`, seed 0, identity.
- Encoder: frozen/eval `dinov2_vitb14_reg`; shorter-edge 448 preprocessing.
- Features: ViT-B layers `[2,5,8,11]`, `layer_norm_mean` fusion.
- Flow: 3 epochs, 2 coupling layers, hidden multiplier 1, LR 2e-4, clamp 1.9,
  tail 0.3/top-k 0.05, lambda-logdet 1e-3.
- Score: latent 1-NN distance weight 1.0 plus NF density penalty weight 0.25;
  density quantile 0.90 and image top 1% aggregation.
- Static memory: `expansion_budget=1.0`.
- Metrics: class-macro i-AUROC, i-AUPRC, p-AUROC, p-AUPRC, and exact
  p-AUPRO at FPR 0.30.

## Evaluation Alignment

The 1/2/8-shot runs and the 4-shot anchor use the same code, data split,
preprocessing, encoder, layers, flow/scoring hyperparameters, support ordering,
and corrected pixel-rank evaluator. Shot count is the only intended factor.
This is a deterministic single-support-order curve, not a multiple-seed
uncertainty estimate.

## Code Modification / Creation

- `scripts/run_flow_latentbank_mvtecad1_static_vitb14reg_shot_sweep.sh`:
  resumable two-GPU launcher with one shared preflight, independent shot
  processes, compact map cleanup, and aggregate JSON generation.
- `preregistration.md`: locked design and acceptance gates before execution.

No method implementation changed for this sweep.

## Added Code Evaluation

- Launcher shell syntax: passed.
- Focused DINOv2 layer and MVTec classic adapter suite: 25 passed.
- Python compilation of the affected backbone and AD1 runner: passed.

## Remote Execution

- Host/container: dsba5, existing `hun_fsad_tta_012` container.
- GPUs: GPU 0 ran shot 1 followed by shot 8; GPU 1 ran shot 2.
- Preflight: 15 categories; layers `(2,5,8,11)`; grid 32x32; four feature
  tensors of 1024x768; encoder frozen/eval.
- Recorded per-category inference-time sums, excluding model loading and exact
  p-AUPRO: 140.14 s (1), 143.26 s (2), and 149.64 s (8).
- Result root:
  `results/remote_runs/dsba5/flow_latentbank_mvtecad1_all15_static_vitb14reg_s1_2_8_20260713_v1`.

## SuperAD Baseline and Unified Metrics

All values are unweighted class-macro percentages. The 4-shot row is the
same-condition result completed on 2026-07-12.

| Shots | i-AUROC | i-AUPRC | p-AUROC | p-AUPRC | p-AUPRO |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 95.60 | 97.89 | 96.79 | 55.16 | 92.64 |
| 2 | 96.97 | 98.50 | 97.20 | 57.05 | 93.52 |
| 4 | 97.21 | 98.74 | 97.37 | 58.12 | 93.82 |
| 8 | **98.18** | **99.12** | **97.55** | **59.07** | **94.24** |

No same-condition VisionAD or SuperAD baseline is available, so strict
superiority remains `BLOCKED_BASELINE`.

## Results and Analysis

All five macro metrics increase from 1 to 2 to 4 to 8 shots. The total 1-to-8
gains are +2.58 i-AUROC, +1.22 i-AUPRC, +0.75 p-AUROC, +3.91 p-AUPRC, and
+1.60 p-AUPRO points. Additional normal support benefits pixel precision most,
while p-AUROC is already high at one shot and consequently saturates earlier.

The scaling curve is not class-wise monotonic. From 4 to 8 shots, p-AUROC
improves for 14/15 classes, p-AUPRC for 10/15, and p-AUPRO for 12/15.
`screw` supplies the largest 1-to-8 image gain (+22.73 i-AUROC) and a large
+15.66 p-AUPRC gain. Conversely, 8-shot p-AUPRC falls relative to 4-shot for
five classes, led by `wood` (-2.67 points); `tile` also loses 0.36 p-AUPRO.
Thus the macro scaling signal is strong, but it does not establish per-class
no-harm.

The audit passed for every shot: 15 metric rows and diagnostics, finite five
metrics, standalone/seed/manifest/aggregate equality, and exact first-N paths.
All classes kept their latent banks fixed at `1024 -> 1024`, `2048 -> 2048`,
or `8192 -> 8192`. No anomaly-map directory remains, and logs contain no
traceback, OOM, NaN, or Inf.

## Continuation Assessment

The requested support-budget sweep is complete. A statistical scaling claim
would require repeating support selection over multiple seeds or ordered
support subsets; this deterministic first-N curve alone cannot estimate
variance. No additional run is required for the present measurement request.

## Conclusion

Verdict: `ACCEPT_MEASUREMENT / BLOCKED_BASELINE`.

The static Flow-LatentBank configuration scales positively in all five macro
metrics through 8 shots, with the best row at 8 shots. The result is valid and
reproducible, while strict SOTA comparison and per-class no-harm claims remain
unsupported.

## Post-Conclusion Storage Cleanup

All three run roots contain only compact metrics, manifests, cleanup evidence,
and logs. Local inspection found zero `anomaly_maps` directories.
