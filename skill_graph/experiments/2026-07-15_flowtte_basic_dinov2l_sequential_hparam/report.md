# FlowTTE Basic DINOv2-L Sequential Hyperparameter Run

- Status: completed
- Protocol: `fmad-experiment-protocol`, diagnostic tuning lane
- Target: MVTec AD2, all eight objects, full `test_public/good,bad`
- Backbone: frozen DINOv2 ViT-L/14 (`dinov2_vitl14`)
- Feature layers: `[5,11,17,23]` unless the layer stage is selected
- Support: exact fixed SuperAD-16 reference paths, seed 0, static bank
- Score: latent distance plus density weight 0.25; DVT `position_mean`
  alpha 1.0; LOO standardization disabled; raw map plus `guided_r8` and
  `closefill_erode` post-processing
- Metrics: `seg_AUROC` is the project evaluator's max-FPR 0.05 AUROC;
  `seg_F1` is the best-threshold morphology F1. Official AUPRO is not
  silently substituted for this evaluator metric.

## Sequential stages

1. **Capacity** (already complete): depth `{1,2,4}` × width `{1,2}`.
2. **Optimization** (already complete): epochs/LR compact cross.
3. **Regularization** (running): `lambda_logdet` `{1e-3,2e-2}` × support
   brightness `{1.0,1.0; 0.8,1.2}`.
4. **Feature layers** (queued): early/current/mid-late/late blocks.

Only the selected winner from a completed stage is passed to the next stage.
Stage 3 candidates are split across dsba3 and dsba5; stage 4 uses the same
host split. Dense anomaly maps are removed by the remote launcher after each
configuration, while compact manifests, metrics, and logs are retained.

## Completed stage anchors

| stage | selected config | guided p-AUROC | guided F1 |
|---|---|---:|---:|
| capacity | `cap_d2_w1` | 0.781937 | 0.441074 |
| optimization | `opt_e3_lr2` (3 epochs, 2e-4) | 0.781937 | 0.441074 |
| regularization | `reg_ld2_b08` | 0.782534 | 0.439453 |
| layers | `layer_late` | 0.785792 | 0.440896 |

## Completed candidate metrics

| stage | candidate | guided p-AUROC | guided F1 |
|---|---|---:|---:|
| regularization | `reg_base` | 0.781939 | 0.440818 |
| regularization | `reg_ld2` | 0.782269 | 0.441400 |
| regularization | `reg_b08` | 0.782193 | 0.438565 |
| regularization | `reg_ld2_b08` **selected** | **0.782534** | 0.439453 |
| layers | `layer_early` | 0.577201 | 0.090947 |
| layers | `layer_current` | 0.782534 | 0.439453 |
| layers | `layer_midlate` | 0.782252 | 0.438456 |
| layers | `layer_late` **selected** | **0.785792** | **0.440896** |

The final selected Basic setting is depth 2, width 1, 3 epochs, learning
rate `2e-4`, `lambda_logdet=2e-2`, support brightness range `0.8,1.2`, and
feature layers `[11,15,19,23]`. Relative to the Stage-1 anchor, the final
guided result changes by `+0.003855` p-AUROC and `-0.000178` F1, so the gain
is ranking-oriented rather than a broad segmentation-F1 improvement.

These are diagnostic results until the exact same-run SuperAD-16,
RN-FMLK/full-memory, and hard-null gates are present. No superiority claim is
made from this tuning ladder alone.
