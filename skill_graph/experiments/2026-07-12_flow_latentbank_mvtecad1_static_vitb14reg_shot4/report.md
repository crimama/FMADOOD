# Static Flow-LatentBank MVTec AD1 ViT-B/14-register 4-shot

Date: 2026-07-12  
Verdict: `ACCEPT_MEASUREMENT / BLOCKED_BASELINE`

## Negative Evidence Intake

This requested run revisits the earliest Flow + Latent Bank configuration as a
controlled baseline, not as a new method claim. It removes the known
anomaly-absorption path by setting the expansion budget to 1.0. The remaining
method is a passive NF latent 1-NN memory baseline and therefore cannot be
promoted as novel or SOTA evidence.

## Motivation

Measure the original all-15 MVTec AD1 4-shot recipe after exactly two requested
structural changes: replace DINOv2 ViT-L/14 with DINOv2-R ViT-B/14 and remove
TTE memory expansion. Keep support selection, preprocessing family, feature
fusion, NF training, score weights, and evaluator geometry fixed where the
backbone depth permits.

## Implementable Design

- Dataset: classic MVTec AD, all 15 categories, full test split.
- Supports: first four `train/good` images, seed 0, identity only.
- Encoder: frozen/eval `dinov2_vitb14_reg`.
- Preprocessing: original shorter-edge 448 recipe, patch-aligned crop.
- Layers: ViT-B `[2,5,8,11]`, the depth-relative counterpart of ViT-L
  `[5,11,17,23]`; `layer_norm_mean` fusion.
- Flow: 3 epochs, 2 coupling layers, hidden multiplier 1, LR 2e-4, clamp 1.9,
  tail 0.3/top-k 0.05, lambda-logdet 1e-3.
- Scoring: latent 1-NN distance weight 1.0 plus NF density penalty weight 0.25;
  top 1% image aggregation.
- Static memory: expansion budget 1.0.
- Explicitly absent: DVT, context/conditioning, register-token conditioning,
  foreground prior, morphology, score calibration, and support augmentation.

## Evaluation Alignment

The historical reference is
`flow_tte_nf_mvtecad1_all15_shot4_20260706_v1_pixelpro`. Dataset, split,
support paths, shorter-edge resolution, Flow hyperparameters, fusion, and metric
semantics are aligned. Two requested factors differ simultaneously: backbone
and TTE. Pixel AUROC/AUPRC also use the corrected overflow-safe rank writer in
the new run rather than the historical float16 histogram. Consequently, deltas
are descriptive configuration deltas, not causal TTE effects.

## Code Modification / Creation

- `src/backbones.py`: make DINOv2 intermediate layers configurable, validate
  them against model depth, and freeze encoder parameters.
- `fmad/backbones/dinov2.py`: carry requested layer indices through wrapper
  construction and resolution reloads.
- `scripts/run_flow_tte_mvtec_ad1.py`: pass the serialized layer contract to
  the shorter-edge backbone.
- `tests/test_dinov2_feature_layers.py`: verify ViT-B layer routing and reject
  the invalid ViT-L layer set.
- `scripts/run_flow_latentbank_mvtecad1_static_vitb14reg_shot4.sh`: fixed
  preflight and benchmark launcher.

## Added Code Evaluation

- Focused backbone/runner/evaluator tests: 31 passed before execution.
- Final full local suite: 412 passed, 5 non-failing Transformer warnings.
- Python compilation and launcher shell syntax: passed.
- Ruff on the new test and modified AD1 runner: passed. The two legacy backbone
  wrapper files retain pre-existing repository-wide lint debt outside this
  experiment's diff.

## Remote Execution

- Host/container: dsba5, fixed project container `hun_fsad_tta_012`.
- GPU: host GPU 0; GPU 1 remained idle and available for recovery.
- Data root: `/home/woojun/dataset/mvtec_ad`.
- Preflight: all 15 categories populated; layers `(2,5,8,11)`; bottle sample
  grid 32x32; four feature tensors each 1024x768; encoder frozen and eval.
- Result root:
  `results/remote_runs/dsba5/flow_latentbank_mvtecad1_all15_shot4_vitb14reg_static_20260712_v1`.
- Sum of recorded per-category execution time: 147.66 seconds, excluding model
  preflight/loading and exact p-AUPRO evaluation.

## SuperAD Baseline and Unified Metrics

No same-condition SuperAD or VisionAD baseline uses this static Flow recipe,
ViT-B/14-register shorter-edge preprocessing, exact first-four supports, and
the same evaluator. Strict superiority is therefore `BLOCKED_BASELINE`.

All values below are unweighted class-macro percentages.

| Configuration | i-AUROC | i-AUPRC | p-AUROC | p-AUPRC | p-AUPRO |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical ViT-L + TTE 1.25 | 96.96 | 98.39 | 96.40 | 57.61 | 93.65 |
| ViT-B-R + static bank 1.0 | **97.21** | **98.74** | **97.37** | **58.12** | **93.82** |
| Descriptive delta | +0.25 | +0.36 | +0.97 | +0.50 | +0.17 |

## Results and Analysis

The requested configuration completed successfully:

- `i_AUROC=0.9721004167`
- `i_AUPRC=0.9874318674`
- `p_AUROC=0.9737474466`
- `p_AUPRC=0.5811603385`
- `p_AUPRO=0.9381603664`

Every category used exactly `000.png` through `003.png`. Initial and final
latent-memory sizes were exactly `4096 -> 4096` for all 15 categories, proving
that no TTE expansion occurred. All metrics are finite, all 15 per-category
rows exist, and the manifest embeds the same metrics as the standalone file.

The mean improves on all five historical fields, but category behavior is
mixed. Image AUROC improves on 8 classes, decreases on 2, and ties on 5;
p-AUROC improves on 9 and decreases on 6. p-AUPRC and p-AUPRO improve on only
6 classes and decrease on 9, despite their positive macro means. `cable` and
`metal_nut` provide the largest localization gains, while `screw` loses 6.60
points image AUROC and 14.61 points p-AUPRC. The result therefore does not
establish broad per-category no-harm.

## Continuation Assessment

Strict method claim now: no. The comparator is historical, two factors changed,
the pixel-rank implementation changed, and no multiple-seed uncertainty is
available.

The result is nevertheless a valid static Flow-LatentBank baseline. If a causal
TTE conclusion is needed, the single next experiment must use the same
ViT-B-R configuration and supports with only `expansion_budget=1.25` restored.
It should be hard-stopped as a TTE direction if static memory remains better on
the macro metrics or if expansion again harms a majority of categories.

## Conclusion

Verdict: `ACCEPT_MEASUREMENT / BLOCKED_BASELINE`.

The requested MVTec AD1 configuration is fully measured and reproducible. It
slightly exceeds the historical aggregate row, while exposing substantial
class asymmetry and supporting no strict superiority or causal TTE claim.

## Post-Conclusion Storage Cleanup

Remote and local audits both report zero `anomaly_maps` directories. Retained
compact evidence consists of `metrics.json`, `metrics_seed=0.json`,
`run_manifest.json`, and `cleanup_evidence.txt`.
