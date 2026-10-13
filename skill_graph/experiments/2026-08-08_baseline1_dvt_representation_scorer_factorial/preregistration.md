# Baseline1 DVT x Representation x Scorer Factorial Preregistration

Date: 2026-08-08
Protocol: `baseline1-dvt-representation-scorer-factorial-20260808`
Status: `DESIGN_FROZEN`
Scope: reduced-object attribution diagnostic, not a method claim

## Negative Evidence Intake

This experiment does not introduce a new method, threshold, support policy, density
term, score-field calibration, query update, morphology, or object-specific tuning.
It reuses the fixed four-object mechanism protocol that already isolated a DVT main
effect and a small positive DVT x Flow interaction with `raw_1nn`.

Known negative evidence constrains the design:

- The 2026-08-07 k=5 default A/B found that `knn_mean_k5` reduces score-tail
  extremes but failed the macro pAUROC no-harm gate, so K5 cannot be promoted as
  a default scorer here.
- Existing scorer-suite metadata labels `shrinkage_mahalanobis` calibration as
  in-sample support distances with optimism bias. Prior Mahalanobis/whitening
  branches are treated as diagnostic hard-null evidence, not as claim-supporting
  scorers.
- The likely failure basin is scorer/calibration redistribution on top of an
  already strong kNN-like latent memory. A positive result can support attribution
  robustness only if the DVT contrast survives the scorer change; it cannot
  support a new deployable method claim.

## Motivation

The previous DVT x Flow factorial showed that DVT explains most of the reduced
four-object Baseline1 improvement under `raw_1nn`. This run asks whether that
attribution is robust to the nearest-neighbor aggregation choice or whether the
apparent DVT effect is partly a 1-NN scorer artifact.

## Fixed Protocol

- Dataset: MVTec AD2 single-image `test_public`.
- Objects: `fruit_jelly, walnuts, vial, fabric`.
- Support: the same fixed SuperAD-selected 16 normal images per object from
  `skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json`.
- Backbone/representation: DINOv3-H+/16, feature layers `[7,15,23,31]`,
  `layer_norm_mean`, resolution 672.
- Flow: existing 2-layer patch-local Flow, 3 epochs, LR `2e-4`, tail `0.3`,
  logdet `0.02`, seed 0.
- No context, no memory update, no TTA, no metadata, no density term
  (`density_weight=0.0`), no score-field calibration, no foreground mode, no
  binary postprocess.
- The runner must write compact JSON metrics only; native anomaly-map TIFFs are
  not retained.

## Arms

Representation/transform arms are the 2 x 2 design:

| arm | DVT | transform |
| --- | --- | --- |
| `fused_raw` | off | identity |
| `fused_flow` | off | Flow |
| `dvt_raw` | position-mean alpha 1.0 | identity |
| `dvt_flow` | position-mean alpha 1.0 | Flow |

Scorer arms:

- Primary scorer A: `raw_1nn`.
- Primary scorer B: `knn_mean_k5`.
- Optional diagnostic scorer: `shrinkage_mahalanobis`, reported only when the
  same run emits it. It is labeled `diagnostic_only=true` with
  `calibration_optimism=in_sample_support_distances` and cannot satisfy any
  attribution robustness gate.

The minimum valid run therefore emits at least eight cells:
`4 representation arms x {raw_1nn, knn_mean_k5}`.

## Metrics

Primary metrics, per object and four-object macro:

- `seg_AUROC_0.05` from `pooled_pauroc_0.05_float16`.
- `seg_F1` from `pooled_oracle_f1_float16`.
- `seg_AUPRO_0.05` from `pooled_aupro_0.05`.

Required coverage checks:

- selected support count is 16 for every object/arm;
- selected support basenames match across all four representation arms;
- prediction map count equals GT map count;
- shape count equals prediction map count;
- non-finite prediction count is zero;
- anomaly-map TIFF count is zero.

## Preregistered Contrasts

For each scorer separately:

- DVT effect without Flow: `dvt_raw - fused_raw`.
- DVT effect with Flow: `dvt_flow - fused_flow`.
- DVT main effect: average of the two DVT effects.
- Flow effect without DVT: `fused_flow - fused_raw`.
- Flow effect with DVT: `dvt_flow - dvt_raw`.
- DVT x Flow interaction:
  `(dvt_flow - dvt_raw) - (fused_flow - fused_raw)`.

Scorer robustness contrasts:

- `K5 - K1` for each representation arm.
- Change in DVT main effect under K5:
  `dvt_main_effect(knn_mean_k5) - dvt_main_effect(raw_1nn)`.
- Change in DVT x Flow interaction under K5:
  `interaction(knn_mean_k5) - interaction(raw_1nn)`.

## Gates

`DVT_ATTRIBUTION_ROBUST_TO_SCORER` passes only if all are true:

1. `raw_1nn` repeats the previous DVT main-effect gate:
   macro `seg_AUROC_0.05 >= +0.005`, positive object count at least 3/4,
   macro `seg_F1 > -0.01`, macro `seg_AUPRO_0.05 > -0.01`, and no object
   collapse below pAUROC `-0.02`, F1 `-0.03`, or AU-PRO `-0.02`.
2. `knn_mean_k5` DVT main effect also has macro `seg_AUROC_0.05 >= +0.005`,
   positive object count at least 3/4, macro `seg_F1 > -0.01`, macro
   `seg_AUPRO_0.05 > -0.01`, and no object collapse by the same thresholds.
3. K5 does not erase the DVT attribution: its DVT main-effect pAUROC is no more
   than `0.01` below the raw-1NN DVT main-effect pAUROC.

`SCORER_DEPENDENT_DVT_ATTRIBUTION` is reported if raw passes but K5 fails any
DVT main-effect gate, or if K5 reduces the DVT main-effect pAUROC by more than
`0.01`.

`K5_DEFAULT_RECONSIDERATION` is explicitly out of scope. Even if K5 preserves
the DVT attribution here, the 2026-08-07 all-object A/B remains the default
scorer decision.

`MAHALANOBIS_DIAGNOSTIC_ONLY` is always true for shrinkage Mahalanobis rows. If
Mahalanobis outperforms K1/K5, the only allowed conclusion is a follow-up
calibration/conditioning audit; it cannot pass the DVT attribution robustness
gate because its support-distance calibration is optimistic and can be
ill-conditioned.

## Execution Boundary

Do not launch the full GPU benchmark until this preregistration, runner command,
aggregator, and tests are committed or reviewed by the leader. The implementation
phase may run only local syntax/unit tests and CLI dry-run/help checks.

