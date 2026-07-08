# FlowTTE NF MVTec AD2 vs SuperAD and SuperADD

Date: 2026-07-07

Protocol: `fmad-experiment-protocol`

Remote:

- Server/container: dsba3, `hun_fsad_tta`
- Host GPU: `3`
- In-container CUDA slot: `0`
- Dataset: `/home/hunim/Volume/DATA/mvtec_ad_2`
- Local pullback root: `results/remote_runs/dsba3`

## Goal

Run the NF-based FlowTTE pipeline on all 8 public MVTec AD2 objects and compare
with the available SuperAD and SuperADD references.

## Method Setting

- Run:
  `flow_tte_nf_mvtecad2_all8_shot16_coreset_dw0_20260707_v1`
- Objects: `can`, `fabric`, `fruit_jelly`, `rice`, `sheet_metal`, `vial`,
  `wallplugs`, `walnuts`
- Split: `test_public/good,bad`
- Few-shot budget: `16` train/good images per object
- Support policy: `dinov2_cls_greedy_coreset`
- Support transforms: `identity`
- Backbone: `dinov2_vitl14`
- Feature layers: `5,11,17,23`
- Feature fusion: `layer_norm_mean`
- Preprocess: MVTec AD2 adapter `no_mask_no_rotation`
- Flow: `flow_epochs=3`, `coupling_layers=2`, `hidden_multiplier=1`,
  `flow_lr=2e-4`, `flow_clamp=1.9`
- NF loss: `tail_weight=0.3`, `tail_top_k_ratio=0.05`,
  `lambda_logdet=1e-3`
- Expansion/scoring: `expansion_budget=1.25`, `density_quantile=0.90`,
  `density_weight=0.0`, `distance_weight=1.0`, `top_percent=0.01`
- Evaluation metrics: `seg_AUROC_0.05`, `seg_F1`
- Cleanup: anomaly maps removed after metric evaluation

## FlowTTE Results

| Object | seg AUROC@0.05 | seg F1 | Best threshold |
| --- | ---: | ---: | ---: |
| can | 0.631038 | 0.005413 | 2.900391 |
| fabric | 0.574536 | 0.142368 | 2.265625 |
| fruit_jelly | 0.630797 | 0.195635 | 3.546875 |
| rice | 0.889664 | 0.598177 | 2.820312 |
| sheet_metal | 0.681807 | 0.288919 | 1.664062 |
| vial | 0.653310 | 0.280517 | 2.929688 |
| wallplugs | 0.772769 | 0.174315 | 3.087891 |
| walnuts | 0.885514 | 0.741347 | 6.484375 |
| **Mean** | **0.714929** | **0.303336** |  |

## Same-Evaluator SuperAD Comparison

Baseline artifact:
`../FMAD-OOD/configs/baselines/recorded_superad16_mvtec_ad2_8object_metrics.json`

| Metric | FlowTTE NF | recorded SuperAD-16 | Delta |
| --- | ---: | ---: | ---: |
| mean seg AUROC@0.05 | 0.714929 | 0.765802 | -0.050873 |
| mean seg F1 | 0.303336 | 0.385534 | -0.082198 |
| AUROC win count | 2/8 | 6/8 |  |
| F1 win count | 2/8 | 6/8 |  |

Class wins for FlowTTE were `can` and `walnuts`. SuperAD won the remaining
classes. This is the strongest local comparison because metric granularity,
object set, split, and evaluator are matched, but `strict_table1_claim_comparable`
is still false because the FlowTTE pipeline does not reproduce all SuperAD
gates and postprocessing details.

## Reported SuperADD Comparison

SuperADD values below are the reported TESTpublic Table 1 numbers from the
paper context, not a same-evaluator rerun.

| Metric | FlowTTE NF | reported SuperADD | Delta |
| --- | ---: | ---: | ---: |
| mean AUROC@0.05 | 0.714929 | 0.839300 | -0.124371 |
| mean F1 | 0.303336 | 0.626113 | -0.322776 |
| AUROC win count | 1/8 | 7/8 |  |
| F1 win count | 1/8 | 7/8 |  |

FlowTTE only wins the reported SuperADD comparison on `can` F1, where SuperADD
reports zero F1. On mean metrics, the current NF setting is far below SuperADD.

## Verdict

`KILL_FOR_CLAIM / NO_CONTINUE` for the current AD2 claim.

The present NF projection/expansion configuration is not competitive with
recorded SuperAD-16 on the same local evaluator, and it is substantially below
reported SuperADD TESTpublic. AD2 exposes the same failure pattern seen earlier:
some categories retain decent AUROC (`rice`, `walnuts`), but thresholded
localization F1 collapses on several objects (`can`, `fabric`, `wallplugs`).

## Artifacts

- Remote:
  `/workspace/results_remote/flow_tte_nf_mvtecad2_all8_shot16_coreset_dw0_20260707_v1`
- Local:
  `results/remote_runs/dsba3/flow_tte_nf_mvtecad2_all8_shot16_coreset_dw0_20260707_v1`
- Metrics:
  `results/remote_runs/dsba3/flow_tte_nf_mvtecad2_all8_shot16_coreset_dw0_20260707_v1/metrics.json`
- Manifest:
  `results/remote_runs/dsba3/flow_tte_nf_mvtecad2_all8_shot16_coreset_dw0_20260707_v1/run_manifest.json`
- Same-evaluator SuperAD comparison:
  `results/remote_runs/dsba3/flow_tte_nf_mvtecad2_all8_shot16_coreset_dw0_20260707_v1/comparison_superad`
- Reported SuperADD comparison:
  `results/remote_runs/dsba3/flow_tte_nf_mvtecad2_all8_shot16_coreset_dw0_20260707_v1/comparison_superadd_reported`
- Cleanup evidence:
  `results/remote_runs/dsba3/flow_tte_nf_mvtecad2_all8_shot16_coreset_dw0_20260707_v1/cleanup_evidence.txt`

## Next

Do not spend more compute on this exact AD2 NF configuration. The next useful
branch should target class-conditional threshold stability and localization F1,
or pivot to incorporating SuperAD/SuperADD-style category gates before another
full AD2 run.
