# FlowTTE NF MVTec AD1 Shot Scaling vs VisionAD

Date: 2026-07-07

Protocol: `fmad-experiment-protocol`

Remote:

- Server/container: dsba3, `hun_fsad_tta`
- Host GPU: `3`
- In-container CUDA slot: `0`
- Dataset: `/workspace/data/MVTecAD`
- Local pullback root: `results/remote_runs/dsba3`

## Goal

Extend the best 1-shot FlowTTE NF setting to `2-shot` and `4-shot`, then
compare against the MVTecAD rows reported by VisionAD.

VisionAD reference values are from Table 1 of "Search is All You Need for
Few-shot Anomaly Detection" and are mean/std over five random seeds.

## Shared FlowTTE NF Setting

- Backbone: `dinov2_vitl14_reg`
- Preprocess: `fmad_shorter_edge`, crop `448`
- Feature layers: `5,11,17,23`
- Feature fusion: `layer_norm_mean`
- Support selection: `first`
- Support transforms: `identity`, `rot90`, `rot180`, `rot270`,
  `flip_vertical`, `flip_horizontal`
- Flow: `flow_epochs=3`, `coupling_layers=2`, `hidden_multiplier=1`,
  `flow_lr=2e-4`, `flow_clamp=1.9`
- NF loss: `tail_weight=0.3`, `tail_top_k_ratio=0.05`,
  `lambda_logdet=1e-3`
- Expansion/scoring: `expansion_budget=1.25`, `density_quantile=0.90`,
  `density_weight=0.0`, `distance_weight=1.0`
- Image aggregation: post-hoc top fraction sweep, best by image AUROC
- Split/classes: full 15-class MVTec AD1 test split

## Results

| Shot | Run | image AUROC | image AP | pixel AUROC | pixel AP | pixel PRO | seg AUROC@0.05 | seg F1 | Top fraction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `flow_tte_nf_mvtecad1_all15_shot1_hp_dist_tte_rotflip_vitl14reg_topsweep_20260706_v1` | 0.980673 | 0.991444 | 0.958969 | 0.534091 | 0.931181 | 0.834212 | 0.562890 | 0.0050 |
| 2 | `flow_tte_nf_mvtecad1_all15_shot2_hp_dist_tte_rotflip_vitl14reg_topsweep_20260707_v1` | 0.981360 | 0.991453 | 0.959126 | 0.530315 | 0.933734 | 0.834619 | 0.562695 | 0.0075 |
| 4 | `flow_tte_nf_mvtecad1_all15_shot4_hp_dist_tte_rotflip_vitl14reg_topsweep_20260707_v1` | 0.984649 | 0.992106 | 0.958491 | 0.531250 | 0.934834 | 0.834450 | 0.562652 | 0.0075 |

## VisionAD Comparison

This table compares only the fields with a direct project-level counterpart:
VisionAD `AUROC`, VisionAD `AUPR`, and VisionAD `PRO` against FlowTTE
`image_AUROC`, `image_AP`, and `pixel_PRO`.

| Shot | FlowTTE image AUROC | VisionAD AUROC | Delta | FlowTTE image AP | VisionAD AUPR | Delta | FlowTTE PRO | VisionAD PRO | Delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 98.067 | 97.400 | +0.667 | 99.144 | 99.000 | +0.144 | 93.118 | 92.500 | +0.618 |
| 2 | 98.136 | 98.100 | +0.036 | 99.145 | 99.300 | -0.155 | 93.373 | 93.200 | +0.173 |
| 4 | 98.465 | 98.600 | -0.135 | 99.211 | 99.500 | -0.289 | 93.483 | 93.700 | -0.217 |

## Verdict

- `1-shot`: FlowTTE NF exceeds VisionAD on image AUROC, image AP/AUPR, and
  PRO under this single-seed project evaluation.
- `2-shot`: FlowTTE NF is slightly higher on image AUROC and PRO, but lower on
  image AP/AUPR.
- `4-shot`: FlowTTE NF is slightly below VisionAD on all three directly
  compared fields.

The scaling behavior is positive for image AUROC and PRO, but the method does
not gain enough at 4-shot to match VisionAD's five-seed mean. Pixel AP remains
low, so the localization confidence ranking still needs work.

## Caveats

- VisionAD reports five-seed mean/std; these FlowTTE runs are single-seed
  support-policy evaluations.
- VisionAD's `pAUROC` column should not be treated as the same metric as this
  project's `seg_AUROC_0.05`. Under that strict interpretation, FlowTTE does
  not match VisionAD pAUROC at any shot: `83.421`, `83.462`, and `83.445`
  versus VisionAD `96.2`, `96.6`, and `96.9`.
- The `first` support policy was chosen because the quick random-support probe
  underperformed it for the 1-shot target run.

## Artifacts

- 1-shot local:
  `results/remote_runs/dsba3/flow_tte_nf_mvtecad1_all15_shot1_hp_dist_tte_rotflip_vitl14reg_topsweep_20260706_v1`
- 2-shot local:
  `results/remote_runs/dsba3/flow_tte_nf_mvtecad1_all15_shot2_hp_dist_tte_rotflip_vitl14reg_topsweep_20260707_v1`
- 4-shot local:
  `results/remote_runs/dsba3/flow_tte_nf_mvtecad1_all15_shot4_hp_dist_tte_rotflip_vitl14reg_topsweep_20260707_v1`
- Metrics file per run: `metrics_best_image_top.json`
- Top fraction sweep per run: `image_top_sweep.json`
- Cleanup evidence per run: `cleanup_evidence.txt`

## Next

For a stricter VisionAD comparison, rerun a predetermined five-seed support
selection policy and report mean/std. For method improvement, prioritize pixel
AP and pAUROC-like localization ranking rather than image AUROC alone.
