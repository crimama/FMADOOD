# FlowTTE NF MVTec AD1 1-Shot Hyperparameter Search

Date: 2026-07-06

Protocol: `fmad-experiment-protocol`

Remote:

- Server/container: dsba3, `hun_fsad_tta`
- Host GPU: `3`
- In-container CUDA slot: `0`
- Dataset: `/workspace/data/MVTecAD`
- Local pullback root: `results/remote_runs/dsba3`

## Goal

Improve the 1-shot MVTec AD1 FlowTTE NF setting until it exceeds the
VisionAD target tuple used in this project:

- `AUROC=0.974`
- `AUPR/AP=0.990`
- `PRO=0.925`

The comparison below uses this project's `image_AUROC`, `image_AP`, and
`pixel_PRO` fields. Pixel-level AUROC/AP are reported separately.

## Search Axes

- NF density score weight: `0.25`, `0.10`, `0.0`
- Distance score: Euclidean and squared Euclidean
- Image score aggregation: post-hoc top fraction sweep
- Support selection: `first`, partial `visionad_seeded_random`
- Support augmentation: identity and VisionAD-style rotations/flips
- Backbone: `dinov2_vitl14`, `dinov2_vitl14_reg`

## Results

| Run | image AUROC | image AP | pixel AUROC | pixel AP | pixel PRO | pAUROC@0.05 | Top fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default | 0.961653 | 0.981620 | 0.960251 | 0.559529 | 0.928296 | 0.834419 | 0.010 |
| distance-only | 0.969044 | 0.985345 | 0.958813 | 0.563473 | 0.928461 | 0.836975 | 0.005 |
| squared distance | 0.969021 | 0.985204 | 0.959096 | 0.560262 | 0.929178 | 0.836029 | 0.005 |
| distance-only + rot/flip | 0.973474 | 0.986022 | 0.955020 | 0.564001 | 0.930410 | 0.838086 | 0.005 |
| distance-only + rot/flip + ViT-L-reg | 0.980673 | 0.991444 | 0.958969 | 0.534091 | 0.931181 | 0.834212 | 0.005 |
| random support seed 0 | 0.950918 | 0.976694 | 0.960007 | 0.538622 | 0.928944 | 0.830277 | 0.005 |
| random support seed 1 | 0.952856 | 0.979376 | 0.957676 | 0.546218 | 0.930146 | 0.833018 | 0.010 |

Best run:

`flow_tte_nf_mvtecad1_all15_shot1_hp_dist_tte_rotflip_vitl14reg_topsweep_20260706_v1`

Best config:

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
- Image aggregation: top fraction `0.005`

## Verdict

The best single-seed setting exceeds the target tuple for the comparison
fields used here:

- `image_AUROC`: `0.980673` vs `0.974`
- `image_AP`: `0.991444` vs `0.990`
- `pixel_PRO`: `0.931181` vs `0.925`

This is not a strict paper-level claim yet because VisionAD reports mean/std
over five random seeds. It also does not beat VisionAD if the target is
interpreted as pixel-level AUROC/AP, because the best run has
`pixel_AUROC=0.958969` and `pixel_AP=0.534091`.

## Artifacts

- Remote best:
  `/workspace/results_remote/flow_tte_nf_mvtecad1_all15_shot1_hp_dist_tte_rotflip_vitl14reg_topsweep_20260706_v1`
- Local best:
  `results/remote_runs/dsba3/flow_tte_nf_mvtecad1_all15_shot1_hp_dist_tte_rotflip_vitl14reg_topsweep_20260706_v1`
- Best metrics:
  `metrics_best_image_top.json`
- Top fraction sweep:
  `image_top_sweep.json`
- Cleanup evidence:
  `cleanup_evidence.txt`

## Next

For a strict paper-comparable statement, rerun the winning config under a
predefined five-seed support policy and report mean/std. The quick random
support probe with seeds 0 and 1 underperformed the `first` policy, so that
follow-up should be framed as robustness validation rather than expected
improvement.
