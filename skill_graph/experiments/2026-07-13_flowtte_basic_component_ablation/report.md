# DINOv2-L Basic Component Ablation

Date: 2026-07-13

## Contract

- MVTec AD2 all eight public objects.
- DINOv2 ViT-L/14, layers 5/11/17/23.
- Resolution 672, except sheet_metal 448.
- Exact fixed SuperAD-selected 16 normal supports, seed 0.
- Static memory, density weight 0.25, fixed close/fill/erode morphology.
- Full Basic versus four independent leave-one-component-out arms.
- The no-Flow arm keeps the support 1-NN bank and replaces the NF transform
  with identity so that scoring remains defined.

## Macro results

All values are percentages. Raw F1 means best-threshold F1 before binary
morphology. The no-RGB arm uses the matched unguided map; all other rows use
guided-r8 maps.

| Arm | Flow + bank | LOO | DVT | RGB guide | p-AUROC@0.05 | Raw F1 | Morph F1 | Delta p-AUROC | Delta Morph F1 |
|---|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| Full Basic | yes | yes | yes | yes | 78.297 | 37.550 | 43.469 | - | - |
| -Flow | no | yes | yes | yes | 77.949 | 36.497 | 42.380 | -0.348 | -1.089 |
| -LOO | yes | no | yes | yes | 78.194 | 38.108 | 44.107 | -0.104 | +0.638 |
| -DVT | yes | yes | no | yes | 76.965 | 37.260 | 43.053 | -1.332 | -0.417 |
| -RGB Guide | yes | yes | yes | no | 77.335 | 34.904 | 41.654 | -0.962 | -1.815 |

## Interpretation

- DVT has the largest continuous-ranking contribution: removing it costs
  1.332 p-AUROC points.
- RGB guidance has the largest segmentation contribution: removing it costs
  2.646 raw-F1 and 1.815 morphology-F1 points.
- The learned Flow contributes positively but modestly: 0.348 p-AUROC and
  1.089 morphology-F1 points over the identity support-bank control.
- LOO standardization is mixed. It adds 0.104 p-AUROC point but reduces raw
  F1 by 0.558 and morphology F1 by 0.638 point. It should not be described as
  an unconditional improvement under this Basic contract.

## Operational decision

LOO standardization was removed from the Basic default after this ablation.
The off arm becomes the new anchor at 78.194 p-AUROC, 38.108 raw F1, and
44.107 morphology F1. The on/off switch remains available for reproduction.

## Audit

- Completed arms: 5/5; completed object evaluations: 40/40.
- dsba3 GPUs 0-3 ran Full, -Flow, -LOO, and -DVT.
- dsba5 GPUs 0-1 split the eight -RGB objects.
- No Traceback, CUDA OOM, or RuntimeError was detected.
- All raw manifests retain shots 16, seed 0, expansion budget 1.0, density
  weight 0.25, coupling depth 2, and hidden multiplier 1.
- Remote and local aggregate artifacts are under
  `flowtte_basic_component_ablation_fixed16_20260713_v1/configs/<arm>`.
