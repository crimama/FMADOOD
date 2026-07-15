# Static Flow-LatentBank MVTec AD1 ViT-B/14-register shot sweep

## Motivation and claim boundary

Extend the accepted static 4-shot measurement to 1, 2, and 8 shots. This is a
support-budget scaling measurement with shot count as the only intended factor;
it is not a new method or SOTA claim.

## Locked design

- Dataset: classic MVTec AD, all 15 categories, complete test split.
- Shots/support: 1, 2, and 8; first N `train/good` images, seed 0, identity.
- Backbone: frozen/eval `dinov2_vitb14_reg`, shorter-edge 448 preprocessing.
- Layers/fusion: `[2,5,8,11]`, `layer_norm_mean`.
- Flow: 3 epochs, 2 coupling layers, hidden multiplier 1, LR 2e-4, clamp 1.9,
  tail weight 0.3/top-k 0.05, lambda-logdet 1e-3.
- Score: latent 1-NN distance weight 1.0 plus NF density weight 0.25; density
  quantile 0.90; top 1% image aggregation.
- Static bank: `expansion_budget=1.0`; initial and final sizes must match.
- Excluded: TTE, DVT, context/register conditioning, foreground prior,
  morphology, score calibration, and support augmentation.
- Metrics: class-macro i-AUROC, i-AUPRC, p-AUROC, p-AUPRC,
  p-AUPRO@FPR0.30 using the corrected rank evaluator.
- Execution: dsba5 existing container; GPU 0 runs shots 1 then 8, while GPU 1
  runs shot 2. Each shot is an independent process.

## Reference and verdict gate

The same-condition 4-shot anchor is
`flow_latentbank_mvtecad1_all15_shot4_vitb14reg_static_20260712_v1` with
metrics 0.972100/0.987432/0.973747/0.581160/0.938160.

- `ACCEPT_MEASUREMENT`: every shot completes all 15 categories; metrics are
  finite; support paths are exactly the first N images; memory is static; and
  standalone and manifest metrics agree.
- `REJECT_INVALID`: any locked component differs, support provenance differs,
  memory grows, outputs are incomplete, or any metric is non-finite.
- Strict SOTA superiority remains `BLOCKED_BASELINE`: no same-condition
  VisionAD/SuperAD comparator is available.
