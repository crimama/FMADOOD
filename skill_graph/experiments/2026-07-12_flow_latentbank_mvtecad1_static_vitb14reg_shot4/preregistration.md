# Static Flow-LatentBank MVTec AD1 ViT-B/14-register 4-shot

## Motivation and claim boundary

Re-run the earliest full 15-class MVTec AD1 Flow + Latent Bank setting while
removing test-time memory expansion and replacing ViT-L/14 with the requested
DINOv2-R ViT-B/14. This is a controlled requested configuration measurement,
not a single-factor TTE ablation because the backbone changes simultaneously.

## Locked design

- Dataset: classic MVTec AD, all 15 categories, complete test split.
- Shots/support: 4, first four `train/good` images, seed 0, identity only.
- Backbone: frozen `dinov2_vitb14_reg` with shorter-edge 448 preprocessing.
- Layers: `[2,5,8,11]`, the depth-relative ViT-B counterpart of the prior
  ViT-L `[5,11,17,23]`; fusion remains `layer_norm_mean`.
- Flow: 3 epochs, 2 coupling layers, hidden multiplier 1, LR 2e-4, clamp 1.9,
  tail weight 0.3/top-k 0.05, lambda-logdet 1e-3.
- Score: latent 1-NN distance weight 1.0 plus the existing NF density penalty
  weight 0.25; density quantile 0.90; image top 1% aggregation.
- Static bank: `expansion_budget=1.0`; initial and final memory sizes must match.
- Excluded: TTE expansion, DVT denoising, context, register-token conditioning,
  score-field calibration, foreground prior, morphology, and support transforms.
- Metrics: class-macro i-AUROC, i-AUPRC, p-AUROC, p-AUPRC, p-AUPRO@FPR0.30.
- Execution: dsba5 fixed project container; host GPU 0, with GPU 1 retained as
  an available recovery resource.

## Reference and verdict gate

Historical reference:
`results/remote_runs/dsba3/flow_tte_nf_mvtecad1_all15_shot4_20260706_v1_pixelpro`.
It scored 0.969631/0.983877/0.964028/0.576137/0.936470 on the five requested
metrics, but used ViT-L/14, TTE budget 1.25, and the legacy float16 pixel-rank
writer. It is contextual rather than a strict paired baseline.

- `ACCEPT_MEASUREMENT`: all 15 categories finish, metrics are finite, selected
  supports match first-four policy, and memory is static for every category.
- `REJECT_INVALID`: any requested component is active, a metric is non-finite,
  support provenance differs, or memory grows.
- Strict superiority remains `BLOCKED_BASELINE` because both backbone and pixel
  metric implementation differ from the historical run.
