# ScopeAD Evaluation on MVTec AD (15 classes, 2-shot & 16-shot)

Hypothesis:
We evaluate whether the new theoretically grounded Renyi multi-scale order aggregation (`scale_order_renyi_cq`)
improves upon the baseline ScopeAD linear0125 configuration on the classic MVTec AD benchmark (15 classes).

## Comparison Setup
- Dataset: MVTec AD (15 classes: bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper)
- Seed: 0 (fixed E9 / Table 5 support manifests for 2-shot and 16-shot).
- Backbone: DINOv2 ViT-L/14, FP16, layers [5, 11, 23], 4 scales (448/672/896/1120).
- Deep mean + 0.125 shallow, Q95 normalization, cap256, 4 bags, CQ99.
- Metrics: Image AUROC (top-1% mean) and Pixel AUROC (full FPR range, matching Table 5).
- Sharded across 4 dsba3 GPUs.
