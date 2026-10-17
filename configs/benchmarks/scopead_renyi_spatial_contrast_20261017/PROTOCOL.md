# Renyi Exponential Order Statistics and Spatial Background Contrast Matching

Hypothesis:
1. Extreme value theory shows that tail anomalies follow exponential/GPD distributions. Under standard exponential random variables, the expected spacing of order statistics follows Renyi representation [0.0625, 0.1458, 0.2708, 0.5208].
2. Resolution-specific background shifts introduce false positive noise across multi-scale maps. Aligning each resolution background median while preserving overall mean removes DC bias across resolutions before order weighting.

## Fixed Comparison Setup
- Dataset: MVTec AD 2 test_public, seed0, VOID_support_2.json (2-shot).
- Backbone: DINOv2 ViT-L/14, FP16, layers [5, 11, 23], 4 scales (448/672/896/1120).
- Deep mean + 0.125 shallow, Q95 normalization, cap256, 4 bags, no CQ.
- Primary metric: AUPRO@0.05, Secondary: pAUROC@0.05.
- 4 arms x 8 classes = 32 cells across 4 dsba3 GPUs.
