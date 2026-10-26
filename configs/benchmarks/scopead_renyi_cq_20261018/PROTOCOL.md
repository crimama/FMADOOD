# ScopeAD Renyi Multi-scale Aggregation with Component Quantile (CQ) Evaluation

Hypothesis:
Earlier suites disabled CQ (prediction_postprocess: none) to isolate pure map-level normalization effects.
In this suite, we combine our theoretically grounded Renyi multi-scale order aggregation with standard CQ99 postprocessing.
We test whether the +0.93pp gain from Renyi order aggregation compounds with CQ smoothing to break the 60.0% AUPRO barrier.

## Comparison matrix (2-shot & 16-shot, seed 0, AD2 full8)
- Arms: q95_mean_cq, scale_order_linear_cq, scale_order_renyi_cq, spatial_contrast_order_renyi_cq
- Total: 4 arms x 8 classes = 32 cells (2-shot), 32 cells (16-shot)
- Evaluated on dsba3 GPUs 0-3 with workers=1.
