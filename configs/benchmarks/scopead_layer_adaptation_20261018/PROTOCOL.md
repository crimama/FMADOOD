# ScopeAD Support-Adaptive Layer Weighting (Stability-Modulated Shallow Fusion)

Hypothesis:
In fixed linear fusion (gamma=0.125), all classes receive identical shallow layer weight.
However, normal support LSO distributions reveal that certain objects have high shallow background variance/noise (e.g. transparent or complex backgrounds in vial, fruit_jelly).
By scaling gamma adaptively according to the relative stability of shallow-vs-deep support distributions (1 / tail spread), we suppress shallow false-alarm noise when shallow features are unstable, while maintaining texture sensitivity when stable.

## Comparison matrix (2-shot & 16-shot, seed 0, AD2 full8)
- Arms: fixed_mean_cq, adaptive_mean_cq, fixed_renyi_cq, adaptive_renyi_cq
- Total: 4 arms x 8 classes = 32 cells (2-shot), 32 cells (16-shot)
- Evaluated on dsba3 GPUs 0-3 with workers=1.
