# Baseline1-core + SemanticAtlasMetric Stage -1 Preregistration

## Objective

Test whether a post-Flow, memory-conditioned semantic atlas distance can reduce support-derived photometric shift inflation while preserving local anomaly evidence.

This is not a Transformer Flow retry and not a Flow retraining experiment. The frozen detector is `Baseline1-core`; only the 1-NN distance used after the Flow projection is changed.

## Frozen Baseline

- Method identity: `Baseline1-core`
- Backbone: DINOv3-H+/16
- Layers: `[7, 15, 23, 31]`
- Feature fusion: `layer_norm_mean`
- DVT: `position_mean`, alpha `1.0`
- Support: fixed SuperAD-16 manifest
- Flow: patch-local MLP affine coupling, 2 coupling layers, hidden multiplier 1
- Stage-A training: 3 epochs, lr `2e-4`, clamp `1.9`, tail weight `0.3`, tail top-k `0.05`, logdet `0.02`
- Score: latent 1-NN, support LOO standardization, `density_weight=0.0`

## Data Custody

Stage -1 is support-only.

- Allowed: fixed 16 `train/good` support images per object and deterministic photometric augmentations derived from them.
- Forbidden: `test_public`, `test_private`, ground-truth masks, public/private filenames, query adaptation, memory updates from query images.
- Objects: `fruit_jelly`, `walnuts`, `vial`, `fabric`.
- Support manifest SHA256: `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`.

## Candidate Mechanism

Name: `SemanticAtlasMetric`.

For clean post-Flow support latents, fit a semantic atlas:

1. Cluster clean support patch latents with deterministic k-means.
2. In each cluster, estimate a rank-1 photometric shift direction from aligned clean-vs-augmentation residuals.
3. Estimate a clean semantic tangent from within-cluster clean neighbor differences.
4. Orthogonalize the shift direction against the clean tangent.
5. Use a bounded local SPD shrinkage matrix for distances involving memory patches from that cluster.

Formula:

```text
u_raw,c = top-1 PCA direction of z_aug - z_clean within cluster c
T_c     = top-r PCA basis of clean neighbor differences within cluster c
u_c     = normalize((I - T_c T_c^T) u_raw,c)
W_c     = I - beta u_c u_c^T
d(q,m)  = || W_cluster(m) (q - m) ||_2
```

Fixed parameters:

- `K=8`
- `semantic_tangent_rank=4`
- `shift_rank=1`
- `beta=0.25`
- `min_cluster_sources=6`
- `min_cluster_rows=512`
- `orthogonal_residual_min_ratio=0.25`
- invalid clusters use identity distance.

`W_c` is memory-conditioned, so this is a pseudolocal distance rather than a symmetric global metric.

## Folds

Use 4 source-identity folds.

- Fit clusters, bases, LOO calibration, and memory on 12 support images.
- Evaluate held shifted-normal and synthetic defect probes on the 4 held support images.
- Held images are never in the fit memory for their fold.

## Arms

- `A0_Baseline1_core`: frozen raw Euclidean latent 1-NN.
- `A1_SemanticAtlasMetric`: semantic atlas with clean-tangent orthogonalization.
- `A2_UnorthogonalizedAtlas`: same clusters and shift basis, no clean-tangent orthogonalization.
- `A3_GlobalRank4Metric`: global post-Flow rank-4 bounded metric, beta-matched.
- `A4_ShuffledAtlas`: valid cluster bases assigned to wrong semantic clusters.
- `A5_RandomAtlas`: deterministic random rank-1 basis per cluster.

No hyperparameter sweep is authorized in this Stage -1 run.

## Metrics

Per object and arm:

- held shifted-normal inflation
- relative shift-inflation reduction vs `A0`
- synthetic local-defect gap retention vs `A0`
- clean LOO mean delta in `A0` LOO std units
- LOO std ratio
- top-10 hard neighbor retention
- pair-distance Spearman
- basis validity and disabled patch mass
- projected shift energy capture
- projected defect energy capture
- shift/defect selectivity

## Gates

`A1_SemanticAtlasMetric` is promotable only if all hold:

- Macro shifted-normal inflation reduction `>= 10%`.
- Positive shift reduction on at least `3/4` objects.
- Positive shift reduction in at least `3/4` folds.
- Macro defect retention `>= 0.97`.
- Every-object defect retention `>= 0.90`.
- For any object with shift reduction `>= 10%`, defect retention `>= 0.95`.
- Absolute clean LOO delta `<= 0.05`.
- LOO std ratio in `[0.8, 1.25]`.
- Top-10 neighbor retention `>= 0.95`.
- Pair-distance Spearman `>= 0.95`.
- Disabled-cluster patch mass `<= 0.20`.
- `A1` shift capture beats `A4` and `A5` by at least `0.10` macro.
- `A1` beats global `A3` by at least `0.05` shift reduction or at least `0.03` defect retention at matched shift behavior.
- `A1` retains defects better than `A2` by at least `0.03` macro with no more than `0.02` shift-reduction loss, otherwise orthogonalization is not credited.

## Kill Conditions

- Any public/private/GT/query-adaptation access.
- Baseline or identity scorer invalidity.
- Shuffled or random atlas passes the same promotion gate.
- Shift reduction improves while macro defect retention falls below `0.97`.
- Global rank-4 explains the same outcome as the semantic atlas.
- Results are indistinguishable from generic local shrinkage.

## Continuation Rule

- `MECHANISM_PASS`: authorize exactly one all-8 support-only replication of `A0`, `A1`, and null controls. Public/test evaluation remains unauthorized.
- `CONTINUE_DIAGNOSTIC`: if all safety gates pass and macro shift reduction is in `[5%, 10%)`, allow one bounded support-only diagnostic rerun with the already frozen safer branch only.
- `KILL_FOR_CLAIM`: if shift and anomaly evidence are not separated, stop this claim path.
