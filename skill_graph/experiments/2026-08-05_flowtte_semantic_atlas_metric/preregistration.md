# FlowTTE SemanticAtlasMetric Stage -1 Preregistration

Date: 2026-08-05  
Maximum verdict: `PROMISING_DIAGNOSTIC`

## Research question

The retained shallow patch-local Flow is used as an embedding projection, not
as a likelihood scorer.  The question is whether the missing 1-NN geometry can
be supplied by a support-only, semantic-local nuisance scorer:

> Can post-Flow photometric shift directions be learned separately inside
> semantic patch regions, removed softly from query-to-memory residuals, and
> kept orthogonal to clean semantic variation so that shifted-normal distance
> falls without suppressing local defect evidence?

This is not a Flow retraining experiment and does not change `Baseline1`.
`SemanticAtlasMetric` is a memory-conditioned local scorer on frozen
`Baseline1-core` latents.  Because the local operator is selected by the
memory patch cluster, it is not claimed to be one symmetric global metric.

## Negative-evidence boundary

- CF-NTR's single global rank-4 nuisance subspace is closed: it captured the
  imposed physical shift but failed held patch-tail drift and stability.
- The default-shift sequence's global hard projection reduced shift while
  degrading anomaly separation; fixed quadrant-local soft projection was safe
  but too weak.
- The paired-loss diagnostic reduced shift by 9.77% but retained only 0.8707
  of synthetic defect contrast.  High clean-neighbor retention alone did not
  prevent anomaly absorption.
- Existing SFLB query-residual projection is query/test-fitted and asymmetric
  in the wrong way for this support-only mechanism test; it is not reused.

Accordingly, a second global projection, a quadrant router, Flow fine-tuning,
or query-adaptive fitting is out of scope.  The new variable is specifically a
post-Flow semantic mixture with clean-tangent orthogonalization.

## Frozen Baseline1-core

- dataset: MVTec AD2 single-image
- support root on dsba5: `/home/hunim/Volume/DATA/mvtec_ad_2`
- objects: `fruit_jelly, walnuts, vial, fabric`
- references: exact fixed SuperAD-selected 16 paths per object
- support manifest SHA-256:
  `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`
- DINOv3-H+/16, layers `[7,15,23,31]`, `layer_norm_mean`, native 672
- DVT: `position_mean`, alpha `1.0`
- patch-local affine Flow: 2 couplings, width multiplier 1, 3 epochs,
  learning rate `2e-4`, clamp `1.9`, tail weight `0.3`, tail top-k `0.05`,
  logdet regularization `2e-2`
- deterministic support brightness in `[0.8,1.2]`, seed 0
- feature standardization enabled
- scorer: raw latent Euclidean 1-NN with support-LOO calibration
- `density_weight=0.0`

No `test_public`, `test_private`, query image, real anomaly, GT, or
filename-condition routing may be read.  Synthetic local defects generated
from the held support identities are safety stress only, not AD performance.

## Source-disjoint evaluation

The 16 support identities are SHA256-sorted into four folds.  In each fold:

- cluster centers, clean tangents, shift bases, memory, and LOO calibration
  are fit from exactly 12 source images;
- clean, held photometric shifts, and sealed synthetic defects from the other
  four identities are evaluated against the 12-source memory;
- an image and every transform/defect derived from it remain in the same fold;
- each source identity is evaluated exactly once.

Training transforms use deterministic seeds disjoint from held transforms.
Patch correspondence is exact.  Identity transforms are excluded from shift
basis fitting and from the relative shift-reduction denominator.

## Semantic atlas

For each fold, deterministic k-means with `K=8` is fit on clean post-Flow
memory patches.  For semantic cluster `c`:

1. The raw shift direction `u_shift,c` is the top PCA direction of aligned
   augmented-minus-clean latent residuals whose clean patch belongs to `c`.
2. The clean semantic tangent `T_c` is top-rank-4 PCA of clean within-cluster
   top-10 neighbor residuals.
3. The safe direction is
   `u_c = normalize((I - T_c T_c^T) u_shift,c)`.
4. If a cluster contains patches from fewer than 6 fit sources, fewer than
   512 rows, or the orthogonal residual norm ratio is below 0.25, it is
   disabled and uses identity scoring.
5. Otherwise its frozen local operator is
   `W_c = I - 0.25 u_c u_c^T`, whose minimum eigenvalue is 0.75.

For a query patch `q` and memory patch `m`, the memory patch chooses the chart:

`d(q,m) = ||W_{c(m)} (q-m)||_2`.

The anomaly evidence is the minimum over memory patches.  Support LOO and all
shift/defect scores are recomputed with the same scorer.  Registers, NLL,
density, cluster distance, and shift energy are not added to the anomaly score.

## Frozen arms

| Arm | Scorer | Purpose |
|---|---|---|
| `A0_baseline1_core` | raw Euclidean latent 1-NN | exact mechanism baseline |
| `A1_semantic_atlas_orth` | K=8 atlas with clean-tangent orthogonalization | proposed device |
| `A2_semantic_atlas_raw` | same atlas without orthogonalization | tests the safety device |
| `A3_global_rank4` | one post-Flow global rank-4 soft operator, beta matched | closed-basin comparator |
| `A4_cluster_shuffle` | semantic clusters receive another cluster's valid basis | routing causal control |
| `A5_random_rank1` | matched rank-1 random basis per cluster, 3 seeds | generic shrinkage null |

The random control is summarized by its median and q95 envelope.  `K`, beta,
tangent rank, validity thresholds, and fallback rules are not tuned after
support results are observed.

## Frozen diagnostics

- held shifted-normal 1-NN inflation and reduction from `A0`;
- per-object and per-fold shift reduction;
- aligned held shift projected-energy capture;
- clean held-source score mean delta in `A0` standard deviations;
- clean-score standard-deviation ratio;
- top-10 neighbor retention and pair-distance Spearman correlation;
- sealed synthetic local-defect gap/energy retention;
- cluster assignment ARI across fold refits;
- disabled-cluster patch mass;
- basis orthonormality, orthogonal residual ratio, minimum eigenvalue,
  deterministic repeat, finite/nonconstant coverage;
- complete fold, transform, defect, and access-ledger parity.

## Pre-registered gates

`A1_semantic_atlas_orth` passes only if every item below passes using unrounded
values:

1. **Identifiability:** held-shift energy capture exceeds the
   `A4_cluster_shuffle` q95-equivalent control and `A3_global_rank4` by at
   least 0.10 macro.
2. **Shift correction:** macro shifted-normal inflation reduction is at least
   10%, at least 3/4 objects improve, and at least 3/4 folds improve.
3. **Orthogonalization value:** versus `A2`, defect retention improves by at
   least 0.03 macro while shift reduction is no more than 0.02 worse.  If this
   fails, orthogonalization receives no contribution credit.
4. **Non-retread:** versus `A3`, `A1` improves shift reduction by at least
   0.05, or improves defect retention by at least 0.03 at matched shift
   reduction.
5. **Anti-absorption:** synthetic defect retention is at least 0.97 macro and
   at least 0.90 for every object; any object with at least 10% shift
   reduction must retain at least 0.95.
6. **Clean geometry:** absolute clean mean delta is at most 0.05 `A0` standard
   deviations, score standard-deviation ratio is in `[0.8,1.25]`, top-10
   neighbor retention is at least 0.95, and pair-distance Spearman is at least
   0.95.
7. **Atlas robustness:** cluster ARI across fold refits is at least 0.70 macro
   and disabled-cluster patch mass is at most 0.20.
8. **Validity/audit:** all metrics are finite and nonconstant; active bases
   are unit/orthogonal within `1e-4`; minimum eigenvalue is at least 0.75;
   source isolation and support custody are exact.
9. **Causal controls:** neither cluster-shuffled nor random q95 satisfies the
   same shift/safety gate.  Otherwise the result is
   `GENERIC_LOCAL_SHRINKAGE`, not semantic shift correction.

## Verdict and continuation

- All gates pass: `PROMISING_DIAGNOSTIC`.  This authorizes one frozen all-eight
  support-only replication, not public/test evaluation.
- Geometry/defect safety passes and shift reduction is 5--10%: exactly one
  constrained support-only follow-up may inspect atlas sparsity/coverage; no
  strength or K sweep is authorized.
- Shift improves but defect retention fails: `ANOMALY_ABSORPTION / KILL`.
- Identifiability, causal controls, source isolation, or audit fails:
  `KILL_FOR_CLAIM / NO_CONTINUE` (audit failure permits exact rerun only).
- Only after a frozen all-eight support-only replication passes may a separate
  preregistration decide whether any TESTpublic access is warranted.

## Execution contract

- host: `dsba5`
- Docker container: `hun_fsad_tta_012`
- selected host GPU: `3`, verified idle before launch; container sees it as
  `cuda:0`
- isolated immutable source snapshot and unique run root
- per-fold/object compact progress rows visible during execution
- retain configs, summaries, access ledger, logs, source/support hashes, and
  checksums; delete only regenerable dense caches after verified pull

