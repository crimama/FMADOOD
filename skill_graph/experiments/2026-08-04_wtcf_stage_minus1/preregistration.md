# WTCF Stage -1 Preregistration

Date: 2026-08-04  
Status: `FROZEN_BEFORE_WTCF_FEATURE_EXTRACTION`

## Question

Can a normal-only Flow be aligned with its deployed exact latent 1-NN score
without erasing unsupported feature directions, and does a repeated
cross-image normal-residual witness add value beyond generic cross-fit or
global-isometry controls?

## Frozen detector contract

- MVTec AD 2 single-image setting;
- DINOv2 ViT-L/14 layers `[5, 11, 17, 23]`, layer-normalized mean fusion;
- 16 normal support images, DVT off, density score off;
- two affine coupling layers, hidden multiplier 1, three initial NLL epochs;
- no query adaptation, memory expansion, routing, public/private test access,
  or ground-truth access;
- final score is exact Euclidean hard 1-NN to one static full-support latent bank.

## Frozen Stage -1 scope

- Objects: `can`, `fruit_jelly` (prior weak), `rice`, `fabric` (prior strong),
  and `sheet_metal` (control).
- Support seeds: `0, 1, 2`. Seed 0 uses the exact fixed SuperAD-selected paths.
  Seeds 1 and 2 use a deterministic SHA-256 ordering of `train/good` paths;
  all three 16-path manifests are sealed before backbone feature extraction.
- Four deterministic image-disjoint folds per object/seed.
- Post-hoc maps use exactly one SHA-first diagnostic source from each fold
  (four sources per object/seed); each source is excluded from every memory or
  linear-null fit used to score its derived C/S/D/SD maps.
- Synthetic C/S/D/SD stimuli are post-hoc non-absorption diagnostics only.
  They cannot affect optimization, rollback, hyperparameters, arm selection,
  or continuation.

## Frozen arms

1. `a6`
2. `a7_identity`
3. `pca_residual`
4. `shrinkage_mahalanobis`
5. `crossfit_only`
6. `global_isometry`
7. `random_witness`
8. `witness_only`
9. `wtcf`

All learned Flow arms start from the exact same fitted A6 state and use 300
updates with a shared deterministic source/row schedule. Linear nulls are fit
only from the same allowed support features. PCA uses rank 32 from at most
4096 deterministic training patches. Mahalanobis uses a diagonal covariance
with shrinkage `0.10` toward the mean variance and exact 1-NN afterward.

## WTCF constants

- Cross-fit soft-1NN temperature `0.05`, held-normal top-tail CVaR fraction
  `0.05`, weak normalized NLL weight `0.05`.
- Mutual cross-image nearest residual candidates are sampled across different
  folds. A witness requires absolute cosine stability at least `0.85`, at
  least three distinct source pairs and three fold-pairs. At most eight
  witnesses are retained.
- Witness JVP sensitivity relative to A6 is allowed in `[0.55, 1.10]` and is
  targeted at no more than `0.85`. Unsupported deterministic probes are held
  in `[0.90, 1.10]`. `global_isometry` applies `[0.90, 1.10]` to both sets.
- Function-space rollback is evaluated every five updates on a frozen clean
  probe. A proposed state is rejected if any A6-relative bound fails:
  latent RMS/effective rank/score standard deviation outside `[0.80, 1.25]`,
  pair-distance Spearman below `0.95`, neighbor retention below `0.80`,
  logdet quantile drift above one A6 standard deviation, coupling-logdet q99
  above `1.25x`, or inverse error above `1e-5`.

## Validity before performance

- identical manifest and fold hashes across arms;
- 100% selected-support access coverage and zero test/label access;
- finite, complete, nonconstant scores;
- soft/hard Spearman at least `0.98`, top-0.5% overlap at least `0.90`, and
  normalized q995 difference at most `0.01`;
- final witness/non-witness JVP bounds and all rollback geometry bounds pass;
- witness count, source-pair diversity, fold diversity, accepted/rejected
  updates, and zero-witness cases are reported.

A zero-witness sentinel is a valid reported outcome but cannot validate the
WTCF-specific mechanism; it triggers an early claim kill and blocks the
remaining Stage -1 cases. Final accepted witness JVP ratios must also reach
the preregistered target ceiling `0.85`, not merely stay below `1.10`.

Failure of any validity condition blocks detector-metric interpretation.

## Keep/kill gate

Relative to A6, WTCF must lower held-normal q995 by at least `0.01` in
A6-normalized score units; remain noninferior on post-hoc synthetic pAUROC,
support-fixed F1, and AU-PRO within `0.005/0.010/0.010`; materially improve at
least two by `0.005/0.010/0.010`; beat each mechanistic/null arm on the joint
gate; avoid any object collapse worse than `0.02/0.03/0.03`; and agree in
direction on at least two of three seeds per object.

For each null, “beat on the joint gate” means WTCF lowers held-normal q995 by
at least `0.005`, is noninferior on all three detector metrics within
`0.005/0.010/0.010`, and materially improves at least two detector metrics by
`0.005/0.010/0.010`. Seed direction agreement requires lower tail plus at
least two nonnegative detector deltas in at least two seeds of each object.

Ties with PCA, random witness, or global isometry kill the WTCF-specific
mechanism claim. A completed failure is terminal for this frozen candidate;
there is no post-result tuning or scientific retry.
