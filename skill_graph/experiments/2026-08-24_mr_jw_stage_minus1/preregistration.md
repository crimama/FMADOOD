# MR-JW Stage -1 preregistration

Date frozen: 2026-08-24, before any MR-JW implementation result or remote
result was observed. Protocol ID: `mr-jw-stage-minus1-20260824`.

## Decision question and claim boundary

This is a fail-closed support-only design-kill assay. It asks whether requiring
one normal support patch to explain the same query location at three image
resolutions exposes a signal that independent per-resolution nearest-neighbour
matching cannot explain. It is not a benchmark comparison or a method claim.
Even `CONTINUE_DIAGNOSTIC` authorizes only a separately preregistered reduced
TESTpublic experiment; it does not authorize public access in this run.

## Failure-basin divergence

TCCR and the two-band DVT branch already showed that generic cross-scale score
fusion, prediction, and positional correction can collapse to phase/duplicate
nulls or weaken small defects. RN selector/score modifications are also closed,
and CIDER-CF showed that synthetic separation can be generator-driven. MR-JW
therefore changes only the witness constraint: all resolution blocks must use
the same `(support image, normalized grid cell)` witness. There is no learned
predictor, score fusion, RN reweighting, support fitting, query update, or
synthetic-trained parameter. Spatial-pair and support-identity shuffles directly
test whether the claimed joint witness, rather than scale count or template
position, causes any effect. Failure against either shuffle closes MR-JW.

## Static custody

- Dataset root: `/home/hunim/Volume/DATA/mvtec_ad_2`.
- The only admissible image paths are the exact 16 `train/good` paths for each
  requested object in
  `skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/superad16_dinov2_reference_paths.json`,
  SHA-256 `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Objects and dsba3 host-GPU mapping are frozen as `can:0`, `fabric:1`,
  `vial:2`, `walnuts:3` in fixed container `hun_fsad_tta_012`.
- Every query is scored leave-one-support-image-out. No directory enumeration,
  held normal outside the manifest, TESTpublic, TESTprivate, ground truth,
  filename condition, query update, support admission, or dense-map retention
  is permitted.

## Frozen representation and mechanism

- Frozen offline DINOv2 ViT-L/14, final patch-token layer 23, fp32.
- Short-edge resolutions `(224, 448, 672)`, all exact ViT/14 patch multiples.
  One loaded backbone is reused; each RGB image is bicubically resized with
  antialiasing, cropped only to a multiple of 14, and mapped to a common 16x16
  normalized-coordinate grid by adaptive average pooling.
- Every grid token is channel-standardized then L2-normalized. No PCA,
  projection, learned scale, calibration, or object-specific parameter exists.
- For query block `q_s(p)` and support witness `r_s(j)` at scale `s`,

  `d_joint(p) = min_j sqrt(mean_s ||q_s(p) - r_s(j)||^2)`.

  The same witness index `j=(support image, normalized grid cell)` is used at
  every scale. Cosine identities compute the exact normalized Euclidean score.

## Frozen arms

- `M0_MRJW`: natural three-resolution joint witness.
- `B1_independent_per_scale`: mean of three independently minimized
  per-resolution Euclidean distances; this is the immediate scale-fusion floor.
- `B0_single_224`, `B0_single_448`, `B2_exact_RN_672`: single-resolution raw
  1-NN arms. The 672 arm is the invariant raw/RN floor.
- `N0_duplicate_448`: three identical 448 blocks with equal `1/sqrt(3)`
  weighting. It must equal `B0_single_448` within the analytic fp32 bound.
- `N1_spatial_pair_shuffle`: retain support image identity and each scale's
  marginal bank, but independently permute 16x16 cells in non-anchor scales.
- `N2_support_identity_shuffle`: retain normalized grid position and every
  scale's marginal bank, but independently permute support-image identity in
  non-anchor scales.

## Sealed interventions and diagnostics

The three fixed evaluation-only families are `G0_donor_patch`,
`G1_material_blotch`, and `G2_structural_scratch`. Each source support image is
altered once per family with a deterministic protocol/object/source seed. The
source image remains excluded from the normal bank. Masks, family identity,
and responses cannot fit features, arms, weights, scales, or gates. A paired
normal nuisance transform (small gamma/RGB gain plus mild blur) measures false
response. The primary row statistic is synthetic-mask pAUROC@FPR<=0.05 of the
paired score change `score(altered)-score(clean)`; nuisance inflation is the
median absolute paired score change normalized by the arm's clean support-LOO
MAD scale.

Implementation validity requires exact support16 custody, finite nonconstant
maps for every arm, no forbidden access, no retained `anomaly_maps/`, and
`N0_duplicate_448 == B0_single_448` within
`32 * finfo(float32).eps`. Clean support-LOO q95 and max/min image-tail ratios
for M0 must not exceed the worse of N1/N2 beyond the same dtype bound.

## Null-derived gate and verdict

No round effect threshold is used. For each aggregate contrast, the frozen
gate is the 95th percentile of deterministic sign-flipped means of centered
pairwise differences among `N0/N1/N2`; all sign patterns are used up to
16 rows and 65,536 deterministic patterns otherwise. This is the G-2 hard-null
distribution. Family and object coverage are directional safeguards rather
than effect-size thresholds.

Strict `CONTINUE_DIAGNOSTIC` requires all validity/safety checks, aggregate
`M0-B1` and `M0-max(N1,N2)` synthetic pAUC margins above their null gates,
`M0-B2_exact_RN_672` no worse than minus its null gate in every family,
nuisance inflation no worse than the better of B1/B2 beyond its null gate,
positive M0-B1 and M0-shuffle margins on at least three of four objects, and
positive M0-B2 retention in at least two of three families.

`NEAR_MISS` is allowed only when all validity/safety checks pass, both primary
aggregate directions are positive, raw-672 retention stays within the null
gate, at least two families and two objects have positive direction, but an
effect does not clear its null gate. A non-positive aggregate M0-B1 or
M0-shuffle margin is `MECHANISM_REFUTED`; other safety/validity failure is
`KILL_FOR_CLAIM`. No bounded rerun or parameter iteration is preregistered.

Expected cost is four parallel dsba3 shards, approximately 0.5--1.5 GPU-hours
total and less than one day elapsed. A failure remains publishable diagnostic
evidence because it directly distinguishes joint-witness causality from
independent scale fusion, scale duplication, spatial alignment, and support
identity shortcuts.
