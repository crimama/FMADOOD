# CIDER-CF Stage -1 preregistration

Date frozen: 2026-08-24, before any CIDER-CF result was observed.  Protocol ID:
`cider-cf-stage-minus1-20260824`.

## Question and custody

Can leave-target-out, cross-synthesis residual consensus define a transferable
low-rank anomaly covariance alternative without learning synthetic labels?
This is a support-only diagnostic on MVTec AD2.  The only admissible paths are
the eight objects' `train/good` directories and the exact SuperAD-16 manifest
with SHA-256 `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
Public/private test and ground-truth enumeration or access is fatal.

## Frozen implementation

- Targets: can, fabric, vial, walnuts; GPU mapping 0, 1, 2, 3 respectively.
- Source: three deterministically hash-ranked normal images from each of the
  other seven objects. Target support is excluded from all source fitting.
- Evaluation: eight hash-ranked target `train/good` images outside support16.
- Frozen DINOv2 ViT-L/14 layer 11, short edge 672, adaptive 16x16 pooling,
  token-wise standardization and L2 normalization.
- Five paired environments: cross-image soft patch, material blotch,
  structural scratch, direct feature-space ascent, and localized frequency
  wave. A deterministic ellipse fixes intervention support across families.
- Normal nuisance is paired exposure/gamma, RGB gain, and mild blur.
- Rank 16; diagonal nuisance variance floor `1e-6`; target H0 is coordinate
  median and `0.9*source nuisance + 0.1*target variance + 1e-4`.
- Each fit uses at most 512 deterministically equally spaced residual patches
  per family, so outside-mask controls do not gain precision merely from area.
- Each of five folds holds one generator out. The primary basis maximizes the
  minimum family energy with a between-family instability penalty.
- Deployed statistic is the single diagonal-plus-low-rank Gaussian LLR. No RN
  fusion, query update, support admission, or target-synthetic fitting occurs.

## Controls and metrics

Primary `M0_CIDER_CF`; controls include pooled-family covariance, outside-mask
fit, and a random same-rank/spectrum basis. Further named arms in the design
remain architectural follow-ups and cannot substitute for these fatal controls.
Metrics are rotating held-family interior pAUROC@.05, held-normal nuisance FPR,
generator-ID balanced accuracy, covariance validity, and hard-null margin.

Strict continuation requires macro pAUROC >=.75, every family >=.65, nuisance
FPR <=.05 macro and <=.08/object, generator probe <=.30 (chance .20), valid
covariances, and >=3/4 object wins over hard nulls. Fatal stop applies to any
forbidden access, any family <.60, generator probe >.35, invalid covariance, or
non-finite maps. Soft continuation is exactly one revision when macro pAUROC
>=.70, nuisance FPR <=.08, probe <=.35, and positive null margin on >=2/4.
Otherwise verdict is `KILL_FOR_CLAIM`. No Stage 1/public experiment is
authorized by anything weaker than the strict gate.
