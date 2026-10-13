# Score-aligned Residual Attribution + kNN Scorer Sentinel Stage -1d

## Scope

Stage -1c showed that a repeatable image-mean direction does not reduce deployed 1-NN tail.
This compact sentinel tests whether score-causing patch residuals instead lie in a support-estimable
rank-4 nuisance subspace, and whether the conclusion depends on 1-NN scoring. It is not a method
or benchmark claim and does not reopen the killed global rank-projection branch.

- target: MVTec AD2 single-image, fixed-16 normal support, sentinel objects `fruit_jelly`, `vial`.
- 4 folds, 12 fit / 4 held; deterministic 256 patch positions per image.
- feature: frozen DINOv3-H+/16 fused feature; no Flow/DVT/query adaptation/test/public/private/GT.
- nuisance samples: fit-support `S_global` and `S_smooth` patch displacements after removing their
  image-mean direction; fixed rank 4, no rank/strength sweep.
- held condition: distinct `U_unseen`; safety: `U+D_local`, `U+D_broad` masks used only for contrast.
- matched null: seeded random orthonormal rank-4 subspace.

## Fixed scorers

The same sampled memory/query is scored with: `1NN`, mean distance of `5NN`, mean distance of
`15NN`, and a fixed soft-min over 15 neighbors with temperature equal to the clean-fit median
1NN distance. No scorer is selected post hoc.

## Gate

For each object: held NN-residual energy captured by rank-4 nuisance subspace must exceed the
random subspace by `.10`; projection must reduce unseen q995 inflation by `>=10%` for primary
1NN and at least two of three secondary scorers; local/broad defect contrast retention must be
`>=.95`; clean pair-distance Spearman and NN retention `>=.95`. Both sentinel objects passing
allows four-object continuation only. Otherwise this support-estimable residual-subspace branch
is stopped. SuperAD/segmentation metrics are `N/A` at this attribution stage.
