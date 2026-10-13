# Patch-Coverage RN TESTpublic Preregistration

Date: 2026-08-15
Status: `PRE-REGISTERED / OPS_READY`
Protocol: `patch-coverage-rn-testpublic-20260815`

## Motivation

The previous all-eight Simple-RN diagnostic showed that full exact RN remains
the strongest arm and that image-level representative support selection is not
reliably better than an equal-budget random control. The next diagnostic keeps
the successful exact normalized 1NN rule fixed, but moves compression to the
unit RN actually uses: spatial patch memory.

## Negative-Evidence Intake

This is not TTA, query adaptation, tangent scoring, support admission, score
fusion, prompt prior, or DVT tuning. It is a patch-memory construction audit
for the full-support RN baseline. The likely failure basins are RN-FMLK
duplication, random-control absorption, and loss of rare normal coverage inside
spatial cells.

## Dataset and Boundary

- Target dataset: MVTec AD2 single-image.
- Split: full `test_public/good,bad` for all eight public objects.
- Support: full `train/good` only, pinned manifest
  `ac084290bd40d171f12e876b45c3baf9b2269010cc2e62a0e91d7c4b80badd53`.
- Selector grid: `16x16`.
- Selector mechanism: for each support source and cell, mean-pool that cell's
  native tokens and normalize the resulting selection descriptor.  Build one
  deterministic farthest-first source ordering per cell, starting nearest the
  cell centroid with `(SHA256(relative_path), relative_path)` tie order.
  P25/P50/P75 are nested source-prefixes; N50 uses the same per-cell source
  quota with protocol/object/cell/path SHA ordering.  Scoring materializes all
  actual native patches from each selected source/cell; pooled descriptors and
  synthetic centroids never enter RN memory.
- Query scoring: unchanged exact normalized 1NN RN.
- Dense maps: RAM-only; compact artifacts only.
- Forbidden: TTA, query-time update, memory update, parameter update at score
  time, test-private access, persisted dense arrays or maps.

## Arms

- `B0_full_RN`
- `P25_cell_kcenter_RN`
- `P50_cell_kcenter_RN`
- `P75_cell_kcenter_RN`
- `N50_cell_random_RN`

## Remote Ownership

- dsba3: `can,fabric,walnuts`.
- dsba5: `fruit_jelly,rice,sheet_metal,vial,wallplugs`.
- Canonical GPU mapping: host GPU 3 to container `cuda:0`.
- Remote controller must honor `TARGET=dsba3|dsba5` for every remote mode,
  including `preflight`, `submit`, `pull`, `cleanup`, `resume`, `stop`, and
  `status`.

## Metrics and Gates

Primary metric: macro `seg_AUROC_0.05` from evaluator-native
`pooled_pauroc_0.05_float16`.

Secondary metrics:

- `seg_F1` from `pooled_oracle_f1_float16`
- `seg_AUPRO_0.05` from `pooled_aupro_0.05`
- `pixel_AP` from `pooled_pixel_ap_float16`
- complete finite nonconstant map coverage

Performance success requires:

- best P arm has macro pAUROC greater than `B0_full_RN + 0.002`;
- F1 and AUPRO harm for that best P arm are not worse than `0.005`;
- `P50_cell_kcenter_RN` beats `N50_cell_random_RN` by more than `0.002`
  macro pAUROC.

Efficiency continuation is justified when any P arm is within `0.002` pAUROC
and `0.005` F1/AUPRO of B0 while reducing patch memory by at least `2x`.

Historical SuperAD16 is context only. This fullshot diagnostic cannot make a
paper-aligned method claim without a same-condition SuperAD/RN-FMLK gate.
