# Simple-RN Representative + DVT TESTpublic Preregistration

Date: 2026-08-15
Status: `PRE-REGISTERED / RUNNING`
Protocol: `simple-rn-representative-dvt-testpublic-20260815`

## Motivation

Recent TESTpublic diagnostics repeatedly failed when they added gates,
tangent scores, or query-time updates on top of exact RN.  The strongest
surviving baseline is the simplest one: frozen DINOv2-L final-layer exact
normalized 1NN RN.  This experiment keeps that inference rule fixed and asks
only whether support-bank representation and support-only DVT improve it.

## Negative-Evidence Intake

This is not a new gating, score-fusion, tangent, TTA, or query-adaptation
branch.  Query updates, parameter updates at score time, private-test access,
and dense-map persistence are forbidden.  The likely failure basins are
RN-FMLK duplication, representative subset undercoverage, DVT anomaly
absorption, and position-shuffle null absorption.

## Dataset and Boundary

- Dataset: MVTec AD2 single-image, all eight public objects.
- Split: full `test_public/good,bad`.
- Support: pinned full `train/good` manifest
  `ac084290bd40d171f12e876b45c3baf9b2269010cc2e62a0e91d7c4b80badd53`.
- Runner: `scripts/run_simple_rn_testpublic.py`.
- Object schema: `simple-rn-testpublic-object-v1`.
- Aggregate: `scripts/aggregate_simple_rn_testpublic.py`.
- Remote sharding: dsba3 `can,fabric,walnuts`; dsba5
  `fruit_jelly,rice,sheet_metal,vial,wallplugs`.
- GPU mapping: host GPU 3 to in-container `cuda:0`.

## Arms

- `B0_full_RN`
- `R16_repr_RN`
- `R32_repr_RN`
- `R64_repr_RN`
- `R128_repr_RN`
- `N64_random_RN`
- `DVT_full_RN`
- `DVT_R64_repr_RN`
- `N_DVT_R64_position_shuffle_RN`

## Metrics and Gates

Primary metric is macro `seg_AUROC_0.05`; secondary metrics are macro
`seg_F1`, `seg_AUPRO_0.05`, `pixel_AP`, and complete map coverage.

Required diagnostic panels:

- representative K trend vs full RN;
- `R64_repr_RN` vs `N64_random_RN`;
- `DVT_full_RN` vs `B0_full_RN`;
- `DVT_R64_repr_RN` vs `R64_repr_RN`;
- `DVT_R64_repr_RN` vs `N_DVT_R64_position_shuffle_RN`.

Strict diagnostic signal requires complete finite nonconstant coverage,
positive macro pAUROC for either representative selection or DVT, no F1
collapse worse than 0.02 for the best arm, and DVT_R64 beating the position
shuffle null.  Historical SuperAD16 remains context only for this full-manifest
runner path.
