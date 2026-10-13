# SPCT B1 Full-Support TESTpublic Stage -1 Preregistration

Date: 2026-08-12  
Protocol ID: `spct-b1-fullsupport-testpublic-20260812`  
Status: `PRE-REGISTERED / PENDING`  
Maximum verdict: `PROMISING_DIAGNOSTIC`

## 1. Boundary and custody

This is a TESTpublic-only launch over all eight MVTec AD2 single-image
objects. The accessible support pool is the full `train/good` population. No
test-private split, no query-time update, no support update, no hidden split,
and no benchmark oracle access is allowed.

The support manifest is the filename-list rewrite of the local
`/home/hun/Volume/DATA/mvtec_ad_2` tree onto the canonical remote prefix
`/home/hunim/Volume/DATA/mvtec_ad_2`. The manifest is frozen at
`full_train_good_paths.json` with SHA-256
`ac084290bd40d171f12e876b45c3baf9b2269010cc2e62a0e91d7c4b80badd53`.

All dense maps remain RAM-only. Public ground truth is opened only after all
maps for the current query batch have been staged in memory.

## 2. Frozen inputs

| object | full `train/good` count | TESTpublic good / bad |
| --- | ---: | ---: |
| can | 412 | 72 / 90 |
| fabric | 387 | 66 / 90 |
| fruit_jelly | 263 | 20 / 60 |
| rice | 313 | 42 / 90 |
| sheet_metal | 137 | 24 / 90 |
| vial | 291 | 35 / 105 |
| wallplugs | 293 | 60 / 90 |
| walnuts | 432 | 60 / 90 |

Resolution is `672` for every object except `sheet_metal`, which is frozen at
`448`. The total TESTpublic count is `600`.

Frozen backbone: DINOv2 L/14 snapshot. No backbone, feature, resolution,
preprocessing, map shape, or score fusion retuning is allowed.

## 3. Arms

| arm | definition |
| --- | --- |
| `B0` | matched raw baseline on the same full-support bank |
| `B1` | unconditional tangent on the same full-support bank |
| `N0` | tangent-shuffle null with tangent assignment permuted |
| `RN_FMLK_full` | conventional normalized final-layer 1NN comparator |

All arms use the same support custody and the same TESTpublic evaluation
surface.

## 4. Metrics and gates

Primary metric: macro `seg_AUROC_0.05`.
Secondary metrics: macro `seg_F1`, macro `seg_AUPRO_0.05`, pixel AP, and
coverage.

The run is not benchmark-comparable with SuperAD16 because the support budget
is different. Absolute TESTpublic performance is the decision target; monotonic
support scaling is not a claim criterion.

Decision gates:

1. `B1` must beat `RN_FMLK_full` on macro `seg_AUROC_0.05`.
2. `B1` must not lose more than `0.01` macro `seg_F1` or macro
   `seg_AUPRO_0.05` versus `RN_FMLK_full`.
3. `B1` must beat both `B0` and `N0` by at least `0.005` macro
   `seg_AUROC_0.05`.
4. No object may lose more than `0.03` macro `seg_AUROC_0.05` versus `B0`.

Passing all gates can support only `PROMISING_DIAGNOSTIC`. It cannot support a
benchmark claim because the support budget is full `train/good`, not the
historical SuperAD16 context budget.

## 5. Execution queues

| queue | machine / GPU | objects |
| --- | --- | --- |
| A | dsba3 host GPU 3 | can, fabric, walnuts |
| B | dsba5 host GPU 3 | fruit_jelly, rice, sheet_metal, vial, wallplugs |

Queues are fixed and sequential within each GPU.
