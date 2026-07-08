# FlowTTE Raw Hard-Null Structural Diagnostic

Date: 2026-07-08

Verdict: `KILL_FOR_MAIN_CLAIM / CONTINUE_AS_DIAGNOSTIC`

## Question

The current best FlowTTE branch is:

```text
DINOv3-H+/16 layers [7,15,23,31]
-> DVT-style position-mean denoise alpha 1.0
-> NF latent projection
-> fixed 16-shot support latent bank
-> latent NN distance + weak density
```

It scores `0.836739` AUROC / `0.527427` F1 on all-eight MVTec AD2
`test_public`. Reported SuperADD context is `0.839300` / `0.626113`.

This diagnostic asks whether the remaining gap is caused by using NF latent
projection instead of a SuperADD-like raw feature nearest-neighbor score field.

## Protocol

- Dataset: MVTec AD2 public test, all 8 objects.
- Objects: `can`, `fabric`, `fruit_jelly`, `rice`, `vial`, `wallplugs`,
  `walnuts`, `sheet_metal`.
- Remote host/container: dsba3, `hun_fsad_tta_012`.
- Host GPUs: `0,1,2`.
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`.
- Code root inside container: `/root/fsad_tta_run`.
- Result root inside container:
  `/root/results_remote/flowtte_raw_hardnull_all8_20260708_v1`.
- Local result archive:
  `results/remote_runs/dsba3/flowtte_raw_hardnull_all8_20260708_v1.tgz`.
- Support: same fixed 16-shot JSON as the current H+ diagnostics, from
  `skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json`.
- Metrics: evaluator object-level `seg_AUROC`, `seg_F1`, reported as all-object
  macro mean.

## Implemented Variants

| Variant | Structure | Key Setting |
|---|---|---|
| `raw_layerwise_tiled_no_dvt` | per-layer raw NN, score-level layer average | H+, 640 tile, 128 overlap, no DVT |
| `raw_layerwise_tiled_dvt` | per-layer raw NN, score-level layer average | H+, 640 tile, 128 overlap, DVT alpha 1.0 |
| `raw_fused_dvt` | fused normalized multi-layer raw NN | H+, no tiling, DVT alpha 1.0 |
| `raw_nn_nf_residual_dvt` | raw fused NN + NF score residual | H+, no tiling, DVT alpha 1.0, residual weight 0.25 |
| `foreground_raw_fused_dvt` | support norm foreground/background split raw NN | H+, no tiling, DVT alpha 1.0 |

## Results

| Method | AUROC | F1 | Delta AUROC vs H+ DVT NF | Delta F1 vs H+ DVT NF |
|---|---:|---:|---:|---:|
| SuperAD-16 recorded context | 0.765802 | 0.385534 | -0.070937 | -0.141893 |
| Reported SuperADD context | 0.839300 | 0.626113 | +0.002561 | +0.098686 |
| H+ DVT NF latent reference | 0.836739 | 0.527427 | +0.000000 | +0.000000 |
| `raw_layerwise_tiled_no_dvt` | 0.759979 | 0.395241 | -0.076760 | -0.132186 |
| `raw_layerwise_tiled_dvt` | 0.771061 | 0.380378 | -0.065678 | -0.147050 |
| `raw_fused_dvt` | 0.829365 | 0.499606 | -0.007374 | -0.027822 |
| `raw_nn_nf_residual_dvt` | 0.833266 | 0.510439 | -0.003473 | -0.016989 |
| `foreground_raw_fused_dvt` | 0.799772 | 0.416134 | -0.036967 | -0.111294 |

Full per-object values are in `per_object.tsv`.

## Analysis

### 1. SuperADD-like raw layer-wise tiled NN is a negative control here

The two high-resolution tiled layer-wise variants are much worse than the
current H+ DVT NF reference:

```text
raw_layerwise_tiled_no_dvt: 0.759979 / 0.395241
raw_layerwise_tiled_dvt:    0.771061 / 0.380378
H+ DVT NF reference:        0.836739 / 0.527427
```

DVT slightly raises AUROC in this tiled layer-wise setting, but it lowers F1.
So the remaining SuperADD gap is not explained by "we should simply remove NF
and copy raw layer-wise tiled NN." Under the current evaluator, support set,
and implementation, that replacement is structurally harmful.

### 2. The fused normalized feature remains important

`raw_fused_dvt` recovers most of the score quality:

```text
raw_fused_dvt: 0.829365 / 0.499606
```

This is far stronger than tiled layer-wise raw NN but still below the current
NF latent reference. This supports the earlier layer-wise diagnostic: the
current fused normalized multi-layer feature is acting as a stabilizer, while
separate per-layer maps introduce poorly calibrated score fields.

### 3. NF residual has signal, but does not beat the current main branch

Adding an NF residual to raw fused NN improves the raw fused baseline:

```text
raw_fused_dvt:            0.829365 / 0.499606
raw_nn_nf_residual_dvt:   0.833266 / 0.510439
gain:                   +0.003901 / +0.010833
```

However, it remains below the H+ DVT NF latent reference by
`-0.003473` AUROC and `-0.016989` F1. NF therefore carries useful residual
normality information, but this hybrid should not replace the current best
method.

### 4. Simple support-norm foreground split is too crude

The foreground/background split variant scores only:

```text
foreground_raw_fused_dvt: 0.799772 / 0.416134
```

This means support feature norm is not a reliable foreground selector across
all objects. Foreground-aware scoring remains a valid structural direction,
but the prior must be better than a global norm quantile.

### 5. `can` remains a class-agnostic failure bucket, not a tuning target

All raw/NF hard-null variants still collapse on `can` F1:

```text
raw_fused_dvt can F1:          0.001897
raw_nn_nf_residual_dvt can F1: 0.001405
```

This should not trigger per-class tuning. It should be used as a failure bucket
for class-agnostic score-field and object-region diagnostics.

## Decision

Do not pivot the main method to raw layer-wise tiled NN or simple foreground
split memory. Keep the current H+ DVT NF latent reference as the best retained
branch.

The only positive diagnostic from this run is:

```text
raw NN + NF residual > raw NN
```

So NF should remain in the method family. The next structural work should focus
on score-field formation and object/foreground priors, not on replacing NF with
raw layer-wise nearest-neighbor scoring.

Recommended next direction:

```text
H+ DVT NF latent reference
-> class-agnostic object/foreground prior stronger than support norm
-> score-field calibration learned from support/test good statistics
-> connected-component fragmentation audit as a measurement, not a tuned fix
```
