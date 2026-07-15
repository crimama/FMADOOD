# AD2 Proposed Full-Normal SuperADD-Style Threshold Report

Date: 2026-07-13  
Verdict: `ACCEPT_MEASUREMENT / BLOCKED_BASELINE`

## Outcome

Using the full MVTec AD2 normal training population improves the data-budget
alignment with SuperADD, but it does not close the binary-localization gap.
The proposed method reaches `83.38` class-macro p-AUROC, only `0.55` point
below SuperADD's reported `83.93`, while the held-out-normal threshold yields
only `23.03` morphology F1. Even the TESTpublic oracle reaches only `54.85`
F1, still `7.76` points below SuperADD's reported `62.61`.

The main failure is therefore not continuous anomaly ranking alone. The
normal-only `p95 * 1.421` threshold is poorly calibrated to this method's
Flow-latent score scale, and a residual mask-quality gap remains after giving
the method an oracle threshold.

## Frozen setting

- Dataset: all eight MVTec AD2 public objects; full `test_public` evaluation.
- Backbone: DINOv3 ViT-H+/16, layers `[7,15,23,31]`; native 672 resolution,
  with `sheet_metal=448` as in the existing AD2 experiments.
- Detector: Flow-LatentBank, 3 Flow epochs, 2 coupling layers, hidden
  multiplier 1, latent 1-NN, LOO standardization, static memory, DVT position
  mean alpha 1.0, density weight 0.25, RGB guided-r8, and fixed
  close/fill/erode morphology (line 17, 16 angles).
- Full-normal split: sorted `train/good`; `index % 8 == 0` is threshold-only,
  and the remaining 7/8 is used for Flow training and prototype construction.
- Prototype selection: exact global `k=100` latent-neighbor distances, global
  mean distance `tau`, ascending count of neighbors below `tau`, stable first
  100,000 selection. The bank remains fixed at 100,000 during testing.
- Fixed threshold: held-out-normal map pixels' 95th percentile times `1.421`,
  with strict `score > threshold`. Oracle uses the TESTpublic raw-best
  threshold. Both arms use the same guided maps and morphology.

The full-normal split and threshold constants follow the released SuperADD
protocol. The exact score-ranking selection follows the paper definition
provided for this experiment and is applied to this method's Flow latents; it
is not a same-code reproduction of SuperADD's upstream patch pipeline.

## Results

All values except thresholds are percentages. `Fixed` denotes the
held-out-normal `p95 * 1.421` threshold.

| Class | p-AUROC | Fixed F1 raw | Fixed F1 morph | Oracle F1 raw | Oracle F1 morph | Fixed thr | Oracle thr |
|---|---:|---:|---:|---:|---:|---:|---:|
| can | 52.00 | 0.02 | 0.02 | 0.03 | 0.03 | 2.9140 | 1.8535 |
| fabric | 95.75 | 35.43 | 36.32 | 63.99 | 75.37 | 2.6183 | 3.6621 |
| fruit_jelly | 80.34 | 52.79 | 53.24 | 54.70 | 54.36 | 4.8293 | 5.1875 |
| rice | 94.32 | 6.89 | 7.21 | 68.24 | 68.25 | 1.2216 | 8.1484 |
| sheet_metal | 92.68 | 6.21 | 6.07 | 68.87 | 68.86 | 1.9170 | 7.9961 |
| vial | 77.59 | 39.84 | 40.12 | 49.93 | 50.59 | 1.8677 | 2.5488 |
| wallplugs | 87.87 | 2.11 | 2.06 | 55.01 | 54.17 | 2.6675 | 6.5820 |
| walnuts | 86.50 | 38.85 | 39.17 | 67.25 | 67.19 | 4.6756 | 9.4062 |
| **Macro** | **83.38** | **22.77** | **23.03** | **53.50** | **54.85** | 2.8389 | 5.6731 |

## Context against reported SuperADD

This table is contextual rather than a strict same-code comparison: the
training-data budget and threshold split are aligned, but the detector,
feature/prototype pipeline, and morphology are not SuperADD's implementation.

| Class | Ours p-AUROC | SuperADD p-AUROC | Delta | Ours fixed morph F1 | SuperADD F1 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| can | 52.00 | 51.61 | +0.39 | 0.02 | 0.00 | +0.02 |
| fabric | 95.75 | 84.20 | +11.55 | 36.32 | 93.74 | -57.42 |
| fruit_jelly | 80.34 | 81.19 | -0.85 | 53.24 | 54.68 | -1.44 |
| rice | 94.32 | 96.11 | -1.79 | 7.21 | 73.31 | -66.10 |
| sheet_metal | 92.68 | 93.18 | -0.50 | 6.07 | 59.54 | -53.47 |
| vial | 77.59 | 82.59 | -5.00 | 40.12 | 64.77 | -24.65 |
| wallplugs | 87.87 | 92.53 | -4.66 | 2.06 | 79.16 | -77.10 |
| walnuts | 86.50 | 90.03 | -3.53 | 39.17 | 75.69 | -36.52 |
| **Macro** | **83.38** | **83.93** | **-0.55** | **23.03** | **62.61** | **-39.58** |

Sources: [SuperADD paper](https://arxiv.org/html/2605.14808),
[official repository](https://github.com/LukasRoom/SuperADD).

## Audit and execution evidence

- dsba3 GPUs 0--3: `can`, `fabric`, `rice`, `walnuts`, `wallplugs`, and
  `sheet_metal`; dsba5 GPUs 0--1: `vial` and `fruit_jelly`.
- Eight finite result rows and all expected manifests, metrics, split records,
  and cleanup evidence are present.
- Per-class split audit passed: prototype and threshold sets are disjoint,
  exhaustive, and exactly reproduce the sorted modulo-8 rule.
- All eight banks record `100000 -> 100000`; no test-time expansion occurred.
- All eight logs contain the explicit `latent_bank_subsampled_100000` stage;
  no traceback, CUDA OOM, or runtime error was found.
- Dense anomaly and threshold-calibration maps were removed only after both
  threshold arms were evaluated; no retained map directory remains.
- Local regression suite: `443 passed`; remote result-bearing run: `v4` only.

Recovered artifacts:

- `results/remote_runs/dsba3/flowtte_ad2_fullnormal_superadd_thresholds_20260713_v4`
- `results/remote_runs/dsba5/flowtte_ad2_fullnormal_superadd_thresholds_20260713_v4`

## Interpretation

The near-match in macro p-AUROC shows that full-normal training plus compact
latent selection preserves a competitive continuous ranking signal. It does
not validate the fixed threshold transfer: `rice`, `sheet_metal`, and
`wallplugs` require oracle thresholds three to seven score units above the
normal-only thresholds, producing large false-positive masks in the fixed
arm. `can` is a separate representation/ranking failure because even its
oracle F1 is effectively zero.

Consequently, the next justified diagnostic is a calibration study that is
still normal-only and class-agnostic (for example, calibrating standardized
score tails rather than multiplying raw guided-map percentiles). It should
not be presented as a SuperADD comparison until the same-code baseline and
metric pipeline are run.
