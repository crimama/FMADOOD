# AD2 Random 16-Shot DINOv2-L/14 Report

Date: 2026-07-13  
Verdict: `ACCEPT_MEASUREMENT / BLOCKED_BASELINE`

## Outcome

The requested random 16-shot DINOv2 ViT-L/14 run completed over all eight
MVTec AD2 public classes. With the rest of the proposed stack frozen, the
class-macro result is `78.11` p-AUROC, `37.19` guided raw F1, and `43.07`
guided plus morphology F1.

This is below the existing DINOv3-H+ fixed-support proposed result by `5.64`
p-AUROC points and `12.31` morphology-F1 points. That delta is informative,
but it is not an isolated backbone ablation: the requested run changes both
the backbone and the support policy from fixed support to seeded random
support. A same-support DINOv2-L versus DINOv3-H+ pair is required to
attribute the difference to the backbone alone.

## Frozen setting

- Dataset: MVTec AD2 single-image, all eight classes, full `test_public`.
- Support: 16 normal images per class, sampled without replacement from the
  sorted `train/good` pool using NumPy `default_rng(0)`; policy
  `visionad_seeded_random`.
- Backbone: DINOv2 ViT-L/14, layers `[5,11,17,23]`; 672 resolution except
  `sheet_metal=448`.
- Detector: Flow-LatentBank, Flow 3 epochs, 2 coupling layers, hidden
  multiplier 1, latent 1-NN with LOO standardization, and a static bank.
- AD2 components: DVT position mean alpha 1, density weight 0.25, RGB
  guided-r8, then fixed close/fill/erode morphology (line 17, 16 angles).
- Disabled: TTE expansion, context/register routing, support transforms,
  brightness augmentation, and tiling.

## Results

All values are percentages. `Guided raw F1` uses the best threshold on the
guided continuous map; `Morph F1` applies the frozen binary morphology at
that threshold.

| Class | p-AUROC | Guided raw F1 | Morph F1 |
|---|---:|---:|---:|
| can | 60.91 | 0.15 | 0.14 |
| fabric | 70.58 | 27.26 | 71.76 |
| fruit_jelly | 74.57 | 33.62 | 34.85 |
| rice | 93.72 | 68.22 | 68.16 |
| sheet_metal | 87.97 | 41.94 | 41.85 |
| vial | 70.27 | 39.97 | 40.94 |
| wallplugs | 77.96 | 14.02 | 14.35 |
| walnuts | 88.91 | 72.30 | 72.53 |
| **Macro** | **78.11** | **37.19** | **43.07** |

For reference, the unguided continuous maps produce `77.21` macro p-AUROC
and `34.81` best-threshold F1. Guided-r8 therefore adds `0.90` p-AUROC point
and morphology raises the guided F1 by `5.89` points overall, although the
large F1 gain is concentrated in `fabric`.

## Context

These rows are contextual rather than strict matched comparisons.

| Setting | Support | Backbone | p-AUROC | Morph F1 | Delta vs this run |
|---|---|---|---:|---:|---:|
| This run | random 16, seed 0 | DINOv2-L/14 | 78.11 | 43.07 | — |
| Existing AD2 proposed | fixed 16 | DINOv3-H+/16 | 83.75 | 55.39 | +5.64 / +12.31 |
| Existing full-normal | sorted 7/8 train | DINOv3-H+/16 | 83.38 | 54.85 oracle | +5.27 / +11.78 |

The weak `can` result remains the largest failure (`60.91` p-AUROC and
near-zero F1), while random-16 DINOv2-L also drops strongly on `fabric` and
`wallplugs`. Thus this seed does not support replacing the existing AD2
configuration with the simpler DINOv2-L random-16 setup.

## Audit and execution evidence

- Result-bearing runs: `v7_first4` and `v7_last4`, one class per dsba3 GPU
  0--3 in two waves.
- All eight manifests exactly match recomputed `default_rng(0)` selections;
  every class has 16 unique normal support paths.
- All eight banks retain identical initial and final sizes; expansion budget
  is 1.0.
- Backbone, layers, per-class resolution, DVT, density, no-context, and
  identity-only support-transform fields match the preregistered contract.
- Eight raw and eight guided metric files are finite. Logs contain no
  traceback, runtime error, CUDA OOM, or disk-full error.
- Dense map cleanup passed: zero anomaly-map directories remain after metric
  generation.
- Local regression suite: `446 passed`; focused aggregation/backbone/guided
  checks: `8 passed`; launcher syntax and diff checks passed.

Recovered aggregate artifact:
`results/remote_runs/dsba3/flowtte_ad2_random16_dinov2l_seed0_20260713_v7_all8`.

## Interpretation and claim boundary

The experiment is a valid single-seed few-shot diagnostic, not a robustness
estimate. The result suggests that the current AD2 performance is not
preserved by simultaneously moving to DINOv2-L/14 and random 16-shot
support. At least three to five preregistered selection seeds are needed to
measure random-support variance. A strict method comparison is also blocked
because a same-condition SuperAD/SuperADD baseline is absent; the verdict is
therefore capped at `BLOCKED_BASELINE`.
