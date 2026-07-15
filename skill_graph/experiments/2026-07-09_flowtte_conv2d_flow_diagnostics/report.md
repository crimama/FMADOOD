# FlowTTE Conv2D Flow Diagnostic

Date: 2026-07-09
Verdict: KILL_FOR_CLAIM / NO_CONTINUE

## Negative Evidence Intake

This branch tests a structural hypothesis rather than a class-specific retune:
the current FlowTTE MLP normalizing flow treats patches independently, so a
2D convolutional flow might preserve local spatial context and improve the
score field. It is not a threshold, morphology, support-selection, or
class-specific hyperparameter branch.

Known failure basin: the current H+ DVT FlowTTE branch is a useful continuous
ranker but a weak mask generator. If spatially aware flow improves raw score
fields, it should lift F1 without requiring post-processing.

## Motivation

Current reference:

```text
DINOv3-H+/16 layers [7,15,23,31]
DVT position_mean alpha=1.0
fixed SuperAD-16 support
MLP NF latent memory distance
no TTE expansion
```

Reference mean on all eight MVTec AD2 objects:

```text
seg_AUROC_0.05=0.836739
seg_F1=0.527427
```

The question for this diagnostic:

> Does replacing patch-independent MLP coupling with a register-free 2D
> convolutional coupling flow improve the latent distance score field?

## Implementable Design

Implemented `normality_mode=conv2d_flow`.

Pipeline:

```text
DINOv3-H+ fused patch feature map
-> DVT position_mean denoise
-> channel affine Conv2D normalizing flow
-> support latent memory bank
-> nearest-neighbor latent distance
-> anomaly map
```

Flow details:

```text
Conv3x3 -> GELU -> Conv3x3 -> GELU -> Conv1x1
channel split affine coupling
2 coupling layers
zero-init final projection
density_weight=0.0 for the first diagnostic
```

## Evaluation Alignment

Target dataset: MVTec AD2 single-image

Data root:

```text
/home/hunim/Volume/DATA/mvtec_ad_2
```

Smoke objects:

```text
can, vial, fabric
```

Shared settings:

```text
Backbone: dinov3_vith16plus
Layers: [7,15,23,31]
Support: fixed SuperAD-16 JSON
DVT: position_mean alpha=1.0
Expansion: 1.0
Metrics: seg_AUROC_0.05, seg_F1
```

This reduced smoke is diagnostic only. It is compared against the same objects
from the H+ DVT baseline, not promoted as an all-eight SuperAD claim.

## Code Modification / Creation

Changed files:

```text
src/flow_tte/conv2d_flow.py
scripts/flow_tte_mvtec_ad2_core.py
scripts/run_flow_tte_mvtec_ad2.py
tests/test_flow_tte_conv2d_flow.py
```

The new estimator preserves feature maps as `B x C x H x W` during flow
training/evaluation, then flattens only the final latent map for the existing
memory bank and score calibration.

## Added Code Evaluation

Local focused tests:

```text
pytest -q tests/test_flow_tte_conv2d_flow.py
2 passed

pytest -q tests/test_flow_tte_conv2d_flow.py \
  tests/test_flow_tte_context.py \
  tests/test_flow_tte_layerwise.py \
  tests/test_flow_tte_score_field.py
24 passed
```

Remote container checks:

```text
python -m py_compile src/flow_tte/conv2d_flow.py \
  scripts/flow_tte_mvtec_ad2_core.py \
  scripts/run_flow_tte_mvtec_ad2.py

python scripts/run_flow_tte_mvtec_ad2.py --help | grep -q conv2d_flow
remote_cli_ok
```

## Remote Execution

Remote:

```text
host: dsba3 147.47.39.144:2222
container: hun_fsad_tta_012
host GPUs: 0,1,2
```

Run:

```text
/workspace/results_remote/flowtte_conv2d_flow_hplus_smoke_can_vial_fabric_20260709_v1
```

Local pullback:

```text
results/remote_runs/dsba3/flowtte_conv2d_flow_hplus_smoke_can_vial_fabric_20260709_v1
```

## SuperAD Baseline and Unified Metrics

Primary comparator for this diagnostic is the current H+ DVT FlowTTE baseline
on the same objects.

| Object | H+ DVT AUROC | Conv2D AUROC | Delta AUROC | H+ DVT F1 | Conv2D F1 | Delta F1 |
|---|---:|---:|---:|---:|---:|---:|
| can | 0.560495 | 0.539136 | -0.021360 | 0.000634 | 0.000376 | -0.000258 |
| vial | 0.746292 | 0.733686 | -0.012605 | 0.434360 | 0.395137 | -0.039222 |
| fabric | 0.968227 | 0.962024 | -0.006202 | 0.697427 | 0.666898 | -0.030529 |
| mean | 0.758338 | 0.744949 | -0.013389 | 0.377474 | 0.354137 | -0.023337 |

## Results and Analysis

The 2D Conv Flow underperforms the H+ DVT baseline on all three smoke objects.
The pattern is consistent rather than noisy:

- AUROC drops on every object.
- F1 drops on every object.
- `can` remains collapsed, so the intended score-field repair does not happen.
- `fabric` stays strong but still loses to the reference, meaning Conv2D flow
  does not add useful spatial consistency beyond the existing DVT+MLP NF branch.

The likely reason is that a shallow convolutional conditioner smooths or
renormalizes local feature fields without fixing the dominant memory-ranking
problem. It adds spatial inductive bias, but the score still uses a flat support
latent memory bank, so foreground/background and object-mode mixing remain.

## Continuation Assessment

Strict method claim: no.

Small next diagnostic for this exact branch: no. The smoke was deliberately
chosen to include weak (`can`, `vial`) and strong (`fabric`) categories; every
metric moved in the wrong direction.

This result does not rule out token-interaction flows or structured memory, but
it argues against spending an all-eight run on this shallow register-free 2D
Conv Flow implementation.

## Cleanup

Dense maps were generated only transiently and removed by `--cleanup-maps`.

Cleanup evidence:

```text
remote anomaly_maps count: 0
local anomaly_maps count: 0
```

## Conclusion

`normality_mode=conv2d_flow` is useful as a code-level diagnostic control, but
not as the next method direction. The next structural direction should not be a
plain local convolutional flow head; it should target the remaining flat-memory
mode mixing or token-level interaction problem directly.
