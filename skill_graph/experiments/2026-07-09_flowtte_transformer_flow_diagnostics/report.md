# FlowTTE Transformer Coupling Flow Diagnostic

Date: 2026-07-09
Verdict: KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC

## Negative Evidence Intake

This branch tests whether the current patch-independent MLP flow is too weak
because it cannot model token-token agreement. It is structurally different
from thresholding, morphology, score-field calibration, support selection, and
the failed shallow 2D Conv Flow branch.

Known failure basin: FlowTTE is a strong continuous patch ranker after
DINOv3-H+ and DVT denoise, but it still produces weak binary masks and
flat-memory mode mixing.

## Motivation

The immediate previous diagnostic, `conv2d_flow`, added local 3x3 spatial
context but lost on `can,vial,fabric`. That suggests plain local convolution is
not enough. The next structural question is:

> Does a register-free Transformer conditioner improve patch-patch agreement
> and latent memory ranking?

## Implementable Design

Implemented `normality_mode=transformer_flow`.

Pipeline:

```text
DINOv3-H+ fused patch feature map
-> DVT position_mean denoise
-> channel affine Transformer coupling flow
-> support latent memory bank
-> nearest-neighbor latent distance
-> anomaly map
```

Flow details:

```text
P in R^{N x C}
channel split P_a, P_b
Linear(P_a) -> 1-layer TransformerEncoder -> Linear -> scale, shift
P_b' = P_b * exp(clamp * tanh(scale)) + shift
no positional embedding
2 coupling layers
zero-init final projection
density_weight=0.0
```

## Evaluation Alignment

Target dataset:

```text
MVTec AD2 single-image
```

Data root:

```text
/home/hunim/Volume/DATA/mvtec_ad_2
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

Primary comparator is the current H+ DVT FlowTTE baseline:

```text
seg_AUROC_0.05=0.836739
seg_F1=0.527427
```

## Code Modification / Creation

Changed files:

```text
src/flow_tte/transformer_flow.py
scripts/flow_tte_mvtec_ad2_core.py
scripts/run_flow_tte_mvtec_ad2.py
tests/test_flow_tte_transformer_flow.py
```

The runner now treats `conv2d_flow` and `transformer_flow` as map-flow modes
that preserve the feature map during flow projection and reuse the existing
memory/scoring path after flattening the latent map.

## Added Code Evaluation

Local verification:

```text
python3 -m py_compile src/flow_tte/transformer_flow.py \
  src/flow_tte/conv2d_flow.py \
  scripts/flow_tte_mvtec_ad2_core.py \
  scripts/run_flow_tte_mvtec_ad2.py

pytest -q tests/test_flow_tte_transformer_flow.py \
  tests/test_flow_tte_conv2d_flow.py \
  tests/test_flow_tte_context.py \
  tests/test_flow_tte_layerwise.py \
  tests/test_flow_tte_score_field.py

26 passed
```

Remote CLI check:

```text
remote_transformer_cli_ok
```

## Remote Execution

Remote:

```text
host: dsba3 147.47.39.144:2222
container: hun_fsad_tta_012
host GPUs: 0,1,2
```

Smoke run:

```text
/workspace/results_remote/flowtte_transformer_flow_hplus_smoke_can_vial_fabric_20260709_v1
results/remote_runs/dsba3/flowtte_transformer_flow_hplus_smoke_can_vial_fabric_20260709_v1
```

All-eight run:

```text
/workspace/results_remote/flowtte_transformer_flow_hplus_all8_20260709_v1
results/remote_runs/dsba3/flowtte_transformer_flow_hplus_all8_20260709_v1
```

## Smoke Result

| Object | H+ DVT AUROC | Transformer AUROC | Delta AUROC | H+ DVT F1 | Transformer F1 | Delta F1 |
|---|---:|---:|---:|---:|---:|---:|
| can | 0.560495 | 0.550074 | -0.010421 | 0.000634 | 0.000578 | -0.000056 |
| vial | 0.746292 | 0.786881 | +0.040589 | 0.434360 | 0.481298 | +0.046938 |
| fabric | 0.968227 | 0.969991 | +0.001765 | 0.697427 | 0.703166 | +0.005740 |
| mean | 0.758338 | 0.768982 | +0.010644 | 0.377474 | 0.395014 | +0.017541 |

The smoke justified one all-eight expansion despite the tiny `can` drop,
because `vial` improved substantially and `fabric` improved without no-harm
violation.

## All-Eight Result

| Object | H+ DVT AUROC | Transformer AUROC | Delta AUROC | H+ DVT F1 | Transformer F1 | Delta F1 |
|---|---:|---:|---:|---:|---:|---:|
| can | 0.560495 | 0.550074 | -0.010421 | 0.000634 | 0.000578 | -0.000056 |
| fabric | 0.968227 | 0.969991 | +0.001765 | 0.697427 | 0.703166 | +0.005740 |
| fruit_jelly | 0.781873 | 0.719945 | -0.061929 | 0.481412 | 0.300790 | -0.180622 |
| rice | 0.947121 | 0.943848 | -0.003273 | 0.711554 | 0.698454 | -0.013099 |
| vial | 0.746292 | 0.786881 | +0.040589 | 0.434360 | 0.481298 | +0.046938 |
| wallplugs | 0.908028 | 0.891167 | -0.016860 | 0.631539 | 0.579718 | -0.051820 |
| walnuts | 0.890238 | 0.874516 | -0.015722 | 0.733291 | 0.670903 | -0.062389 |
| sheet_metal | 0.891639 | 0.892381 | +0.000743 | 0.529204 | 0.582991 | +0.053787 |
| mean | 0.836739 | 0.828600 | -0.008139 | 0.527427 | 0.502237 | -0.025190 |

## Results and Analysis

Transformer coupling produced real positive signal on:

```text
vial: +0.046938 F1
sheet_metal: +0.053787 F1
fabric: +0.005740 F1
```

But it harmed:

```text
fruit_jelly: -0.180622 F1
walnuts: -0.062389 F1
wallplugs: -0.051820 F1
rice: -0.013099 F1
```

This means token interaction is not a pure no-harm replacement for the fused
MLP flow. The branch supports the mechanism hypothesis only partially:
Transformer conditioning can improve some structured or boundary-sensitive
objects, but without mode-aware routing/calibration it can distort object modes
and hurt class-agnostic robustness.

## Continuation Assessment

Strict method claim: no.

Continuation: yes, but not as plain `transformer_flow`.

The next useful diagnostic should keep the positive token-interaction evidence
but constrain it with a mode/reliability mechanism:

```text
mode-level structured memory + per-mode calibration
or
Transformer-flow residual/ensemble gated by baseline score reliability
```

Do not tune this branch class-by-class. The all-eight harm shows the next step
must be class-agnostic reliability or mode calibration, not Transformer
hyperparameter search.

## Cleanup

Dense maps were generated only transiently and removed by `--cleanup-maps`.

Cleanup evidence:

```text
remote anomaly_maps count: 0
local anomaly_maps count: 0
```

## Conclusion

Plain register-free Transformer coupling flow is killed as a replacement for
the current H+ DVT FlowTTE baseline. It remains useful evidence that token
interaction can help selected objects, especially `vial` and `sheet_metal`.
The next structural direction should be mode-level structured memory or
reliability-gated use of the Transformer score, not a broader Transformer
hyperparameter sweep.
