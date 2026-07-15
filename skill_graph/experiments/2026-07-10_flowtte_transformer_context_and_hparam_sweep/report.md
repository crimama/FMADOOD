# FlowTTE Transformer Context Status And HParam Sweep Plan

## Scope

This note records the completed and in-progress analysis before starting a
class-agnostic hyperparameter sweep on the best current FlowTTE structure.

Dataset and evaluator remain MVTec AD2 TESTpublic with all 8 objects and the
same `seg_AUROC_0.05` / `seg_F1` metrics.

## Current Best Structure

The best retained FlowTTE method structure is still:

```text
DINOv3-H+/16 backbone
layers [7,15,23,31]
layer_norm_mean fused patch feature
DVT-lite position_mean denoise, alpha=1.0
patch-wise NF latent projection
fixed support latent memory bank
no TTE memory expansion
latent NN distance + weak density term
```

Current all-8 reference:

| Method | AUROC_0.05 | F1 |
|---|---:|---:|
| H+ DVT FlowTTE reference | 0.836739 | 0.527427 |
| MLP map run with density weight 0.0 | 0.837518 | 0.523904 |
| reported SuperADD context | 0.839300 | 0.626113 |

The earlier structured-memory number `0.839300 / 0.626113` is the reported
SuperADD context, not our structured-memory run. The actual structured-memory
run was `0.832426 / 0.524593`, so it is not the current best method.

## Completed Structural Diagnostics

### Transformer Flow Without Register Context

Patch-only Transformer Flow was tested as a replacement for the patch-wise MLP
coupling flow.

| Method | AUROC_0.05 | F1 |
|---|---:|---:|
| MLP dw0 control | 0.837518 | 0.523904 |
| Transformer dw0 | 0.828600 | 0.502237 |

Image-level failure-map analysis showed that Transformer Flow reduced the
bad-image GT-vs-background score gap across all classes. This suggests the
unconstrained token mixer smooths or globally normalizes the score field before
memory distance, weakening anomaly contrast.

Verdict: do not use patch-only Transformer Flow as the main scorer.

### DINOv3 CLS/Register Tokens Inside Transformer Flow

Implemented prefix-token conditioning:

```text
[CLS/register/dummy prefix tokens ; patch tokens]
-> Transformer conditioner
-> scale/shift only for patch tokens
-> patch latent memory and scoring only
```

The register/CLS tokens are not inserted into the memory bank and are not scored
as patch tokens.

Completed all-8 interim results:

| Variant | Status | AUROC_0.05 | F1 |
|---|---|---:|---:|
| Transformer + CLS prefix | complete | 0.826397 | 0.495254 |
| Transformer + register prefix | complete | 0.826433 | 0.494640 |
| Transformer + CLS+register prefix | complete | 0.826379 | 0.494712 |
| Transformer + random dummy prefix | complete | 0.829330 | 0.503940 |
| Transformer + learned dummy prefix | complete | 0.829324 | 0.503923 |

Conclusion: real DINOv3 global tokens do not recover the patch-only Transformer
Flow loss. Dummy prefix tokens perform slightly better than CLS/register
prefixes and near patch-only Transformer Flow, which means the current evidence
does not support the claim that raw DINOv3 register tokens improve the
Transformer Flow score field. All Transformer-context variants remain clearly
below the H+ DVT reference.

## Hyperparameter Sweep Target

Because the structural variants have not beaten the H+ DVT reference, the
next sweep targets the retained best structure, not Transformer Flow.

Sweep constraints:

- no class-specific tuning;
- all 8 objects every run;
- same fixed 16-shot support JSON;
- use dsba3 GPUs `0,1,2`;
- delete dense anomaly maps after metrics are recorded;
- compare against the H+ DVT reference and SuperADD context.

Primary search axes:

```text
DVT alpha: 0.75, 1.0, 1.25, 1.5
density_weight: 0.0, 0.1, 0.25, 0.5
flow depth/clamp: coupling_layers 1/2/4, clamp 1.2/1.9/2.5
tail modeling: tail_weight 0.0/0.3/0.6, tail_top_k_ratio 0.03/0.05/0.10
regularization: lambda_logdet 1e-4/1e-3/1e-2
support brightness: 1.0,1.0 and 0.9,1.1
calibration_sample_size: 0, 4096, 8192
```

The first remote pass will use a broad but finite class-agnostic grid, then
continue with a narrower second pass around the best all-8 settings.

## HParam Sweep Progress

Remote controller:

```text
/workspace/results_remote/flowtte_hparam_extreme_20260710_v1_controller.log
```

Leaderboard:

```text
/workspace/results_remote/flowtte_hparam_extreme_20260710_v1_leaderboard.tsv
```

Density-weight block completed:

| Variant | AUROC_0.05 | F1 | Note |
|---|---:|---:|---|
| density_weight 0.05 | 0.837668 | 0.525329 | AUROC up, F1 down |
| density_weight 0.10 | 0.837575 | 0.526275 | below reference F1 |
| density_weight 0.15 | 0.837298 | 0.526882 | below reference F1 |
| density_weight 0.20 | 0.836890 | 0.527247 | closest to reference F1 |
| density_weight 0.35 | 0.835211 | 0.527009 | F1 below reference |
| density_weight 0.50 | 0.833294 | 0.525192 | worse |

Interim read: lowering density improves AUROC slightly but does not beat the
H+ DVT reference on F1. The density axis alone is not the missing SuperADD F1
gap.

Calibration and DVT-alpha block:

| Variant | AUROC_0.05 | F1 | Note |
|---|---:|---:|---|
| calibration_sample_size 0 | 0.836739 | 0.527427 | reproduces reference |
| calibration_sample_size 8192 | 0.836454 | 0.527256 | no improvement |
| DVT alpha 0.75 | 0.834495 | 0.522175 | worse |
| DVT alpha 1.25 | 0.832375 | 0.515905 | worse |
| DVT alpha 1.50 | 0.826311 | 0.498589 | worse |
| DVT alpha 2.00 | 0.814195 | 0.470521 | much worse |

Interim read: H+ backbone is already near the useful DVT alpha maximum.
Over-subtracting the position artifact field damages the score field.

Flow-geometry and likelihood block, partial:

| Variant | AUROC_0.05 | F1 | Note |
|---|---:|---:|---|
| flow_clamp 1.2 | 0.834439 | 0.527520 | tiny F1 gain, AUROC lower |
| flow_clamp 1.5 | 0.835431 | 0.527892 | best F1 so far, AUROC lower |
| flow_clamp 2.5 | 0.837105 | 0.524526 | AUROC up, F1 down |
| flow_clamp 3.5 | 0.836725 | 0.516025 | harmful |
| coupling_layers 1 | 0.832981 | 0.525124 | worse |
| coupling_layers 4 | 0.836800 | 0.511609 | harmful |
| tail_weight 0.0 | 0.835868 | 0.527428 | F1 tied, AUROC lower |
| tail_weight 0.6 | 0.836328 | 0.527305 | no improvement |
| tail_top_k_ratio 0.03 | 0.836070 | 0.526821 | no improvement |
| tail_top_k_ratio 0.10 | 0.836602 | 0.527852 | F1 gain, below logdet |
| lambda_logdet 1e-4 | 0.836395 | 0.527266 | no improvement |
| lambda_logdet 1e-2 | 0.836202 | 0.528875 | stage-1 best so far |
| flow_lr 1e-4 | 0.831622 | 0.524206 | worse |
| flow_lr 5e-4 | 0.825097 | 0.473424 | harmful |
| flow_epochs 5 | 0.832978 | 0.495342 | harmful |
| brightness 0.95,1.05 | 0.837818 | 0.527503 | AUROC gain, small F1 gain |
| brightness 0.90,1.10 | 0.837991 | 0.527758 | AUROC gain, F1 below logdet |
| brightness 0.80,1.20 | 0.838230 | 0.528203 | best AUROC, F1 below logdet |

Interim read: the strongest positive direction is now stronger log-det
regularization, not only lower affine clamp. `lambda_logdet=1e-2` improves F1
by `+0.001448` over the H+ DVT reference while lowering AUROC by about
`-0.000537`. This is still a small diagnostic gain, but it is large enough to
justify a narrow logdet-focused phase-2 sweep.

Stage-1 final read:

| Selection | Variant | AUROC_0.05 | F1 | Delta vs H+ DVT reference |
|---|---|---:|---:|---:|
| best F1 | `lambda_logdet=1e-2` | 0.836202 | 0.528875 | +0.001448 F1 |
| best AUROC | `brightness 0.80,1.20` | 0.838230 | 0.528203 | +0.001491 AUROC / +0.000775 F1 |

Neither setting approaches the reported SuperADD F1 context (`0.626113`).
The tuning signal is real but small, so it remains diagnostic rather than a
method-level improvement claim.

## Phase-2 Narrow Sweep

Started a second class-agnostic controller that waits for stage-1 completion
and then searches the only observed positive region:

```text
/workspace/results_remote/flowtte_hparam_phase2_20260710_v2_controller.log
/workspace/results_remote/flowtte_hparam_phase2_20260710_v2_leaderboard.tsv
```

Phase-2 keeps the same best retained structure and full all-8 setting, then
tests:

- logdet grid around `3e-3..3e-2`;
- logdet with density, tail, LR, and epoch variants;
- fine clamp grid around `1.3..1.8`;
- clamp with density weights `0.15..0.30`;
- clamp with tail-weight and tail-ratio variants;
- clamp with logdet regularization variants;
- clamp with learning-rate / epoch variants;
- clamp with mild support-brightness augmentation.

Gate: this remains `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC` unless the final
all-8 result shows a meaningful F1 increase without sacrificing AUROC enough
to break the ranker interpretation.

Phase-2 logdet grid partial results:

| Variant | AUROC_0.05 | F1 | Note |
|---|---:|---:|---|
| lambda_logdet 3e-3 | 0.836369 | 0.527643 | below stage-1 best |
| lambda_logdet 5e-3 | 0.836336 | 0.527966 | small F1 gain |
| lambda_logdet 7.5e-3 | 0.836279 | 0.528424 | approaching stage-1 best |
| lambda_logdet 1.5e-2 | 0.835972 | 0.529521 | new best over stage-1 |
| lambda_logdet 2e-2 | 0.835673 | 0.529874 | current best F1 |
| lambda_logdet 3e-2 | 0.835028 | 0.529609 | F1 falls, AUROC drops further |

Partial read: F1 peaks around `lambda_logdet=2e-2` and starts falling by
`3e-2`. This suggests stronger logdet regularization improves the binary mask
operating surface, but the AUROC drop shows it is trading off some continuous
ranking quality. Current best over the H+ DVT reference is `+0.002446` F1 and
`-0.001066` AUROC.

## Phase-3 Follow-Up

Because phase-2 showed the useful region is near `lambda_logdet=2e-2`, a
third controller was staged to run after phase-2 completes:

```text
/workspace/results_remote/flowtte_hparam_phase3_20260710_v3_controller.log
/workspace/results_remote/flowtte_hparam_phase3_20260710_v3_leaderboard.tsv
```

Phase-3 focuses on:

- fine logdet values `1.75e-2`, `2.25e-2`, `2.5e-2`, `2.75e-2`;
- `lambda_logdet=2e-2` with density and tail variants;
- `lambda_logdet=2e-2` with support-brightness augmentation;
- `lambda_logdet=2e-2` with lower affine clamp.

This is still class-agnostic and all-eight only. It is intended to test whether
the F1 gain around logdet regularization can be improved without collapsing
AUROC further.
