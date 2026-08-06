# Same16 DVT × Density Factorial Attribution

Date frozen: 2026-07-24
Status: `DESIGN_FROZEN / DIAGNOSTIC_ONLY`

## Negative Evidence Intake

The completed Same16 public-shadow run retained Basic after A6 failed its
no-harm promotion gate. That comparison changed two components together:
support-fitted position-mean DVT and the density penalty. This experiment is
the single detector-matched causal confirmation allowed after A6 confirmation.
It is not a density sweep, shot-scaling run, new detector, or method claim.

The likely failure basin is calibration/ranking redistribution across objects.
The experiment is worth running only because the unresolved bundle prevents
attribution of the observed Basic-to-A6 pAUROC/F1 movement.

## Motivation

Separate DVT, density, and their interaction under the exact completed v6
public-shadow contract.

Claim: the nominal Basic-to-A6 difference can be attributed to DVT, density,
or a DVT×density interaction.

Boundary: MVTec AD2 `test_public` development evidence only. No TESTprivate,
shot-scaling, method-promotion, or paper-level claim is authorized.

## Implementable Design

Reuse, without rerunning:

- `basic`: DVT on, density `0.25`
- `a6`: DVT off, density `0.0`
- same-condition SuperAD-16 and RN-FMLK rows

Run only the missing cells:

| Cell | DVT | Density weight |
|---|---|---:|
| `dvt_only` | `position_mean`, alpha `1.0` | `0.0` |
| `density_only` | `none` | `0.25` |

Everything else remains identical to v6: DINOv2 ViT-L/14, layers
`[5,11,17,23]`, `layer_norm_mean`, Flow/fused normality, 3 epochs, exact
fixed-16 reference set, full eight objects, original-resolution maps, no RGB
guide, no threshold calibration, no morphology, no TTA, no query update, and
no memory expansion.

## Evaluation Alignment

- Target dataset: MVTec AD2 single-image
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- Population: all eight objects and full `test_public/good,bad`
- Support SHA-256:
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`
- Baseline source:
  `same16_baseline_metric_contract_20260723_v6`
- Metrics:
  `seg_AUROC_0.05`, `seg_F1_raw`, `seg_AUPRO_0.05`, plus complete map,
  shape, finite, and nonconstant coverage

For each metric, report:

- DVT effect at density `0` and `0.25`
- density effect with DVT off and on
- average DVT and density main effects
- interaction:
  `basic - dvt_only - density_only + a6`
- per-object deltas, positive-object count, object-bootstrap 95% interval,
  and object-collapse flags

## Frozen Attribution Gates

Use 2,000 deterministic object-bootstrap replicates, seed `0`.

Material bounds:

- pAUROC: `0.005`
- F1: `0.010`
- AUPRO: `0.010`

A positive factor credit requires its average main effect to meet the material
bound, have a strictly positive 95% lower confidence bound, improve at least
six of eight objects, and cause no cross-metric object collapse.

An interaction is supported when its absolute macro magnitude meets the
metric material bound and its 95% interval excludes zero. A supported
interaction blocks simple independent-component attribution.

Object-collapse bounds are unchanged from v6:

- pAUROC below `-0.02`
- F1 below `-0.03`
- AUPRO below `-0.02`

## Decision

Allowed verdicts:

- `DVT_MAIN_EFFECT`
- `DENSITY_MAIN_EFFECT`
- `DVT_DENSITY_INTERACTION`
- `MIXED_OR_INCONCLUSIVE_ATTRIBUTION`

Every verdict remains `KILL_FOR_CLAIM / DIAGNOSTIC_ONLY`. No shot curve or
new detector promotion follows automatically. Dense maps are deleted after
metrics, coverage, aggregation, and verdict are recorded.
