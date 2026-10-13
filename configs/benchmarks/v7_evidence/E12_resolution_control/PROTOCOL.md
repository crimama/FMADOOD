# E12 — Single high-resolution control for RSG3

Registered 2026-09-25 before execution. This is a development comparison on the
public test population; masks are evaluator-only and no sealed holdout is used.

## Hypothesis and comparison boundary

The four-scale RSG3 gain may come from seeing a 1120-pixel input alone, or from
combining scales. Compare the exact current RSG3 implementation with one 1120
branch while holding support, data, backbone, preprocessing, fitting, query,
normalization, gating, post-processing, and evaluator fixed.

Three arms separate scale composition from memory-bank capacity:

- `e12_rsg3_multiscale`: `[448, 672, 896, 1120]`, 256 tokens per support image
  at each scale. This is the paired RSG3 control.
- `e12_rsg3_hr1120_256`: `[1120]`, 256 tokens per support image. This is the
  single high-resolution arm with the original per-scale bank setting.
- `e12_rsg3_hr1120_1024`: `[1120]`, 1024 tokens per support image. This is the
  capacity-matched high-resolution control: its total bank-token budget matches
  the four-scale arm (4 × 256 per support image).

The single-scale arms intentionally differ in their declared resolution list;
the capacity-matched arm also changes only the token budget to match total bank
capacity. Other RSG3 settings are copied from `E10_ad2_strata/e10_rsg3.yaml`.

## Fixed conditions

- Dataset: MVTec AD 2, all 8 classes.
- Split: `mvtec_ad2_test_public_strata_comparison_no_label_selection`; the
  `regular` stratum is primary, with the other declared strata reported
  separately and as a pooled descriptive result.
- Support: seed 0, 16-shot, the same fixed
  `configs/methods/VOID_support_16.json` manifest in all arms.
- Backbone: frozen DINOv2 ViT-L/14; feature layers `[5, 11, 23]`; same
  preprocessing and shared pipeline as the paired control.
- Method: RSG3, gated shallow residual, support-quantile tail normalization,
  residual-spike fusion, component-quantile post-processing; unchanged across
  arms.
- Primary metric: full8 macro AUPRO@0.05 on `regular`. Secondary metrics:
  pAUROC@0.05 and per-stratum AUPRO@0.05.
- Runtime budget: 24 class evaluations total (3 arms × 8 classes), one already
  assigned dsba3 GPU UUID, no retries. Maps and fitted weights remain disabled.

## Decision rule

Use 0.86pp as the project noise threshold. Let `Δ = multiscale −
capacity-matched 1120-only` on regular AUPRO:

- `Δ > 0.86pp`: evidence that multiscale composition adds a meaningful gain
  beyond maximum input resolution and total bank capacity for this support draw.
- `|Δ| ≤ 0.86pp`: no resolved multiscale advantage at this noise floor; the
  current evidence does not rule out high-resolution-only as a sufficient cause.
- `Δ < −0.86pp`: the capacity-matched single-resolution arm is meaningfully
  better on this development draw.

The original-budget 1120-only arm shows whether an unadjusted single-scale
control is competitive. Report all class values and strata; do not convert this
single seed into a population-level robustness claim. If any arm has incomplete
coverage, invalid provenance, or support/evaluator mismatch, stop interpretation
and report the execution failure without substituting cells.
