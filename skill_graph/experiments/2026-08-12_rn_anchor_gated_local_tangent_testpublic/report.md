# RN-Anchor Gated Local Tangent TESTpublic Report

Date: 2026-08-12  
Status: `PRE-REGISTERED / NOT RUN`  
Verdict ceiling: `PROMISING_DIAGNOSTIC`

## Scope

This report tracks the all-eight TESTpublic run for RN-anchored gated local
tangent scoring. No result is recorded yet.

## Frozen contract

- Dataset: MVTec AD2 single-image `test_public`.
- Support: full per-object `train/good` population.
- Required arms: `A0_RNFMLK_final_layer_1nn_exact_anchor`,
  `A1_RN_anchor_unconditional_local_tangent`,
  `A2_support_confidence_gated_tangent`, `A3_capped_gated_tangent`,
  `N0_part_shuffled_tangent_control`.
- Optional oracle: `O0_oracle`, excluded from aggregate claims.
- Primary metric: macro `seg_AUROC_0.05`.
- Secondary metrics: macro `seg_F1`, macro `seg_AUPRO_0.05`, coverage.
- Required contrasts: `A3-A0`, `A3-N0`, object wins, and object no-harm counts.

## RN-anchor rule

`A3` must preserve exact RN anchor/distance identity and may only apply the
support-LOO gate as `d_rn*(1-0.25*gate)`. Tangent score may affect admission but
must not enter magnitude. Shards must record anchor and distance identity hashes
and the aggregate rejects cross-object hash drift.

## Claim boundary

Strict signal requires `A3_capped_gated_tangent >
A0_RNFMLK_final_layer_1nn_exact_anchor` on macro pAUROC, F1, and AUPRO, with
complete coverage and a primary win over `N0_part_shuffled_tangent_control`.
SuperAD16 dominance is not claimable from full-support evidence; it becomes
comparable only under the same support budget and support manifest.

## Aggregation

Aggregator: `scripts/aggregate_rn_gated_tangent_testpublic.py`.

No aggregate artifact exists yet.
