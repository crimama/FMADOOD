# RN-Anchor Gated Local Tangent TESTpublic Log

## 2026-08-12 — preregistered

- Frozen all-eight TESTpublic scope, exact full `train/good` support custody,
  and required arms `A0_RNFMLK_final_layer_1nn_exact_anchor`,
  `A1_RN_anchor_unconditional_local_tangent`,
  `A2_support_confidence_gated_tangent`, `A3_capped_gated_tangent`, and
  `N0_part_shuffled_tangent_control`.
- Recorded `A0_RNFMLK_final_layer_1nn_exact_anchor` as mandatory hard null.
  Optional `O0_oracle` is allowed only as excluded diagnosis.
- Locked `A3` to exact RN anchor/distance with bounded support-LOO
  multiplicative correction `d_rn*(1-0.25*gate)`. Tangent score cannot enter
  magnitude.
- Added fail-closed aggregate contract and focused tests before any result
  intake.
