# ATAC-SB-v2 Stage -1 support-only preregistration

Status before results: `DESIGN / SUPPORT-ONLY GATE ELIGIBLE`  
Protocol: `atac-sb-v2-stage-minus1-20260824`  
Claim ceiling: validity diagnostic only; no public metrics or method claim.

## Frozen custody

- Target: MVTec AD2 single-image, data root
  `/home/hunim/Volume/DATA/mvtec_ad_2`.
- Objects: exact same16 support paths from
  `skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/superad16_dinov2_reference_paths.json`.
- Manifest SHA-256:
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Backbone: frozen `dinov2_vitl14`, layers `[5,11,17,23]`, primary layer 23;
  existing resize/crop custody (672, except sheet_metal 448).
- No `test_public`, `test_private`, ground truth, labels, metadata, query
  fitting, query memory, or anomaly maps.

## Frozen mechanism

For resized RGB canvas `x` in fp32, obtain the existing exact two-level Haar
high-pass `H2(x)`. Normalize per image (all RGB/spatial elements) as
`Hhat = H2 / max(1, ||H2||_inf)`. This per-image choice preserves one shared
RGB phase direction and gives `||Hhat||_inf <= 1` without channel-specific
rescaling. Define `b(x)=4*x*(1-x)`, `d=b*Hhat`, and use
`x_plus=x+0.05*d`, `x_zero=x`, `x_minus=x-0.05*d`. The locality audit repeats
the same frozen direction at epsilon `0.10`; no epsilon is selected from data.

Because `0 <= b <= 1` and `0.05*4*x*(1-x) <= min(x,1-x)`, both branches are in
`[0,1]` by construction. The implementation still records observed bounds
and rejects values outside the tolerance `1e-7`.

## Ordered gates

1. Synthetic proof fixtures: finite branches, box bounds, midpoint identity.
2. Real exact-support branches: finite/nonconstant maps, midpoint identity
   `<=1e-6`, no-op identity, and unscaled linear central residual accepted only
   against predeclared fp32/fp64 arithmetic bounds.
3. Epsilon locality Spearman `rho >= 0.90`, scale ratio `[0.5,2.0]`.
4. Support leave-one-out calibration scale finite with `max/min <=2`.
5. Intervention confounds each have absolute Spearman `<=0.30` against
   patch `|d|`, barrier `b`, and `|Hhat|` activity.

Any failed gate is `KILL / FATAL_SUPPORT_GATE`; no post-result epsilon,
normalization, layer, or tolerance tuning is allowed. Passing one object allows
the pre-registered all-eight support-only validity shards, not public scoring.

## Decision boundary

All gates pass: `PROMISING_SUPPORT_VALIDITY` and freeze a separate held-object
mechanism assay. Any fatal gate: `KILL_FOR_CLAIM / NO_CONTINUE` for this exact
candidate. Neither outcome supports SuperAD/RN/public metrics at this stage.

