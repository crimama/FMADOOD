# Fig.1 Pilot: Uncalibrated Equal-Weight Mean Baseline

## Purpose

This directory contains benchmark configurations for Figure 1 teaser figure: comparing an uncalibrated equal-weight 3-layer mean baseline against calibrated shallow-fusion arms from void_instrumentation.

**Dataset:** MVTec AD 2, test_public split  
**Backbone:** DINOv2-ViT-L/14  
**Protocol:** Few-shot learning (1, 2, 4, 8, 16 support shots per class)  
**Seed:** 0  
**Evaluation:** 8 classes (can, fabric, fruit_jelly, rice, sheet_metal, vial, wallplugs, walnuts)  
**Mechanism Audit:** Enabled  

## Arm Definitions

### mean3 Arms (New)

- **raw_mean3_sK**: Uncalibrated equal-weight mean (score_normalization='none')
  - shallow_mode = 'mean3'
  - score_normalization = 'none' (raw distances, divisor=1)
  - Fused output: (layer_5 + layer_11 + layer_23) / 3

- **qa_mean3_sK**: Calibrated equal-weight mean (score_normalization='tailscale')
  - shallow_mode = 'mean3'
  - score_normalization = 'tailscale' (Q_alpha normalization)
  - Fused output: (layer_5 + layer_11 + layer_23) / 3

### Reference Arms (from void_instrumentation)

- **maps_off_s16**: Deep-only baseline (no shallow layer)
- **maps_all_s16**: All shallow layer (unthresholded)
- **maps_gated_s16**: Gated residual shallow (tau-gated spike)

## Configuration Details

All configs use:
- **Resolutions:** [448, 672, 896, 1120]
- **Bank:** 256 tokens/source, 4 bags per resolution, k=1 nearest neighbor
- **Feature layers:** [5, 11, 23] (shallow, mid, deep)
- **Postprocess:** Component quantile (CQ), q=1.0, percentile=99.0
- **Fusion:** `residual_spike` remains configured for the matched support threshold and audit; `mean3` returns the equal three-layer mean.
- **Evaluation metric contract:** mvtec_ad2_float16_streaming_v1
- **Fusion diagnostic:** Enabled

### Map-Saving Variants (s16 only)

The configs prefixed with `maps_*_s16` enable anomaly map persistence:
- `maps_raw_mean3_s16.yaml`
- `maps_qa_mean3_s16.yaml`
- `maps_off_s16.yaml` (void reference, off mode)
- `maps_all_s16.yaml` (void reference, all mode)
- `maps_gated_s16.yaml` (void reference, gated mode)

## Design Notes

- **Post-hoc diagnostic ablation**: This benchmark is NOT a method change to fusion_factorial; it is a post-hoc control configuration to measure the baseline (uncalibrated mean) and compare against the calibrated shallow-fusion architectures from void_instrumentation.
- **Support manifests**: All configs share support manifests with void_instrumentation (configs/methods/VOID_support_K.json).
- **Matched pipelines**: The scoring configs differ from `void_s{K}_off.yaml` only in `shallow_mode` and `score_normalization`; map variants additionally enable `save_anomaly_maps`.
- The decision and case-selection rules were registered before the pilot in `PROTOCOL_20260929.md`.

## Related Documents

- `results/comparisons.md` — Cumulative comparison index with metrics
- `skill_graph/current.md` — Research state and assumptions
