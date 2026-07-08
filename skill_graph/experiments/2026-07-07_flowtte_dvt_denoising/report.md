# FlowTTE DVT-Style Position Denoising Probe

Date: 2026-07-07
Verdict: CONTINUE_DIAGNOSTIC / PROMISING_COMPONENT

## Scope

This run tests a DVT-inspired feature denoising probe, not a full DVT reproduction.
The implemented transform estimates a support-only position artifact field:

```text
G_pos = mean_support_feature_at_position - global_support_feature_mean
p_clean = p_raw - alpha * G_pos
```

No per-image neural field optimization and no trained DVT student are used.

## Shared Setting

- Dataset: MVTec AD2 TESTpublic
- Backbone: `dinov3_vitl16`, layers `[5,11,17,23]`, `feature_fusion=layer_norm_mean`
- Support: 16-shot, fixed DINOv3 no-context coreset support JSON
- Flow: 3 epochs, 2 coupling layers, hidden multiplier 1
- Memory: static no-TTE, `expansion_budget=1.0`
- Scoring: latent distance + density weight 0.25
- Metrics: `seg_AUROC_0.05`, `seg_F1`

## Reduced Alpha Sweep

| variant | AUROC_0.05 | F1 |
|---|---:|---:|
| base_no_dvt | 0.747299 | 0.285387 |
| dvt_pos_a05 | 0.755856 | 0.301430 |
| dvt_pos_a10 | 0.778458 | 0.336546 |
| identity_dvt_pos_a10 | 0.776635 | 0.345968 |
| identity_no_dvt | 0.746022 | 0.293262 |


Reduced result: alpha 1.0 was best among Flow variants. The identity diagnostic also improved, so the denoising effect is not only an NF-interaction artifact.

## All-8 Comparison

| method | AUROC_0.05 | F1 | delta AUROC vs base | delta F1 vs base | note |
|---|---:|---:|---:|---:|---|
| SuperAD-16 recorded | 0.765802 | 0.385534 | -0.031941 | -0.052266 | recorded comparator from component ablation context |
| SuperADD reported | 0.839300 | 0.626112 | +0.041557 | +0.188312 | reported Table 1 context comparator, not same-run artifact |
| Flow no-context base | 0.797743 | 0.437800 | +0.000000 | +0.000000 | DINOv3 Flow-LatentBank no-TTE, no DVT |
| CLS soft w10 | 0.805427 | 0.447118 | +0.007684 | +0.009318 | best previous component ablation variant |
| DVT position denoise alpha 1.0 | 0.825207 | 0.468348 | +0.027464 | +0.030548 | support-fitted position artifact field, Flow transform retained |


DVT position denoising improves the no-context base by `+0.027464` AUROC and `+0.030548` F1.
It also improves over the previous best `CLS soft w10` by `+0.019780` AUROC and `+0.021230` F1.

## Per-Object Delta vs Base

| object | base AUROC | DVT AUROC | delta AUROC | base F1 | DVT F1 | delta F1 | artifact L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| can | 0.676314 | 0.649947 | -0.026367 | 0.002963 | 0.002580 | -0.000383 | 0.403000 |
| fabric | 0.755118 | 0.898430 | +0.143312 | 0.325347 | 0.521358 | +0.196011 | 0.301285 |
| fruit_jelly | 0.813073 | 0.799020 | -0.014053 | 0.541089 | 0.517707 | -0.023382 | 0.375811 |
| rice | 0.946106 | 0.949752 | +0.003646 | 0.695460 | 0.692959 | -0.002501 | 0.285537 |
| sheet_metal | 0.765878 | 0.888738 | +0.122860 | 0.427553 | 0.517492 | +0.089939 | 0.354991 |
| vial | 0.705036 | 0.714034 | +0.008998 | 0.394468 | 0.390019 | -0.004449 | 0.444731 |
| wallplugs | 0.852729 | 0.851421 | -0.001308 | 0.418772 | 0.432228 | +0.013455 | 0.298930 |
| walnuts | 0.867691 | 0.850313 | -0.017378 | 0.696749 | 0.672444 | -0.024304 | 0.322536 |


## Interpretation

The gain is not uniform. Large improvements on `fabric` and `sheet_metal` dominate the mean, while `can`, `fruit_jelly`, and `walnuts` drop. This supports the idea that a position artifact component exists and can hurt ranking, but a single global alpha is too blunt for all object types.

Next diagnostic should test object-adaptive alpha or a lower-risk denoising gate, especially because some object classes lose anomaly separation after position field subtraction.
