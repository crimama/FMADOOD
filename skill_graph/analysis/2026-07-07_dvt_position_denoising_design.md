# DVT-Style Position Denoising Probe

## Purpose

This probe tests whether DVT's position-dependent, input-independent artifact view is useful for the current Flow-LatentBank no-TTE pipeline.

It is not a full DVT reproduction. The implemented module does not run per-image neural field optimization or train a lightweight denoiser student. Instead, it fits a simple support-only artifact field:

```text
G_pos = mean_support_feature_at_position - global_support_feature_mean
p_clean = p_raw - alpha * G_pos
```

This keeps the memory bank fixed and applies the same support-fitted transform to support and test features.

## Pipeline Placement

```text
DINOv3 multi-layer patch features
-> support-fitted position artifact denoising
-> NF transform or identity transform
-> support latent memory
-> nearest-distance anomaly map
```

Context/register features are not denoised in this probe. The denoiser only touches patch features used for localization.

## Current Ablation

Reduced MVTec AD2 objects:

```text
can, fabric, vial, wallplugs
```

Variants:

```text
base_no_dvt
dvt_pos_a05
dvt_pos_a10
identity_no_dvt
identity_dvt_pos_a10
```

The first three compare Flow-LatentBank with and without position denoising. The identity variants check whether the effect comes from NF interaction or from patch-space denoising itself.

## Success Criterion

Proceed to all-eight-object evaluation only if reduced results improve the primary metrics without clear object-level collapse:

```text
seg_AUROC_0.05
seg_F1
```

Secondary diagnostic:

```text
dvt_artifact_l2_mean in run_manifest.json
```
