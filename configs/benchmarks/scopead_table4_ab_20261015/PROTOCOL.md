# Table 4 A/B — deep feature-concatenation banks

## Question and boundaries
Fill the exact missing A/B cells in PaperWorks/outlines/Experiments.md Table 4.
A versus B tests resolution combination with matched configured cap summed
over resolutions. B versus the historical C tests a common reference-token
match across deep layers versus independent layerwise nearest neighbours.
No assumption is made about which arm wins.

- A: one resolution 1120, cap1024/source, concatenate layers [11,23].
- B: resolutions 448/672/896/1120, cap256/source/resolution, concatenate [11,23].
- Both: no score calibration, no shallow features or shallow fusion; 4 bags,
  k=1, mean resolution fusion, frozen DINOv2 ViT-L/14, FP16 extraction.
- CQ ON (percentile99, component maximum, no dilation), matching the existing
  Table 4 C/D/E/F source results. Do not mix with the CQ-off diagnostic suite.
- Same existing k9 noise guard. A routes to its sole 1120 resolution; B retains
  [448,672]. Inspect activations and report any effective-resolution limitation.
- All bank sampling is sparse. Cap matching concerns configured sampled token
  slots, not unique patches, computation, or equality of memory to layerwise C.
- Concat normalizes each layer and the concatenation, then uses one nearest
  reference token per query patch. No new method implementation.

## Population, metrics and budget
MVTec AD2 test_public, eight classes, previously seen development population.
Seed0, 2/16-shot with VOID_support_2.json / VOID_support_16.json. Preserve
no_mask_no_rotation preprocessing, support identities, backbone state and the
mvtec_ad2_float16_streaming_v1 full-native-pixel evaluation contract. Labels
are used only by evaluation, never fitting or prediction.

Primary: class-macro AUPRO@0.05, higher is better. Secondary:
pAUROC@0.05, higher is better. Budget: four configs times eight classes =
32 class evaluations, timeout10800 seconds per class. Stop after this matrix;
retain all outcomes and failures. No outcome-dependent selection or tuning.

## Execution and validation
Use scripts/experiments.py run in the existing dsba3 hun_fsad_tta_012 container
on the same four assigned GPU UUIDs. Prioritize this matrix after the calibration
diagnostics. Use a staged immutable release; do not activate or modify a checkout
used by another experiment. Save metrics/config/manifest and retain logs remotely.
Dense anomaly maps and fitted weights remain disabled.

Before interpreting A/B, verify support/query/backbone hashes, evaluation coverage,
caps, matching mode, and actual noise-guard activations. Compare historical C only
after the provenance audit; declare any source/backend or execution mismatch.
Use the existing collector for canonical Markdown/CSV registration after completion.
