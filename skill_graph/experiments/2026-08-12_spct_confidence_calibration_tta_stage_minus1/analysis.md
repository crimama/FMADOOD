# SPCT Confidence, Calibration, Synthetic Gate, and Key-TTA Analysis

Date: 2026-08-12
Status: `DESIGN FROZEN / NOT RUN`

## Evidence intake

The immediately preceding SPCT diagnostic is valid and complete. Increasing
support from 16 images to the full `train/good` population improved macro R1
diagnostic pAUROC from `0.689831` to `0.769953` (`+0.080122`, paired 95% CI
`[+0.039551,+0.117081]`, four of four objects). Correct source and part
correspondence was causal: full R1 beat source shuffle by `+0.163114` and part
shuffle by `+0.258039`.

Two limitations define this continuation. First, the local tangent gain is
heterogeneous: it is material on `can` and `vial` and approximately absent on
`fabric` and `walnuts`. Second, full support increased held-normal FPR by
`+0.055402`. The prior PCA32 additive synthetic-margin head was exactly
inactive, so it must not be retuned as another additive score.

Earlier negative evidence also constrains the design:

- stable-channel and fixed-position retrieval keys did not beat their hard
  nulls; full anomaly values must remain untouched;
- naive online test-memory expansion is confounded by contamination,
  calibration drift, and stream order;
- the SO-CFM synthetic hinge was inactive and did not beat mask shuffle;
- calibration effects must be evaluated at fixed support-derived thresholds,
  not only by rank metrics invariant to monotone scaling.

## Research question

Can the already-positive full-support correspondence signal be converted into
a safer score by learning *when* local tangent projection is trustworthy,
calibrating its normal tail with support-image blocks, and using synthetic
anomalies only to supervise activation? After that support-only path is fixed,
can a strictly key-only query refinement recover part of the clean-key oracle
gap without modifying anomaly values, memory, tangents, or thresholds?

## Method boundary

This is a support-only synthetic Stage -1 diagnostic on the same paired
Probe16 panel as SPCT. It is not a public-test benchmark and cannot create a
SuperAD performance claim. Both `support16` and `full_train_good` budgets run.
All arms run even when an earlier gate is weak.
