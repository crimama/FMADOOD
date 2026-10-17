# Paired reference-bank distribution diagnosis — 2026-10-17

User authorized the diagnostic proposed after rejecting synthetic-anomaly heuristics.
Hypothesis: support source-LOO Q95 is estimated against S-1 normal sources while
queries use S sources, creating observation-dependent relative-scale mismatch.
This experiment measures that mismatch; it does not assume it causes oracle gains.

## Fixed conditions and budget

- AD2 test_public, previously seen development, seed0, normal VOID manifests for
  2/16 shots, eight classes; 16 pipeline cells total.
- Unchanged Q95, FP16 DINOv2-L, layers5/11/23, resolutions448/672/896/1120,
  four random bags, cap256/source/bag, cosine1NN, linear0125, CQoff.
- All queries retain the existing pipeline/evaluator and Q95 metrics. Primary
  descriptive quantities are normal-query Q95/support-Q95 ratios and empirical
  threshold exceedance; AUPRO@.05 and pAUROC@.05 are baseline parity checks.
- No trainable parameters, fitted weights, preprocessing, support/query changes.
- Two currently free assigned GPUs, physical0/2, same existing hun_fsad_tta_012
  container, immutable stage-only snapshot. Preserve other jobs and active checkout.
- Timeout10800sec/cell; stop after16 cells. No automatic new method or GT fitting.

## Paired comparison

Capture256 image-relative nearest native-token probes/resolution from the existing
query backbone pass. After unchanged prediction returns, use evaluation GT only
to select wholly mask-empty query images; labels never enter fitting/scoring.
For each selected query, bag and layer, compare exactly the SAME probes against:

1. The original full S-source bank.
2. Every S-1-source bank obtained by excluding one original source, preserving
   all remaining original token identities. Average conditions equally when
   constructing the excluded-reference empirical distribution.

Compute all exclusions from per-source minimum distances using one matrix
product, not additional backbone passes. Assert source removal never decreases
1NN distance and full-distance parity with original captured raw scores<=2e-5.
No query/calibration refit, bank expansion, or anomaly synthesis.

## Interpretation limits and decision

- Excluded/full normal-query ratio isolates reference removal on same queries.
- Excluded-query/support-LOO ratio also includes support/query distribution shift
  and token sampling differences. It is not a pure reference-size measurement.
- Preserve per-image quantiles/exceedance, all48 observation/bag records, normal
  image identities, diagnostics overhead and original pipeline provenance. Spatial
  tokens, exclusions and bags are dependent; no iid-token CI, p-value or FPR guarantee.
- Support Q95 uses the exact fitted divisor including existing numerical floor;
  retain raw Q95 and clamp flag. Undefined near-zero ratios are missing, not zero.
- A large common factor cannot change within-class pixel ranking by itself;
  compare variation across resolutions/layers as well as average magnitude.
- Only after inspecting the observed relative mismatch, decide whether matched
  reference cross-fitting merits a controlled candidate. Do not tune a divisor
  with normal query labels or present a development diagnostic as deployable fitting.
- Existing Weibull bank-size correction did not consistently improve AUPRO;
  this paired measurement tests remaining assumptions rather than repeats it.

## Outputs

Use existing experiments.py run/collect and existing remote helper. Save standard
metrics/config/manifests/status/logs; embed compact reference_bank_diagnostic in
run_manifest.json so standard pull retains it. No dense maps, features, weights
or new cumulative ledger. Diagnostic timing is not inference-cost evidence.
