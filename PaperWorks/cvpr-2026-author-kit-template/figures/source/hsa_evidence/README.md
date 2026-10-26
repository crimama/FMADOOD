# Direct HSA figure evidence — 2026-10-26

Post-hoc analysis of **seen MVTec AD 2 test_public**, not an untouched final
evaluation or a new fitted method. Both arms reuse the same 2-shot/seed-0
support, query, frozen features and nearest-neighbor distances for eight classes.
GT is used after prediction for counts, diagnostic thresholds and representative
image selection. No CQ or noise guard is applied.

## Protocol

Question: does full ScopeAD recover defects missed by raw L23/1120 at a similar
normal-tail false-alarm budget, and does it also lose previously detected defects?
This compares complete pipelines and does not isolate module-wise causation.

Thresholds are frozen from Table 6: every 100th normal pixel within each native
image, including normal background of bad images, is pooled per class and arm
across all acquisition conditions; Q99 is the threshold. No per-query or
per-condition refit. Counts use all native pixels. Actual FPR is measured,
not forced to equal 1%; thresholds are diagnostic, not deployment thresholds.

## Files and execution

- Class JSON: cache-index path/SHA256, cached configuration, threshold-source
  and executed-script SHA256, parity audits, histograms, native transitions,
  per-image counts and selection rule.
- Class example NPZ: authorized RGB, GT and binary display maps, maximum edge
  640 pixels. No full-resolution floating anomaly maps were transferred.
- table6_threshold_source.json: original /tmp/hsa_direct_validation.json.
- replay_source_v1.py and replay_source_v2.py: immutable executed source copies.
  V2 adds the can representative fallback (all its defects are <500 pixels);
  scoring and statistics are identical. Every class records its script hash.
  Canonical script is ../analyze_hsa_evidence.py (V2). An initial mask-path
  failure produced no results; retained V1 has the corrected test_public path.
- figure_summary.json: plot summaries and representative/failure identities.

The existing dsba3 container ran two CPU workers, two Torch threads each, without
fitting, GPU use or checkout updates. Remote source cache roots:

    /workspace/results_local_compatible/scopead_mechanism_20261020
    /var/tmp/fsad-analysis/scopead_mechanism_20261020

New reports and retained logs:

    /var/tmp/fsad-analysis/scopead_hsa_figure_20261026

Replay one class in that container:

    python3 analyze_hsa_evidence.py --class-name can --output /path/to/new/diagnostic

Regenerate figures locally from the paper directory:

    MPLCONFIGDIR=/tmp/scopead-mpl ../../.venv/bin/python figures/source/draw_hsa_evidence.py

## Verified evidence and limits

1,084 queries; 4,425,357,784 normal and 29,106,600 defect pixels. Class recalls
reproduce Table 6 within 1e-4; threshold differences are below 2e-5. Histogram
coverage and exact TP/FP transition identities pass for each image. Original
pipeline replay errors in cache records are checked to be <=2e-5.

Eight-class macro recovery is 13.5689%, regression 3.8538%, net +9.7151 pp
(9.72 rounded before subtraction). Raw/full recalls are 36.2358%/45.9509%;
subtracting separately rounded table cells gives 9.71 pp instead. Exact macro
normal FPR is 0.99086%/0.99236%. Seven classes improve; walnuts loses 3.3617 pp.
Nonregular recall is 36.8849%/46.2964%, FPR 1.05783%/1.05009%, using the same
pooled class thresholds. These are existing exposure/shift conditions, not
synthetic transformations.

Main example class vial is the upper median by eight-class net gain, not the
best class. Image selection uses the recorded closeness rule on nonregular bad
images; this remains post-hoc GT-based selection. Walnuts is a counterexample.
Display score/threshold t differs from support component calibration c_m.
Density curves are class macros; undisplayed tails remain in the denominator.

No claim follows of absolute background score suppression, improvement of all
ambiguous pairs, identical FPR per image, module-wise causation, multi-seed
significance, or unseen-shift generalization. The appendix also reports the
unfavorable baseline-conditioned pairwise diagnostic in
results/remote_runs/dsba3/scopead_hsa_overlap_20261024/audit_v2/comparison.md.
That comparator already includes Q95/layer fusion and differs from raw L23.
