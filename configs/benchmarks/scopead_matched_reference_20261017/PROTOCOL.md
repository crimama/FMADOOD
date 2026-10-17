# Matched-reference source cross-fitting — 2026-10-17

Hypothesis: the source-LOO S-1/full-query S reference mismatch creates unequal
relative calibration errors, especially at 2-shot. Hold out one support image,
fit its Q95 against the remaining source bank, and use the identical bank for
query scoring. Average predictions over every held-out source and original bag.
This is a standard cross-fitting construction; no novelty or coverage guarantee
is claimed. Native support/query distance sampling differences may remain.

## Controlled development comparison

AD2 previously seen test_public development; seed0; exact VOID_support_2.json;
all eight classes; FP16 DINOv2-L layers5/11/23; 448/672/896/1120; four original
bags; cap256 per source; cosine1NN; linear0125; unchanged noise guard; CQ off.
Backbone, data, preprocessing, source selection, initial token identities and
fusion are fixed. No training or query-derived fitting. No maps or weights.

1. q95_full: existing Q95, full S-source query bank, pooled support-LOO divisor.
2. q95_excluded_pooled: query each S-1-source fold; existing pooled Q95 divisor.
3. q95_excluded_matched: same query folds as (2), but each fold uses the Q95
   of its own held-out support source against that exact S-1-source bank.

The primary comparison (3)-(2) isolates fold-specific calibration with identical
query banks. (2)-(1) measures changing the query reference/fold aggregation.
(3)-(1) determines whether the complete candidate improves on current ScopeAD.
At 2-shot each fold uses one source; all 2 folds are averaged equally. Total
support budget stays 2, but effective per-fold reference support decreases to 1.
Noise routing still uses the original complete normal support set.

Primary metric AUPRO@0.05 (higher better); secondary pAUROC@0.05. This is a seen
development comparison, not a sealed final test. Labels enter only evaluation.
24 cells total, 3 configurations x 8 classes, four assigned GPUs in existing
hun_fsad_tta_012. Timeout10800sec/cell. Stop after these cells; no automatic new
parameter sweep, extra shots/seeds or submission. Retain remote logs and compact
results/config/support/source provenance; no periodic monitoring automation.

Prerequisites: verify physical source exclusion, held-out sample Q95 and equal
fold score average against a brute-force implementation; verify baseline bank
hash equality. Use immutable stage-only local source release, preserve current.
