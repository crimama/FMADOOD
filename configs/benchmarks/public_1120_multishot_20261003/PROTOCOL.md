# AD2 public 1120 multi-shot / multi-seed extension

Authorized 2026-10-03: complete all proposed experiments on dsba3, sequentially.

Hypothesis: SCARF retains its localization advantage when external comparators
receive the same maximum input short edge (1120), across support draws and shots.
This compares full systems, not identical feature observations or compute budgets.

## Budget and order

PatchCore / SuperAD / AnomalyDINO: 1,2,4,8,16 shots x seeds 0,1,2 x full8.
Reuse the 24 completed 16-shot seed0 class runs from public_1120_20261003.
New comparator budget 336: first 16-shot seeds1/2, then 1/2/4/8-shot seeds0/1/2.
Then supplement SCARF 1/4/8-shot seeds1/2: 48 class runs, cap256 / four scales.
Total new budget 384. Four GPU UUIDs, two disjoint classes per GPU, one config
and one class at a time per GPU through scripts/experiments.py run. No OOM or
performance-based pruning. A failed shard stops; preserve logs for diagnosis.
Per-class timeout 21600 seconds. No automatic container replacement or cleanup.

## Fixed conditions

AD2 test_public full8 is seen development data. Primary class-macro AUPRO@.05
(higher better); secondary pAUROC@.05 and I-AUROC. Three-seed mean and sample SD
are computed after full8 per-seed completion. GT/labels are evaluation-only.
Frozen DINOv2-L/14, native aspect ratio and unchanged method/scoring settings.
Comparators preserve original FP32, layers [5,11,17,23] or AnomalyDINO [23],
and override dataset and backbone short edge to1120 for every class.
Ordered per-shot/seed manifests are taken from existing Table4 AnomalyDINO
configs and shared by all methods; record their SHA256 in plan.json.
SCARF retains the user-selected FP16 execution, cap256 and four scales
448/672/896/1120. It is a separate precision/execution condition: do not pool
new FP16 seeds with historical FP32 seed0 as an unqualified three-seed table
entry. Collection must explicitly audit/label precision, or reuse a complete
matching precision set before replacing that table row.
Cost profiling on; maps and weights off. Retain remote execution logs and pull
only compact metrics/config/manifest/status evidence.

## Source and collection

Execute the exact immutable source release used by the completed 1120 pilot;
stage configuration files using run_remote.sh update-code --stage-only, without
activating or changing that source. Provenance records both execution release
and staged config release. No commit/push.
Collect via scripts/experiments.py collect --input-dirs, preserving canonical
Markdown and CSV histories. Verify full8 coverage, finite metrics, resolution,
normal support count, and support/query/GT fingerprints before paired claims.
No untouched holdout or fusion-only causality claim.
