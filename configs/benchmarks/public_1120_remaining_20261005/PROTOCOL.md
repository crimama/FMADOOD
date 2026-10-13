# Remaining Table4 comparators at effective input1120

Authorized 2026-10-05, following the completed PatchCore/SuperAD/AnomalyDINO suite.
Hypothesis: SCARF retains its public localization advantage against the remaining
methods when their effective backbone input is1120. This is a whole-system
resolution ablation, not equal compute, precision, feature layers or backbones.

## Conditions and budget
AD2 test_public seen development; full8; shots1/2/4/8/16; seeds0/1/2.
Primary AUPRO@.05, secondary pAUROC@.05 and I-AUROC; higher is better.
Same ordered normal support manifests as prior1120 suite, SHA256 in plan.json.
GT/labels only enter evaluation. No query-based fitting or hyperparameter tuning.
FP32 comparator inference; native method scoring, augmentation, coreset and fitting.
FastRecon frozen ImageNet ResNet50: square1120, 15% coreset, lambda0.3.
Explicit backbone resolution_ablation flag enables the adapter extension; historical
368 default is retained. Grid140x140; smoothing sigma4 remains unchanged.
RAD frozen DINOv3-B/16: square1280 resize, center1120 crop; preserves512:448
resize/crop ratio and effective input1120, with unchanged256 score-map stage.
SubspaceAD frozen DINOv2-registers-G/14: square1120, layers22-28, 30 rotations,
99% PCA; two-pass support-only fitting remains unchanged.
RegAD excluded from this launch: released STN FC geometry and adapter require224.
Resizing a dataset to1120 while the STN receives224 is not a valid1120 comparator.
Architecture/checkpoint adaptation would require a separate experiment decision.
SCARF comparator uses completed cap256 FP16 four-scale448/672/896/1120 results.

45 configs x8 classes =360 new class runs, no performance-based pruning.
Four GPU UUIDs0-3, disjoint two-class shards; sequential config/class per GPU.
Class timeout21600s; failures stop the affected shard and retain logs for diagnosis.
Maps/weights off; cost profiling on. Normal supports and full8 coverage, finite
metrics, effective resolutions and support/query/GT hashes must be audited before
paper aggregation. Mean and sample SD are over three full8 seed macros.

## Source and outputs
Stage local snapshot without activating current. Copy the immutable execution
release from prior1120 comparison into an isolated source directory, replacing
only src/fmad/methods/fastrecon_paper.py with the declared input-size extension.
Record file-level hashes and changed paths. Reuse existing container and weights.
No commit/push, container replacement, checkout update or artifact deletion.
Remote root /results/public_1120_remaining_20261005; source and config paths in
execution_provenance.json. Existing generic scripts/experiments.py run/collect.
Collect compact artifacts and update canonical Markdown/CSV together, then
paper Table4a (effective1120 ablations); native Table4 values remain traceable.
