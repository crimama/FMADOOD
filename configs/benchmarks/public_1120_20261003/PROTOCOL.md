# AD2 public: single-resolution 1120 comparator ablation

Recorded 2026-10-03 KST before launch. User requests PatchCore, SuperAD and
AnomalyDINO at 1120 using all four existing dsba3 GPUs.

## Question and scope

Does raising the external comparators' short edge to 1120 reduce the observed
gap to SCARF? This is a resolution ablation of the project's Table 4 settings.
SCARF's four scales remain a distinct full-system configuration; matching the
maximum resolution does not match feature observations, banks or computation.
No SCARF rerun or other comparator is included in this 24-cell budget.

## Fixed conditions

- MVTec AD 2 `test_public`, all eight classes, seen development data.
- 16 normal support images/class, seed 0, ordered fixed manifest
  `configs/methods/VOID_support_16.json`, identical to Table 4 seed-0 settings.
- Frozen DINOv2 ViT-L/14, original FP32 feature execution. PatchCore/SuperAD
  layers [5,11,17,23]; AnomalyDINO layer [23]. No trainable parameters.
- Change short edge to 1120 for every class, including sheet_metal. Native
  aspect ratio, preprocessing, method-specific retrieval/scoring, support bank
  rules, smoothing and evaluator remain fixed. Token count increases with
  resolution; no additional token cap or coreset reduction is introduced.
- `dataset_config.resolution=1120` explicitly replaces class defaults; the
  previous pipeline overwrote `backbone_config.smaller_edge_size` per class.
  The new optional override preserves default behavior when absent.
- Primary metric: class-macro AUPRO@.05, higher better. Secondary: standardized
  pAUROC@.05 and I-AUROC. F1 oracle is diagnostic only.
- Evaluation labels/GT are used only after prediction. No fitting/prediction
  label access, threshold tuning, class selection or hyperparameter sweep.
- Cost profiling enabled at inference batch 1; retain fitting time, inference
  time and memory evidence. Times are descriptive, not an isolated speed
  comparison against historical runs from other environments.
- No anomaly maps or fitted weights. Save metrics, config, manifests and
  training/fitting logs; full execution logs remain remote.

## Execution and validity

Existing container `hun_fsad_tta_012`; fixed GPU UUIDs and disjoint outputs.
One worker/GPU; 3 configurations x 2 classes each, 24 total class-runs.

| GPU | UUID | Classes |
|---|---|---|
| 0 | GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f | can, fruit_jelly |
| 1 | GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae | fabric, walnuts |
| 2 | GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639 | rice, wallplugs |
| 3 | GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62 | sheet_metal, vial |

GPU3 waits for successful completion of the existing fusion control shard
`/results/scarf_v16_fusion_20261003/controls/gpu3/status.json`. Preserve that
work and its immutable source. Stage a new immutable release with the existing
`scripts/run_remote.sh update-code --stage-only`; do not swap active source.
Launch from the staged release with its explicit PYTHONPATH and shared lock.

Use `scripts/experiments.py run`; retain status, PID/command, GPU UUID, release
hash, config digests and predecessor evidence. Per-class timeout 21600 seconds.
Stop the affected shard on execution failure; do not treat OOM/timeout as a
scientific loss. Retry only a diagnosed safe issue within these same identities.
No performance-based stopping. Full8 macro requires all eight unique classes
and finite, shape-complete coverage. Verify each recorded resolution is 1120
and support/query/GT content matches the baseline before paired interpretation.

## Collection

Pull compact quantitative/config/manifest/status evidence with the existing
remote helper. Combine disjoint GPU shards through `scripts/experiments.py
collect --input-dirs`, updating the canonical Markdown and CSV together.
Record per-class deltas and provenance in the suite's comparison.md. Preserve
runtime/code differences when comparing to Table 4, and do not claim exact
fusion-only causality or untouched final evaluation.
