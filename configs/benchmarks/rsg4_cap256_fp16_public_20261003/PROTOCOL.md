# RSG4 cap256 FP16 public validation — 2026-10-03

## Question and boundary
Does the FP16 optimized execution preserve the canonical FP32 RSG4 cap256 public performance? Both arms retain frozen DINOv2 ViT-L/14 layers [5,11,23], scales [448,672,896,1120], cap256 per source, 4 bags, k1, support tail calibration, residual gate, CQ, and noise guard. FP16 changes backbone autocast, CUDA feature retention, reusable calibration distances, shared bag distances and chunk512; downstream remains FP32. This compares execution plus precision, not autocast alone. No trainable parameters.

## Data and metrics
AD2 test_public, full8, 16 normal supports/class. Same fixed manifest within each seed, preprocessing, query/GT content and evaluator. Existing public development population; labels used only for evaluation, never fitting/prediction. Primary AUPRO@0.05 (higher is better); secondary pAUROC@0.05 and image AUROC; oracle F1 is diagnostic. No private evaluation or threshold tuning. Maps and weights off.

## Sequence and budget
Start paired seed0, eight classes per arm (16 class-runs), on four already-exposed dsba3 GPU UUIDs, one worker/GPU, two classes per shard. Fresh canonical FP32 comparator audits the stored seed0 evidence. Per-class timeout 7200 seconds. Existing container and retained remote logs. Stop a failing shard on execution/coverage/nonfinite error; preserve artifacts and retry only a diagnosed safe failure.

If seed0 passes a practical screening tolerance (macro AUPRO drop <=0.10 percentage points and worst-class drop <=0.50 pp; pAUROC and image AUROC reported separately), extend identical paired protocol to seed1 and seed2 (32 further class-runs). This tolerance is a development decision rule, not statistical equivalence. If it fails, report class differences and keep FP32 as the verified setting. Do not select changed hyperparameters on private scores.

## Registration
Use scripts/experiments.py run for each GPU shard and collect --input-dirs for their disjoint full8 combination. Register verified metrics and provenance in the canonical paired Markdown/CSV ledger; store paired precision/execution audit and image metrics in the run report. Generic collector may mark precision-different arms unmatched; a separate declared-ablation audit must validate common content and intentional differences, without relabeling this as a method-only comparison.
