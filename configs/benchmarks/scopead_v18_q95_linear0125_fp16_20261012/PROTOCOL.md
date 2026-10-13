# ScopeAD v18 evidence completion — 2026-10-12

## Question and frozen method
Complete the missing Outline_v18 evidence while retaining q95 after local-center failed its development acceptance rule. This is evidence completion, not another calibration search. Frozen DINOv2 ViT-L/14 layers 5/11/23, FP16 autocast, q95 source-excluded support distances, linear0125, cap256/source, four bags, CQ99, top1% mean, k9 noise guard. No trainable parameters. Ground truth is evaluator-only.

## Matrix and budget
1. Tables 10/12/12a: AD2 seed0 16shot, MR cap256, 1120 cap256, 1120 cap1024, MR cap1024; 32 class evaluations. All four arms enable cost_profile and native condition_strata. Single-resolution noise guard uses [1120], as in the prior single-resolution diagnostic; disclose that difference. MR256 vs 1120cap1024 matches nominal total sampled vectors; verify actual bank counts before claiming matched capacity. MR cap256 remains the main setting.
2. Tables 4/5b: shots 1/4/8, seeds 1/2, AD2 full8, MVTec full15, VisA full12; 210 new class evaluations. Reuse existing seed0 and 2/16shot evidence. AD2 uses the public_v13 seed manifests shared with Table4 comparators. MVTec/VisA use deterministic support selection with seed, unpaired with external methods. No deep-only runs.
3. Figure2: AD2 seed0 shots1/2/4/8 with native acquisition strata; 32 class evaluations. The 16shot results come from step1. Regular/Shift1 full8 and Shift2/3 seven classes require explicit actual coverage.
4. Table7 Ours: MVTec15classes, 16shot seed0, clean plus existing severity3 brightness/contrast/defocus_blur/gaussian_noise roots; 75 evaluations. Fixed seed0 support manifest. Verify clean support basename/content equality across conditions before interpreting performance differences.
5. Component costs: each AD2 class, all four cost arms, 12 deterministic normal/anomalous queries; 32 descriptive profiles. Report exclusive synchronized absolute time and percentage shares. Instrumentation overhead means component sums are distinct from uninstrumented full-query timing. Profile configs run independently to avoid treating different-resolution maps as execution-equivalence candidates. No saved maps or fitted weights.

Total: 349 class evaluations plus 32 component profiles. One worker on each existing dsba3 GPU UUID0–3, disjoint class shards, sequential phases, existing container only. Timeout10800seconds/class. Stop after this matrix. A failed phase stops its shard; retain logs and resume the same invalid_run identity only.

## Metrics and validity
AD2 primary AUPRO@.05, secondary pAUROC@.05 and image AUROC. MVTec/VisA main I-AUROC and full-FPR P-AUROC; corruption localization uses AUPRO/pAUROC@.05. Aggregate each seed's full class macro first, then mean and sample SD over seeds0/1/2. No favorable class/corruption selection. Verify support/query/GT and bank provenance within matched arms. Resolution/cap differences are explicit ablations. Cost and strata instrumentation are evaluation metadata additions. Repeatedly used public populations are seen development data, not untouched final evaluation. Lambda and q are fixed. Old gated results do not substitute for linear0125.

## Further v18 gaps
Table6 linear0125 private inference/threshold/submission; Table7 missing external methods; final qualitative panels remain separate followups. They are not launched in this batch. Private map output must be explicitly configured; browser login is not assumed current.

## Outputs and collection
Exact YAMLs, plan, runtime status, compact metrics/manifests and remote logs are retained. Pull compact outputs with the existing remote helper; logs stay remote. Collect complete affected phases using scripts/experiments.py collect to update canonical Markdown and CSV together. Component metrics.json files are descriptive profiles, not benchmark metrics; never import them as AUROC rows. Register only verified complete statistics in the outline.


## Correction record (2026-10-12)

The suite initially run under scopead_v18_q95_completion_fp16_20261012 was configured with layer_fusion residual_spike, support-quantile tau q99, and gamma 0.125. That computes mean(deep) + 0.125 * ReLU(shallow - tau) and is not the frozen linear0125 equation. Its artifacts and original configs are retained as historical evidence and are excluded from linear0125 claims. This corrected suite uses layer_fusion deep_plus_shallow, computing mean(deep) + 0.125 * shallow with no threshold or activation. All other declared conditions remain matched.
