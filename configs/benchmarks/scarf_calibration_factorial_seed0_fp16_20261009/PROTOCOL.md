# SCARF score calibration factorial — seed 0 development

## Question and hypotheses

Test whether support-derived score calibration improves SCARF map scoring, and whether its effect interacts with shallow residual fusion, resolution, support count, bank capacity, or the calibration leave-one-out estimator. Primary contrast: deep-only, q95 tailscale minus deep-only, no normalization, paired at the same dataset, shot, resolution, cap, and support manifest. MAD is a supplementary robust-scale alternative. Normalization rescales per-layer scores before fusion, so the fixed gamma=0.125 can have a different relative contribution after calibration than on raw distances. Interpret normalization within each fusion arm; do not treat cross-normalization gamma contrasts as an isolated calibration effect. The tau parameter does not explain the linear-fusion contrast.

## Fixed conditions and method boundary

Frozen DINOv2 ViT-L/14, feature layers 5/11/23, FP16 autocast; seed 0; existing exact support manifests; four bags; 256 sampled support tokens/source except the declared Stage B 1024-capacity controls; CQ99 final map, top-1% final-map image score, k=9 low-resolution noise guard; no training/adaptation, anomaly maps, or saved weights. No dataset object, ground truth, or evaluation labels are passed into fitting or prediction. Query labels/ground truth are evaluator-only. The populations have been seen during development: AD2 public full8, MVTec AD full15, VisA full12. This is not holdout or final-evaluation evidence.

## Stages and budget

Run in order A → B → C. Stage A is 4 resolutions (448/672/896/1120) × shots 1/2/4/8/16 × fusion (deep-only off, linear0125 all layers) × normalization (none/MAD/tailscale): 30 configs per dataset. Order configs by shot, then fusion, then normalization for paired outcomes. Stage B uses shots 2/16 at 1120 only, with 256 cap and the same 2×3 fusion/normalization factorial (12 configs/dataset), plus deep-only none/tailscale at cap 1024 (4 configs/dataset). The cap control holds nominal stored token count near 4×256; query computation is not matched. The guard stays enabled; since its declared resolutions must be a subset of the configured scales, the 1120-only arms declare `[1120]`. Record activations: any firing routes the query through its declared resolution and is a resolution-treatment confound requiring review; analyze non-firing queries separately if needed. Stage C uses shots 2/16 and q95 tailscale with each fusion mode, but `loo_estimator: token` (4 configs/dataset).

Stage C compares within-image leave-token-out calibration against the usual leave-source-out calibration; it does not include each token as its own nearest neighbor. At one shot, the existing source estimator already falls back to token LOO, so that condition is not in Stage C. This is a calibration-reference estimator contrast, not a self-including exact-self arm.

Total: 150 configs, 50 per dataset; 1,750 logical class runs (A 1,050; B 560; C 140). Run all fresh under the current code snapshot because prior support-matched artifacts have mismatched feature-bank digests. Keep runtime and hardware fixed within each paired dataset/object comparison. Historical matches are listed as candidates in plan.json only, with no planned reuse.

## Metrics and decision limits

Higher is better. AD2: AUPRO@0.05 primary, pAUROC@0.05 secondary. MVTec/VisA: image AUROC and full-FPR pixel AUROC, reported separately. Report every dataset and shot with paired class outcomes and macro differences; no favorable-subset selection. Seed 0 is development evidence and cannot establish significance or seed stability. Seed 1/2 confirmation is pending evidence and separate scheduling. A runtime/provenance failure invalidates that cell; it is not a method-performance result.

Per-class timeout is 10,800 seconds. No performance-based early stopping. Retry only recoverable runtime failures within the declared budget and retain failed-attempt provenance. Keep logs, metrics, config and manifest provenance; maps and weights remain off. `fusion_diagnostic` stays false; `ablation_bank_audit` stays true to record bank/scales provenance.
