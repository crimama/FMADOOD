# ScopeAD calibration distributions — AD2 public development

## Question
Does normal-reference q95 scaling reduce observation scale imbalance on evaluation images, and does that change improve fused localization? Distribution alignment and performance improvement are separate endpoints. Mixed or negative outcomes remain in the report.

## Paired comparison
- MVTec AD 2 test_public, all eight classes; previously seen development population.
- Seed 0; 2 and 16 shots, existing VOID_support_2.json / VOID_support_16.json manifests. Verify support, query and bank digests across none/q95 arms before interpreting deltas.
- Frozen DINOv2 ViT-L/14, FP16 extraction, layers 5/11/23, resolutions 448/672/896/1120, 4 bags, 256 bank tokens/source, deep mean + 0.125 shallow.
- Only score_normalization differs within each shot pair: none vs tailscale (normal support LOO q95). No fitting on query images or labels. No weight tuning.
- CQ OFF in BOTH arms. Other existing settings including the k=9 noise guard remain fixed. Diagnostic fails if routed resolutions change within a class; inspect activations before comparing.
- Fresh CQ-off comparison; old CQ-on results are not the comparator.

## Measurements and figures
1. Per-observation raw and normalized distance distributions on the SAME evaluation coordinates, separated into normal/defect pixels only after prediction. Bag-mean values; normalization occurs before bag mean.
2. Actual fused-score normal/defect distributions for none/q95, with paired class AUPRO@0.05 (primary, higher better), pAUROC@0.05 (secondary). Full native-pixel metrics use the unchanged evaluator.
3. Observation contribution summaries: coefficients 0.125 for layer5, 0.5 for layer11/23, divided by resolution count. These are descriptive contributions, not an independent causal ablation.
4. Class-by-class figures at both shots; do not select only favorable cases. Illustrative case selection must be labeled post hoc.

Diagnostic sampling: at most 256 native coordinates per image/normal-or-defect stratum; deterministic identity-based sampling; at most 2048 coordinates per class/stratum in a priority reservoir. Same coordinates and priorities across both arms. This is an image-capped sample, not an unbiased full-population pixel distribution. OpenCV bilinear diagnostic sampling; predictions retain torch bilinear. No saved dense maps, weights or extra backbone calls. Compact samples and per-image sample means reside in run_manifest.json object metadata.

## Budget and execution
32 class evaluations = 8 classes x 2 shots x 2 normalization arms. Existing dsba3 container and four assigned GPU UUIDs. User requested immediate execution alongside existing PatchCore work: GPU memory usage was 2–3 GiB of 48 GiB and utilization 0%; 40 CPU cores with load about 5.4. No waiting for predecessor jobs. Stage an immutable source release without activating it; run by pinned absolute release path. Per-class timeout 10800 seconds; stop at this matrix, no outcome-dependent early stopping. Concurrent execution means runtime is not an isolated cost measurement.

Launch correction: initial immediate launch failed before any class evaluation because the runner requires a new output directory. Launcher state remains in gpuN; evaluation output uses gpuN/evaluation. Preserve failed launch logs.

Local validation: audit-on/off prediction parity for none and q95, reconstruction of actual fusion from grids, bounded paired sampling, error cleanup. Record source release SHA and full runner provenance. No numerical claim until paired cells complete and provenance passes. Collection uses scripts/experiments.py collect with all four disjoint shards and updates the canonical Markdown/CSV together after completion.
