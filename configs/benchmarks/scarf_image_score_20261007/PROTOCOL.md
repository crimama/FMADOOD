# SCARF image score: top-1% mean and image-level residual gate - 2026-10-07

Authorized by the user on 2026-10-07: report SCARF I-AUROC with the top-1% mean
image score, and test a separate image-level residual threshold.

Hypothesis (post-hoc, from the RSG4 MVTec AD/VisA diagnosis): the token-level
support q99 residual gate lets about 1% of normal tokens pass by construction,
so the image maximum of a normal image almost always contains a residual spike.
A support-calibrated image-level threshold (tau_img = support LOO shallow
maximum, q = 1.0) removes these spikes from the image score while keeping the
token-level gate for the pixel map.

Method change: image score only. The pixel anomaly map is unchanged, so AUPRO,
pAUROC and full pixel AUROC must reproduce the source runs.
- Primary image score (paper metric): top-1% mean of the final SCARF map,
  k = max(1, floor(N * 0.01)), the same rule as SuperAD and PatchCore.
- Recorded candidates per query: max_final (previous rule), top1pct_mean_final,
  max_image_gate, top1pct_mean_image_gate. The image-gate map uses the same deep
  base, bags, scales, interpolation and CQ post-processing, with tau_img in place
  of the token tau. Candidate AUROCs are evaluation-only.

Conditions: each config copies its source config exactly and adds only
image_score_aggregation=top1pct, image_score_source=final,
image_residual_tau_q=1.0 and image_score_candidates=true (see plan.json).
- AD2 test_public full8 (Table 4 SCARF): reference_matching_fast FP16 cap256,
  448/672/896/1120, bags 4, shots 1/2/4/8/16 x seeds 0/1/2 = 120 class runs.
- MVTec AD full15 shots 1/2/4/8 and VisA full12 shots 1/2/4/8/16 (Table 5
  SCARF): fusion_factorial FP32 cap256 gated, seed 0, paired E9 manifests =
  60 + 60 class runs.

Execution: dsba3 existing container, GPU UUIDs 0-3, container python3
(3.10, torch 2.1.2+cu118) as in the source runs. Isolated source = release
6d01d1873437e5cd2d38ce7b86fb50998b95794fc17904951d1a7aa9b85cfb6d plus
reference_matching.py, reference_matching_fast.py, fusion_factorial.py and
pipeline.py from the local tree (diff audited: image-score options and the
default-off negative residual option only).

Splits: AD2 public is seen development; MVTec AD and VisA test sets were used
in the post-hoc diagnosis that motivated tau_img. Labels and GT enter
evaluation only. The image-gate threshold q = 1.0 is fixed before the run; no
other value is tuned. Results are development evidence; a paper claim for the
image gate needs a fixed-rule confirmation.

Validation: pixel metrics and max_final I-AUROC must match the source runs;
primary score equals the top1pct_mean_final candidate; all candidates finite.
Maps and weights off. Remote logs retained.
