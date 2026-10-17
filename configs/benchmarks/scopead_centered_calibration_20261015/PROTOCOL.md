# Centered normal-reference scaling — public development comparison

## Hypothesis and method boundary
Normal support LOO q95 division aligns an upper reference point but leaves observation centers unequal. Test whether subtracting a support-only fraction of the median improves fused localization and normal-query alignment. This is a development hypothesis, not an established improvement or novelty claim. No backbone parameters are trained.

Transform each observation and bag by (d − rho Q50) / max(Q95 − rho Q50, 1e-4). Q50 and Q95 come from the existing normal support leave-source-out distances. Keep negative scores. No query statistics or labels enter fitting or prediction. The unchanged denominator floor may prevent exact Q95=1 for degenerate samples; retain and report it.

## Controlled matrix
- Four arms: none (raw), q95 (rho=0), partial (rho=0.5), centered (rho=1).
- MVTec AD 2 test_public, all eight classes; already seen development data.
- 2 and 16 shots, seed 0, existing VOID_support_2/16.json manifests.
- Frozen DINOv2 ViT-L/14, FP16, layers 5/11/23, resolutions 448/672/896/1120, four bags, 256 tokens/source.
- CQ off; deep mean + 0.125 shallow (ablation_shallow_mode=all), existing k=9 noise guard and evaluator fixed. No other parameter changes.
- Fresh none and q95 controls run with the same new instrumentation. Verify support/query identities and bank feature digests before computing paired deltas.
- Primary AUPRO@0.05, secondary pAUROC@0.05, higher is better. F1 oracle is diagnostic only. Performance and distribution alignment are separate endpoints.

## Measurements
Retain the existing paired native-pixel sampling after bag mean for normal/defect and fused maps. Add support LOO Q50/Q95/offset/scale and transformed quantiles per observation/bag. On mask-empty query images only, sample native feature-grid raw and normalized distances BEFORE interpolation or bag mean, at matched coordinates. Reservoir cap: 256 tokens/image then 1024/observation/bag. Also count all native tokens exceeding their raw support Q95, normalized 1, and below 0. Normalized >1 corresponds to the raw reference tail for tailscale only (subject to floor); it has no such meaning for raw-distance control.

Native samples are image-capped descriptive distributions, not full pixel quantiles. Threshold counters use all native tokens on mask-empty images; these are not the evaluator's full-resolution FPR. Labels/masks are used only by the diagnostic after prediction. A native token on a mixed normal/defect image is not assigned a pixel label. No dense maps or weights are saved.

## Execution and stopping
64 class evaluations = 4 arms × 2 shots × 8 classes. Four assigned GPU UUIDs in the existing dsba3 hun_fsad_tta_012 container; two classes/lane. New candidates run first. Start immediately without predecessor waits; retain concurrent PatchCore CPU work. Observed GPU usage before launch: 2–3.2 GiB, utilization 0%; host load about 6 on 40 cores. Cost timing is not an isolated benchmark. Stop at this fixed matrix, no outcome-dependent stopping or extra tuning. Per-class timeout 10800 seconds.

Use scripts/experiments.py run and collect. Stage an immutable source snapshot with run_remote.sh update-code --stage-only, pin its absolute path, and preserve the active checkout. No git push, container replacement or GPU expansion. Save metrics/config/manifests and retain logs remotely; pull compact provenance. Keep mixed/negative outcomes. Collect all four shards to the canonical comparison Markdown and paired CSV.

## Local validation
78 targeted tests passed: existing reference/fast paths and diagnostic, centered formula, rho=0 default equality, nondegenerate reference/fast agreement, support threshold transformation, negative scores, denominator floor and invalid rho. Tiny nearly collinear features can amplify existing floating-point differences in LOO matrix multiplication near the 1e-4 floor; nondegenerate normalized features test path equivalence, with a separate finite-floor test.
