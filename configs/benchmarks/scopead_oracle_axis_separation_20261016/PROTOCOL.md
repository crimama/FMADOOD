# Oracle axis separation: performance interventions
User requested separation of the existing48-scale oracle improvement.
Seen AD2 test_public,seed0,VOID normal support2/16,full8,CQoff,FP16,
same frozen backbone/banks/preprocessing/4resolutions/linear0125 and metrics.
16 class/shot pipeline cells, each extracts normal Q95 native observations ONCE
and reuses RAM components for ALL6 oracle interventions (96 searches total).
No maps/checkpoints exported. Query GT used ONLY in explicit oracle analysis.
Keep full baseline support/query/GT/bank manifests; retain original Q95 metrics.
Six positive multiplicative families, shape[resolution4,bag4,layer3]:
- layer:3parameters, shared over resolutions/bags.
- resolution:4parameters, shared over bags/layers.
- bag:4parameters, shared over resolutions/layers (global bag-index effect).
- resolution_layer:12parameters, shared over bags.
- resolution_layer_fixed_layer:12search parameters but9effective degrees;
  additionally enforce arithmetic mean multiplier=1 for EACH layer.
  Projection finds c per layer with mean(clip(c*raw,1/8,8))=1, by48bisection steps.
  This preserves each layer's total coefficient (including shallow0.125).
- full:48parameters, independent per observation; same search as previous oracle.
All families include unchanged original Q95 as final candidate.
Search: DE seed0,96population,20generations,plus initialQ95 objective=2017calls
PER family. Bounds1/8..8, sampled normal128/image and defect16/GTregion with
same importance weights and float16-tied sampled AUPRO@.05 as previous oracle.
Each family's top3 sampled candidates re-evaluated using FULL original-pixel
streaming AUPRO@.05 and pAUROC@.05. Same baseline and full reconstruction audits.
Select each family independently by full-pixel AUPRO, include Q95 original.
Each class retains ALL6 selected scales, metrics, dimensions, search settings,
sample/full agreement and baseline differences in oracle_scale_diagnostic.axis_ablation.
Primary standard metrics use the full48 family solely for collector compatibility;
axis-specific values are explicit diagnostic results in the SAME manifest/report.
No causal additive attribution or global optimum claimed. Families are nested
in places but finite optimization can make observed scores non-monotone; this
alone is not evidence that extra parameters hurt. No cross-arm coefficient
projection substituted for actual reoptimization. No ratio threshold for success.
Quantities to compare: layer-only vs resolution-only vs bag-only, resolution_layer
vs full (extra bag flexibility), fixed-layer vs unrestricted resolution_layer.
No assumption that full-layer mean constraint fixes actual score contribution.
Report8class effects, macro, and matched provenance; all are GT-using diagnosis,
not a deployed normal-support estimator or independently evaluated performance.
Per-cell timeout21600sec. End after16cells; no adaptive expansion/restarts.
Existing dsba3 container hun_fsad_tta_012, existing four GPU UUIDs, immutable source.
No active checkout switch, no other processes stopped, no periodic monitoring.
