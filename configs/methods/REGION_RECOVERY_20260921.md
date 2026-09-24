# Fixed proposal 1: guided region recovery

Locked before evaluation on 2026-09-21. Hypothesis: bounded expansion of
high-score seeds recovers partially detected regions and improves overall
AUPRO@0.05 over RSG3. No claim of novelty: region growing/hysteresis are
established; AnomalyDINO (WACV 2025) and UniVAD (CVPR 2025) provide related
frozen matching/component-aware context, not evidence for this rule.

RR_A is exact RSG3 with recovery disabled; RR_B enables recovery only.
Frozen DINOv2-L layers 5/11/23, resolutions 448/672/896/1120, four banks,
tailscale, residual-spike fusion, original CQ99/max remain fixed. No trainable
parameters or extra backbone calls. Same normal support16 manifest, seed0,
preprocessing, query population, evaluation and per-class device for A/B.

Fixed rule: raw q99 seeds; raw q95 candidate pixels supported by at least two
resolution q95 masks; RGB L2 <=0.1 and layer5 adjacent-cell cosine >=0.9
barriers; synchronous four-neighbour growth for four steps on max-edge1024
work grid. Nearest feature-cell edges are approximate, not semantic masks.
Preserve CQ output; lift only reached native raw q95..q99 pixels to raw q99.
No dense feature interpolation, query history, labels, masks or GT in method.

Budget: full8 x A/B =16 evaluations, one candidate, no sweep or automatic
replication. Split test_public_seen_development: repeatedly seen public
development, labels used only by evaluation. This is not final evaluation.
Primary screening: positive full8 macro AUPRO delta with macro pAUROC loss
at most0.5 percentage points. Report every class delta and flag AUPRO losses
over2pp for review, not a post-hoc exclusion. Report timing and available memory
evidence. A screening pass warrants replication, not a final adoption claim.
Failure stops this fixed candidate; runtime failure is not method failure.

Use existing generic runner/collector, local and dsba3 existing containers and
assigned GPUs, paired A/B on each device. Retain configs, manifests, metrics,
fit/timing logs and remote execution logs. Maps/weights disabled. No push,
container replacement, resource expansion, or unrelated proposal execution.
