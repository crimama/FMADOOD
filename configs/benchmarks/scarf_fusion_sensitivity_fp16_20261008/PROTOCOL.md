# FP16 residual controls and quantile sensitivity — 2026-10-08

Authorized: user requested experiments 1 (fusion controls) and 3 (quantile
sensitivity). Reuse the existing dsba3 container and its four assigned GPU UUIDs.

## Hypotheses and fixed conditions

1. Support-calibrated excess fusion has a useful conditional advantage over
   deep-only, a selected global linear shallow weight, and hard gating.
2. Local changes around q=.95 and beta=.99 have bounded performance effects.
   Report observed changes; do not assume stability or superiority beforehand.

MVTec AD 2 test_public, full8, seed 0; this is previously seen development data.
Frozen DINOv2 ViT-L/14, layers [5,11,23], FP16 autocast, 448/672/896/1120,
cap256 per source, four bags, k=1, query chunk512, fast calibration-bank reuse
and shared bag distances. Support manifests: VOID_support_2.json and
VOID_support_16.json, identical within shot. CQ99 and the existing k9 lowres
noise guard are fixed across arms. Routing can use two scales for detected
noisy queries, exactly as in the existing final FP16 configuration; four-scale
configuration is retained. No backbone adaptation or trainable parameters.

Image score uses the final map's top-1% mean. Separate image residual threshold
and candidate diagnostics are disabled in every arm. Cost instrumentation is
disabled in every arm; this study measures performance rather than timing.
Query labels/GT are evaluator-only and cannot enter fitting or prediction.
Primary: full8 macro AUPRO@.05 (higher better). Secondary: standardized partial
pixel AUROC@.05; class regressions and paired deltas are reported. Oracle F1
remains a diagnostic.

## Arms and budget

For d=(z11+z23)/2, s=z5, t=normal-support q99/q95:

- Excess (default): d+relu(s-t).
- Deep: d, with all three feature layers and identical support banks retained.
- Hard gate: d+s*1[s>t].
- Linear: d+lambda*s, lambda in {0,.125,.25,.5,1,2}. Deep is lambda=0.

Eight fusion arms at 2 and 16 shots: 128 class runs.
Sensitivity at 16 shots adds q=.90/.98 with beta=.99, and beta=.95/.98/.995
with q=.95: 40 class runs. Default gated_s16 is the shared sensitivity control.
Total: 21 configurations, 168 class runs. All controls are rerun under one
release/runtime because historical controls used FP32 and a different pipeline.

Select one global lambda using only 2-shot full8 macro AUPRO over the six
linear candidates. Break unscaled metric ties within 1e-10 toward smaller
lambda. Record this choice before interpreting 16-shot results. The complete
16-shot weight grid is transfer/sensitivity evidence, not a second selection
objective. Use of development labels for lambda selection is explicit.

## Execution and validity

Use scripts/experiments.py run, one worker per GPU, timeout 10800 s/class.
Each shard runs the same plan on two disjoint classes:

| GPU | UUID | Classes |
|---|---|---|
| 0 | GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f | can, fruit_jelly |
| 1 | GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae | fabric, walnuts |
| 2 | GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639 | rice, wallplugs |
| 3 | GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62 | sheet_metal, vial |

Wait for the previous table5 suite to complete before using its devices.
Validate locally, then install a fixed local snapshot with run_remote.sh.
No git push, container replacement, map/weight outputs, or artifact cleanup.
Stop an affected shard on execution errors; retain evidence and diagnose a safe
retry within this budget. No performance-based stopping or extra grid search.

Save config/support hashes, source digest, runtime, command, status, logs and
metrics. Collect compact results with run_remote.sh pull and the existing
experiments.py collector, updating the canonical Markdown/CSV together.
Verify support/query/GT and bank provenance before interpreting paired deltas.
Update Outline_v17 tables 8/11 only from verified complete class scopes; keep
the historical FP32 ablation conditions distinct from this final FP16 study.
