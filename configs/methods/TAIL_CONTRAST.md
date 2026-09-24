# TailContrast: tail-focused paired contrast adaptation

Development hypothesis: BCMA B2's small gain is limited by detached hard-normal
scores and baseline imitation. Direct normal-tail gradients and shift-stable
defect contrast may improve low-FPR ranking. No novelty or improvement claim is
made before the controlled comparison.

| Config | Hard-normal ranking | Paired contrast | Low-rank adapter |
|---|---|---|---|
| tail_contrast_D0.yaml | off | off | off |
| tail_contrast_D1.yaml | on | off | off |
| tail_contrast_D2.yaml | on | on | off |
| tail_contrast_D3.yaml | on | on | on |

D0 is exactly BCMA B2 (same context metric, training and sampling). D1 replaces
the synthetic ranking sampler with sampled top-5%-normal-tail plus random
normal pixels; BOTH positive and normal negative score values receive gradients.
The top-tail index selection alone is detached. No test labels/data are used.

D2 replaces clean baseline imitation and shifted-to-baseline imitation with:
`L_normal = mean((s(Tx)-s(x))^2)` and, on synthetic defect pixels,
`L_contrast = mean(((s(TAx)-s(Tx))-(s(Ax)-s(x)))^2)`.
`L_total = L_rank + L_normal + L_contrast`. D1 instead retains the original
unit-weight clean and shifted baseline losses. D2/D3 keep_loss remains a
diagnostic only, explicitly logged as keep_loss_optimized=false.

D3 adds `normalize(z + U V^T z)` before the existing context metric, on BOTH
query and bank features. Rank=16; U starts at zero and V random. Backbone
weights, raw bank features and token identities remain frozen. The 0.2 metric
bound is relative to adapted-feature NN, NOT the original baseline. The extra
32768 parameters (DINOv2-L dimension 1024) and changed feature conditions are
declared, not hidden as baseline-equivalent compute.

For D1-D3, four no-grad score maps become temporary differentiable loss
variables. Their gradients are backpropagated through recomputed scores one
view at a time, before a single optimizer step. This preserves all cross-view
gradients while bounding retained model graphs. Frozen feature caches last one
step, on CPU; dense maps/checkpoints are not saved. Log total/component losses,
family/source, normal score gradient, gradient norm, and mean scores.

Matched conditions: seed0, normal random16, frozen DINOv2 ViT-L/14 layer23,
448+672, source-excluded banks during fitting, full-bank NN, bank256/source,
64 steps, lr0.001, margin0.05, no_mask_no_rotation. Do not change the two
synthetic families or photometric distribution in this ablation. Both
resolutions see each training step; B3 correspondence is absent throughout.

Primary AUPRO@0.05; secondary pAUROC@0.05; F1 oracle diagnostic. All 8 AD2
test_public classes are seen development, not a sealed holdout. First use one
seed; no search/repeats. D0-D3 run sequentially in the existing local project
container. Budget: 32 class runs, <=3600s/class (32h hard ceiling); likely
runtime is longer than BCMA due to score replay and must be measured. Stop on
execution errors/nonfinite loss, preserve artifacts, do not silently retry.
Compare final metrics, per-class regressions, support/query content hashes,
and D0/B2 reproduction. A training loss drop alone is not success. Update
`results/comparisons.md` and its canonical CSV with completed results.
Use the shared `scripts/experiments.py run --configs ...` for sequencing and
`scripts/experiments.py collect --output-dir ... --watch` for host-side ledger
updates. New statuses carry their plan; see `docker/local-bcma.md` for the
explicit plan options required by the currently running older queue.

Prior-work sanity check (2026-09-11): tail selection relates to ICML 2022
[When AUC meets DRO](https://proceedings.mlr.press/v162/zhu22g.html), but our
sampled surrogate is not exact AUPRO optimization. Feature modulation relates
to WACV 2024 [ReConPatch](https://openaccess.thecvf.com/content/WACV2024/html/Hyun_ReConPatch_Contrastive_Patch_Representation_Learning_for_Industrial_Anomaly_Detection_WACV_2024_paper.html).
These generic components are not themselves novel. Synthetic-only
discrimination may fail on real subtle defects; shift invariance may suppress
color defects; the low-rank adapter may overfit 16 supports. No quantitative
gain is assumed.
