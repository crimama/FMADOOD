# DINOv3 Attention Facets Stage -1 Preregistration

Date: 2026-08-10  
Status: frozen before metric execution  
Target: MVTec AD2 single-image, `test_public/good,bad`

## Negative Evidence Intake

This experiment is not a retune of the closed invariant-channel,
stable-key/full-value, positional-key, DVT, or score-field branches. Those
branches modify hidden channels, retrieval position, or post-backbone scores.
This assay opens a previously unmeasured internal evidence source: the actual
DINOv3 attention projections before and after RoPE (`Q`, `K`, and `V`).

The expected failure basin is anomaly absorption: a facet or self-self mixing
operator can reduce normal nuisance tails while smoothing small defects into
normal texture. A mean improvement cannot override an object-level collapse,
hard-null match, F1 loss, or component-coverage loss.

This is a diagnostic lane. It is not directly comparable to paper-aligned
SuperAD because the backbone is frozen DINOv3-H+/16 rather than DINOv2-L/14.
The maximum Stage -1 verdict is `PROMISING_DIAGNOSTIC`.

## Motivation and Claim

- Claim: useful nuisance-stable, defect-preserving evidence may remain in
  DINOv3-H+ attention facets even when post-block hidden descriptors fail.
- Evidence sought: a real Q/K/V facet must improve localization over the
  dimension-matched hidden-token baseline and beat wrong-layer/spatial nulls.
- Boundary: fixed support-16, frozen inference, no DVT, no Flow, no query
  adaptation, no test/private fitting, and no output-map postprocessing.
- Positioning: mechanism assay first; self-self propagation and cross-layer
  routing are gated continuations rather than independent sweeps.

## Frozen Data and Backbone Contract

- `target_dataset=MVTec AD2 single-image`
- Remote data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- Local mounted data root: `/home/hun/Volume/DATA/mvtec_ad_2`
- Support manifest:
  `skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json`
- Support SHA-256:
  `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`
- Exactly 16 frozen `train/good` images per object.
- Backbone: `facebook/dinov3-vith16plus-pretrain-lvd1689m`
- Runtime contract: transformers `4.56.2`, patch size 16, 32 blocks,
  4 register tokens.
- Stage A layers: `[15,31]`; layer expansion is forbidden inside Stage A.
- Resolution: 672, except `sheet_metal=448` if Stage C is reached.
- Prefix tokens: CLS and all register tokens are excluded from descriptors and
  patch affinities.

## Stage A: Direct Facet Matching

Objects: `can,fabric,rice,walnuts`.

Fixed candidate arms:

- `hidden`: post-block hidden-token control using the same two layers.
- `k_pre_rope`: query/support K before RoPE.
- `k_post_rope`: query/support K after RoPE.
- `v`: query/support V.
- `q`: query/support Q after RoPE.
- `qquery_ksupport`: asymmetric Q-query/K-support cosine-equivalent 1-NN.
- `equal_ensemble`: equal average of support-LOO-standardized member score
  maps; raw distance scales may not be averaged.

Hard nulls:

- `k_wrong_layer`: L15 query K matched against L31 support K.
- `k_query_roll`: K query map rolled by `(H//2,W//2)` before localization
  evaluation.

All descriptors are L2-normalized per patch and per layer before fixed layer
concatenation. Euclidean 1-NN on these descriptors is therefore
cosine-equivalent. Dense maps may exist in RAM only until evaluation and must
not be retained on disk.

Primary metrics:

- `seg_AUROC_0.05`
- `seg_F1`

Secondary diagnostics:

- `seg_AUPRO_0.05`
- image AUROC
- positive/nonconstant map coverage
- per-object deltas and weak/strong no-harm

Stage A continuation gate:

1. all custody, coverage, finite-map, shape, and access-ledger checks pass;
2. at least one real facet improves macro `seg_AUROC_0.05` and `seg_F1`
   versus `hidden`;
3. the same facet beats `k_wrong_layer` and `k_query_roll` on both primary
   metrics;
4. no object loses more than `0.03` absolute `seg_F1` or
   `seg_AUPRO_0.05` versus `hidden`;
5. no public/private/GT information is used for feature selection, fitting,
   temperature selection, or score calibration.

If no real facet satisfies all conditions, verdict is
`KILL_FOR_CLAIM / NO_CONTINUE`. A positive result is
`PROMISING_DIAGNOSTIC`, not a method claim.

## Stage B: One-Step Side Propagation

Stage B is authorized only if Stage A passes. It uses the Stage-A winning
facet/layer contract without retuning and preserves the raw DINO path.

Fixed arms:

- `identity_v`: raw V, no token mixing.
- `qk_to_v`: normalized QK affinity propagating V.
- `qq_to_v`, `kk_to_v`, `vv_to_v`.
- `qkv_to_v`: equal QQ/KK/VV affinity ensemble.
- `local_vv_to_v`: VV affinity restricted to a fixed radius of two patches.

Fixed nulls are uniform affinity, spatially rolled affinity, and row-shuffled
affinity. Temperature is fixed before public evaluation using support-only
entropy matching; no public metric sweep is allowed.

Stage B continues only when a propagation arm improves `seg_F1` and
`seg_AUPRO_0.05`, does not reduce `seg_AUROC_0.05`, beats every affinity null,
and passes the same per-object no-harm gate.

## Stage C: Mid-to-Late Routing

Stage C is authorized only if Stage B passes and expands once to all eight
objects. Fixed candidates are:

- `kk_L15_to_v_L31`
- `qk_L15_to_v_L31`
- `vv_L15_to_v_L31`

Controls are no routing, wrong-layer routing, and spatial-roll routing. The
all-eight shards are fixed as `can,fabric`, `fruit_jelly,rice`,
`vial,wallplugs`, and `walnuts,sheet_metal`.

## GPU Allocation and Sequential Gate

Only GPUs verified idle immediately before submission may be used. The frozen
Stage-A assignment is:

- local host GPU 0: `can`
- dsba3 host GPU 2: `fabric`
- dsba3 host GPU 3: `rice`
- dsba5 host GPU 3 / container CUDA slot 0: `walnuts`

Stages are sequential: all Stage-A shards finish and aggregate before any
Stage-B submission; all Stage-B shards finish and aggregate before Stage C.
Within a stage the four assigned GPUs run concurrently. A newly occupied GPU
is skipped rather than preempting another process.

## Result and Cleanup Contract

Every shard must record host GPU ID, container CUDA slot, source snapshot hash,
support SHA, object list, full test count, exact facet/layer contract, runtime
versions, and access ledger. Aggregation must be validity-first. After a
verdict, remove every regenerable `anomaly_maps/` tree remotely and locally;
preserve compact JSON/TSV metrics, logs, hashes, and the report.
