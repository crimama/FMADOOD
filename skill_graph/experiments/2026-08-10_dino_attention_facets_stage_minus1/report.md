# DINOv3 Attention Facets Stage -1

Date: 2026-08-10  
Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`

## Negative Evidence Intake

This assay opened the actual DINOv3-H+ attention projections rather than
retuning the closed hidden-channel, positional-retrieval, DVT, Flow, or
post-score branches. The experiment was validity-first and explicitly guarded
against the known anomaly-absorption failure mode with wrong-layer/spatial
nulls and object-level F1/AU-PRO no-harm gates.

The first immutable submission (`..._v1`) terminated before metric execution
because a provisional `torchvision::nms` compatibility patch was reintroduced
after smoke verification and captured in the snapshot. That submission is
invalid and contributes no measurements. The patch was removed, its absence
was locked by a regression test and a launch-time source invariant, and the
complete result below is from the new immutable run
`dino_attention_facets_stage_a_20260810T100000Z_v2` only.

## Motivation

Determine whether pre/post-RoPE K, Q, or V descriptors from frozen DINOv3-H+
contain nuisance-stable, defect-preserving localization evidence beyond the
dimension-matched post-block hidden-token control.

## Implementable Design

- Fixed support: 16 `train/good` images per object, manifest SHA-256
  `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`.
- Backbone: `facebook/dinov3-vith16plus-pretrain-lvd1689m`, transformers
  `4.56.2`, frozen inference, resolution 672, layers 15 and 31.
- Descriptors: hidden, K before RoPE, K after RoPE, V, Q after RoPE, asymmetric
  Q-query/K-support, and a support-LOO-standardized equal ensemble.
- Hard nulls: wrong-layer K matching and spatially rolled K query maps.
- Scorer: per-layer/patch L2 normalization, fixed concatenation, raw 1-NN,
  native-resolution evaluation. Query data never update memory or parameters.
- Dense maps existed in RAM only until evaluation and were not written to disk.

The extractor reconstructs Q/K/V directly from the audited HF DINOv3 blocks,
applies patch-only RoPE, excludes CLS/register tokens, and leaves the original
model forward path untouched. A real-model parity test measured maximum
absolute forward-output difference `0.0` and no residual hooks.

## Evaluation Alignment

- Dataset: MVTec AD2 single-image `test_public/good,bad`.
- Objects: `can,fabric,rice,walnuts`.
- Images: 600 total (`240` normal, `360` anomalous).
- Primary metrics: macro `seg_AUROC_0.05` and `seg_F1`.
- No-harm metrics: per-object `seg_F1` and `seg_AUPRO_0.05`, floor `-0.03`.
- Ground truth was opened only after every arm's score map had been produced;
  it was used only for evaluation, never selection or calibration.
- All four shard manifests, access ledgers, support hashes, coverage fields,
  snapshot hashes, and compact-artifact checksums passed.

## Code Modification / Creation

- `src/flow_tte/dinov3_attention_facets.py`: audited pre/post-RoPE Q/K/V
  extraction without hooks or model mutation.
- `scripts/run_dino_attention_facets_stage_minus1.py`: fixed-support Stage A
  runner, support-LOO ensemble calibration, RAM-only native maps, and compact
  custody artifacts.
- `scripts/aggregate_dino_attention_facets_stage_minus1.py`: multi-shard
  validity, hard-null, macro, and object no-harm gates.
- `scripts/orchestrate_dino_attention_facets_stage_a.py` and
  `scripts/run_dino_attention_facets_stage_a_orchestrator.sh`: immutable
  snapshot, GPU UUID/idle preflight, launch/status/pull/checksum/cleanup flow.
- Focused tests cover facet extraction, runner contracts, and aggregation.

## Added Code Evaluation

- Focused suite: `16 passed`.
- Ruff on extractor, runner, aggregator, orchestrator, and focused tests:
  passed.
- Python compilation and shell syntax checks: passed.
- Real DINOv3-H+ smoke: layers 15/31, finite Q/K/V tensors, nonzero RoPE
  deltas, original-forward parity max absolute delta `0.0`, zero leaked hooks.
- End-to-end two-image 672-resolution smoke completed all nine arms with fixed
  support-16 before the full run.

## Remote Execution

All GPUs that were idle at the submission boundary were used. The four shards
shared archive SHA-256
`fe5914046173f3808a4537bf4c195e96905b4b5f83505aa008315dc49d9ab5fc`
and runner SHA-256
`b9f038df1a63b3cf982face2202c7fd497c6fa50725786bb1c93ac6f071f9b15`.

| Object | Host / GPU | Container slot | Images | Start UTC | Complete UTC |
|---|---|---:|---:|---|---|
| can | local GPU 0 | 0 | 162 | 10:06:14 | 10:13:53 |
| fabric | dsba3 GPU 2 | 2 | 156 | 10:07:07 | 10:27:41 |
| rice | dsba3 GPU 3 | 3 | 132 | 10:06:22 | 10:24:57 |
| walnuts | dsba5 GPU 3 | 0 | 150 | 10:06:19 | 10:22:42 |

Every shard was checksum-verified after pull. The source snapshots were then
removed from the execution containers; compact remote/local metrics and logs
were retained.

## SuperAD Baseline and Unified Metrics

This is a DINOv3-H+/16 mechanism diagnostic, not a paper-aligned DINOv2-L/14
SuperAD comparison. SuperAD and RN-FMLK fields are therefore `N/A`, strict
comparability is false, and no benchmark-superiority claim is made.

## Results and Analysis

All validity gates passed, but no real facet passed the frozen continuation
gate.

| Arm | Macro seg-AUROC@0.05 | Macro F1 | Macro AU-PRO@0.05 | Delta AUROC vs hidden | Delta F1 vs hidden |
|---|---:|---:|---:|---:|---:|
| hidden | 0.809196 | 0.500209 | 0.376642 | — | — |
| V | 0.775897 | 0.440216 | 0.381428 | -0.033298 | -0.059992 |
| K pre-RoPE | 0.760275 | 0.330466 | 0.396284 | -0.048921 | -0.169742 |
| equal ensemble | 0.750195 | 0.349977 | 0.368545 | -0.059000 | -0.150232 |
| Q | 0.737624 | 0.320942 | 0.311882 | -0.071571 | -0.179267 |
| K post-RoPE | 0.708282 | 0.252939 | 0.324033 | -0.100914 | -0.247270 |
| Q-query/K-support | 0.530857 | 0.039717 | 0.072263 | -0.278339 | -0.460492 |

`V` was the best real arm by the frozen macro ordering and clearly beat both
hard nulls, but it did not beat hidden and violated object no-harm:

| Object | Hidden AUROC / F1 / AU-PRO | V AUROC / F1 / AU-PRO | V delta F1 | V delta AU-PRO |
|---|---|---|---:|---:|
| can | 0.512141 / 0.000362 / 0.037925 | 0.536502 / 0.000407 / 0.053338 | +0.000045 | +0.015413 |
| fabric | 0.930949 / 0.627395 / 0.446625 | 0.787282 / 0.397427 / 0.390665 | -0.229968 | -0.055960 |
| rice | 0.943826 / 0.701070 / 0.673004 | 0.918594 / 0.689076 / 0.611972 | -0.011994 | -0.061032 |
| walnuts | 0.849868 / 0.672008 / 0.349012 | 0.861212 / 0.673956 / 0.469736 | +0.001948 | +0.120725 |

The mechanism is object-dependent: V helps `can` and `walnuts`, but absorbs
useful defect evidence on `fabric` and `rice`. Direct K/Q facets are even less
stable, and support-only standardization plus equal ensembling does not repair
the heterogeneity. The hard nulls validate that the facet maps contain spatial
signal, but spatial signal alone is not nuisance-invariant, defect-preserving
evidence.

Canonical aggregate:
`results/stage_a_dino_attention_facets/aggregate/dino_attention_facets_stage_a_20260810T100000Z_v2/summary.json`
(SHA-256 `d3982180c55effdf9b67019cbe80ba90f7065a5ca9e2765eb893d1387f7b5656`).

## Continuation Assessment

Stage A is valid and negative. `passing_facets=[]`; therefore Stage B
one-step self-self propagation and Stage C mid-to-late routing remain locked
and were not executed. Starting them would violate the preregistered
sequential gate.

This closes direct, global use of the tested Q/K/V facet descriptors and their
equal ensemble as a nuisance-invariant replacement for hidden features. It
does not show that attention facets are meaningless: their object-specific
gains can remain diagnostic, but support-only evidence in this protocol does
not identify when they are safe.

## Conclusion

`KILL_FOR_CLAIM / NO_CONTINUE`.

Actual DINOv3 attention facets are not a generally safer feature space than
post-block hidden tokens under fixed support-16. The strongest facet, V,
exhibits the same central failure already seen elsewhere in the project:
normal-side improvement on some objects does not predict defect preservation
on another object. No DINO V-V/QQ/KK propagation experiment is licensed by
this result.

## Post-Conclusion Storage Cleanup

The valid v2 execution wrote no dense maps. All four v2 immutable source
snapshots were removed only after checksum-verified pull. Compact JSON/TSV
metrics, access ledgers, hashes, and logs were retained locally; the aggregate
contains no private-test artifact.
