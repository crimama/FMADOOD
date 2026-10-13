# `can` Collapse Attribution Preregistration

Date: 2026-08-09

Protocol ID: `can-collapse-attribution-all8-20260809`

Tier: mechanism attribution diagnostic; not a method or SuperAD claim

## Negative Evidence Intake

Eleven consecutive module-level branches were closed because they improved some
objects and lost others under an all-eight macro gate. The official private
anchor and the per-object public decomposition now show why that framing was
wrong: the macro deficit is not diffuse.

Baseline1-core public K1 per object:

| object | pAUROC | F1 | AU-PRO | AP | comp. recall | normal FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| can | 0.5734 | 0.0010 | 0.1158 | 0.0002 | 0.135 | 0.00562 |
| fruit_jelly | 0.7858 | 0.4870 | 0.4000 | 0.4608 | 0.447 | 0.00112 |
| fabric | 0.9641 | 0.6778 | 0.4785 | 0.6285 | 0.391 | 0.00177 |
| sheet_metal | 0.8917 | 0.5317 | 0.4841 | 0.5105 | 0.360 | 0.00001 |
| wallplugs | 0.9102 | 0.6584 | 0.5250 | 0.6564 | 0.400 | 0.00044 |
| walnuts | 0.8911 | 0.7313 | 0.5747 | 0.7381 | 0.538 | 0.00035 |
| rice | 0.9466 | 0.7140 | 0.6839 | 0.7331 | 0.600 | 0.00001 |
| vial | 0.7536 | 0.4403 | 0.8305 | 0.4680 | 1.000 | 0.00585 |

Macro is a plain eight-object mean. Removing `can` moves macro pAUROC
`0.8396 -> 0.8776`, F1 `0.5302 -> 0.6058`, AU-PRO `0.5116 -> 0.5681`. The
public SuperADD F1 gap would fall from `-0.0959` to `-0.0203` and the public
pAUROC position would move from `+0.0003` to `+0.0383`. One object carries
`79%` of the headline F1 gap.

`can` pAUROC `0.5734` and AP `0.00022` are near chance. This is not a
threshold, region-coherence, or scorer symptom, so none of the closed branches
address it.

## Relationship to the Weak-Object Rule

The 2026-07-08 rule states that weak objects are failure buckets and no-harm
checks, not tuning targets. That rule is retained. This protocol performs
**attribution**, not tuning: every arm is defined class-agnostically, applied
identically to all eight objects, and `can` is one row among eight. No
per-object parameter, threshold, layer, backbone, or support policy may be
selected from this experiment's output. A per-object conclusion is permitted
only as a diagnostic statement about where signal is lost.

## Research Question

Why does `can` collapse to chance while seven objects do not? Exactly one of
three mutually exclusive attributions must be selected, or the protocol
returns `INCONCLUSIVE`.

| Hypothesis | Statement | Downstream implication if selected |
| --- | --- | --- |
| H-A support insufficiency | 16 normal references do not cover `can` normal variation | support efficiency / normal coverage is the research target |
| H-B representation deficiency | DINOv3-H+ `[7,15,23,31]` does not encode `can` defects | backbone / layer / resolution is the research target |
| H-C pipeline signal loss | separable signal exists but the deployed path destroys it | the DVT/Flow/1-NN path is the research target |

## Dataset and Frozen Baseline Contract

- `target_dataset=MVTec AD2 single-image`
- local data root: `/home/hun/Volume/DATA/mvtec_ad_2`
- remote data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- objects: `can,fabric,fruit_jelly,rice,sheet_metal,vial,wallplugs,walnuts`
- population: full `test_public/good,bad`
- backbone: frozen DINOv3-H+ `[7,15,23,31]`, `layer_norm_mean`, resolution 672
  except `sheet_metal=448`
- Flow/scorer: existing Baseline1 Phase-3 contract, 3 epochs, density weight 0,
  `raw_1nn`
- forbidden: query-time update, memory expansion, condition/filename/split
  metadata, `test_private`, `test_private_mixed`, and any private-derived value
- SuperAD/SuperADD/RN-FMLK fields are `N/A`; `comparable=false`; no `KEEP`
  verdict is reachable in this stage.

## Block A1 — Support Ladder (tests H-A)

Nested class-agnostic normal-reference ladder drawn from `train/good` only:

- `S16`: the exact frozen DINOv3-H+ Baseline1-core support-16 manifest,
  unchanged anchor. This is the same manifest used by the official private
  submission and by the `raw`/`position_current`/`position_lowpass` anchors, so
  the ladder base is directly comparable to every DINOv3-H+ result cited above.
- `S64`: `S16` plus 48 images, seed 0, deterministic sorted-name sampling
- `S128`: `S64` plus 64 images, seed 0, same rule

`S16 ⊂ S64 ⊂ S128` is a hard requirement. `128 <= min_object(train/good) = 137`
(`sheet_metal`), so the ladder is equal-budget for all eight objects. No object
receives a different ladder. Deployed Baseline1-core pipeline is otherwise
unchanged.

Arms: 3 × 8 objects = 24 runs.

## Block A2 — Per-Layer Representation Probe (tests H-B vs H-C)

Pure representation probe with DVT off, Flow off, density off, raw patch 1-NN
against the S16 support bank. This removes every downstream stage so that a
failure is attributable to the features themselves.

Layer arms: `7`, `15`, `23`, `31`, and `fused[7,15,23,31]` under the same
`layer_norm_mean` rule. Arms: 5 × 8 objects = 40 runs.

## Block A3 — Ordered Stage Path (localizes H-C)

Ordered capture `pre_dvt_feature -> post_dvt_feature -> flow_latent ->
deployed_final` at S16, all eight objects, reusing the existing
`2026-08-08_dvt_small_defect_signal_path_attribution` machinery and artifacts
where custody verification passes. Any stage/object cell whose custody check
fails is rerun rather than reused. Primary readout is per-object pixel AUROC
and AP at each stage.

## Primary Metrics

- Per-object pooled pixel AUROC (full range) and pixel AP on
  `test_public/good,bad`.
- Secondary: `seg_AUROC_0.05`, `seg_F1` (raw oracle), `seg_AUPRO_0.05`,
  `gt_component_recall_at_oracle`, `normal_image_pooled_fpr_at_oracle`,
  `good_pixel_p99_9`.
- All metrics use the existing float16 tie-inclusive original-resolution
  evaluator. Public labels are used for diagnosis only; no threshold, layer, or
  support policy is selected from them.

## Validity Gates

All must pass before any hypothesis is read:

1. all eight objects complete in every arm; full map/GT coverage; finite,
   nonconstant score maps; no retained `anomaly_maps/`;
2. exact support manifest custody: `S16` basenames equal the frozen DINOv3-H+
   Baseline1-core manifest
   `skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json`
   with SHA-256
   `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`, and
   `S16 ⊂ S64 ⊂ S128` verified by basename set inclusion per object;
3. `S64`/`S128` membership drawn only from `train/good`; zero intersection with
   any `test_public`, `validation`, or private path;
4. identical preprocessing, resolution policy, evaluator, and metric contract
   across all arms of a block;
5. A2 arms confirm DVT off, Flow identity, density weight `0`, and scorer
   `raw_1nn`, asserted from the emitted run config, not from the launch command;
6. no query update, no metadata/condition input, no private access;
7. duplicated local/dsba5 arm parity: maximum absolute difference of shared
   continuous metrics `<= 1e-3`.

## Frozen Attribution Rule

Let `best_layer_auroc(o)` be the maximum Block A2 pixel AUROC over the five
layer arms for object `o`, and let `deployed_auroc(o)` be the S16 deployed
pixel AUROC.

Evaluate in this fixed order and stop at the first match:

1. **H-B (representation deficiency)** if
   `best_layer_auroc(can) < 0.60` **and** at least `6/7` control objects have
   `best_layer_auroc >= 0.75`.
2. **H-C (pipeline signal loss)** if
   `best_layer_auroc(can) - deployed_auroc(can) >= 0.10` **and**
   `best_layer_auroc(can) >= 0.70`.
3. **H-A (support insufficiency)** if `can` S128 pixel AUROC improves over S16
   by `>= 0.10` **and** that improvement is at least `2x` the median S128-minus-S16
   improvement of the seven control objects.
4. Otherwise `INCONCLUSIVE`.

Rules 1 and 3 can both be inspected, but the verdict is whichever fires first
in the stated order. This ordering is fixed before execution and may not be
changed after seeing results.

## Anti-Overfitting and Stop Conditions

- No parameter, layer, resolution, sigma, k, threshold, or support size may be
  swept beyond the arms listed above.
- A2 layer results may not be used to change the deployed layer set. Selecting
  a layer from this diagnostic is forbidden; only the attribution verdict
  transfers.
- If the A1 ladder is non-monotone for `can` (S64 or S128 below S16 by more
  than `0.02` AUROC), record it and do not retune; treat it as evidence
  against H-A.
- If validity gates fail, the protocol reports `INVALID` and no hypothesis is
  read.

## Execution and Cleanup

- Terra implements the ladder sampler, the layer-probe arm wiring, the compact
  per-object aggregator, and focused regression tests. No new scoring
  mathematics is introduced; all arms route through existing runners.
- Block A1 runs on the local RTX 4090 container `hun_fsad_tta_local_stage1a0`.
- Block A2 runs on dsba5 container `hun_fsad_tta_012`, host GPU 3, container
  CUDA slot 0, via an immutable snapshot.
- Block A3 reuses existing artifacts where custody passes; reruns otherwise.
- One arm is deliberately duplicated across local and dsba5 for parity.
- Preserve compact JSON/log/summary/parity artifacts only; verify absence of
  `anomaly_maps/` after the conclusion.

## Reporting

The report must state the verdict, the three hypothesis readouts with their
numbers, the per-object table, the macro-without-`can` counterfactual, and an
explicit statement of what the verdict does and does not license. No method
claim, no SuperAD comparison, and no per-object configuration may be published
from this protocol.
