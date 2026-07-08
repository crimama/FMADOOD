# Flow-LatentBank no-TTE shot16 DINOv2/DINOv3 diagnostic

Date: 2026-07-07
Protocol: `fmad-experiment-protocol`
Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`

## 1. Motivation

The immediate question was whether Flow-LatentBank without TTE can be compared to `SuperAD-16` under the same few-shot budget, and whether replacing the backbone with DINOv3 moves the method closer to reported SuperADD.

Claim: static latent-bank scoring may avoid the ranking collapse introduced by test-time memory expansion, while a stronger DINOv3 representation may recover some SuperADD-like localization strength.

Evidence target: all eight public MVTec AD2 objects, full `test_public/good,bad`, `seg_AUROC_0.05` and `seg_F1`.

Boundary: this is still a passive latent-memory diagnostic. DINOv2 shot16 is the closest same-budget/evaluator SuperAD comparison. DINOv3 is a backbone diagnostic and not a strict SuperAD/SuperADD claim.

Negative-evidence intake: earlier FlowTTE 16-shot expansion lost to recorded SuperAD-16 (`0.714929`/`0.303336` vs `0.765802`/`0.385534`). This run tests whether removing TTE and changing representation has a measurable gradient rather than retuning the killed expansion setting.

## 2. Implementable Design

Target dataset: MVTec AD2 single-image.
Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`.
Objects: `can`, `fabric`, `fruit_jelly`, `rice`, `vial`, `wallplugs`, `walnuts`, `sheet_metal`.
Split: full `test_public/good,bad`.
Few-shot budget: 16 train/good support images per object.

Shared Flow-LatentBank config:

| Parameter | Value |
| --- | --- |
| score mode | `latent_distance` |
| support transforms | `identity` |
| feature fusion | `layer_norm_mean` |
| flow epochs | `3` |
| coupling layers | `2` |
| hidden multiplier | `1` |
| flow lr | `2e-4` |
| flow clamp | `1.9` |
| tail weight / top-k | `0.3` / `0.05` |
| lambda logdet | `1e-3` |
| density quantile | `0.90` |
| expansion budget | `1.0` no-TTE |
| distance / density weight | `1.0` / `0.25` |
| top percent | `0.01` |

Conditions:

| Condition | Backbone | Support policy | Purpose |
| --- | --- | --- | --- |
| DINOv2 no-TTE | `dinov2_vitl14` | `dinov2_cls_greedy_coreset` | SuperAD-16 same-shot/budget/evaluator comparison |
| DINOv3 no-TTE | `dinov3_vitl16` | `dinov3_cls_greedy_coreset` | SuperADD-context backbone diagnostic |

Strict gate: beat recorded SuperAD-16 on both mean AUROC and F1 under the DINOv2 matched setting, then still require RN-FMLK/hard-null gates before any paper claim.

Continuation gate: DINOv3 should produce a clear positive gradient over DINOv2 no-TTE and not collapse F1, even if it remains below reported SuperADD.

## 3. Evaluation Alignment

SuperAD-16 source: `/home/hun/Volume/RESEARCH/FMAD-OOD/configs/baselines/recorded_superad16_mvtec_ad2_8object_metrics.json`.

That baseline is recorded from `dinov2_vitl14`, 16-shot DINOv2 CLS greedy coreset, full eight objects, full TESTpublic, and `src.post_eval.eval_segmentation` metrics.

DINOv2 no-TTE alignment: dataset, split, object set, shot count, DINOv2 CLS coreset policy, backbone family, and evaluator are aligned. It is the closest available SuperAD-16 comparison, but `strict_table1_claim_comparable=false` because this pipeline is not SuperAD's full method/postprocess path.

DINOv3 no-TTE alignment: dataset/split/objects/shot count/evaluator are aligned, but support selection and backbone are changed. This is a diagnostic comparison, not a strict SuperAD-16 gate.

SuperADD comparison: reported TESTpublic Table 1 values are context only, not a same-evaluator rerun.

## 4. Code Modification / Creation

Implemented DINOv3 support and support-policy routing:

- `scripts/dinov3_backbone.py`: HuggingFace DINOv3 ViT wrapper with CLS/register token handling and torch pytree compatibility shim.
- `scripts/run_flow_tte_mvtec_ad2.py`: DINOv3 backbone branch and stricter `reference_budget_matched` manifest flag.
- `scripts/flow_tte_support.py`: CLS greedy coreset aliases for `dinov2_*` and `dinov3_*` policies.
- `tests/test_dinov3_backbone.py`: DINOv3 model-id, token-offset, detection, and compatibility tests.
- `tests/test_mvtec_classic_adapter.py`: support-policy alias coverage.

## 5. Added Code Evaluation

Local checks passed:

- `python3 -m pytest tests/test_dinov3_backbone.py tests/test_mvtec_classic_adapter.py -q`: 15 passed.
- `python3 -m py_compile scripts/dinov3_backbone.py scripts/run_flow_tte_mvtec_ad2.py scripts/flow_tte_support.py`: passed.

Remote preflight passed after installing `transformers==4.56.2` and restoring `numpy==1.26.4`. The official `facebook/dinov3-vitl16-pretrain-lvd1689m` checkpoint was gated, so the run used `camenduru/dinov3-vitl16-pretrain-lvd1689m`.

## 6. Remote Experiment Execution

Remote host: dsba3.
Container: `hun_fsad_tta_012`.
Host GPUs: `0,1,2`; in-container CUDA slots: `0,1,2`.

Object chunks:

| CUDA slot | Objects |
| ---: | --- |
| 0 | `can,fabric,fruit_jelly` |
| 1 | `rice,vial,wallplugs` |
| 2 | `walnuts,sheet_metal` |

DINOv2 no-TTE remote root:
`/workspace/results_remote/flow_latentbank_mvtecad2_all8_shot16_vitl14_notte_dw025_20260707_v1`

DINOv2 local pullback:
`results/remote_runs/dsba3/flow_latentbank_mvtecad2_all8_shot16_vitl14_notte_dw025_20260707_v1`

DINOv3 no-TTE remote root:
`/workspace/results_remote/flow_latentbank_mvtecad2_all8_shot16_dinov3vitl16_notte_dw025_20260707_v1`

DINOv3 local pullback:
`results/remote_runs/dsba3/flow_latentbank_mvtecad2_all8_shot16_dinov3vitl16_notte_dw025_20260707_v1`

## 7. Evaluation Results and Analysis

Unified metric schema: `superad_<metric>`, `method_<metric>`, `delta_vs_superad`, `comparable` fields are stored in `summary.json`; object rows are stored in `comparison_rows.tsv`.

| Method / baseline | Backbone | Support | TTE | Mean AUROC_0.05 | Mean F1 | Delta AUROC vs SuperAD-16 | Delta F1 vs SuperAD-16 | Delta AUROC vs SuperADD | Delta F1 vs SuperADD | Comparable note |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SuperAD-16 recorded | dinov2_vitl14 | DINOv2 CLS coreset 16 | n/a | 0.765802 | 0.385534 | 0.000000 | 0.000000 | -0.073498 | -0.240578 | same-evaluator baseline |
| Flow-LatentBank no-TTE | dinov2_vitl14 | DINOv2 CLS coreset 16 | no | 0.761782 | 0.350391 | -0.004020 | -0.035144 | -0.077518 | -0.275722 | closest SuperAD-16 budget/evaluator comparison |
| Flow-LatentBank no-TTE | dinov3_vitl16 | DINOv3 CLS coreset 16 | no | 0.797743 | 0.437800 | +0.031941 | +0.052266 | -0.041557 | -0.188312 | backbone diagnostic; not same support/backbone as SuperAD |
| SuperADD reported | not rerun | paper Table 1 | n/a | 0.839300 | 0.626112 | +0.073498 | +0.240578 | 0.000000 | 0.000000 | reported context, not same-evaluator rerun |


Object-level values:

| Object | SuperAD AUROC/F1 | DINOv2 no-TTE AUROC/F1 | DINOv3 no-TTE AUROC/F1 | SuperADD reported AUROC/F1 |
| --- | ---: | ---: | ---: | ---: |
| can | 0.586950/0.001553 | 0.647997/0.001661 | 0.676314/0.002963 | 0.516100/0.000000 |
| fabric | 0.687853/0.275239 | 0.696520/0.275325 | 0.755118/0.325347 | 0.842000/0.937400 |
| fruit_jelly | 0.797842/0.453155 | 0.741353/0.321203 | 0.813073/0.541089 | 0.811900/0.546800 |
| rice | 0.928397/0.665867 | 0.915980/0.656086 | 0.946106/0.695460 | 0.961100/0.733100 |
| vial | 0.692650/0.368248 | 0.692449/0.361069 | 0.705036/0.394468 | 0.825900/0.647700 |
| wallplugs | 0.775900/0.199382 | 0.756382/0.105793 | 0.852729/0.418772 | 0.925300/0.791600 |
| walnuts | 0.883834/0.718787 | 0.886771/0.735338 | 0.867691/0.696749 | 0.900300/0.756900 |
| sheet_metal | 0.772990/0.402043 | 0.756804/0.346652 | 0.765878/0.427553 | 0.931800/0.595400 |

Key observations:

- DINOv2 no-TTE is nearly tied with SuperAD-16 on mean AUROC (`-0.004020`) but remains lower on mean F1 (`-0.035144`). It wins AUROC on 3/8 objects and F1 on 3/8 objects.
- DINOv3 no-TTE improves over DINOv2 no-TTE by `+0.035961` AUROC and `+0.087409` F1.
- DINOv3 no-TTE is above recorded SuperAD-16 context by `+0.031941` AUROC and `+0.052266` F1, but this is not same-backbone/support comparable.
- DINOv3 no-TTE is still below reported SuperADD by `-0.041557` AUROC and `-0.188312` F1. It wins 2/8 objects on AUROC and 1/8 on F1.
- Compared with the earlier 16-shot TTE expansion run, DINOv2 no-TTE improves by `+0.046853` AUROC and `+0.047054` F1. That supports the no-TTE direction, though density-weight settings differ.

Strict method claim now: no. The closest matched DINOv2 run does not beat SuperAD-16 on both metrics, and RN-FMLK/hard-null gates are absent.

Small next diagnostic justified: yes. The DINOv3 backbone swap gives a substantial positive gradient and moves the method above recorded SuperAD-16 context, while still leaving a clear F1 gap to reported SuperADD.

## 8. Conclusion

Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.

For the user's first comparison target, Flow-LatentBank no-TTE at SuperAD's 16-shot budget is close to SuperAD-16 in AUROC but still loses the stricter claim gate because F1 remains lower. For the second target, DINOv3 improves the Flow-LatentBank branch materially, but it does not close the gap to reported SuperADD and is not a strict same-condition comparison.

Retained asset: DINOv3 support is now runnable in the Flow-LatentBank AD2 pipeline, and the evidence says backbone quality matters more than TTE expansion in this branch.

Single next experiment: run a DINOv3 no-TTE support/control ablation that separates backbone from support selection: same DINOv2 SuperAD-16 reference images with DINOv3 features, versus DINOv3 CLS coreset. Hard-stop if the gain disappears under the same reference set or if F1 remains far below SuperADD.

## 9. Post-Conclusion Storage Cleanup

Both new runs used `--cleanup-maps`; `run_manifest.json` records `cleanup_anomaly_maps=true`.

Local cleanup evidence: no `anomaly_maps/` directories remain under either pulled result root.

Remote cleanup evidence: no `anomaly_maps/` directories remain under either completed remote result root after the dsba3 check.

## Artifacts

- Summary: `skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_shot16_dinov2_dinov3/summary.json`
- Object rows: `skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_shot16_dinov2_dinov3/comparison_rows.tsv`
- DINOv2 result comparison: `results/remote_runs/dsba3/flow_latentbank_mvtecad2_all8_shot16_vitl14_notte_dw025_20260707_v1/comparison_superad16`
- DINOv3 result comparison: `results/remote_runs/dsba3/flow_latentbank_mvtecad2_all8_shot16_dinov3vitl16_notte_dw025_20260707_v1/comparison_superadd_reported`
