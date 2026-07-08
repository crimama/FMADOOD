# Flow-LatentBank no-TTE all8 diagnostic

Date: 2026-07-07
Verdict: PROMISING_DIAGNOSTIC

## Negative Evidence Intake

This experiment isolates one failure basin of FlowTTE-LatentBank: test-time memory expansion can absorb low-density or ambiguous test patches, increasing normality volume while decreasing useful ranking density.

The candidate is not a new scoring mechanism. It is a direct ablation that removes the TTE reservoir by setting `expansion_budget=1.0`, which makes the reservoir capacity zero. The NF latent projection and latent-memory distance score remain unchanged.

Known limits still apply. Flow-LatentBank is a passive feature/memory method and remains below SuperAD-16 context. This run can support a mechanism diagnostic about TTE harm, not a paper-level method claim.

## Motivation

The question was: what happens if TTE is removed from FlowTTE-LatentBank, leaving a static Flow-LatentBank?

Claim: under the current latent-bank setup, test-time expansion is not helping; static latent memory may preserve anomaly ranking better.

Evidence: same-condition all-eight MVTec AD2 run comparing `expansion_budget=1.0` no-TTE against `expansion_budget=1.25` TTE.

Boundary: 4-shot first-support diagnostic with `dinov2_vitl14_reg`, not SuperAD Table 1 protocol.

Positioning: ablation evidence for whether future FlowTTE work should keep memory expansion at all.

## Implementable Design

Target dataset: MVTec AD2 single-image.

Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`.

Objects: all eight public objects: `can`, `fabric`, `fruit_jelly`, `rice`, `vial`, `wallplugs`, `walnuts`, `sheet_metal`.

Split: full `test_public/good,bad`.

Shared config:

| Parameter | Value |
| --- | --- |
| shots | `4` |
| support policy | `first` |
| support transforms | `identity` |
| backbone | `dinov2_vitl14_reg` |
| feature fusion | `layer_norm_mean` |
| score mode | `latent_distance` |
| flow epochs | `3` |
| coupling layers | `2` |
| hidden multiplier | `1` |
| flow lr | `2e-4` |
| density quantile | `0.90` |
| distance weight | `1.0` |
| density weight | `0.25` |
| top percent | `0.01` |

Conditions:

| Condition | Expansion budget | Interpretation |
| --- | ---: | --- |
| Flow-LatentBank | 1.0 | no TTE, static memory |
| FlowTTE-LatentBank | 1.25 | test-time memory expansion |

Strict claim gate: unavailable because same-condition SuperAD 4-shot first-support baseline and RN-FMLK/hard-null are not present.

Diagnostic continuation gate: no-TTE should outperform same-condition TTE on mean `seg_AUROC_0.05` and mean `seg_F1`, with object wins on a majority of classes.

## Evaluation Alignment

The ablation is aligned for the internal question because dataset, split, object set, support paths, backbone, NF config, score mode, and evaluator are matched between no-TTE and TTE.

SuperAD comparison is context only. The recorded SuperAD artifact is 16-shot DINO CLS coreset, while this run is 4-shot first-support.

## Code Modification / Creation

No code change was required.

Created artifacts:

- `skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_all8_vitl14reg/report.md`
- `skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_all8_vitl14reg/summary.json`
- `skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_all8_vitl14reg/comparison_rows.tsv`
- `results/remote_runs/dsba3/flow_latentbank_mvtecad2_all8_shot4_vitl14reg_notte_20260707_v1/comparison_vs_tte/summary.json`
- `results/remote_runs/dsba3/flow_latentbank_mvtecad2_all8_shot4_vitl14reg_notte_20260707_v1/comparison_vs_tte/comparison_rows.tsv`

## Added Code Evaluation

No new code was added.

Config verification:

- `expansion_budget=1.0` creates reservoir capacity 0.
- no-TTE manifest confirms `initial_memory_size == final_memory_size` for all eight objects.
- TTE manifest confirms memory growth from initial support memory to 1.25x budget for all eight objects.

## Remote Execution

Remote host: dsba3.

Container: `hun_fsad_tta_012`.

Host GPUs: `0,1,2`; in-container CUDA slots: `0,1,2`.

The existing `hun_fsad_tta` container exposed only host GPU `3`, so a separate `hun_fsad_tta_012` container was created from the same `hun_fsad_tta_image:latest` image with the same workspace/data/cache mounts and host GPUs `0,1,2`.

Object chunks:

| CUDA slot | Objects |
| ---: | --- |
| 0 | `can,fabric,fruit_jelly` |
| 1 | `rice,vial,wallplugs` |
| 2 | `walnuts,sheet_metal` |

No-TTE remote result:

`/workspace/results_remote/flow_latentbank_mvtecad2_all8_shot4_vitl14reg_notte_20260707_v1`

No-TTE local pullback:

`results/remote_runs/dsba3/flow_latentbank_mvtecad2_all8_shot4_vitl14reg_notte_20260707_v1`

TTE baseline remote result:

`/workspace/results_remote/flowtte_latentbank_mvtecad2_all8_shot4_vitl14reg_tte_20260707_v1`

TTE baseline local pullback:

`results/remote_runs/dsba3/flowtte_latentbank_mvtecad2_all8_shot4_vitl14reg_tte_20260707_v1`

After completion, no `run_flow_tte_mvtec_ad2.py` process remained and all three exposed GPUs returned to 1 MiB memory use.

## SuperAD Baseline and Unified Metrics

Same-condition SuperAD baseline: `BLOCKED_BASELINE`.

Context-only SuperAD-16 source:

`/home/hun/Volume/RESEARCH/FMAD-OOD/configs/baselines/recorded_superad16_mvtec_ad2_8object_metrics.json`

Context-only comparison:

| Method | Mean AUROC_0.05 | Mean F1 | Delta AUROC vs SuperAD-16 context | Delta F1 vs SuperAD-16 context |
| --- | ---: | ---: | ---: | ---: |
| SuperAD-16 context | 0.765802 | 0.385534 | 0.000000 | 0.000000 |
| Flow-LatentBank no-TTE | 0.733798 | 0.297527 | -0.032004 | -0.088007 |
| FlowTTE-LatentBank TTE | 0.712458 | 0.266227 | -0.053344 | -0.119307 |

The SuperAD rows are not a strict gate because reference budget and support policy differ.

## Results and Analysis

Same-condition no-TTE vs TTE:

| Object | no-TTE AUROC | TTE AUROC | Delta | no-TTE F1 | TTE F1 | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| can | 0.576989 | 0.619259 | -0.042271 | 0.000609 | 0.001459 | -0.000850 |
| fabric | 0.687677 | 0.595938 | +0.091740 | 0.243356 | 0.157578 | +0.085778 |
| fruit_jelly | 0.716786 | 0.616367 | +0.100419 | 0.289808 | 0.163486 | +0.126322 |
| rice | 0.926475 | 0.924423 | +0.002052 | 0.667107 | 0.666288 | +0.000819 |
| vial | 0.687931 | 0.654681 | +0.033250 | 0.337461 | 0.279502 | +0.057959 |
| wallplugs | 0.725154 | 0.809637 | -0.084483 | 0.072587 | 0.177903 | -0.105316 |
| walnuts | 0.829177 | 0.790068 | +0.039109 | 0.488845 | 0.434725 | +0.054120 |
| sheet_metal | 0.720194 | 0.689287 | +0.030907 | 0.280446 | 0.248875 | +0.031571 |
| mean | 0.733798 | 0.712458 | +0.021340 | 0.297527 | 0.266227 | +0.031300 |

No-TTE wins 6/8 objects on AUROC and 6/8 objects on F1. TTE helps `can` and `wallplugs`, but hurts the other six objects enough that the mean drops.

The result supports the interpretation that the current TTE memory expansion is often degrading ranking rather than improving it. Removing TTE preserves the NF latent projection benefit while avoiding expansion-driven anomaly absorption.

## Continuation Assessment

Strict method claim now: no.

Small next diagnostic justified: yes.

Single-step continuation contract: keep Flow-LatentBank as the stronger branch for this latent-memory family and test whether a class-conditional or gated expansion policy can recover `can`/`wallplugs` without degrading the six objects where static memory wins. Hard stop the expansion branch if a gated TTE variant cannot beat static Flow-LatentBank mean AUROC/F1 and cannot improve `can`/`wallplugs` without harming the six no-TTE wins.

## Conclusion

Verdict: `PROMISING_DIAGNOSTIC`.

This run gives clear evidence that, for the current `dinov2_vitl14_reg` latent-bank setting on all eight MVTec AD2 objects, removing TTE improves the internal same-condition metric. The result does not beat recorded SuperAD-16 context and is not a strict method claim, but it changes the direction of the FlowTTE branch: the static Flow-LatentBank baseline is stronger than the current expansion version.

## Post-Conclusion Storage Cleanup

Remote cleanup evidence:

- no `anomaly_maps/` directories remain under either completed remote result root.
- all chunk `cleanup_evidence.txt` files contain `cleanup_anomaly_maps=true`.

Local cleanup evidence:

- no `anomaly_maps/` directories remain under either local pullback root.
