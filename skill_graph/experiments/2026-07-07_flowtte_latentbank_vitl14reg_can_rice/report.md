# FlowTTE-LatentBank dinov2_vitl14_reg can/rice diagnostic

Date: 2026-07-07
Verdict: KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC

## Negative Evidence Intake

This is not a new method claim. It is a backbone-control diagnostic for the existing FlowTTE-LatentBank branch: freeze a normalizing-flow projection, expand the latent memory bank, and score by latent memory distance.

Known risk remains the same as earlier FlowTTE-LatentBank evidence: passive feature surgery can look positive on a weak subset while F1 or category coverage collapses. The previous full MVTec AD2 16-shot run lost to recorded SuperAD-16 and reported SuperADD, so this reduced run cannot revive the branch as a claim.

The narrow question here is whether replacing `dinov2_vitl14` with `dinov2_vitl14_reg` improves the same reduced 4-shot `can,rice` configuration enough to keep register features as a candidate backbone for later diagnostics.

## Motivation

The user asked to run FlowTTE-LatentBank with `dinov2_vitl14_reg`. The hypothesis is that register tokens may stabilize patch embeddings before NF latent projection, reducing ranking-collapse sensitivity in at least some objects.

Claim: register backbone improves the current FlowTTE-LatentBank reduced run.

Evidence: paired current-code run against `dinov2_vitl14` under the same support paths, split, scoring mode, and hyperparameters.

Boundary: reduced two-object 4-shot diagnostic only. It is not a Table 1-style or deployable AD2 method claim.

Positioning: backbone-selection evidence for FlowTTE-style latent-bank experiments.

## Implementable Design

Target dataset: MVTec AD2 single-image.

Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`.

Objects and split: `can,rice`, full `test_public/good,bad`.

Support policy: first 4 `train/good` images per object, identity transform only.

Candidate method: `FlowTTE-LatentBank`, `score_mode=latent_distance`, `feature_fusion=layer_norm_mean`.

Backbone pair:

| Run | Backbone |
| --- | --- |
| Control | `dinov2_vitl14` |
| Candidate | `dinov2_vitl14_reg` |

Core hyperparameters:

| Parameter | Value |
| --- | --- |
| `flow_epochs` | `3` |
| `coupling_layers` | `2` |
| `hidden_multiplier` | `1` |
| `flow_lr` | `2e-4` |
| `flow_clamp` | `1.9` |
| `tail_weight` | `0.3` |
| `tail_top_k_ratio` | `0.05` |
| `lambda_logdet` | `1e-3` |
| `expansion_budget` | `1.25` |
| `distance_weight` | `1.0` |
| `density_weight` | `0.25` |
| `top_percent` | `0.01` |

Strict gate: unavailable because there is no same-condition SuperAD 4-shot `can,rice` baseline and no RN-FMLK/hard-null branch for this exact pair.

Early continuation gate: register backbone should improve the paired non-register run on mean `seg_AUROC_0.05` and mean `seg_F1` without catastrophic object collapse.

Unified metric schema:

| Field | Value |
| --- | --- |
| `superad_seg_AUROC_0.05` | `BLOCKED_BASELINE` |
| `superad_seg_F1` | `BLOCKED_BASELINE` |
| `method_seg_AUROC_0.05` | candidate mean metric |
| `method_seg_F1` | candidate mean metric |
| `delta_vs_superad` | not comparable |
| `comparable` | `false` |

## Evaluation Alignment

The design directly tests the backbone question by holding the FlowTTE pipeline, support paths, expansion budget, score mode, split, and objects fixed.

SuperAD comparison is not aligned for a strict claim. The available recorded SuperAD artifact is 16-shot DINO CLS coreset over all eight public MVTec AD2 objects. This diagnostic is 4-shot first-support on only `can,rice`.

The result is therefore comparable only to the paired current-code `dinov2_vitl14` run. Recorded SuperAD-16 can/rice values are reported as context only.

## Code Modification / Creation

No code changes were required. The existing `scripts/run_flow_tte_mvtec_ad2.py` CLI already supports `--backbone-model dinov2_vitl14_reg`.

Created artifacts:

- `skill_graph/experiments/2026-07-07_flowtte_latentbank_vitl14reg_can_rice/report.md`
- `skill_graph/experiments/2026-07-07_flowtte_latentbank_vitl14reg_can_rice/summary.json`
- `skill_graph/experiments/2026-07-07_flowtte_latentbank_vitl14reg_can_rice/comparison_rows.tsv`
- `results/remote_runs/dsba3/flow_tte_latentbank_mvtecad2_can_rice_shot4_vitl14reg_20260707_v1/comparison_backbone_pair/summary.json`
- `results/remote_runs/dsba3/flow_tte_latentbank_mvtecad2_can_rice_shot4_vitl14reg_20260707_v1/comparison_backbone_pair/comparison_rows.tsv`

## Added Code Evaluation

No new code was added. Existing runner metadata confirmed:

- `target_dataset=MVTec AD2 single-image`
- `score_mode=latent_distance`
- `feature_fusion=layer_norm_mean`
- `support_policy=first`
- `support_transforms=["identity"]`
- `strict_method_claim_supported=false`
- `strict_table1_claim_comparable=false`

## Remote Execution

Remote host: dsba3 via the existing FMAD remote config.

Container: `hun_fsad_tta`.

Host GPU: `3`; in-container CUDA slot: `0`.

Control remote result:

`/workspace/results_remote/flow_tte_latentbank_mvtecad2_can_rice_shot4_vitl14_current_20260707_v1`

Control local pullback:

`results/remote_runs/dsba3/flow_tte_latentbank_mvtecad2_can_rice_shot4_vitl14_current_20260707_v1`

Candidate remote result:

`/workspace/results_remote/flow_tte_latentbank_mvtecad2_can_rice_shot4_vitl14reg_20260707_v1`

Candidate local pullback:

`results/remote_runs/dsba3/flow_tte_latentbank_mvtecad2_can_rice_shot4_vitl14reg_20260707_v1`

After completion, no `run_flow_tte_mvtec_ad2.py` process remained and GPU memory returned to 1 MiB.

## SuperAD Baseline and Unified Metrics

Same-condition SuperAD baseline: `BLOCKED_BASELINE`.

Reason: no same-condition SuperAD artifact exists for this exact reduced 4-shot first-support `can,rice` run.

Context-only SuperAD source:

`/home/hun/Volume/RESEARCH/FMAD-OOD/configs/baselines/recorded_superad16_mvtec_ad2_8object_metrics.json`

Context-only SuperAD-16 can/rice mean:

| Metric | SuperAD-16 context | Register candidate | Delta |
| --- | ---: | ---: | ---: |
| `seg_AUROC_0.05` | 0.757674 | 0.771840 | +0.014167 |
| `seg_F1` | 0.333710 | 0.333811 | +0.000101 |

This context does not support a claim because reference budget, support selection, and object scope differ.

## Results and Analysis

Paired current-code comparison:

| Object | `dinov2_vitl14` AUROC | `dinov2_vitl14_reg` AUROC | Delta | `dinov2_vitl14` F1 | `dinov2_vitl14_reg` F1 | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| can | 0.630632 | 0.619280 | -0.011352 | 0.003365 | 0.001459 | -0.001906 |
| rice | 0.900884 | 0.924400 | +0.023516 | 0.627806 | 0.666163 | +0.038357 |
| mean | 0.765758 | 0.771840 | +0.006082 | 0.315586 | 0.333811 | +0.018226 |

Interpretation:

Register features improved the mean, mostly by improving `rice`. The `can` result moved in the wrong direction and remains effectively collapsed in F1 for both backbones. This means `dinov2_vitl14_reg` is a better reduced-run backbone candidate, but the evidence is not robust enough to claim a method improvement.

## Continuation Assessment

Strict method claim now: no.

Small next diagnostic justified: yes, but only as a backbone-control diagnostic.

Single-step continuation contract: if FlowTTE-LatentBank is revisited, run a support-matched all-eight AD2 backbone pair with `dinov2_vitl14` vs `dinov2_vitl14_reg` before making register the default. Hard stop if the mean gain does not survive beyond `rice` or if `can`-like F1 collapse remains unchanged.

## Conclusion

Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.

Asset retained: a clean paired-register backbone result showing `dinov2_vitl14_reg` gives a small mean improvement under the reduced `can,rice` 4-shot latent-bank setting.

The result should not be presented as a SuperAD/SuperADD win. It is useful only for deciding whether future FlowTTE diagnostics should include `dinov2_vitl14_reg` as a controlled backbone option.

## Post-Conclusion Storage Cleanup

Remote cleanup evidence:

- `flow_tte_latentbank_mvtecad2_can_rice_shot4_vitl14_current_20260707_v1`: `anomaly_maps_absent`, `cleanup_anomaly_maps=true`
- `flow_tte_latentbank_mvtecad2_can_rice_shot4_vitl14reg_20260707_v1`: `anomaly_maps_absent`, `cleanup_anomaly_maps=true`

Local cleanup evidence:

- no `anomaly_maps/` directories remain under either local pullback root.
