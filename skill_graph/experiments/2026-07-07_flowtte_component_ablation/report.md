# FlowTTE Component Ablation

Date: 2026-07-07
Verdict: KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC

## Negative Evidence Intake

This is not a method-promotion run. The branch stays in the diagnostic lane
because prior FlowTTE/Flow-LatentBank results remain below reported SuperADD
and because this component ablation uses the DINOv3 coreset support set, not
the exact recorded SuperAD-16 reference image set.

The failure basin being tested is whether previous gains are actually coming
from a meaningful FlowTTE component or from a near-identity kNN/feature-quality
effect. The main risk is over-crediting Normalizing Flow or register tokens
without a matched component control.

## Motivation

The immediate question was: after the register/CLS analysis runs, which pieces
of the current FlowTTE/Flow-LatentBank pipeline are responsible for the useful
metric movement?

The tested components were:

- DINOv3 patch latent distance with static memory.
- learned NF latent projection.
- NF density penalty and NLL-only scoring.
- TTE memory expansion.
- CLS/register context retrieval penalties.
- register-conditioned NF and hybrid structural register usage.

## Implementable Design

Target dataset: MVTec AD2 single-image.

Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`.

Objects: all eight public TESTpublic objects:
`can,fabric,fruit_jelly,rice,vial,wallplugs,walnuts,sheet_metal`.

Shared candidate config:

- `dinov3_vitl16`, layers `[5,11,17,23]`, `feature_fusion=layer_norm_mean`
- 16 train/good support images per object
- support policy:
  `fixed_json=/workspace/fsad_tta/skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json`
- full `test_public/good,bad`
- `flow_epochs=3`, `coupling_layers=2`, `hidden_multiplier=1`
- `flow_lr=2e-4`, `flow_clamp=1.9`, `lambda_logdet=1e-3`
- `density_quantile=0.90`, `distance_weight=1.0`, `top_percent=0.01`
- primary metrics: `seg_AUROC_0.05`, `seg_F1`

Internal baseline:

- `flow_noctx_base`: learned NF latent distance + NF density penalty,
  static support memory, no TTE.

Component variants:

- `identity_no_nf_dist`: no learned NF transform, distance only.
- `identity_density`: no learned NF transform, pseudo density penalty.
- `flow_dist_only`: learned NF latent distance only.
- `nf_nll_only`: NF NLL score only.
- `flow_tte_budget125`: TTE expansion with `expansion_budget=1.25`.
- `cls_soft_w5`, `cls_soft_w10`: CLS context soft distance penalty.
- `register_soft_w5`: register context soft distance penalty.
- `clsreg_soft_w5`: CLS+register context soft distance penalty.
- `cls_topm4`: CLS top-M memory selection.
- `register_condnf`: register-conditioned NF only.
- `hybrid_regcond_cls_topm4`: register-conditioned NF + CLS top-M.

## Evaluation Alignment

All component variants share the same support image paths as
`flow_noctx_base`; manifest support checks passed for all 12 variants.

Baseline context:

- recorded SuperAD-16:
  `/home/hun/Volume/RESEARCH/FMAD-OOD/configs/baselines/recorded_superad16_mvtec_ad2_8object_metrics.json`
- reported SuperADD TESTpublic Table 1 context:
  mean `AUROC_0.05=0.839300`, mean `F1=0.626113`

The SuperAD/SuperADD numbers are context comparators here, not strict promotion
comparators, because the component ablation uses the DINOv3 coreset reference
set rather than the exact recorded SuperAD-16 reference set, and SuperADD is a
reported-table comparator rather than a same-run artifact.

## Code Modification / Creation

Created or updated:

- `src/flow_tte/config.py`: added `flow_transform_mode`.
- `src/flow_tte/trainer.py`: added identity transform mode for no-NF control.
- `src/flow_tte/pipeline.py`: split flow-conditioning contexts from memory
  retrieval contexts.
- `scripts/flow_tte_mvtec_ad2_core.py`: separate flow and memory context
  extraction.
- `scripts/run_flow_tte_mvtec_ad2.py`: CLI flags for transform/context source
  split; manifest now records dynamic all-eight vs reduced claim scope for
  future runs.
- `scripts/flow_tte_morphology_audit.py`: map morphology audit.
- `scripts/run_flow_tte_component_ablation_remote.sh`: first all-8 ablation
  launcher.
- `scripts/run_flow_tte_component_ablation_followup_remote.sh`: follow-up
  component launcher.
- `tests/test_flow_tte.py`, `tests/test_flow_tte_context.py`: transform/context
  regression coverage.

Note: completed run manifests were generated before the dynamic
`claim_scope` label patch, so some already-pulled manifests still say
`reduced-object few-shot diagnostic only`. Their `objects` and diagnostics
confirm all eight objects.

## Added Code Evaluation

Local checks:

- `uv run ruff check src/flow_tte/config.py src/flow_tte/trainer.py src/flow_tte/pipeline.py scripts/flow_tte_mvtec_ad2_core.py scripts/run_flow_tte_mvtec_ad2.py tests/test_flow_tte.py tests/test_flow_tte_context.py scripts/flow_tte_morphology_audit.py`: passed.
- `pytest tests/test_flow_tte.py tests/test_flow_tte_context.py -q`: 18 passed.
- `uv run basedpyright src tests`: passed.
- `bash -n scripts/run_flow_tte_component_ablation_remote.sh`: passed.
- `bash -n scripts/run_flow_tte_component_ablation_followup_remote.sh`: passed.
- `python3 scripts/run_flow_tte_mvtec_ad2.py --help`: passed.

Known verification gap:

- `uv run basedpyright scripts/run_flow_tte_mvtec_ad2.py` still reports
  existing script-level argparse/import typing debt. This predates the manifest
  label patch and is not resolved in this experiment report.

## Remote Execution

Remote:

- host: `147.47.39.144`
- container: `hun_fsad_tta_012`
- host GPUs: `0,1,2`
- in-container CUDA slots: `0,1,2`

Remote roots:

- `/home/hunim/Volume/FMAD-OOD-remote/results_remote/flowtte_component_ablation_all8_20260707_v1`
- `/home/hunim/Volume/FMAD-OOD-remote/results_remote/flowtte_component_ablation_followup_all8_20260707_v1`
- `/home/hunim/Volume/FMAD-OOD-remote/results_remote/flowtte_morphology_reduced_20260707_v1`

Local pullbacks:

- `results/remote_runs/dsba3/flowtte_component_ablation_all8_20260707_v1`
- `results/remote_runs/dsba3/flowtte_component_ablation_followup_all8_20260707_v1`
- `results/remote_runs/dsba3/flowtte_morphology_reduced_20260707_v1`

Summary artifacts:

- `component_summary.tsv`
- `component_summary.json`
- `object_deltas.tsv`
- `morphology_reduced_summary.tsv`

## SuperAD Baseline and Unified Metrics

| component | AUROC_0.05 | F1 | delta AUROC vs base | delta F1 vs base | note |
|---|---:|---:|---:|---:|---|
| SuperAD-16 recorded | 0.765802 | 0.385534 | -0.031941 | -0.052266 | context baseline |
| Flow no-context base | 0.797743 | 0.437800 | 0.000000 | 0.000000 | static memory, NF latent + density |
| identity no-NF distance | 0.785304 | 0.424348 | -0.012439 | -0.013452 | no learned NF |
| identity + density | 0.795172 | 0.441508 | -0.002571 | +0.003708 | pseudo-density control |
| flow distance only | 0.781339 | 0.405754 | -0.016404 | -0.032046 | learned NF latent only |
| NF NLL only | 0.682066 | 0.225544 | -0.115677 | -0.212256 | density alone |
| TTE budget 1.25 | 0.769788 | 0.420321 | -0.027955 | -0.017479 | test-time expansion |
| CLS soft w5 | 0.801613 | 0.442286 | +0.003870 | +0.004486 | CLS context penalty |
| CLS soft w10 | 0.805427 | 0.447118 | +0.007684 | +0.009318 | best component |
| register soft w5 | 0.798287 | 0.438193 | +0.000544 | +0.000393 | weak mean gain |
| CLS+register soft w5 | 0.800546 | 0.440939 | +0.002803 | +0.003139 | weaker than CLS-only |
| CLS top-M4 | 0.797035 | 0.433893 | -0.000708 | -0.003907 | hard memory selection |
| register-conditioned NF | 0.796554 | 0.436152 | -0.001189 | -0.001648 | structural register |
| register NF + CLS top-M4 | 0.793241 | 0.426666 | -0.004502 | -0.011134 | hybrid structural |

Best current variant:

- `cls_soft_w10`: mean `AUROC_0.05=0.805427`, mean `F1=0.447118`
- vs recorded SuperAD-16 context: `+0.039625` AUROC, `+0.061584` F1
- vs reported SuperADD context: `-0.033873` AUROC, `-0.178995` F1

## Results and Analysis

Main performance drivers:

1. **Static no-TTE memory is essential.** Reintroducing TTE expansion with
   `expansion_budget=1.25` drops the base by `-0.027955` AUROC and
   `-0.017479` F1. This supports the earlier ranking-collapse interpretation:
   memory expansion absorbs too much test-time normal/anomalous variation and
   weakens patch ranking.

2. **CLS context as a soft retrieval penalty is the clearest positive
   component.** `CLS soft w10` is the best all-8 variant and improves the base
   by `+0.007684` AUROC and `+0.009318` F1. It wins on 5/8 objects by AUROC
   and 6/8 by F1. `CLS soft w5` also improves both metrics, so the signal is
   not a single lucky weight.

3. **NF NLL is not a standalone scoring module in the current form.**
   `nf_nll_only` collapses to `0.682066` AUROC and `0.225544` F1. NF density
   is useful only as a weak correction term paired with patch latent distance,
   not as the main anomaly map.

4. **Learned NF latent projection alone is not the dominant source of the
   gain.** `flow_dist_only` is worse than both the base and the no-NF distance
   control. `identity_density` slightly improves F1 but not AUROC. This means
   the current advantage is best explained as DINOv3 patch distance + density
   calibration + context correction, not as a pure learned NF metric-space win.

5. **Register tokens have weak or negative metric contribution in the tested
   structural forms.** `register_soft_w5` is almost neutral
   (`+0.000544` AUROC, `+0.000393` F1), while `register_condnf`,
   `cls_topm4`, and the hybrid structural variant are below the base.
   `CLS+register` is weaker than CLS alone, suggesting register information
   dilutes the useful CLS context when merged directly.

Morphology diagnostic on reduced objects (`fabric,can,wallplugs,vial`):

| run | bad area | good area | bad IoU | bad recall | bad precision | good components |
|---|---:|---:|---:|---:|---:|---:|
| no context | 0.007205 | 0.002347 | 0.138007 | 0.314742 | 0.234284 | 11.412 |
| CLS w10 | 0.007961 | 0.002489 | 0.136341 | 0.315419 | 0.233365 | 11.990 |
| register-conditioned NF | 0.006869 | 0.002118 | 0.144403 | 0.321188 | 0.251256 | 10.620 |
| register top-M4 | 0.008135 | 0.003083 | 0.128432 | 0.329700 | 0.225523 | 14.156 |

This explains why register is not useless but is not currently a mean-metric
winner: register-conditioned NF slightly reduces good false-positive area and
fragmentation and improves bad precision on the reduced audit, but it does not
translate into stronger all-8 AUROC/F1. Register top-M increases fragmentation
and false-positive area.

## Continuation Assessment

Strict method claim now: no.

Reasons:

- best variant remains below reported SuperADD by `-0.033873` AUROC and
  `-0.178995` F1;
- SuperAD-16 comparison is contextual rather than strict because this ablation
  does not use the exact recorded SuperAD-16 reference image set;
- RN-FMLK/hard-null dominance is not established in this component run.

Small next diagnostic is justified:

- keep `Flow-LatentBank no-TTE + CLS soft context`, with `w=10` as the current
  component choice;
- stop spending GPU on structural register-conditioned NF/top-M unless the goal
  is specifically false-positive morphology rather than mean metric;
- next useful small diagnostic should test SuperADD-aligned preprocessing,
  masking, or post-threshold morphology on top of `CLS soft w10`.

Hard stop condition for this branch:

- if SuperADD-aligned localization components do not close the F1 gap while
  preserving the `CLS soft w10` AUROC gain, this Flow-LatentBank variant should
  remain diagnostic only.

## Conclusion

The major positive component is **CLS-conditioned soft memory distance**.

The current best method instance is:

```text
DINOv3 ViT-L/16 patch features
+ static support latent bank, no TTE
+ latent distance as main score
+ weak density penalty
+ CLS soft context penalty, weight 10
```

Register tokens should not be treated as the main performance driver for the
current pipeline. Their useful role, if any, is secondary morphology/FP control,
not direct all-8 mean AUROC/F1 improvement.

Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.

## Post-Conclusion Storage Cleanup

Dense `anomaly_maps/` cleanup:

- first component ablation remote root: maps count `0`
- follow-up component ablation remote root: maps count `0`
- morphology reduced remote root: `before_tiff_count=2432`,
  `after_tiff_count=0`
- local pullbacks exclude maps or have no `anomaly_maps/` directories

Preserved compact artifacts: metrics, manifests, logs, summary TSV/JSON,
object deltas, morphology audit TSV/JSON, cleanup evidence.
