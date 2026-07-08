# FlowTTE Structural Register Diagnostics

Date: 2026-07-07
Verdict: KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC

## Negative Evidence Intake

This is not a repeat of the previous register soft-penalty sweep. The prior
branch added `w * context_distance` directly to patch retrieval distance. This
run tests two structural uses of DINOv3 register tokens:

- `register_topm4`: register context selects the top-4 support image context
  groups before patch latent nearest-neighbor retrieval.
- `register_condnf`: register context conditions the normalizing flow itself,
  while patch latent distance remains the localization score.

The likely failure basin is that register-only context is not sufficiently
anomaly-aligned for localization retrieval. It may describe global nuisance
variation but still be weaker than CLS for selecting useful references.

## Motivation

The goal was to test whether register tokens should be used structurally rather
than as an additive score penalty. The motivating claim was:

> Register is not localization evidence; it should route or condition
> patch-level normality modeling.

## Implementable Design

Target dataset: MVTec AD2 single-image.
Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`.
Objects: all eight public objects:
`can,fabric,fruit_jelly,rice,vial,wallplugs,walnuts,sheet_metal`.
Split: full `test_public/good,bad`.

Shared setting:

- Backbone: `dinov3_vitl16`
- Feature layers: `[5, 11, 17, 23]`
- Feature fusion: `layer_norm_mean`
- Support: 16-shot `dinov3_cls_greedy_coreset`, seed 0
- TTE: disabled by `expansion_budget=1.0`
- Flow: 2 coupling layers, 3 epochs, hidden multiplier 1
- Score: `latent_distance`, `distance_weight=1.0`, `density_weight=0.25`
- Primary metrics: `seg_AUROC_0.05`, `seg_F1`

Variants:

| Method | Register use | Key setting |
|---|---|---|
| `register_topm4` | support image context-group routing | `context_mode=top_m`, `context_top_m=4` |
| `register_condnf` | conditional NF | `flow_condition_mode=context`, `context_mode=none` |

## Evaluation Alignment

The strict SuperAD-16 paper-aligned claim gate is not satisfied because these
runs use DINOv3 and DINOv3 CLS coreset support selection, not the DINOv2
SuperAD-16 reference policy. The directly comparable diagnostic baseline is the
existing non-fixed DINOv3 no-context Flow-LatentBank run.

Reference comparison context:

- No-context DINOv3 diagnostic: `0.797743` AUROC / `0.437800` F1
- CLS w10 diagnostic: `0.805427` AUROC / `0.447118` F1
- Recorded SuperAD-16 context: `0.765802` AUROC / `0.385534` F1
- Reported SuperADD context: `0.839300` AUROC / `0.626113` F1

Only the no-context and CLS w10 rows are same implementation-family diagnostics.
SuperAD/SuperADD numbers are context, not strict gate wins.

## Code Modification / Creation

Modified:

- `src/flow_tte/config.py`
- `src/flow_tte/context_query.py`
- `src/flow_tte/flow.py`
- `src/flow_tte/trainer.py`
- `src/flow_tte/memory.py`
- `src/flow_tte/scoring.py`
- `src/flow_tte/pipeline.py`
- `scripts/flow_tte_mvtec_ad2_core.py`
- `scripts/run_flow_tte_mvtec_ad2.py`
- `tests/test_flow_tte.py`
- `tests/test_flow_tte_context.py`

Added:

- `scripts/run_flow_tte_structural_context_remote.sh`

## Added Code Evaluation

Local checks:

- `pytest tests/test_flow_tte.py tests/test_mvtec_classic_adapter.py -q`: 27 passed
- `uv run ruff check ...`: all checks passed
- `uv run basedpyright src/flow_tte/... tests/test_flow_tte.py`: 0 errors
- `python3 -m py_compile ...`: passed
- `python3 scripts/run_flow_tte_mvtec_ad2.py --help`: new CLI flags visible

Remote checks in `hun_fsad_tta_012`:

- `python3 -m py_compile ...`: passed
- CLI help exposed `--context-mode`, `--context-top-m`,
  `--flow-condition-mode`

Remote pytest was not used because the container does not expose a `pytest`
command in PATH.

## Remote Execution

Container: `hun_fsad_tta_012`.
Host GPUs: `0,1,2`; in-container CUDA slots: `0,1,2`.

Run roots:

- `results/remote_runs/dsba3/flow_latentbank_mvtecad2_all8_shot16_dinov3vitl16_register_topm4_coreset_notte_dw025_20260707_v1`
- `results/remote_runs/dsba3/flow_latentbank_mvtecad2_all8_shot16_dinov3vitl16_register_condnf_coreset_notte_dw025_20260707_v1`

Each run used three chunks:

- GPU0: `can,fabric,fruit_jelly`
- GPU1: `rice,vial,wallplugs`
- GPU2: `walnuts,sheet_metal`

After completion, no `run_flow_tte_mvtec_ad2.py` process remained and GPUs
0/1/2 were idle.

## SuperAD Baseline and Unified Metrics

Diagnostic table:

| Method | n | `seg_AUROC_0.05` | `seg_F1` | Delta AUROC vs no-context | Delta F1 vs no-context |
|---|---:|---:|---:|---:|---:|
| no-context DINOv3 | 8 | 0.797743 | 0.437800 | 0.000000 | 0.000000 |
| CLS w10 diagnostic | 8 | 0.805427 | 0.447118 | +0.007684 | +0.009318 |
| register top-M=4 | 8 | 0.798846 | 0.434411 | +0.001103 | -0.003389 |
| register-conditioned NF | 8 | 0.796554 | 0.436152 | -0.001189 | -0.001648 |

Raw aggregate artifacts:

- `skill_graph/experiments/2026-07-07_flowtte_structural_register/summary.tsv`
- `skill_graph/experiments/2026-07-07_flowtte_structural_register/summary.json`

## Results and Analysis

`register_topm4` gives a tiny mean AUROC gain but loses F1. The gain is mostly
from `fabric` (`+0.030842` AUROC, `+0.007261` F1), while `vial`,
`wallplugs`, and `can` degrade. This suggests register-only support routing is
too coarse or not reliably aligned with local defect evidence.

`register_condnf` is more balanced per-object but still below the no-context
mean. It improves `can`, `fruit_jelly`, `sheet_metal`, and `wallplugs` F1, but
the `fabric` drop dominates the mean. This says conditional NF is plausible as
a mechanism, but register-only context is not enough in the current form.

The strongest related diagnostic remains `CLS w10`, not a register-only
variant. That is consistent with the previous observation that CLS carries the
more useful global retrieval signal for this current DINOv3 setup.

## Continuation Assessment

Strict method claim now: no.

Small next diagnostic justified: yes, but not as a broad register sweep. The
bounded continuation is to test the same structural mechanisms with the already
positive global context source:

- `CLS top-M` or `CLS-conditioned NF`, one setting only
- same 16-shot DINOv3 coreset/no-TTE setup
- hard stop if it fails to beat the `CLS w10` diagnostic on both AUROC and F1

Do not continue with register-only top-M weights or register-only conditional
NF variants unless a separate diagnostic shows register context is better than
CLS for support routing.

## Conclusion

Register-only structural usage does not currently improve Flow-LatentBank
enough to support a method claim. It is cleaner than additive register penalty
conceptually, but empirically weaker than CLS-based context. The useful asset is
the implemented structural context path: top-M context-group memory routing and
conditional NF are now available for controlled follow-up experiments.

## Post-Conclusion Storage Cleanup

Both remote and local result roots were checked after pullback. No
`anomaly_maps/` directories remain. Each chunk preserves
`cleanup_evidence.txt` with `cleanup_anomaly_maps=true`.
