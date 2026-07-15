# FlowTTE Session Handoff

Updated: 2026-07-10 12:10 KST

## Post-Handoff Update

This section supersedes the 08:42 KST runtime snapshot retained below for
historical context.

- The phase-3 controller completed at `2026-07-10 11:14:44 KST`.
- The requested follow-up kept the H+ DVT fused MLP method fixed and changed
  only identity support to the exact prior SuperAD eight-rotation orbit.
- The matched identity control scored `0.836389/0.527370`; rotation-8 scored
  `0.819251/0.506201`, a delta of `-0.017137/-0.021168`.
- The augmentation branch is closed as `KILL_FOR_CLAIM / NO_CONTINUE`.
  `sheet_metal` and `fabric` regress severely, and only `fruit_jelly` reaches
  `+0.01` F1 improvement, so the pre-registered transform-aware DVT follow-up
  is not justified.
- Exact OpenCV transform parity, full-batch-equivalent support-row
  microbatching, paired manifest/support-path equality, runtime safety, and
  zero retained anomaly maps were verified.
- The executed runs recorded a matching nine-file v1 method hash. A final
  audit expanded future validation to the actual 48-file split-root closure
  (including `src/utils.py`) and confirmed stale v1 markers are rejected before
  GPU work. The remote evaluator differs from the local checkout, so paired
  deltas are valid but bit-for-bit local reproduction of absolute metrics is a
  documented caveat in the report.
- Remote GPUs `0,1,2` are idle after completion.

Full report:
`skill_graph/experiments/2026-07-10_flowtte_hplus_dvt_superad_rotation8/report.md`.

## 0. Read This First

This document is for a new session taking over the FlowTTE work. The current
active line is **FlowTTE H+ DVT fixed-memory latent bank** on **MVTec AD2
single-image**, all 8 public objects.

Current high-level verdict:

```text
FlowTTE H+ DVT = strong diagnostic baseline
strict method claim vs SuperADD = not supported
current tuning branch = KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC
```

Do not restart old class-specific tuning. All current experiments must stay
class-agnostic and all-object unless explicitly labeled as a reduced diagnostic.

## 1. Default Repo And Remote State

Local repo:

```text
/home/hun/Volume/RESEARCH/FSAD-TTA
```

Remote execution:

```text
host: hunim@147.47.39.144:2222
container: hun_fsad_tta_012
remote repo: /workspace/fsad_tta
remote data: /home/hunim/Volume/DATA/mvtec_ad_2
host GPUs used: 0,1,2
```

Do not use or delete other containers. Dense `anomaly_maps/` should be removed
after result analysis and summary artifacts are recorded.

Current remote status at handoff:

```text
phase-2 hparam controller: running
phase-3 hparam controller: running but waiting for phase-2 completion
phase-2 latest partial leaderboard rows: 32 variants + header
phase-3 leaderboard: not yet created
```

Useful status command, without embedding credentials:

```bash
ssh -p 2222 hunim@147.47.39.144 \
  "docker exec hun_fsad_tta_012 bash -lc 'cat /workspace/results_remote/flowtte_hparam_phase2_20260710_v2_leaderboard.tsv 2>/dev/null || true; cat /workspace/results_remote/flowtte_hparam_phase3_20260710_v3_leaderboard.tsv 2>/dev/null || true; pgrep -af run_flow_tte_hparam_phase || true'"
```

## 2. Current Method Structure

The retained method is:

```text
DINOv3-H+/16 image
-> feature layers [7,15,23,31]
-> layer_norm_mean fusion
-> DVT-lite position_mean feature denoise, alpha=1.0
-> patch-wise MLP normalizing flow latent projection
-> fixed 16-shot support latent memory bank
-> no test-time memory expansion
-> test patch to support latent nearest-neighbor distance
-> weak density score term
-> continuous anomaly map
```

Important config defaults:

```text
backbone_model=dinov3_vith16plus
feature_layers=7,15,23,31
feature_fusion=layer_norm_mean
dvt_denoise_mode=position_mean
dvt_denoise_alpha=1.0
normality_mode=fused
expansion_budget=1.0
score_mode=latent_distance
density_weight=0.25
support=static fixed 16-shot JSON
```

Fixed support JSON:

```text
/workspace/fsad_tta/skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json
```

Current reference performance:

| Method | AUROC_0.05 | F1 |
|---|---:|---:|
| H+ DVT FlowTTE reference | 0.836739 | 0.527427 |
| reported SuperADD context | 0.839300 | 0.626113 |

Interpretation:

- AUROC is close to SuperADD.
- F1 is still far lower by about `0.0987`.
- The main gap is not only ranking, but score-field-to-mask quality:
  fragmentation, threshold sensitivity, boundary imprecision, and
  foreground/background/object-mode confusion.

## 3. Original Problem Framing

The motivating interpretation was:

```text
few-shot TTE memory expansion
-> normality volume expands
-> normal density decreases
-> anomaly / expanded-normal ranking can collapse
```

The method moved toward **no-TTE fixed support memory** to avoid anomaly
absorption. Normalizing Flow was introduced as a learned latent projection and
density model, not as DeCoFlow continual learning. In the current best branch,
the NF is mainly used to project DINO patch features before latent memory
distance; NLL/density is only a weak auxiliary score.

## 4. DVT-Lite Feature Denoise

The implemented DVT-style denoise is not the full DVT paper pipeline. It is a
lightweight support-fitted positional artifact subtraction:

```text
support feature maps: X_j[h,w,c]
position_mean[h,w,c] = mean_j X_j[h,w,c]
global_mean[c] = mean_{j,h,w} X_j[h,w,c]
artifact[h,w,c] = position_mean[h,w,c] - global_mean[c]
denoised X = X - alpha * artifact
```

Code:

```text
src/flow_tte/denoising.py
```

Why it matters:

- It reduces position-dependent support artifacts before NF projection.
- It helped the H+ branch at `alpha=1.0`.
- Larger alpha values are harmful. Stage-1 hparam sweep showed:

| DVT alpha | AUROC_0.05 | F1 |
|---:|---:|---:|
| 0.75 | 0.834495 | 0.522175 |
| 1.00 | 0.836739 | 0.527427 |
| 1.25 | 0.832375 | 0.515905 |
| 1.50 | 0.826311 | 0.498589 |
| 2.00 | 0.814195 | 0.470521 |

Conclusion: `alpha=1.0` should remain the default unless a new structural
reason appears.

## 5. Register Token Analysis

Register-related literature was interpreted as:

```text
patch tokens = local normality / defect evidence / pixel localization
register or CLS = global context / style / nuisance / shift
```

The intended direction was to use DINOv3 register tokens as condition/context,
not as direct localization maps.

Tried directions:

- register/CLS context routing and memory conditioning;
- CLS weighting variants;
- register-conditioned NF ideas;
- Transformer Flow with raw CLS/register prefix tokens.

Most recent Transformer-prefix implementation:

```text
[CLS/register/dummy prefix tokens ; patch tokens]
-> Transformer conditioner
-> scale/shift only for patch tokens
-> patch latent memory and scoring only
```

All-eight results:

| Variant | AUROC_0.05 | F1 |
|---|---:|---:|
| Transformer + CLS prefix | 0.826397 | 0.495254 |
| Transformer + register prefix | 0.826433 | 0.494640 |
| Transformer + CLS+register prefix | 0.826379 | 0.494712 |
| Transformer + random dummy prefix | 0.829330 | 0.503940 |
| Transformer + learned dummy prefix | 0.829324 | 0.503923 |

Conclusion:

- Raw DINOv3 CLS/register tokens did not improve Transformer Flow.
- Dummy prefix tokens did slightly better than real CLS/register prefixes.
- Current evidence does not support using raw register tokens inside the
  Transformer Flow as a direct improvement.
- Keep the conceptual claim that registers are global context tokens, but do
  not claim performance gain from the current register implementation.

## 6. Structural Diagnostics Summary

Reference:

| Method | AUROC_0.05 | F1 |
|---|---:|---:|
| H+ DVT FlowTTE reference | 0.836739 | 0.527427 |
| reported SuperADD context | 0.839300 | 0.626113 |

Completed structural diagnostics:

| Branch | Result | Verdict |
|---|---:|---|
| Morphology on saved H+ maps | F1 0.542316 | helps, but not enough |
| no-NF identity feature distance | 0.832461 / 0.524804 | below reference |
| support-position score calibration | 0.785133 / 0.429653 | harmful |
| support-position z-score | 0.701830 / 0.278144 | harmful |
| foreground feature-energy prior | 0.834598 / 0.526807 | baseline-tied |
| layer-wise Flow-LatentBank | 0.828210 / 0.499110 | below reference |
| layer-wise CLS topM4 | 0.829923 / 0.508863 | below reference |
| structured image top-M memory | 0.832426 / 0.524593 | below reference |
| conditional CLS NF | 0.832374 / 0.512126 | below reference |
| foreground/background flow mixture | 0.834712 / 0.519250 | below reference |
| local contrast score-field | 0.806200 / 0.438345 | harmful |
| Conv2D Flow smoke mean | 0.744949 / 0.354137 | no continue |
| Transformer Flow all-eight | 0.828600 / 0.502237 | diagnostic only |
| Transformer + CLS/register prefixes | about 0.826 / 0.495 | below Transformer baseline |

Main conclusions:

- NF removal does not improve the all-eight mean.
- More spatial or token mixing is not automatically better.
- Foreground/background separation alone is insufficient.
- Local contrast and support-position score calibration are harmful.
- Layer-wise late fusion underperformed fused normalized feature.
- Transformer Flow has positive object-specific signals, but all-eight no-harm
  fails and image analysis shows local anomaly contrast is weakened.
- The method is currently a strong continuous ranker but weak binary mask
  generator.

## 7. Transformer Flow Failure Analysis

Architecture-only comparison:

```text
same DINOv3-H+, same DVT alpha=1.0, same fixed support, density_weight=0.0
baseline: patch-wise MLP flow
candidate: Transformer coupling flow
```

Mean result:

| Method | AUROC_0.05 | F1 |
|---|---:|---:|
| MLP dw0 | 0.837518 | 0.523904 |
| Transformer dw0 | 0.828600 | 0.502237 |

Image-level failure analysis showed:

- Transformer Flow reduced GT-vs-background score gap across every class.
- Positive cases such as `fabric`, `vial`, and `sheet_metal` looked more like
  score-field smoothing than stronger anomaly ranking.
- `fruit_jelly`, `walnuts`, `wallplugs`, and `rice` drove all-eight failure.
- `can` remained collapsed in both MLP and Transformer settings.

Conclusion:

```text
unconstrained token interaction before memory distance
-> smoother / more globally consistent field
-> weaker local defect contrast
```

Do not continue plain Transformer Flow hyperparameter tuning. If revisited,
it needs a constrained or residual/gated mechanism.

## 8. SuperADD Gap Interpretation

SuperADD is structurally similar at the high level:

```text
DINOv3 patch feature + memory bank + NN distance
```

But it differs in important settings:

- high-resolution/tiled execution;
- layer-wise raw feature distance and scale normalization;
- coreset memory;
- held-out threshold calibration;
- binary morphology/postprocessing;
- possibly stronger practical object/mask operating-point calibration.

Important interpretation:

- Our AUROC nearly matches reported SuperADD context, so the continuous ranker
  is not completely broken.
- The F1 gap is mostly score-field/mask formation, not just backbone capacity.
- Morphology lifted F1 by about `+0.0149`, but this still leaves most of the
  SuperADD F1 gap.
- Therefore, structural novelty cannot be claimed from postprocessing alone,
  but future work must account for binary mask formation if comparing to
  SuperADD F1.

## 9. Hyperparameter Tuning Summary

The latest tuning intentionally targets the retained best structure, not
Transformer Flow.

Constraints:

```text
all 8 MVTec AD2 public objects
fixed 16-shot support JSON
no class-specific tuning
same DINOv3-H+ [7,15,23,31]
DVT position_mean alpha=1.0 unless swept
cleanup anomaly_maps after completed result analysis
```

Stage-1 remote run:

```text
remote: /workspace/results_remote/flowtte_hparam_extreme_20260710_v1
local:  results/remote_runs/dsba3/flowtte_hparam_extreme_20260710_v1
script: scripts/run_flow_tte_hparam_extreme_remote.sh
```

Stage-1 best by F1:

| Variant | AUROC_0.05 | F1 | Delta F1 vs ref |
|---|---:|---:|---:|
| lambda_logdet=1e-2 | 0.836202 | 0.528875 | +0.001448 |

Stage-1 best by AUROC:

| Variant | AUROC_0.05 | F1 | Delta AUROC vs ref |
|---|---:|---:|---:|
| brightness 0.80,1.20 | 0.838230 | 0.528203 | +0.001491 |

Stage-1 harmful axes:

- DVT alpha above 1.0;
- more coupling layers;
- higher flow learning rate;
- longer flow training;
- high affine clamp;
- density weight sweep alone.

Phase-2 remote run:

```text
remote: /workspace/results_remote/flowtte_hparam_phase2_20260710_v2
local partial: results/remote_runs/dsba3/flowtte_hparam_phase2_20260710_v2_partial
script: scripts/run_flow_tte_hparam_phase2_remote.sh
status at handoff: running
```

Phase-2 latest partial top F1 rows:

| Variant | AUROC_0.05 | F1 |
|---|---:|---:|
| lambda_logdet=2e-2 | 0.835673 | 0.529874 |
| lambda_logdet=3e-2 | 0.835028 | 0.529609 |
| lambda_logdet=1.5e-2 | 0.835972 | 0.529521 |
| lambda_logdet=1e-2 + tail_top_k_ratio=0.10 | 0.836370 | 0.529346 |
| lambda_logdet=1e-2 + density_weight=0.20 | 0.836753 | 0.528860 |

Current tuning interpretation:

- The best F1 region is near `lambda_logdet=2e-2`.
- F1 gain is real but small: `+0.002446` over H+ DVT reference.
- AUROC drops by about `-0.001066`, so this trades some continuous ranking for
  a slightly better F1 surface.
- This is still far from SuperADD F1 and should remain diagnostic.

Phase-3 queued:

```text
remote: /workspace/results_remote/flowtte_hparam_phase3_20260710_v3
script: scripts/run_flow_tte_hparam_phase3_remote.sh
status at handoff: waiting for phase-2 completion
```

Phase-3 tests:

- fine logdet values `1.75e-2`, `2.25e-2`, `2.5e-2`, `2.75e-2`;
- `lambda_logdet=2e-2` with density and tail variants;
- `lambda_logdet=2e-2` with brightness augmentation;
- `lambda_logdet=2e-2` with lower affine clamp.

## 10. What To Do Next

Immediate next session steps:

1. Poll phase-2 and phase-3 controller/leaderboard status.
2. When phase-2 completes, pull summaries and verify `anomaly_maps/` cleanup.
3. Let phase-3 finish if it has already started; otherwise decide whether the
   partial phase-2 signal is worth the remaining GPU time.
4. Update:
   - `skill_graph/experiments/2026-07-10_flowtte_transformer_context_and_hparam_sweep/report.md`
   - `tasks/todo.md`
   - `skill_graph/log.md`
   - `skill_graph/index.md`
5. Compare final best against:
   - H+ DVT reference: `0.836739 / 0.527427`
   - reported SuperADD context: `0.839300 / 0.626113`

Decision rule:

```text
If final tuning only gains <= about 0.003 F1 and lowers AUROC,
do not claim a method improvement. Record as diagnostic evidence that
logdet regularization slightly adjusts the mask operating surface.
```

Recommended research direction after tuning:

```text
score-field/mask formation structure
not class-specific hparam tuning
not raw register-prefix Transformer Flow
not plain Conv2D/Transformer flow replacement
```

Potential next structural directions, if continuing:

- constrained residual score-field correction that preserves local anomaly
  contrast;
- reliability-aware foreground/object prior rather than hard suppression;
- support-stat calibration that does not subtract position artifacts from the
  final score field directly;
- morphology/threshold policy only as a controlled comparator, not as the
  method claim itself;
- gated ensemble between MLP local contrast and Transformer smoothing, using a
  class-agnostic reliability signal.

## 11. Important Files

Current method and flow code:

```text
src/flow_tte/config.py
src/flow_tte/denoising.py
src/flow_tte/trainer.py
src/flow_tte/scoring.py
src/flow_tte/memory.py
src/flow_tte/conv2d_flow.py
src/flow_tte/transformer_flow.py
scripts/flow_tte_mvtec_ad2_core.py
scripts/run_flow_tte_mvtec_ad2.py
scripts/run_flow_tte_dvt_denoising_all8_remote.sh
```

Current hparam scripts:

```text
scripts/run_flow_tte_hparam_extreme_remote.sh
scripts/run_flow_tte_hparam_phase2_remote.sh
scripts/run_flow_tte_hparam_phase3_remote.sh
```

Key reports:

```text
skill_graph/analysis/2026-07-09_flowtte_current_method_results_issues.md
skill_graph/experiments/2026-07-10_flowtte_transformer_context_and_hparam_sweep/report.md
skill_graph/experiments/2026-07-10_flowtte_transformer_failure_image_analysis/report.md
skill_graph/experiments/2026-07-09_flowtte_transformer_flow_diagnostics/report.md
skill_graph/experiments/2026-07-09_flowtte_conv2d_flow_diagnostics/report.md
skill_graph/experiments/2026-07-09_flowtte_patch_structure_flow_diagnostics/report.md
skill_graph/experiments/2026-07-09_flowtte_structured_memory_diagnostics/report.md
```

Key local result roots:

```text
results/remote_runs/dsba3/flowtte_hparam_extreme_20260710_v1
results/remote_runs/dsba3/flowtte_hparam_phase2_20260710_v2_partial
results/remote_runs/dsba3/flowtte_transformer_failure_map_analysis_20260710_v1
results/remote_runs/dsba3/flowtte_transformer_dw0_maps_all8_20260710_v1
results/remote_runs/dsba3/flowtte_mlp_dw0_maps_all8_20260710_v1
```

## 12. Current Open Risks

- Phase-2 and phase-3 were still running at handoff, so final tuning numbers may
  supersede the partial table above.
- The SuperADD number is recorded context, not necessarily a fully rerun
  same-code artifact inside this repo. Treat it as reported context unless a
  same-condition SuperADD artifact is explicitly available.
- Current hparam gains are too small for a method claim.
- `can` remains essentially collapsed across many variants; do not tune only
  for `can`, because the user explicitly rejected class-specific tuning.
- Register claims must be phrased carefully: conceptually useful as global
  context, but current experiments did not show performance gain.

## 13. Handoff Verdict

```text
Current branch:
  FlowTTE H+ DVT hparam tuning

Verdict:
  KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC

Reason:
  Best partial hparam result improves F1 from 0.527427 to 0.529874, but this is
  small, lowers AUROC, and remains far below SuperADD F1 0.626113.

Asset to keep:
  H+ DVT fixed-memory FlowTTE is a strong diagnostic baseline and a useful
  platform for studying score-field/mask formation failures.
```
