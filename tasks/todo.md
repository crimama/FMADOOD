# FSAD-TTA Task Log

## 2026-07-13 Few-Shot AD2 Basic/Ours+/Ours++ Model Ladder

- Status: design documented; execution not started
- Protocol: `fmad-experiment-protocol`
- Target: MVTec AD2 single-image, all eight objects, full TESTpublic
- Primary claim: DINOv2-L Basic versus same-run, exact-reference SuperAD-16
- Scaling variants: Ours+ DINOv2-R and Ours++ DINOv3-H+
- Explicit exclusion: no `Ours++ Full` candidate or new full-normal run
- [x] Freeze model roles, comparison boundaries, metrics, and claim language
- [x] Incorporate prior H+ sweep negatives and bounded tuning priorities
- [ ] Reproduce exact-reference Basic/SuperAD/RN-FMLK anchors
- [x] Implement dual-server parallel controller with duplicate-safe queues
- [ ] Run complete Basic depth-width factorial: layers 1/2/4 x hidden 1/2
- [ ] Run anchor-capacity log-det/brightness 2x2 factorial, excluding the
  duplicate anchor configuration
- [ ] If positive, run density/kNN/memory/LOO robustness gates
- [ ] Freeze Basic and run paper-aligned SuperAD comparison
- [ ] Run Basic random shots 1/2/4/8/16 over seeds 0--4
- [ ] Transfer the frozen detector to Ours+ and Ours++
- [ ] Produce final fair-comparison, shot-curve, scaling, and ablation tables
- Design:
  `skill_graph/analysis/2026-07-13_fewshot_ad2_model_ladder_experiment_plan.md`

## 2026-07-10 DARC Resolution and Correspondence Gates

- Status: Gate 1 and raw-ladder pilot completed; raw DARC-v1 terminal stopped
- Protocol: `fmad-experiment-protocol`, staged mechanism diagnostic
- Target: MVTec AD2 single-image; public labels are development/shadow only
- Development gate: MVTec AD1 normal images plus deterministic synthetic thin cues
- Remote: dsba3 `147.47.39.144:2222`, fixed container `hun_fsad_tta_012`
- Requested resources: host GPUs `0,1,2,3`; container CUDA slots recorded separately
- Retained anchor: 672 H+ `[7,15,23,31]` + DVT + MLP FlowTTE
- Candidate family: layer-7 native crops with paired 16-pixel/8-pixel sampling,
  then `G0/L0/L1/R1` and coarse-only fallback
- Resource labels:
  - `P16-random`: genuine 16-image access, seeds `0,1,2`; diagnostic only
  - `M16-fullpool`: full-pool selection to 16 memories; never called few-shot
  - `P16-superad`: exact same SuperAD-selected 16 paths for paper-aligned comparison
  - `Pfull`: full normal-pool access, reported separately
- Checklist:
  - [x] Audit dsba3 GPU/container/process state and dataset population
  - [x] Freeze protocol-parity evaluator and exact pixel-center coordinate tests
  - [x] Implement paired resolution scorer and deterministic synthetic cue suite
  - [ ] Run Gate 0 protocol parity
  - [x] Run Gate 1 resolution-only diagnostic
  - [ ] If Gate 1 passes, run `G0/L0/L1/R1` and operational factorial nulls
  - [ ] Run frozen reduced AD2 shadow set: can/fabric/vial/wallplugs/rice
  - [x] Run the pre-metric AD2 raw-ladder pilot: can, seed 0, fold 0,
        G0/L0/L1/R1, full test_public good+bad
  - [ ] Run all-eight P16/Pfull only when preceding gates remain scientifically valid
  - [x] Pull compact artifacts, judge gates, update report/log/index
  - [x] Remove completed local/remote `anomaly_maps/` after analysis
- Report:
  `skill_graph/experiments/2026-07-10_flowtte_darc_resolution_correspondence/report.md`
- Raw-pilot result:
  - G0: `0.667650` pAUROC@.05 / `0.017797` oracle F1
  - L0: `0.547831` / `0.000488`
  - L1: `0.579249` / `0.000827`
  - R1: `0.550352` / `0.001647`
  - R1-G0: `-0.117298` pAUROC, `-0.001312` AP,
    `-0.016150` oracle F1, `-0.083333` component recall
  - Coverage: all queries accepted `5/5` registrations; token-weighted fallback
    `2.370760%`; no missing/duplicate/nonfinite maps
- Gate 1 context: `passed=true` over 720 rows; AP gain `+0.096791`
  (`+90.72%` relative), component-recall lower bound `+0.004167`, and
  broad-control pAUROC delta `+0.001697`; fixed-threshold deployment remained
  disallowed
- Compact artifact:
  `results/remote_runs/dsba3/darc_ad2_raw_ladder_can_s0f0_full8_20260711_v2`
- Cleanup evidence: valid root `648 -> 0` TIFFs; all related raw-ladder roots
  now contain `0` TIFFs; canonical evaluator `12/12` files, eight completion
  records, and eight coverage files remain; local compact pullback has no TIFFs
- Frozen verdict: `KILL_FOR_CLAIM / CONTRACT_CONFLICT`; both the L1-L0
  continuation clause and R1-G0 stop clause fired
- Operational decision: stop raw R1 and do not promote it
- Actual-data pilot contract:
  `skill_graph/analysis/2026-07-11_flowtte_darc_ad2_raw_ladder_pilot.md`

## 2026-07-10 FlowTTE H+ DVT Exact SuperAD Rotation-8 Ablation

- Status: completed
- Protocol: fmad-experiment-protocol, matched augmentation ablation lane
- Target: MVTec AD2 single-image, all 8 public objects
- Remote container: `hun_fsad_tta_012`; host GPUs `0,1,2`
- Fixed method: `dinov3_vith16plus`, layers `[7,15,23,31]`,
  `feature_fusion=layer_norm_mean`, pooled DVT `position_mean alpha=1.0`,
  fused patch-wise MLP flow, fixed ordered 16-source support JSON, seed `0`,
  and `calibration_sample_size=4096`
- Changed variable:
  - identity control: `16` effective support views
  - candidate: exact SuperAD rotations `0,45,...,315`, `128` effective views
- Matched result:
  - identity: `0.836389` AUROC / `0.527370` F1
  - rotation-8: `0.819251` AUROC / `0.506201` F1
  - delta: `-0.017137` AUROC / `-0.021168` F1
  - largest failure: `sheet_metal -0.111020` AUROC / `-0.119605` F1
  - only continuation-threshold positive: `fruit_jelly +0.011249` F1
- Runtime/provenance:
  - exact OpenCV transform parity verified for all eight angles
  - full-batch-equivalent row microbatching added for the 8x support bank
  - worst-case `can` smoke passed without OOM at about 24.5 GiB observed
  - manifests, selected support paths, support hash, and the executed v1
    method-bundle hash matched; no unexpected paired config difference
  - post-run review expanded the method guard to the actual 48-file
    `FSAD_ROOT`/`PROJECT_ROOT` closure, including `src/utils.py`; the updated
    controller rejects stale v1 markers before GPU work
  - remote/local evaluator files are not bit-identical, so absolute local
    reproduction is a documented caveat; both paired arms used the same remote
    evaluator
  - local and remote retained `anomaly_maps/` count: `0`
- Local pullbacks:
  - `results/remote_runs/dsba3/flowtte_hplus_dvt_identity_cal4096_all8_20260710_v1`
  - `results/remote_runs/dsba3/flowtte_hplus_dvt_superad_rotation8_all8_20260710_v1`
  - `results/remote_runs/dsba3/flowtte_hplus_dvt_superad_rotation8_can_smoke_20260710_v1`
- Report:
  `skill_graph/experiments/2026-07-10_flowtte_hplus_dvt_superad_rotation8/report.md`
- Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`

## 2026-07-10 FlowTTE H+ Extreme Hyperparameter Sweep

- Status: running
- Protocol: fmad-experiment-protocol, diagnostic continuation lane
- Target dataset: MVTec AD2 single-image, all 8 public objects
- Remote container: `hun_fsad_tta_012`
- Host GPUs: `0,1,2`
- Fixed structure: `dinov3_vith16plus`, layers `[7,15,23,31]`,
  `feature_fusion=layer_norm_mean`, DVT `position_mean alpha=1.0`,
  fixed 16-shot support JSON, no-TTE, latent NN distance scoring
- Reference: H+ DVT FlowTTE `0.836739` AUROC / `0.527427` F1
- Stage-1 completed:
  - best F1: `lambda_logdet=1e-2`, `0.836202` AUROC / `0.528875` F1
  - best AUROC: support brightness `0.80,1.20`, `0.838230` AUROC /
    `0.528203` F1
  - harmful axes: higher DVT alpha, more coupling layers, higher LR,
    more epochs
  - cleanup evidence: stage-1 local and remote `anomaly_maps/` count `0`
- Phase-2 running:
  - current best partial result: `lambda_logdet=2e-2`,
    `0.835673` AUROC / `0.529874` F1
  - logdet `3e-2` reduced F1 and further reduced AUROC
- Phase-3 queued after phase-2:
  - focuses on `lambda_logdet ~= 2e-2` with density, tail, brightness,
    and clamp combinations
- Local pullbacks:
  - `results/remote_runs/dsba3/flowtte_hparam_extreme_20260710_v1`
  - `results/remote_runs/dsba3/flowtte_hparam_phase2_20260710_v2_partial`
- Report:
  `skill_graph/experiments/2026-07-10_flowtte_transformer_context_and_hparam_sweep/report.md`
- New-session handoff:
  `skill_graph/analysis/2026-07-10_flowtte_session_handoff.md`
- Interim verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`

## 2026-07-09 FlowTTE Transformer Coupling Flow Diagnostic

- Status: completed
- Protocol: fmad-experiment-protocol, structural diagnostic lane
- Target dataset: MVTec AD2 single-image
- Remote container: `hun_fsad_tta_012`
- Host GPUs: `0,1,2`
- Method: `normality_mode=transformer_flow`
- Shared settings: `dinov3_vith16plus`, layers `[7,15,23,31]`, fixed
  SuperAD-16 support, DVT position mean denoise `alpha=1.0`, no-TTE,
  `density_weight=0.0`, `calibration_sample_size=4096`
- Local verification:
  - `python3 -m py_compile src/flow_tte/transformer_flow.py src/flow_tte/conv2d_flow.py scripts/flow_tte_mvtec_ad2_core.py scripts/run_flow_tte_mvtec_ad2.py`
  - `pytest -q tests/test_flow_tte_transformer_flow.py tests/test_flow_tte_conv2d_flow.py tests/test_flow_tte_context.py tests/test_flow_tte_layerwise.py tests/test_flow_tte_score_field.py`: `26 passed`
  - CLI help exposed `transformer_flow`
  - remote py_compile + CLI check: `remote_transformer_cli_ok`
- Smoke result:
  - same-object H+ DVT baseline mean: `0.758338` AUROC / `0.377474` F1
  - Transformer flow mean: `0.768982` AUROC / `0.395014` F1
  - `vial` improved by `+0.046938` F1; `can` was slightly lower but already
    collapsed
- All-eight result:
  - H+ DVT baseline: `0.836739` AUROC / `0.527427` F1
  - Transformer flow: `0.828600` AUROC / `0.502237` F1
  - positives: `vial`, `sheet_metal`, `fabric`
  - failures: `fruit_jelly`, `walnuts`, `wallplugs`, `rice`
- Local pullbacks:
  - `results/remote_runs/dsba3/flowtte_transformer_flow_hplus_smoke_can_vial_fabric_20260709_v1`
  - `results/remote_runs/dsba3/flowtte_transformer_flow_hplus_all8_20260709_v1`
- Cleanup evidence:
  - remote `anomaly_maps/`: `0` left
  - local `anomaly_maps/`: `0` left
- Report:
  `skill_graph/experiments/2026-07-09_flowtte_transformer_flow_diagnostics/report.md`
- Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`

## 2026-07-09 FlowTTE Conv2D Flow Diagnostic

- Status: completed
- Protocol: fmad-experiment-protocol, reduced-object structural diagnostic
- Target dataset: MVTec AD2 single-image
- Remote container: `hun_fsad_tta_012`
- Host GPUs: `0,1,2`
- Objects: `can,vial,fabric`
- Split: full `test_public/good,bad`
- Method: `normality_mode=conv2d_flow`
- Shared settings: `dinov3_vith16plus`, layers `[7,15,23,31]`, fixed
  SuperAD-16 support, DVT position mean denoise `alpha=1.0`, no-TTE,
  `density_weight=0.0`, `calibration_sample_size=4096`
- Local verification:
  - `pytest -q tests/test_flow_tte_conv2d_flow.py`: `2 passed`
  - `pytest -q tests/test_flow_tte_conv2d_flow.py tests/test_flow_tte_context.py tests/test_flow_tte_layerwise.py tests/test_flow_tte_score_field.py`: `24 passed`
  - remote `py_compile` + CLI `conv2d_flow` help check: `remote_cli_ok`
- Smoke result:
  - same-object H+ DVT baseline mean: `0.758338` AUROC / `0.377474` F1
  - Conv2D flow mean: `0.744949` AUROC / `0.354137` F1
  - all three objects lost F1; `can` stayed collapsed
- Local pullback:
  `results/remote_runs/dsba3/flowtte_conv2d_flow_hplus_smoke_can_vial_fabric_20260709_v1`
- Cleanup evidence:
  - remote `anomaly_maps/`: `0` left
  - local `anomaly_maps/`: `0` left
- Report:
  `skill_graph/experiments/2026-07-09_flowtte_conv2d_flow_diagnostics/report.md`
- Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`

## 2026-07-09 FlowTTE Patch-Structure Flow Diagnostics

- Status: completed
- Protocol: fmad-experiment-protocol, all-object structural diagnostic lane
- Target dataset: MVTec AD2 single-image
- Remote container: `hun_fsad_tta_012`
- Host GPUs: `0,1,2`
- Objects: all 8 public MVTec AD2 objects
- Split: full `test_public/good,bad`
- Fixed reference: H+ DVT FlowTTE baseline,
  `dinov3_vith16plus`, layers `[7,15,23,31]`, fixed 16-shot support,
  no-TTE, DVT position mean denoise `alpha=1.0`,
  `seg_AUROC_0.05=0.836739`, `seg_F1=0.527427`
- Implemented variants:
  - `conditional_cls`
  - `conditional_xy`
  - `conditional_cls_xy`
  - `foreground_flow_mixture`
  - `local_contrast`
- Local verification:
  - `python3 -m py_compile scripts/flow_tte_mvtec_ad2_core.py scripts/run_flow_tte_mvtec_ad2.py scripts/flow_tte_score_field.py`
  - `python3 scripts/run_flow_tte_mvtec_ad2.py --help`
  - `uv run ruff check ...`
  - `python3 -m pytest tests/test_flow_tte_score_field.py tests/test_flow_tte_layerwise.py -q`
- Execution:
  - completed variants: `conditional_cls`, `foreground_flow_mixture`,
    `local_contrast`
  - runtime-blocked variants: `conditional_xy`, `conditional_cls_xy`
- Mean results:
  - H+ reference: `seg_AUROC_0.05=0.836739`, `seg_F1=0.527427`
  - `conditional_cls`: `0.832374`, `0.512126`
  - `foreground_flow_mixture`: `0.834712`, `0.519250`
  - `local_contrast`: `0.806200`, `0.438345`
  - reported SuperADD context: `0.839300`, `0.626113`
- Interpretation:
  - none of the completed structural variants beats the H+ reference F1
  - foreground/background flow mixture is closest but still loses by
    `-0.008177` F1 and does not solve the `can` collapse
  - local contrast is broadly harmful
  - coordinate-conditioned NF (`xy`, `cls_xy`) is runtime-blocked in this path
    and should not be pursued without first fixing/justifying the mechanism
- Cleanup evidence:
  - remote `anomaly_maps/`: `0` left
  - local `anomaly_maps/`: `0` left
  - dsba3 GPUs `0,1,2` idle after completion
- Report:
  `skill_graph/experiments/2026-07-09_flowtte_patch_structure_flow_diagnostics/report.md`
- Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`

## 2026-07-08 FlowTTE Score-Field and Layer-wise Structural Diagnostics

- Status: completed
- Protocol: fmad-experiment-protocol, all-object structural diagnostic lane
- Target dataset: MVTec AD2 single-image
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- Remote container: `hun_fsad_tta_012`
- Host GPUs: `0,1,2`; in-container CUDA slots: `0,1,2`
- Objects: all 8 public MVTec AD2 objects
- Split: full `test_public/good,bad`
- Fixed reference: H+ DVT FlowTTE baseline,
  `dinov3_vith16plus`, layers `[7,15,23,31]`, fixed 16-shot support,
  no-TTE, DVT position mean denoise `alpha=1.0`
- Score-field variants:
  - baseline: `0.836739` AUROC / `0.527427` F1
  - support_position_center: `0.785133` / `0.429653`
  - support_position_zscore: `0.701830` / `0.278144`
  - foreground_energy: `0.834598` / `0.526807`
  - center_plus_foreground: `0.808839` / `0.471613`
- Layer-wise variants:
  - fused baseline: `0.836739` / `0.527427`
  - layer-wise no-context score fusion: `0.828210` / `0.499110`
  - layer-wise CLS topM4 routed: `0.829923` / `0.508863`
- Interpretation: support-position score calibration is harmful; foreground
  prior is baseline-tied and not robust; per-layer Flow banks with score-level
  fusion underperform the fused normalized feature. CLS routing partially
  recovers the layer-wise drop but remains below baseline.
- Cleanup evidence: all completed local and remote result roots have `0`
  retained `anomaly_maps/` directories.
- Local pullbacks:
  - `results/remote_runs/dsba3/flowtte_scorefield_structural_all8_20260708_v4`
  - `results/remote_runs/dsba3/flowtte_layerwise_ctx_cls_topm4_all8_20260708_v1`
  - `results/remote_runs/dsba3/flowtte_layerwise_noctx_all8_20260708_v1`
- Reports:
  - `skill_graph/experiments/2026-07-08_flowtte_scorefield_structural/report.md`
  - `skill_graph/experiments/2026-07-08_flowtte_layerwise_context_routed/report.md`
- Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`

## 2026-07-08 FlowTTE H+ Priority Sequence Diagnostic

- Status: completed
- Protocol: fmad-experiment-protocol, priority diagnostic continuation lane
- Target dataset: MVTec AD2 single-image
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- Remote container: `hun_fsad_tta_012`
- Host GPUs: `0,1,2`; in-container CUDA slots: `0,1,2`
- Objects: all 8 public MVTec AD2 objects
- Split: full `test_public/good,bad`
- Shared reference: H+ backbone-only FlowTTE DVT alpha `1.0`,
  `dinov3_vith16plus` layers `[7,15,23,31]`, fixed 16-shot support,
  no-TTE, no context.
- Priority 1: threshold/morphology on saved H+ maps.
  - raw H+ NF: `seg_AUROC_0.05=0.836739`, `seg_F1=0.527427`
  - close/fill: `seg_F1=0.541344`
  - close/fill/erode: `seg_F1=0.542316`
  - delta vs raw H+ NF: `+0.014888` F1
  - gap vs reported SuperADD F1: `-0.083797`
- Priority 2: no-NF identity feature-distance control.
  - `flow_transform_mode=identity`, `density_weight=0.0`
  - mean `seg_AUROC_0.05=0.832461`, `seg_F1=0.524804`
  - delta vs raw H+ NF: `-0.004278` AUROC, `-0.002623` F1
- Interpretation: morphology explains a measurable but insufficient part of
  the F1 gap. Removing the learned NF projection does not improve the all8
  mean, so NF latent projection is not the primary mean-metric bottleneck.
  Remaining weak objects are analysis buckets only; they should not be used
  for class-specific hyperparameter tuning. The next valid diagnostic should
  apply one class-agnostic rule across all eight objects.
- Cleanup evidence: both priority result roots have `0` local/remote
  `anomaly_maps/` directories; dsba3 GPUs `0,1,2` were idle after completion.
- Local pullbacks:
  - `results/remote_runs/dsba3/flowtte_hplus_postprocess_all8_20260708_v1`
  - `results/remote_runs/dsba3/flowtte_hplus_identity_feature_nn_all8_20260708_v1`
- Report:
  `skill_graph/experiments/2026-07-08_flowtte_hplus_priority_sequence/report.md`
- Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`

## 2026-07-08 FlowTTE DVT H+ Backbone-Only AD2 Diagnostic

- Status: completed
- Protocol: fmad-experiment-protocol, backbone-only SuperADD alignment lane
- Target dataset: MVTec AD2 single-image
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- Remote container: `hun_fsad_tta_012`
- Host GPUs: `0,1,2`; in-container CUDA slots: `0,1,2`
- Objects: all 8 public MVTec AD2 objects
- Split: full `test_public/good,bad`
- Candidate: previous best FlowTTE DVT alpha `1.0` branch with only the
  backbone changed to `dinov3_vith16plus`, layers `[7,15,23,31]`
- Held fixed: 16-shot fixed support JSON, no-TTE, no register/CLS context,
  `density_weight=0.25`, latent NN scoring, DVT position mean denoise
- Local pullback:
  `results/remote_runs/dsba3/flowtte_dvt_hplus_backbone_only_all8_20260708_v1`
- Mean metrics:
  - previous DINOv3-L DVT alpha `1.0`: `seg_AUROC_0.05=0.825207`,
    `seg_F1=0.468348`
  - H+ backbone-only: `seg_AUROC_0.05=0.836739`, `seg_F1=0.527427`
  - delta vs previous: `+0.011532` AUROC, `+0.059079` F1
  - recorded SuperAD-16 context: `0.765802`, `0.385534`; H+ delta
    `+0.070937`, `+0.141893`
  - reported SuperADD context: `0.839300`, `0.626113`; H+ delta
    `-0.002561`, `-0.098686`
- Interpretation: H+ backbone explains a substantial part of the gap and
  nearly closes AUROC, but the remaining F1 gap is still large. Next
  diagnostic should keep H+ fixed and isolate threshold/morphology or raw
  layer-wise NN scoring rather than broad hyperparameter tuning.
- Cleanup evidence: local and remote `anomaly_maps/` directory count is `0`;
  GPUs `0,1,2` were idle after completion.
- Report:
  `skill_graph/experiments/2026-07-08_flowtte_hplus_backbone_only/report.md`
- Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`

## 2026-07-07 FlowTTE Component Ablation

- Status: completed
- Protocol: fmad-experiment-protocol, all-eight-object component diagnostic lane
- Target dataset: MVTec AD2 single-image
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- Remote container: `hun_fsad_tta_012`
- Host GPUs: `0,1,2`; in-container CUDA slots: `0,1,2`
- Objects: all 8 public MVTec AD2 objects
- Split: full `test_public/good,bad`
- Few-shot setting: 16 train/good support images per object
- Shared config: `dinov3_vitl16`, fixed DINOv3 no-context coreset support
  paths, no-TTE unless ablated, `density_weight=0.25`
- Main result:
  - no-context base: `seg_AUROC_0.05=0.797743`, `seg_F1=0.437800`
  - best component: `CLS soft w10`, `0.805427`, `0.447118`
  - delta vs base: `+0.007684` AUROC, `+0.009318` F1
  - TTE budget `1.25`: `0.769788`, `0.420321`
  - NF NLL only: `0.682066`, `0.225544`
  - register-conditioned NF: `0.796554`, `0.436152`
- Interpretation: the primary positive component is CLS-conditioned soft
  memory distance on top of static no-TTE latent distance. NF NLL is not a
  standalone scoring module, and structural register usage is not a mean-metric
  driver in the tested forms.
- Cleanup evidence: component/follow-up remote roots have `0` retained
  anomaly-map TIFFs; morphology reduced audit maps were cleaned from
  `2432 -> 0`.
- Report:
  `skill_graph/experiments/2026-07-07_flowtte_component_ablation/report.md`
- Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`

## 2026-07-07 FlowTTE Register Failure Analysis Design

- Status: completed Phase A analysis.
- Protocol: fmad-experiment-protocol diagnostic continuation lane.
- Goal: explain why `register_topm4` and `register_condnf` are weaker than
  `CLS w10` before running more tuning.
- Phase A: context separability, support-routing replacement,
  latent-distance inflation, and conditional-NF distortion diagnostics.
- Result:
  - CLS context separability wins 6/8 objects; mean delta `+0.008771`.
  - register context separability wins 2/8 objects; mean delta `+0.001372`.
  - register top-M has better nearest-neighbor retention than CLS but does not
    explain F1 gains.
  - register-conditioned NF improves NLL bad-good separation on 7/8 objects,
    but prior segmentation metrics remain below no-context on mean.
- Phase B: only if continued, preserve maps for a reduced
  `fabric,can,wallplugs,vial` morphology audit and clean them afterward.
- Design:
  `skill_graph/analysis/2026-07-07_flowtte_register_failure_analysis_design.md`
- Report:
  `skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/report.md`

## 2026-07-07 FlowTTE Structural Register AD2 Diagnostics

- Status: completed
- Protocol: fmad-experiment-protocol, structural context diagnostic lane
- Target dataset: MVTec AD2 single-image
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- Remote container: `hun_fsad_tta_012`
- Host GPUs: `0,1,2`; in-container CUDA slots: `0,1,2`
- Objects: all 8 public MVTec AD2 objects
- Split: full `test_public/good,bad`
- Few-shot setting: 16 train/good support images per object
- Shared config: `dinov3_vitl16`, `dinov3_cls_greedy_coreset`, no-TTE
  `expansion_budget=1.0`, `density_weight=0.25`
- Structural variants:
  - register top-M routing: `context_mode=top_m`, `context_top_m=4`
  - register-conditioned NF: `flow_condition_mode=context`, `context_mode=none`
- Mean metrics:
  - no-context DINOv3 diagnostic: `seg_AUROC_0.05=0.797743`,
    `seg_F1=0.437800`
  - CLS w10 diagnostic: `0.805427`, `0.447118`
  - register top-M=4: `0.798846`, `0.434411`
  - register-conditioned NF: `0.796554`, `0.436152`
- Cleanup evidence: remote and local `anomaly_maps/` count under completed
  result roots is `0`; chunk `cleanup_evidence.txt` files are preserved.
- Report:
  `skill_graph/experiments/2026-07-07_flowtte_structural_register/report.md`
- Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`

## 2026-07-07 Flow-LatentBank no-TTE DINOv3 Fixed-Reference AD2 Diagnostic

- Status: completed
- Protocol: fmad-experiment-protocol, support-control diagnostic lane
- Target dataset: MVTec AD2 single-image
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- Remote container: `hun_fsad_tta_012`
- Host GPUs: `0,1,2`; in-container CUDA slots: `0,1,2`
- Objects: all 8 public MVTec AD2 objects
- Split: full `test_public/good,bad`
- Few-shot setting: 16 train/good support images per object
- Candidate: `Flow-LatentBank` no-TTE with `dinov3_vitl16`, but forced to use
  the exact DINOv2 SuperAD-16 reference image set via
  `fixed_json=skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/superad16_dinov2_reference_paths.json`
- Local pullback:
  `results/remote_runs/dsba3/flow_latentbank_mvtecad2_all8_shot16_dinov3vitl16_fixed_dinov2ref_notte_dw025_20260707_v1`
- Mean metrics:
  - fixed-reference DINOv3: `seg_AUROC_0.05=0.800727`,
    `seg_F1=0.437437`
  - DINOv3 CLS coreset control: `0.797743`, `0.437800`;
    fixed-reference delta `+0.002983` / `-0.000363`
  - DINOv2 same-reference control: `0.761782`, `0.350391`;
    fixed-reference delta `+0.038945` / `+0.087046`
  - recorded SuperAD-16: `0.765802`, `0.385534`;
    fixed-reference delta `+0.034925` / `+0.051902`
  - reported SuperADD: `0.839300`, `0.626113`;
    fixed-reference delta `-0.038573` / `-0.188676`
- Interpretation: DINOv3 retains the gain when constrained to the exact
  DINOv2 SuperAD-16 reference image set. The prior DINOv3 gain is therefore
  mostly backbone/feature quality, not DINOv3 coreset support-selection luck.
- Cleanup evidence: local and remote `anomaly_maps/` count under the completed
  result root is `0`.
- Report:
  `skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/report.md`
- Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`

Next constrained diagnostic:

- Keep the fixed DINOv2 SuperAD-16 reference set and test one
  SuperADD-aligned localization component at a time, starting with
  preprocessing/masking or post-threshold morphology. Do not spend the next
  run on another support-selection policy sweep.

## 2026-07-07 Flow-LatentBank no-TTE 16-shot DINOv2/DINOv3 AD2 Diagnostic

- Status: completed
- Protocol: fmad-experiment-protocol, all-eight-object SuperAD-16 budget lane
- Target dataset: MVTec AD2 single-image
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- Remote container: `hun_fsad_tta_012`
- Host GPUs: `0,1,2`; in-container CUDA slots: `0,1,2`
- Objects: all 8 public MVTec AD2 objects
- Split: full `test_public/good,bad`
- Few-shot setting: 16 train/good support images per object
- Candidate A: `Flow-LatentBank` no-TTE with `dinov2_vitl14`,
  `dinov2_cls_greedy_coreset`, `expansion_budget=1.0`
- Candidate B: `Flow-LatentBank` no-TTE with `dinov3_vitl16`,
  `dinov3_cls_greedy_coreset`, `expansion_budget=1.0`
- Shared config: `score_mode=latent_distance`,
  `feature_fusion=layer_norm_mean`, support transforms `identity`,
  `flow_epochs=3`, `density_weight=0.25`
- SuperAD-16 source:
  `/home/hun/Volume/RESEARCH/FMAD-OOD/configs/baselines/recorded_superad16_mvtec_ad2_8object_metrics.json`
- SuperADD context source:
  `results/remote_runs/dsba3/flow_tte_nf_mvtecad2_all8_shot16_coreset_dw0_20260707_v1/comparison_superadd_reported/comparison_rows.tsv`
- Local pullbacks:
  - `results/remote_runs/dsba3/flow_latentbank_mvtecad2_all8_shot16_vitl14_notte_dw025_20260707_v1`
  - `results/remote_runs/dsba3/flow_latentbank_mvtecad2_all8_shot16_dinov3vitl16_notte_dw025_20260707_v1`
- Mean metrics:
  - SuperAD-16: `seg_AUROC_0.05=0.765802`, `seg_F1=0.385534`
  - DINOv2 no-TTE: `seg_AUROC_0.05=0.761782`,
    `seg_F1=0.350391`; delta vs SuperAD-16 `-0.004020` /
    `-0.035144`
  - DINOv3 no-TTE: `seg_AUROC_0.05=0.797743`,
    `seg_F1=0.437800`; delta vs SuperAD-16 context `+0.031941` /
    `+0.052266`
  - Reported SuperADD: `seg_AUROC_0.05=0.839300`,
    `seg_F1=0.626113`; DINOv3 delta `-0.041557` / `-0.188312`
- Interpretation: DINOv2 no-TTE is the closest SuperAD-16 comparable run and
  does not pass the strict claim gate because F1 remains below SuperAD-16.
  DINOv3 gives a clear positive backbone gradient, but it is diagnostic only
  because support/backbone differ and reported SuperADD is not a same-evaluator
  rerun.
- Cleanup evidence: remote/local `cleanup_anomaly_maps=true`, no retained
  `anomaly_maps/` under the two completed result roots.
- Report:
  `skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_shot16_dinov2_dinov3/report.md`
- Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`

Next constrained diagnostic:

- Separate DINOv3 representation gain from support-selection gain by running
  DINOv3 features on the exact DINOv2 SuperAD-16 reference image set, then
  compare against the DINOv3 CLS coreset run. Hard stop if the gain disappears
  under the same reference set or if F1 remains far below SuperADD.

## 2026-07-07 Flow-LatentBank No-TTE AD2 All8 Diagnostic

- Status: completed
- Protocol: fmad-experiment-protocol, all-eight-object ablation lane
- Target dataset: MVTec AD2 single-image
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- Remote container: `hun_fsad_tta_012`
- Host GPUs: `0,1,2`; in-container CUDA slots: `0,1,2`
- Objects: all 8 public MVTec AD2 objects
- Split: full `test_public/good,bad`
- Few-shot setting: 4 train/good support images per object
- Shared config: `dinov2_vitl14_reg`, `score_mode=latent_distance`,
  `feature_fusion=layer_norm_mean`, support policy `first`,
  support transforms `identity`
- Candidate: `Flow-LatentBank`, no TTE, `expansion_budget=1.0`
- Same-condition baseline: `FlowTTE-LatentBank`, TTE,
  `expansion_budget=1.25`
- Local pullbacks:
  - `results/remote_runs/dsba3/flow_latentbank_mvtecad2_all8_shot4_vitl14reg_notte_20260707_v1`
  - `results/remote_runs/dsba3/flowtte_latentbank_mvtecad2_all8_shot4_vitl14reg_tte_20260707_v1`
- Mean metrics:
  - no-TTE: `seg_AUROC_0.05=0.733798`, `seg_F1=0.297527`
  - TTE: `seg_AUROC_0.05=0.712458`, `seg_F1=0.266227`
  - no-TTE minus TTE: `+0.021340` AUROC, `+0.031300` F1
- Object wins: no-TTE wins 6/8 objects on AUROC and 6/8 on F1.
- No-TTE memory check: all objects have `initial_memory_size == final_memory_size`.
- Same-condition SuperAD baseline: `BLOCKED_BASELINE`; recorded SuperAD-16
  is context only because this run is 4-shot first-support.
- Cleanup evidence: remote/local `cleanup_anomaly_maps=true`, no retained
  `anomaly_maps/`.
- Verdict: `PROMISING_DIAGNOSTIC`

Next constrained diagnostic:

- Treat static Flow-LatentBank as the stronger branch than current TTE.
  If expansion is revisited, test one class-agnostic expansion gate against
  this static all8 baseline and hard-stop it unless the all-object mean improves
  without losing the no-TTE wins as a no-harm check.

## 2026-07-07 FlowTTE-LatentBank dinov2_vitl14_reg AD2 Reduced Diagnostic

- Status: completed
- Protocol: fmad-experiment-protocol, reduced-object backbone-control lane
- Target dataset: MVTec AD2 single-image
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- Remote container: `hun_fsad_tta`
- Host GPU: `3`; in-container CUDA slot: `0`
- Objects: `can`, `rice`
- Split: full `test_public/good,bad`
- Few-shot setting: 4 train/good support images per object
- Method: `FlowTTE-LatentBank`, `score_mode=latent_distance`,
  `feature_fusion=layer_norm_mean`
- Control run:
  `results/remote_runs/dsba3/flow_tte_latentbank_mvtecad2_can_rice_shot4_vitl14_current_20260707_v1`
- Register run:
  `results/remote_runs/dsba3/flow_tte_latentbank_mvtecad2_can_rice_shot4_vitl14reg_20260707_v1`
- Mean metrics:
  - `dinov2_vitl14`: `seg_AUROC_0.05=0.765758`,
    `seg_F1=0.315586`
  - `dinov2_vitl14_reg`: `seg_AUROC_0.05=0.771840`,
    `seg_F1=0.333811`
  - Delta register vs non-register: `+0.006082` AUROC,
    `+0.018226` F1
- Category asymmetry: `rice` improved by `+0.023516` AUROC and `+0.038357`
  F1, while `can` decreased by `-0.011352` AUROC and `-0.001906` F1.
- Same-condition SuperAD baseline: `BLOCKED_BASELINE`; recorded SuperAD-16
  is context only because this run is reduced-object 4-shot first-support.
- Cleanup evidence: remote/local `cleanup_anomaly_maps=true`, no retained
  `anomaly_maps/`.
- Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`

Next constrained diagnostic:

- If FlowTTE-LatentBank is revisited, run a support-matched all-eight AD2
  backbone pair before making `dinov2_vitl14_reg` the default. Hard stop if
  the gain remains rice-only or if can-like F1 collapse persists.

## 2026-07-07 FlowTTE Method Pivot

- Status: recorded
- Decision: keep the method name **FlowTTE**.
- Current prototype label for internal distinction: `FlowTTE-LatentBank`.
- Next branch label: `FlowTTE-NFScore-TTA`.
- Method pivot:
  - Current: train NF once on few-shot normal support, freeze it, project
    support/query patches into latent `z`, expand a latent memory bank, and
    score mainly by latent memory distance.
  - Next: use the NF itself as the scoring module with patch NLL as the primary
    anomaly score, then perform conservative test-time adaptation on selected
    pseudo-normal patches.
- DeCoFlow-inspired pieces to reuse:
  - NF likelihood/NLL as the main anomaly score.
  - Tail-aware loss for normal tail coverage.
  - Adapter-only or auxiliary-coupling updates instead of full NF fine-tuning.
  - Lightweight feature alignment before the NF.
- Explicit non-goal: do not port DeCoFlow's full continual-learning protocol or
  task-routing setup directly.
- Method note:
  `skill_graph/methods/flowtte_vnext_nf_scoring_tta.md`

Next constrained implementation:

- Add `score_mode = latent_distance | nf_nll`, verify static NF-NLL scoring
  first, then add adapter-only TTA if the static score is viable.

## 2026-07-07 FlowTTE-NFScore Static NLL Reduced Diagnostic

- Status: completed
- Protocol: fmad-experiment-protocol, reduced diagnostic lane
- Target dataset: MVTec AD2 single-image
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- Remote container: `hun_fsad_tta`
- Host GPU: `3`; in-container CUDA slot: `0`
- Objects: `can`, `rice`
- Split: full `test_public/good,bad`
- Few-shot setting: 4 train/good support images per object
- Method: `FlowTTE-NFScore` static raw NLL scoring
- Config: `score_mode=nf_nll`, `expansion_budget=1.0`,
  `distance_weight=0.0`, `density_weight=0.0`
- Remote result:
  `/workspace/results_remote/flow_tte_nfscore_mvtecad2_can_rice_shot4_static_nll_20260707_v1`
- Local pullback:
  `results/remote_runs/dsba3/flow_tte_nfscore_mvtecad2_can_rice_shot4_static_nll_20260707_v1`
- Metrics: `mean_segmentation_au_roc=0.674336`,
  `mean_segmentation_f1=0.127093`.
- Immediate baseline: previous `FlowTTE-LatentBank` can/rice 4-shot run with
  `mean_segmentation_au_roc=0.765772`, `mean_segmentation_f1=0.315589`.
- Delta vs LatentBank: `AUROC=-0.091436`, `F1=-0.188495`.
- Same-condition SuperAD baseline: `BLOCKED_BASELINE` for this reduced 4-shot
  diagnostic; recorded SuperAD-16 used only as context.
- Cleanup evidence: remote/local `cleanup_anomaly_maps=true`, no retained
  `anomaly_maps/`.
- Verdict: `KILL_FOR_CLAIM / NO_CONTINUE_STATIC_NLL`

Next constrained diagnostic:

- Do not scale raw static NF-NLL to full AD2. Any continuation should first add
  calibration or anchored adapter-only TTA and compare against this negative
  static-NLL gate plus the latent-bank baseline.

## 2026-07-06 FlowTTE NF Reduced Diagnostic

- Status: completed
- Protocol: fmad-experiment-protocol, reduced-object diagnostic lane
- Target dataset: MVTec AD2 single-image
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- Remote container: `hun_fsad_tta`
- Host GPU: `3`; in-container CUDA slot: `0`
- Objects: `can`, `rice`
- Split: full `test_public/good,bad`
- Few-shot setting: 4 train/good support images per object
- Method: FlowTTE NF projection with memory expansion enabled
- Remote result: `/workspace/results_remote/flow_tte_nf_can_rice_shot4_20260706_v1`
- Local pullback: `results/remote_runs/dsba3/flow_tte_nf_can_rice_shot4_20260706_v1`
- Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`

Next constrained diagnostic:

- Replace `first_n_train_good_seed_fixed` support with SuperAD-style DINO CLS coreset
  selection and run a same 4-shot static-vs-TTE ablation.
- Hard stop if `can` F1 remains near zero or TTE expansion is matched by a static
  no-expansion branch.

## 2026-07-06 FlowTTE NF MVTec AD1 Reduced Diagnostic

- Status: completed
- Protocol: fmad-experiment-protocol, reduced-object diagnostic lane
- Target dataset: MVTec AD1 classic single-image
- Remote AD1 source: local `/home/hun/Volume/DATA/MVTecAD` copied into
  container `/workspace/data/MVTecAD`
- Remote container: `hun_fsad_tta`
- Host GPU: `3`; in-container CUDA slot: `0`
- Objects: `bottle`, `hazelnut`
- Split: full `test/good` plus all defect-type test folders
- Few-shot setting: 4 train/good support images per object
- Method: FlowTTE NF projection with memory expansion enabled
- Remote result:
  `/workspace/results_remote/flow_tte_nf_mvtecad1_bottle_hazelnut_shot4_20260706_v2_metrics4`
- Local pullback:
  `results/remote_runs/dsba3/flow_tte_nf_mvtecad1_bottle_hazelnut_shot4_20260706_v2_metrics4`
- Extended metrics: `image_AUROC=0.9989285714`,
  `pixel_AUROC=0.9886983420`, `image_AP=0.9994360902`,
  `pixel_AP=0.7963940594`.
- SuperAD baseline source: `BLOCKED_BASELINE`; no same-condition AD1
  `bottle,hazelnut` SuperAD artifact was found in FSAD-TTA/FMAD-OOD records.
- Verdict: `BLOCKED_BASELINE`

Next constrained diagnostic:

- Run or port a same-condition SuperAD/official-superad AD1 baseline for
  `bottle,hazelnut`, same 4 support images, same DINOv2 preprocessing, and the
  same `seg_AUROC_0.05`/`seg_F1` evaluator before making any method claim.

## 2026-07-06 FlowTTE NF MVTec AD1 Full 15-Class Evaluation

- Status: completed
- Protocol: fmad-experiment-protocol, full-dataset diagnostic lane
- Target dataset: MVTec AD1 classic single-image
- Remote AD1 source: local `/home/hun/Volume/DATA/MVTecAD` copied into
  container `/workspace/data/MVTecAD`
- Remote container: `hun_fsad_tta`
- Host GPU: `3`; in-container CUDA slot: `0`
- Objects: all 15 MVTec AD1 classes
- Split: full `test/good` plus all defect-type test folders
- Few-shot setting: 4 train/good support images per object, seed 0
- Method: FlowTTE NF projection with memory expansion enabled
- Remote result:
  `/workspace/results_remote/flow_tte_nf_mvtecad1_all15_shot4_20260706_v1_pixelpro`
- Local pullback:
  `results/remote_runs/dsba3/flow_tte_nf_mvtecad1_all15_shot4_20260706_v1_pixelpro`
- Metrics: `image_AUROC=0.9696312263`, `image_AP=0.9838773037`,
  `pixel_AUROC=0.9640275682`, `pixel_AP=0.5761367314`,
  `pixel_PRO=0.9364702334`, `seg_AUROC_0.05=0.8426347705`,
  `seg_F1=0.5839155076`.
- `pixel_PRO`: AU-PRO integrated over FPR `[0, 0.30]`.
- SuperAD baseline source: `BLOCKED_BASELINE`; no same-condition full AD1
  SuperAD artifact was found in FSAD-TTA/FMAD-OOD records.
- Verdict: `BLOCKED_BASELINE`

Next constrained diagnostic:

- Run or port a same-condition SuperAD/official-superad AD1 full 15-class
  baseline using the same support images, same DINOv2 preprocessing, same map
  resolution, and the same `image_AUROC`/`image_AP`/`pixel_AUROC`/`pixel_AP`/
  `pixel_PRO`/`seg_AUROC_0.05`/`seg_F1` metric suite.

## 2026-07-06 FlowTTE NF MVTec AD1 VisionAD-Preprocess Shot Sweep

- Status: completed
- Protocol: fmad-experiment-protocol, full-dataset diagnostic lane
- Target dataset: MVTec AD1 classic single-image
- Remote container: `hun_fsad_tta`
- Host GPU: `3`; in-container CUDA slot: `0`
- Objects: all 15 MVTec AD1 classes
- Shot settings: `1`, `2`, `4` train/good support images per object
- Support selection: VisionAD-style seeded random without replacement, seed `1`
- Support transforms: identity, rotations 90/180/270, vertical flip,
  horizontal flip
- Backbone/preprocess alignment: `dinov2_vitl14_reg`, resize `448`, center
  crop `392`, layers `4..18`, mean raw layer tokens then L2 normalize
- Remote results:
  - `/workspace/results_remote/flow_tte_nf_mvtecad1_all15_visionadprep_shot1_20260706_v1`
  - `/workspace/results_remote/flow_tte_nf_mvtecad1_all15_visionadprep_shot2_20260706_v1`
  - `/workspace/results_remote/flow_tte_nf_mvtecad1_all15_visionadprep_shot4_20260706_v1`
- Local pullbacks:
  - `results/remote_runs/dsba3/flow_tte_nf_mvtecad1_all15_visionadprep_shot1_20260706_v1`
  - `results/remote_runs/dsba3/flow_tte_nf_mvtecad1_all15_visionadprep_shot2_20260706_v1`
  - `results/remote_runs/dsba3/flow_tte_nf_mvtecad1_all15_visionadprep_shot4_20260706_v1`
- Mean metrics:
  - 1-shot: `image_AUROC=0.9146122376`, `image_AP=0.9583436853`,
    `pixel_AUROC=0.9082760398`, `pixel_AP=0.2922858364`,
    `pixel_PRO=0.8146454113`, `seg_AUROC_0.05=0.7130165644`,
    `seg_F1=0.3690930279`
  - 2-shot: `image_AUROC=0.9283035242`, `image_AP=0.9603264786`,
    `pixel_AUROC=0.9106212922`, `pixel_AP=0.3061433600`,
    `pixel_PRO=0.8237909435`, `seg_AUROC_0.05=0.7237545457`,
    `seg_F1=0.3856519296`
  - 4-shot: `image_AUROC=0.9370056437`, `image_AP=0.9661652540`,
    `pixel_AUROC=0.9121581905`, `pixel_AP=0.3129367435`,
    `pixel_PRO=0.8296951933`, `seg_AUROC_0.05=0.7282588031`,
    `seg_F1=0.3923758461`
- Compared with the previous non-VisionAD-preprocess 4-shot full run, all mean
  metrics decreased; largest drops were `pixel_AP=-0.263200`,
  `seg_F1=-0.191540`, and `pixel_PRO=-0.106775`.
- SuperAD/VisionAD comparable baseline source: `BLOCKED_BASELINE`; no
  same-condition artifact was found in local records.
- Verdict: `BLOCKED_BASELINE / NEGATIVE_ALIGNMENT_DIAGNOSTIC`

Next constrained diagnostic:

- Do not switch the main FlowTTE NF config to the VisionAD-preprocess setting
  without additional adaptation. If VisionAD alignment is pursued, run a
  controlled ablation separating register backbone, 392 crop, 15-layer fusion,
  and rotation/flip support expansion.

## 2026-07-06 FlowTTE NF MVTec AD1 1-Shot Hyperparameter Search

- Status: completed
- Protocol: fmad-experiment-protocol, full-dataset hyperparameter search lane
- Target dataset: MVTec AD1 classic single-image
- Remote container: `hun_fsad_tta`
- Host GPU: `3`; in-container CUDA slot: `0`
- Objects: all 15 MVTec AD1 classes
- Few-shot setting: 1 train/good support image per object
- Main target tuple from the VisionAD paper table used in this project:
  `AUROC=0.974`, `AUPR/AP=0.990`, `PRO=0.925`
- Search axes:
  - NF score density term: `density_weight={0.25,0.10,0.0}`
  - Distance score: Euclidean vs squared distance
  - Image score aggregation: top fraction sweep
  - Support policy: `first` vs VisionAD seeded random support
  - Support augmentation: identity vs rotations/flips
  - Backbone: `dinov2_vitl14` vs `dinov2_vitl14_reg`
- Best run:
  `/workspace/results_remote/flow_tte_nf_mvtecad1_all15_shot1_hp_dist_tte_rotflip_vitl14reg_topsweep_20260706_v1`
- Local pullback:
  `results/remote_runs/dsba3/flow_tte_nf_mvtecad1_all15_shot1_hp_dist_tte_rotflip_vitl14reg_topsweep_20260706_v1`
- Best configuration:
  `dinov2_vitl14_reg`, `fmad_shorter_edge`, crop `448`, layers
  `5,11,17,23`, `layer_norm_mean`, support transforms
  `identity,rot90,rot180,rot270,flip_vertical,flip_horizontal`,
  `density_weight=0.0`, `expansion_budget=1.25`,
  `density_quantile=0.90`, image top fraction `0.005`.
- Best metrics:
  `image_AUROC=0.980673`, `image_AP=0.991444`,
  `pixel_AUROC=0.958969`, `pixel_AP=0.534091`,
  `pixel_PRO=0.931181`, `seg_AUROC_0.05=0.834212`,
  `seg_F1=0.562890`.
- Verdict: `TARGET_BEATEN_FOR_IMAGE_AUROC_IMAGE_AP_AND_PRO_SINGLE_SEED`

Important caveat:

- The winning comparison is single-seed and uses the project metric suite
  `image_AUROC`, `image_AP`, and `pixel_PRO`. It does not beat VisionAD if the
  target is interpreted as pixel-AUROC/pixel-AP; `pixel_AUROC=0.958969` and
  `pixel_AP=0.534091` remain below that interpretation.
- VisionAD paper results are mean/std over five random seeds. The random support
  seed probe here was stopped after seeds 0 and 1 because both were far below
  the `first` support image policy on image AUROC: `0.950918` and `0.952856`.

Next constrained diagnostic:

- Run the winning config over at least five deterministic support selections
  only if the paper claim must be strict mean-over-seeds. Otherwise use the
  best single-seed config as the current shot-1 method setting.

## 2026-07-07 FlowTTE NF MVTec AD1 Shot Scaling vs VisionAD

- Status: completed
- Protocol: fmad-experiment-protocol, full-dataset shot-scaling lane
- Target dataset: MVTec AD1 classic single-image
- Remote container: `hun_fsad_tta`
- Host GPU: `3`; in-container CUDA slot: `0`
- Objects: all 15 MVTec AD1 classes
- Shot settings: `1`, `2`, `4`
- Method: best 1-shot FlowTTE NF setting extended to 2/4-shot:
  `dinov2_vitl14_reg`, `fmad_shorter_edge`, crop `448`, layers
  `5,11,17,23`, `layer_norm_mean`, rot/flip support expansion,
  `density_weight=0.0`, `distance_weight=1.0`, `expansion_budget=1.25`.
- Local pullbacks:
  - `results/remote_runs/dsba3/flow_tte_nf_mvtecad1_all15_shot1_hp_dist_tte_rotflip_vitl14reg_topsweep_20260706_v1`
  - `results/remote_runs/dsba3/flow_tte_nf_mvtecad1_all15_shot2_hp_dist_tte_rotflip_vitl14reg_topsweep_20260707_v1`
  - `results/remote_runs/dsba3/flow_tte_nf_mvtecad1_all15_shot4_hp_dist_tte_rotflip_vitl14reg_topsweep_20260707_v1`
- Metrics:
  - 1-shot: `image_AUROC=0.980673`, `image_AP=0.991444`,
    `pixel_AUROC=0.958969`, `pixel_AP=0.534091`,
    `pixel_PRO=0.931181`, `seg_AUROC_0.05=0.834212`,
    `seg_F1=0.562890`.
  - 2-shot: `image_AUROC=0.981360`, `image_AP=0.991453`,
    `pixel_AUROC=0.959126`, `pixel_AP=0.530315`,
    `pixel_PRO=0.933734`, `seg_AUROC_0.05=0.834619`,
    `seg_F1=0.562695`.
  - 4-shot: `image_AUROC=0.984649`, `image_AP=0.992106`,
    `pixel_AUROC=0.958491`, `pixel_AP=0.531250`,
    `pixel_PRO=0.934834`, `seg_AUROC_0.05=0.834450`,
    `seg_F1=0.562652`.
- VisionAD comparison on project-level `image_AUROC`/`image_AP`/`pixel_PRO`:
  1-shot wins all three; 2-shot wins AUROC and PRO but loses AP; 4-shot loses
  all three slightly.
- Strict pAUROC caveat: this project's `seg_AUROC_0.05` remains far below
  VisionAD pAUROC values and should not be reported as a win.
- Verdict: `PARTIAL_KEEP_FOR_AD1_1SHOT_IMAGE_METRICS / NO_STRICT_VISIONAD_CLAIM`

## 2026-07-07 FlowTTE NF MVTec AD2 vs SuperAD and SuperADD

- Status: completed
- Protocol: fmad-experiment-protocol, full AD2 diagnostic lane
- Target dataset: MVTec AD2 single-image
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- Remote container: `hun_fsad_tta`
- Host GPU: `3`; in-container CUDA slot: `0`
- Objects: all 8 public MVTec AD2 objects
- Split: `test_public/good,bad`
- Few-shot setting: `16` train/good support images per object
- Support policy: `dinov2_cls_greedy_coreset`
- Backbone/preprocess: `dinov2_vitl14`, AD2 adapter `no_mask_no_rotation`
- Local pullback:
  `results/remote_runs/dsba3/flow_tte_nf_mvtecad2_all8_shot16_coreset_dw0_20260707_v1`
- FlowTTE mean metrics: `seg_AUROC_0.05=0.714929`,
  `seg_F1=0.303336`.
- Same-evaluator recorded SuperAD-16 baseline:
  `seg_AUROC_0.05=0.765802`, `seg_F1=0.385534`; deltas
  `-0.050873` and `-0.082198`, FlowTTE wins 2/8 classes.
- Reported SuperADD TESTpublic reference: `AUROC_0.05=0.839300`,
  `F1=0.626113`; deltas `-0.124371` and `-0.322776`, FlowTTE wins 1/8
  classes in reported-context comparison.
- Cleanup evidence: `cleanup_anomaly_maps=true`; local pullback contains no
  `anomaly_maps` directory.
- Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`

Next constrained diagnostic:

- Do not continue this exact NF AD2 configuration. If AD2 remains the target,
  the next branch should avoid class-conditional threshold/localization tuning
  and instead test a shared structural mechanism with all-object no-harm gates.

## 2026-07-08 FlowTTE DVT Position Denoise Alpha Sweep

- Status: completed
- Protocol: fmad-experiment-protocol, bounded diagnostic lane
- Target dataset: MVTec AD2 single-image
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- Remote container: `hun_fsad_tta_012`
- Host GPUs: `0,1,2`
- Objects: all 8 public MVTec AD2 objects
- Method: Flow-LatentBank no-TTE with DVT-style support position-mean
  artifact subtraction before latent scoring.
- Alpha grid: `0.00` no-DVT, `0.25`, `0.50`, `0.75`, `1.00`.
- Best fixed alpha: `1.00`, `seg_AUROC_0.05=0.825207`,
  `seg_F1=0.468348`; deltas vs no-DVT are `+0.027464` AUROC and
  `+0.030548` F1.
- SuperAD/SuperADD context: alpha `1.00` is above recorded SuperAD-16
  (`0.765802` / `0.385534`) but below reported SuperADD, especially F1
  (`0.839300` / `0.626113`).
- Oracle diagnostic: object-wise alpha selection upper bound is `0.833359`
  AUROC / `0.474177` F1 when selected by AUROC and `0.827210` AUROC /
  `0.476974` F1 when selected by F1.
- Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.
- Report:
  `skill_graph/experiments/2026-07-08_flowtte_dvt_alpha_sweep/report.md`.

Next constrained diagnostic:

- Implement a no-label support-only alpha selector using support compactness or
  held-out support self-consistency across `alpha={0,0.25,0.5,0.75,1.0}`.
- Hard-stop if the selector cannot beat both no-DVT and fixed alpha `1.0` on
  mean AUROC/F1 without increasing object-level harm.

## 2026-07-08 FlowTTE SuperADD-Aligned Settings Run

- Status: completed fallback diagnostic; full H+ alignment blocked by gated HF
  access
- Protocol: fmad-experiment-protocol, settings-alignment diagnostic lane
- Target dataset: MVTec AD2 single-image
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- Remote container: `hun_fsad_tta_012`
- Host GPUs: `0,1,2`
- Objects: all 8 public MVTec AD2 objects
- Method structure preserved: Flow-LatentBank no-TTE with DVT
  `position_mean` alpha `1.0`, NF latent distance + density weight `0.25`.
- Intended full SuperADD-aligned settings: `dinov3_vith16plus`, layers
  `[7,15,23,31]`, backbone resolution `640`, tile patch `640`, overlap `128`,
  image resize factor `0.625`, support brightness range `[0.8,1.2]`.
- Full H+ blocker: `facebook/dinov3-vith16plus-pretrain-lvd1689m` was not
  cached and remote download returned `401 Unauthorized`.
- Executed fallback settings: `dinov3_vitl16`, layers `[5,11,17,23]`, backbone
  resolution `640`, tile patch `640`, overlap `128`, image resize factor
  `0.625`, support brightness range `[0.8,1.2]`.
- Remote result:
  `/home/hunim/Volume/FMAD-OOD-remote/results_remote/flowtte_superadd_preproc_vitl16_all8_20260708_v1`
- Local pullback:
  `results/remote_runs/dsba3/flowtte_superadd_preproc_vitl16_all8_20260708_v1`
- Mean result: `seg_AUROC_0.05=0.762906`, `seg_F1=0.384584`.
- Delta vs previous FlowTTE DVT alpha `1.0`: `-0.062301` AUROC,
  `-0.083764` F1.
- Delta vs reported SuperADD context: `-0.076394` AUROC, `-0.241529` F1.
- Local checks passed: focused `ruff`, focused `basedpyright`, `py_compile`,
  launcher `bash -n`, CLI help, and 22 focused tests.
- Cleanup evidence: chunk runs used `--cleanup-maps`; local pullback contains
  no `anomaly_maps` directory.
- Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`.
- Report:
  `skill_graph/experiments/2026-07-08_flowtte_superadd_aligned_setup/report.md`.

## 2026-07-09 FlowTTE Structured Memory Diagnostics

- Status: completed
- Protocol: fmad-experiment-protocol, diagnostic continuation lane
- Target dataset: MVTec AD2 single-image
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- Remote container: `hun_fsad_tta_012`
- Host GPUs: `0,1,2`
- Implemented:
  - feature-derived memory context sources: `feature_avg3*`,
    `image_feature_mean*`
  - lazy `TorchMemoryBank` context grouping for top-M only
  - `calibration_sample_size` cap for support leave-one-out score calibration
- Main run:
  `flowtte_structmem_image_mean_ch16_topm4_cal4096_all8_20260709_v1`
- Result:
  - H+ DVT baseline: `seg_AUROC_0.05=0.836739`, `seg_F1=0.527427`
  - Structured memory: `seg_AUROC_0.05=0.832426`, `seg_F1=0.524593`
  - Delta: `-0.004313` AUROC, `-0.002835` F1
- Cleanup evidence:
  local and remote completed result roots contain `0` `anomaly_maps`
  directories.
- Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`
- Report:
  `skill_graph/experiments/2026-07-09_flowtte_structured_memory_diagnostics/report.md`

Next constrained diagnostic:

- Do not continue `image_feature_mean_ch16 top-M` as a method candidate.
- Reuse the execution fixes for support-stat score calibration or
  foreground/background score suppression.
- Hard-stop if mean AUROC/F1 does not beat the H+ DVT baseline or if
  `vial`/`fruit_jelly` losses repeat.

## 2026-07-10 Transformer Flow Failure Image Analysis

- Status: completed
- Protocol: fmad-experiment-protocol, architecture-only failure analysis
- Target dataset: MVTec AD2 single-image
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- Remote container: `hun_fsad_tta_012`
- Host GPUs: `0,1,2`
- Compared:
  - MLP fused flow: `normality_mode=fused`, `density_weight=0.0`
  - Transformer Flow: `normality_mode=transformer_flow`, `density_weight=0.0`
- Fixed settings: DINOv3-H+/16, layers `[7,15,23,31]`, DVT
  `position_mean alpha=1.0`, fixed SuperAD-16 support, no-TTE,
  context/register disabled.
- Mean result:
  - MLP dw0: `seg_AUROC_0.05=0.837518`, `seg_F1=0.523904`
  - Transformer dw0: `seg_AUROC_0.05=0.828600`, `seg_F1=0.502237`
  - Delta: `-0.008918` AUROC, `-0.021666` F1
- Image-level conclusion: Transformer Flow reduces bad-image
  GT-vs-background score gap for every class. Positive F1 cases (`fabric`,
  `vial`, `sheet_metal`) are better interpreted as threshold/fragmentation
  side effects, not robust local anomaly-ranking gains.
- Cleanup evidence: generated dense anomaly maps were deleted after
  extracting CSV/JSON and representative panels; pulled metric roots contain
  `0` TIFF maps.
- Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.
- Report:
  `skill_graph/experiments/2026-07-10_flowtte_transformer_failure_image_analysis/report.md`

Next constrained diagnostic:

- Do not continue unconstrained all-token Transformer Flow.
- Test local-contrast-preserving structures instead: local/patch flow or raw
  latent score with class-agnostic object/foreground prior, support-stat score
  calibration, and limited neighborhood-aware distance.
# 2026-07-11 FlowTTE Image-Disjoint Support-Consensus S0

- Status: closed and discarded; `KILL_FOR_CLAIM / NO_CONTINUE`.
- Reduced four-object macro base `0.815615/0.471988`; residual
  `0.815634/0.472022`; delta `+0.000019/+0.000033`.
- `can` regressed, q25 was worse, and shuffled support IDs matched or beat the
  candidate. No decoder or all-eight continuation is justified.
- Implementation, tests, launchers, preregistration, and local/remote raw run
  artifacts were removed. Only the result report remains.
- Report:
  `skill_graph/experiments/2026-07-11_flowtte_support_consensus_s0/report.md`.
# 2026-07-12 MVTec AD1 VisionAD-aligned ViT-B/14-register sweep

- [x] Lock all-15 classic MVTec AD, shots 1/2/4/8/16, seed 1.
- [x] Lock frozen `dinov2_vitb14_reg`, resize 448, center crop 392, grid 28x28.
- [x] Repair AD1 RunConfig drift and expose current frontier structural flags.
- [x] Add requested metric aliases and tie-safe i-AUPRC semantics.
- [x] Add two-GPU shot launcher and local contract tests.
- [x] Resolve dsba5 SSH/preset and fixed container mapping for GPUs 0,1.
- [x] Run remote preflight and the full five-shot sweep.
- [x] Pull, validate, summarize, conclude, and clean regenerable maps.
- Status: completed measurement; verdict `BLOCKED_BASELINE` for strict VisionAD
  superiority because Flow-head training and evaluator geometry are unmatched.
- Report:
  `skill_graph/experiments/2026-07-12_flowtte_mvtecad1_visionad_vitb14reg/report.md`.

# 2026-07-12 Static Flow-LatentBank MVTec AD1 ViT-B/14-register 4-shot

- [x] Lock the historical all-15 AD1 4-shot configuration and two requested
  changes: ViT-B/14-register and no TTE.
- [x] Make shorter-edge DINOv2 extraction honor ViT-B layers 2/5/8/11.
- [x] Add focused layer-contract tests and a fixed remote launcher.
- [x] Run dsba5 model/data/GPU preflight.
- [x] Execute the complete 15-category benchmark.
- [x] Pull, audit, compare, report, and verify dense-map cleanup.
- Result: `0.972100/0.987432/0.973747/0.581160/0.938160` for
  i-AUROC/i-AUPRC/p-AUROC/p-AUPRC/p-AUPRO.
- Static-memory audit: all 15 categories `4096 -> 4096`; exact first-four
  supports; local/remote anomaly-map directories both zero.
- Verdict: `ACCEPT_MEASUREMENT / BLOCKED_BASELINE`.
- Report:
  `skill_graph/experiments/2026-07-12_flow_latentbank_mvtecad1_static_vitb14reg_shot4/report.md`.

# 2026-07-13 FlowTTE quantile-match/guided stack + two-view grid shift

- [x] Implement the four-arm retained-map stack evaluator using the existing
  q4096 quantile matcher and RGB guided-r8 module in the fixed order.
- [x] Add synthetic composition-order, resized-tensor shift/back-alignment,
  and mean/max combiner tests.
- [x] Implement a metrics-only two-view runner with original-bank arm A,
  phase-matched full-refit arm C, shared shifted-query extraction, and exact
  view0 anchor parity stop.
- [x] Add credential-free, idempotent CPU and four-GPU remote controllers.
- [x] Run the full local suite: `418 passed`.
- [ ] Run Task A on dsba3 and verify all three 1e-6 references.
- [ ] Run Task B on dsba3 only after Task A passes; poll through eight objects,
  pull compact artifacts, and print the preregistered gates.
- Status: remote execution blocked in the current managed session before SSH;
  both launch attempts failed with `socket: Operation not permitted`. No remote
  files or retained anchor maps were touched.
- Report:
  `skill_graph/experiments/2026-07-13_flowtte_gridshift_2view/report.md`.

# 2026-07-13 Static Flow-LatentBank MVTec AD1 1/2/8-shot sweep

- [x] Lock the accepted 4-shot ViT-B/14-register, shorter-edge 448, no-TTE
  protocol with shot count as the only intended factor.
- [x] Add a resumable two-GPU launcher and preflight.
- [x] Run 1/2/8 shots over all 15 categories on dsba5 GPUs 0 and 1.
- [x] Pull and audit metrics, supports, static memory, logs, and map cleanup.
- [x] Combine the new results with the same-condition 4-shot anchor and report.
- Result at 8 shots: `0.981751/0.991162/0.975455/0.590661/0.942405` for
  i-AUROC/i-AUPRC/p-AUROC/p-AUPRC/p-AUPRO.
- Verdict: `ACCEPT_MEASUREMENT / BLOCKED_BASELINE`.
- Report:
  `skill_graph/experiments/2026-07-13_flow_latentbank_mvtecad1_static_vitb14reg_shot_sweep/report.md`.

# 2026-07-13 Static Flow-LatentBank MVTec AD1 DVT-alpha diagnostic

- [x] Lock the accepted 4-shot static baseline and alpha-only design.
- [x] Pre-register alpha `{0,0.25,0.5,0.75,1.0}`, identity-path parity, macro
  retention, and class no-harm gates.
- [x] Validate and execute the all-15 sweep on dsba5 GPUs 0 and 1.
- [x] Pull, audit, compare against no-DVT, report, and clean dense maps.
- Result: alpha `0.5` gives the best pixel macros
  (`0.975438/0.583405/0.942154`) but loses `0.005036/0.002020` on image
  AUROC/AUPRC and harms 9--10 of 15 classes on each pixel metric.
- No nonzero alpha passes the pre-registered AD1 all-five retention gate.
- Verdict: `KILL_FIXED_DVT_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.
- Report:
  `skill_graph/experiments/2026-07-13_flow_latentbank_mvtecad1_static_vitb14reg_dvt_alpha_sweep/report.md`.

# 2026-07-13 Static Flow-LatentBank MVTec AD1 component 0/1/2 diagnostic

- [x] Lock the accepted 4-shot DVT-off contract and independent component arms.
- [x] Implement AD1 CLS soft-context wiring and guided-r8 post-hoc adapter.
- [x] Add focused regression tests and resumable dsba5/dsba3 launchers.
- [x] Run density0 and CLS-soft-w10 on dsba5 GPUs 0/1.
- [x] Run identity/guided-r8 as two disjoint shards on dsba5 GPUs 0/1 after
  cancelling the dsba3 wait per the user's updated allocation.
- [x] Pull, audit, aggregate per-class metrics, clean maps, and report.
- Result: density0 is the only retention/positive-gate passer at
  `0.973680/0.987876/0.973087/0.584241/0.938218`; guided-r8 reaches
  `0.971024/0.987666/0.977740/0.621455/0.942004` but misses image-AUROC
  retention by 0.0077 percentage point beyond the allowed loss.
- Verdict:
  `KEEP_DENSITY0_DIAGNOSTIC / KILL_CLS_SOFT_W10 / KILL_FIXED_GUIDED_R8_FOR_AD1_CLAIM / BLOCKED_BASELINE`.
- Preregistration:
  `skill_graph/experiments/2026-07-13_flow_latentbank_mvtecad1_static_vitb14reg_component012/preregistration.md`.
- Report:
  `skill_graph/experiments/2026-07-13_flow_latentbank_mvtecad1_static_vitb14reg_component012/report.md`.

# 2026-07-13 MVTec AD1 density0 + guided-r8 hybrid shot sweep

- [x] Lock the raw-image/refined-pixel two-output contract and same-shot
  1/2/4/8 controls before execution.
- [x] Implement and locally verify the hybrid metric combiner and remote
  two-GPU launcher.
- [x] Execute shots 1/2/4/8 on dsba5 GPUs 0/1.
- [x] Pull, audit, compare by shot/class, clean maps, and report.
- Result: every shot passes retention and positive gates. The 1/2/4/8-shot
  p-AUPRC values are `0.599343/0.614666/0.624378/0.630672`, gains of
  `+4.778/+4.420/+4.322/+4.001` points over same-shot controls; all other
  macro losses stay within 0.10 point.
- Audit: 4-shot raw numeric max-diff `0.0`; exact first-N supports; all class
  banks remain static at `1024/2048/4096/8192`; no catastrophic class loss or
  retained dense-map binary.
- Verdict: `PROMISING_DIAGNOSTIC / BLOCKED_BASELINE`.
- Preregistration:
  `skill_graph/experiments/2026-07-13_mvtecad1_density0_guided_hybrid_shot_sweep/preregistration.md`.
- Report:
  `skill_graph/experiments/2026-07-13_mvtecad1_density0_guided_hybrid_shot_sweep/report.md`.

# 2026-07-13 AD2 default binary morphology

- [x] Promote the fixed class-agnostic `close/fill/erode` profile to the AD2
  FlowTTE runner default.
- [x] Preserve raw continuous `seg_AUROC`, raw best threshold, and raw F1 while
  reporting postprocessed F1 and explicit morphology provenance.
- [x] Keep `--binary-postprocess none` as the matched raw ablation switch.
- [x] Add CLI/default/evaluator regression coverage and run the full suite.
- Contract: 17-pixel line closing over 16 orientations, external-contour fill,
  then one 3x3 erosion, applied only after the raw best threshold.
- Verification: `432 passed`, Python compilation and `git diff --check` pass.

# 2026-07-13 AD2 default RGB guide + binary morphology

- [x] Make half-scale guided-r8 the canonical AD2 continuous-map default.
- [x] Preserve close/fill/erode as the subsequent default binary postprocess.
- [x] Add independent `--rgb-guide none` and `--binary-postprocess none`
  ablation switches and explicit manifest provenance.
- [x] Pin historical AD2 diagnostic launchers to their original unguided map
  contract.
- [x] Add default, raw-ablation, and in-place refinement regression coverage.
- Verification: `436 passed`, CLI help, Python compilation, shell syntax, and
  `git diff --check` pass.

# 2026-07-13 AD2 proposed full-normal SuperADD/oracle thresholds

- [x] Recover the official SuperADD split and threshold constants from its
  released implementation: sorted modulo-8 holdout, p95, factor 1.421.
- [x] Pre-register the unchanged proposed method and two-threshold evaluation.
- [x] Implement disjoint 7/8 prototype + 1/8 calibration scoring, guided
  calibration maps, paired threshold metrics, manifests, and cleanup.
- [x] Verify the full local suite (`443 passed`) and both remote environments.
- [x] Discard pre-metric v1/v2 attempts and freeze the user-specified global
  k=100, mean-tau, ascending-score 100k latent selection.
- [x] Detect and discard the miswired v3 run; verify fused FlowTTE memory
  selector coverage and launch v4 with an explicit 100k stage assertion.
- [x] Execute all eight objects across dsba3 GPUs 0–3 and dsba5 GPUs 0–1.
- [x] Pull, audit, aggregate, report, and verify dense-map cleanup.
- Result: p-AUROC `0.833807`; held-out-normal raw/morph F1
  `0.227669/0.230264`; TESTpublic oracle raw/morph F1
  `0.535016/0.548522`.
- Audit: exact disjoint/exhaustive modulo-8 splits, eight static
  `100000 -> 100000` banks, finite rows, clean logs, and zero retained dense
  map directories.
- Verdict: `ACCEPT_MEASUREMENT / BLOCKED_BASELINE`; continuous ranking nearly
  matches reported SuperADD, but fixed-threshold and oracle mask F1 do not.
- Report:
  `skill_graph/experiments/2026-07-13_flowtte_ad2_fullnormal_superadd_thresholds/report.md`.

# 2026-07-13 AD2 random 16-shot DINOv2-L/14

- [x] Pre-register the all-eight, full-TESTpublic, random seed-0 diagnostic.
- [x] Parameterize the established proposed launcher without changing its
  historical defaults.
- [x] Verify launcher/config and remote DINOv2-L/14 preflight.
- [x] Execute four dsba3 GPU shards, pull results, and audit exact supports,
  static memories, finite metrics, logs, and dense-map cleanup.
- [x] Compare with the existing proposed/full-normal context and record the
  single-seed `BLOCKED_BASELINE` verdict.

Result: random-16 DINOv2-L/14 reaches `0.781110` p-AUROC, `0.371855`
guided raw F1, and `0.430721` guided+morphology F1. Exact seeded supports,
static memories, clean logs, and map cleanup passed for all eight classes.
Report:
`skill_graph/experiments/2026-07-13_flowtte_ad2_random16_dinov2l/report.md`.
# 2026-07-15 MVTec AD2 Shift-Factorized Latent Bank

- [x] Freeze the 16-shot DINOv2-L/14 Basic-default comparison contract.
- [x] Implement robust uncentered low-rank support-1NN residual projection.
- [x] Pass focused projection and FlowTTE regressions (`25 passed`).
- [x] Pass a one-object remote mechanism smoke through projection and scoring.
- [x] Launch six all-object variants on dsba3 GPUs 0--3 and dsba5 GPUs 0--1.
- [ ] Collect pAUROC@0.05/F1 and compute official AU-PRO@0.05 from retained maps.
- [ ] Compare against the exact same-support baseline and record keep/kill verdict.
- Dense maps are temporarily retained only for official AU-PRO@0.05 evaluation.
- Remote roots: `/workspace/results_remote/sflb_ad2_rank_trim_20260715_v1`.
