# Skill Graph Index

## Methods

- `flowtte_vnext_nf_scoring_tta`: FlowTTE method pivot from frozen-NF latent
  memory expansion to NF-NLL scoring with conservative test-time adaptation.

## Experiments

- `2026-07-08_flowtte_hplus_priority_sequence`: prioritized H+ diagnostics
  after the backbone-only run. Morphology on saved H+ maps improved mean F1
  from `0.527427` to `0.542316`, but still left an F1 gap to reported
  SuperADD. A no-NF identity feature-distance control scored
  `0.832461/0.524804`, slightly below H+ NF latent. Conclusion: postprocess
  helps but is insufficient, and NF projection is not the primary all8
  bottleneck. Remaining weak objects are analysis buckets only; next steps must
  stay class-agnostic. Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.
- `2026-07-08_flowtte_hplus_backbone_only`: backbone-only SuperADD alignment
  diagnostic for the best current FlowTTE-DVT branch. Changing only the
  backbone from `dinov3_vitl16` layers `[5,11,17,23]` to SuperADD's
  `dinov3_vith16plus` layers `[7,15,23,31]` improved mean
  `seg_AUROC_0.05/F1` from `0.825207/0.468348` to `0.836739/0.527427`.
  AUROC is nearly tied with reported SuperADD (`0.839300`), but F1 remains
  below (`0.626113`), so the remaining gap likely comes from
  threshold/morphology/high-resolution/scoring settings. Verdict:
  `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.
- `2026-07-08_flowtte_superadd_aligned_setup`: ran the SuperADD-style
  non-structural settings diagnostic for Flow-LatentBank no-TTE with DVT
  position denoise. Full DINOv3-H+/16 alignment was blocked by gated HF
  access, so the executed fallback used `dinov3_vitl16`, layers
  `[5,11,17,23]`, 640/128 tiled extraction, `0.625` resize factor, and
  `[0.8,1.2]` support brightness. Mean `seg_AUROC_0.05=0.762906`,
  `seg_F1=0.384584`, below the previous DVT alpha `1.0` run. Verdict:
  `KILL_FOR_CLAIM / NO_CONTINUE`.
- `2026-07-08_flowtte_dvt_structural_analysis`: structural diagnosis of the
  DVT-style position-mean denoise probe for Flow-LatentBank no-TTE. Finds
  raw feature compaction but latent NN expansion, and identifies query-side
  score-separation correction as the main mechanism rather than support memory
  volume compression.
- `2026-07-08_flowtte_dvt_alpha_sweep`: DVT-style support position-mean
  denoise alpha sweep for Flow-LatentBank no-TTE on all-eight MVTec AD2.
  Fixed alpha `1.0` improves no-DVT mean AUROC/F1, but object-level harm
  keeps the result diagnostic and motivates a support-only alpha selector.
- `2026-07-07_flowtte_component_ablation`: all-eight MVTec AD2 component
  ablation for FlowTTE/Flow-LatentBank, identifying static no-TTE memory and
  CLS soft context distance as the main positive components while ruling out
  NF-NLL-only and structural register variants as current mean-metric drivers.
- `2026-07-07_flowtte_register_failure_analysis`: Phase A failure-mode
  analysis for DINOv3 register/CLS context in FlowTTE, measuring context
  separability, retrieval retention/inflation, and register-conditioned NF
  density distortion on all-eight MVTec AD2.
- `2026-07-07_flowtte_structural_register`: structural DINOv3 register
  diagnostics for Flow-LatentBank, testing register top-M memory routing and
  register-conditioned NF on all-eight MVTec AD2 16-shot no-TTE.
- `2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3`: Flow-LatentBank
  no-TTE all-eight MVTec AD2 diagnostic using `dinov3_vitl16` with the exact
  DINOv2 SuperAD-16 reference image set, isolating backbone gain from support
  selection.
- `2026-07-07_flow_latentbank_no_tte_shot16_dinov2_dinov3`: Flow-LatentBank
  no-TTE 16-shot all-eight MVTec AD2 comparison against recorded SuperAD-16
  with `dinov2_vitl14`, plus a `dinov3_vitl16` backbone diagnostic against
  reported SuperADD context.
- `2026-07-07_flow_latentbank_no_tte_all8_vitl14reg`: Flow-LatentBank
  no-TTE all-eight MVTec AD2 4-shot ablation using `dinov2_vitl14_reg`,
  compared against same-condition FlowTTE-LatentBank expansion.
- `2026-07-07_flowtte_latentbank_vitl14reg_can_rice`: FlowTTE-LatentBank
  reduced MVTec AD2 `can,rice` 4-shot backbone-control run comparing
  `dinov2_vitl14_reg` against current-code `dinov2_vitl14`.
- `2026-07-06_flow_tte_nf_can_rice_shot4`: FlowTTE NF 4-shot reduced
  diagnostic on MVTec AD2 `can,rice`.
- `2026-07-06_flow_tte_nf_mvtecad1_bottle_hazelnut_shot4`: FlowTTE NF
  4-shot reduced diagnostic on MVTec AD1 `bottle,hazelnut`.
- `2026-07-06_flow_tte_nf_mvtecad1_all15_shot4_pixelpro`: FlowTTE NF
  4-shot full 15-class evaluation on MVTec AD1 with `pixel_PRO`.
- `2026-07-06_flow_tte_nf_mvtecad1_visionadprep_shot_sweep`: FlowTTE NF
  1/2/4-shot full 15-class MVTec AD1 sweep with VisionAD-aligned backbone and
  preprocessing.
- `2026-07-06_flow_tte_nf_mvtecad1_shot1_hparam_search`: FlowTTE NF 1-shot
  full 15-class MVTec AD1 hyperparameter search targeting VisionAD-level
  image AUROC/AP and PRO.
- `2026-07-07_flow_tte_nf_mvtecad1_shot_scaling`: FlowTTE NF 1/2/4-shot
  full 15-class MVTec AD1 scaling comparison against VisionAD reported
  AUROC/AUPR/PRO.
- `2026-07-07_flow_tte_nf_mvtecad2_superad_superadd`: FlowTTE NF 16-shot
  full MVTec AD2 run with same-evaluator SuperAD and reported SuperADD
  comparisons.
- `2026-07-07_flowtte_clsreg_w5_nonfixed_reference`: FlowTTE DINOv3
  `CLS+register` weight-5 all-eight MVTec AD2 diagnostic without the fixed
  SuperAD-16 reference JSON, using DINOv3 CLS greedy coreset support.
- `2026-07-07_flowtte_nfscore_static_can_rice`: FlowTTE-NFScore static raw
  NLL reduced `can,rice` diagnostic on MVTec AD2.

## Analyses

- `2026-07-08_flowtte_current_structural_problem_diagnosis`: current FlowTTE
  structural bottleneck synthesis. The method is now best viewed as a strong
  DINOv3-H+ + DVT continuous patch ranker with weak score-field calibration
  and foreground/background separation. NF removal, register routing, and
  morphology alone do not close the SuperADD F1 gap; next diagnostics must be
  class-agnostic and all-object.
- `2026-07-07_register_without_nf_design`: NF를 제외하고 DINOv3 register를
  활용하는 방향 설계. register를 직접 anomaly map/score로 쓰지 않고 patch
  residualization, score calibration, shortcut suppression, morphology prior,
  support coverage에 쓰는 후보를 정리.
- `2026-07-07_flowtte_register_failure_analysis_design`: planned
  failure-mode analysis for DINOv3 register/CLS context in FlowTTE,
  decomposing context quality, support routing, conditional-NF latent
  distortion, and map morphology before any further tuning.
- `2026-07-07_flow_mvtecad2_results_summary`: consolidated Flow-family MVTec
  AD2 result summary with SuperAD-16 and reported SuperADD comparisons.
