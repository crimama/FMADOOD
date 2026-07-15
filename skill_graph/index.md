# Skill Graph Index

## Methods

- `flowtte_vnext_nf_scoring_tta`: FlowTTE method pivot from frozen-NF latent
  memory expansion to NF-NLL scoring with conservative test-time adaptation.

## Experiments

- `2026-07-13_flowtte_ad2_hplus_guided_r8_morph`: all-eight matched AD2 H+
  evaluation of fixed RGB guided-r8 and default close/fill/erode morphology.
  Mean AUROC/F1 changes from `0.836739/0.542316` to
  `0.837528/0.553859`; the mean gate passes but `can` fails the class harm
  gate with `-0.036604` AUROC.
- `2026-07-10_flowtte_darc_resolution_correspondence`: staged experiment cycle
  testing the upstream `can` failure hypothesis with paired native 16px/8px
  layer-7 sampling, detached coarse correspondence, and the `G0/L0/L1/R1`
  mechanism ladder while retaining the current 672 H+ DVT MLP flow as a coarse
  no-harm anchor. AD2 public is explicitly development/shadow, and P16-random,
  P16-SuperAD-selected, M16-fullpool, and Pfull are kept separate. Status:
  the actual full-public `can` raw-ladder cell is complete. G0/L0/L1/R1
  pAUROC@.05/F1 were `0.667650/0.017797`, `0.547831/0.000488`,
  `0.579249/0.000827`, and `0.550352/0.001647`; R1 lost `0.117298` pAUROC
  and `0.016150` F1 versus G0 despite `5/5` registrations and only `2.37%`
  fallback. The preceding synthetic Gate 1 passed and supports unrestricted
  high-resolution G0, not the hard-local endpoint. Compact metrics/audits were
  retained and raw TIFF cleanup was verified. Frozen verdict:
  `KILL_FOR_CLAIM / CONTRACT_CONFLICT`; operationally stop raw R1.
- `2026-07-10_flowtte_hplus_dvt_superad_rotation8`: matched all-eight
  diagnostic that changes only identity support to the exact prior SuperAD
  rotations while retaining H+ fused features, pooled position-mean DVT, and
  the MLP flow. Rotation-8 scored `0.819251/0.506201` versus the capped
  identity control `0.836389/0.527370`, with severe `sheet_metal` and
  `fabric` regressions. Exact rotation parity, full-batch-equivalent row
  microbatching, manifest/path provenance, and runtime cleanup were verified.
  Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`.
- `2026-07-10_flowtte_transformer_context_and_hparam_sweep`: completed
  DINOv3 CLS/register prefix Transformer Flow diagnostics and started
  class-agnostic extreme tuning for the retained H+ DVT FlowTTE structure.
  Register/CLS prefix variants stayed below the H+ DVT reference. Stage-1
  tuning found `lambda_logdet=1e-2` as best F1 (`0.836202/0.528875`) and
  support brightness `0.80,1.20` as best AUROC (`0.838230/0.528203`) versus
  reference `0.836739/0.527427`. Phase-2 partial best is
  `lambda_logdet=2e-2` (`0.835673/0.529874`); phase-3 is queued around
  `lambda_logdet ~= 2e-2`. Verdict:
  `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.
- `2026-07-09_flowtte_transformer_flow_diagnostics`: implemented
  register-free Transformer coupling flow for FlowTTE and ran H+ DVT fixed
  support smoke plus all-eight MVTec AD2. Smoke improved
  `can,vial,fabric` mean, but all-eight fell to `0.828600/0.502237` versus
  H+ DVT baseline `0.836739/0.527427`. Positive object-level signal survived
  on `vial`, `sheet_metal`, and `fabric`; broad no-harm failed. Verdict:
  `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.
- `2026-07-09_flowtte_conv2d_flow_diagnostics`: reduced-object structural
  diagnostic for register-free 2D Conv Flow under the DINOv3-H+ DVT fixed
  support setting. On `can,vial,fabric`, Conv2D flow scored mean
  `0.744949/0.354137` vs the same-object H+ DVT baseline
  `0.758338/0.377474`, losing on every object. Verdict:
  `KILL_FOR_CLAIM / NO_CONTINUE`.
- `2026-07-09_flowtte_structured_memory_diagnostics`: structured memory
  diagnostics for FlowTTE on all-eight MVTec AD2. Adds feature-derived memory
  contexts, lazy context grouping, and calibration sample cap. Image-level
  top-M sub-bank routing scored `0.832426/0.524593`, slightly below the H+
  DVT baseline `0.836739/0.527427`. Verdict:
  `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.
- `2026-07-09_flowtte_patch_structure_flow_diagnostics`: all-eight MVTec AD2
  structural diagnostics for patch-independent FlowTTE NF using DINOv3-H+ and
  DVT `alpha=1.0`. Completed CLS-conditioned NF (`0.832374/0.512126`),
  foreground/background flow mixture (`0.834712/0.519250`), and local contrast
  (`0.806200/0.438345`); none beats the H+ reference
  `0.836739/0.527427`. Coordinate-conditioned `xy` and `cls_xy` are
  runtime-blocked. Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`.
- `2026-07-09_flowtte_object_prior_score_calibration`: stronger
  class-agnostic object/foreground prior and support-score calibration
  diagnostic on the retained H+ DVT NF branch. RGB object prior
  `0.778452/0.374071`, RGB-feature product `0.767204/0.346625`,
  support-score reliability `0.828178/0.505659`, and RGB+reliability
  `0.780889/0.376619` all stayed below the H+ DVT NF reference
  `0.836739/0.527427`. Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`.
- `2026-07-08_flowtte_raw_hardnull`: raw hard-null structural diagnostic
  under the H+ DVT fixed-support setting. SuperADD-like raw layer-wise tiled
  NN underperformed badly (`0.759979/0.395241` no-DVT,
  `0.771061/0.380378` DVT). Raw fused DVT was closer
  (`0.829365/0.499606`), and raw+NF residual improved it
  (`0.833266/0.510439`) but still stayed below the H+ DVT NF latent reference
  (`0.836739/0.527427`). Verdict:
  `KILL_FOR_MAIN_CLAIM / CONTINUE_AS_DIAGNOSTIC`.
- `2026-07-08_flowtte_scorefield_structural`: all-eight H+ DVT FlowTTE
  score-field structural diagnostic. Support-position calibration was harmful
  (`0.785133/0.429653` center, `0.701830/0.278144` z-score vs baseline
  `0.836739/0.527427`), and support feature-energy foreground prior was only
  baseline-tied (`0.834598/0.526807`). Verdict:
  `KILL_FOR_CLAIM / NO_CONTINUE`.
- `2026-07-08_flowtte_layerwise_context_routed`: layer-wise Flow-LatentBank
  diagnostic with score-level layer fusion. No-context layer-wise scored
  `0.828210/0.499110`; CLS topM4 routed scored `0.829923/0.508863`, both
  below the fused H+ DVT baseline `0.836739/0.527427`. Verdict:
  `KILL_FOR_CLAIM / NO_CONTINUE`.
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

- `2026-07-11_flowtte_support_consensus_s0`: discarded result of the
  image-disjoint support-consensus falsification on
  `can,fabric,fruit_jelly,rice`. Exact Phase-3 base is
  reproduced (`0.815615/0.471988`); true residual reaches only
  `0.815634/0.472022`, while shuffled support IDs reach
  `0.815648/0.472008`. `can` does not improve, q25 harms macro metrics, flat
  distance reconstruction error is `0.0`, and all map coverage is `1.0`.
  Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`; do not train the gated decoder.
- `2026-07-10_flowtte_darc_experiment_design`: frozen DARC implementation,
  coordinate, support-budget, evaluator, gate, GPU scheduling, and stop-rule
  specification for the resolution/correspondence experiment cycle.
- `2026-07-10_flowtte_session_handoff`: new-session handoff for the current
  FlowTTE line. Summarizes the retained H+ DVT fixed-memory method, register
  and Transformer findings, DVT-lite mechanism, structural diagnostic verdicts,
  stage-1 hparam sweep, phase-2 partial best (`lambda_logdet=2e-2`,
  `0.835673/0.529874`), queued phase-3, remote/container paths, and immediate
  next actions.

- `2026-07-10_flowtte_transformer_failure_image_analysis`: architecture-only
  image-level analysis of `normality_mode=transformer_flow` against matched
  MLP fused flow (`density_weight=0.0`). Transformer Flow lowers mean
  AUROC/F1 and reduces bad-image GT-vs-background score gap across all
  classes, so its positive cases are better interpreted as score-field
  smoothing rather than robust anomaly-ranking improvement.
- `2026-07-09_flowtte_current_method_results_issues`: current FlowTTE method
  summary, result table, and main bottleneck diagnosis. The method is best
  viewed as a strong continuous patch ranker but weak binary mask generator;
  recent NF-conditioning, foreground/background mixture, and local contrast
  structural variants do not close the SuperADD F1 gap.
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
# 2026-07-12

- [Static Flow-LatentBank MVTec AD1 ViT-B/14-register 4-shot report](experiments/2026-07-12_flow_latentbank_mvtecad1_static_vitb14reg_shot4/report.md):
  all-15 first-support measurement with no TTE. Scores
  `0.972100/0.987432/0.973747/0.581160/0.938160`; all category memories remain
  `4096 -> 4096`. Verdict: `ACCEPT_MEASUREMENT / BLOCKED_BASELINE`.
- [Preregistration](experiments/2026-07-12_flow_latentbank_mvtecad1_static_vitb14reg_shot4/preregistration.md)
- [FlowTTE MVTec AD1 VisionAD-aligned ViT-B/14-register report](experiments/2026-07-12_flowtte_mvtecad1_visionad_vitb14reg/report.md):
  complete all-15-category shots 1/2/4/8/16 measurement on dsba5 GPUs 0,1.
  Best pixel result at 16 shots is `0.959495` p-AUROC, `0.425467` p-AUPRC,
  and `0.875318` p-AUPRO. Verdict: `BLOCKED_BASELINE` for strict VisionAD
  comparison because the flow-head training and evaluator are unmatched.
- [Preregistration](experiments/2026-07-12_flowtte_mvtecad1_visionad_vitb14reg/preregistration.md)

# 2026-07-13

- [Few-shot AD2 Basic/Ours+/Ours++ model-ladder plan](analysis/2026-07-13_fewshot_ad2_model_ladder_experiment_plan.md):
  excludes Ours++ Full; fixes Basic as the exact-reference DINOv2-L SuperAD
  comparison, Ours+ as DINOv2-R scaling, and Ours++ as a 16-shot DINOv3-H+
  comparison to reported full-normal SuperADD context. Includes bounded
  hyperparameter gates informed by prior negative sweeps.

- [AD2 random 16-shot DINOv2-L/14 report](experiments/2026-07-13_flowtte_ad2_random16_dinov2l/report.md):
  all-eight seed-0 random-support diagnostic with the proposed AD2 stack
  otherwise frozen. Macro p-AUROC is `0.781110` and guided morphology F1 is
  `0.430721`. Exact support/static-memory audit passed. Verdict:
  `ACCEPT_MEASUREMENT / BLOCKED_BASELINE`; the existing H+ comparison is
  contextual because both backbone and support policy differ.
- [Preregistration](experiments/2026-07-13_flowtte_ad2_random16_dinov2l/preregistration.md)

- [AD2 proposed full-normal SuperADD-style threshold report](experiments/2026-07-13_flowtte_ad2_fullnormal_superadd_thresholds/report.md):
  exact sorted 7/8 prototype + 1/8 threshold split and global k=100
  score-ranked 100k Flow-latent bank across all eight public objects. Macro
  p-AUROC is `0.833807`; held-out-normal morphology F1 is `0.230264`, while
  TESTpublic-oracle morphology F1 is `0.548522`. Verdict:
  `ACCEPT_MEASUREMENT / BLOCKED_BASELINE`.
- [Preregistration](experiments/2026-07-13_flowtte_ad2_fullnormal_superadd_thresholds/preregistration.md)

- AD2 FlowTTE runner default: half-scale RGB `guided-r8` continuous-map
  refinement followed by fixed class-agnostic `closefill_erode` binary
  morphology at the refined map's best threshold. `--rgb-guide none` and
  `--binary-postprocess none` provide independent matched ablation paths.

- [MVTec AD1 density0 + guided-r8 hybrid shot-sweep report](experiments/2026-07-13_mvtecad1_density0_guided_hybrid_shot_sweep/report.md):
  fixed two-output diagnostic using the density0 raw map for image scoring and
  its guided-r8 refinement for pixel metrics at shots 1/2/4/8. All four shots
  pass locked gates; p-AUPRC improves by `+4.00` to `+4.78` points with no
  catastrophic class loss. Verdict: `PROMISING_DIAGNOSTIC / BLOCKED_BASELINE`.
- [Preregistration](experiments/2026-07-13_mvtecad1_density0_guided_hybrid_shot_sweep/preregistration.md)
- [Static Flow-LatentBank MVTec AD1 component 0/1/2 report](experiments/2026-07-13_flow_latentbank_mvtecad1_static_vitb14reg_component012/report.md):
  DVT-off 4-shot isolated density removal, CLS soft retrieval, and RGB
  guided-r8 diagnostic. Density removal alone passes the locked all-five
  retention and positive gates; guided-r8 strongly improves localization but
  narrowly fails image-AUROC retention. Verdict:
  `KEEP_DENSITY0_DIAGNOSTIC / KILL_CLS_SOFT_W10 / KILL_FIXED_GUIDED_R8_FOR_AD1_CLAIM / BLOCKED_BASELINE`.
- [Preregistration](experiments/2026-07-13_flow_latentbank_mvtecad1_static_vitb14reg_component012/preregistration.md)
- [Static Flow-LatentBank MVTec AD1 ViT-B/14-register 1/2/8-shot report](experiments/2026-07-13_flow_latentbank_mvtecad1_static_vitb14reg_shot_sweep/report.md):
  same-condition extension of the accepted 4-shot measurement. All five macro
  metrics increase through 8 shots; best row is
  `0.981751/0.991162/0.975455/0.590661/0.942405`. Verdict:
  `ACCEPT_MEASUREMENT / BLOCKED_BASELINE`.
- [Preregistration](experiments/2026-07-13_flow_latentbank_mvtecad1_static_vitb14reg_shot_sweep/preregistration.md)
- [Static Flow-LatentBank MVTec AD1 DVT-alpha report](experiments/2026-07-13_flow_latentbank_mvtecad1_static_vitb14reg_dvt_alpha_sweep/report.md):
  alpha-only `{0,0.25,0.5,0.75,1.0}` diagnostic on the accepted 4-shot static
  baseline. Alpha 0.5 gives the best pixel macros but fails image retention
  and class no-harm; alpha 1 loses 2.63 i-AUROC points. Verdict:
  `KILL_FIXED_DVT_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.
- [Preregistration](experiments/2026-07-13_flow_latentbank_mvtecad1_static_vitb14reg_dvt_alpha_sweep/preregistration.md)
- [FlowTTE q4096/guided stack and two-view grid-shift smoke](experiments/2026-07-13_flowtte_gridshift_2view/report.md):
  fixed-order module-stack evaluator plus an exact-parity, metrics-only
  two-view diagnostic. Local implementation is verified; remote evidence is
  blocked by the current session's outbound SSH restriction.
# 2026-07-15

- [MVTec AD2 Shift-Factorized Latent Bank rank/trim diagnostic](experiments/2026-07-15_sflb_ad2_rank_trim/report.md):
  shared low-rank test-shift estimation from trimmed support-1NN latent
  residuals; six all-object variants running across dsba3/dsba5.
