# Skill Graph Log

- 2026-07-08: Implemented and ran the FlowTTE raw hard-null structural
  diagnostic on all-eight MVTec AD2 using dsba3 `hun_fsad_tta_012` GPUs
  0,1,2. Tested SuperADD-like raw layer-wise tiled NN, raw fused NN, raw fused
  NN plus NF residual, and a support-norm foreground split under the H+
  backbone/DVT/fixed-support setting. Results: raw layer-wise tiled no-DVT
  `0.759979/0.395241`, raw layer-wise tiled DVT `0.771061/0.380378`, raw fused
  DVT `0.829365/0.499606`, raw+NF residual `0.833266/0.510439`, foreground
  raw fused `0.799772/0.416134`. Conclusion: simply copying raw layer-wise
  tiled NN is a strong negative control; NF carries useful residual signal but
  the current H+ DVT NF latent reference `0.836739/0.527427` remains best.
  Verdict: `KILL_FOR_MAIN_CLAIM / CONTINUE_AS_DIAGNOSTIC`.
  Report: `skill_graph/experiments/2026-07-08_flowtte_raw_hardnull/report.md`.
- 2026-07-08: Implemented and ran score-field and layer-wise structural
  diagnostics for the current H+ DVT FlowTTE branch on all-eight MVTec AD2
  using dsba3 `hun_fsad_tta_012` GPUs 0,1,2. Support-position calibration was
  harmful (`support_position_center=0.785133/0.429653`,
  `support_position_zscore=0.701830/0.278144` vs baseline
  `0.836739/0.527427`). Support feature-energy foreground prior was
  baseline-tied (`0.834598/0.526807`) and not robust. Layer-wise score fusion
  also underperformed: no-context `0.828210/0.499110`, CLS topM4 routed
  `0.829923/0.508863`. Conclusion: the current fused normalized multi-layer
  feature is a stabilizer; separate per-layer Flow banks and simple
  support-stat score-field correction are not main performance drivers.
  Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`.
  Reports:
  `skill_graph/experiments/2026-07-08_flowtte_scorefield_structural/report.md`,
  `skill_graph/experiments/2026-07-08_flowtte_layerwise_context_routed/report.md`.
- 2026-07-08: Diagnosed the current FlowTTE structural bottleneck from the
  accumulated component, DVT, register, H+ backbone, SuperADD-style setting,
  and H+ priority experiments. Conclusion: the strongest branch is a good
  DINOv3-H+ + DVT continuous patch ranker, but it lacks a class-agnostic
  foreground/background and score-field calibration mechanism that converts
  ranking into SuperADD-level spatial F1. NF removal, register routing, and
  morphology alone do not solve the gap; weak objects should remain failure
  buckets and no-harm checks, not tuning targets. Analysis:
  `skill_graph/analysis/2026-07-08_flowtte_current_structural_problem_diagnosis.md`.
- 2026-07-08: Ran the H+ priority diagnostic sequence after the backbone-only
  run left a reported SuperADD F1 gap. Priority 1 evaluated threshold
  morphology on saved H+ maps: close/fill/erode improved mean F1 from
  `0.527427` to `0.542316` (`+0.014888`) but still trailed reported SuperADD
  F1 by `-0.083797`. Priority 2 reran all-eight MVTec AD2 with the same H+
  backbone/support/DVT setup but `flow_transform_mode=identity` and
  `density_weight=0.0`; it scored `0.832461` AUROC / `0.524804` F1, slightly
  below the H+ NF latent reference. Conclusion: morphology helps, but NF
  latent projection is not the primary mean-metric bottleneck; weak objects are
  failure buckets for class-agnostic diagnostics, not targets for class-specific
  hyperparameter tuning. Verdict:
  `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.
  Report: `skill_graph/experiments/2026-07-08_flowtte_hplus_priority_sequence/report.md`.
- 2026-07-08: Ran the backbone-only SuperADD alignment diagnostic for the best
  current FlowTTE-DVT branch. The run kept fixed support, no-TTE, DVT
  `alpha=1.0`, no context, and latent NN scoring, while changing only the
  backbone to `dinov3_vith16plus` with layers `[7,15,23,31]`. H+ model files
  were copied as cache artifacts to dsba3 and loaded offline; no HF token was
  copied to the remote container. Result on all 8 MVTec AD2 public objects:
  mean `seg_AUROC_0.05=0.836739`, `seg_F1=0.527427`, improving the previous
  DINOv3-L DVT best by `+0.011532` AUROC and `+0.059079` F1. This nearly
  closes the reported SuperADD AUROC gap (`-0.002561`) but leaves F1 below by
  `-0.098686`, with `can` still F1-collapsed. Verdict:
  `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.
  Report: `skill_graph/experiments/2026-07-08_flowtte_hplus_backbone_only/report.md`.
- 2026-07-07: Wrote the NF-free register usage design. Based on the component
  ablation, direct register retrieval/top-M and register-conditioned scoring
  are deprioritized. The preferred register roles are score calibration,
  patch-feature residualization, shortcut suppression, morphology priors, and
  support coverage diagnostics. First recommended preflight:
  register-conditioned score calibration vs raw patch kNN on
  `fabric,can,wallplugs,vial`. Design:
  `skill_graph/analysis/2026-07-07_register_without_nf_design.md`.
- 2026-07-07: Ran all-eight MVTec AD2 FlowTTE component ablations on dsba3
  `hun_fsad_tta_012` GPUs 0,1,2 using the fixed DINOv3 no-context 16-shot
  support set. The no-context Flow-LatentBank base scored
  `seg_AUROC_0.05=0.797743`, `seg_F1=0.437800`. The best component was
  `CLS soft w10` with `0.805427`/`0.447118`, improving the base by
  `+0.007684` AUROC and `+0.009318` F1. TTE expansion, NF-NLL-only scoring,
  and structural register variants all underperformed the base. Conclusion:
  static no-TTE latent distance plus CLS-conditioned soft retrieval is the
  current positive mechanism; register is at most a secondary morphology
  control. Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.
  Report: `skill_graph/experiments/2026-07-07_flowtte_component_ablation/report.md`.
- 2026-07-07: Ran FlowTTE register failure Phase A analysis on dsba3
  `hun_fsad_tta_012` GPUs 0,1,2 using the exact DINOv3 no-context 16-shot
  support set. CLS had the strongest context separability on 6/8 objects
  (mean bad-good min-distance delta `+0.008771` vs register `+0.001372`).
  Register top-M retained nearest support patches slightly better than CLS
  but did not explain a F1 gain. Register-conditioned NF improved NLL
  bad-good separation on 7/8 objects while still losing segmentation mean in
  the prior structural run. Verdict:
  `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.
  Report: `skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/report.md`.
- 2026-07-07: Designed the next FlowTTE register failure analysis instead of
  starting another hyperparameter sweep. The plan decomposes the previous
  structural-register result into context separability, support-routing
  replacement, latent-distance/NF distortion, and a conditional reduced
  map-morphology audit. Design:
  `skill_graph/analysis/2026-07-07_flowtte_register_failure_analysis_design.md`.
- 2026-07-07: Implemented and ran structural register diagnostics for
  Flow-LatentBank on all 8 MVTec AD2 public objects using dsba3
  `hun_fsad_tta_012` GPUs 0,1,2. Tested `register top-M=4` memory routing
  and `register-conditioned NF` under 16-shot DINOv3 coreset no-TTE settings.
  `register top-M=4` scored `seg_AUROC_0.05=0.798846`,
  `seg_F1=0.434411`; `register-conditioned NF` scored `0.796554`,
  `0.436152`, both below the prior `CLS w10` diagnostic
  (`0.805427`/`0.447118`). Verdict:
  `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.
  Report: `skill_graph/experiments/2026-07-07_flowtte_structural_register/report.md`.
- 2026-07-07: Ran the fixed-reference DINOv3 Flow-LatentBank no-TTE diagnostic
  on all eight MVTec AD2 objects. The run forced `dinov3_vitl16` to use the
  exact DINOv2 SuperAD-16 reference image set and scored
  `seg_AUROC_0.05=0.800727`, `seg_F1=0.437437`. This is effectively tied
  with the DINOv3 CLS coreset run (`+0.002983` AUROC, `-0.000363` F1), but
  clearly above the DINOv2 same-reference run (`+0.038945` AUROC,
  `+0.087046` F1). Conclusion: the DINOv3 gain is mainly backbone/feature
  quality, not support-selection luck. Verdict:
  `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.
  Report: `skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/report.md`.
- 2026-07-07: Ran Flow-LatentBank no-TTE at SuperAD's 16-shot MVTec AD2
  budget and added a DINOv3 backbone diagnostic. The closest SuperAD-aligned
  DINOv2 run (`dinov2_vitl14`, DINOv2 CLS coreset 16) scored
  `seg_AUROC_0.05=0.761782`, `seg_F1=0.350391`, nearly matching recorded
  SuperAD-16 AUROC but below its F1 (`0.765802`/`0.385534`). The DINOv3
  diagnostic improved to `0.797743`/`0.437800`, above SuperAD-16 context but
  still below reported SuperADD (`0.839300`/`0.626113`). Verdict:
  `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.
  Report: `skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_shot16_dinov2_dinov3/report.md`.
- 2026-07-07: Consolidated all Flow-family MVTec AD2 results into a single
  summary table with key settings, SuperAD-16 same-evaluator context, and
  reported SuperADD TESTpublic context. Current conclusion: no strict AD2
  method claim; static `Flow-LatentBank no-TTE` is the strongest retained
  Flow branch, but it remains below SuperAD-16 and SuperADD on all8 means.
  Analysis: `skill_graph/analysis/2026-07-07_flow_mvtecad2_results_summary.md`.
- 2026-07-07: Ran the Flow-LatentBank no-TTE ablation on all eight MVTec AD2
  public objects using dsba3 host GPUs `0,1,2` in `hun_fsad_tta_012`.
  With `dinov2_vitl14_reg`, 4-shot first support, and static memory
  (`expansion_budget=1.0`), no-TTE beat the same-condition TTE baseline
  (`expansion_budget=1.25`) by `+0.021340` mean AUROC and `+0.031300` mean
  F1, winning 6/8 objects on both metrics. Verdict: `PROMISING_DIAGNOSTIC`.
  Report: `skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_all8_vitl14reg/report.md`.
- 2026-07-07: Ran FlowTTE-LatentBank with `dinov2_vitl14_reg` on MVTec AD2
  `can,rice` 4-shot and paired it against a current-code `dinov2_vitl14`
  control. Register improved mean AUROC/F1 from `0.765758`/`0.315586` to
  `0.771840`/`0.333811`, but the gain was concentrated on `rice` while `can`
  got worse. Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.
  Report: `skill_graph/experiments/2026-07-07_flowtte_latentbank_vitl14reg_can_rice/report.md`.
- 2026-07-07: Kept the method name **FlowTTE** and recorded the next
  methodology branch: move from frozen-NF latent memory expansion to NF-NLL
  scoring with conservative test-time adaptation. Internal distinction:
  `FlowTTE-LatentBank` for the current prototype and `FlowTTE-NFScore-TTA`
  for the next branch.
  Note: `skill_graph/methods/flowtte_vnext_nf_scoring_tta.md`.
- 2026-07-07: Implemented config-driven `score_mode=latent_distance|nf_nll`
  and ran the first `FlowTTE-NFScore` static raw-NLL diagnostic on MVTec AD2
  `can,rice` 4-shot. Mean AUROC/F1 was `0.674336`/`0.127093`, below the
  previous `FlowTTE-LatentBank` reduced baseline by `-0.091436` AUROC and
  `-0.188495` F1. Verdict: `KILL_FOR_CLAIM / NO_CONTINUE_STATIC_NLL`.
  Report: `skill_graph/experiments/2026-07-07_flowtte_nfscore_static_can_rice/report.md`.
- 2026-07-06: Ran FlowTTE NF 4-shot reduced diagnostic on dsba3
  (`hun_fsad_tta`, host GPU 3) for MVTec AD2 `can,rice`.
  Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.
  Report: `skill_graph/experiments/2026-07-06_flow_tte_nf_can_rice_shot4/report.md`.
- 2026-07-06: Ran FlowTTE NF 4-shot reduced diagnostic on dsba3
  (`hun_fsad_tta`, host GPU 3) for MVTec AD1 `bottle,hazelnut`.
  Extended metrics rerun added image/pixel AUROC/AP. Verdict:
  `BLOCKED_BASELINE`.
  Report: `skill_graph/experiments/2026-07-06_flow_tte_nf_mvtecad1_bottle_hazelnut_shot4/report.md`.
- 2026-07-06: Ran FlowTTE NF 4-shot full 15-class MVTec AD1 evaluation on
  dsba3 (`hun_fsad_tta`, host GPU 3) with image/pixel AUROC/AP plus
  `pixel_PRO`. Mean metrics: `image_AUROC=0.969631`,
  `image_AP=0.983877`, `pixel_AUROC=0.964028`, `pixel_AP=0.576137`,
  `pixel_PRO=0.936470`. Verdict: `BLOCKED_BASELINE`.
  Report: `skill_graph/experiments/2026-07-06_flow_tte_nf_mvtecad1_all15_shot4_pixelpro/report.md`.
- 2026-07-06: Ran FlowTTE NF 1/2/4-shot full 15-class MVTec AD1 sweep on
  dsba3 (`hun_fsad_tta`, host GPU 3) with VisionAD-aligned backbone and
  preprocessing (`dinov2_vitl14_reg`, `448 -> 392`, layers `4..18`,
  rot/flip support expansion). Mean metrics at 4-shot:
  `image_AUROC=0.937006`, `image_AP=0.966165`,
  `pixel_AUROC=0.912158`, `pixel_AP=0.312937`,
  `pixel_PRO=0.829695`. Verdict:
  `BLOCKED_BASELINE / NEGATIVE_ALIGNMENT_DIAGNOSTIC`.
  Report: `skill_graph/experiments/2026-07-06_flow_tte_nf_mvtecad1_visionadprep_shot_sweep/report.md`.
- 2026-07-06: Ran FlowTTE NF 1-shot full 15-class MVTec AD1 hyperparameter
  search on dsba3 (`hun_fsad_tta`, host GPU 3). Best run used
  `dinov2_vitl14_reg`, `fmad_shorter_edge`, rot/flip support expansion,
  `density_weight=0.0`, and image top fraction `0.005`. Metrics:
  `image_AUROC=0.980673`, `image_AP=0.991444`,
  `pixel_AUROC=0.958969`, `pixel_AP=0.534091`,
  `pixel_PRO=0.931181`. Verdict:
  `TARGET_BEATEN_FOR_IMAGE_AUROC_IMAGE_AP_AND_PRO_SINGLE_SEED`.
  Report: `skill_graph/experiments/2026-07-06_flow_tte_nf_mvtecad1_shot1_hparam_search/report.md`.
- 2026-07-07: Extended the best FlowTTE NF MVTec AD1 config to 2-shot and
  4-shot full 15-class runs on dsba3 (`hun_fsad_tta`, host GPU 3). Against
  VisionAD's reported MVTecAD AUROC/AUPR/PRO tuple, 1-shot wins all three
  project-level counterparts, 2-shot wins AUROC/PRO but loses AP, and 4-shot
  is slightly below on all three. Strict pAUROC remains below VisionAD.
  Report: `skill_graph/experiments/2026-07-07_flow_tte_nf_mvtecad1_shot_scaling/report.md`.
- 2026-07-07: Ran FlowTTE NF 16-shot DINO CLS coreset on all 8 MVTec AD2
  public objects and compared with recorded SuperAD-16 plus reported SuperADD
  TESTpublic. Mean FlowTTE `seg_AUROC_0.05=0.714929`, `seg_F1=0.303336`;
  recorded SuperAD-16 is higher by `+0.050873` AUROC and `+0.082198` F1.
  Reported SuperADD is higher by `+0.124371` AUROC and `+0.322776` F1.
  Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`.
  Report: `skill_graph/experiments/2026-07-07_flow_tte_nf_mvtecad2_superad_superadd/report.md`.
- 2026-07-07: Ran FlowTTE DINOv3 fixed-reference no-TTE register-context
  soft-penalty ablation on all 8 MVTec AD2 public objects using dsba3
  `hun_fsad_tta_012` GPUs 0,1,2. Tested `cls`, `register`, and
  `cls_register` context sources at weights `1.0` and `5.0`. Best diagnostic
  result was `cls`, weight `5.0`: `seg_AUROC_0.05=0.805848`,
  `seg_F1=0.443443`, improving no-context by `+0.005121` AUROC and
  `+0.006006` F1 but remaining below reported SuperADD by `-0.033452` AUROC
  and `-0.182670` F1. Verdict:
  `CONTINUE_DIAGNOSTIC / NOT_SUFFICIENT_FOR_SUPERADD_CLAIM`.
  Report: `skill_graph/experiments/2026-07-07_flowtte_register_context_ablation/report.md`.
- 2026-07-07: Ran FlowTTE DINOv3 `CLS+register`, context weight `5.0`,
  16-shot non-fixed reference diagnostic on all 8 MVTec AD2 public objects
  using dsba3 `hun_fsad_tta_012` GPUs 0,1,2. This removed the fixed
  SuperAD-16 reference JSON and selected support with DINOv3 CLS greedy
  coreset seed 0. Mean `seg_AUROC_0.05=0.800546`, `seg_F1=0.440939`,
  improving the directly comparable non-fixed no-context baseline by
  `+0.002803` AUROC and `+0.003139` F1. It remains below the fixed-reference
  `CLS+register w5` diagnostic by `-0.004054` AUROC and `-0.000902` F1.
  Verdict: `CONTINUE_DIAGNOSTIC`; not strict SuperAD-16 comparable because
  the reference image set differs.
  Report: `skill_graph/experiments/2026-07-07_flowtte_clsreg_w5_nonfixed_reference/report.md`.
- 2026-07-08: Ran the Flow-LatentBank no-TTE DVT position-mean denoise alpha
  sweep on all 8 MVTec AD2 public objects using dsba3 `hun_fsad_tta_012`
  GPUs 0,1,2. Tested alpha `0.25`, `0.50`, `0.75` and compared against
  no-DVT plus the previous alpha `1.00` run. Fixed alpha `1.00` was best by
  mean (`seg_AUROC_0.05=0.825207`, `seg_F1=0.468348`), above no-DVT by
  `+0.027464` AUROC and `+0.030548` F1, but it still harms several objects
  and remains below reported SuperADD in F1. Oracle alpha upper bound was
  `0.833359` AUROC / `0.474177` F1 when selected by AUROC and `0.827210`
  AUROC / `0.476974` F1 when selected by F1. Verdict:
  `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`; next constrained branch is a
  no-label support-only alpha selector.
  Report: `skill_graph/experiments/2026-07-08_flowtte_dvt_alpha_sweep/report.md`.
- 2026-07-08: Ran structural diagnostics for the DVT alpha `1.0` probe on
  all 8 MVTec AD2 public objects using dsba3 `hun_fsad_tta_012` GPUs 0,1,2.
  The probe compacts raw support feature variance (`0.451323` denoised/raw)
  and feature LOO distance (`0.882623`), but expands NF latent LOO distance
  (`1.188963`). Query-side denoising explains most of the score-separation
  gain: final bad-good separation is `0.633344` raw/raw, `0.270657`
  support-only denoise, `0.668832` query-only denoise, and `0.681876`
  both-denoise. Verdict: the mechanism is ranking-separation stabilization,
  not simple NF latent memory volume compression.
  Report: `skill_graph/experiments/2026-07-08_flowtte_dvt_structural_analysis/report.md`.
- 2026-07-08: Ran the FlowTTE SuperADD-style non-structural settings
  diagnostic. Full DINOv3-H+/16 alignment was blocked because
  `facebook/dinov3-vith16plus-pretrain-lvd1689m` is gated and remote HF access
  returned `401 Unauthorized`. The executed fallback used `dinov3_vitl16`,
  layers `[5,11,17,23]`, backbone resolution `640`, 640/128 tiled extraction,
  `0.625` resize factor, `[0.8,1.2]` support brightness, and DVT alpha `1.0`
  on dsba3 `hun_fsad_tta_012` GPUs 0,1,2. Mean result:
  `seg_AUROC_0.05=0.762906`, `seg_F1=0.384584`, below the previous DVT alpha
  `1.0` run (`0.825207`/`0.468348`) and reported SuperADD context
  (`0.839300`/`0.626113`). Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`.
  Report: `skill_graph/experiments/2026-07-08_flowtte_superadd_aligned_setup/report.md`.
