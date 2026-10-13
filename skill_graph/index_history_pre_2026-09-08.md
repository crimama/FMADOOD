# Skill Graph Index

- [Common pipeline refactor, 2026-09-08](analysis/2026-09-08_common_pipeline_refactor/report.md):
  active runtime consolidated into `src/fmad`; exact-shot support and evaluation are
  runner-owned, historical method runners retired, final metrics and provenance retained.
  This is an engineering change, not a new benchmark result.

- [PSAR-D1 detector checkpoint report](experiments/2026-08-28_psar_d1_checkpoint/report.md),
  [preregistration](experiments/2026-08-28_psar_d1_checkpoint/preregistration.md),
  [static contract](experiments/2026-08-28_psar_d1_checkpoint/static_contract.json),
  [output schema](experiments/2026-08-28_psar_d1_checkpoint/output_schema.json):
  the q95 operational repair passed and the three canonical object packages
  completed, but `fabric_repeat` exhausted remote storage after 495 raw maps.
  The complete 24-package barrier failed before GT/evaluation, giving ordered
  gate 1 `INVALID_FAIL_CLOSED / CLOSE PSAR-D1`. Because readable TESTpublic maps
  already crossed `revision_budget=0`, no rerun, all-eight confirmation, or
  LoRA branch is authorized.

- [Post-ECO detector-first future direction](analysis/2026-08-28_post_eco_detector_first_future_direction.md):
  a five-stage external-evidence and adversarial-design loop selected PSAR-D1
  as the next bounded non-ECO generation (`0.56 -> 0.56 -> 0.70 -> 0.72 ->
  0.72`). Close five static contract gaps, then run an immutable
  `fabric/vial/walnuts` detector-metric checkpoint against exact SuperAD-16,
  RN-FMLK, an absolute-head identity, and two unpaired hard nulls. LoRA remains
  closed until an unchanged all-eight frozen-head positive plus a separately
  preregistered capacity-gap result.
- [ECO-O1 lifted-J0/G2 operator sentinel report](experiments/2026-08-28_eco_o1_lifted_j0_g2_operator_sentinel/report.md),
  [preregistration](experiments/2026-08-28_eco_o1_lifted_j0_g2_operator_sentinel/preregistration.md),
  [attempt ledger](experiments/2026-08-28_eco_o1_lifted_j0_g2_operator_sentinel/attempts.md):
  the sole authorized sentinel generation ended
  `INVALID_FAIL_CLOSED / END_ECO_FAMILY` after three pre-result invocations.
  The final invocation imported the exact passed G2 N0 bytes but the frozen G2
  validator required the parent artifact's snapshot identity to equal the new
  O1 snapshot identity. No O1 scientific result, detector map, benchmark
  metric, or LoRA path was produced. The detector bridge/checkpoint are not
  authorized, and no further ECO support-only repair is allowed.
- [ECO-G2 Native-Core Stage -1 report](experiments/2026-08-28_eco_g2_nativecore_stage_minus1/report.md),
  [preregistration](experiments/2026-08-28_eco_g2_nativecore_stage_minus1/preregistration.md):
  the unchanged frozen support-only N0/N1 contract passed all 20 committed
  masks and all eight held/readout targets with an exact repeat. Verdict
  `PROMISING_DIAGNOSTIC / OPERATOR_ONLY_SENTINEL_AUTHORIZED / NO_LORA`;
  exactly one operator-only sentinel is permitted before the mandatory
  detector-bridge-or-terminate branch.
- [Post-G2 detector/metric bridge policy](analysis/2026-08-28_eco_post_g2_detector_metric_bridge_policy.md):
  prospective state machine that closes ECO on N0/N1 failure, permits at most
  one operator-only sentinel after N1 success, and then requires a valid
  detector bridge plus a frozen three-object metric checkpoint or family
  closure. The O1 invalid terminal resolved this branch to family closure;
  the frozen policy itself remains unchanged.

- [ECO-Audit v2 Stage -1 report](experiments/2026-08-27_eco_audit_v2_stage_minus1/report.md):
  frozen activity/spatial/operator audit completed on dsba3 GPUs 0--3 with
  four held shards and an exact held-0 repeat. Support source-footprint Q10,
  active-global/active-boundary coordinates, and N5 passed, but the required
  eroded active interior was empty or singleton in 32/32 cells and J0/G2 was
  independently operator-confounded in 32/32. Official verdict
  `UNRESOLVED_OR_INVALID`; research verdict
  `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`. AGIAS and LoRA remain closed.
- [ECO preregistration](experiments/2026-08-27_eco_audit_v2_stage_minus1/preregistration.md)

- [NCFC 실패 분석·문헌 조사·향후 방향](analysis/2026-08-27_ncfc_postmortem_literature_future_direction.md):
  ECO-Audit v2를 즉시 frozen 진단으로 선택하고, genuine active mismatch일
  때만 AGIAS action-identification → shallow sidecar → capacity-gated protected
  LoRA를 여는 falsification-first 연구 사다리.

## Methods

- `Baseline1`: canonical name for the previously retained/default
  DINOv3-H+/16 + DVT patch-local MLP Flow. Full method specification:
  `skill_graph/methods/baseline1.md`. Completed paired-loss v4 artifacts retain
  the legacy arm ID `V0_default` for checksum and schema compatibility.

## Integrated Ledger

- [Source-only shift–anomaly residual separation literature and gate design](analysis/2026-08-25_source_only_shift_anomaly_residual_separation_literature.md):
  surveys paired/group disentanglement, shared/private factorization, concept
  erasure, invariant/equivariant learning, and synthetic-anomaly restoration.
  The three-stage adversarial loop converged at `0.64 -> 0.59 -> 0.46` with
  `DESIGN_STAGE / REVISE`; one CRLP-Predict source-only gate is the only bounded
  next action.

- [Few-shot AD under Distribution Shift: 통합 실험 Ledger](analysis/2026-08-07_fewshot_distribution_shift_integrated_experiment_ledger.md):
  target-specific module이 필요한 논리, requirement R1--R9, closed/blocked
  experiment branch, current no-go gate, 그리고 완료된 Stage -1a1의 제한사항을 한 문서로 통합한다.

- [External Auxiliary Evidence Feasibility Audit](analysis/2026-08-08_external_auxiliary_evidence_feasibility_audit.md):
  MVTec AD2 single-image에서 deployable external auxiliary evidence가 있는지 감사했다.
  public/private filename condition token은 benchmark oracle이라
  `DESIGN_KILL_METADATA_ORACLE`, per-image acquisition metadata는 `NO_METADATA`;
  다음 허용 후보는 named frozen general image-only prior feasibility뿐이다.

## Experiments

- `2026-08-28_eco_o1_lifted_j0_g2_operator_sentinel`: the only authorized O1
  generation ended before scientific output after three immutable, separately
  sealed attempts. Attempt 003 passed `58/58` P0 tests and exact P1 bytes, but
  the frozen G2 parent validator rejected the exact N0 artifact because parent
  G2 snapshot custody cannot equal current O1 snapshot custody. Verdict
  `INVALID_FAIL_CLOSED / END_ECO_FAMILY`; this is an invalid execution closure,
  not a detector no-gain result. Detector bridge, metric checkpoint, all-eight,
  and LoRA remain unopened.
- [Report](experiments/2026-08-28_eco_o1_lifted_j0_g2_operator_sentinel/report.md),
  [Preregistration](experiments/2026-08-28_eco_o1_lifted_j0_g2_operator_sentinel/preregistration.md),
  [Attempt ledger](experiments/2026-08-28_eco_o1_lifted_j0_g2_operator_sentinel/attempts.md)

- `2026-08-28_eco_g2_nativecore_stage_minus1`: valid v5 completed on the fixed
  dsba3 container and GPUs 0--3. N0 native strict-core counts were `76--247`;
  N1 active counts were `80--183`, minimum coverage was `0.9674773123`, and
  held-0 repeat bytes were identical. This resolves the pooled-grid population
  ambiguity only. J0/G2 operator repair, anomaly maps, benchmarks, and LoRA
  remain unproven and closed.
- [Report](experiments/2026-08-28_eco_g2_nativecore_stage_minus1/report.md),
  [Preregistration](experiments/2026-08-28_eco_g2_nativecore_stage_minus1/preregistration.md)

- `2026-08-27_eco_audit_v2_stage_minus1`: no-training `can x J0/G2`
  attribution audit completed through P0/P1/P2, four held shards, and an exact
  repeat. Structural validity and custody passed, but all 32 cells were
  scientifically invalid: required active interior was empty/singleton and the
  image operator was independently unbounded. Active-global and N5 passed
  32/32, supporting one geometry-only feasibility diagnostic but not AGIAS or
  LoRA.
- [Report](experiments/2026-08-27_eco_audit_v2_stage_minus1/report.md),
  [Preregistration](experiments/2026-08-27_eco_audit_v2_stage_minus1/preregistration.md)

- `2026-08-27_ncfc_s1_stage_minus1`: frozen DINOv2 block-11 `can x J0`
  commutation sentinel completed on dsba3 GPUs 0--3. Both readouts passed
  normal translation, RN safety, and all hard-null cells; anomaly correction
  cosine was `0.956--0.993`. Strict residual commutation passed only `3/8`
  cells per composition order and failed mainly on held base 1 outside the
  anomaly mask. Verdict `NORMAL_ONLY_NO_ANOMALY_SIGNAL / END_NCFC`; no LoRA,
  public/test access, or metric reinterpretation is authorized.
- [Report](experiments/2026-08-27_ncfc_s1_stage_minus1/report.md),
  [Preregistration](experiments/2026-08-27_ncfc_s1_stage_minus1/preregistration.md)

- `2026-08-25_crlp_predict_stage_minus1`: source-only CRLP-Predict all-eight
  diagnostic completed after a GPU0 validity sentinel on dsba3 GPUs 0--3. All
  `96/96` cells were valid, but Gate A/B passed `0/96`: direction signals were
  dominated by magnitude/residual-spectrum/generator shortcuts, while learned
  shift correction met neither the 5% RN-tail gain nor any matched-control
  margin. Verdict `KILL_FOR_CLAIM / NO_CONTINUE`; public/test/GT stayed locked.
- [Report](experiments/2026-08-25_crlp_predict_stage_minus1/report.md),
  [Preregistration](experiments/2026-08-25_crlp_predict_stage_minus1/preregistration.md)

- `2026-08-25_cug_rn_stage_minus1`: sequential dsba3 support-only CUG-RN-S1
  run completed the GPU0 can sentinel and authorized four-object panel on GPUs
  0--3. Panel validity/safe-positive/full-causal counts were `4/4`, `4/4`, and
  `0/4`; matched nulls absorbed M0's apparent benefit, so all-eight and public
  access remained locked. Verdict `NEAR_MISS / ONE_BOUNDED_ITERATION`.
- [Report](experiments/2026-08-25_cug_rn_stage_minus1/report.md),
  [Preregistration](experiments/2026-08-25_cug_rn_stage_minus1/preregistration.md)

- `2026-08-24_atac_stage_minus1`: local and dsba3 can support-16 validity
  gates both killed the frozen unclipped ATAC-v1 path before public access.
  Excursion was `0.0096239--0.0162178` on all 16 supports versus the `0.005`
  limit; dsba3 additionally exposed a strict fp32 linear-control failure.
  Verdict `KILL_FOR_CLAIM / NO_CONTINUE`; compact hash custody passed and no
  anomaly maps remain.
- [Report](experiments/2026-08-24_atac_stage_minus1/report.md),
  [Remote preregistration](experiments/2026-08-24_atac_stage_minus1/remote_preregistration.md)

- `2026-08-20_sam_region_refinement_stage1`: completed the frozen SAM
  region-refinement Stage-1 on local RTX 4090 (four Phase-1 objects, five arms,
  564 query maps). Max-lift refinement lost macro AU-PRO `-0.0831` versus B0
  (`0/4` objects positive; `fabric -0.2346`), anomaly-conditioned prompting
  contributed nothing beyond map-agnostic lifting (`N2 > M1`), and the
  component-recall endpoint is structurally insensitive to max-lift operators
  (exact `0.0000` deltas). Anomaly preservation passed (`>=0.97` everywhere).
  Verdict `KILL_FOR_CLAIM / NO_CONTINUE_SAM_MAX_LIFT`; strike 1 for the
  foundation-model-region-prior family. Any redesign must first repair the
  global-oracle measurement contract.
- [Report](experiments/2026-08-20_sam_region_refinement_stage1/report.md),
  [Preregistration](experiments/2026-08-20_sam_region_refinement_stage1/preregistration.md)

- `2026-08-13_transport_quotient_rn_testpublic`: completed all-eight
  full-support TESTpublic transport-quotient RN run. `B0_RN1` ranked first at
  macro pAUROC/F1/AU-PRO `0.749400/0.336069/0.435772`; `M0_topk_orbit` lost
  `-.097088/-.152026/-.145875`, lost pAUROC on every object, and nearly matched
  `N0_shuffled_tangent`. Verdict `VALID_NO_SIGNAL`; research decision
  `KILL_FOR_CLAIM / NO_CONTINUE`.
- [Preregistration](experiments/2026-08-13_transport_quotient_rn_testpublic/preregistration.md),
  [Report](experiments/2026-08-13_transport_quotient_rn_testpublic/report.md)

- `2026-08-12_rn_anchor_gated_local_tangent_testpublic`: preregistered all-eight
  TESTpublic RN-anchor gated tangent run.
  `A0_RNFMLK_final_layer_1nn_exact_anchor` is mandatory hard null;
  `A3_capped_gated_tangent` preserves exact RN anchor/distance and only applies
  support-LOO bounded correction `d_rn*(1-0.25*gate)`. SuperAD16 context is
  budget-incomparable unless support budget matches.
- [Preregistration](experiments/2026-08-12_rn_anchor_gated_local_tangent_testpublic/preregistration.md),
  [Report](experiments/2026-08-12_rn_anchor_gated_local_tangent_testpublic/report.md)

- `2026-08-12_spct_b1_fullsupport_testpublic`: full-support TESTpublic preregistration for
  B1 unconditional tangent against matched raw, tangent shuffle, and
  RN_FMLK_full. Verdict ceiling: `PROMISING_DIAGNOSTIC`; benchmark-comparable
  is false because the support budget is full `train/good`, not SuperAD16.
- [Preregistration](experiments/2026-08-12_spct_b1_fullsupport_testpublic/preregistration.md),
  [Report](experiments/2026-08-12_spct_b1_fullsupport_testpublic/report.md)

- `2026-08-10_cross_image_calibrated_region_completion_stage_minus1`: exact P16
  leave-one-support-image-out tail calibration was valid and cross-machine
  reproducible, but seeded aligned low-pass completion lost AU-PRO/component recall
  versus current DVT (`-.009657/-.083333`) and calibrated max
  (`-.004691/-.080729`). Current-only morphology matched its component recall and
  beat its AU-PRO. Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`.
- [Report](experiments/2026-08-10_cross_image_calibrated_region_completion_stage_minus1/report.md),
  [Preregistration](experiments/2026-08-10_cross_image_calibrated_region_completion_stage_minus1/preregistration.md)

- `2026-08-08_scale_selective_residual_dvt_diagnostic`: fixed Gaussian low-pass DVT repaired the macro
  small top-1 loss, retained `67.48%/71.71%` of medium AP/top-5 benefit, and beat an equal-energy shuffled
  null on medium metrics. Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.
- [Report](experiments/2026-08-08_scale_selective_residual_dvt_diagnostic/report.md),
  [Preregistration](experiments/2026-08-08_scale_selective_residual_dvt_diagnostic/preregistration.md)

- `2026-08-08_baseline1_dvt_representation_scorer_factorial`: four-object DVT × Flow factorial
  repeated with K1/K5 from the same evaluation. K5 retained most DVT pAUROC/F1 effect but reduced
  positive objects from `3/4` to `2/4`. Verdict: `SCORER_DEPENDENT_DVT_ATTRIBUTION /
  EFFECT_SIZE_MOSTLY_RETAINED`; K1 remains default.
- [Report](experiments/2026-08-08_baseline1_dvt_representation_scorer_factorial/report.md),
  [Preregistration](experiments/2026-08-08_baseline1_dvt_representation_scorer_factorial/preregistration.md)

- `2026-08-08_positional_key_full_value_retrieval_stage_minus1k`: same-coordinate full-feature
  top-5 retrieval versus global, rolled-position, and source-permuted nulls. P1 improved tail/Jaccard
  versus global but not wrong-position nulls (`0/4`), and local retention fell to `.8851`
  (`fabric=.6422`). Verdict: `POSITION_OVERCONSTRAINT / NO_CONTINUE`.
- [Report](experiments/2026-08-08_positional_key_full_value_retrieval_stage_minus1k/report.md),
  [Preregistration](experiments/2026-08-08_positional_key_full_value_retrieval_stage_minus1k/preregistration.md)

- `2026-08-08_stable_key_full_value_retrieval_stage_minus1j`: stable channels select exact top-5
  neighbors while full features compute anomaly distance. Full value preserved local/broad defect
  contrast (`1.0572/1.0155`), but relevance passed `1/4` and retrieval mechanism `0/4`.
  Verdict: `HARD_NULL_ABSORBED`.
- [Report](experiments/2026-08-08_stable_key_full_value_retrieval_stage_minus1j/report.md),
  [Preregistration](experiments/2026-08-08_stable_key_full_value_retrieval_stage_minus1j/preregistration.md)

- `2026-08-08_dino_invariant_channel_readout_stage_minus1`: support-paired lower-50% DINO channel
  readout versus identity and score-shuffled null. It reduced held-shift displacement on `4/4`
  objects (`0.6078` versus `1.0052`) but retained only `0.4941/0.5949` local/broad defect contrast
  and clean NN identity `0.6719`. Verdict: `ANOMALY_ABSORPTION`; no trainable diagonal-gate continuation.
- [Report](experiments/2026-08-08_dino_invariant_channel_readout_stage_minus1/report.md),
  [Preregistration](experiments/2026-08-08_dino_invariant_channel_readout_stage_minus1/preregistration.md)

- `2026-08-08_dvt_small_defect_signal_path_attribution`: all-8 ordered signal-path diagnostic.
  Small top-1% coverage is unchanged pre-DVT, falls `-.011667` post-DVT, and remains `-.011532`
  through Flow latent/final; medium gain also first appears at DVT. Verdict: `LOSS_AT_DVT_SUBTRACTION`.
- [Report](experiments/2026-08-08_dvt_small_defect_signal_path_attribution/report.md),
  [Preregistration](experiments/2026-08-08_dvt_small_defect_signal_path_attribution/preregistration.md)

- `2026-08-08_dvt_positional_causal_controls`: current DVT를 equal-norm position shuffle 및
  spatially constant global centering과 비교한 all-8 causal diagnostic. Medium current-minus-shuffled
  AP/top-5 coverage `+.09024/+.08731`, current-minus-global `+.07571/+.07321`로 positional
  specificity는 통과했지만 small top-1 coverage `-.01153`로 safety는 실패했다. Verdict:
  `POSITION_SPECIFIC_MEDIUM_GAIN_WITH_SMALL_DEFECT_COST`; unconditional DVT tuning은 종료한다.
- [Report](experiments/2026-08-08_dvt_positional_causal_controls/report.md),
  [Preregistration](experiments/2026-08-08_dvt_positional_causal_controls/preregistration.md)

- `2026-08-08_dvt_alignment_defect_scale_attribution`: support-only leave-one-out positional
  explainability와 fixed GT component scale로 all-8 DVT 이득을 귀인했다. Medium `1--5%` defect는
  AP/top-5% coverage `+.07571/+.07321`; alignment 상위 eligible group은 `+.10908/+.10830`으로
  두 gate를 통과했다. Small top-1% coverage는 `-.01153`, large는 underpowered. Verdict:
  `ALIGNMENT_SCALE_ATTRIBUTION_SUPPORTED` (diagnostic only); local/dsba5 parity passed.
- [Report](experiments/2026-08-08_dvt_alignment_defect_scale_attribution/report.md),
  [Preregistration](experiments/2026-08-08_dvt_alignment_defect_scale_attribution/preregistration.md)

- `2026-08-08_baseline1_dvt_all8_rank_coverage`: all-eight, bad-image-only threshold-free
  validation of the DVT localization effect. Pixel AP/full AUROC/top-5% coverage improved by
  `+.01064/+.00817/+.00635`; AP/AUROC were positive on `7/8`, coverage on `6/8`, and non-fabric
  macro remained positive. Top-1% concentration was slightly negative. Verdict:
  `GENERAL_LOCALIZATION_PRIOR_SUPPORTED` (diagnostic only); local/dsba5 parity passed.
- [Report](experiments/2026-08-08_baseline1_dvt_all8_rank_coverage/report.md)

- `2026-08-08_baseline1_dvt_condition_attribution`: post-scoring-only condition
  attribution of DVT-off/on Flow arms. Defect-pixel TPR rose by approximately
  `+9.7%p` across regular/photometric/geometric conditions and on `3/4` objects;
  strict shifted-normal FP suppression held on only `1/4`. Verdict:
  `DEFECT_EVIDENCE_GAIN`; DVT is an object-aligned localization prior, not a
  demonstrated distribution-shift corrector.
- [Report](experiments/2026-08-08_baseline1_dvt_condition_attribution/report.md)

- `2026-08-08_external_auxiliary_evidence_feasibility_audit`: read-only FMAD
  audit after the nuisance-orbit design kill. Dataset filenames expose public
  `regular/overexposed/underexposed/shift_*` and private `regular/mixed`, so
  metadata-token routing is forbidden as benchmark-oracle use. No per-image
  sidecar or useful embedded metadata was found; paired acquisition violates
  the current single-image task. Verdict:
  `NO_METADATA / DESIGN_KILL_METADATA_ORACLE / NEXT_FROZEN_IMAGE_ONLY_PRIOR_FEASIBILITY`.
- [Report](experiments/2026-08-08_external_auxiliary_evidence_feasibility_audit/report.md)

- `2026-08-08_nuisance_orbit_anomaly_persistence_design_audit`: FMAD
  negative-evidence intake for the proposed query nuisance-orbit anomaly
  persistence experiment. It retreads the rejected query counterfactual
  component-persistence branch (`evaluator` score `0.32`, `reject`) and the
  same failure basin as transformed support union/patchwise-min
  (`+0.001790` pAUROC, `8.53%` tail reduction, `-0.000049` F1, `NO_CONTINUE`)
  plus rotation-8 support augmentation (`KILL_FOR_CLAIM / NO_CONTINUE`).
  Verdict: `DESIGN_KILL_CLOSED_BRANCH`; no code/GPU run.
- [Report](experiments/2026-08-08_nuisance_orbit_anomaly_persistence_design_audit/report.md)

- `2026-08-07_layer7_fused_split_interface_stage_minus1h`: layer 7에서 고른 fixed 75%
  inlier patch로 fused condition을 계산한 support-only split-interface audit. C0 대비 local/broad
  drift 감소가 객체별 `-0.2--21.8%` / `.4--13.6%`로 25% gate에 미달했고 tail도 세 객체에서
  악화했다. Strict/weak pass `0/4`, parity mismatch 0. Verdict:
  `KILL_FOR_CLAIM / NO_CONTINUE_SPLIT_INTERFACE`.
- [Report](experiments/2026-08-07_layer7_fused_split_interface_stage_minus1h/report.md),
  [Preregistration](experiments/2026-08-07_layer7_fused_split_interface_stage_minus1h/preregistration.md)

- `2026-08-07_layerwise_defect_spillover_stage_minus1g`: fixed DINO layers와 fused
  representation에서 local defect의 receptive-field spillover를 측정한 support-only audit.
  Layer 7은 localization `4/4`였지만 condition contamination `.214--.688`로 feasibility는
  `0/4`; 나머지 layer localization은 `0/4`였다. Local/dsba5 parity mismatch 0. Verdict:
  `KILL_FOR_CLAIM / CONTINUE_LAYER_INTERFACE_DIAGNOSTIC`.
- [Report](experiments/2026-08-07_layerwise_defect_spillover_stage_minus1g/report.md),
  [Preregistration](experiments/2026-08-07_layerwise_defect_spillover_stage_minus1g/preregistration.md)

- `2026-08-07_spatial_consensus_conditioner_stage_minus1f`: full-grid `4x4`
  block consensus가 defect-invariant global condition을 식별하는지 검증한 support-only audit.
  C2 oracle relative error `.51--.63`, local/broad contamination drift `.65--3.95`, object pass
  `0/4`; local/dsba5 parity mismatch 0. Verdict:
  `KILL_FOR_CLAIM / NO_CONTINUE_SPATIAL_CONSENSUS`.
- [Report](experiments/2026-08-07_spatial_consensus_conditioner_stage_minus1f/report.md),
  [Preregistration](experiments/2026-08-07_spatial_consensus_conditioner_stage_minus1f/preregistration.md)

- `2026-08-07_conditional_normalization_transport_stage_minus1e`: query global
  shift를 제거하는 대신 normal support memory를 query domain으로 이동하는 support-only
  assay. Median location(E2)과 median+MAD diagonal transport(E3)는 모두 `0/4`; E2의
  defect-conditioned location drift가 `.218--.807`로 contamination gate를 전부 실패했다.
  Local/dsba5 parity mismatch는 0이다. Naive moment alignment는 중단하며 defect-invariant
  condition identifiability 전에는 Flow Matching/Diffusion/DiT로 확장하지 않는다.
- [Report](experiments/2026-08-07_conditional_normalization_transport_stage_minus1e/report.md),
  [Preregistration](experiments/2026-08-07_conditional_normalization_transport_stage_minus1e/preregistration.md)

- `2026-08-07_score_aligned_residual_attribution_stage_minus1`: two-object compact
  sentinel testing a support-derived rank-4 patch-residual subspace and fixed
  1NN/5NN/15NN/soft-min scorers. Capture margin was only `.053/.072` versus `.10`;
  1NN and 15NN did not improve, 5NN gained only about `2%`, soft-min was object-dependent,
  and NN geometry failed. Verdict: `KILL_FOR_CLAIM / NO_FOUR_OBJECT_CONTINUATION`.
- [Report](experiments/2026-08-07_score_aligned_residual_attribution_stage_minus1/report.md),
  [Preregistration](experiments/2026-08-07_score_aligned_residual_attribution_stage_minus1/preregistration.md)

- `2026-08-07_shared_direction_held_transform_intervention_stage_minus1`: fixed rank-1,
  no-tuning probe of the Stage -1b shared direction on an unseen photometric transform.
  Defect contrast and geometry were retained, but unseen normal 1-NN tail reduction was
  effectively zero and did not beat the random-direction null (`0/4`). Verdict:
  `KILL_FOR_CLAIM / NO_CONTINUE_SHARED_DIRECTION_INTERVENTION`.
- [Report](experiments/2026-08-07_shared_direction_held_transform_intervention_stage_minus1/report.md),
  [Preregistration](experiments/2026-08-07_shared_direction_held_transform_intervention_stage_minus1/preregistration.md)

- `2026-08-07_support_shared_photometric_direction_stage_minus1`: support-only
  cross-image diagnostic independent from spatial smoothness. The shared global
  and smooth photometric transform directions transferred across held normal
  images better than a source-random global-photometric null on all four
  objects. Local/dsba5 compact checksums and parity passed (maximum delta
  `1.41e-6`). This is direction-identifiability evidence only, not a nuisance
  selector or method claim. Verdict: `PROMISING_DIAGNOSTIC / CONTINUE_DESIGN_ONLY`.
- [Report](experiments/2026-08-07_support_shared_photometric_direction_stage_minus1/report.md),
  [Preregistration](experiments/2026-08-07_support_shared_photometric_direction_stage_minus1/preregistration.md)

- `2026-08-07_global_local_conditional_smoothness_stage_minus1`: separately
  preregistered mask-free follow-up to Stage -1a0. It restricts smoothness to
  active DVT-displacement neighbor pairs, preventing unchanged local-defect
  patches from inflating an all-patch average. Global coherence and effective
  energy coverage exceeded defect aggregates on `4/4` objects, but conditional
  smoothness was lower than defect on every object (`0/4`, required `>=3/4`).
  Local GPU 0/dsba5 GPU 3 input ledgers matched and all 11 metrics reproduced
  within tolerance (maximum delta `.0009765625`). Verdict:
  `INCONCLUSIVE_OR_NOT_SUPPORTED / NO_NEW_MODULE / NO_PUBLIC_ACCESS`.
- [Report](experiments/2026-08-07_global_local_conditional_smoothness_stage_minus1/report.md),
  [Preregistration](experiments/2026-08-07_global_local_conditional_smoothness_stage_minus1/preregistration.md)

- `2026-08-07_global_local_coherence_stage_minus1`: four-object fixed-16
  support-only audit of the proposed global-nuisance/local-defect factorisation
  assumption, run deterministically on local GPU 0 and dsba5 GPU 3. Every
  object had larger global than defect displacement coherence and positive
  shift+defect contrast retention (`.819`--`.984` broad), but the preregistered
  spatial-smoothness condition held only for `fabric` (`1/4`, required `>=3/4`).
  Cross-lane parity passed with maximum aggregate delta `6.3e-7`; no
  public/private/test/ground-truth path was accessed. Verdict:
  `INCONCLUSIVE_OR_NOT_SUPPORTED / NO_NEW_MODULE / NO_PUBLIC_ACCESS`.
- [Report](experiments/2026-08-07_global_local_coherence_stage_minus1/report.md),
  [Preregistration](experiments/2026-08-07_global_local_coherence_stage_minus1/preregistration.md)

- `2026-08-06_flow_latent_geometry_stage_minus1`: measurement-only JVP
  diagnostic asking whether the A6 Flow separates proven normal-variation
  directions from defect-like ones, or whether Normal Inflation is a geometric
  identifiability failure of the map itself. Witness discovery accepted six
  directions from 5,832 candidates across nine cases, so `8/9` fell below the
  preregistered minimum of three and the separation margin `M` is blocked;
  where computable it read `1.0046 / 1.0470 / 1.0748` against identity `1.0`.
  Exploratory observations held without exception: random `unsupported` probes
  were amplified more than `defect_proxy` in all `3/3` measurable cases
  (`+0.034`-`+0.039`), `principal` contracted in `9/9` (`0.773`-`0.912`) while
  every other family expanded, effective rank roughly doubled (`102`-`166` to
  `255`-`329`), and principal-direction curvature exceeded defect curvature by
  an order of magnitude. Suggests the 16-shot support carries no test-time
  shift direction, so the whitening-like map amplifies rather than suppresses
  nuisance variation - which would retro-explain WTCF, `crossfit_only`,
  `pca_residual`, and WAFF. Untested threat: witness scarcity may instead
  reflect the inherited `0.85` cosine criterion at ~10^3 dimensions. Verdict:
  `INCONCLUSIVE_WITNESS_SENTINEL / H1_BLOCKED / EXPLORATORY_ONLY`.
- [Report](experiments/2026-08-06_flow_latent_geometry_stage_minus1/report.md),
  [Preregistration](experiments/2026-08-06_flow_latent_geometry_stage_minus1/preregistration.md)

- `2026-08-05_tte_collapse_decomposition`: attempted to attribute the
  2026-07-07 `flow_tte_budget125` verdict (`-0.027955` pAUROC) that closed the
  test-time branch. Pre-run reanalysis of the recorded artifact showed the macro
  loss is `79.5%` one object (`fabric` `-0.177735`), macro median is
  `-0.011959 / -0.000030`, and excluding `fabric` the arm reads
  `-0.006558 / +0.008230`. The preregistered replication gate then failed:
  under Baseline1-core the same reservoir rule gives `fabric` `+0.0086` pAUROC,
  opposite in sign to the historical `-0.177735`, so no mechanism decomposition
  may be read. Contamination proved near-absent (`0.01%`-`0.61%`) and the
  calibration arms were shown to be structurally inert against oracle metrics.
  The confound was found to be DVT: the recovered 2026-07-07 manifest carries no
  `dvt_denoise_mode` key, so that arm ran without DVT, while every Baseline1
  diagnostic arm had it on. With `--dvt-mode none` the `fabric` sign flips to
  `-0.0625`, and the production runner at the manifest configuration reproduces
  the collapse outright (`Δseg_AUROC = -0.182033` versus the recorded
  `-0.177735`). Mechanism: removing only ground-truth defect patches from
  absorption candidates swings `fabric` by `+0.0859`, so defect
  self-confirmation dominates and stream non-stationarity is secondary. Verdict:
  `HISTORICAL_COLLAPSE_REPRODUCED / MECHANISM_ATTRIBUTED_TO_DEFECT_CONTAMINATION
  / NO_METHOD_CLAIM`. The recorded number is not evidence against test-time
  expansion as a class; it is evidence against an unfiltered NLL gate under a
  representation that admits defect patches. Under Baseline1 (DVT on)
  contamination is `0.04%` and expansion is mildly positive. The branch is
  reopened for design work, conditional on any absorption policy carrying an
  explicit defect-rejection mechanism.

- `2026-08-05_flowtte_semantic_atlas_metric`: completed the four-object,
  fixed-16 support-only test of a semantic-cluster-local nuisance metric over
  frozen Baseline1 latents. A1 preserved synthetic defect contrast (`0.9992`)
  and clean geometry but reduced shifted-normal inflation by only `0.6788%`,
  regressed on `walnuts`, and had unstable fold-refit charts (`0.5274` ARI).
  The global rank-4 control captured more shift energy, while
  orthogonalization added no required safety margin over raw local directions.
  Verdict: `KILL_FOR_CLAIM / RETAIN_BASELINE1 / NO_CONTINUE / NO_PUBLIC_ACCESS`.

- `2026-08-04_flowtte_paired_loss_diagnostic`: completed the four-object,
  fixed-16 support-only diagnostic of pair-aligned losses on the strongest
  retained DINOv3-H+ DVT patch-local MLP Flow. V3 reduced shifted-normal
  inflation by `9.77%` across `4/4` objects but retained only `0.8707` macro
  synthetic defect contrast (`.7372` fruit_jelly, `.6683` vial). Normalized
  pairing was safer but weak; random pairing demonstrated destructive
  alignment. All-8 and public access were blocked. Verdict:
  `KILL_FOR_CLAIM / RETAIN_BASELINE1 / NO_PUBLIC_ACCESS`.


- `2026-08-04_shift_conditioned_patch_local_flow`: implemented the
  DINOv3-H+/16 task-register plus bounded residual FiLM conditioning over the
  frozen patch-local MLP Flow and completed the four-object fixed-16
  support-only gate on dsba5 GPU 3. The shift predictor beat zero on all four,
  but correct-register conditioning increased held shifted-normal inflation by
  `20.41%` macro, beat both zero/shuffled controls on `0/4`, failed identity on
  `3/4`, and lost defect contrast on `fruit_jelly/walnuts`. Public/private/test
  access remained zero. Verdict:
  `EARLY_VALIDITY_KILL / KILL_FOR_CLAIM / NO_PUBLIC_ACCESS`.

- `2026-08-04_wtcf_stage_minus1`: implemented the A6-centered WTCF
  support-only falsification and completed the preregistered `can/seed0`
  sentinel on dsba5 GPU 3. Five stable repeated-residual witnesses were found,
  and WTCF passed access, score, proxy, and function-geometry checks, but its
  witness JVP ratios remained `.9236--.9493` rather than reaching the frozen
  `<=.85` target. Only 5 updates were accepted and 59 blocks were rolled back.
  The controller stopped before the remaining 14 cases and before any
  test/label access. Verdict:
  `EARLY_VALIDITY_KILL / NO_FULL_STAGE / NO_PUBLIC_ACCESS`.

- `2026-07-24_same16_dvt_density_factorial`: completed the exact-support
  all-eight A6/DVT-only/density-only/Basic 2×2 on dsba3. DVT accounts for most
  of Basic's pAUROC/F1 macro advantage over A6
  (`+0.012423/+0.005457`) but loses AU-PRO (`-0.008910`), is strongly
  object-heterogeneous, crosses zero under object bootstrap, and triggers
  collapse flags. Density is sub-material and interaction is sub-material.
  Verdict: `MIXED_OR_INCONCLUSIVE_ATTRIBUTION / KILL_FOR_CLAIM`; no shot
  scaling or further DVT/density sweep.

- `2026-07-23_same16_baseline_metric_contract`: frozen all-eight
  Basic/SuperAD-16/RN-FMLK rerun with exact shared support identities/hash and
  original-resolution pAUROC/raw F1/AU-PRO coverage. Final Basic/SuperAD/RN
  metrics are `0.772635/0.761857/0.745312` pAUROC,
  `0.355676/0.368423/0.331950` F1, and
  `0.533408/0.534924/0.425133` AU-PRO. Verdict:
  `KILL_FOR_CLAIM / NO_SHOT_SCALING`.

- `2026-07-23_a6_dominant_layer_public_shadow`: frozen A0/A6/A7/A6-L17/L17
  confirmation plus independent post-run adjudication. A6 must beat its A7
  hard null without any primary-metric collapse, and shot scaling requires
  dominance over both same-run SuperAD and RN-FMLK. A6 failed promotion and
  A6-L17 regressed across all metrics. Verdict:
  `KILL_FOR_CLAIM / RETAIN_BASIC / NO_SHOT_SCALING`.

- `2026-07-19_dinov2l_default_mechanism_stage_m1`: completed the all-eight
  DINOv2-L/14 Basic `[5,11,17,23]` support-only mechanism audit on dsba3 GPUs
  0--3. DVT raised SD patch pAUROC `+0.00985` but lost fixed F1 `-0.09908`,
  inflated shifted-normal tails `+0.22644`, and reduced NN identity. Flow
  without DVT improved SD pAUROC/F1 `+0.02625/+0.03424` on all objects with no
  resolved tail increase; density was immaterial and fixed fusion lost to the
  single-layer envelope. All 130,560 rows, custody hashes, compact checksums,
  and an independent byte-identical reaggregation passed. TTA/query updates
  and public/private access were zero. Verdict:
  `MECHANISM_AUDIT_COMPLETE / KILL_FOR_CLAIM / NO_STAGE_M2`.

- `2026-07-19_gb_cmf_stage_minus1a`: completed the preregistered four-case,
  five-arm support-only GB-CMF diagnostic on dsba3 GPUs 0--3. GB-CMF failed
  joint feasibility in three cases, geometry on all 16 held sources, raw
  noninferiority, hard-null superiority, and seed-direction agreement. Its
  pAUROC delta versus matched nonfactorial was `-0.023865` with a descriptive
  95% interval `[-0.033540,-0.014036]`. The main failure is structural:
  stochastic signed dual terms and a parameter proximal anchor did not bound
  grouped worst-risk or flow/logdet geometry. Verdict:
  `KILL_FOR_CLAIM / NO_PUBLIC_ACCESS / NO_STAGE_MINUS1B`.

- `2026-07-19_waff_hplus_stage_minus1`: completed the valid support-only
  fixed-sum distance/NLL weighting assay. No weight in the frozen 101-point
  simplex jointly satisfied shifted-normal tail and shifted-defect retention
  for any can/rice seed case; minimum violations ranged from `.030534` to
  `.308887`. L2 and public access were stopped. Verdict:
  `REDESIGN_INFEASIBLE_CONSTRAINT_SURFACE / NO_PUBLIC_ACCESS`.

- `2026-07-18_default_shift_sequence`: completed the preregistered all-eight
  default FlowTTE distribution-shift sequence on dsba3 GPUs 0--3 without TTA
  or TESTprivate. Density increased shifted-normal tails; routed support banks
  failed performance/tail/separation gates; guided fallback retained `can`
  harm and collapsed score separation; local-soft nuisance shrinkage preserved
  safety but reduced tail inflation only `5.95%`. All dense maps and isolated
  snapshots were removed. Verdict:
  `TESTPUBLIC_SHADOW_ONLY / NEGATIVE_OR_MIXED_CAUSAL_SIGNAL`.

- `2026-07-18_cfntr_stage_minus1`: completed the all-eight, support-only
  CF-NTR identifiability preflight on dsba3 GPUs 0--3 without reading
  TESTpublic/private. The physical rank-4 tangent is specific (energy capture
  `0.711801`, hard-null margin `0.509387`) and preserves synthetic defect
  evidence (`0.930158`), but stability tails fail (`0.732572 < 0.75`) and
  held-out p95 drift reduction is only `0.262992` versus the frozen `0.65`
  gate. The failure intensifies with depth because an image-mean tangent does
  not model patch-conditioned late-layer drift. Verdict:
  `KILL_FOR_CLAIM / NO_CONTINUE`.

- `2026-07-17_sdig_nodyr_stage_a`: completed the all-eight MVTec AD2
  TESTpublic shadow on dsba3 GPUs 0--3 with exact SuperAD-selected reference-16.
  NODyR scored `0.729019/0.326280` pAUROC/F1, only
  `+0.000773/+0.000602` over the best single macro null and below the frozen
  `+0.005/+0.005` threshold. The object-wise intrinsic envelope beat it in
  all eight objects. Shifted-good score tails inflated strongly and NODyR was
  nearly identical to `state_angle`, so transition direction did not resolve
  acquisition/defect identifiability. All 6,504 generated maps were validated
  and removed. Verdict:
  `TESTPUBLIC_SHADOW_ONLY / NEGATIVE_OR_MIXED_CAUSAL_SIGNAL`.

- `2026-07-17_sdig_nodyr_method_design`: completed a five-stage solution-design
  cycle centered on separating acquisition shift from local defect evidence.
  The selected `SDIG-NODyR` design freezes one cross-layer transition-direction
  scorer, requires it to beat a per-bootstrap direct-state/RD/shuffle null
  envelope as a standalone map, and permits image-disjoint support-only fixed
  calibration only after that raw gate passes. Candidate scores were
  `0.59/0.32/0.65/0.69/0.79`. Verdict:
  `CONDITIONAL_KEEP_DESIGN_ONLY`; empirical validation is pending.

- `2026-07-16_dscr_paia_condition_audit`: completed a preregistered three-object
  paired acquisition audit of frozen G0 versus density D0 on dsba3. Regular
  macro D0-G0 pAUROC/F1 was `-0.022554/-0.024119`; shifted-mean delta was
  `-0.029614/-0.022535`, so the interaction was only
  `-0.007060/+0.001584`. The density loss is already present under regular
  acquisition, led by `vial`, rather than being principally induced by a
  shared distribution shift. Prior pooled results reproduced within `1.1e-6`;
  maps were validated and removed. Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`.

- `2026-07-16_dscr_hres_density_shadow`: completed a five-object, seed-0/fold-0
  MVTec AD2 compact public mechanism diagnostic on dsba3/dsba5. The
  unrestricted high-resolution raw-distance G0 arm scored macro
  `0.565902/0.080168` pAUROC/F1 versus D0 density NLL
  `0.539851/0.068325` and calibrated D0c `0.539858/0.066906`. Density improved
  pAUROC on only `rice` and violated no-harm on `can` and `vial`; all maps were
  validated and removed after compact artifacts were retained. Verdict:
  `KILL_FOR_CLAIM / NO_CONTINUE`.

- `2026-07-15_flowtte_basic_dinov2l_sequential_hparam`: completed the
  sequential DINOv2 ViT-L/14 Basic depth/width, epoch/LR, log-det/brightness,
  and feature-layer ladder on all eight MVTec AD2 public objects. Final
  selection is depth 2, width 1, 3 epochs, LR `2e-4`, log-det `2e-2`, support
  brightness `0.8,1.2`, and layers `[11,15,19,23]`, scoring
  `0.785792/0.440896` guided p-AUROC/F1. Diagnostic only pending matched
  SuperAD/RN-FMLK/hard-null gates.

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

- `2026-08-07_cross_domain_reading_list_for_structure_search`: 비-AD 분야
  reading list for the structure-search framework. Groups ~20 papers against the
  framework's open design gaps: identifiability limits (§2), coherent-nuisance
  separation (§3), NLL-vs-deployed-score mismatch (R9), bi-Lipschitz geometry
  audit (R8), support-conditioned static transform (R7), constrained transport
  (§6 Flow Matching), and conformal tail statistics (R2/R6). Priority three are
  feature collapse / DKL bi-Lipschitz, SNGP distance awareness, and LooC
  selective invariance. Reference material only; no empirical claim.
- [Reading list](analysis/2026-08-07_cross_domain_reading_list_for_structure_search.md)

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

# 2026-08-07

- [Baseline1 k=5 scorer default A/B](experiments/2026-08-07_baseline1_knn5_default_ab/report.md):
  fixed K1-versus-K5 scorer-only comparison on all eight MVTec AD2 public
  objects. K5 reduced the good-pixel extreme tail but changed macro
  pAUROC/F1/AU-PRO by `-0.004222/-0.002285/-0.003457` and failed the pAUROC
  no-harm plus object-collapse gates. Verdict: `RETAIN_K1 / K5_NOT_DEFAULT`.

# 2026-08-08

- [Frozen CLIP prior separability Stage -1i report](experiments/2026-08-08_frozen_clip_prior_separability_stage_minus1i/report.md):
  support-only diagnostic replacing metadata auxiliary variables with a frozen
  pixel-derived CLIP ViT-L/14@336 patch prior versus DINOv3-H+ layer 7. Strict
  and continuation gates were both `0/4`; verdict:
  `KILL_FROZEN_CLIP_PRIOR / NO_CONTINUE`.
- [Preregistration](experiments/2026-08-08_frozen_clip_prior_separability_stage_minus1i/preregistration.md)

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
- `2026-08-04_shift_conditioned_patch_local_flow`: preregistered a
  DINOv3-H+ follow-up to the failed all-token Transformer Flow. A learned
  support-relative shift register may read but not update patch tokens and can
  alter the frozen MLP Flow only through a bounded residual branch. The frozen
  protocol fixes initialization/scorer/shift-target issues and requires a
  support-only held-shift plus synthetic-defect retention pass before the
  reduced TESTpublic smoke.
- [Baseline1 mechanism attribution (2026-08-08)](experiments/2026-08-08_baseline1_mechanism_attribution/report.md)
  — DINOv3-H+ representation × Flow factorial; Flow main-mechanism gate failed.
- [Baseline1 DVT × Flow factorial (2026-08-08)](experiments/2026-08-08_baseline1_dvt_flow_factorial/report.md)
  — DVT main effect and DVT×Flow interaction both pass the reduced-object mechanism gate.

# 2026-08-09

- [Two-band protected-residual DVT diagnostic](experiments/2026-08-09_two_band_protected_residual_dvt/report.md):
  all-eight B0/B1/B2 test of low-band positional correction plus an untouched high residual. Medium gain
  preservation reached `96.44%/95.69%`, but small AP/top-1 and AU-PRO collapsed. Verdict:
  `ANOMALY_ABSORPTION / NO_CONTINUE`.

# 2026-08-10

- [DINOv3 attention-facet Stage -1](experiments/2026-08-10_dino_attention_facets_stage_minus1/report.md):
  valid fixed-support-16 direct Q/K/V assay over `can,fabric,rice,walnuts`
  (600 full public images) on four available GPUs. `V` was the best real facet
  but changed macro pAUROC/F1 by `-0.033298/-0.059992` versus hidden and
  violated `fabric/rice` no-harm. Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`;
  Stage B V-V/QQ/KK propagation and Stage C routing were not authorized.
- [Preregistration](experiments/2026-08-10_dino_attention_facets_stage_minus1/preregistration.md)

# 2026-08-11

- [TCCR Stage -1 compact diagnostic](experiments/2026-08-11_tccr_stage_minus1/report.md):
  exact P16-anchored cross-scale excess diagnostic on local GPU 0 and dsba5 GPU
  3, with dsba3 excluded. TCCR collapsed to macro AU-PRO/pAUROC
  `0.013620/0.491648` versus P16 `0.552158/0.806379`; phase/null validity and
  frozen cross-machine parity failed. Verdict: `KILL_NO_CONTINUE`.
- [DB-FSPC Gate A](experiments/2026-08-11_db_fspc_gate_a/report.md):
  dsba5 GPU 3 support-only Qwen3-VL-2B/vLLM audit. Runtime feasibility passed,
  but graph parsing/stability/equivariance/null gates failed; verdict:
  `GATE_A_FAIL / NO_GATE_B`. It is label-free and emits no anomaly score.
- [Support-Orbit Velocity Mismatch Stage -1](experiments/2026-08-11_support_orbit_velocity_mismatch_stage_minus1/report.md):
  support-only static SOVM diagnostic over local GPU 0 `can/vial/walnuts` and
  dsba5 GPU 3 `fabric`. Aggregate validity passed and the run ended
  `ORACLE_ONLY / NO_CONTINUE`: M0 macro pAUROC was `0.674283`
  (`local=0.741302`, `broad=0.607263`), normal FPR was `0.045711`, and the
  spatial/source-shuffle control remained the only strong positive control.
  Learned velocity/code/time did not survive, but spatial/source correspondence
  is still the best next pivot.
- [Preregistration](experiments/2026-08-11_support_orbit_velocity_mismatch_stage_minus1/preregistration.md)

# 2026-08-12

- [SPCT confidence/calibration/synthetic-gate/key-TTA Stage -1](experiments/2026-08-12_spct_confidence_calibration_tta_stage_minus1/report.md):
  valid paired support16/full-support four-object diagnostic. All four proposed
  selection/calibration/TTA mechanisms were `NO_SIGNAL`, while full support
  improved B0/B1 pAUROC by `+0.064629/+0.080122` with 4/4 wins. Verdict:
  `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`; the absolute-best usable full-support
  B1 arm advances directly to an all-eight TESTpublic evaluation.

# 2026-08-14

- [RNC-PA method discovery](analysis/2026-08-14_rn_preserving_neighborhood_partial_assignment.md):
  five-stage literature/evidence loop after the valid negative TQRN result.
  Promoted `Regional Normal Correspondence Partial Assignment` as
  `DESIGN_STAGE / CONDITIONAL_KEEP`: exact RN remains a lower-bounded unary,
  while source-grouped sparse partial assignment tests regional correspondence
  validity. Only a compact structural-null diagnostic is licensed next.

# 2026-08-15

- [CF-GC-RN Stage -1](experiments/2026-08-15_cross_fitted_geometry_conditioned_rn_stage_minus1/report.md):
  reduced-object full-support TESTpublic test of leave-center-out spatial
  coherence as RN reference selection rather than an additive anomaly penalty;
  immutable rice/vial run launched on dsba5/dsba3 GPU3.
- [Simple-RN representative + DVT TESTpublic](experiments/2026-08-15_simple_rn_representative_dvt_testpublic/report.md):
  TTA-free all-eight exact RN diagnostic for representative support subsets,
  random64, support-only DVT, and position-shuffle DVT null.
- [Patch-Coverage RN TESTpublic](experiments/2026-08-15_patch_coverage_rn_testpublic/report.md):
  TTA-free all-eight exact RN diagnostic that compresses normal memory inside
  16x16 spatial patch cells while preserving the exact 1NN scoring rule.
- [Protected Residual-Consensus RN Stage -1](experiments/2026-08-16_protected_residual_consensus_rn_stage_minus1/report.md):
  protected test-time feature-residual consensus re-ranks frozen RN top-5
  references while preserving the selected raw RN distance and exact fallback;
  final verdict `KILL_FOR_CLAIM / NO_CONTINUE`, with exact RN1 retained.
# 2026-08-16 — Fundamental canonical transport architecture

- [Design analysis](analysis/2026-08-16_fundamental_canonical_transport_architecture.md): FACT-RN, an offline intervention-learned support-conditioned canonical representation. Design-stage `REVISE`; held-object mechanism assay required before implementation promotion.

# 2026-08-24 — FACT-RN Stage -1 sentinel verdict

- [FACT-RN Stage -1 report](experiments/2026-08-17_factorized_canonical_transport_rn_stage_minus1/report.md):
  valid `can` support-only sentinel, but the learned arm was nearly identity-like
  and matched by the part-frame shuffle. Preregistered verdict
  `KILL_FOR_CLAIM / NO_CONTINUE`; no remote/public expansion.

# 2026-08-24 — CIDER-CF synthetic-counterfactual design

- [CIDER-CF design](analysis/2026-08-24_cider_synthetic_counterfactual_design.md):
  leave-target-out, leave-generator-out nuisance-whitened defect covariance and
  one conformalized likelihood-ratio statistic. Five-stage design score `0.88`;
  `DESIGN_STAGE / CONDITIONAL_KEEP`, support-only Stage -1 next.

# 2026-08-24 — MR-JW support-only mechanism assay

- [MR-JW Stage -1 report](experiments/2026-08-24_mr_jw_stage_minus1/report.md):
  exact support-16, four-object, three-resolution joint-witness diagnostic on
  dsba3 GPUs 0--3. Independent scale NN beat MR-JW on all four objects, the
  rowwise best shuffle also won in aggregate, and clean LOO stability failed
  on three objects. Verdict: `MECHANISM_REFUTED / NO_CONTINUE`; no public/GT
  access.

# 2026-08-25 — CUG-RN margin/topology meta-update

- [CUG-RN-S1 support-only panel](experiments/2026-08-25_cug_rn_stage_minus1/report.md):
  valid and safe-positive on `4/4`, but full causal activity `0/4`; verdict
  `NEAR_MISS / ONE_BOUNDED_ITERATION`.
- [CUG-RN-S2-MTP margin/topology panel](experiments/2026-08-25_cug_rn_s2_mtp_stage_minus1/report.md):
  separately named direct-margin plus global-RN topology retry. All four
  objects and 48 rows were valid, but normal margin, anomaly margin, topology,
  and full activity all passed `0/48`; even the source fits satisfied none of
  the four joint constraints. Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`; no
  all-eight/public/GT access.

# 2026-08-26 — Support-adapted residual distillation

- [TARCD-LoRA-S1 preregistration](experiments/2026-08-26_tarcd_lora_stage_minus1/preregistration.md):
  one-seed target-support LoRA diagnostic using paired
  `x,T(x),A(x),T(A(x))`, frozen-teacher canonical targets, explicit RN and
  anomaly-margin auxiliary losses, support-disjoint held bases, matched
  shuffled/random/static controls, and no public/test/GT access.
- [TARCD-LoRA-S1 report](experiments/2026-08-26_tarcd_lora_stage_minus1/report.md):
  valid support-only panel on `4/4` objects and `12/12` cells, but paired
  correction, full preservation, and cell success each `0/12`; mechanism
  `0/4`. M1 underperformed canonical KD on q99 reduction in every cell and
  whole-control repeatability was incomplete. Verdict
  `KILL_FOR_CLAIM / NO_CONTINUE`; public/test/private/GT remained closed.
- [TARCD-LoRA-S2-DCR preregistration](experiments/2026-08-26_tarcd_lora_s2_dcr_stage_minus1/preregistration.md):
  separately named single-seed loss-mechanism successor with direct paired
  correction, anomaly recomposition, slack RN guards, matched synthetic
  ablations, support-disjoint held bases, and no public/test/private/GT access.
- [TARCD-LoRA-S2-DCR report](experiments/2026-08-26_tarcd_lora_s2_dcr_stage_minus1/report.md):
  valid four-object, twelve-cell dsba3 support-only panel. Direct supervision
  improved S1 correction direction and scale, but absolute correction and
  hard-null gates remained `0/12`, synthetic/RN contributions were partial,
  and no object passed. Verdict `KILL_FOR_CLAIM / NO_CONTINUE`; no public or
  additional-seed branch.
- [TARCD-LoRA-S3-VR preregistration](experiments/2026-08-26_tarcd_lora_s3_vr_stage_minus1/preregistration.md):
  first sequential one-change repair of S2, replacing only the singular
  zero-start correction loss with normalized vector Huber under matched
  support-only randomization and unchanged mechanism/null gates.
- [TARCD-LoRA-S3-VR report](experiments/2026-08-26_tarcd_lora_s3_vr_stage_minus1/report.md):
  bounded-gradient vector Huber fixed the S2 conditioning pathology, but the
  preregistered run failed bitwise repeatability and a separately labeled
  four-GPU observational panel missed both held improvement floors, with
  mechanism `0/4`. Verdict `KILL_FOR_CLAIM / NO_CONTINUE`.
- [SCOPE-LoRA Cycle 1 report](experiments/2026-08-26_scope_lora_cycle1/report.md):
  valid four-GPU rank-4 feasibility panel with positive held explained energy
  and normal-tail reduction, but a preregistered anomaly-commutation failure
  kept P1 closed. Verdict
  `INSUFFICIENT_PROJECTOR_SIGNAL / CONTINUE_ONE_DIAGNOSTIC`.
- [SCOPE-LoRA Cycle 2 CEB preregistration](experiments/2026-08-26_scope_lora_cycle2_ceb/preregistration.md):
  one-change score-firewall diagnostic using calibration-support normal gaps,
  held G2 mask pAUROC, spatial-shuffle and static/max-stream hard nulls before
  any oracle-routed LoRA training.
- [SCOPE-LoRA Cycle 2 CEB report](experiments/2026-08-26_scope_lora_cycle2_ceb/report.md):
  normal q99 reduction and synthetic no-harm passed, but the bypass added
  exactly zero held anomaly-ranking value over its projected-teacher stream
  and failed both spatial and static-null gates. Verdict
  `INSUFFICIENT_PROTECTED_SCORE_SIGNAL / END_OUTER_LOOP`.
- [Oracle-Family-Expert LoRA preregistration](experiments/2026-08-27_oracle_family_expert_lora_stage_minus1/preregistration.md):
  three independent J0/J1/J2 DINOv2 QKV-LoRA experts test whether removing
  cross-family averaging is sufficient to beat static correction while
  retaining held synthetic anomalies.
- [Oracle-Family-Expert LoRA report](experiments/2026-08-27_oracle_family_expert_lora_stage_minus1/report.md):
  valid `12/12`-cell dsba3 panel improved the global hard-null margin to
  `-0.0832`, retained nonnegative q99 in `11/12`, and produced the first
  strict synthetic contribution in `1/12`. Anomaly protection failed on
  `can/vial` and progressive synthetic evidence remained insufficient.
  Verdict `INSUFFICIENT_ORACLE_FAMILY_SIGNAL / END_FAMILY_ROUTING_BRANCH`.
- [Protected Asymmetric-PCGrad QKV-LoRA preregistration](experiments/2026-08-27_protected_pcgrad_qkv_lora_stage_minus1/preregistration.md):
  one-sided global PCGrad protects anomaly/recomposition/RN gradients while
  retaining the same shared rank-4 DINOv2 QKV-LoRA, schedule, controls, and
  support-disjoint held-G2 assay.
- [Protected Asymmetric-PCGrad QKV-LoRA report](experiments/2026-08-27_protected_pcgrad_qkv_lora_stage_minus1/report.md):
  valid four-object dsba3 panel found zero P2 conflicts and `0/12` protected
  contribution; shuffled controls confirmed the implementation by producing
  `88/94/52/85` conflicts with protected updates. Global hard-null margin
  regressed to `-0.1442` despite positive q99 reduction. Verdict
  `INSUFFICIENT_PROTECTED_PCGRAD_SIGNAL / END_SYNTHETIC_QKV_LORA_BRANCH`.
- [CRFR-S1 preregistration](experiments/2026-08-27_crfr_s1_stage_minus1/preregistration.md):
  frozen-DINO content/nuisance factorization with crossed recomposition,
  exact-RN deployment, shuffled/zero-code and no-cross controls, held G2, and
  fail-closed support-only gates.
- [CRFR-S1 report](experiments/2026-08-27_crfr_s1_stage_minus1/report.md):
  exact-repeat four-object dsba3 panel was stable but nuisance-code identity
  was unused: causal factorization `0/12`, anomaly isolation `2/12`, synthetic
  contribution `1/12`, and positive hard-null cells `0/12`. Global hard-null
  margin was `-0.3956` despite q99 reduction `+0.0461`; verdict
  `INSUFFICIENT_CRFR_SIGNAL / END_SUPPORT_ONLY_SYNTHETIC_REPRESENTATION_BRANCH`.
- [SIAI-S1 J0 oracle preregistration](experiments/2026-08-27_siai_s1_j0_oracle_stage_minus1/preregistration.md):
  can-only signed-exposure true-action feasibility sentinel with static-mean,
  wrong-sign, shifted-RN, and paired-clean controls under remote-only custody.
- [SIAI-S1 J0 oracle report](experiments/2026-08-27_siai_s1_j0_oracle_stage_minus1/report.md):
  exact-repeat result found genuine action-use and pixel-reconstruction signals
  but failed correction/anomaly cosine gates; positive-EV clipping isolated as
  the next problem. Verdict `INSUFFICIENT_SIAI_SIGNAL / END_SIAI_S1`.
- [SIAI-D1 J0 sign/saturation report](experiments/2026-08-27_siai_d1_j0_sign_diagnostic_stage_minus1/report.md):
  exact-repeat posthoc diagnostic confirmed that zero-clipping negative EV
  restores normal correction (`0.9100`) and RN topology, while positive EV
  collapses under about `30%` clipping. Negative-EV anomaly cosine remained
  `0.8097`, so the frozen verdict is `NO_REVERSIBLE_SIGN_SIGNAL`; explicit
  pixel inversion ends and any successor must target feature commutation.
- [SAEC-LoRA-S1 preregistration](experiments/2026-08-27_saec_lora_stage_minus1/preregistration.md):
  one zero-start rank-4 block-11 MLP LoRA tests cross-base shared-action
  displacement and held-G2 synthetic anomaly commutation against shuffled
  action and no-commutation controls.
- [SAEC-LoRA-S1 report](experiments/2026-08-27_saec_lora_stage_minus1/report.md):
  exact-repeat four-object dsba3 panel found a bounded correct-vs-shuffled
  relation signal, but M1 lost to matched N2 on every object. Macro relation
  gain was `-0.003369`, action identification `0/4`, synthetic contribution
  `0/16`, and base-ID shortcut failure `4/4`. Verdict
  `NO_IDENTIFIABLE_ACTION / END_SAEC_RELATION`; no SAEC sweep or SuperAD claim.

- [2026-09-08 shift-residual full-feature ablation](experiments/2026-09-08_shift_residual_ablation/design.md)
  — frozen A0-A3 contract; implementation pending, no execution started.

- SRA-AD2-ALL8-SAME16-S1-v1: single-seed 20260906 implementation matrix
  ([design](experiments/2026-09-08_shift_residual_ablation/design.md),
  [report](experiments/2026-09-08_shift_residual_ablation/report.md)); 32
  heads, 1084 TESTpublic queries, execution pending.

- 2026-09-08 shift-residual ablation: canonical design/report are in
  INTEGRATION_CHECK for the frozen single-seed 20260906 A0-A3 matrix
  (32 heads, 1084 TESTpublic queries). Existing baselines are reused via
  baseline_reuse_manifest.json; candidate-vs-historic FP16/FP32 rows are
  comparable=false, with historical_pair_valid tracked separately. No
  training/evaluation start is recorded pending READY/launch evidence.

- 2026-09-08 shift-residual ablation is DEPLOYING: critic/local preflight
  ready, remote PID/launch receipt pending. Canonical evaluation is querywise
  prediction -> GT metric update -> immediate feature/map discard; inventory
  for scheduling is allowed, but no global barrier or second metric pass.

- 2026-09-08 run_v2 is RUNNING (feature extraction only; no metrics or
  finished heads yet). Launch receipt:
  local results/remote_runs/dsba1/sra_ablation_20260908/run_v2/launch_receipt.json
  SHA-256 d6a24e01806e80bf76df6550ded4b8bde230e81b162407047af1252982838fc0.
  Remote root and immutable source snapshot are recorded in the canonical
  report; v1 packaging logs are retained. Baseline precision caveat and
  dense_maps=false remain in force.
