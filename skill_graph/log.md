# Skill Graph Log

# 2026-08-06 — Representation space Normal Inflation & 9-arm Stage -1 analysis documented

- Documented the mathematical and empirical analysis of Normal Inflation caused by distance-based anomaly scoring under domain shift in feature/representation space.
- Analyzed 9 solution paradigms across MVTec AD 2 objects (`can`, `fabric`, `fruit_jelly`):
  - Linear SubspaceAD (`pca_residual`) suffered severe detector metric drops (AU-PRO -0.11 to -0.32p).
  - Shrinkage Mahalanobis distance degraded to raw Euclidean metric due to high-dimensional ill-conditioned covariance.
  - Unconstrained Soft-1NN Risk (`crossfit_only`) caused Anomaly Absorption (Proxy audit FAILED on `can` and `fruit_jelly`).
  - Witness-Tangent Cross-Fit (`wtcf`) passed Proxy Audit safety, reduced normal tail ($q_{995}$), and preserved anomaly sensitivity.
- Report written to `skill_graph/analysis/2026-08-06_normal_inflation_representation_space_stage_minus1_analysis.md`.

# 2026-08-05 — SemanticAtlasMetric stopped support-only

- Completed the preregistered fixed-16, four-object SemanticAtlasMetric
  Stage -1 on dsba5 GPU 3 without test/public/private/GT access.
- The proposed semantic-local orthogonal metric preserved synthetic defect
  contrast (`0.9992` macro) and clean neighbor geometry, but reduced
  shifted-normal inflation by only `0.6788%` against the `10%` gate and
  regressed on `walnuts`.
- Fold-refit semantic clustering was unstable (`0.5274` macro ARI); the
  proposed atlas captured less shift energy than the closed global-rank-4
  comparator, and orthogonalization did not improve defect retention over the
  raw local direction.
- Verdict: `KILL_FOR_CLAIM / RETAIN_BASELINE1 / NO_CONTINUE / NO_PUBLIC_ACCESS`.
- Exact local reaggregation reproduced the normalized remote verdict SHA-256
  `081877b6bdb869217f83c13f921e36a59a3f3525700eda80c39a60046a43f044`.
- Artifacts:
  `skill_graph/experiments/2026-08-05_flowtte_semantic_atlas_metric/`

# 2026-08-04 — FlowTTE paired-loss diagnostic stopped support-only

- Completed the fixed-16, four-object loss assay on dsba5 GPU 3 using the
  strongest retained DINOv3-H+ DVT patch-local MLP Flow. Register, FiLM,
  Transformer, query adaptation, and public/private/test access were excluded.
- V3 (raw pair + NLL weight 0.1) was strongest: `9.77%` macro shifted-normal
  reduction and `4/4` positive objects, but only `0.8707` macro synthetic
  defect retention, including `.7372` on `fruit_jelly` and `.6683` on `vial`.
  V2 reached `2.84%/.8401`; normalized C1 reached `2.19%/.9654`; random C2
  exposed destructive alignment with only `.4424` defect retention.
- A post-run adjudicator correction limited experiment-wide numerical validity
  to the V0 sentinel instead of deliberately bad controls. It changed the
  label from `EARLY_VALIDITY_KILL` to `KILL_FOR_CLAIM` without changing any
  GPU result or candidate gate. All 20 compact hashes passed, the access ledger
  remained entirely false, the remote snapshot was removed, and GPU 3 returned
  idle. The retained/default method is now named `Baseline1`; pursue
  nuisance-subspace-restricted correction from that fixed reference.
  Report:
  `skill_graph/experiments/2026-08-04_flowtte_paired_loss_diagnostic/report.md`.


# 2026-08-04 — Shift-conditioned patch-local Flow stopped support-only

- Implemented the preregistered DINOv3-H+/16 task-register extractor, bounded
  residual FiLM conditioner over a frozen patch-local MLP Flow, paired
  photometric losses, E0/E1/E5 and zero/shuffled controls, fail-closed
  aggregator, and immutable dsba5 GPU-3 controller.
- The v5 four-object fixed-16 mechanism run learned the relative shift proxy on
  every object, but correct-register conditioning worsened shifted-normal raw
  1-NN inflation by `20.41%` macro and beat both null registers on `0/4`.
  Identity failed on `3/4`; synthetic defect retention collapsed to `.5396`
  on `fruit_jelly` and `.6846` on `walnuts` despite a `.9593` macro caused by
  `vial` amplification.
- Post-training logdet cancellation also missed the frozen `1e-5` limit, but
  the independent shift, identity, and retention failures make the negative
  conclusion robust to that numerical tolerance. Verdict:
  `EARLY_VALIDITY_KILL / KILL_FOR_CLAIM / NO_PUBLIC_ACCESS`. All compact hashes
  and four checkpoints passed pullback verification; v1--v5 source snapshots
  were removed. Report:
  `skill_graph/experiments/2026-08-04_shift_conditioned_patch_local_flow/report.md`.

# 2026-08-04 — WTCF Stage -1 stopped at directional validity gate

- Implemented WTCF cross-fit soft 1-NN alignment, repeated normal-residual
  witness discovery, directional JVP controls, function-space rollback, nine
  matched/null arms, and a validity-first dsba5 GPU-3 controller.
- The valid v6 `can/seed0` sentinel found five stable witnesses and passed
  support custody, finite/nonconstant scores, soft-hard proxy, function
  geometry, and training-finite checks. WTCF preserved pair-distance rank
  (`.9983`) and neighbor identity (`.9297`) but accepted only 5 updates and
  rejected 59 blocks. Witness JVP ratios stayed `.9236--.9493`, above the
  preregistered `.85` contraction ceiling.
- The controller issued `EARLY_VALIDITY_KILL_NO_FULL_STAGE`; the remaining 14
  support-only cases and all public/private/test/ground-truth access were
  blocked. Compact v2--v6 evidence was pulled, all v1--v6 snapshots and
  invalid v1--v5 remote results were removed, and GPU 3 returned idle. Report:
  `skill_graph/experiments/2026-08-04_wtcf_stage_minus1/report.md`.

# 2026-07-24 — Same16 and A6 joint v6 completed after evaluator recovery

- Completed all 56 object/arm runs on dsba3. The original controller failed
  closed because Basic/mechanism imported stale `/workspace` evaluation code.
  Preserved maps were re-evaluated with verified isolated-snapshot imports;
  all 40 affected rows passed metric-schema and coverage checks, with 40
  legacy backups retained.
- Basic/SuperAD/RN-FMLK scored `0.772635/0.761857/0.745312` pAUROC,
  `0.355676/0.368423/0.331950` F1, and
  `0.533408/0.534924/0.425133` AU-PRO. Basic gains pAUROC but loses F1 and
  AU-PRO to SuperAD.
- A6 failed promotion; A6-L17 regressed across all metrics. Adjudication:
  `A6_NOT_PROMOTED_RETAIN_BASIC / KILL_FOR_CLAIM_NO_SHOT_SCALING`.
  All 7,588 dense maps and the source snapshot were removed after compact
  pullback and cleanup verification.

# 2026-07-23 — Same16 and A6 joint v6 launched on dsba3

- Implemented the all-eight same-reference Basic/SuperAD/RN-FMLK metric
  contract plus A0/A6/A7/A6-L17/L17 public-shadow confirmation. The evaluator
  now ignores the AD2 `ground_truth` metadata directory, reports
  original-resolution pAUROC/raw F1/AU-PRO with coverage, and validates exact
  support identities plus SHA-256.
- Added independent baseline/mechanism aggregation and post-run adjudication.
  A6 requires both simplification and A7 hard-null gates; L17 repair requires
  no collapse on any primary metric. Random-16 is authorized only after the
  selected detector beats both same-run SuperAD and RN-FMLK across all three
  macro metrics, wins at least 5/8 objects per metric, and has no collapse.
- OpenSSH was installed rootlessly, the existing fixed container
  `hun_fsad_tta_012` passed CUDA/data/support/DINOv2 checks and 27 focused
  tests, and joint run `same16_baseline_metric_contract_20260723_v6` started
  on GPUs 0--3. Results use `/var/tmp/fsad_tta/results_remote` because the
  container overlay has 7.8 TB free while `/workspace` has 57 GB. Last
  observed state was controller PID `283559` alive with 14/56 object/arm
  metrics complete. Verdict remains `BLOCKED_BASELINE` until aggregation and
  adjudication complete.
  Reports:
  `skill_graph/experiments/2026-07-23_same16_baseline_metric_contract/report.md`,
  `skill_graph/experiments/2026-07-23_a6_dominant_layer_public_shadow/report.md`.

# 2026-07-19 — DINOv2-L/14 Basic default mechanism audit completed

- Completed the frozen all-eight C/S/D/SD assay on dsba3 GPUs 0--3 with exact
  support-16, layers `[5,11,17,23]`, static memory, and zero TTA/query/public/
  private access. The valid v4 contains 130,560 compact rows and 8,192 paired
  factorial cases; all object parity checks had zero NN-index mismatches.
- DVT A0-A1 gained `+0.009853` SD patch pAUROC but lost `-0.099084` fixed F1,
  increased q.995 normal-tail inflation `+0.226439`, and reduced NN identity
  `-0.048839`. Flow without DVT gained `+0.026251/+0.034241` SD pAUROC/F1 on
  all eight objects with no resolved tail increase, whereas Flow with DVT
  retained ranking gain but lost F1 retention and inflated tails. Density had
  no material contribution; fused identity features lost `-0.017146/-0.034825`
  SD pAUROC/F1 to the per-object layer envelope.
- Corrected a post-compute access-ledger assertion from 32 total reads to the
  audited 160 total (128 allowlisted content hashes plus 32 decoded images),
  added regression coverage, and reaggregated the unchanged rows. Strict RGB
  custody, all compact hashes, and independent local byte parity passed. The
  v1--v4 isolated snapshots were removed while compact/failure evidence was
  retained. Verdict: `MECHANISM_AUDIT_COMPLETE / KILL_FOR_CLAIM /
  NO_PUBLIC_ACCESS / NO_STAGE_M2`. Report:
  `skill_graph/experiments/2026-07-19_dinov2l_default_mechanism_stage_m1/report.md`.

# 2026-07-19 — DINOv2-L/14 Basic default mechanism audit frozen

- Corrected the audit reference from DINOv3-H+ and the tuned DINOv2 layer-late
  variant to the operational DINOv2-L/14 Basic default: layers
  `[5,11,17,23]`, exact SuperAD-selected 16 supports, static memory, DVT
  position mean, two-layer Flow, distance plus density, and LOO off.
- Froze an all-eight, source-disjoint support-only C/S/D/SD assay with causal
  DVT-side, Flow, density, fused-feature, single-layer, SuperAD, and RN-FMLK
  controls. Intermediate credit requires matched changes in normal-tail,
  neighbor, geometry, and shifted-defect-retention diagnostics.
- TTA, query update/transformation, routing, TESTprivate, metric-selected retry,
  and post-result default tuning are forbidden. No experiment was launched.
- Design:
  `skill_graph/analysis/2026-07-19_dinov2l_default_mechanism_experiment_plan.md`.

- 2026-07-19: Completed valid GB-CMF Stage -1A on dsba3 GPUs 0--3 with four
  frozen can/rice cases and five arms per case, exactly 20 fits. GB-CMF minus
  default had descriptive median `+0.011818/+0.011919` pAUROC/F1 but both 95%
  intervals crossed zero; versus matched nonfactorial, pAUROC was
  `-0.023865` with interval `[-0.033540,-0.014036]`. Three cases were not
  jointly tail/drop feasible, all 16 held sources failed geometry, and both
  objects disagreed across seeds. Tiny parameter moves (`.0031--.0051`)
  produced pair-rank, logdet-share, coupling-logdet, and score-scale failures,
  while the soft/hard proxy passed 15/16 sources. The frozen signed augmented
  term could reward already-satisfied cells, latest-cell duals did not enforce
  grouped worst risk, and geometry was post-hoc rather than optimized. All 67
  compact hashes and local reaggregation passed; each case opened only its 16
  whitelisted support images with zero forbidden access; the remote snapshot
  was removed and GPUs returned idle. Verdict:
  `KILL_FOR_CLAIM / NO_PUBLIC_ACCESS / NO_STAGE_MINUS1B`. Report:
  `skill_graph/experiments/2026-07-19_gb_cmf_stage_minus1a/report.md`.

- 2026-07-19: Completed WAFF-H+ valid v6 support-only feasibility testing on
  the default H+ flow. All four can/rice cases passed runtime/parity audits,
  but no point in the frozen 101-point fixed-sum distance/NLL simplex jointly
  met shifted-normal tail and shifted-defect retention. Minimum joint
  violations were `.030534`, `.033264`, `.308887`, and `.063389`. This closes
  post-flow component reweighting rather than blaming the optimizer. L2 and
  public access were not launched. Verdict:
  `REDESIGN_INFEASIBLE_CONSTRAINT_SURFACE / NO_PUBLIC_ACCESS`. Report:
  `skill_graph/experiments/2026-07-19_waff_hplus_stage_minus1/report.md`.

- 2026-07-18: Completed the four-stage default FlowTTE distribution-shift
  diagnostic on all eight MVTec AD2 public objects using exact SuperAD-16
  support and dsba3 GPUs 0--3, with no TTA, query adaptation, or TESTprivate.
  Stage 1 showed density `-0.000779/+0.003526` pAUROC/F1 versus distance while
  worsening shifted-good inflation; guided gained F1 but harmed `can` AUROC by
  `-0.036604`, and morphology gains were object-concentrated. Stage 2 routed
  support banks changed pAUROC/F1 only `+0.000602/-0.000291`, reduced tail
  inflation `2.71%`, and reduced separation; support LOO route accuracy ranged
  from 0% to 76.79%. Stage 3 fallback gained `0.000848` pAUROC over guided but
  lost `0.004352` F1, retained `can` harm, and reduced separation `0.021247`.
  Stage 4 local-soft preserved macro/no-harm/separation but reduced tail
  inflation only `5.95%`; global-hard reduced it `18.42%` while erasing
  anomaly performance and separation. All compact pulls and checksums passed;
  dense maps and both isolated snapshots were removed. Verdict:
  `TESTPUBLIC_SHADOW_ONLY / NEGATIVE_OR_MIXED_CAUSAL_SIGNAL / KEEP_FROZEN_DEFAULT`.
  Report:
  `skill_graph/experiments/2026-07-18_default_shift_sequence/report.md`.

- 2026-07-18: Completed CF-NTR Stage -1 on dsba3 GPUs 0--3 using only the
  exact SuperAD-selected 16 train/good references per object and sealed
  synthetic shifts/defects. The valid v3 run passed specificity, defect
  retention, and audit gates but failed stability and drift correction:
  physical energy capture was `0.711801`, while held-out p95 drift reduction
  was `0.262992` (95% LCB `0.256818`) and every object stayed below `0.40`.
  Layerwise reduction collapsed from `0.485229` at layer 5 to `0.098899` at
  layer 23, showing that the global image-mean rank-4 tangent does not capture
  spatially conditional late-layer distribution shift. All 136,192 maps were
  finite/nonconstant; checksums and exact coverage passed; dense artifacts and
  the isolated remote snapshot were removed while compact evidence was
  retained. Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`. Report:
  `skill_graph/experiments/2026-07-18_cfntr_stage_minus1/report.md`.

- 2026-07-18: Completed the preregistered SDIG-NODyR Stage A TESTpublic shadow
  on dsba3 GPUs 0--3 using the exact SuperAD-selected 16 normal references and
  full eight-object MVTec AD2 population. NODyR scored macro
  `0.729019/0.326280` pAUROC/F1 versus the best single macro null,
  `state_angle`, at `0.728246/0.325678`; the `+0.000773/+0.000602` margin
  missed the frozen `+0.005/+0.005` signal threshold, and the object-wise null
  envelope beat NODyR on both metrics in all eight objects. Shifted-good p95
  inflated from regular `0.133087` to `0.162501/0.165181/0.168161`, while
  NODyR/state-angle image mean and p95 correlations were
  `0.999887/0.999636`. This identifies the candidate as an almost exact state
  prediction reparameterization that still tracks acquisition shift. All
  1,084 images and 6,504 maps passed coverage/equivalence audits; dense maps
  and compact caches were removed. Verdict:
  `TESTPUBLIC_SHADOW_ONLY / NEGATIVE_OR_MIXED_CAUSAL_SIGNAL`. Report:
  `skill_graph/experiments/2026-07-17_sdig_nodyr_stage_a/report.md`.

- 2026-07-17: Completed a five-stage distribution-shift identifiability design
  cycle after consolidating the position-correction, hard-local, density,
  support-consensus, and condition-aware regressions. Candidate design scores
  were `0.59 -> 0.32 -> 0.65 -> 0.69 -> 0.79`; the selected
  `SDIG-NODyR` path first tests a standalone frozen-backbone depth-transition
  direction map against direct-state/RD/shuffle nulls, and only after a raw
  pass permits image-disjoint support-only fixed calibration. Query-view
  persistence was rejected and optional fusion cannot rescue failed raw
  evidence. Verdict: `CONDITIONAL_KEEP_DESIGN_ONLY`; no empirical method
  improvement is claimed. Report:
  `skill_graph/experiments/2026-07-17_sdig_nodyr_method_design/report.md`.

- 2026-07-15: Completed the sequential DINOv2 ViT-L/14 Basic tuning ladder
  on all eight MVTec AD2 public objects with fixed SuperAD-16 support,
  static latent bank, DVT position-mean alpha 1.0, density 0.25, RGB
  guided-r8, and close/fill/erode morphology. Capacity selected depth 2 /
  width 1 (`0.781937/0.441074`); optimization retained 3 epochs and LR
  `2e-4`; regularization selected `lambda_logdet=2e-2` with support
  brightness `0.8,1.2` (`0.782534/0.439453`); layer selection selected
  `[11,15,19,23]` (`0.785792/0.440896`). Relative to the Stage-1 anchor,
  final p-AUROC improved `+0.003855` while F1 changed `-0.000178`, so the
  result is diagnostic and not a broad segmentation improvement. All remote
  configs completed with zero recorded errors and dense maps cleaned.
  Report: `skill_graph/experiments/2026-07-15_flowtte_basic_dinov2l_sequential_hparam/report.md`.

- 2026-07-13: Completed the all-eight matched AD2 H+ RGB guided-r8 and default
  morphology evaluation. The retained fixed-support Flow-LatentBank + LOO +
  DVT + density control exactly reproduced `0.836739/0.527428` before
  morphology. Guided-r8 + morphology reaches `0.837528/0.553859`, improving
  over matched morphology control by `+0.000789/+0.011544`. The mean gate
  passes, but `can` loses `0.036604` AUROC and fails the class harm gate.
  Dense maps were cleaned locally/remotely and dsba3 GPUs 0--3 returned to
  1 MiB/0%. Verdict:
  `PASS_MEAN_GATE / FAIL_CLASS_HARM_GATE / BLOCKED_BASELINE`. Report:
  `skill_graph/experiments/2026-07-13_flowtte_ad2_hplus_guided_r8_morph/report.md`.
- 2026-07-11: Completed the actual MVTec AD2 public `can` DARC raw-ladder
  pilot on dsba3 `hun_fsad_tta_012` GPUs 0,1,2,3. The valid v2 cell used
  P16-random seed 0, fold 0, full `72 good + 90 bad`, and produced 162
  coverage rows plus 648 finite native maps with no missing/duplicate IDs.
  G0/L0/L1/R1 pAUROC@.05 and oracle F1 were respectively
  `0.667650/0.017797`, `0.547831/0.000488`, `0.579249/0.000827`, and
  `0.550352/0.001647`. R1-G0 was `-0.117298` pAUROC, `-0.001312` AP,
  `-0.016150` F1, and `-0.083333` component recall. All queries accepted
  5/5 registrations and fallback was only `2.370760%`, so the failure is the
  noisy hard-local correspondence/reconstruction field rather than coverage,
  threshold, or morphology. The prior v1 label-path seed defect was
  invalidated before metrics and the v2 content-SHA identity was exhaustively
  verified. After evaluation, the valid root's 648 TIFFs and all related
  raw-ladder TIFFs were removed while canonical compact metrics and shard audit
  records were retained. The preceding 720-row AD1 synthetic Gate 1 passed
  (`+0.096791` AP), supporting unrestricted high-resolution G0 but not the
  hard-local terminal. Because the pilot's L1-L0 continuation and R1-G0 stop
  clauses both fired without frozen precedence, the verdict is
  `KILL_FOR_CLAIM / CONTRACT_CONFLICT`; operationally, raw R1 is stopped. Report:
  `skill_graph/experiments/2026-07-10_flowtte_darc_resolution_correspondence/report.md`.
- 2026-07-10: Started the staged DARC resolution/correspondence experiment
  cycle for the FlowTTE `can` structural failure. Frozen protocol separates
  evaluator parity, paired 16px/8px sampling, `G0/L0/L1/R1`, reduced AD2
  shadow, and untouched final-claim gates; it also separates P16-random,
  P16-SuperAD-selected, M16-fullpool, and Pfull normal access. Execution target
  is dsba3 `hun_fsad_tta_012` on host GPUs 0,1,2. Interim verdict:
  `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`. Design:
  `skill_graph/analysis/2026-07-10_flowtte_darc_experiment_design.md`; report:
  `skill_graph/experiments/2026-07-10_flowtte_darc_resolution_correspondence/report.md`.
- 2026-07-10: Implemented and completed the exact SuperAD rotation-8 support
  augmentation diagnostic for the retained H+ DVT MLP FlowTTE path. The
  paired all-eight candidate used the exact same ordered 16 source images and
  all the same method/scoring settings as a matched identity control; only the
  transform orbit and derived effective support views changed (`16 -> 128`).
  Rotation-8 scored `0.819251/0.506201` versus identity
  `0.836389/0.527370`, regressing by `-0.017137` AUROC and `-0.021168` F1.
  `sheet_metal` fell by `-0.111020/-0.119605` and `fabric` F1 fell by
  `-0.042623`; only `fruit_jelly` improved F1 by at least `0.01`. Added exact
  OpenCV transform parity tests and full-batch-equivalent support-row
  microbatching; the worst-case smoke completed without OOM, paired manifests
  and executed v1 provenance hashes matched, and retained anomaly-map count is
  `0`. Post-run review expanded the guard to the actual 48-file split-root
  dependency closure and verified that it rejects stale v1 markers. Verdict:
  `KILL_FOR_CLAIM / NO_CONTINUE`. Report:
  `skill_graph/experiments/2026-07-10_flowtte_hplus_dvt_superad_rotation8/report.md`.
- 2026-07-10: Created a new-session handoff document for the current FlowTTE
  work, including the retained H+ DVT method structure, register/Transformer
  negative evidence, DVT-lite explanation, completed structural diagnostics,
  hparam tuning status, remote/container paths, and next-step checklist.
  Handoff:
  `skill_graph/analysis/2026-07-10_flowtte_session_handoff.md`.
- 2026-07-10: Recorded the completed Transformer/register prefix analysis and
  launched the FlowTTE H+ extreme class-agnostic hyperparameter sweep on dsba3
  `hun_fsad_tta_012` GPUs 0,1,2. Stage-1 all-eight sweep completed with
  best F1 from `lambda_logdet=1e-2` (`0.836202/0.528875`) and best AUROC from
  support brightness `0.80,1.20` (`0.838230/0.528203`) versus the H+ DVT
  reference `0.836739/0.527427`; cleanup verified `0` retained
  `anomaly_maps/` locally and remotely. Phase-2 is running and currently shows
  `lambda_logdet=2e-2` as best partial F1 (`0.835673/0.529874`), with phase-3
  queued around `lambda_logdet ~= 2e-2`. Verdict remains
  `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`. Report:
  `skill_graph/experiments/2026-07-10_flowtte_transformer_context_and_hparam_sweep/report.md`.
- 2026-07-09: Implemented and ran the FlowTTE patch-structure diagnostic suite
  for the hypothesis that patch-independent NF processing is the bottleneck.
  Shared setting: all-eight MVTec AD2, DINOv3-H+/16, layers `[7,15,23,31]`,
  fixed 16-shot support, DVT `position_mean alpha=1.0`, dsba3
  `hun_fsad_tta_012` GPUs 0,1,2. Completed variants: `conditional_cls`
  `0.832374/0.512126`, `foreground_flow_mixture` `0.834712/0.519250`, and
  `local_contrast` `0.806200/0.438345`; none beats the H+ reference
  `0.836739/0.527427`. Coordinate-conditioned `xy` and `cls_xy` were
  runtime-blocked with no usable metrics. Cleanup verified `0` retained
  `anomaly_maps/` locally and remotely. Verdict:
  `KILL_FOR_CLAIM / NO_CONTINUE`. Report:
  `skill_graph/experiments/2026-07-09_flowtte_patch_structure_flow_diagnostics/report.md`.
- 2026-07-09: Implemented and ran class-agnostic object/foreground prior plus
  score-field calibration diagnostics on the retained H+ DVT NF FlowTTE branch
  using dsba3 `hun_fsad_tta_012` GPUs 0,1,2. Tested RGB object prior
  `0.778452/0.374071`, RGB-feature product prior `0.767204/0.346625`,
  support-score reliability `0.828178/0.505659`, and RGB prior plus reliability
  `0.780889/0.376619`, all below the H+ DVT NF reference
  `0.836739/0.527427`. Conclusion: hard class-agnostic objectness suppression
  is not the answer; it suppresses anomaly evidence together with nuisance
  background. Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`.
  Report:
  `skill_graph/experiments/2026-07-09_flowtte_object_prior_score_calibration/report.md`.
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
- 2026-07-09: Ran FlowTTE structured-memory diagnostics on all 8 MVTec AD2
  public objects using dsba3 `hun_fsad_tta_012` GPUs 0,1,2. Added
  feature-derived memory context sources, lazy context grouping, and
  calibration sample cap to make H+ structured retrieval feasible. The
  executed all-8 method used `image_feature_mean_ch16`, `context_mode=top_m`,
  `context_top_m=4`, DVT alpha `1.0`, fixed SuperAD-16 support, and
  `calibration_sample_size=4096`. Mean result: `seg_AUROC_0.05=0.832426`,
  `seg_F1=0.524593`, below the H+ DVT baseline `0.836739/0.527427`.
  Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.
  Report: `skill_graph/experiments/2026-07-09_flowtte_structured_memory_diagnostics/report.md`.
- 2026-07-09: Implemented and smoke-tested `normality_mode=conv2d_flow`, a
  register-free 2D convolutional affine-coupling flow that preserves the
  DINOv3-H+ feature map during flow projection and flattens only the final
  latent map for existing memory distance scoring. Smoke on `can,vial,fabric`
  used fixed SuperAD-16 support, layers `[7,15,23,31]`, DVT `alpha=1.0`,
  no-TTE, and `density_weight=0.0`. Mean result was
  `seg_AUROC_0.05=0.744949`, `seg_F1=0.354137`, below the same-object H+ DVT
  baseline `0.758338/0.377474`; all three objects lost F1. Verdict:
  `KILL_FOR_CLAIM / NO_CONTINUE`.
  Report: `skill_graph/experiments/2026-07-09_flowtte_conv2d_flow_diagnostics/report.md`.
- 2026-07-09: Implemented `normality_mode=transformer_flow`, a register-free
  Transformer affine-coupling flow over DINOv3-H+ patch tokens, and ran
  smoke plus all-eight diagnostics on dsba3 `hun_fsad_tta_012` GPUs 0,1,2.
  Smoke on `can,vial,fabric` improved same-object mean from
  `0.758338/0.377474` to `0.768982/0.395014`, mainly through `vial`.
  All-eight result fell below the H+ DVT baseline: `0.828600/0.502237`
  versus `0.836739/0.527427`. Positive signal remained on `vial`,
  `sheet_metal`, and `fabric`, but `fruit_jelly`, `walnuts`, and
  `wallplugs` produced broad no-harm failure. Verdict:
  `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.
  Report: `skill_graph/experiments/2026-07-09_flowtte_transformer_flow_diagnostics/report.md`.
- 2026-07-10: Ran architecture-only image-level failure analysis for
  `normality_mode=transformer_flow` on all 8 MVTec AD2 public objects. The
  matched control used `normality_mode=fused`, `density_weight=0.0`; all
  other core settings were fixed: DINOv3-H+/16, layers `[7,15,23,31]`,
  DVT `position_mean alpha=1.0`, fixed SuperAD-16 support, no-TTE,
  context/register disabled, dsba3 `hun_fsad_tta_012` GPUs 0,1,2. Mean
  architecture-only result was MLP `0.837518/0.523904` versus Transformer
  `0.828600/0.502237`. Per-image analysis showed negative bad-image
  GT-vs-background score-gap deltas for every class, including `vial` and
  `sheet_metal` where global F1 improved. Conclusion: Transformer Flow mainly
  smooths/regularizes the score field and weakens local anomaly contrast, so
  it is not a robust method branch. Dense maps were regenerated for analysis
  and then removed; local pulled run roots contain `0` map TIFFs. Verdict:
  `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.
  Report: `skill_graph/experiments/2026-07-10_flowtte_transformer_failure_image_analysis/report.md`.
- 2026-07-11: Closed and discarded the image-disjoint support-consensus S0
  branch. On `can,fabric,fruit_jelly,rice`, Phase-3 base was
  `0.815615/0.471988` and residual was `0.815634/0.472022`
  (`+0.000019/+0.000033`). `can` regressed, q25 was worse, and shuffled IDs
  matched or beat the candidate. Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`.
  Implementation, tests, launchers, preregistration, and local/remote run
  artifacts were deleted; only the result report remains:
  `skill_graph/experiments/2026-07-11_flowtte_support_consensus_s0/report.md`.
# 2026-07-12 — MVTec AD1 VisionAD-aligned ViT-B/14-register sweep completed

- Ran all 15 classic MVTec AD categories for shots 1/2/4/8/16, seed 1, on
  dsba5 GPUs 0 and 1 with frozen `dinov2_vitb14_reg`, 448-to-392 input,
  layers 2/5/8/11, and verified 28x28/768 features.
- Class-macro results (i-AUROC/i-AUPRC/p-AUROC/p-AUPRC/p-AUPRO):
  - 1: `0.756873/0.881598/0.874048/0.231153/0.652445`
  - 2: `0.887245/0.935944/0.942296/0.360746/0.815653`
  - 4: `0.915777/0.954135/0.950294/0.388857/0.848569`
  - 8: `0.940539/0.966574/0.956502/0.417193/0.866856`
  - 16: `0.939385/0.965973/0.959495/0.425467/0.875318`
- Rejected an incomplete first run after detecting total raw-float16 pixel-score
  overflow. The clean rerun uses monotonic signed-log1p uint16 histogram ranks;
  exact float32 AUPRO remains integrated through FPR 0.30.
- Audit passed: five finite summaries, 15 per-category rows and support
  diagnostics per shot, exact requested unique support counts, no runtime error,
  and zero retained anomaly-map directories locally or remotely.
- Local validation: `407 passed`; Ruff, shell syntax, and Python compilation
  passed.
- Compact pullback:
  `results/remote_runs/dsba5/flowtte_mvtecad1_visionad_vitb14reg_s1_2_4_8_16_20260712_v1`.
- Report:
  `skill_graph/experiments/2026-07-12_flowtte_mvtecad1_visionad_vitb14reg/report.md`.
- Verdict: `BLOCKED_BASELINE`; encoder/input alignment is valid, but Flow-head
  training and evaluator geometry prevent a strict paired VisionAD comparison.

# 2026-07-12 — Static Flow-LatentBank MVTec AD1 ViT-B/14-register completed

- Re-ran the original all-15 4-shot, first-support, shorter-edge 448
  Flow-LatentBank recipe with the two requested changes: frozen
  `dinov2_vitb14_reg` layers 2/5/8/11 and no TTE (`expansion_budget=1.0`).
- Result (i-AUROC/i-AUPRC/p-AUROC/p-AUPRC/p-AUPRO):
  `0.972100/0.987432/0.973747/0.581160/0.938160`.
- Historical ViT-L + TTE row was
  `0.969631/0.983877/0.964028/0.576137/0.936470`; descriptive deltas are
  positive on all five means but are not causal because backbone and pixel-rank
  writer also differ.
- Audit passed: 15 finite category rows, exact first-four paths, memory
  `4096 -> 4096` for every category, and zero dense-map directories locally
  and remotely.
- Local verification: 412 tests passed; focused Ruff, compilation, and launcher
  syntax passed.
- Report:
  `skill_graph/experiments/2026-07-12_flow_latentbank_mvtecad1_static_vitb14reg_shot4/report.md`.
- Verdict: `ACCEPT_MEASUREMENT / BLOCKED_BASELINE`.

# 2026-07-13 — FlowTTE q4096/guided stack and grid-shift implementation

- Added fixed-order retained-map stack evaluation and a metrics-only two-view
  grid-shift smoke with original-bank arm A and phase-matched refit arm C.
- Local verification passed: 418 tests, Python compilation, and shell syntax.
- Remote execution remains pending because the managed session rejected both
  SSH launch attempts with `socket: Operation not permitted`; no remote or
  anchor artifacts were changed.
- Report:
  `skill_graph/experiments/2026-07-13_flowtte_gridshift_2view/report.md`.
- Verdict: `BLOCKED_DATA` for remote execution evidence.

# 2026-07-13 — Static Flow-LatentBank MVTec AD1 shot sweep completed

- Extended the accepted static ViT-B/14-register 4-shot setting to 1, 2, and
  8 shots with the same first-support, shorter-edge 448, Flow 3-epoch, latent
  1-NN, and no-TTE contract.
- Class-macro 1/2/4/8-shot results rise monotonically on all five requested
  metrics; the 8-shot row is
  `0.981751/0.991162/0.975455/0.590661/0.942405`.
- Audit passed across 45 class-runs: exact first-N supports, static memories,
  finite metrics, identical metric copies, clean logs, and zero retained maps.
- Report:
  `skill_graph/experiments/2026-07-13_flow_latentbank_mvtecad1_static_vitb14reg_shot_sweep/report.md`.
- Verdict: `ACCEPT_MEASUREMENT / BLOCKED_BASELINE`.

# 2026-07-13 — Static Flow-LatentBank MVTec AD1 DVT-alpha sweep completed

- Swept only support position-mean DVT alpha `{0,0.25,0.5,0.75,1.0}` on the
  accepted 4-shot, static ViT-B/14-register Flow-LatentBank setting.
- Alpha 0 exactly reproduced the accepted no-DVT metrics. No nonzero alpha
  passed the all-five retention gate; image metrics degraded monotonically.
- Alpha 0.5 gave the best p-AUROC/p-AUPRC/p-AUPRO
  (`0.975438/0.583405/0.942154`), but the mean gain was dominated by
  `transistor` and 9--10 of 15 classes degraded on each pixel metric.
- Audit passed across 75 class-runs: exact supports, fixed `4096 -> 4096`
  banks, finite and internally identical metrics, clean logs, and no maps.
- Report:
  `skill_graph/experiments/2026-07-13_flow_latentbank_mvtecad1_static_vitb14reg_dvt_alpha_sweep/report.md`.
- Verdict: `KILL_FIXED_DVT_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.

# 2026-07-13 — Static Flow-LatentBank MVTec AD1 component 0/1/2 completed

- Independently evaluated density removal, frozen CLS soft retrieval at the
  fixed AD2 weight 10, and RGB guided-r8 on the accepted DVT-off 4-shot static
  ViT-B/14-register contract.
- Density removal is the only locked-gate passer:
  `0.973680/0.987876/0.973087/0.584241/0.938218`, including +0.3080-point
  p-AUPRC with no macro loss beyond 0.10 point.
- CLS soft retrieval fails image-AUROC and p-AUPRO retention. Guided-r8 raises
  p-AUROC/p-AUPRC/p-AUPRO by +0.3993/+4.0295/+0.3844 points and improves
  p-AUPRC in all 15 classes, but i-AUROC loss is -0.1077 point and therefore
  fails the preregistered all-five gate.
- Identity regeneration exactly reproduced all five baseline metrics. Audit
  passed exact supports, static `4096 -> 4096` memories, clean logs, 15-class
  aggregation, and zero retained arrays. Full regression: 424 passed.
- Report:
  `skill_graph/experiments/2026-07-13_flow_latentbank_mvtecad1_static_vitb14reg_component012/report.md`.
- Verdict:
  `KEEP_DENSITY0_DIAGNOSTIC / KILL_CLS_SOFT_W10 / KILL_FIXED_GUIDED_R8_FOR_AD1_CLAIM / BLOCKED_BASELINE`.

# 2026-07-13 — AD2 close/fill/erode promoted to runner default

- The AD2 FlowTTE CLI now defaults to fixed class-agnostic binary morphology:
  17-pixel directional closing across 16 angles, contour fill, and one 3x3
  erosion at the raw best threshold.
- Continuous `seg_AUROC`, `best_thre`, and `seg_F1_raw` remain auditable;
  `seg_F1` records the postprocessed binary result. The manifest records the
  profile and threshold source, and `--binary-postprocess none` restores raw
  evaluation.
- Local verification: 432 tests passed, plus compilation and diff checks.

# 2026-07-24 — Same16 DVT × density factorial attribution completed

- Completed the exact-support, all-eight MVTec AD2 public-shadow 2×2 using
  reused A6/Basic and new DVT-only/density-only cells in fixed dsba3 container
  `hun_fsad_tta_012`.
- A6/DVT-only/density-only/Basic scored
  `0.759284/0.771822/0.760328/0.772635` pAUROC,
  `0.350216/0.356363/0.350910/0.355676` raw F1, and
  `0.538267/0.530338/0.543298/0.533408` AU-PRO.
- DVT's average main effect was `+0.012423/+0.005457/-0.008910`, while
  density was `+0.000928/+0.000003/+0.004050`; the interaction was
  `-0.000232/-0.001381/-0.001961`.
- DVT pAUROC was driven heavily by `sheet_metal` and its CI crossed zero;
  DVT also collapsed sheet-metal AU-PRO and walnut F1. Density and interaction
  stayed below material bounds. No factor received attribution credit.
- The local/remote summary SHA-256 matched, all 16 new metrics/manifests were
  present, logs were clean, dense-map directories were zero, and the isolated
  snapshot was removed.
- Report:
  `skill_graph/experiments/2026-07-24_same16_dvt_density_factorial/report.md`.
- Verdict: `MIXED_OR_INCONCLUSIVE_ATTRIBUTION / KILL_FOR_CLAIM`; retain Basic
  descriptively, but do not open shot scaling or another DVT/density sweep.

# 2026-07-13 — AD2 RGB guide and morphology promoted to operational defaults

- The canonical AD2 FlowTTE CLI now applies half-scale RGB guided-r8
  (`radius=8`, `epsilon=0.01`) to the continuous anomaly maps before the
  already-default 17px/16-angle close/fill/3x3-erode binary morphology.
- The run manifest records guide activation, exact variant, ordering, absence
  of ground-truth use, and refined-map count. `--rgb-guide none` and
  `--binary-postprocess none` independently restore matched ablation paths.
- Historical AD2 experiment launchers explicitly pin `--rgb-guide none` so
  their previously reported configurations do not change on rerun.
- This is a user-selected operational default, not a revision of the locked
  experimental verdict: the all-eight mean improved, while the `can` AUROC
  class-harm gate still failed.

# 2026-07-13 — AD2 proposed full-normal SuperADD-style threshold run completed

- Ran all eight MVTec AD2 public objects with the frozen proposed H+ setting,
  using the sorted 7/8 normal prototype split and disjoint 1/8 normal
  threshold split on dsba3 GPUs 0--3 plus dsba5 GPUs 0--1.
- Applied the user-specified exact global k=100 mean-tau score ranking to Flow
  latents and retained a static 100,000-entry bank for every class.
- Macro results: p-AUROC `0.833807`; held-out-normal raw/morph F1
  `0.227669/0.230264`; TESTpublic oracle raw/morph F1
  `0.535016/0.548522`.
- The reported SuperADD context is `0.8393/0.6261`: ranking is close
  (`-0.005493` AUROC), but fixed morphology F1 is lower by `-0.395836` and
  oracle morphology F1 remains lower by `-0.077578`.
- Audit passed exact disjoint/exhaustive splits, eight finite rows, static
  `100000 -> 100000` banks, clean logs, and dense-map cleanup. Local suite:
  443 passed.
- Report:
  `skill_graph/experiments/2026-07-13_flowtte_ad2_fullnormal_superadd_thresholds/report.md`.
- Verdict: `ACCEPT_MEASUREMENT / BLOCKED_BASELINE`.

# 2026-07-13 — AD2 random 16-shot DINOv2-L/14 completed

- Ran all eight MVTec AD2 public classes with exact NumPy seed-0 random
  16-shot support, DINOv2-L/14 layers 5/11/17/23, and the remaining proposed
  Flow-LatentBank + DVT + density + guided-r8 + morphology stack frozen.
- Macro results: p-AUROC `0.781110`, guided raw F1 `0.371855`, and guided
  morphology F1 `0.430721`.
- This is below the existing DINOv3-H+ fixed-support proposed context by
  `0.056418` p-AUROC and `0.123138` morphology F1, but the comparison is not
  a backbone-only ablation because support policy also changes.
- Audit passed exact recomputed support selection, 16 unique paths per class,
  static memories, all configuration fields, clean logs, and zero retained
  dense-map directories. Local regression: `446 passed`.
- Report:
  `skill_graph/experiments/2026-07-13_flowtte_ad2_random16_dinov2l/report.md`.
- Verdict: `ACCEPT_MEASUREMENT / BLOCKED_BASELINE`; single-seed diagnostic.

# 2026-07-13 — Few-shot AD2 model-ladder plan frozen

- Defined Basic (DINOv2-L), Ours+ (DINOv2-R), and Ours++ (DINOv3-H+) while
  explicitly excluding Ours++ Full and any new full-normal candidate run.
- Basic owns the exact-reference SuperAD claim; Ours+ and Ours++ are backbone
  scaling variants, with SuperADD used only as external full-normal context.
- Prior H+ negatives prevent a broad deep-Flow sweep, but depth and width stay
  explicit through one complete six-run factorial: layers 2/4/6 x hidden
  1/2. Eight layers is excluded.
- Capacity is followed by a four-row log-det/brightness factorial; its anchor
  overlaps the selected capacity row, requiring only three new runs.
- Design:
  `skill_graph/analysis/2026-07-13_fewshot_ad2_model_ladder_experiment_plan.md`.
- Added `scripts/run_flow_tte_basic_hparam_parallel_remote.sh`: nine unique
  Basic configurations are assigned once across dsba3 GPUs 0--3 and dsba5
  GPUs 0--1, with per-GPU queues, offline DINOv2 preflight, exact support
  rebasing, restart-safe object skipping, aggregation, and dense-map cleanup.
- Verification: assignment/factorial tests plus focused checks `10 passed`;
  full regression `448 passed`; launcher syntax and diff checks passed.
# 2026-07-15 — MVTec AD2 Shift-Factorized Latent Bank launched

- Implemented robust low-rank support-1NN residual projection as an explicit
  test-environment shift model, verified 25 focused tests, passed remote
  fruit-jelly projection/scoring smoke, and launched six all-object rank/trim
  variants across dsba3 GPUs 0--3 and dsba5 GPUs 0--1.
- Report: `skill_graph/experiments/2026-07-15_sflb_ad2_rank_trim/report.md`.
- Status: `RUNNING / CONTINUE_DIAGNOSTIC`; official AU-PRO@0.05 pending.

# 2026-07-16 — DSCR high-resolution density shadow diagnostic completed

- Ran the frozen five-object, first-good-4/first-bad-4 public shadow cell on
  dsba3 GPUs 0--3 and dsba5 GPU 0 with identical G0/D0/D0c support, folds,
  feature geometry, and runner hash.
- G0/D0/D0c macro pAUROC were `0.565902/0.539851/0.539858`; macro oracle F1
  were `0.080168/0.068325/0.066906`; macro fixed F1 were
  `0.009808/0.008520/0.008450`.
- Density improved pAUROC on only `rice` and oracle F1 on only `fabric` and
  `rice`. `can` exceeded the 0.03 pAUROC harm bound, and `vial` exceeded both
  the 0.03 pAUROC and 0.05 F1 bounds. Normal-tail calibration did not repair
  the ranking.
- Audit passed 40 logical queries, 120 finite/nonconstant arm maps, exact arm
  ID parity, five completion manifests, identical settings, and zero retained
  `anomaly_maps/` after cleanup. Focused evaluator test: `1 passed`; lint was
  unavailable because `ruff` was not installed.
- Report:
  `skill_graph/experiments/2026-07-16_dscr_hres_density_shadow/report.md`.
- Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`; do not launch the optional
  whitening/Mahalanobis follow-up or reopen the killed hard-local branch.

# 2026-07-16 — DSCR paired acquisition interaction audit completed

- Reran frozen G0/D0 on dsba3 for `fabric`, `rice`, and `vial`, with one good
  and one bad source identity under regular, overexposed, shift-1, and shift-2.
- Regular macro D0-G0 pAUROC/F1 was `-0.022554/-0.024119`; the shifted mean was
  `-0.029614/-0.022535`, yielding only `-0.007060/+0.001584` interaction.
- `vial` already lost `0.128838/0.129513` at regular, while `rice` responded in
  opposite directions to overexposure (`+0.155902` pAUROC) and shift-1
  (`-0.030151`). The failure is object/condition heterogeneous, not a shared
  shift-induced density defect.
- Pooled `fabric/rice` reproduced the prior dsba3 results essentially exactly;
  dsba3 `vial` reproduced the prior dsba5 result within `1.1e-6`, closing the
  host-difference concern.
- Audit passed exact 24-query coverage, finite/nonconstant float32 maps, arm-ID
  parity, three completion manifests, 18 local tests, and zero retained PAIA
  TIFFs after cleanup.
- Report:
  `skill_graph/experiments/2026-07-16_dscr_paia_condition_audit/report.md`.
- Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`; do not condition or retune D0.
# 2026-08-04 — Shift-Conditioned Patch-Local Flow preregistered

- Froze a DINOv3-H+/16, DVT position-mean, fixed-16 follow-up that keeps the
  Stage-A MLP Flow and raw latent 1-NN scorer while adding only a bounded
  augmentation-supervised image-shift residual branch.
- Corrected four pre-execution defects: simultaneous zero head/gain gradient
  deadlock, silent normalized-scorer replacement, absolute augmentation
  prediction, and the missing Stage-B NLL definition.
- Added a support-only held-shift/defect-retention gate before any AD2 public
  image access and fixed dsba5 host GPU 3 / container cuda:0 snapshot custody.
- Preregistration:
  `skill_graph/experiments/2026-08-04_shift_conditioned_patch_local_flow/preregistration.md`.
