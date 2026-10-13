# Current research state

> Latest execution reviewed: 2026-09-25; older sections retain dated context.
> This file is the live project status, not a
> historical experiment log. It records confirmed configuration and pending
> decisions only; it must not invent results.

> For completed-run result lookup, read only the latest summary in the [compact comparison index](../results/comparisons.md) first; read its macro table for all-method comparisons. Do not routinely load this file's historical sections. Its paired CSV remains `results/comparisons/psma_mr_vs_history_20260909/psma_mr_vs_history_20260909.csv`.
> After each completed run, update that same Markdown ledger and its paired CSV.
> Detailed comparison history through 2026-09-14 is preserved in `results/comparisons/details/history_through_20260914.md`; future collector details live at `<run-root>/comparison.md`.

## ACTIVE — Codex operational takeover, 2026-09-25

The user transferred experiment operations and research reasoning from the
Orca Claude `0924-Main` session to Codex. Claude acknowledged that it will idle;
existing GPU processes remain running. Coding is assigned to GPT-6 Sol workers.
The active scope remains the registered [v7 evidence batch](../configs/benchmarks/v7_evidence/PROTOCOL_20260924.md),
not an unrestricted method search. Manuscript `.tex` and official E3 submission
remain outside this execution scope.

- E1_v2, I1a Gaussian, and E9 MVTec/VisA have completed on dsba3 and their
  metrics/config/manifests are pulled and collected. The E11 shot-shift run was
  interrupted at the user's request; retain its incomplete artifacts. Do not
  reactivate that suite while finishing the remaining analyses below.
- E4 clean and Gaussian were already complete, despite stale Claude progress
  reports. Gaussian full12 is 60.00588 → 64.76544 AUPRO (+4.75956pp).
  Finish shot/impulse/speckle with the same immutable local snapshot
  `FSAD-TTA-snapshot-v7` (package hash `2e78ffd985463933c96a0f812813e8b61f8865c4dd1a11766c60f4c8e96ea3b1`).
  All 12 clean support noise reference values are positive, so the zero-reference
  guard fix does not invalidate this matched E4 continuation. Shot, impulse, and
  speckle are now complete and collected; finish the registered G/P interpretation.
- E5's original signed-drop formula was correct; the later sign erratum is
  superseded in place. MAD replacement does not meet the relief criterion;
  this does not establish absence of support-side causes or interactions.
- E7 must use regular CQ metrics: gated token − source = +0.53308pp, not the
  reported whole-split +0.56254pp. The registered threshold is not met; support
  scarcity is not established as the exclusive cause of the 1-shot difference.
- E10's same-support acquisition comparison supports absolute AUPRO leadership
  in the tested strata, not minimum degradation or stable fixed-threshold FPR.
- E12 resolution control is complete and collected (24/24, no failures). On
  regular full8, four-scale RSG3 scores 61.25986 AUPRO / 77.94223 pAUROC;
  1120-only with matched total bank capacity scores 59.19861 / 76.05768.
  The +2.06125pp AUPRO difference clears the registered 0.86pp floor; the
  multiscale arm also leads both single-scale arms in shift_1/2/3. This is one
  fixed 16-shot support draw on the public development population, not a
  multi-seed generalization result. See the [E12 comparison](../results/v7_evidence/E12_resolution_control/comparison.md).
- Pooled-bank RNG equivalence, finite zero-reference guard behavior, and import
  scope/provenance fixes are validated in local commit `a9b9e36` (103 targeted
  tests passed; independent review approved). Not pushed or activated remotely.
  The I1b gate is closed: full15 I1a Gaussian `lowres_pool` is 52.82, below 72.009.
- E1_v2 is collected: C0 passes for seeds 1–4, all 8 classes are support-paired,
  and including the existing seed-0 instrumentation gives R1 in 5/5 draws
  (mean G(2)−G(16)=+3.78pp). Gated AUPRO is 56.07±2.39 at 2 shots and
  59.48±0.60 at 16 shots; this does not establish cross-shot equivalence.
- I1a Gaussian full15 is collected at 52.82 AUPRO for `lowres_pool`, below the
  72.009 gate; I1b is closed by its registered condition.
- E9 MVTec/VisA is collected with complete support-hash pairing against stored
  RSG3 draws. Under the +0.86pp claim rule, RSG3 leads on VisA at 4/8/16 shots,
  ranks first by only +0.55pp at 2 shots, loses at 1 shot, and does not clear the
  lead rule on MVTec at any shot. See the run comparison reports for exact ranks.
- E11 shot-shift was stopped at the user's request after 8 cells completed;
  partial artifacts remain preserved and are not collected as a complete suite.

## COMPLETED — RSG4 noise-guarded remedy, 2026-09-23

Official naming decision (2026-09-26): **RSG4** denotes RSG3 plus the selected
query-conditional `noise_guard_action: lowres` rule. Historical `NR_lowres`
labels remain unchanged as run identities. The canonical MVTec AD 2 config is
[`tail_bag_RSG4.yaml`](../configs/methods/tail_bag_RSG4.yaml).

[Attribution protocol](../configs/benchmarks/noise_attribution/PROTOCOL_20260923.md)
complete: 120/120 cells, 0 failures, local + dsba3. C0 control exact
(clean 85.3614, noise 64.6741; every class diff 0). Relief vs control drop
-20.687: deep-only +1.646 (A1 fail), lowres [448,672] +5.087 (A2 PASS,
clean cost -0.752), no-CQ -0.692 (A3 fail). The 896/1120 branches carry the
failure. bottle/zipper collapse in every arm and also in SuperAD (31.1/19.5);
only PatchCore holds them (61.3/30.3), plausibly via its 3x3 feature pooling.
Values: `results/noise_attribution/attribution_summary.json` (one-off aggregator removed 2026-09-24).

[Remedy protocol](../configs/benchmarks/noise_remedy/PROTOCOL_20260923.md),
locked before attribution results were inspected; the fixed mapping selects
`lowres`. `noise_guard` in reference_matching.py (Immerkaer sigma / support
median > k=9, k fixed from unlabeled image statistics). Codex implementation
reviewed: fixed an unreported bug (lowres_deep_only also ran NLM), default
resolutions, float NLM strength, BGR order; tests strengthened to exact
equality with directly configured remedies (87 targeted tests pass).
G1 PASS offline on full populations: 0/7984 firings across MVTec AD clean,
brightness/contrast/blur, MVTec AD 2 test_public; 100% on every noise root.
Held-out roots from `scripts/prepare_corrupted_mvtec_seeded.py` are
pixel-identical local vs dsba3; library impulse noise is unseeded, replaced.
Dedicated MVTec AD 2 RSG4 execution completed on 2026-09-26: seed 0 full8
1/2/4/8/16-shot scaling plus seed 1--2 full8 2/16-shot endpoint replication,
72/72 class cells with no failures. Every run group fired the guard 0/1084
queries (largest observed ratio 2.4795 versus k=9); the seed-0 16-shot result
reproduces matched RSG3 exactly at 59.14577 AUPRO / 77.10613 pAUROC. Across
three seeds, 2-shot is 55.61652+/-2.69990 AUPRO and 75.64451+/-0.52560 pAUROC;
16-shot is 59.62938+/-0.79638 and 77.23270+/-0.29345 (sample SD). Seed-0
scaling shows the main gain from 1 to 2 shots, followed by an AUPRO plateau.
See the canonical ledger and per-run comparison reports for provenance.

Remedy stage: 180/180 cells + 15 matched-hardware G2 cells, 0 failures after
one fix. Attempt 1 failed every FIRED cell: FusionFactorial.predict_batch
required one gate grid per configured resolution; now per resolution actually
scored (`last_resolutions`), with a regression test that fails on the old code.
All locked rules PASS (`results/noise_remedy/remedy_summary.json`; one-off aggregator removed 2026-09-24):
- G1: 0/7984 guardrail firings (offline, full populations).
- G2: guarded clean == NA_gated clean on 15/15 classes when run on the same GPU.
  Cross-GPU (A6000 vs 4090) differs by ~1e-5pp only; the first G2 attempt mixed
  machines, which a byte-identity test must never do.
- P1: Gaussian s3 64.674 -> 69.009 (+4.335), 13/15 classes; bottle -2.55,
  zipper -0.69. Equals NA_lowres exactly on 15/15 classes.
- P2: held-out shot +9.103, impulse +4.650, speckle +13.011, gaussian s5 +12.156,
  gaussian s1 +0.798 (below floor); 4/5 wins, mean +7.944.
Held-out gains exceed the development gain, and grow with severity
(s1 +0.80, s3 +4.34, s5 +12.16). Still below SuperAD 71.10 (-2.10) and
PatchCore 74.03 (-5.02) on Gaussian s3; the residual is bottle/zipper, which no
tested component fixes (candidate: PatchCore-style local feature pooling).

## COMPLETED — Coverage-void instrumentation, 2026-09-23

[Locked protocol](../configs/benchmarks/void_instrumentation/PROTOCOL_20260922.md)
(V3 null correction registered after one pilot cell, before matrix collection).
MVTec AD 2 full8 `test_public`, seed0, nested manifests 1/2/4/8/16-shot
(`configs/methods/VOID_support_*.json`), arms gated/all/off, 6 acquisition
strata per cell. 120/120 cells, 0 failures; gated local, all/off dsba3.
Control reproduces: 16-shot regular 61.26 / shift_1 59.40 = prior table.
The (since removed) aggregator enforced balanced
object panels and de-duplicates a pilot re-run (both bugs found mid-run).

Verdicts: H1 FAIL, H2 PASS, H3 PASS, H4 PASS-but-vacuous, H5 FAIL.
Calibration null confirmed (16-shot regular gate rate 0.994x of 1%). The
latent "coverage void" hypothesis is rejected: V5 rises WITH shots
(2/4/8/16: 0.64/0.74/0.95/1.00, LOO reference drift), and explains neither
axis (loss~logV5 R2 0.097 shift, 0.016 shot, opposite signs). H4's rule lacked
a per-axis informativeness precondition. 1-shot uses the leave-token-out
fallback (reference_matching.py), so its V5 (3.25) is a different estimator.

Component result (regular, macro AUPRO): gating gain 7.93/11.49/7.50/6.66/6.51
for 1/2/4/8/16-shot; shift_1 6.73/10.02/5.09/4.60/4.61. Shallow net vs deep-only
below floor at every shot. Per-class check (2026-09-23): gating gain is larger
at 2-shot than 16-shot in 7/8 classes (paired, robust). BUT gated 2->16 macro
change 0.44pp is CANCELLATION, not flatness: per-class changes span
-6.52 (can) .. +7.34 (walnuts). "gated@2 > ungated@16" holds 7/8 classes but is
unpaired across support sets and concentrated in fabric/wallplugs. Do not claim
"support dependence removed"; Outline_v6 title/abstract/4.3.1 overclaim this
and need correction. Seed replication is required for any cross-shot claim.

## COMPLETED — External-method corruption comparison, 2026-09-22

User authorized immediate local + dsba3 execution for a Table-1-style
cross-method robustness table in PaperWorks Section 8.1. [Locked protocol](../configs/benchmarks/corruption_compare/PROTOCOL_20260922.md):
MVTec AD full15, normal16 seed0, clean plus brightness/contrast/defocus-blur/
Gaussian-noise severity3. Six published-method comparators (RegAD, RD4AD,
PatchCore, FastRecon, RN-FMLK, SuperAD) use their existing Table-1 DINOv2-L
settings; existing RSG3 artifacts are reused after provenance validation.
Completed all450 new method-condition-class evaluations; together with75 reused
RSG3 cells, all525 cells are finite and shape-complete. Clean support files in
all corruption roots were byte-checked against the clean root; only queries are
corrupted. [Final analysis](../results/comparisons/corruption_compare_20260922/comparison.md):
RSG3 leads AUPRO under brightness (84.4245) and contrast (83.1513), and leads
the brightness/contrast/blur mean (83.1665). Across all four corruptions,
SuperAD is first (79.8073), PatchCore second (79.0452), and RSG3 third
(78.5434); RSG3 retention is92.01%, with a -20.6873pp Gaussian-noise failure.
Claim competitive high absolute performance and mild-shift stability, not
universal robustness. AUPRO/pAUROC use the same streaming evaluator; no target
adaptation, maps, or weights. This is seed0 development evidence.

## COMPLETED — Source-fixed threshold and paired mechanism audit, 2026-09-22

Final collection: 32/32 complete and registered in the canonical paired ledger.
[Full analysis](../results/comparisons/source_mechanism_20260922/comparison.md):
all32 prior FF metric dictionaries and bank hashes match; within-class bank and
calibration hashes match across arms. CQ source1% D−B shift PRO +8.92754pp,
but FPR increase +0.72606pp worse and only4/8 classes meet no-worse inflation.
The prespecified composite mechanism gate fails. D−E shift PRO is only
+0.68055pp with +0.01961pp worse FPR inflation. Retain empirical shift detection
performance; do not claim nuisance rejection, source-threshold stability,
shot scaling, or seed stability. Initial sheet_metal evaluator interpolation
failure was recovered on the same GPU with prediction-matched PyTorch projection;
scoring/tolerances unchanged, failed evidence retained, 20 relevant tests pass.

User authorized immediate local + dsba3 execution of the proposed mechanism
and fixed-source-threshold study. [Locked protocol](../configs/methods/SOURCE_MECHANISM_20260922.md).
SM_B/C/D/E reproduce FF_B/C/D/E;16-shot seed0 full8,32 class evaluations.
Same frozen backbone/banks/resolutions/CQ. Extra16 support-disjoint regular
train/good images calibrate source thresholds before target evaluation: explicitly
16-support+16-calibration normal data, not16 total; development exposure declared.
Primary source nominal FPR1%, sensitivity0.1%/5%, raw and CQ, all native pixels
under float16 histogram contract. Source thresholds are never target-tuned.
Record fixed-point normal FPR, defect TPR/region-balanced PRO, per-layer raw and
normalized responses, deep/shallow/residual contributions, and scene-paired
regular→shift_1 changes. GT is evaluator-only; no dense maps or weights saved.

Local GPU access had failed inside the existing idle container while the host
GPU remained healthy. Restarted that SAME container (only sleep infinity was
running), with no recreation/allocation change. CUDA tensor smoke recovered.
Local and remote each81 tests pass. Actual DINO normal16 can smoke: observed
prediction exactly matches uninstrumented prediction; component projection
reconstruction maximum error1.90735e-6. Calibration pool validated for all8 classes.
Snapshot160ad2e8eb2a20e1f32abefc138109278666d192416c3535aca6ce844924bd9f.
Existing local4090 can/fabric; existing dsba3 GPUs0/1/2/3 sheet_metal,
rice/vial, fruit_jelly/wallplugs, walnuts. Explicit UUIDs and isolated remote
Python3.11.16/Torch2.6.0+cu124. Generic runner order D,B,C,E;14400s/class.
Status roots: `results/local_runs/bcma_docker_20260911/source_mechanism_20260922/lane_0`
and remote `/workspace/results_remote/source_mechanism_20260922/lane_0..3`.
Compact evidence was collected into `results/comparisons/source_mechanism_20260922`.
No shot/seed expansion was launched in this stage.
Interpretation must distinguish lower mechanical score contributions from
improved source-threshold transfer without worse defect PRO. Prior AUPRO
evidence remains valid regardless of whether this mechanism hypothesis passes.

## COMPLETED — RSG3 scaling × shallow-residual factorial, 2026-09-22

User authorized the planned five-arm comparison. [Fixed protocol](../configs/methods/FF_PROTOCOL_20260922.md):
FF_A MAD+ungated-add, FF_B q95+ungated-add, FF_C MAD+q99-residual,
FF_D q95+q99-residual (exact RSG3), FF_E q95+deep-only. Normal16 seed0,
fixed SD support manifest, full8,40 class evaluations on seen test_public.
Frozen backbone, four resolutions and four banks, CQ and common evaluator fixed.
Raw/CQ condition metrics and gate activation are evaluated after prediction;
no query labels enter method fitting/scoring. No maps or weights exported.
Local/remote each72 tests passed. Real DINOv2 normal16 can smoke gives exact
FF_D/reference readout equality (max absolute error0, native1024×2232).
Snapshot d4fa3a92beea8f7bbcd3a3ed87b8f4403cde8b61f79a9a54e17511d6dcf11a2b.
Existing local GPU: can/fabric; existing dsba3 GPUs0/1/2/3: sheet_metal,
rice/vial, fruit_jelly/wallplugs, walnuts. Explicit UUIDs, paired arms stay
on the same device/runtime. Remote Python3.11.16/Torch2.6.0+cu124.
Generic runner order D,A,B,C,E; timeout14400 seconds/class. Status roots:
`results/local_runs/bcma_docker_20260911/fusion_factorial_20260922/lane_0`
and remote `/workspace/results_remote/fusion_factorial_20260922/lane_0..3`.
Completed40/40, no recorded runtime retries,1,084 queries/arm. Canonical ledger
and paired CSV are collected.114 remote compact files match local SHA256;
class-wise support/query/GT, bank features, code/backbone/runtime match across
arms. D reproduces RR_A AUPRO/pAUROC exactly for all8 classes.
[Detailed analysis and conclusion](../results/comparisons/fusion_factorial_20260922/comparison.md)
and its analysis.json preserve the condition/gate/raw-CQ evidence.

Decision: retain RSG3 on the primary AUPRO metric. D=59.14576 versus E=58.67094;
the strongest evidence is residual gating over ungated addition (D−B+5.13007pp,
6/8 wins), plus tail scaling under the gate (D−C+1.66385pp,8/8 wins).
Shallow contribution D−E is small (+0.47482pp,5/8 wins; pAUROC−0.24459pp),
only+0.12630pp after excluding can. Screening criteria qualify seed replication,
not a final claim. No new experiment or default change was started by collection.

Shift suppression is not supported: paired good-image gate activation increases
in8/8 classes under shift_1 (macro+0.26778pp). Wallplugs gate effect flips from
regular+12.58832pp to shift_1−15.81119pp AUPRO. Shift_1 PRO at FPR0.1% is
3.07803pp worse for D than E after CQ. Raw-score contrasts retain the main
positive mean effects, so gains are not only CQ artifacts; extreme-FPR behavior
is a separate limitation. Normal16 alone cannot establish shot dependence.
Next proposed priority is paired D/E seed1–4 full8 (64 additional evaluations),
not authorized/launched automatically. Stop the completed40-cell budget here.

## COMPLETED — Performance-first proposal 1: region recovery, 2026-09-21

User authorized only proposal1. Comparator returns to the stronger RSG3;
QR1/geometry is not the objective. [Fixed protocol](../configs/methods/REGION_RECOVERY_20260921.md):
RR_A recovery-off versus RR_B feature/RGB/multiscale-guided region expansion,
16shot, seed0, fixed normal support16, full8,16 evaluations, seen development.
Only postprocessing changes; no backbone adaptation or extra backbone calls.
Local and dsba3 targeted tests each82 passed/1 skipped. Normal-only cached
DINOv2 smoke:18293/2285568 pixels raised, maximum unchanged, four backbone
calls per query in both arms. This is execution evidence, not performance.
Snapshot723131adc4791de0e460f279d316291063fa79770067a97ab50ead364379e272.
Existing local GPU can/fabric; dsba3 GPU0 sheet_metal, GPU1 rice/vial,
GPU2 fruit_jelly/wallplugs, GPU3 walnuts. Paired A/B stay on same device.
Runner timeout14400 seconds per class allows heavy four-scale sheet_metal;
not a sweep. Maps/weights off; logs retained. Status roots:
`results/local_runs/bcma_docker_20260911/region_recovery_20260921/lane_0`
and remote `/workspace/results_remote/region_recovery_20260921/lane_0..3`.
No other proposal or default change authorized by this trial.

Completed16/16 without retry. [Detailed result](../results/comparisons/region_recovery_20260921/comparison.md):
RR_A reproduces RSG3 at59.14576/77.10613 AUPRO/pAUROC; RR_B gives
59.09510/77.04420, deltas -0.05065/-0.06193 pp. AUPRO improves in3/8
classes and declines in5/8; no >2pp class loss. Mean per-image time increases
4.08076s to4.22155s (+3.45%). Recovery is active (0.19--0.70% native pixels
raised per class; no constant fallback), so this is a performance failure rather
than an inactive implementation. The positive-macro primary screen fails;
stop this fixed rule, do not adopt/sweep/replicate it, and retain RSG3.

## COMPLETED — Sequential trace-control and confidence trial, 2026-09-21

The user authorized sequential follow-up experiments. The trace-matched E
control completed16/16: the fixed directional/operator-effect criterion passes
at both shots, but this does not overturn D's failure to beat QR1. See the
[trace-matched analysis](../results/comparisons/pg_trace_match_20260921/comparison.md).
All64 operators match D's contraction trace within float32 rounding. Spectrum
and nuisance-energy suppression remain unmatched; no pure-direction causal
or unknown-defect-preservation claim.

The second single candidate attenuates QR1 by spatial agreement of unlabelled
query residual means. [Locked rule/budget](../configs/methods/SPATIAL_CONFIDENCE_20260921.md):
16 evaluations, full8, seed0, nested1/16shot, same frozen backbone/pipeline.
Normal-only smoke and55 local/remote targeted tests pass (one opt-in skip).
No tuning/sweep; stop the fixed rule if its QR1 improvement/safety gate fails.
Started only after E collection. Reuse local can/fabric; dsba3 physicalGPU0
sheet_metal, GPU1 rice/vial, GPU2 fruit_jelly/wallplugs, GPU3 walnuts, matching
the completed per-class comparators. Source snapshot6914f8ae3233c3e943a7e3c60161e9069979d843c9ff14c0238ce9e00ab5e363.
Both stages are now complete32/32 with no runtime retries. SC1 fails its
QR1 improvement gate at both shots despite improving on original features;
secondary safety gates pass. [Completed analysis](../results/comparisons/spatial_confidence_20260921/comparison.md).
Retain QR1 as comparator within this sequence; neither geometry nor this
confidence rule is adopted. Shared pipeline/provenance and all96 recovered
remote compact artifacts were verified. No additional search started.
Repeatedly seen development, not sealed final evaluation. No default change.

## COMPLETED — Protected normal-contrast geometry candidate, 2026-09-21

User authorized the proposed4-arm implementation/comparison, then explicitly
re-enabled local experiments. Current assignment: local existing container/GPU
for can/fabric; dsba3 existing container physicalGPU2 for the other6 classes.
All arms and both shots for each class stay on the same device/runtime.
[Protocol](../configs/methods/PROTECTED_GEOMETRY_20260921.md):64 evaluations,
PG_A QR0, PG_B QR1, PG_C unprotected closed-form nuisance contraction, PG_D
stable-normal-local-contrast-protected contraction. Frozen DINOv2-L, original
R1 pipeline and nested1/16shot seed0 support. C/D transform both support/query
features and refit support MAD in that space; no gradients or synthetic defects.
Linear protected-subspace invariance is only BEFORE final normalization, not
a guarantee of defect preservation. Related-work and existing GuardedMetric
overlap are acknowledged; no novelty claim is established by implementation.
41 targeted tests passed remotely,1 skipped. Geometry-off exact QR0/QR1 parity
and operator boundedness/protection are tested. No hyperparameter sweep or
automatic replication. Use actual status files, not this note, for completion:
local `results/local_runs/bcma_docker_20260911/protected_geometry_20260921/lane_0`
and remote `/workspace/results_remote/protected_geometry_20260921/lane_0`.

## COMPLETED — QR1 mechanism audit, remote-only at execution, 2026-09-21

User authorized finishing direction controls, then paired nuisance alignment,
defect preservation and fixed-correspondence/5NN readout diagnostics.
[Locked audit protocol](../configs/methods/FEATURE_MECHANISM_20260920.md).
QM reached60/64 after a local retry; the remaining C/D16 can/fabric runs move
to dsba3 after local OOM. Preserve the original partial shard and error logs.
User now prohibits local compute: experiments and tests run only in existing
dsba3 container on physicalGPU2 UUID GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639.
Existing isolated Python3.11/Torch2.6 runtime; recovery changes hardware relative
to local, which is a declared comparability limitation.42 targeted tests pass
remotely. Recovery is sequenced before the16 new full8 1/16shot audit runs;
failure stops the chain. No method/default or trainable-parameter change.
Remote status roots: `/workspace/results_remote/query_mechanism_recovery_20260920`
and `/workspace/results_remote/feature_mechanism_20260920`. Now complete: QM64/64
and FM16/16, registered in the canonical ledger;66 newly recovered compact files
match remote SHA256, repeat collection is a no-op. See the
[completed analysis](../results/comparisons/feature_mechanism_20260920/comparison.md).
Direction specificity, normal-shift alignment/residual stability and limited
fixed-NN/5NN readout gates pass; mean defect/normal separation fails. High
differential alignment is partly expected from the mean-residual estimator,
not independent proof of nuisance identification. QR1 changes the reference
bank, not backbone/query features. Keep it as a correction baseline; a stronger
task-inductive-bias claim requires a distinct selection/preservation principle,
not relabeling this translation. Planned diagnostic suite stops here; no new
method/sweep launched by collection. The local compute prohibition at that time
was lifted by the user for the subsequent protected-geometry experiment.

## PRIOR — QR1 direction-specific mechanism controls, 2026-09-20

User authorized experiments on the mechanism of the observed QR1 benefit.
[Protocol](../configs/methods/QUERY_MECHANISM_20260920.md):64 evaluations,
QM_A baseline, B fitted QR1, C equal-norm random direction, D reversed direction;
nested1/16shot seed0, full8, unchanged R1 two-scale comparator. Preserve actual
acquisition diagnostics and compare within-image normal/defect score margins,
ranking metrics and calibrated operating points. Norm matching is before bank
normalization, not score-magnitude matching. No cross-query donor data, new
trainable parameters, or default change.36 tests pass, one skipped; controls
finite/repeatable with identical banks. Results pending; no automatic replication.

## COMPLETED — Real acquisition diagnosis: shift exists, proposed cause fails, 2026-09-20

User authorized continuing with actual acquisition shift. Official AD2 paper
and local files confirm public lighting-condition images and per-condition GT.
[Protocol](../configs/methods/REAL_ACQUISITION_20260920.md):32 runs, QR0/QR1,
nested1/16shot seed0, full8, all public images. Primary shift_1 versus regular,
secondary exposure/other lights; paired scene sets and GT dimensions audited.
Condition-wise ranking metrics and fixed-source-threshold FPR/TPR are retained,
with leave-one-calibration-image-out sensitivity, not confidence intervals.
34 tests passed, one skipped. No method change, private evaluation or default
change. All32 runs completed and collected;4,336 diagnostic records verified.
H0 passes:1shot QR0 shift1 normal FPR+2.686091pp,7/8 positive. H1 fails:
1-minus16shot FPR interaction−1.841566pp,0/8 positive. H2 fails: correction
FPR−0.458443pp / defect TPR+0.343966pp; both-lower3/8. These decisions persist
at the tested LOO threshold endpoints. Wallplugs drives macro FPR degradation.
No new method is promoted. See [analysis/audit](../results/comparisons/real_acquisition_20260920/comparison.md).
This is seen-development, not final validation; calibration transfer versus
ranking loss remains an open distinction, not an established causal account.

## COMPLETED — Few-shot/shift diagnosis: framing gates fail, 2026-09-20

User authorized testing the new problem framing, not another CM variant.
[Protocol](../configs/methods/SHIFT_DIAGNOSIS_20260920.md):48 evaluations,
full8 seed0, nested1/16shot, QR0/QR1/spatial-only QR3. Fixed original R1 pipeline
at448/672 layers11/23, distinct from RSG3. Paired synthetic brightness0.7
diagnostics use16 separate normal calibration images and8 good/8 bad queries;
labels remain evaluator-only. This is not real acquisition-shift or strict
one-image deployment evidence. Evaluate the predeclared interaction/tradeoff
gates before proposing a new method.32 tests pass, one skipped; single-support
fit/predict smoke passes all three arms. All48 runs completed and were collected.
H1 baseline shot×brightness FPR interaction−0.172079pp, positive2/8 (required5).
H2 joint shifted FPR/TPR reduction occurs3/8 at1shot (required4); spatial
exclusion does not resolve the macro tradeoff. Do not promote a new method from
this framing. QR1 improves original-public AUPRO+0.95695/+0.66381pp at1/16shot,
but this is separate from the failed mechanism claim. See
[analysis and audit](../results/comparisons/shift_diagnosis_20260920/comparison.md).
No defaults changed or further runs launched. Real-shift provenance and
calibration transfer remain open questions, not established causes.

## COMPLETED — Coherent local correspondence pilot, not promoted, 2026-09-20

The user authorized the next experiment. [Fixed protocol](../configs/methods/COHERENT_MATCHING_20260919.md):
CM_A/B/C/D, full8 seed0, 32 evaluations, same frozen RSG3 baseline and normal
support selection. C measures shared-window versus independent-reference cost;
D is an appearance-only normal-marginal magnitude control, not a query-matched
causal control. No default change. Promotion requires +1pp AUPRO, pAUROC >=-0.5pp,
at least four improving classes and a positive leave-largest-gain-out macro.
63 unit/regression tests passed; actual DINOv2 normal-image A/B map max error
was zero, C/D finite. Synthetic splice diagnostic passed; this does not establish
real-world nuisance invariance. Existing local/dsba3 containers and assigned GPUs
were reused. All32 evaluations completed without failures and were collected in
the canonical ledger. C−A AUPRO−0.080607pp / pAUROC−0.188339pp, improving3/8
classes; excluding the best-gain class gives−0.210787pp AUPRO. The promotion
gate fails; no replication or default change. See the
[collected results and audit](../results/comparisons/coherent_matching_20260919/comparison.md).

## COMPLETED — Local relational control: small effect, not adopted, 2026-09-19

The current user authorized sequential experiments on local and dsba3. The frozen
experimental comparator is RSG3/DINOv2, 16 normal supports, seeds0–4, full8; this does
not change any implementation default. LR_A/B/C/D compare the baseline, exact
zero-weight control, sorted local star-edge discrepancy, and a shuffled-locality
null. There are no trainable parameters or pipeline changes.

The [recorded protocol](../configs/methods/LOCAL_RELATION_20260919.md) is resolved:
32 screening + 96 replication evaluations completed without failures. All five
full8 comparisons are collected and verified. See the
[final analysis and provenance audit](../results/comparisons/local_relation_20260919/comparison.md).
Mean paired C−A AUPRO +0.109944pp / pAUROC -0.040428pp; C−D AUPRO +0.118767pp.
C beats A and D in 5/5 seeds, but the +1pp practical adoption threshold fails.
Five classes regress in every seed; walnuts drives the macro gain. This is a
small, narrow effect, not zero effect or a rejection of all geometric methods.

No default change or further CQ/calibration/shot/dataset expansion is scheduled
for this candidate. Those conditional branches were skipped by the original
gate, not left unfinished. All data are seen development data. Novelty, intrinsic
backbone geometry repair and generalization remain unestablished. Historical
next-action lists below are not execution authorization. Remote logs and the
existing containers are retained; all 128 results and 16 full8 macros are indexed
in the canonical ledger with previous records preserved.

## OPEN LEAD — Tail-calibrated fusion TC1: mechanism confirmed, adoption not yet, 2026-09-16

32/32 local, no failures. MS1 base; only `feature_layers` and the new
`score_normalization` differ. Macro x100:

| Arm | layers | normalisation | AUPRO | pAUROC |
|---|---|---|---:|---:|
| TC0 | [5,23] | mad (control, = LF2) | 54.11123 | 76.53857 |
| **TC1** | [5,23] | **tailscale** | **57.25588** | **77.73743** |
| TC2 | [5,23] | tail (-log(1-F)) | 57.04788 | 77.63368 |
| TC3 | [5,11,23] | tailscale | 57.11975 | 75.92282 |

**Mechanism test PASSED** (pre-declared: fruit_jelly >= +1.2pp and vial >= +1.0pp
over TC0). TC1 gives +13.29 and +9.34; TC2 +15.13/+9.90; TC3 +12.34/+9.37 —
exceeded roughly tenfold. TC1's per-class gain tracks the pre-measured layer-5
tail blowup at **pearson r = +0.934** (n=8), and fabric, pre-declared as not
explained by the mechanism (0.89x blowup), indeed does not recover. The
prediction held in both directions.

Confirmed mechanism: dividing each (layer, resolution) component by its
leave-source-out normal **MAD** standardises the centre, while AUPRO@0.05
operates in the tail. A component whose normal tail blows up (fruit_jelly
L5@1120 q95 = 35.70 vs 4.87 elsewhere) then eats the FPR budget under mean
fusion. `tailscale` divides by that component's own LOO normal q95 instead —
still linear, so magnitude is preserved — and the blowup disappears. LF2's
collapse on fruit_jelly and vial was a calibration artefact, not layer 5 being
uninformative there.

**Adoption test FAILED.** Pre-declared line was macro AUPRO >= 57.61
(MS1 + 0.86pp floor). TC1 is +0.506 over MS1, inside the floor. **No default is
changed; MS1 remains the reference detector on the primary metric.**

Recorded for what it is: TC1 and TC2 are the first configurations ever to pass
the older CA/CB joint target (AUPRO >= 56.75 and pAUROC >= 77.24) that CA1-4,
CB1-6, MS and HR all failed. TC1's pAUROC 77.737 is the highest full8 value in
the ledger (previous best CA3 77.566) and TC1 beats both MS1 and CB5 on both
metrics at once.

**This supersedes the 2026-09-15 closure of the layer axis.** That closure was
correct on its own terms — no support-only *selector* exists, and five criteria
failed at 5/8. Calibration turns out to be a different lever that needs no
selector: `tailscale` is fitted per component from supports alone, is not
class-conditional, and selects nothing. The `[5, 23]` stack is usable after all.

Next, in priority order.
1. **Seeds.** >= 3 bank seeds for TC1 and for MS1 before any claim; the +0.506
   AUPRO margin is currently unresolvable against the floor. This is now the
   binding step, not a further method search.
2. `tailscale` on the `[11, 23]` stack and on CB5's bagged bank — the gain here
   was demonstrated only where a blowup existed, and CB5 is the other top arm.
3. `tail_quantile` is fixed at 0.95 and untuned; any tuning must use supports
   only and be declared before measurement.
4. `can` is unmoved by every arm (23.4-23.8) and still caps the macro.

Code: `score_normalization: mad|tailscale|tail` plus `tail_quantile` in
reference_matching.py; default `mad` is byte-identical and covered by
`test_tail_normalisation_defaults_off_and_is_byte_identical`. 34 tests in
test_reference_matching.py, 619 passing overall (3 pre-existing failures in
test_psma_mr_dsba1_orchestration.py, unrelated and present before this change).
[Report](../results/local_runs/bcma_docker_20260911/tail_calib_20260915/comparison.md).

## DECISION — Primary metric declared: AUPRO@0.05, 2026-09-15

User declaration (2026-09-15), recorded before the layer results were produced.

- **Primary: `seg_AUPRO@0.05`.** Rationale, in this order: AUPRO normalises per
  connected component, so small defect regions count equally with large ones, and
  region-level localisation is the stated target of this project (CVPR 2027
  region-coherence direction). The separation from prior methods also happens to
  be larger on this metric (MS1 vs SuperAD-16: AUPRO +3.26pp vs pAUROC +0.24pp),
  but that is a consequence, not the reason — recording it the other way round
  would be post-hoc metric selection.
- **Secondary guardrail: `seg_AUROC@0.05` (pAUROC).** A candidate maximises AUPRO
  without a material pAUROC regression; pAUROC alone cannot carry a decision.
- `seg_F1_oracle` stays a diagnostic: its threshold is chosen with evaluation
  ground truth and it is never used for selection.
- Consequence for the open trade-off: MS1 (56.750/76.434) is the reference
  detector on the primary metric; MS2/CA3 remain the pAUROC-priority variants.

## Closed — `can` root cause identified; bank and resolution axes closed, 2026-09-15

Read-only diagnostics on the local GPU (scratch, not benchmark runs; token-level
NN-distance and token AUROC against a 16-support bank, `_regular` variants).
These are proxies for AUPRO, not substitutes.

1. **Not a bank-coverage problem.** `can` normal tokens match the 16-support bank
   *better than any other class* (median NN cosine distance 0.026, vs rice 0.075,
   walnuts 0.101, fabric 0.118). Defect tokens sit almost on top of them (0.043).
   An earlier rotation/coverage hypothesis was measured and **refuted**.
2. **Not a resolution problem.** Token AUROC for `can` is flat across a 3.5x input
   range: 448 0.707 / 672 0.646 / 896 0.674 / 1120 0.685 / 1568 0.702. `fabric`
   over the same range climbs monotonically 0.784 -> 0.888. Tiled/high-resolution
   inference for `can` is therefore **closed**, including the >=1344 idea.
3. **It is a representation problem, and the layer stack was the cause.** Token
   AUROC by depth at res 448 (15 queries): L5 0.800 / L11 0.766 / L17 0.729 /
   L23 0.677, strictly decreasing. `fabric` is the opposite and peaks at L17.
   `reference_matching.py` hard-rejected any stack other than [11, 23], so every
   R1-derived suite (R/P, QR, RC, MC, VR, CG, CR, TF, BC, HR, MS, CA, CB) scored
   `can` on the two layers that carry least of its signal. This explains why
   `can` never moved (13.3-15.3) through every bank and resolution experiment.
4. Supersedes the earlier reading that only feature adaptation moves `can`: D3's
   24.3/67.6 is largely recoverable by layer choice alone, with frozen features
   and no training (LA2 23.45/64.56).

Code change: the R1/R2 layer whitelist is removed (R0 still pinned to [23]); the
[11, 23] default is unchanged and covered by `test_r1_default_layers_are_unchanged`.

## Closed — Layer ablation LA0-LA3 (`can` only) and LF2 [5,23] full8, 2026-09-15

`can`-only ablation on the MS1 base, layers only (x100): LA0 [11,23] 14.90512 /
58.13908 (reproduces the recorded MS1 `can` to 3e-5) · LA1 [5,11] 20.05515 /
59.12422 · **LA2 [5,23] 23.44978 / 64.56015** · LA3 [5,11,23] 21.99259 / 62.34070.
All three shallow arms passed the pre-declared `can` rule (> LA0 + 2.5pp); LA2's
+8.54pp is 3.5x the BC-derived `can` sampling range. Shallow must be paired with
deep, not with the middle: [5,23] > [5,11,23] > [5,11] >> [11,23].

Full8 LF2 = [5,23] vs the recorded MS1 row, 8/8 complete, no failures:

| Class | MS1 AUPRO | LF2 AUPRO | d | \|d\|/sd |
|---|---:|---:|---:|---:|
| wallplugs | 38.74531 | 49.74502 | **+11.00** | 10.0 |
| can | 14.90509 | 23.44978 | **+8.54** | 8.3 |
| sheet_metal | 67.19308 | 65.83558 | -1.36 | 2.3 |
| walnuts | 72.33211 | 69.14530 | -3.19 | 2.8 |
| rice | 66.81898 | 63.08384 | -3.74 | 5.3 |
| fabric | 59.42012 | 50.65189 | -8.77 | 4.1 |
| vial | 77.37467 | 68.02675 | -9.35 | 18.7 |
| fruit_jelly | 57.21231 | 42.95165 | -14.26 | 23.8 |
| **macro** | **56.75021** | **54.11123** | **-2.639** | |

Pre-declared pass rule (macro AUPRO >= 57.61 and pAUROC >= 75.43): **FAILED on
the primary metric** (pAUROC passed at 76.53857, +0.10). **[5,23] is not adopted;
the default stack stays [11, 23].**

Findings. (1) Optimal feature depth is strongly class-dependent and the per-class
effect is an order of magnitude larger than anything else in this ledger: every
class moves 2.3x-23.8x its own sampling sd, where the whole 28-stage R1 add-on
programme sat inside 1x. (2) The sign is anti-correlated with the MS1 per-class
score — depth helps exactly where the deep stack was failing (wallplugs, can, the
two worst MS1 classes) and hurts where it was working (vial, fruit_jelly, rice).
(3) macro pAUROC is flat (+0.10) while macro AUPRO falls 2.64: the two metrics
decouple under a layer change, which is why the primary-metric declaration had to
precede the run.

Diagnostic upper bound, NOT a method and never to be quoted as a score: picking
the better of {MS1, LF2} per class using TEST AUPRO gives macro 59.19326 (+2.44
over MS1). This is oracle selection with evaluation labels; it only sizes the
opportunity. The legitimate version needs a selector that reads only the supports.
[can report](../results/local_runs/bcma_docker_20260911/layer_ablation_can_20260915/comparison.md) ·
[full8 report](../results/local_runs/bcma_docker_20260911/layer_full8_20260915/comparison.md).

## CLOSED — Layer axis: no support-only selector found; router abandoned, 2026-09-15

Round 2 of the same 8-point frame (same ground truth, same pre-declared 7/8 pass
line) tested the two candidates that construct the missing numerator:

- **(A) `synthetic_auroc`** — CutPaste patches (6 per image, side 0.5-2% of the
  short edge, identical settings for every class) pasted onto a support, scored
  against a bank of the other 15 clean supports; token AUROC of pasted vs rest.
- **(C) `dynamic_range`** — median distance to a random bank token over median
  distance to the nearest bank token, i.e. how much farther *different* normal
  content sits than the matched neighbour.

Result: **(A) 5/8, (C) 5/8.** Both at chance, both below the pass line, and both
wrong on `can` and `wallplugs` — the same two positive cases that defeated the
round-1 normal-tail statistics. Pearson r vs the AUPRO gap: (A) +0.308,
(C) -0.322 (slightly inverted).

Five support-only criteria have now been tested across two rounds
(`fpr_budget_ratio`, `q99/q50`, `excess_kurtosis`, `synthetic_auroc`,
`dynamic_range`). All five score 5/8 and all five fail on both positive cases.
The agreement is not coincidence: on every property measurable from normal
supports, the deep stack genuinely *is* better — tighter normals, larger dynamic
range, equal synthetic separation. What makes shallow win on `can` and
`wallplugs` is invisible to all of them.

Why (A) failed is the most informative part, and is quantified: for `can` the
synthetic separation is 0.7457 (deep) vs 0.7453 (shallow), a gap of 0.0004, while
the *real* defects give 0.677 vs 0.800. **CutPaste corruptions are detected
equally well by both stacks; real `can` defects are not.** A pasted patch is a
region of wrong-but-coherent content, which deep semantic features represent
fine; the real defect is a sub-token speck on printed text, which they discard.
Naive self-supervised corruption does not span the real defect manifold for these
classes, so it cannot rank layers for them.

**Decision: the layer axis is closed as a method.** MS1 ([11, 23], four scales,
bank cap 256) remains the reference detector. Reasons: no selector exists after
five attempts; the oracle ceiling is only +2.44pp macro even with a perfect
selector; and any further synthetic-defect design that is tuned until it "works"
would be fitting the corruption model to knowledge of the test defects, which is
prohibited here.

What survives, and is worth keeping as analysis rather than method: optimal
feature depth is strongly class-dependent, the per-class effect (2.3x-23.8x each
class's sampling sd) dwarfs every other lever in this ledger, its sign is
anti-correlated with the MS1 per-class score, and it is not predictable from
normal supports by any of five natural criteria. That last clause is a real
limitation of few-shot anomaly detection, not merely a failed experiment.

Reopening condition: a mechanism argument for why a specific support-only
statistic should see the numerator, declared before it is measured. Trying more
statistics against this same 8-class answer key is now selection on 8 points and
must not be treated as evidence.

## Closed — Support-only router criterion (B), normal-tail statistics: NO SIGNAL, 2026-09-15

Feasibility probe before building a layer router: can a statistic computed from
the 16 supports alone predict which stack wins per class? Frame: 8 classes, the
LF2-vs-MS1 winner as ground truth, pre-declared pass line 7/8 (6/8 weak, p~0.14;
random 4/8). Primary candidate declared in advance was `fpr_budget_ratio` =
q95/q50 of the leave-source-out normal score, chosen because AUPRO@0.05 caps FPR
at 0.05 so the operating threshold sits near the normals' 95th percentile.

Result: **5/8 for all three statistics** (`fpr_budget_ratio`, `q99/q50`,
`excess_kurtosis`) — at chance, and below the pass line. No post-hoc rescue is
available since all three fail together.

The failure is diagnostic, not noisy: **all three are wrong on both positive
cases (`can`, `wallplugs`)**, the only classes where shallow wins. For `can` the
deep stack has the *lighter* normal tail (1.6915 vs 1.7419) precisely because it
absorbs everything, defects included. Pearson r between the statistic gap and the
AUPRO gap is +0.57..+0.61 but is carried by fruit_jelly and vial, with no
discriminative power where it is needed.

Root cause, and the reason this generalises: the quantity wanted is
defect-response / normal-response. Supports measure only the **denominator**. Any
purely-normal statistic is blind to the numerator, so this family is closed on
principle, not just empirically. This is the degenerate solution predicted before
the run: an objective fitted to normals alone selects the most absorbing layer.

Consequence for the router design: the objective must be **contrastive**. The two
surviving candidates construct a numerator — (A) synthetic defects pasted on
supports, (C) dynamic range (distance to different normal content over distance to
the matched neighbour). Probe (A)+(C) runs in the same 8-point frame with the same
pre-declared 7/8 rule.

Caveat: 8 binary predictions is a weak test in either direction. What is
established is that normal-tail statistics are unusable and systematically
inverted on the positive cases, not that no support-only signal exists.

## Closed — Bank capacity / sampling-noise diagnostic BC0–BC4 complete, 2026-09-14

40/40 local runs complete, no failures, collected 20:40 KST (merged view of the
single local lane; `--reference-method reference_matching_R1` audit: content
matched 8/8; metrics within 1e-8 only 2/8, but the largest per-class deviation is
1.3e-6 (×100 units), i.e. GPU nondeterminism, not a code change).

| Stage | Bank | AUPRO | pAUROC |
|---|---|---:|---:|
| BC0 | cap 256, seed 0 (R1 control) | 52.47034 | 75.99215 |
| BC1 | cap 256, bank_seed 1, same supports | 52.37557 | 76.20999 |
| BC2 | cap 256, bank_seed 2, same supports | 53.23841 | 76.42696 |
| BC3 | cap 1024 (all 448 tokens) | 53.44014 | 76.90004 |
| BC4 | cap 4096 (all tokens, deterministic) | 52.83772 | 76.79544 |

Sampling-noise floor (BC0/1/2, identical supports, only the 256-token subsample
changed): macro AUPRO range 0.86pp, pAUROC range 0.44pp; per-class AUPRO sd
0.5–2.1pp (fabric 2.12, walnuts 1.12, wallplugs 1.10, can 1.03). Consequence:
every R1-derived suite of 2026-09-11..14 (QR/RC/MC/VR/CR/CG/TF, 28 stages,
AUPRO 52.3–53.6 / pAUROC 75.5–76.1) lies inside this floor; none of their
per-class or macro deltas is evidence. Capacity: BC3/BC4 raise pAUROC by
+0.9/+0.8pp (above every R1-family value) with AUPRO +1.0/+0.4pp, but the AUPRO
gain does not clear the 0.86pp floor; coverage alone does not explain the
fabric/wallplugs gap to SuperAD-16. Decision: R1 scoring add-ons are closed;
future claims require ≥3 bank seeds (or a deterministic full bank) and effects
beyond the floor. [Report](../results/local_runs/bcma_docker_20260911/bank_capacity_20260914/comparison.md).

## Closed — MCP1 masked context prediction (external Codex session): STOPPED, 2026-09-15 11:05 KST

A parallel Codex session implemented `masked_prediction_MCP1` (frozen DINOv2-L,
target layer 11 predicted from layer-23 context of a separately image-masked
input, cross-attention predictor, 128 AdamW steps/class, four scales, no bank)
and launched 8 classes at ~10:5x. Completed local runs: fruit_jelly 38.03/68.76,
vial 20.43/59.71 versus MS1 57.21/76.84 and 77.37/68.95 - a 19 pp and 57 pp
AUPRO collapse. This repeats the masked_prediction P0-P2 family already closed
in this ledger (14-16 AUPRO). The four dsba3 runs still in flight were stopped
with the user's approval and the GPUs reassigned to CB; the two completed local
runs stay as evidence. Do not reopen feature/pixel masked-prediction without a
new mechanism argument.

## CLOSED — Layer axis closed for FUSION as well as selection, 2026-09-16 10:30 KST

Two independent attempts to recover the class-dependent layer optimum with a
fixed, class-agnostic rule both failed. Together with the five failed
support-only selectors, the layer axis is now closed in every direction we can
justify.

### TB4 / TB5 — fixed fusion rules over tail-calibrated layers (full8)

| arm | layer fusion | AUPRO | pAUROC |
|---|---|---:|---:|
| TB1 ([11,23]) | mean | **57.744** | **77.350** |
| TB3 ([5,11,23]) | mean | ~56.8 | ~76.4 |
| TB5 ([5,11,23]) | quantile 0.90 | 55.675 | 76.594 |
| TB4 ([5,11,23]) | max | 54.337 | 76.359 |

Pass rule (AUPRO > 57.744 AND pAUROC >= 77.350): **both FAIL**, and they fail
monotonically in how disjunctive the rule is: mean > q0.90 > max. The mechanism
sub-test (`can` > 20 AND `fabric` > 55) fails for both - TB4 has can 23.16 but
fabric 48.65, TB5 can 23.36 / fabric 50.72.

The prediction that tailscale would rescue max was **wrong**. Putting layers in
common FPR units is necessary but not sufficient: a maximum still takes the worst
case over three layers at every pixel, so every layer's normal-side noise enters
the score, and `wallplugs` collapses (42.38 mean -> 35.46 q0.90 -> 29.34 max).
Disjunctive fusion buys the shallow layer's defect sensitivity at the price of
all three layers' false-positive mass.

### LR — per-pixel local reliability weighting (reduced 4-class probe)

Base [5,11,23] + bagging + tailscale; classes `can`, `fabric`, `wallplugs`, `rice`.

| arm | mode | 4-class AUPRO | pAUROC | can | fabric |
|---|---|---:|---:|---:|---:|
| LR0 | off (control) | 47.252 | 75.165 | 22.93 | 50.16 |
| LR1 | local scale | 45.978 | 75.240 | 20.92 | 51.34 |
| LR2 | local scale + precision weights | 46.035 | 75.490 | 19.00 | 49.57 |
| LR3 | as LR2, lambda=2 (more local) | 39.720 | 71.433 | 10.64 | 47.42 |
| LR4 | as LR2, lambda=32 (more global) | 47.672 | 75.778 | 22.45 | 50.62 |

1. **Implementation sanity: PASSED.** LR4 (lambda=32) sits within 0.2-1.6 sd of
   LR0 on every class, as required of a configuration that should converge to the
   global scale. The local path is wired correctly, so the rest is readable.
2. **Mechanism: FAILED.** No arm reaches `can` > 20 AND `fabric` > 55; fabric
   never exceeds 51.34.
3. **Performance: FAILED.** Every genuinely local arm is BELOW the control, and
   the ordering is monotone in how local the estimate is: lambda=32 (47.67) >
   off (47.25) > weight (46.04) > scale (45.98) > lambda=2 (39.72). The more the
   scale is estimated locally, the worse it gets.

**Why it fails, and it is not an implementation detail.** Dividing by a locally
estimated normal scale is self-defeating for this task. A defect region is
exactly a place where the local neighbourhood of bank tokens is atypical, so the
local leave-source-out distances there are already large; dividing the defect's
own distance by that inflated local scale suppresses the defect. The global
constant does not have this problem because it is dominated by the (vastly more
numerous) ordinary normal locations. LR3 makes this explicit: the most local
setting destroys `can` (22.93 -> 10.64).

The denominator/numerator argument therefore survives in a sharper form: a
support-only statistic is not merely blind to the defect response, it is
**positively correlated with it**, so using it as a divisor cancels the signal it
was meant to calibrate. This is why local calibration cannot substitute for
knowing which layer sees the defect.

### Consequence

The layer axis is closed: per-class selection (5 criteria at chance), fixed
fusion (mean/quantile/max, all fail), and per-pixel reliability weighting (fails,
with a mechanism). The remaining oracle gap of -1.45 vs TB1 is recorded as an
**analysis result** - class-dependent feature depth that is provably not
recoverable from normal supports - and not as headroom. TB1 + CQ (58.671 /
77.351) stands as the leading configuration.

## DECLARED BEFORE RUN — LR: local reliability weighting (reduced 4-class probe), 2026-09-16 10:05 KST

This is the response to the observation that the contribution so far is a stack
of small engineering choices with no single idea, and that the class-dependent
layer optimum is being left on the table (oracle 59.19 vs TB1 57.74).

**Why this is not the closed selector family.** The recorded closure says: a
support-only statistic measures only the DENOMINATOR of defect-response /
normal-response, so it cannot rank layers. That argument is about choosing a
layer PER CLASS. It does not apply to a per-pixel weighting, because the
quantity there is "how tight is this layer's normal model AT THIS LOCATION",
which is purely a denominator quantity and is directly measurable from the 16
supports leave-source-out. One fixed rule for all eight classes; no per-class
choice; no test data.

**Mechanism.** `_fit_scales_for_resolution` already computes a leave-source-out
nearest-neighbour distance for every bank token, then discards all of it into a
single scalar per (resolution, layer, bag). Keeping it per token gives a local
estimate of the normal model's looseness. A deep layer is tight on object
structure and loose on fine print; a shallow layer is the reverse. So:

    now:  s_l(q) = d_l(q) / q95_l                    one scalar per layer
    LR1:  s_l(q) = d_l(q) / sigma_l(q)               local, k nearest bank tokens
    LR2:  sum_l w_l(q) s_l(q) / sum_l w_l(q),  w_l = 1 / sigma_l(q)^2
    shrinkage: sigma = (k*local + lambda*global) / (k + lambda)

Inverse-variance weighting is the optimal combination of independent noisy
estimates; uniform mean (today) and max (TB4) are its two degenerate limits.
The current global tailscale is exactly LR at lambda -> infinity, so the existing
method is a special case and lambda is the ablation axis.

**Reduced probe, local GPU, 4 classes x 5 arms = 20 runs.** Classes chosen for
diagnostic power, not for a macro number: `can` (visible only at layer 5),
`fabric` (only at layer 23) - the decisive pair - plus `wallplugs` (sub-patch,
both layers) and `rice` (fine texture, the class global tailscale already helped).
Base for every arm: [5,11,23] + 4 bagged banks + tailscale.
LR0 off (= TB3 control) · LR1 scale · LR2 weight · LR3 weight lambda=2 ·
LR4 weight lambda=32.

Pass rules, declared now:
1. **Implementation sanity**: LR4 (large lambda) must be close to LR0. If it is
   not, the local path is wrong and nothing else may be read.
2. **Mechanism**: LR1 or LR2 achieves `can` > 20 AND `fabric` > 55 at the same
   time - the exact condition arithmetic-mean fusion failed three times over
   ([11,23] 17.58/55.60, [5,11,23] 22.93/50.16, [5,23] 24.39/46.80).
3. **Performance**: 4-class macro above LR0 by more than each class's BC sampling
   sd.
4. **On failure**: local reliability cannot substitute for the numerator either;
   the layer axis closes for good and the -1.45 gap is recorded as an analysis
   result, not headroom.

A reduced probe cannot produce a full8 claim; survivors go to full8 afterwards.

## DECLARED BEFORE RUN — TB4/TB5 layer-fusion rule, 2026-09-16 09:30 KST

Motivated by the user's point that the class-dependent layer optimum is real and
that a fixed class-agnostic setting is leaving measurable performance behind: the
oracle per-class choice of {MS1, [5,23]} is 59.19 versus TB1's 57.74, i.e. -1.45,
which is four times everything the whole bank-composition axis produced.

What is closed is (a) test-label per-class selection and (b) choosing the layer
per class from normal-support statistics (five criteria, all at chance). A third
option was never tried: a FIXED rule, identical for all eight classes, that lets
the score decide per pixel which layer to trust. That is not per-class selection.

Code inspection showed layer fusion has always been a hard-coded arithmetic mean;
the configurable `scale_fusion` applies to the resolution axis only. So one cell
of a 2x2 is untested:

| | mad | tailscale |
|---|---|---|
| mean over layers | CB5 57.075 | TB1 57.744 |
| max over layers | SF1 55.814 (scale-axis proxy, negative) | **untested** |

Max previously failed because under `mad` each layer is divided by its own MAD,
which standardises the centre but not the tail, so the fattest-tailed component
wins every maximum and eats the FPR budget. `tailscale` divides by the LOO q95,
putting layers in common FPR units, which is precisely the precondition a maximum
needs. If `can` is only visible at layer 5 and `fabric` only at layer 23, a
calibrated max picks whichever fires without knowing the class.

New arms on [5,11,23] + bagging + tailscale: **TB4** = `layer_fusion: max`,
**TB5** = `layer_fusion: quantile, layer_fusion_q: 0.9` (max's softened form,
guards against one runaway component).

Pass rules, declared now:
- Detector: AUPRO > 57.744 (TB1) AND pAUROC >= 77.350 (TB1).
- **Mechanism sub-test**: `can` > 20 AND `fabric` > 55 simultaneously - the exact
  condition arithmetic mean failed three times over
  ([11,23] can 17.58 / fabric 55.60; [5,11,23] 22.93 / 50.16; [5,23] 24.39 / 46.80).
- On failure: no fixed rule, mean or max, can hold both ends. The layer axis is
  then closed for fusion as well as for selection, and the remaining -1.45 is
  recorded as an analysis contribution (class-dependent depth that is not
  predictable from supports), not as recoverable headroom.

Implementation: `layer_fusion` / `layer_fusion_q` in reference_matching.py,
R1-only, default `mean` byte-identical, bag averaging order unchanged.
115 passed / 16 skipped.

## DECLARED BEFORE RUN — TB1 clean re-run + TB3 three-layer fusion, 2026-09-16 08:20 KST

Two arms, 16 runs, dsba3 GPU0-3.
- **TB1 (re-run)**: identical config to the TB1 above; needed only because that
  run root is `invalid_run` and therefore has no collector row. Must reproduce
  57.744 / 77.350; any deviation beyond float noise invalidates the earlier read.
- **TB3**: [5, 11, 23] + 4 bagged random-256 banks + tailscale. Motivation:
  TB1 ([11,23]) is the fabric/AUPRO arm and TB2 ([5,23]) is the can/pAUROC arm;
  the layer stack is a per-class trade-off that cannot be selected from supports,
  so the only legitimate move is to FUSE all three layers into one score. TC3
  tested [5,11,23]+tailscale without bagging (57.120 / 75.923); bagging is worth
  +0.5 to +0.7 elsewhere and its pAUROC contribution is what TC3 lacked.

Pass rules, declared now:
- TB1 re-run: reproduce 57.744 / 77.350 within 0.05.
- TB3 (detector only): AUPRO > 57.744 (TB1) AND pAUROC >= 77.350 (TB1).
- TB3 + CQ (the real target): AUPRO > 58.671 AND pAUROC >= 77.351, i.e. beat the
  current leader on the primary metric without giving up its guardrail.
- Secondary, mechanism: does adding layer 5 keep can above 20 (TB2 24.39) while
  keeping fabric above 55 (TB1 55.60)? If not, three-layer fusion cannot hold both
  ends and the layer axis is closed for fusion as well as for selection.

## RESULT — TB1 + CQ is the new leader; the two levers are orthogonal by class, 2026-09-16 08:15 KST

TB1 = [11,23] + 4 bagged random-256 banks + tailscale (8/8, numbers valid; the
enclosing run root is marked invalid because TB2 failed in the same lanes on a
stale remote snapshot - a missed `update-code` on my relaunch, since fixed).
TB2 = [5,23] + bagging + tailscale, 8/8 clean. CQ = frozen
`component_quantile(percentile=99, q=1.0)` applied offline to dumped maps; the
TB1 dump reproduces the TB1 run to 3 decimals (57.743/77.349 vs 57.744/77.350).

| config | AUPRO | pAUROC |
|---|---:|---:|
| MS1 | 56.750 | 76.434 |
| CB5 | 57.075 | 77.096 |
| TC1 | 57.256 | 77.737 |
| TB2 ([5,23]+bag+tail) | 57.601 | **78.127** |
| TB1 ([11,23]+bag+tail) | 57.744 | 77.350 |
| TC1+CQ | 57.988 | 77.510 |
| CB5+CQ | 58.041 | 77.129 |
| **TB1+CQ** | **58.671** | 77.351 |

Pre-declared rule for TB1+CQ (AUPRO > 58.041 AND pAUROC >= 77.096): **PASSED**,
and the separate additivity check (CQ gain >= +0.7) passed at +0.927.

**Why the levers compose here but not on TC1.** CQ's gain by base: CB5 +0.966,
TB1 +0.927, TC1 +0.732. The mechanism is that tailscale and CQ repair different
classes. Over CB5, tailscale (TB1) gains on the fine-texture classes - rice
+1.39 (2.0 sd), sheet_metal +1.08 (1.8 sd) - because there the high-resolution
component carries a fat tail that eats the FPR budget; it barely touches fabric
(+0.36), whose measured tail blowup is 0.89x, i.e. absent. CQ gains on the
sub-patch classes - fabric +3.27, wallplugs +5.80 - by completing partially
covered components. The two sets overlap only at wallplugs. On the [5,23] stack
(TC1) tailscale had already recovered wallplugs to 48.49, so CQ had less left to
do there (+2.86 instead of +5.80) and the combination was sub-additive. On
[11,23] the two act on disjoint failures and add almost fully.

TB1+CQ vs CB5 per class (AUPRO, sd units): wallplugs +7.87 (7.2) · fabric +3.72
(1.8) · rice +2.08 (2.9) · sheet_metal +1.56 (2.6) · can +0.13 (0.1) ·
vial -0.54 (1.0) · walnuts -1.00 (0.9) · fruit_jelly -1.05 (1.7). Five of eight
(sign p=0.36), but four gains exceed 1.8 sd while no loss exceeds 1.7.

**TB2 is the pAUROC record (78.127)** and keeps `can` at 24.39 - by far the best
value that class has ever had - but its AUPRO (57.601) is below TB1's because
[5,23] costs fabric (46.80 vs 55.60). The layer stack remains a per-class
trade-off that cannot be selected from supports.

Ledger note: TB1's run root is `invalid_run` for collector purposes, so TB1 and
TB1+CQ are not CSV rows yet. A clean TB1 re-run is required before any claim.

## RESULT — CQ x TC1 are only partly additive; two co-leaders, 2026-09-16 00:55 KST

The frozen CQ transform was applied to TC1's maps (dumped locally, 1084 maps;
the dump run reproduces the ledger TC1 row 57.25588 / 77.73743 exactly).

| config | AUPRO | pAUROC |
|---|---:|---:|
| CB5 | 57.075 | 77.096 |
| TC1 ([5,23] tailscale) | 57.256 | 77.737 |
| **CB5 + CQ** | **58.041** | 77.129 |
| **TC1 + CQ** | 57.988 | **77.510** |

CQ helps on both bases (+0.966 on CB5, +0.732 on TC1) but the two levers are
**sub-additive**: the combination does not reach 57.256 + 0.966. The overlap is
visible per class - CQ gives wallplugs +5.80 on CB5 but only +2.86 on TC1,
because TC1 has already lifted wallplugs from 39.69 to 48.49. Both levers repair
the same deficiency (sub-patch components scoring too low) by different means,
so their gains partly coincide. fabric is the exception and is genuinely
additive (+3.27 on CB5, +3.77 on TC1).

Consequence: there is no single winner. CB5+CQ leads the primary metric by 0.053
over TC1+CQ, which is far inside the 0.86 pp floor; TC1+CQ leads pAUROC by 0.381.
Treat them as co-leaders pending seeds. TC1+CQ is arguably the better candidate
since it wins the guardrail without a meaningful AUPRO loss, and TC1 also carries
`can` at 23.78 vs CB5's 17.20 - the only configuration that has ever moved that
class without a per-class choice.

Still open and running: TB1 ([11,23] + bagging + tailscale) and TB2 ([5,23] +
bagging + tailscale), i.e. whether bagging and tailscale compose. A first
partial TB1 (6/8, discarded for a config error in TB2 but numerically valid)
showed wallplugs +2.69 and rice +1.39 over CB5, so tailscale does act on the
[11,23] stack too - more than the handover predicted.

## RESULT — Component-max fill (CQ) PASSES on full8, 2026-09-15 23:05 KST

Frozen transform `component_quantile(percentile=99.0, q=1.0)` applied to CB5's
saved maps for all eight classes. Maps came from two dump runs (MD1 four classes,
MD2 four classes) whose online metrics reproduce the recorded CB5 row exactly to
4 decimals; the offline harness's `identity` pass reproduces the CB5 full8 macro
(57.075 / 77.096) exactly, so the only difference below is the transform.

| | AUPRO | pAUROC |
|---|---:|---:|
| SuperAD-16 | 53.492 | 76.186 |
| MS1 | 56.750 | 76.434 |
| CB5 | 57.075 | 77.096 |
| **CB5 + CQ** | **58.041** | **77.129** |

Pre-declared pass rule (AUPRO > 57.075 AND pAUROC >= 76.434): **PASSED on both**.
The macro AUPRO gain of +0.966 is the first result in this ledger to clear the
0.86 pp sampling floor over the previous best, and pAUROC does not regress
(+0.033).

Per class (AUPRO, delta vs CB5, in units of that class's BC sampling sd):
wallplugs 39.69 -> 45.48 (+5.80, **5.3 sd**) · fabric 55.24 -> 58.51 (+3.27,
1.5 sd) · rice +0.68 (1.0) · sheet_metal +0.42 (0.7) · can -0.23 (0.2) ·
vial -0.59 (1.1) · walnuts -0.73 (0.7) · fruit_jelly -0.88 (1.4).

Honest reading of the strength. This is NOT a broad improvement: it wins on
4 of 8 classes (sign test p=0.64, not significant). It is one very large,
unambiguous gain (wallplugs at 5.3 sd) plus a solid second (fabric), against
four small losses that are all inside 1.5 sd. The macro number is carried by
those two classes. What makes it credible rather than lucky is that the two
winners are exactly the sub-patch-defect classes the mechanism predicts, the
effect is monotone in the fill level q (0.75 -> 1.0) on the independent 4-class
probe, and the parameters were frozen before this run.

Mechanism: AUPRO@0.05 scores each connected defect component once, normalised by
its own area, so a component that is detected but only partially covered is
heavily penalised. Filling each above-threshold component to its own maximum
completes such components. It cannot create detections that are not already
above the 99th percentile, and it leaves every below-threshold pixel untouched,
which is why pAUROC (driven by the normal-pixel ordering) is unchanged.

Status: CQ is currently EVALUATION-SIDE only - it is a post-processing step in
`scripts/map_postprocess_eval.py`, not in the detector. To become a method it
must be implemented inside the prediction path and re-run online; the number
above is exact but is not a collector row and is not in the CSV ledger.

## LOCKED BEFORE FULL8 — Component-max fill (CQ) selected on the 4-class probe, 2026-09-15 22:05 KST

Selection was made on the four diagnostic classes only (fabric, can, wallplugs,
vial) using CB5's saved maps; the transform and its parameters are frozen here
BEFORE the full8 run, and will not be re-tuned on the full8 result.

Chosen transform: `component_quantile(percentile=99.0, q=1.0)` - threshold the
map at its own 99th percentile (a per-image, label-free statistic), take the
8-connected components of that mask, and replace every pixel inside a component
by that component's maximum score. Pixels below the threshold keep their
original value, so the normal-pixel ordering that sets the FPR budget is
untouched. One parameter set for all eight classes; no per-class choice.

4-class probe (CB5 maps, identity baseline 47.597/69.559):

| q (component fill level) | AUPRO | pAUROC |
|---|---:|---:|
| 0.75 | 47.43 (-0.17) | 69.77 |
| 0.80 | 48.21 (+0.61) | 69.75 |
| 0.90 | 48.86 (+1.26) | 69.81 |
| 0.95 | 49.21 (+1.62) | 69.86 |
| 0.97 | 49.37 (+1.77) | 69.88 |
| 0.99 | 49.55 (+1.95) | 69.91 |
| **1.00** | **49.66 (+2.06)** | **69.92 (+0.36)** |

Monotone in q all the way to 1.0, i.e. filling a detected component to its own
maximum is strictly better than partial filling - the direct prediction of
AUPRO being per-component normalised. Threshold is flat over p in [98, 99.5]
(+1.2 to +1.6) and collapses at p=95 (-3.05), so the rule is "level up only what
is already clearly above the normal background". Per class at q=1.0:
fabric +3.27, wallplugs +5.80 (pAUROC +2.33), can -0.23, vial -0.59 - the gain
is concentrated exactly in the sub-patch-defect classes that every score-side
axis failed to move.

Full8 verification plan: dump CB5 maps for the remaining four classes
(fruit_jelly, rice, sheet_metal, walnuts) on the local GPU, then evaluate all
eight offline with the frozen transform. Pass rule (unchanged, declared earlier):
macro AUPRO > 57.075 (CB5) with pAUROC not below 76.434 (MS1).

## Planned/active — Map-level axis, 2026-09-15 evening (user direction: performance first, seeds deferred)

User decision: multi-seed claim-validity work is deferred; the bottleneck is that
no configuration is clearly better than MS1/CB5, so the next work must look for a
real gain. Every axis explored so far changes HOW THE SCORE IS BUILT (bank,
resolution, layer). The map itself has never been touched, and AUPRO@0.05 - the
declared primary metric - is component-normalised and FPR-capped, so it is
sensitive to map post-processing. The only clean positive lever in the older
project memory is also map-level (frequency-separated DVT, AU-PRO +0.0105), which
is larger than anything the whole CB suite produced.

Track 1 (GPU, dsba3 GPU0/1) — **scale fusion SF0-SF2** on the CB5 base
(four scales, four bagged random-256 banks, k=1, layers [11,23]).
Per-resolution maps are currently combined by arithmetic mean; SF0 = mean
(control, must reproduce CB5), SF1 = elementwise max, SF2 = rank-normalised
per-resolution token scores then mean. Motivation: rice prefers fine scales and
fabric coarse ones (measured in HR/MS), and "visible at any scale" is a fixed
rule, not a per-class selection, so it does not fall under the closed
support-only-selector family. 24 runs.

Track 2 (CPU, parallel) — **offline map post-processing harness**. The evaluator
(`StreamingSegmentationAccumulator`) is a standalone class, so saved maps can be
re-scored without a GPU. MD1 = CB5 with `save_anomaly_maps: true` on four
diagnostic classes (fabric, can, wallplugs, vial; ~4 GB compressed) run once;
after that, candidate transforms (local contrast, high-pass, morphology,
region-quantile pooling) are searched on CPU in seconds instead of ~30 GPU-min
per variant. Only survivors go to a full8 GPU check.

Pass rule declared before results, both tracks: macro AUPRO@0.05 > 57.08 (CB5)
with pAUROC not below 76.43 (MS1); per-class effects are read against the
BC-derived per-class sampling sd. No per-class or test-label selection anywhere;
the harness is a search tool over FIXED transforms, and any transform adopted
must use one parameter setting for all eight classes.

## Latest execution — Bank-composition suite CB1–CB6 complete and collected, 2026-09-15 15:35 KST

48/48 (12 local fruit_jelly/vial + 36 dsba3), no failures. All arms on the MS1
four-scale base [448,672,896,1120]. The original remote lane of 2026-09-15 00:36
never started (`--devices must match REMOTE_GPU_DEVICES count`, 77 silent
retries); relaunched remote-only 11:07 after stopping the external MCP1 runs.

| Stage | Bank | k | AUPRO | pAUROC |
|---|---|---|---:|---:|
| MS1 (control) | random 256 | 1 | 56.750 | 76.434 |
| MS2 (control) | full | 1 | 55.487 | 77.239 |
| CB1 | full | 5 | 56.416 | 77.055 |
| CB2 | full | 10 | 56.808 | 76.961 |
| CB3 | full, trim 10 % | 1 | 55.701 | 76.573 |
| CB4 | full, trim 10 % | 3 | 56.237 | 76.354 |
| **CB5** | **4 x random 256, bagged** | 1 | **57.075** | **77.096** |
| CB6 | 4 x random 256 + trim | 1 | 56.935 | 76.170 |

Verdict under the two rules, reported separately to avoid post-hoc selection.
(a) The CB pre-declared joint rule (AUPRO >= 56.75 AND pAUROC >= 77.24, written
before the primary-metric declaration): **no arm passes**; CB5 misses pAUROC by
0.14. (b) The declared decision policy (maximise AUPRO@0.05 without a material
pAUROC regression): **CB5 wins** - highest AUPRO in the ledger (57.075) with
pAUROC 77.096, which is +0.66 over MS1 and only -0.14 versus MS2, i.e. no
regression at all relative to the reference detector.

Strength of the CB5 vs MS1 result: the macro AUPRO margin (+0.325) is inside the
0.86 pp single-bank sampling floor, so the macro number alone is not evidence.
The per-class pattern is: CB5 wins 7 of 8 classes (one-sided sign test p=0.035),
and the two largest moves are can +2.29 (2.2x its sampling sd) and fabric -4.18
(2.0x). CB5 is also a four-bank average, so it is less subsample-dependent than
any single-bank arm by construction. Treat CB5 > MS1 as supported but modest;
it needs support-seed replication before any paper claim.

Mechanism findings.
1. Bagging sparse banks is the winner and it works by variance reduction, not by
   coverage: CB5 beats MS1 on seven of eight classes (can +2.29, walnuts +1.44,
   wallplugs +0.94, vial +0.89, sheet_metal +0.61, rice +0.50, fruit_jelly +0.10)
   and loses only fabric (-4.18). It also gives can its best value in the whole
   project (17.20/61.29 vs 14.91/58.14), the first movement on that class from
   any bank-side change.
   Note this partially amends the `can` root-cause section above: that section
   states `can` never moved (13.3-15.3) through any bank or resolution change.
   Bagging is a bank-side change and it did move `can`, to 17.20/61.29 (CB5) and
   16.59/60.84 (CB6). That does not weaken the layer finding (LA2 [5,23] reaches
   23.45 with frozen features) but it does mean the bank axis was not completely
   flat for `can`; the two effects are plausibly additive and untested together.
2. The kNN-mean ladder does not saturate at k=5 on the classes that suffer
   absorption: fabric 45.98 (k=1) -> 50.29 (k=3) -> 52.01 (k=5) -> 54.27 (k=10),
   at a cost of only -0.38 pAUROC. An earlier reading of "saturates at k=5" came
   from two classes only and was wrong.
3. Trimming rare normal tokens interacts with density and is class-dependent:
   in the full-bank regime it is a net loss (CB3 below MS2 on both metrics), in
   the bagged-sparse regime it produces the best fabric (62.32, above MS1 59.42
   and near the 2-scale HR2 62.79) and the best wallplugs in the ledger
   (41.46/75.95) while costing fruit_jelly, vial, walnuts and sheet_metal.
4. The class split is now explicit: sub-patch-defect classes (fabric, wallplugs)
   want a sparse, outlier-trimmed bank; large-defect classes (fruit_jelly, vial,
   walnuts, sheet_metal) want a dense bank with a larger k. No single fixed
   configuration satisfies both, which is the empirical case for a support-only
   adaptive bank configuration (never test-label selection).

Cross-reference added 2026-09-15 after independent verification of this section's
numbers (all eight CB5 per-class deltas and both macro deltas reproduce exactly;
no correction needed). Two cautions for finding 4, from the layer work recorded
above in this file:

- Under the declared primary metric (AUPRO@0.05), CB5's +0.32pp over MS1 sits
  inside the 0.86pp sampling floor, so it is not yet an improvement. Adoption
  needs >= 3 bank seeds for the MS1 side of the comparison.
- The proposed support-only adaptive bank configuration faces the obstacle that
  just closed the layer axis. There, the same shape of result held (per-class
  optimum, effects far outside the floor, no single fixed setting for all eight),
  and **five** natural support-only criteria — normal-tail ratio, q99/q50,
  excess kurtosis, synthetic CutPaste separation and dynamic range — all scored
  5/8 at chance, each failing on the same positive classes. Before building an
  adaptive bank selector, run the equivalent cheap 8-point feasibility check on
  the bank statistic first; do not assume a support-only signal exists because
  the per-class effect is large.

[Report](../results/local_runs/bcma_docker_20260911/bank_composition_20260915/comparison.md).

## Closed — RD-P1 reverse distillation pilot: FAIL, 2026-09-15

RD1 (WRN50 teacher, strides 4/8/16): fabric 37.25/51.95, can 13.50/59.52,
rice 18.62/65.56, walnuts 59.55/74.47. RD2 (DINOv2-L teacher @1120): can
10.91/54.42, fabric 30.14/66.13, rice 29.67/82.32, walnuts 65.79/90.61. All 8 runs
complete and collected 01:45 KST (ledger rows reference_matching-independent method
`reverse_distillation_RD1/RD2`); no class beats MS1 on either metric. Reading: the
student does not generalise to the over/under-exposed and shifted normal test
variants from 16 supports — normal false positives dominate, as the guardrail
predicted. Dense supervision from 16 images is not sufficient without an external
"non-normal" penalty (EfficientAD-style) or much stronger augmentation. Family A
closed in this configuration; no retune. Non-bank next candidate remains the
region-coherence prior (needs SAM weights copied into the offline container).

## Closed — Coverage–absorption suite CA1–CA4 complete and collected, 2026-09-15 11:09 KST

32/32, no failures. Merged view (local fruit_jelly/vial + dsba3 lane_0..3).
All arms on the MS1 four-scale base [448,672,896,1120].

| Stage | Bank | k | AUPRO | pAUROC |
|---|---|---|---:|---:|
| MS1 (control) | random 256 | 1 | **56.750** | 76.434 |
| MS2 (control) | full | 1 | 55.487 | 77.239 |
| CA1 | coreset 1024 | 1 | 53.625 | 77.382 |
| CA2 | full | 3 | 56.140 | 77.129 |
| CA3 | coreset 1024 | 3 | 54.558 | **77.566** |
| CA4 | coreset 4096 | 1 | 54.940 | 77.324 |

Pass rule (AUPRO >= 56.75 and pAUROC >= 77.24): **no arm passes**. CA2 is the
closest (AUPRO -0.61, pAUROC -0.11) and is the best AUPRO/pAUROC compromise so
far; CA3 has the highest pAUROC in the whole ledger.

Mechanism findings. (1) Coreset is a NEGATIVE result: greedy k-center keeps the
rare/extreme normal tokens, which are exactly what sub-patch defects match, so
fabric AUPRO drops further than with a full bank (MS2 45.98 -> CA1 41.50) while
pAUROC rises. Coverage is not what the sparse random bank was buying. (2)
kNN-mean (k=3) is a partial positive: on the full bank it recovers fabric
45.98 -> 50.29 and is neutral to slightly positive elsewhere (walnuts 73.95,
sheet_metal 68.54); it is the only lever so far that reduces absorption. (3) The
fabric gap to MS1 (59.42) is still unexplained by density or coverage; MS1 wins
fabric and wallplugs only, and loses the other six classes. (4) can is unmoved
by every bank composition (13.3-15.3).

Follow-up: CB1-CB6 (k=5/k=10 ladder, support-only outlier trimming, bagged
sparse banks) tests whether rare-normal-token removal plus stronger kNN
averaging reaches the joint target.
[Report](../results/local_runs/bcma_docker_20260911/coverage_absorption_20260915/comparison.md).

## Latest execution — Multi-scale suite MS0–MS2 complete and collected, 2026-09-15 00:06 KST

24/24 complete, no failures; merged view (local lane_0 fruit_jelly/vial + dsba3
lane_0..3), compact files only, full logs on dsba3 /results/multi_scale_20260914.
R1 code unchanged; only `resolutions` and `bank_tokens_per_source` differ.

| Stage | Scales | Bank | AUPRO | pAUROC | vs HR0 (52.470/75.992) |
|---|---|---|---:|---:|---|
| MS0 | [896,1120] | full | 53.91454 | 75.82232 | +1.44 / −0.17 |
| MS1 | [448,672,896,1120] | cap 256 | **56.75021** | 76.43401 | **+4.28 / +0.44** |
| MS2 | [448,672,896,1120] | full | 55.48657 | **77.23926** | **+3.02 / +1.25** |

Pre-declared pass rule (AUPRO ≥ 54.97 and pAUROC ≥ 75.99): MS1 and MS2 both
pass; MS0 fails on pAUROC. Both clear the 0.86pp sampling floor by 3–5×. MS1 is
the highest full8 AUPRO in the ledger (SuperAD-16 53.49, Basic-P16 53.34, MC1
53.61); MS2 pAUROC 77.24 is above SuperAD-16 (76.19) and 0.02 below Basic-P16.
Per-class (AUPRO/pAUROC) MS1 vs MS2: can 14.91/58.14 vs 13.96/58.51;
fabric 59.42/72.01 vs 45.98/72.90; fruit_jelly 57.21/76.84 vs 58.10/78.12;
rice 66.82/93.24 vs 67.65/93.58; sheet_metal 67.19/77.54 vs 68.58/81.07;
vial 77.37/68.95 vs 78.85/68.97; wallplugs 38.75/74.78 vs 37.48/73.98;
walnuts 72.33/89.98 vs 73.30/90.79. MS2 wins pAUROC on 7/8 classes and AUPRO
on 5/8; the MS1 macro AUPRO lead is fabric alone (+13.4pp).

Findings. (1) Four-scale averaging removes the high-resolution losses of HR2
(rice, sheet_metal, vial back to or above HR0) while keeping most sub-patch
gains. (2) Full bank raises pAUROC everywhere (coverage) but collapses the
fabric AUPRO gain (62.8 → 46.0): dense normal banks absorb sub-patch defects
into near-duplicate normal neighbours; the 256-token subsample acted as an
implicit regulariser. This coverage-vs-absorption trade-off is now the central
structural question. (3) can stays 13.8–14.9 at every scale/bank: needs a
feature-level lever. Single seed, seen development split; no final claim.

Decision: adopt **MS1 (R1, scales [448,672,896,1120], cap 256) as the new
reference detector** for further work, with MS2 as the pAUROC-priority variant;
docs/method.md still describes R1 and must be updated. R1 single-scale add-ons
remain closed. [Report](../results/local_runs/bcma_docker_20260911/multi_scale_20260914/comparison.md).

## Next — structural direction (updated 2026-09-15 evening, after the layer-axis closure)

Reference detector: **CB5** (four-scale R1 [448,672,896,1120], four bagged
random-256 banks, k=1, layers [11,23]) on the declared primary metric; MS1 stays
the single-bank control and MS2/CA3 the pAUROC-priority variants.

Deleted from the plan: the earlier item "support-only adaptive bank
configuration". The layer-axis closure above establishes on principle - not just
empirically - that a statistic computed from normal supports measures only the
denominator of defect-response / normal-response, and five such criteria all
failed at chance and were inverted on exactly the positive cases. Per-class
adaptive *bank density* would be the same construction on the same 8-point answer
key and must not be attempted without a new mechanism argument declared first.

Ordered next steps:
1. Bag ladder (no code, cheap - sparse banks): 8 and 16 bags at cap 256, and 4
   bags at cap 1024. Question: does the variance-reduction gain continue, and
   where does it saturate? Pass = AUPRO > 57.08 with pAUROC not below 76.43.
   This is the only lever currently on a rising curve.
2. Bagging x k, never yet combined: CB5 with k=3 and k=5. Both are independently
   positive (bagging 7/8 classes; k monotone on the absorption classes) and they
   act on different failure modes, so the combination is the cheapest remaining
   chance of a joint AUPRO+pAUROC gain.
3. Per-bag trimming as a bag-level (not global) hyperparameter: trim only some of
   the bags, so fabric-style sparse-trimmed evidence and vial-style dense
   evidence coexist inside one averaged score, with no per-class selection.
4. Support-seed replication of CB5 vs MS1 (seeds 1, 2) before any paper claim.
   CB5 is already bank-seed averaged; support selection is the remaining
   uncontrolled source of variance.
5. `can`: bank and resolution are closed by the diagnostics above; layer choice is
   closed as a *selectable* method. What remains legitimate is (a) reporting the
   class-dependent-depth finding as analysis, and (b) testing whether bagging and
   a shallow stack are additive on `can` alone as a mechanism probe, not as a
   full8 candidate.
6. Region-coherence scoring (SAM/segmentation prior on CB5 maps) is now the
   largest untried structural axis and is the one aligned with the stated CVPR
   2027 direction; it needs SAM weights copied into the offline container.

## Closed — Resolution diagnostic HR0–HR2 complete, 2026-09-14

24/24 complete, no failures, collected 21:3x KST as a merged view (local lane_0
HR0 + dsba3 remote lane_0..3 HR1/HR2; compact files only, full logs remain on
dsba3 at /results/high_resolution_20260914/remote). HR0 = R1 [448,672], cap 256;
HR1 = [672,896]; HR2 = [896,1120]; everything else identical to R1. Collector
withholds Δ because `resolutions` differ (by design); HR0 reproduces R1 to 1.3e-6.

| Stage | AUPRO | pAUROC | vs HR0 |
|---|---:|---:|---|
| HR0 [448,672] | 52.47034 | 75.99215 | — |
| HR1 [672,896] | 53.93733 | 75.42452 | +1.47 / −0.57 |
| HR2 [896,1120] | 54.96973 | 74.93654 | +2.50 / −1.06 |

Per-class HR2−HR0 (AUPRO/pAUROC; sampling sd from BC0–2 in brackets):
fabric +24.97/+0.50 [2.1], walnuts +4.50/+0.07 [1.1], wallplugs +3.94/−6.14 [1.1],
fruit_jelly +1.50/+1.00 [0.6], can +0.88/+2.35 [1.0] (HR1 can 18.46 = +4.97),
vial −0.45/−0.79 [0.5], rice −7.26/−2.79 [0.7], sheet_metal −8.08/−2.64 [0.6].
HR2 AUPRO 54.97 is the highest full8 seen-development value in the ledger
(SuperAD-16 53.49, MC1 53.61) and clears the 0.86pp sampling floor; its pAUROC
loss is concentrated in wallplugs, rice and sheet_metal. Per-class best-of-scale
upper bound (diagnostic only, not a method): 57.75 / 75.66.

Interpretation (hypothesis, single seed): input scale is a first-order lever
where reference selection was not; the effect sign is class-dependent. Sub-patch
defect classes (fabric, walnuts, wallplugs; GT area medians 0.01–0.04%) gain;
fine-texture classes (rice, sheet_metal) lose. Confound to resolve first: cap 256
was held fixed while tokens per image grew 2.5–6×, so bank coverage per unit
area fell to 4–10%, which BC3/BC4 showed matters for pAUROC. can is not rescued
by scale alone (13.5 → 14.4/18.5) and needs a feature-level lever.
[Report](../results/local_runs/bcma_docker_20260911/high_resolution_20260914/comparison.md).

## Closed — Correspondence Guard CG0–CG3 complete, 2026-09-14

All 32 runs complete, no failures, collected 19:42 KST. CG0 53.13413/76.12941 =
QR1 (8 classes within 1e-8); CG1 53.61380/76.02219 = MC1; CG2 53.60877/76.02265;
CG3 53.60460/76.02195. Joint target failed (CG3 pAUROC < CG0). CG2/CG3 are a
no-op relative to CG1 (|Δ| < 0.01pp). Guard diagnostics: MC1 changes the nearest
reference for 3.0–4.5% of tokens and the mode for 0.4–1.0%; its per-token score
reduction is 0.0014–0.0035 (~2% of mean score); the guard removed ~1e-5 per token
(0.3–1% of that benefit). Cause: the cost equals the QR1 gap, which is ~0 exactly
where bounded MC1 deformations switch references. Even a perfect guard could
recover at most MC1's 0.107pp pAUROC loss. Conclusion: the QR→MC→VR→CG
correspondence-adjustment line is closed; MC1's pAUROC loss is anomaly
absorption by full-query bank deformation (fruit_jelly −0.93, sheet_metal −0.49,
vial −0.30), not correspondence switching. All R1-derived suites (QR/RC/MC/VR/
CR/CG/TF, 28 stages) lie in AUPRO 52.3–53.6 / pAUROC 75.5–76.1; none exceed
QR1 pAUROC. can remains 13.4/57.3 for every frozen-feature method (D3 24.3/67.6).
[Collected report](../results/local_runs/bcma_docker_20260911/correspondence_guard_20260914/comparison.md).

## Latest execution — CRR CR0–CR3 full8 complete and collected, 2026-09-14

All32 class runs completed with no recorded failure. Local GPU0:8/8 at12:57:32
KST. dsba3 GPU0/1/2/3:8/8/4/4 at13:34:32 /13:46:20 /13:42:27 /13:05:48.
Collection did not launch, restart, stop or schedule experiments; existing
containers, remote logs and artifacts are retained. No source/config changes.

| Stage | AUPRO x100 | pAUROC x100 | F1 oracle x100 |
|---|---:|---:|---:|
| CR0 / exact R1 | 52.47033 | 75.99215 | 36.65951 |
| CR1 / residual + local tail | 52.98213 | 75.63799 | 36.65437 |
| CR2 / + cross-image tail | 53.13319 | 75.52551 | 36.39412 |
| CR3 / + reference geometry | 52.29146 | 75.91410 | 36.58088 |

Equal-class full8 mean, seed0 normal16,448+672,seen TESTpublic development.
CR2 is this suite's AUPRO best: +0.66286pp vs CR0, but pAUROC -0.46664pp.
CR3-CR2: AUPRO -0.84173pp/pAUROC +0.38859pp; CR3 is below CR0 on both.
No learned candidate meets the locked joint goal of AUPRO >= CR0 and pAUROC
> CR0. Retain R1 as the joint-metric reference; CR2 is only an AUPRO-priority
trade-off candidate. CR2's wallplugs +5.70226pp dominates its AUPRO gain;
posthoc other-seven mean is -0.05706pp. This does not replace the full8 score.
Single seen-development seed: no significance, final generalization or unique
causal failure claim. Geometry and global-tail families are not ruled out.

All32 metrics agree with summary.tsv/manifest, with complete finite,
nonconstant and shape-correct coverage. All8 quartets share content hashes,
common conditions, code, backbone/banks/MAD; trained triplets share64 episodes,
compact descriptor hashes and positive-sampling sequences. Logs confirm
0/64 sequential optimizer updates, finite values, fit_started/fit_finished.
The additional64 step=null descriptor-cache events are not optimizer steps.
Cross-source pairs: CR1=0; CR2/3 about93.8%, confirming activation, not benefit.
CR0 exactly reproduces all8 historical R1 and TF0 class metrics. Code SHA and
Python/packages match local/remote; differing hardware/kernel/glibc preclude
controlled speed claims. Frozen backbone; parameters0/129; fit calls32/544.
33MiB compact training pool released after fit; dense maps and weights off.

Basic remote pull hit host file permissions; recovered only92 allowlisted
compact files through the existing container, verified all SHA256s, without
changing permissions or copying full logs. Existing generic collector used
on5 lanes; consolidated32 class+4 full8 macro rows in canonical Markdown/CSV.
All548 historical rows unchanged;584 total. Lane macros are not duplicated.
[Combined result and interpretation](../results/local_runs/bcma_docker_20260911/cross_image_reference_20260914/comparison.md)
links collection_audit.json, training_audit.json, per-run metrics and transfer
hashes. docs/method.md now distinguishes completed CRR evidence from hypotheses.

## CRR execution setup and launch record — 2026-09-14

User authorized implementation -> review -> local/remote GPU0/1/2/3 setup ->
Bash launch -> experiments, then explicitly corrected dsba2 to dsba3.
Implemented cross_image_reference.py and CR0--CR3
YAMLs through the existing CLI. [Locked protocol](../configs/methods/CROSS_IMAGE_REFERENCE.md)
and [design](analysis/2026-09-14_cross_image_reference_design.md) describe the
bounded R1 residual, cross-image tail ranking and reference geometry. CR2-CR1
and CR3-CR2 isolate additions; CR1-TF3 does not. No backbone/common-pipeline edit.
Full8 planned, seed0 normal16,448+672,64 updates, seen TESTpublic development;
joint target AUPRO >= fresh CR0 and pAUROC > fresh CR0. Maps/weights off.

Local validation:152 tests passed,5 skipped across method/common-pipeline,
collector/launcher and remote-safety tests. Separate real cached DINOv2 CUDA
full16-bank/full-resolution smoke passed all four stages (two trained updates,
no benchmark metrics). CR0 exactly matches R1; trained stages share descriptor
and episode hashes, with finite outputs and frozen backbone. Bash syntax passes.
Ruff is absent, so no Ruff check. Existing local container fsad-tta-local-bcma
is compatible (Torch2.6.0+cu124/CUDA12.4); no container/package changes needed.
Validated/running package SHA:
bb8f51c90acb1da0db72647da87fb03dc6be4ea33cbab253802e820e20854275.

Local launch12:28:55 KST, parent PID1470978/session1470978 survives the shell.
Generic scripts/run_experiments.sh -> experiments.py run -> existing CLI;
can/fabric run CR0 -> CR1 -> CR2 -> CR3,8 total class runs,one worker on GPU0
UUID GPU-0070b668-714f-bcbb-6c46-f0b68c9b7f6b. Initial status running CR0/can,
0/8 completed, training/evaluation process1471079 on that UUID; no failure
recorded. GPU utilization71%/2089MiB at follow-up, not an efficiency benchmark.
No other methods or extra seeds queued. Per-class timeout14400s.
Follow-up: CR0/can and CR0/fabric completed,2/8 local class runs; status now
running CR1/can with an empty failures list. Remaining local stages stay queued.
Latest12:45:09 KST: local5/8 complete, CR2/fabric active, failures empty.
Local PID1470978 was not restarted or interrupted by the remote correction.

The initial dsba2 connection blocker is RESOLVED by the user's dsba3 correction.
Existing hun_fsad_tta_012 is running; all four RTX A6000 GPUs were idle and
passed CUDA arithmetic. All8 normal-data paths are available at /data/mvtec_ad_2,
with existing /model-cache. No container/package/data changes or git push.
The configured physical GPU0/1/2/3 UUIDs are respectively:
GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f,
GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae,
GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639,
GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62.
run_remote.sh update-code activated immutable local snapshot
5b6ea70b4fb2d9d1c45b2f754beb4ed5a4b1ed0405b9a12b5eed584cb3a02e12.
Package SHA and all four YAML byte hashes exactly match the active local run;
Python and recorded package versions match. Host kernel/glibc differ
(local6.8.0-138/glibc2.36; remote6.8.0-139/glibc2.35), so do not claim identical
hardware/OS or controlled wall-clock comparison. Remote152 tests passed with
5 skipped; separate real DINOv2 full-bank smoke passed all four stages
(two trained updates, finite/nonconstant predictions, exact CR0/R1 parity).
Validation log: hun_fsad_tta_012:/tmp/fmad-execution-logs/run.nMoI5DCO.
Launched remaining24 class runs at12:44:19 KST with generic run_experiments.sh
--backend remote, one worker per UUID and CR0->CR1->CR2->CR3 within each lane.
Detached parent/session1496638 survives the initiating shell. Lane assignments:
remote0 fruit_jelly/wallplugs, remote1 rice/walnuts, remote2 sheet_metal,
remote3 vial. At12:45 all four lane statuses were running CR0 with no failures;
all four had fit_started with16 supports. Host process106378/106376/106379/106377
respectively maps to physical GPU0/1/2/3 (1804/1856/2244/1804MiB at inspection).
This verifies actual allocation, not just queued commands. Remote0/24 complete
at this initial check; local and remote total32 planned class runs. No duplicate
local launch, changed source/config, container replacement or data download.
Remote originals: hun_fsad_tta_012:/results/cross_image_reference_20260914/remote/lane_0..3.
Local launch output: results/experiment_launch/cross_image_reference_20260914/remote.launch.log.
Remote helper log: hun_fsad_tta_012:/tmp/fmad-execution-logs/run.qqnLG6do;
per-lane launch logs are beside the remote lane directories. Keep full logs
remote; recover compact metrics/config/manifests only during later collection.

Local originals: results/local_runs/bcma_docker_20260911/cross_image_reference_20260914/local/lane_0.
Launch logs: results/experiment_launch/cross_image_reference_20260914/local.launch.log
and local/lane_0.launch.log. Status.json in the original root records the exact
config order/commands; per-class stdout and training.jsonl remain there.
Descriptor cache emits step=null/phase=descriptor_cache before numeric optimizer
steps. Only compact pool counts/hashes persist after fit; no dense accumulation.
No suite metrics have been collected or compared yet. Initial CR0/can fitting
finished12:29:07 KST in5.804s; subsequent stdout reached its summary output.
Collect verified completions with the existing helper and preserve partial
versus full8 scope; no cumulative comparison update during this launch turn.

## Previous execution — TBF TF0--TF3 full8 complete and collected, 2026-09-14

All32 class runs completed without recorded failures: local8 finished10:29:39
KST; remote GPU0/1/2/3 completed8/8/4/4 at10:43:57/10:52:55/10:58:09/
10:24:42 KST. All four remote GPUs were idle at11:02:49 inspection.
No experiment is active or scheduled by collection. Containers and originals
are retained; no new training/evaluation, source/config edit, restart or cleanup.

| Stage | AUPRO x100 | pAUROC x100 | Oracle F1 x100 | AUPRO delta vs TF0 (pp) |
|---|---:|---:|---:|---:|
| TF0 / exact R1 | 52.47033 | 75.99215 | 36.65951 | +0.00000 |
| TF1 / global scalar | 52.70446 | 75.79109 | 36.68387 | +0.23413 |
| TF2 / + normal tail | 52.61031 | 75.79416 | 36.62025 | +0.13998 |
| TF3 / + local gate | 52.76262 | 75.51276 | 35.96161 | +0.29229 |

Full8 macro weights classes equally, not the five unequal lanes. TF3 is best
AUPRO within this suite, but pAUROC -0.47940pp and oracle F1 -0.69791pp vs
TF0. Targeted pAUROC recovery without primary loss was NOT achieved.
TF2-TF1 AUPRO -0.09415pp/pAUROC +0.00307pp: no useful aggregate tail benefit.
TF3-TF2 AUPRO +0.15231pp/pAUROC -0.28141pp: local conditioning adds a tradeoff.
TF3-TF0 gains concentrate in wallplugs +6.76877pp; sheet_metal -4.31818pp.
Posthoc exclusion of wallplugs gives -0.63292pp AUPRO across the other seven
classes. Retain TF0/R1 as the pAUROC-preserving reference, not an unconditional
promotion of TF3. No unique failure mechanism or general family rejection is
established. Single seen-development seed; no significance/final-holdout claim.

All32 metrics agree with summary.tsv/manifests and have full finite/nonconstant,
shape-correct prediction coverage. All logs have fit_started/fit_finished,
finite values and expected0/64 sequential steps. All8 within-class quartets
match support/query/GT protocol, common config, code, software environment,
backbone weights and frozen MAD scales. All8 TF1/TF2/TF3 triplets share exactly
64 actual episode records, including four views/masks/coordinates/shift and
source exclusion. No query fitting or GT, no backbone updates.

TF0 exactly reproduces previous R1 metrics/coverage and MAD on all8 classes.
Six old remote support paths use /home/hunim/Volume/DATA/mvtec_ad_2/ instead
of /data/mvtec_ad_2/: content/identity/GT hashes and ordered relative paths
match. This is an audited path alias, not literal old/new path-string equality.
Fresh TF0 is the strict same-code baseline; old R1 has a different package SHA.

Generic collect processed all5 roots; canonical Markdown/CSV consolidate52
lane-scoped rows into36 full8 rows, preserving512 historical rows (548 total).
All92 recovered remote compact files are SHA256-identical to originals.
Host permission errors were bypassed through read-only existing-container
transfer without changing permissions. Full remote logs remain remote;
bounded audit summaries only. No dense maps/checkpoints were recovered.
Audit: results/local_runs/bcma_docker_20260911/tail_aware_fusion_20260914/
collection_audit_20260914.json and collection_remote_audit_20260914.json.
Compact remote: results/remote_runs/tail_aware_fusion_20260914_dsba3/lane_0..3.

Observed summed fit times TF0/TF1/TF2/TF3:88.49/1668.53/1668.51/1660.27s;
mean per-query times .661/.918/.895/1.192s. Fit encoder calls32/544/544/544
per class; query calls2 for all stages. Mixed4090/A6000 and concurrent lanes,
so these are recorded costs, not an equal-hardware latency benchmark.

### Preserved implementation and launch provenance

User authorized local GPU0 plus dsba3 GPUs0/1/2/3 to implement and execute
the bounded fusion design. Existing containers fsad-tta-local-bcma and
hun_fsad_tta_012 are running; all five GPUs were idle at preflight.
New method tail_aware_fusion.py and four YAML configs are implemented through
the existing CLI. Locked protocol: configs/methods/TAIL_AWARE_FUSION.md.
TF0 exact R1; TF1 global fusion; TF2 sampled normal-tail ranking; TF3 local
65-parameter gate. TF1/2 one parameter, TF0 zero; backbone/references/MAD frozen.
Full8 seed0 normal16,448+672,64 updates for TF1--3, same metric/preprocessing.
TF1--3 use identical episode hashes, positive coordinates and reference exclusions.
Normal tail is top5% of at most4096 sampled native pixels per normal view, not
exact dataset-wide pAUROC. No query fitting, dense-map accumulation or weights.
Primary AUPRO must not decrease while targeted pAUROC improves for a joint win.
Completed metrics above supersede the initial launch status below.

Local validation:135 passed,4 skipped in focused regression suite; separately
one actual cached DINOv2 CUDA smoke passes all four stages at448/672 (TF1--3
two updates on three normal can supports, no benchmark metrics). TF0 exact R1,
finite outputs, same episode hashes/scales, frozen backbone,1/65 gate params.
Remote immutable snapshot activated:
1f36c150db82307f11045dc72856e419d756ce13d08c7ec0db7621836a710ef8.
No git push, container replacement, data copy or cleanup. Remote validation
also passed135 tests (4 skipped) plus the actual cached-backbone smoke.
Both environments: Torch2.6.0+cu124 / CUDA12.4; all four remote devices passed
CUDA arithmetic. Matching package code SHA:
668d54ddd9cb33b975e67c1846bc5ea70fe811f389c1dee7714f10885f0b1654.

Launched10:08:41 KST2026-09-14 through one detached generic Bash command;
parent PID1249970/session1249970 survived the initiating shell. Five lanes
verified running TF0 with assigned-device processes and no recorded failures.
Placement (same as R/P suite): local0 can/fabric; dsba3 GPU0 fruit_jelly/wallplugs,
GPU1 rice/walnuts, GPU2 sheet_metal, GPU3 vial. One worker per device; each lane
runs TF0->TF1->TF2->TF3.32 total class runs,0 completed at initial status check.
Follow-up10:11 KST: local TF0 can/fabric finished and exactly match previous R1
saved metrics/coverage, complete support/query/GT protocol and MAD scales.
Remote TF0 fruit_jelly/vial also finished; GPU3 TF1/vial reached step13/64 with
finite loss/gradient and changing gate weight. No failure in any lane.
Source/configs remained unchanged while these runs were active; all are now done.

- Local originals: results/local_runs/bcma_docker_20260911/tail_aware_fusion_20260914/local/lane_0.
- Local launch/training stdout: results/experiment_launch/tail_aware_fusion_20260914/local/lane_0.launch.log;
  class-specific logs and training.jsonl are under the local original root.
- Remote originals: hun_fsad_tta_012:/results/tail_aware_fusion_20260914/remote/lane_0..3
  (host /home/hunim/Volume/FMAD-OOD-remote/results_local_compatible/tail_aware_fusion_20260914/remote).
- Parent/remote launch logs: results/experiment_launch/tail_aware_fusion_20260914/launcher.log
  and remote.launch.log. Remote helper log: /tmp/fmad-execution-logs/run.qBZ2mAJe.
- Collection and compact-artifact verification are complete; canonical
  results/comparisons.md and its paired CSV are updated. No collector remains active.

## Previous execution — independent R/P full8 complete and collected, 2026-09-14

All64 final class runs are complete and verified: local16 finished03:12:08 KST;
dsba3 GPU0/1/2/3 completed16/16/8/8 at03:39:14/03:50:41/05:17:32/02:59:26.
No experiment remains active or scheduled by collection. Existing containers
are retained; no stop/start, code/config edit, new evaluation or cleanup.
The5 failed original R1 attempts are preserved and excluded. Repair/resume
skipped24 completed D3/L3C/R0 runs and produced40 successful candidate runs.

Seed0, normal16, frozen DINOv2 ViT-L14,448+672, shared no_mask_no_rotation,
seen TESTpublic development. Full8 macro weights each class equally, not
each GPU lane. Primary AUPRO@.05; pAUROC secondary, oracle F1 diagnostic.
No significance or untouched-final claim.

| Stage | AUPRO x100 | pAUROC x100 | Oracle F1 x100 | AUPRO delta vs L3C (pp) |
|---|---:|---:|---:|---:|
| D3 | 48.60330 | 77.65589 | 35.94865 | +0.64127 |
| L3C | 47.96204 | 77.15604 | 36.40674 | +0.00000 |
| R0 | 46.66157 | 75.42997 | 34.11109 | -1.30047 |
| R1 | 52.47033 | 75.99215 | 36.65951 | +4.50830 |
| R2 | 42.92744 | 73.04516 | 27.39787 | -5.03459 |
| P0 | 14.73310 | 54.76474 | 7.03519 | -33.22894 |
| P1 | 16.01451 | 56.38920 | 9.41695 | -31.94753 |
| P2 | 14.24413 | 55.08400 | 9.31401 | -33.71791 |

R1 is best-primary in THIS batch: +4.50830pp versus L3C, but pAUROC
-1.16388pp, so not an across-metric win. R1-R0 +5.80876pp with7/8 class gains
supports the combined scaled multi-layer readout; it does not isolate layers
from normalization. Largest gains are walnuts/wallplugs/vial; can declines.
R2-R1 -9.54289pp with7/8 declines, especially rice -25.72031pp and
sheet_metal -16.54659pp. The coupled context/reference mechanism did not help.
P1-P0 +1.28141pp is a partial recovery, chiefly fruit_jelly/wallplugs;
P2-P1 -1.77038pp. All P variants are far below anchors despite valid training.
P2/can AUPRO0 is actual measured zero with finite/nonconstant complete outputs.

Recommended base for subsequent method work: R1. Isolate feature-layer choice
and support normalization before assigning mechanism credit. No implementation
or experiment follows automatically from collection. Historical SuperAD-16
53.49241 uses a different original evaluator/preprocessing contract: no claim
that R1 beats it. Full detailed tables remain in canonical comparisons.md.

All64 metrics match original summaries/manifests; all queries have full finite/
nonconstant, shape-correct prediction coverage. All8 class octets match recorded
support/query/GT protocol, within-class software environment, backbone weights
and metric contract. D3/L3C64 steps, R0/R1/R2 zero updates, P128 steps;
all64 original logs have fit_finished and expected finite sequential updates.
All8 P triplets have identical128 actual episode hash sequences and1568 fit
backbone calls; P1 changes uncertainty flag only (+129 trainable parameters),
P2 synthetic supervision only. All8 R1/R2 pairs have exactly equal MAD scales.

Comparability is declared-method, not globally identical features/compute:
D3/L3C/R0 layer23; R1/R2/P layers11/23. R1-R0 includes normalization;
R2-R1 combines context and shared-reference selection. P uses18 encoder calls
per query versus R2 calls. Trainable parameters D3/L3C100384, R0/R1/R2 zero,
P0 529792, P1/P2 529921. Observed mean per-query time R1 .662s versus
P4.624–4.650s; mixed hardware/concurrency, not controlled latency benchmarks.
Maps/checkpoints off. Only compact results/provenance pulled; full remote logs
were summarized on dsba3, not copied.

Original D3/L3C/R0 code SHA:
f22b0607728b985215e12022c3b5f1b6d1c4b6f9445f73adbe3c1d071b63b212.
Repaired R1/R2/P code SHA:
27f1bbf62c96b44d471829ba91f9eeb95c77181a63e067a39054948d6253e0ce.
Only package diff: capped query CPU->bank device in R1/R2 calibration;
config bytes unchanged. Original hashes remain recorded. Generic collector's
strict cross-code/feature guard remains intact; the full8 report explicitly
audits those declared method contrasts instead of asserting equal-code runs.
Final remote snapshot:
c2c129a5bcaee17dcb7b1882f0c9555928acd7d9f154b80b85e0c5ec3cc7e481.

Generic collect processed all5 roots. Canonical Markdown and paired CSV
replace104 per-lane rows with72 full8 rows, preserving440 historical rows
(512 total). No duplicate cumulative table/helper. Remote permission fallback
recovered compact artifacts via read-only container transfer; all184 remote
compact files are byte-identical to originals. Five original failed R1 configs
and training logs remain traceable without being counted as successful runs.

Local original root:
results/local_runs/bcma_docker_20260911/reference_prediction_20260914/local/lane_0.
Compact remote root:results/remote_runs/reference_prediction_20260914_dsba3.
Remote originals:hun_fsad_tta_012:/results/reference_prediction_20260914/remote.
Audit and original-log summaries:
results/local_runs/bcma_docker_20260911/reference_prediction_20260914/collection_audit_20260914.json
and collection_log_audit_20260914.json. Design:configs/methods/REFERENCE_PREDICTION.md.
Fixed devices: localGPU0 can/fabric; dsba3GPU0 fruit_jelly/wallplugs,
GPU1 rice/walnuts, GPU2 sheet_metal, GPU3 vial.

## Previous execution — FBPA full8 complete and collected, 2026-09-13

All32 final runs are complete and verified: local24/24 finished23:33:47 KST,
dsba3 8/8 finished00:10:26 KST on September13. Local M2/sheet_metal was
intentionally stopped01:17:41 and restarted23:15:18 under the same conditions.
Its interrupted attempt is preserved and excluded. No new experiment active
or scheduled by collection. Full local/remote logs and results are retained.

Seed0/normal16/frozen DINOv2 ViT-L14 layer23/448+672/64 updates,
no_mask_no_rotation, seen TESTpublic development. Primary: full8 macro
AUPRO@.05; pAUROC secondary; oracle F1 diagnostic. No final-holdout claim.

| Stage | AUPRO x100 | pAUROC x100 | Oracle F1 x100 | AUPRO delta vs L3C (pp) |
|---|---:|---:|---:|---:|
| L3C | 47.96117 | 77.14863 | 36.41526 | — |
| M0 | 47.35899 | 76.69564 | 36.01830 | -0.60218 |
| M1 | 47.70336 | 76.99095 | 36.51869 | -0.25781 |
| M2 | 47.73587 | 76.93707 | 36.40540 | -0.22530 |

M1-M0 +0.34437pp supports family-balanced packing in this seen seed0 full8
comparison, not a uniquely proven gradient mechanism. Four class gains/four
declines; dominated by vial +3.06554pp and wallplugs +1.00584pp. Local6 alone
declines0.21941pp. M2-M1 +0.03251pp is weak/mixed: pAUROC -0.05388pp,
oracle F1 -0.11330pp. M2 still trails L3C by0.22530pp, with3/8 class gains.
Neither candidate is a new best-primary method. Historical L3C47.97341
remains distinct from this batch's47.96117; do not substitute old baselines.

All32 original training logs confirm steps0..63 and fit_finished with finite
loss/gradient values; final metrics have full finite/nonconstant shape coverage.
All8 class quartets match support/query/GT protocol, common effective config,
backbone, code, metric contract and within-host environment. All8 M0/M1/M2
triplets have identical128 actual episode manifests (inputs/masks/seeds/
teacher/reference population); M1 adds family_balance only, M2 source_pairing
only. M0 is not K1. L3C differs in reference/RNG construction, so L3C contrasts
are method-level comparisons, not isolated packing/bank attribution.

All runs train100384 parameters with backbone frozen. M0/M1/M2 per-class fit
extraction calls1856/1856/1600; main-view calls1024/1024/768. Summed final-run
fit time L3C/M0/M1/M2:3241.02/3213.33/3215.48/2816.34s. M2-M1 -12.41%,
excluding interrupted work; not queue wall time or an equal-hardware timing
benchmark. CUDA fit peaks essentially unchanged. Same device per class across
methods: six classes localGPU0/450W, vial/wallplugs dsba3 physicalGPU2.
Maps/checkpoints off. No dense map or remote full-log pullback.

Generic collect ran on both roots. Canonical comparisons.md and paired CSV
replace40 interim local6/remote2 rows with36 full8 rows, preserving404
historical rows. Remote pull recovered4 summaries; host permissions blocked25
compact files, recovered byte-identically through the existing stopped
container. Eight remote logs were summarized remotely without copying them.
No code/config edit, new evaluation, container restart, cleanup or new helper.
Design: configs/methods/FAMILY_BALANCED_PAIRED.md.
Code SHA256: ce423a34f66f373427bc44539335af2a5a1206f0f79db1515c4b20bd98644910.
Remote snapshot: 7ed14b60b67e9177ac01a8fc8b71f4e7ae9ff0fc8c07ac2770d92aba79676259.
Local root: results/local_runs/bcma_docker_20260911/family_balanced_paired_20260912_local.
Compact remote: results/remote_runs/family_balanced_paired_20260912_dsba3_gpu2.
Remote originals: hun_fsad_tta_012:/results/family_balanced_paired_20260912_dsba3_gpu2.
Audit: local root/collection_audit_20260913.json. Both queues and the local
collector have finished; resume history and original logs remain on disk.

## Latest execution — L1/L2/L3 full8 complete and collected, 2026-09-12

All48 runs completed without recorded failures: local36/36 at20:58:19 KST,
dsba3 12/12 at19:27:16 KST; both queues started17:54:05 KST.
Independent branches, not cumulative components: K1 -> L1 -> L2C -> L2 -> L3C -> L3.
Seed0/normal16/64 real updates/448+672, seen TESTpublic development.
Primary full8 macro AUPRO@.05; pAUROC secondary, oracle F1 diagnostic only.

| Stage | AUPRO x100 | pAUROC x100 | Oracle F1 x100 | AUPRO delta vsK1 (pp) |
|---|---:|---:|---:|---:|
| K1 | 47.71643 | 76.74333 | 36.22386 | — |
| L1 | 47.70081 | 77.16674 | 36.88045 | -0.01562 |
| L2C | 31.93149 | 70.16059 | 29.32723 | -15.78494 |
| L2 | 32.82129 | 70.45137 | 29.61005 | -14.89514 |
| L3C | 47.97341 | 77.16807 | 36.43473 | +0.25697 |
| L3 | 47.93244 | 77.13435 | 36.43894 | +0.21601 |

L1 operating-region ranking has no primary full8 gain: local6 +0.45701pp
does not survive remote classes, especially vial -2.61600pp. pAUROC +0.42341pp
and oracle F1 +0.65659pp are secondary tradeoffs, not primary success.
L2 same-shift restoration beats canonical-target L2C by0.88980pp, but is
below K1 on all8 classes. Target-choice benefit does not validate the restorer.
L3 exact meta-update is -0.04096pp vs same-exposure pooled-ERM L3C. L3C is
the simpler best-primary candidate within this batch, not a significant or
paper-ready winner. Its gain cannot be attributed to meta-learning; two-source
training exposure and both-source bank exclusion differ from K1.
No new experiment is active or scheduled by result collection.

All48 artifacts and original training logs pass complete finite/nonconstant
prediction coverage, steps0..63 plus fit_finished, normal16, frozen-backbone,
and maps/checkpoint-off checks. All8 class sextets match full support/query/GT
protocol, effective common config, backbone, and within-host environment.
Python/package versions match across hosts; kernel/glibc differ. Same device
per class across methods: local GPU0 observed450W for six classes, dsba3
physicalGPU2 for vial/wallplugs, one worker/GPU. Mixed-device, not equal compute.
K1/L1/L3C/L3 train100384 parameters; L2C/L2 train32768 with different scoring.
L3C/L3 use8 training views vsK1's4. Summed class fit time K1 1658.70s,
L3C 3234.90s, L3 3337.96s; not wall time or inference latency.

Matched config contrasts: L1 adds operating-region flags only; L2C/L2 differ
only by restoration_target; L3C/L3 only by cross_environment_meta.
Read-only log aggregation confirms recorded source/family/synthetic-selection
sequences match for all8 pairs in these contrasts. All512 L3 virtual steps
are nonzero; L3C has512 zero virtual steps. The switch executed, but this
does not demonstrate synthetic-transfer-to-real-defect generalization.
Detailed design: configs/methods/OPERATING_RESTORE_TRANSFER.md.
Prelaunch149 targeted tests passed in each existing container; full-shape
random-feature CUDA smoke passed7.459GiB peak without loading the backbone.

Common code SHA256:
91d308922d66b3ceacc0c09200c3c09e7323beedc967b2ad4b7d9c0dd4292312.
Remote immutable snapshot:
be30187a3e25dc5232b81977d796f29409cadb3ba49cb80c4afd3ea8709d0afd.
Local root: results/local_runs/bcma_docker_20260911/operating_restore_transfer_20260912_local.
Remote originals: hun_fsad_tta_012:/results/operating_restore_transfer_20260912_dsba3_gpu2.
Compact copy: results/remote_runs/operating_restore_transfer_20260912_dsba3_gpu2.
Generic pull encountered host permissions on compact files; read-only transfer
through the existing container recovered37 status/config/metrics/manifests,
all byte-identical. Full training logs remain remote; scalar summaries only.
Generic collect ran on both roots, then full8 canonical Markdown/CSV replaced
60 interim rows with54 rows, preserving350 historical rows. Historical K1/K3
47.69026/47.65616 remain references; use current K1 47.71643 for this batch.
No code/config edits, new experiments/evaluation, restart, GPU change or
artifact deletion during collection. Single-seed seen-development evidence only.

## Previous execution — K1/K3 complete and collected, 2026-09-12

User requested the next methodology after K2 analysis. K3 keeps K1's synthesis
and loss/controller, projecting the actual Adam parameter displacement against
the current support preservation gradient and available first-order budget
slack. It does not claim finite-step/all-support preservation or performance
improvement. No new confidence/synthetic/shift component is combined.

Method: `src/fmad/methods/projected_update.py`; on/off config:
`configs/methods/projected_update_K3.yaml`; design and future matched comparison:
`configs/methods/PROJECTED_UPDATE.md`. Off reproduces K1; both switches off H1.
One clean scoring VJP reuses the current frozen features, no extra backbone
extraction/trainable parameters/inference branch or cross-step dense storage.
Adam proposal moments are retained; actual rounded step diagnostics are scalar.

133 focused CPU tests passed in both existing local/dsba3 containers, including
off parity, actual Adam-momentum projection, zero/tiny-gradient/slack cases,
unchanged synthesis/backbone calls, gradient/state preservation, shared config,
and output/runner regressions. Local container was idle before source edits.
User then authorized launching K1 -> K3, full8/seed0/normal16/448+672/64 steps,
seen TESTpublic, primary full8 AUPRO@.05, at most two workers per assigned GPU.
Local six classes: can/fabric/fruit_jelly/rice/sheet_metal/walnuts; dsba3 GPU2:
vial/wallplugs. No extra seed/sweep. Maps/weights off, no Git push/provisioning.
Activated remote immutable snapshot:
`f99c089e2035d57fc849caef5887485925ae6c7cd5008ec5a835823ced85fbe5`.

Initial queues started 14:41:17/18 KST. Local rebooted at14:42:57; its last
training logs were14:42:10. No completed local metrics existed. Cause is unknown;
this is interrupted execution, not evidence against either method. Remote work
survived the local outage and completed4/4 (no recorded failures) at14:59:07 KST.
Do not relaunch remote work. Remote compact artifacts were subsequently
recovered and verified during full8 collection below.

User requested retry. Started the SAME local container and checked an actual
CUDA allocation. Current GPU0 power is450W after reboot, retained without a
power/driver change; original interrupted attempts were500W. Both new local
K1/K3 controls run at450W; do not treat the old500W attempts as compute-matched.
Archived two incomplete can/fabric worker trees under the same queue's
`interrupted/reboot_20260912_144257/`, outside active method stages. Original
logs/configs/status snapshot are preserved; no files were deleted. Recorded
the host interruption as invalid_run, then used the existing CLI helper's
`--resume`. No checkpoint exists: incomplete class fits restart from step0.
Local resumed15:07:05 KST; K1 can/fabric processes were verified on GPU0, followed
by the remaining K1 classes and K3. One interruption record remains in status.

Local queue: results/local_runs/bcma_docker_20260911/projected_update_20260912_local.
Remote: hun_fsad_tta_012:/results/projected_update_20260912_dsba3_gpu2.
Resume log: results/local_runs/bcma_docker_20260911/projected_update_20260912_local.resume1.launch.log.
One generic local collect --watch process was restarted, updating the canonical
Markdown/CSV; its log is results/projected_update_20260912_collect.resume1.log.
No source/config synchronization or modification during recovery.

Collection complete: local12/12 finished15:45:23 KST, remote4/4 at14:59:07.
All16 final results pass coverage/finite/64-step/normal16/100384-parameter and
output-policy checks. All8 pairs match full protocol/content, common effective
config (excluding method/switch/output path), code/backbone/software environment.
Code hash: a10803343ea5211db5fbfcbe868f3dbaaef7087a9a39dfc9786fb600cc028e42.
Full8 x100 K1/K3: AUPRO47.69026/47.65616, pAUROC76.74015/76.70431,
oracle F1 36.21609/36.11109. Deltas -0.03410/-0.03583/-0.10500pp.
Five class AUPRO gains, three declines; vial -0.27628pp. Valid comparison,
no primary improvement or vial recovery; no promotion of K3 over K1.

Read-only original-log aggregation: K3 projection89/512 steps, preservation
replay error0 at512/512, max first-order residual1.0722e-11. K1/K3 synthetic
selection/source/family sequences match, weak345/512 both. Mean P/b falls
0.82623 ->0.72348; vial1.47532 ->1.01179 despite worse AUPRO. Reducing this
surrogate alone is insufficient here; no causal/generalization claim follows.
Sum class fit time2302.36 ->2339.86s (+1.63%), not queue wall time/equal compute.

Generic collect ran for both roots. Permission-denied compact files recovered
read-only through the existing remote container after generic pull;13 status/
config/metrics/manifest files byte-identical. Full remote training logs stayed
remote. Local compact root: results/remote_runs/projected_update_20260912_dsba3_gpu2.
Canonical Markdown/CSV replace20 local6/remote2 interim rows with18 full8 rows,
preserving332 historical rows; archived interrupted attempts remain excluded.
No new evaluation, code change, restart or GPU-setting change during collection.
Single-seed seen-development evidence only. No next experiment is active.

## Latest execution — H1/K1/K2 complete and collected, 2026-09-12

User authorized method implementation and experiments after the K1 diagnosis.
K2 replaces frozen-teacher candidate selection with current-student paired
tail/random ranking-pool selection. Design/budget: configs/methods/STUDENT_BOUNDARY.md.
No loss/controller/inference change; K2 off reproduces K1, both off H1.
108 focused tests passed in both existing containers. Generic parallel collect
active-status rendering is fixed and tested before any new run.
Remote immutable source snapshot:
`17aaa3c579b3eded83be4a6c2a841f8985d6ba5eb497b9eb9307f2fce6ab2ffc`.
Local container had NVML/CUDA access failure despite a healthy host. It was
idle (sleep infinity only); restarting the SAME container restored CUDA.
No recreation, driver/power change or Git push. Host GPU0 power is now 500W;
all new paired controls must use the current conditions, not assume the old
300W runs are compute-matched.

Locked queue: H1 -> K1 -> K2, seed0/normal16/448+672/64 steps, seen TESTpublic,
full8 macro AUPRO@.05 primary; 24 class runs max, 3600s/class, no sweep.
Local GPU0 six classes and dsba3 GPU2 vial/wallplugs, two workers/GPU.
Completed roots: results/local_runs/bcma_docker_20260911/student_boundary_20260912_local
and hun_fsad_tta_012:/results/student_boundary_20260912_dsba3_gpu2.
Both queues completed with no recorded failures: local18/18 at12:34:37 KST,
dsba3 6/6 at12:04:02 KST. Assigned GPUs are idle; collector/launcher have exited.
Full8 x100 H1/K1/K2 AUPRO:47.41422/47.68475/47.55686;
pAUROC:76.54504/76.72671/76.62417; oracle F1:35.81383/36.18544/35.97395.
K2-K1 primary delta -0.12789pp: no primary improvement. Vial +2.06683pp vsK1
and +0.09190pp vsH1, but only2/8 classes improve vsK1 (vial/sheet_metal).
All24 outputs valid and all8 triplets match full protocol/content/common
effective config/code/backbone/recorded environment. K2 preview/replay error
is0 at all512 steps; feasible290/512, strength<1 at177/512 (34.57%).
Correct objective preview and class recovery do not establish broad benefit
or a causal mechanism; this remains single-seed seen-development evidence.
The initial host background launchers exited before remote/collector startup;
no remote class run was created. Separate-session launch restored them without
restarting or duplicating the local queue. Retain both launch logs as evidence.
Remote compact artifacts retrieved through the existing container after host
pull permission errors (no chmod/full-log transfer). JSON copies differ only
by an added terminal newline. Canonical full8 Markdown/CSV replace21 local6
rows with27 full8 rows and preserve305 historical rows. No next run launched.
Maps/weights remain off; candidate selection retains scalar diagnostics only.

## Previous completed execution — H1/K1 collected, 2026-09-12

User requested a single-core-problem improvement after the multi-angle audit.
K1 targets easy/misaligned synthetic supervision by changing only synthetic
strength construction relative to a frozen, source-excluded normal boundary.
Design and locked budget: `configs/methods/BOUNDARY_SYNTHETIC.md`.
Method: `src/fmad/methods/boundary_synthetic.py`; config switch off reproduces H1.
H1 ranking, preservation/controller, shift pairing, inference and evaluator
remain fixed; no J1/I1/I2/I3 or F2 reliability component is combined.

86 targeted tests passed in each existing local/dsba3 container, including
H1-off parity, source exclusion, frozen-teacher selection, unchanged RNG,
exact replay, scalar-cache/output policy and collector tests. CUDA checks
passed on the assigned GPUs. No Git push or container recreation.
Remote immutable source snapshot:
`1b9677fab5b80bd7a679aa01a23ee7398aba89f4cabc4a3d47a967d1491046d0`.

Completed comparison: H1 -> K1, seed0/normal16/448+672/64 steps, seen TESTpublic,
full8 macro AUPRO@0.05 primary; 16 class runs maximum, 3600s/class, no sweep.
Two workers per GPU. Local GPU0: can, fabric, fruit_jelly, rice, sheet_metal,
walnuts; dsba3 physical GPU2 only: vial, wallplugs. All pairs stay on the same
device. No additional inference cost/parameters; teacher candidate forwards
increase fitting compute and are declared in the design. Maps/weights off.

- Local status: `results/local_runs/bcma_docker_20260911/boundary_synthetic_20260912_local/status.json`.
- Remote status: `hun_fsad_tta_012:/results/boundary_synthetic_20260912_dsba3_gpu2/status.json`.
- Local completed12/12 at10:54:15 KST; remote4/4 at10:34:01 KST. Both statuses
  complete, no recorded failures; neither GPU is running this comparison now.
- Full8 x100: H1 AUPRO47.40645 / pAUROC76.52947 / F1 oracle35.78126;
  K1 AUPRO47.70115 / pAUROC76.73356 / F1 oracle36.20219.
  Changes: +0.29470pp / +0.20410pp / +0.42092pp respectively. Six classes improve;
  vial -1.89786pp and sheet_metal -0.15140pp. A modest single-seed seen-development
  gain, not statistical/generalization evidence or proof of the synthetic mechanism.
- All8 pairs match protocol/support/query/GT/common config/code/backbone/recorded
  runtime. All16 pass finite/coverage/64-step/normal16/100384-parameter/output checks.
- Generic terminal collect succeeded after its earlier live parallel-status
  KeyError('object'). Remote compact artifacts were read in the existing container
  after host pull permission failures; no permission changes or full log pull.
  Local copy: results/remote_runs/boundary_synthetic_20260912_dsba3_gpu2.
- Canonical CSV/Markdown replace local6 interim14 rows with full8 18 rows and
  preserve287 historical rows. No new experiment, restart or source fix was made
  during status checking. Generic collector's active-status fix remains pending.

## Previous completed execution — J1, 2026-09-12

J1 hard-pair tail risk has completed and been collected (16/16, no failures).
Locked design: `configs/methods/TAIL_RISK.md`. It retains H1's exact sampled
normal/synthetic pairs and changes only mean hinge reduction to the upper
25% mean. The new method `src/fmad/methods/tail_risk.py` has a `tail_risk`
on/off switch; off reproduces H1. No I1/I2/I3 component is combined.

Both local and dsba3 passed 96 targeted tests (off parity, gradients, replay,
fixed common config, frozen backbone, output policy, and collector checks).
Ruff was not installed; no lint success is claimed. New source snapshot:
`6e792a707969c2eb01ed78aa1173de8c222f86ced3705909c9f0171b9761e9c1`.
Completed remote runs used that immutable release. Neither queue is active.

H1 -> J1, two workers per GPU, single seed0/normal16/448+672/64 steps,
`test_public_seen_development`, full8 AUPRO@0.05 primary, no sweep, 16 class
runs maximum, 3600s/class. Local six classes: can, fabric, fruit_jelly, rice,
sheet_metal, walnuts. dsba3 physical GPU2 only: vial, wallplugs. Each class's
two methods run on the same device. Maps/checkpoints remain off.

- Local: `results/local_runs/bcma_docker_20260911/tail_risk_20260912_local/status.json`.
- Remote: `hun_fsad_tta_012:/results/tail_risk_20260912_dsba3_gpu2/status.json`.
- Each root has a sibling `.launch.log` with exit code and per-class logs.
  Local finished 09:44:39 KST, remote 09:28:28 KST; both exit 0.

Full8 AUPRO@0.05 x100 H1 47.40352 -> J1 47.53549 (+0.13197pp), pAUROC
76.53814 -> 77.00359, F1 oracle 35.80016 -> 36.49949. Six classes improve;
vial -3.03246pp and fabric -0.23413pp decline. All eight pairs have matched
content/config/code/backbone/recorded runtime, and all16 final artifacts pass
finite/coverage/64-step/parameter checks. Canonical CSV/Markdown contain 16
class + 2 full8 macro rows; historical rows are unchanged.

This is a small observed single-seed development gain, not statistical or
sealed-final superiority. Vial final P/b rises 0.67770 -> 1.41967. J1's final
tail loss is approximately 4x sampled mean loss for every class, so hard-pair
selection cannot yet be credited separately from loss reweighting. No new
experiment or method revision was launched during completion checking.

The host had rebooted outside this task (boot 08:39:43 KST). Loaded NVIDIA
driver and NVML now match at 580.178.04. The same stopped local container was
started, not recreated; its CUDA check passed. dsba3's assigned GPU was idle
and its CUDA check also passed before launch. No host reboot, driver install,
container replacement, Git push or GPU expansion was performed by this task.

## Previous completed comparison — 2026-09-12

Completed comparison launched at 06:22:21 KST: H1 control -> I1 selective
preservation -> I2 paired local margin -> I3 frozen-reference retrieval.
These are three independent H1 extensions, not cumulative components.
85 tests passed locally and the same 85 on dsba3 before launch. Configs and
equations: `configs/methods/THREE_DIRECTIONS.md`. Method files are under
`src/fmad/methods/`, each with a config on/off switch verified to reproduce
H1 when off. Shared CLI, fitting loop and evaluation remain unchanged.

New assignment (all four methods use the same device for each class):

- Local RTX4090 GPU 0: can, fabric, fruit_jelly, rice, sheet_metal, walnuts;
  `results/local_runs/bcma_docker_20260911/three_directions_20260912_local/status.json`.
- dsba3 physical GPU 2 only: vial, wallplugs;
  `/results/three_directions_20260912_dsba3_gpu2/status.json` in `hun_fsad_tta_012`.

Two workers per GPU. Both queues completed: local 24/24 at 07:14:19 KST,
dsba3 8/8 with wrapper exit 0 at 06:47:43 KST; no recorded failures.
Source snapshot:
`bedcfcda2cf3963ff5105eac9f81bf4faae721176dc9deea59266f523bd5b960`.
All 32 class runs (8 control, 24 candidate), seed0/normal16/64 steps/448+672,
seen-development, were collected without new evaluation. Primary full8
AUPRO@0.05 x100: H1 47.41448, I1 47.30383 (-0.11065pp), I2 46.95461
(-0.45986pp), I3 47.37585 (-0.03863pp). No candidate clears the primary
metric. I1/I3 small changes do not establish superiority, equivalence, or
family-wide failure; one seed gives no variance estimate.

All 24 candidate/control pairs match support/query/GT, common config, code,
backbone and recorded runtime within class. All 32 final artifacts pass
coverage/non-finite checks and record 100,384 trainable parameters, 64 steps,
frozen backbone, maps/weights off. The canonical CSV/Markdown now contain
32 class + 4 full8 macro rows instead of the local6 interim rows. Remote
metrics were retrieved; configs/manifests/training logs remain remotely.

Main observed feedback: I2 paired ranking is zero on 306/512 steps; mean
synthetic uplift 0.33731 exceeds mean margin 0.05584. Same-definition support
P/b falls from H1 0.60677 to I2 0.21664, yet all aggregate metrics decline.
Easy synthetic-objective satisfaction is not evidence of real-defect
improvement. Loss of normal-tail competition and synthetic-to-real mismatch
remain competing post-hoc explanations. I1 loses vial/wallplugs; I3's vial
gain is offset mainly by wallplugs. No new method or experiment was started
during collection. See the latest full8 analysis in results/comparisons.md.

Environment note: the 06:54 local automatic NVIDIA update caused host
NVML/kernel mismatch (580.178.04 versus loaded 580.173.02). The existing
container retained matching 580.173.02 libraries, and its post-completion
GPU query/CUDA arithmetic check passed. No restart or failure was recorded;
this does not certify stability after a future restart.

Previous H1/G1 comparison is now complete (16/16, no failures): local ended
01:12:57 KST, dsba3 ended 01:42:56 KST. Full8 AUPRO x100 G1 46.71464,
H1 47.42062 (+0.70598pp); pAUROC 76.75832 -> 76.54791. H1 improves 3/8
classes, driven by vial +8.25242pp, but declines on 5/8. Vial final P/b
4.16909 -> 0.67699; can/fruit_jelly also reduce P/b but lose performance.
All eight pairs match content/code/environment/backbone and parameters within
class. This replaces the earlier local4-only intermediate verdict. The
canonical CSV/Markdown now contain the full8 result and class provenance.

Previous completion (2026-09-12 00:27 KST): GARD G1/G2 completed 16/16 class
runs in 32.04 minutes, without recorded failures. G1 AUPRO@0.05 = 46.71704,
G2 = 46.32650 (x100); G2-G1 = -0.39054pp. D3 remains the stronger recent
local method reference at 48.96953. F1/G1/G2 support/query/GT content,
backbone provenance, and environment match for all 8 classes. These are
seed0/normal16 seen-development results, not a sealed or multi-seed claim.
See the cumulative ledger for class-level evidence and interpretation.

Completed candidate (2026-09-12): H1 addresses only the zero/delayed
preservation coefficient observed in G1 vial training. The sole config change
from G1 is `immediate_preservation: true`: detached current-step
`clip(P/b, lambda_init, lambda_max)` replaces EMA-dual accumulation. Synthetic
gating, confidence, shared pipeline and all other G1 settings remain unchanged.
50 targeted local-container tests passed before its original launch. Benchmark
status: **16/16 complete**. Full8 AUPRO improves, but the gain is not uniform;
hard-budget satisfaction and causal attribution are not established.
See the H1 section in
`configs/methods/RELATIONAL_TAIL_GARD.md` for the one-comparison boundary.

The completed original G1 -> H1 comparison used this historical split:

- Local `fsad-tta-local-bcma`, GPU
  `GPU-0070b668-714f-bcbb-6c46-f0b68c9b7f6b`: can, fabric, fruit_jelly, rice.
- dsba3 `hun_fsad_tta_012`, physical GPU 2
  `GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639`: sheet_metal, vial, wallplugs, walnuts.

Each device ran two independent class workers; G1 completed its assigned
classes before H1 started on that same device. Seed0, normal16, 448+672,
64 steps, `test_public_seen_development`; primary full8 AUPRO@0.05.
Budget: 16 class runs total, 3600-second timeout per class, no sweep. Match
support/query/GT hashes and evaluator provenance before interpreting paired
deltas; the combined full8 scope uses mixed hardware, identical within each
class's method pair. Historical all-local results remain references.

Local status: `results/local_runs/bcma_docker_20260911/preservation_h1_20260912_local/status.json`.
Remote status: `/results/preservation_h1_20260912_dsba3_gpu2/status.json` inside
the existing dsba3 container. Each queue has a sibling `.launch.log` and
per-class logs. Both used detached Docker execution; remote execution held
`/tmp/fmad-remote-execution.lock` throughout. Its completed remote source snapshot was
`87244c8a01bac3fa87eea04dec32deed05f4f8672033b18627bf0adcf743d08d`.
Maps/weights are off; retain remote logs/provenance and retrieve metrics only.

BCMA B0–B3 completed all 32 class runs; B2 AUPRO@0.05 = 46.83252 (×100),
versus B0 46.66157. See the ledger for matched conditions and historical
comparisons. The statements below reviewed on September 8 are earlier context.

TailContrast D0–D3 completed all 32 class runs on September 11. D3 AUPRO@0.05
48.96953 vs D2 47.14817 (x100), with 6/8 class gains but a -13.34778pp vial
regression. These are seed0 normal16, 448+672 test_public seen-development
results, not a sealed final evaluation. D0 reproduces B2 on all 8 classes.

The user authorized implementation and sequential E1–E3 experiments following
the D3 analysis. GuardedMetric uses an original-space bounded rank-16 metric,
then nuisance-direction contraction protection, then weak synthetic/hard-positive
training. See `configs/methods/GUARDED_METRIC.md` for the locked design,
assumptions, changed parameter/compute conditions, and 24-class-run budget.
Reuse the same `fsad-tta-local-bcma` container/GPU; maps/checkpoints remain off.
The existing D3 results are the reference, not a requested rerun. No new-stage
performance is asserted before completed artifacts exist. Execution status:
`results/local_runs/bcma_docker_20260911/guarded_metric_20260911/status.json`.
The generic `scripts/experiments.py` sequences the shared CLI and collects
completed metrics into the same cumulative Markdown/CSV. See the status file
for actual running/completed state rather than inferring it from this note.

## Direction

The project is using the refactored common FMAD-OOD pipeline. New work should
change the method behind the shared support-selection, backbone, fitting,
prediction, and evaluation interfaces so controlled method comparisons remain
meaningful. The current research direction is method/performance-first for a
CVPR 2027 submission: a short hypothesis may proceed to implementation and a
controlled development comparison before a detailed mechanism analysis. Add
mechanism, ablation, and generalization evidence after a candidate is shown to
be worth extending.

## Remote execution policy

Current user direction (2026-09-12): this experiment may use the local GPU and
**only dsba3 physical GPU 2**. This overrides the earlier all-four-GPU request;
the four-GPU private preset is not permission to use the other devices.
Select GPU 2 by UUID explicitly. Match the local `fsad-tta-local-bcma` environment and
data/protocol, update code only when its selected version changes, and retrieve
only evaluation metrics (`scripts/run_remote.sh pull --metrics-only`). Keep
configs/manifests/logs remotely; maps and checkpoints remain off. Existing local
experiments are not stopped, migrated, or duplicated by this setup request.

Setup completed on September 12 after SSH access recovered: reused/restarted
`hun_fsad_tta_012`, retaining its original Python 3.10/Torch 2.1.2 environment.
The helper selects an isolated copy of the local Python 3.11.16/Torch
2.6.0+cu124 runtime. Imports, CUDA tensor smoke, CLI help, and 56 targeted
remote tests passed. OS/GPU differ, so application compatibility does not imply
bitwise-identical execution. Four GPU UUIDs are configured/visible. Data file
name/size inventory and DINOv2 repository/checkpoint hashes match local assets.
No model/data download, remote benchmark, or local experiment restart occurred.

The user explicitly selected local code over the legacy remote checkout.
`configs/remote/dsba3.env` now chooses `REMOTE_CODE_MODE=local`; `update-code`
uploads changed content-addressed snapshots, validates the archive, and switches
`/workspace/.local-sync/current` under the experiment lock. Identical code
skips transfer (verified live). Old code/results remain untouched. Logs and
source hashes remain remote; `pull --metrics-only` retrieves metrics only.
`--workers 4` alone still uses one device; multi-GPU experiment scheduling is
not part of this environment-only task and requires explicit device assignment.

Use one designated existing container throughout the research cycle. The active
default procedure is local edit/test, authorized commit/push, container
`update-code` (`git pull --ff-only`); dsba3 uses the explicitly authorized
local-snapshot exception above. Then execute on explicitly assigned GPU(s), with compact
quantitative result/provenance pullback. Restart the same stopped container;
never recreate it automatically. A missing container or incompatible exposure
is a configuration blocker, not permission to provision a replacement.

See [remote runtime instructions](../README.md#remote-experiments-reuse-one-existing-container)
for prerequisites, logging, locking, the result allowlist, and dsba3 setup
details. Environment checks are not benchmark performance evidence.

## Retained methods and baseline status

- Retained comparison methods are `superad` (SuperAD) and `rn_fmlk` (RN-FMLK).
- The current default configuration is `superad` on `mvtec_ad2` with
  `dinov2_vitl14`, seed `0`, and `protocol.shots: 16`.
- An actual GPU baseline run for this refactored configuration is pending. No
  GPU metric or improvement is asserted here.

## Protocol and metric status

- The common runner now selects an exact number of seeded normal support images.
  This differs from the earlier SuperAD internal 10% coreset protocol, so old
  results are not automatically comparable. Use a matching fixed manifest and
  matching conditions when making historical comparisons.
- The retained evaluation contract is
  `mvtec_ad2_float16_streaming_v1`, including `seg_AUROC_0.05` and
  `seg_AUPRO_0.05`. It also exposes `seg_F1_oracle`, whose best threshold is
  selected with evaluation ground truth; it is an oracle diagnostic, not a
  deployable threshold. The primary metric for the current paper comparison
  is still pending an explicit declaration; do not infer it from an old report
  or from the presence of a metric in a config.
- When reusing historical numbers, distinguish pixel partial-AUROC@0.05 from
  AU-PRO@0.05. Do not assume that a reported value under one label is the other.
- CPU normalized kNN now uses the same cosine-distance definition as GPU search;
  old CPU results based on the inverted L2 score are not equivalent.
- No holdout is presumed untouched. Each run must record the actual data split,
  support/query identities, and any label access used for tuning or evaluation.

## Execution limits to address before relying on final evidence

- The current runner uses the dataset's default `test_public` query/GT paths;
  a configured development/final partition and label-exposure record are not
  implemented. Declare the actual development population now. Before claiming
  sealed final evaluation, provide and verify an appropriate evaluation path;
  repeated seeds on a seen population do not remove tuning exposure.
- Training-session failures are logged, but prediction/evaluation failures are
  not covered by a durable run-level failure record. The final run manifest
  and metrics are written after object processing. Until this is addressed,
  preserve command/exit/error evidence in the experiment record and explicitly
  mark interrupted runs incomplete or invalid; absence of final files is not
  a completed negative result.
- These are known implementation gaps, not completed fixes or a requirement
  to perform broad infrastructure cleanup before method development.

## CVPR 2027 schedule

The official CVPR 2027 call lists the following 2026 Anywhere-on-Earth (AoE)
deadlines:

- Registration: 2026-11-10 AoE
- Paper submission: 2026-11-16 AoE
- Supplementary material: 2026-11-23 AoE

See the [official CVPR 2027 call for papers](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)
for the authoritative schedule. These dates inform planning; they are not
automatic experiment gates in this repository.

## Suggested next milestones (not hard gates)

1. Explicitly declare the primary metric and comparison protocol.
2. Run and record the pending GPU SuperAD and RN-FMLK baselines under matching
   conditions.
3. Implement the next candidate as a method-only change in the common runner.
4. Compare it on the same support, seed, protocol, and declared metric before
   deciding whether to spend budget on ablations or generalization.
5. Add the paper-ready repeated runs and analyses that support the resulting
   claim, labeling any post-hoc hypothesis accordingly.

## Document roles

Read this file for current status, `skill_graph/todo.md` for the active task
checklist, and `skill_graph/log.md` or dated experiment folders for evidence.
Archived README and working-memory files are historical references only.
