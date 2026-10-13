# Skill Graph TODO

> Active queue reviewed 2026-09-25. Dated completed entries are retained. The
> pre-transition checklist remains available in
> [`todo_history_pre_2026-09-08.md`](todo_history_pre_2026-09-08.md). Its old
> approvals, bans, and experiment-specific stop conditions are historical
> evidence, not current instructions.

## Active v7 evidence takeover — 2026-09-25

- [x] Transfer operations from Claude to Codex; retain running local/remote artifacts.
- [x] Correct E5 signed-drop explanation, E7 regular-stratum analysis, and E10 matched-class interpretation.
- [x] Register completed E4 Gaussian full12 through the canonical collector.
- [ ] Complete E4 shot/impulse/speckle full12 with the existing immutable local snapshot; assess G/P and firing counts.
- [x] Validate GPT-6 Sol fixes for pooled RNG, zero-reference guard, and import scope/provenance; 103 tests pass, independent review approved, local commit a9b9e36.
- [x] Collect E1_v2 (128/128); C0 support pairing passes, R1 passes on five seeds (+3.78pp mean difference), and R2 reports 56.07±2.39 (2-shot) vs 59.48±0.60 (16-shot).
- [x] Collect I1a Gaussian full15 (30/30); lowres_pool AUPRO 52.82 is below the 72.009 gate, so I1b is closed.
- [x] Collect E9 MVTec/VisA (435/435, 420/420); per-class support hashes match the stored RSG3 draws. RSG3 clears the +0.86pp lead threshold on VisA at 4/8/16 shots and on no MVTec shot; see each run comparison for rankings.
- [x] Complete and collect E12 resolution control (24/24): multiscale RSG3 exceeds 1120-only with matched total bank capacity by +2.06125pp on primary regular AUPRO; all three acquisition shifts also favor multiscale. Retain the single-seed development limitation.
- [ ] E11 shot-shift was stopped at the user's request before collection (8 completed cells preserved; four lanes marked `invalid_run`). Do not resume until the preceding analyses are complete.
- [x] Align Outline_v7 and experiment_results with confirmed observations, pending gates, and limitations; refresh again when pending results complete.
- [ ] Update remote code only after current snapshot users exit; no automatic push or official submission.

## Completed noise attribution + guarded remedy — 2026-09-23

- [x] Attribution: 896/1120 branches carry the Gaussian-noise failure (relief +5.09pp).
- [x] Noise guard (k=9) + lowres remedy; G1/G2/P1/P2 all pass; held-out mean +7.94pp.
- [ ] Decide whether to pursue the residual bottle/zipper gap (local feature pooling).
- [ ] Add the remedy result to Outline_v6 section 4.5 and the corruption table.

## Completed coverage-void instrumentation — 2026-09-23

- [x] Lock protocol and H1-H5 rules; nested 1/2/4/8/16 manifests; 120 cells local + dsba3.
- [x] Collect with balanced panels + dedup; verdict H1/H5 FAIL, latent-void hypothesis rejected.
- [ ] Correct Outline_v6 overclaim (macro flatness is per-class cancellation).
- [ ] Seed replication of the shot-axis comparison before any cross-shot claim.
- [ ] Update Outline_v6 4.5 (cross-method corruption table) and 4.7 (audit 32/32, gate failed).

## Completed external-method corruption comparison — 2026-09-22

- [x] Lock full15 normal16 seed0 clean + four-corruption protocol for six
  published comparators and existing RSG3; verify corruption-root support is clean.
- [x] Validate comparator configs and launch clean locally plus four dsba3 lanes.
  - Local: RegAD/RD4AD clean. dsba3 GPU0: clean FAISS methods then brightness;
    GPUs1/2/3: contrast/defocus blur/Gaussian noise. FAISS GPU runtime smoke passes.
- [x] Complete450 new cells; verify support/config/evaluator provenance and finite coverage.
- [x] Collect method × corruption AUPRO/pAUROC and degradation table; replace the
  Section 8.1 internal-variant proxy with direct external-method results.
  - All525 cells including reused RSG3 are complete. Mean shifted AUPRO:
    SuperAD79.8073, PatchCore79.0452, RSG378.5434. RSG3 leads brightness and
    contrast but loses20.6873pp under Gaussian noise; claims are bounded accordingly.

## Completed source-fixed mechanism audit — 2026-09-22

- [x] Freeze32-cell B/C/D/E normal16 seed0 protocol, plus explicit16 normal
  calibration images; evaluator-only source thresholds and paired scene analysis.
- [x] Add non-mutating layer/component tracing and fixed FPR/TPR/PRO evaluation;
  local/remote81 tests pass, actual DINO prediction parity exact.
- [x] Restore GPU access by restarting the idle existing local container;
  launch existing local/dsba3 lanes without resource expansion.
- [x] Complete32 cells; verify original FF scores, support/bank/calibration pairing
  and native component reconstruction, then collect compact results.
  - Full8 registered; all32 FF metrics and banks exactly reproduced; paired hashes pass.
- [x] Assess primary source FPR1% regular→shift_1 FPR inflation and target PRO,
  component changes and LOO sensitivity; preserve negative/tradeoff findings.
  - Full8: D−B shift PRO +8.92754pp but FPR inflation +0.72606pp worse;
    only4/8 favorable. Prespecified composite gate fails.

## Completed RSG3 factorial — 2026-09-22

- [x] Lock five-arm40-cell protocol with true ungated-add controls and fixed banks.
- [x] Implement evaluator-only raw/CQ, acquisition and gate diagnostics; local and
  remote72 tests pass, actual frozen-DINO normal16 RSG3 readout parity exact.
- [x] Launch D/A/B/C/E on existing local and dsba3 devices with paired conditions.
- [x] Complete40 cells, verify coverage/content/bank hashes and D baseline parity.
- [x] Pull compact results, update canonical paired ledger via generic collector,
  and report scaling/gate/shallow contrasts plus raw/CQ and condition evidence.
- [x] Retain RSG3; separate strong gate/scaling effects from small shallow gains,
  failed shift-suppression framing and extreme-FPR tradeoffs. Seed replication
  qualifies for proposal only; no additional run started.

## Historical QR1 mechanism controls — 2026-09-20

- [x] Freeze64-run direction-control protocol, preserve data/pipeline and bank RNG;
  36 tests passed, one skipped, finite/repeatable same-bank smoke passed.
- [ ] Finish QM_A/B/C/D at1/16shot full8 on local/dsba3.
- [ ] Collect and verify provenance/compact artifacts; assess direction-specific
  AUPRO gains separately from normal/defect mean-margin mechanism gates.
- [ ] Report regular versus actual shift, shot dependence and threshold sensitivity
  without interpreting single-seed direction controls as novelty or causal proof.

## Completed real acquisition diagnosis — 2026-09-20

- [x] Verify official acquisition provenance, local paired scenes and per-condition GT.
- [x] Freeze32-run protocol, add condition metrics and calibration sensitivity;
  34 tests passed, one skipped.
- [x] Complete local/dsba3 QR0/QR1 at1/16shot on full8 and collect compact results.
- [x] Check primary shift_1 problem/shot-interaction/correction gates and
  threshold sensitivity; report other conditions without selecting new gates.
- [x] H0 passes, H1/H2 fail; close32-run budget, retain all artifacts, no new
  method/default change. Separate actual shift evidence from unsupported causes.

## Completed shot/shift diagnosis — 2026-09-20

- [x] Freeze48-evaluation protocol, nested support manifests, evaluator-only
  normal calibration and paired synthetic shift;32 tests pass, one skipped.
- [x] Finish local/dsba3 QR0/QR1/spatial-only QR3 at1/16shot on full8.
- [x] Pull/verify compact artifacts and collect through existing generic helper.
- [x] Report shot×shift normal FPR interaction, correction/defect TPR tradeoff,
  and spatial exclusion effect; apply the predeclared framing gates.
- [x] Both framing gates fail (H1:2/8, H2:3/8); close48-run budget without
  a new-method implementation or automatic replication. Preserve uncertainty
  and synthetic-versus-real shift limitations in the detailed report.

## Completed coherent-matching queue — 2026-09-20

- [x] Freeze CM_A/B/C/D and 32-evaluation protocol; implement method-only change.
- [x] Pass 63 tests, synthetic splice check, real DINOv2 zero-control and finite
  active/magnitude-control smoke. Local/remote resources verified idle.
- [x] Complete full8 seed0 on local and dsba3; retain remote logs.
- [x] Pull compact provenance, verify paired support/config/backbone, collect
  with the existing generic collector, and evaluate the predeclared gate.
- [x] Gate failed: C−A AUPRO−0.080607pp, improving3/8 classes. Close this
  candidate without replication or default change; all32 runs are indexed.

## Completed local-relation queue — 2026-09-19

- [x] Implement LR_A/B/C/D with unchanged RSG3 banks/pipeline and normal-only
  calibration; freeze the [protocol](../configs/methods/LOCAL_RELATION_20260919.md).
- [x] Validate 61 method/reference tests and actual-backbone zero-weight CUDA
  parity; validate the generic completed-shard collector/launcher with 55 tests.
- [x] Launch the initial 32 class evaluations in the existing local and dsba3
  containers; retain remote logs and save compact provenance, no maps/weights.
- [x] Complete full8 collection: all32 valid, zero-control max metric error
  2.69e-8; 92 remote compact hashes verified. Preserve 1,102 old CSV rows,
  add36 full8-scoped rows, verify repeat collection is a no-op. C>B and C>D:
  screening passed, not adoption; see the combined detailed report.
- [x] Finish launched paired A/C/D seed1–4 replication (96 class runs),
  audit and collect each full8 seed, and close the study with all D controls.
- [x] A/C completed for all five seeds: mean paired AUPRO +0.109944pp fails
  the predeclared +1pp adoption threshold (pAUROC -0.040428pp passes its guard).
  C>D in all five seeds, mean +0.118767pp AUPRO. Do not expand this candidate;
  the small positive effect is not zero and is not a general failure of geometry.
- [x] Audit all128 matched runs, 384 byte-preserved combined evidence files,
  and 324 remote compact SHA256 matches. Preserve all1,102 historical CSV rows;
  register144 unique new rows (128 class +16 full8 macro), total1,246. Retain
  remote logs/containers, maps and weights off. Conditional expansion skipped
  because adoption failed; no triggered experiment remains pending.

## Completed CRR execution queue — local and dsba3 four GPUs

- [x] Implement cross_image_reference.py and CR0--CR3 config-only ablations;
  preserve the common pipeline, source exclusions and bounded scalar caching.
- [x] Validate152 focused tests (5 skipped) plus separate real cached DINOv2
  full16-bank CUDA smoke for all stages; CR0 exact R1, finite trained outputs,
  identical descriptor/episode hashes and frozen backbone. Bash syntax passes.
  Ruff absent. No package install or container recreation.
- [x] Launch local can/fabric CR0->CR3 at12:28:55 KST, PID1470978, GPU0 UUID
  GPU-0070b668-714f-bcbb-6c46-f0b68c9b7f6b. Verify live CR0/can process1471079,
  running status and correct GPU.8 local class runs; no extra seeds/sweeps.
- [x] User corrected remote to dsba3 GPU0/1/2/3. Verify existing container,
  data/cache and all four physical GPU UUIDs/CUDA; sync immutable snapshot
  5b6ea70b4fb2d9d1c45b2f754beb4ed5a4b1ed0405b9a12b5eed584cb3a02e12.
  Local/remote package SHA and YAML hashes match; remote152 tests pass,5 skip.
- [x] Remote actual DINOv2 full-bank smoke passes all CR stages (two trained
  updates, no benchmark metrics). Launch24 remaining class runs12:44:19 KST
  remote-only, parent/session1496638. Verify all four GPU UUID processes and
  lane CR0 fit_started/running statuses without failures. Original local lane
  untouched;5/8 complete and CR2/fabric active at12:45:09.
- [x] Collect all32 completed runs; last completion13:46:20 KST. All92 remote
  compact files verified against original SHA256; full logs remain remote.
  Generic collect processed5 lanes; consolidate32 class +4 full8 macro rows,
  preserving548 historical rows (584 total). Matched conditions and CR0/R1/TF0
  parity confirmed; finite complete metrics and expected0/64 optimizer steps.
  CR2 AUPRO53.13319/pAUROC75.52551: +0.66286/-0.46664pp vs CR0.
  CR3 AUPRO52.29146/pAUROC75.91410: -0.17887/-0.07805pp vs CR0.
  Joint target not met; retain R1 reference. No new run or code/config change.

## Completed CRR design

- [x] Design [CRR CR0--CR3](analysis/2026-09-14_cross_image_reference_design.md):
  R1-preserving residual control -> cross-image tail ranking -> source-separated
  reference geometry. Define matched controls, leakage boundaries, output/memory
  constraints and the future full8 seed0 joint-metric comparison.
  No implementation or experiment was requested/executed in that design turn;
  later implementation/execution authorization and current status are above.

## Previously completed TBF execution queue

- [x] Implement TBF tail_aware_fusion.py, TF0--TF3 config-only ablations and
  locked design/protocol. Keep R1 banks/scales/backbone fixed; native sampled
  tail ranking and bounded global/local fusion only. No dense stored maps.
- [x] Local135 tests pass (4 skipped), plus actual cached DINOv2 CUDA smoke
  for all stages; exact TF0/R1 parity and identical TF1/2/3 episodes/scales.
- [x] Activate validated local-only snapshot in existing dsba3 container:
  1f36c150db82307f11045dc72856e419d756ce13d08c7ec0db7621836a710ef8.
- [x] Same remote validation:135 passed,4 skipped plus actual DINOv2 CUDA
  smoke; identical code/Torch/CUDA. Launch32 full8 class runs10:08:41 KST
  through generic run_experiments.sh on local0 and dsba3 GPUs0/1/2/3.
  Detached parent1249970, one lane/GPU; all five active on TF0 at launch check.
- [x] Collect all32 completed TBF runs; last completion10:58:09 KST. Verify
  all92 remote compact files against originals; retain full remote logs there.
  Generic collect processed5 roots; consolidate to36 full8 rows, preserving
  previous512 rows (548 total) in comparisons.md and paired CSV.
  All8 matched support/query/GT/scales quartets, identical64 episode triplets,
  finite complete results/logs. TF0 exactly reproduces previous R1 metrics;
  remote absolute support-path aliases are explicitly documented.
  AUPRO TF0/1/2/3:52.47033/52.70446/52.61031/52.76262;
  pAUROC:75.99215/75.79109/75.79416/75.51276. TF3 +0.29229pp AUPRO but
  -0.47940pp pAUROC vs TF0: targeted pAUROC recovery was not achieved.
  No next experiment scheduled; no source/config edit or new evaluation.

## Previously completed R/P queue

- [x] Implement two independent R/P families: reference_matching.py R0/R1/R2
  and masked_prediction.py P0/P1/P2, registered through the existing CLI.
  Record assumptions, feature/compute differences and locked full8 comparison
  in configs/methods/REFERENCE_PREDICTION.md. Maps/checkpoints off.
- [x] Implement generic scripts/run_experiments.sh: one Bash invocation,
  sequential configs per GPU and parallel disjoint GPU lanes, using existing
  experiments.py and run_remote.sh rather than method-specific queue scripts.
- [x] Pass88 focused tests in each existing local/remote container and real
  cached-backbone/full-width predictor CUDA checks. Verify matching Torch/CUDA
  and code hash. Sync immutable snapshot through existing remote helper only.
- [x] Launch64 planned class runs00:45 KST2026-09-14: D3 -> L3C -> R0 -> R1 ->
  R2 -> P0 -> P1 -> P2 on localGPU0 plus dsba3GPU0/1/2/3, one worker per GPU.
  At00:49 all five GPUs active; actual training/fit logs confirmed. Parent
  PID457405 survives detached; current.md records roots/provenance and placement.
- [x] Finish/validate all64 R/P suite runs: local16 +remote48; final completion
  05:17:32 KST2026-09-14. Exclude5 original R1 failures, preserve all artifacts.
  No active experiment; all64 final metrics/logs and full query coverage valid.
- [x] Repair shared R1/R2 CPU-staged-query/CUDA-bank calibration failure.
  Reproduce with3 CUDA regression failures, then pass97 tests in each existing
  container and real DINOv2 R1/R2 fit/predict at448+672. Add P CUDA fit tests.
  Generic resume audits byte-identical config relocation between immutable
  releases, preserving original plans/config hashes and failure history.
- [x] Resume all5 GPU lanes02:11:18 KST2026-09-14, parent583054. Preserve and
  skip24 D3/L3C/R0 completions;40 runs pending at restart, all lanes start R1.
  Record changed code hash explicitly; configs/support/metrics stay unchanged.
- [x] Recover compact remote artifacts and verify184 byte-identical files;
  remotely audit48 logs without full-log pull. Generic collect processed5 roots;
  canonical Markdown/CSV now72 full8 rows, preserving440 historical rows.
  All8 support/query/GT octets matched; all8 P triplets share128 episode hashes.
  AUPRO D3/L3C/R0/R1/R2/P0/P1/P2:48.60330/47.96204/46.66157/52.47033/
  42.92744/14.73310/16.01451/14.24413. R1-L3C +4.50830pp but lower pAUROC;
  R2-R1 -9.54289pp. Best continuation R1; no new experiment authorized here.
  Explicitly declare feature/compute changes and audited code repair;
  preserve old hashes and strict collector guards. Single seen seed only.

## Previously completed execution work

- [x] Design FBPA M0/M1/M2 around family-balanced then same-source paired
  updates; identical128 episode identities/64 updates/fixed per-episode bank.
  Integer-only schedule consistency verified. See FAMILY_BALANCED_PAIRED.md.
  This initial design stage did not start implementation or experiments.
- [x] Subsequent user authorization: implement FBPA method/configs and verify
  173 local focused tests (4 skipped); full deck provenance, one averaged
  update, off L3C parity, frozen backbone and no dense persistent outputs.
- [x] Validate/sync same code in existing dsba3 container:173 tests pass there;
  host52 remote-workflow/security tests also pass. Launch L3C -> M0 -> M1 ->
  M2 at22:47:54/55 KST, localGPU0/dsba3GPU2 only;32 full8 class runs,seed0.
  First real training steps and GPU process placement verified on both hosts.
  Snapshot7ed14b60b67e9177ac01a8fc8b71f4e7ae9ff0fc8c07ac2770d92aba79676259.
- [x] Honor user stop at01:17:41 KST on2026-09-13: terminate only active local
  M2/sheet_metal, preserve22 local completions and interrupted artifacts;
  dsba3 already8/8 complete. SIGTERM is user interruption, not method failure.
- [x] Subsequent user resume at23:15:18 KST: start existing stopped local
  container, verify identical package code/config/Python/packages and GPU0
  CUDA allocation, then existing --resume skips22 completed local runs.
  M2/sheet_metal restarts from scratch, then M2/walnuts; remote8 not rerun.
  Runner319785/training319818, generic collector320193; no code/config edits.
- [x] Collect/verify all32 FBPA final runs: local24 finished23:33:47 KST,
  remote8 finished00:10:26 KST on2026-09-13. Exclude preserved interrupted
  sheet_metal attempt. All32 logs/coverage valid; all8 class quartets matched,
  all8 M0/M1/M2 triplets have identical128 actual episode manifests.
  Full8 AUPRO L3C/M0/M1/M2:47.96117/47.35899/47.70336/47.73587.
  M1-M0 +0.34437pp; M2-M1 +0.03251pp but M2-L3C -0.22530pp.
  No new best-primary method. M2 summed fit time -12.41% vsM1, peaks unchanged.
  Canonical Markdown/CSV replace40 interim rows with36 full8 rows, preserve404
  historical rows. Remote compact recovery complete; no full-log/map pull.
  Collector finished. No new experiment scheduled. Audit: local FBPA root/
  collection_audit_20260913.json.
- [x] Implement independent L1/L2/L3 methods plus L2C/L3C controls and K1-off
  parity including forced weak synthesis;149 local targeted tests pass.
  Full-shape L3 CUDA second-order smoke passes (7.459GiB graph peak).
- [x] Sync validated immutable code to existing dsba3 container, repeat149 tests,
  launch K1 -> L1 -> L2C -> L2 -> L3C -> L3 on localGPU0/dsba3GPU2,1worker/GPU.
  Both queues running17:54:05 KST; local K1/can, remote K1/vial verified on GPU.
  Snapshot be30187a3e25dc5232b81977d796f29409cadb3ba49cb80c4afd3ea8709d0afd.
- [x] Collect all48 final class results: local36/36 complete20:58:19 KST,
  dsba3 12/12 complete19:27:16 KST, no recorded failures. All48 valid; all8
  class sextets match common conditions. Replace60 interim rows with54 full8
  rows in canonical Markdown/CSV, preserving350 historical rows. Maps/weights off.
  AUPRO x100 K1/L1/L2C/L2/L3C/L3:47.71643/47.70081/31.93149/32.82129/
  47.97341/47.93244. L1-K1 -0.01562pp; L2-L2C +0.88980pp but L2 belowK1
  on all8 classes; L3-L3C -0.04096pp. L3C small primary gain +0.25697pp
  is seen-seed0 development evidence, not a meta-learning or final claim.
  All512 meta steps active; no next experiment launched during collection.
- [x] Implement K3 actual-Adam parameter displacement projection, preserving
  K1 synthesis/loss and exact off parity; 133 focused CPU tests pass locally.
  Method/config/design: projected_update.py, projected_update_K3.yaml,
  configs/methods/PROJECTED_UPDATE.md. No new experiment or remote sync.
- [x] Launch K1 -> K3 full8 seed0/64-step comparison with two workers/GPU;
  133 tests pass in both containers, remote snapshot f99c089e2035d57fc849caef5887485925ae6c7cd5008ec5a835823ced85fbe5.
- [x] Resume local queue at15:07:05 KST after host reboot14:42:57: same container,
  preserved incomplete attempts/status/logs, existing --resume, GPU0 now450W
  for both controls, no source/power changes. Remote4/4 already complete;
  no remote rerun. Generic local collector restarted.
- [x] Collect K1/K3 local12 and remote4 final results; all16 valid, all8 pairs
  matched. Canonical full8 Markdown/CSV replace20 interim rows with18 rows,
  preserving332 historical rows and excluding interrupted attempts.
  K1/K3 AUPRO47.69026/47.65616 (-0.03410pp); vial -0.27628pp, no primary gain.
  Projection89/512, replay error0; P/b decreased without the intended metric
  improvement. No next experiment launched; single-seed development only.
- [x] Implement K2 student-aligned synthetic selection and config on/off parity;
  unchanged H1 loss/controller/inference; 108 tests passed in both containers.
  Design and 24-run budget: configs/methods/STUDENT_BOUNDARY.md.
- [x] Fix/test generic collector's parallel active-status KeyError before launch.
- [x] Launch/verify H1 -> K1 -> K2 on local GPU0 six classes and dsba3 GPU2
  vial/wallplugs, two workers/GPU, seed0/normal16/full8/64 steps, maps/weights off.
- [x] Collect K2 comparison: all24 outputs valid, all8 triplets matched;
  K2 preview/replay error0 on512 steps. Full8 K2-K1 AUPRO -0.12789pp (no primary
  gain), vial +2.06683pp vsK1. Canonical CSV/Markdown replace21 local6 rows with27
  full8 rows, preserving305 historical rows. No additional experiment launched.
- [x] Implement K1 boundary-conditioned synthetic construction only; no loss,
  inference, parameter-count or shared pipeline change. 86 targeted tests pass
  in both existing containers; design: configs/methods/BOUNDARY_SYNTHETIC.md.
- [x] Launch H1 -> K1, seed0/normal16/full8, local GPU0 six classes and dsba3
  GPU2 vial/wallplugs, two workers/GPU, 16 runs maximum. Both queues running H1.
- [x] Collect H1/K1 (16/16 complete, no failures): verify8 matched pairs and16
  valid results; retrieve remote2 compact results and replace14 local6 rows with18
  full8 canonical rows. Preserve287 historical rows. H1 47.40645 -> K1 47.70115
  AUPRO x100 (+0.29470pp), six gains with vial -1.89786pp. No next run active.
- [x] After H1/K1 completes, fix/test generic collect's parallel active-status
  rendering (KeyError 'object'); fixed before K2, not during an active run.
- [x] Implement J1 hard-pair tail risk as an independent H1 extension, with
  config-off parity and unchanged shared pipeline; 96 tests passed on both
  local and dsba3. Design: configs/methods/TAIL_RISK.md.
- [x] Collect H1 -> J1 (16/16 complete): local GPU0 six classes, dsba3 GPU2
  vial/wallplugs. Eight pairs pass matched-condition checks; canonical full8
  CSV/Markdown updated. H1 47.40352 -> J1 47.53549 AUPRO x100 (+0.13197pp),
  six classes improve but vial -3.03246pp. K1 is tracked separately above.
- [x] 2026-09-12: finish G1/H1 collection and replace partial rows with
  verified full8 metrics in the canonical CSV/Markdown.
- [x] Implement independent I1/I2/I3 H1 extensions and verify config-off
  parity, exact replay, frozen backbone, output policy, and common pipeline.
  85 tests passed on each of local and dsba3.
- [x] Collect H1/I1/I2/I3 (32 class runs) launched 06:22 KST on local GPU 0
  and dsba3 GPU 2 only. All 24 candidate/control pairs pass matched-condition
  checks; 32 final results are valid. Replace local6 interim rows with full8
  rows and analysis in the canonical CSV/Markdown. H1 47.41448, I1 47.30383,
  I2 46.95461, I3 47.37585 AUPRO x100; no candidate improves the primary.
  No next experiment is running or automatically scheduled by this result.

Earlier general milestones below remain separate from the active comparison:

- [ ] Before the baseline run, explicitly declare the primary metric and the
  development-versus-final evaluation allocation. Keep the common data split,
  support policy, preprocessing, seed policy, and metric contract fixed; reserve
  an untouched final evaluation for the chosen comparison.
- [ ] Before any sealed-final claim, verify the actual evaluation path and
  development/final separation; the current runner does not yet configure that
  partition. Preserve failure/exit evidence for interrupted runs until durable
  run-level error recording covers prediction and evaluation failures. See
  [known execution limits](current.md#execution-limits-to-address-before-relying-on-final-evidence).
- [ ] Run an actual GPU baseline check for the retained common pipeline using
  the currently configured and authorized execution target. Record the command,
  environment, seed, support selection, runtime, and quantitative result. Do
  not treat a smoke test, fixture, or setup receipt as a benchmark result.
- [ ] Select at most two initial candidate hypotheses before implementation and
  development comparison. Candidate selection is pending; do not infer it from
  a historical report or manufacture a result.
- [ ] Implement the selected candidate(s) behind the `src/fmad` method
  interface while preserving the shared runner, dataset, evaluation, and result
  persistence path.
- [ ] Compare the initial candidates under the declared development protocol,
  then select continuation based on observed performance and validity evidence.
  Prioritize implementation correctness and same-condition performance
  comparison; a runtime failure does not by itself invalidate a method family.
- [ ] If a candidate shows a repeatable development signal, run the smallest
  justified ablation, generalization, efficiency, and post-hoc analyses. Label
  post-hoc observations as such and do not present them as preregistered gates.
- [ ] Save quantitative results, training logs, and reproduction metadata for
  each completed run. Keep anomaly maps and fitted weights disabled unless a
  specific diagnostic or reproducibility need opts them in.

## Completion checks

- [ ] Compare candidates under identical support-image identities (or an
  explicitly documented fixed-reference manifest), seed, preprocessing, and
  evaluation metrics.
- [ ] Verify no data/ground-truth leakage across method fitting, development
  selection, and final evaluation.
- [ ] Record whether each run is complete, invalid, incomplete, or merely
  diagnostic before interpreting its numbers.
- [ ] Update [`current.md`](current.md) and append a dated entry to
  [`log.md`](log.md) when the active state changes.

## Historical boundary

The dated experiment records remain useful for constraints, failures, and
quantitative evidence. They do not require deleted `tasks/` files, restore
retired runners, or impose a blanket ban on a new method. Any continuation of
an old branch must be recorded as an explicit current-state choice before
execution. This is a reproducibility and scope-recording requirement, not an
extra permission gate; current user and higher-priority instructions still
control.
