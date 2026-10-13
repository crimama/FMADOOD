# Skill Graph Log

## 2026-09-25 — E12 resolution control completed and collected

- Completed all 24 class evaluations (three arms × eight classes), with no
  failures. All arms used the same code snapshot, seed-0 16-shot support
  manifest, and matching per-class/per-stratum query lists; the declared
  evaluator contract and provenance were retained.
- On regular full8, multiscale RSG3 is 61.25986 AUPRO / 77.94223 pAUROC;
  1120-only at matched total bank capacity is 59.19861 / 76.05768. The
  +2.06125pp AUPRO difference exceeds the registered 0.86pp threshold.
  Multiscale also leads both 1120-only arms in shift_1, shift_2, and shift_3.
- Interpretation is limited to this fixed support draw on the public
  development population. Detailed strata and class metrics are in
  `results/v7_evidence/E12_resolution_control/comparison.md`; cumulative
  ledger updated in `results/comparisons.md` and its paired CSV.

## 2026-09-25 — Prior v7 evidence collected; E11 stopped by user request

- Stopped the four active E11 shift-shot lanes. All four remote statuses are
  `invalid_run`; 12 completed cells and partial artifacts are retained, with no
  E11 collection or plot presented as complete. No E11 run process remains.
- Pulled and collected E1_v2 (128/128), I1a Gaussian (30/30), E9 MVTec
  (435/435), and E9 VisA (420/420). All expected metrics, configs, and manifests
  are present locally. The broad pull also reported permission-denied files in
  unrelated historical result roots; this did not prevent transfer of these
  four requested suites.
- E1 C0 passes for new seeds 1–4; all class-level arms have identical support
  hashes. With existing seed-0 cells, R1 passes 5/5 draws (mean G(2)−G(16)
  +3.78pp). Gated AUPRO is 56.07±2.39 at 2 shots and 59.48±0.60 at 16 shots.
- I1a Gaussian lowres_pool is 52.82 AUPRO, below the preregistered 72.009
  threshold; I1b is closed. E9 support hashes pair with stored RSG3 draws for
  every class and shot. RSG3 clears the +0.86pp lead criterion on VisA at
  4/8/16 shots and on no MVTec shot. Detailed rankings are in the suite reports.
- E4 shot, impulse, and speckle runs are complete and collected; the registered
  G/P interpretation remains outstanding.

## 2026-09-22 — External-method corruption comparison complete

- Completed450 new MVTec AD full15 class evaluations for RegAD, RD4AD,
  PatchCore, FastRecon, RN-FMLK and SuperAD under clean plus four severity-3
  corruptions; combined with75 existing RSG3 cells, all525 cells have finite,
  shape-complete coverage. Normal16 seed0, clean support, frozen DINOv2-L,
  common evaluator, no target adaptation, maps or weights.
- Retained the initial PatchCore dependency failure as execution evidence.
  Repeated nine RegAD cells affected by a transient NumPy downgrade; final
  aggregation uses NumPy2.4.6 throughout. Normalized support identities match,
  and corruption-root `train/good` files had been byte-checked against clean.
- Mean shifted AUPRO@0.05 is SuperAD79.8073, PatchCore79.0452, RSG378.5434.
  RSG3 leads brightness84.4245 and contrast83.1513 and leads the three mild
  shifts at83.1665, but Gaussian noise falls to64.6741 (-20.6873pp).
- Added the detailed report at
  `results/comparisons/corruption_compare_20260922/comparison.md` and replaced
  PaperWorks Section8.1's internal-variant proxy with direct external-method
  AUPRO and pAUROC tables. The conclusion is bounded to competitive absolute
  performance and mild-shift stability, not universal robustness.

## 2026-09-14 — CRR CR0–CR3 complete, verified and collected

- All32 runs complete, no failures; final dsba3 lane1 ended13:46:20 KST.
  Local8 + remote24; no new training/evaluation, restart, source/config edit,
  container change, cleanup or subsequent experiment scheduled.
- Recovered92 compact remote artifacts through the existing container after
  host rsync permission errors; all SHA256s match. Full logs remain remote.
  Audited32 metric/summary/manifest identities, finite full query coverage,
  matched class conditions and CR0 exact historical R1/TF0 parity.
- Expected0/64 optimizer steps verified; each trained run's64 descriptor-cache
  progress events are separate. Episode/descriptor/positive-sample hashes
  match CR1/2/3. Frozen backbone,0/129params,32/544fit encoder calls.
- Full8 AUPRO CR0/1/2/3:52.47033/52.98213/53.13319/52.29146;
  pAUROC:75.99215/75.63799/75.52551/75.91410.
  CR2 +0.66286pp primary but -0.46664pp pAUROC vs CR0; CR3 below CR0 on both.
  Joint target not achieved. CR2 gains concentrate in wallplugs; no causal
  diagnosis or final-evaluation claim from this single seen seed.
- Generic collect processed5 lanes. Canonical CSV/Markdown now contain36 new
  CRR full8 records (32 class+4 macro),548 historical rows preserved,584 total.
  Suite comparison.md/collection_audit.json/training_audit.json retain details.
  Current state, TODO and method overview updated; R1 remains reference.

## 2026-09-14 — User corrected dsba2 to dsba3; four remote GPUs launched

- The user explicitly selected dsba3 physical GPUs0/1/2/3, resolving the
  connection-name blocker. Reused hun_fsad_tta_012 and existing data/model cache;
  all four RTX A6000 devices idle and CUDA arithmetic passed before launch.
- Activated local snapshot through run_remote.sh update-code:
  5b6ea70b4fb2d9d1c45b2f754beb4ed5a4b1ed0405b9a12b5eed584cb3a02e12.
  Package SHA bb8f51c90acb1da0db72647da87fb03dc6be4ea33cbab253802e820e20854275
  and all four YAML hashes match local; Python/packages match. Kernel/glibc
  differences are declared in current.md; no identical-host performance claim.
- Remote152 tests pass,5 skipped; separate real cached DINOv2 full16-bank
  smoke passes CR0--CR3 at448/672, two updates for learned heads, exact CR0/R1
  parity. Validation logs remain remote at /tmp/fmad-execution-logs/run.nMoI5DCO.
- Launched24 remote class runs12:44:19 KST through generic Bash remote-only,
  parent/session1496638. Lane0 fruit_jelly/wallplugs, lane1 rice/walnuts,
  lane2 sheet_metal, lane3 vial; each CR0->CR3,one worker per verified GPU UUID.
  All four actual GPU processes and CR0 fit_started/running statuses confirmed,
  failures empty. Remote helper log /tmp/fmad-execution-logs/run.qqnLG6do.
- Original local lane untouched:5/8 complete, CR2/fabric active12:45:09 KST.
  No new code/config changes to running methods, git push, container/package
  replacement, dataset download or dense-output enabling. No suite collection
  yet; retain remote logs and recover only compact result/provenance artifacts.

## 2026-09-14 — CRR implemented and local lane launched; dsba2 unresolved

- User authorized implementation/review/setup/launch on local plus dsba2 GPUs
  0/1/2/3. Added one registered method and four config-only CR0--CR3 ablations;
  reuse generic CLI/Bash, no method-specific queue or ledger script.
-152 focused checks passed,5 skipped, plus separate actual cached DINOv2 CUDA
  smoke (full16 banks,448/672,all stages,two trained updates). CR0 exact R1;
  matching episode/descriptor hashes, source exclusion and frozen backbone.
  Bash syntax passes; Ruff unavailable. Locked CROSS_IMAGE_REFERENCE.md.
- Launched local can/fabric CR0->CR3,8 class runs,12:28:55 KST, parent/session
  1470978. Initial CR0/can process1471079 runs on local GPU0 UUID
  GPU-0070b668-714f-bcbb-6c46-f0b68c9b7f6b. Package SHA
  bb8f51c90acb1da0db72647da87fb03dc6be4ea33cbab253802e820e20854275.
- dsba2 preset and name resolution absent; asked user for connection details
  or corrected server name. No remote actions or substitution of dsba3.
  Remaining24 remote runs pending verified connection/environment/GPU mapping.
  No suite results collected yet, no ledger update, no container replacement
  or data copy. CR0/can subsequently reached its normal summary output.

## 2026-09-14 — CRR next-method design only

- Designed CR0--CR3: exact R1, bounded residual local-ranking control,
  cross-image normal-tail ranking, then source-separated reference descriptors.
  All trained CR stages omit the former shift-pair loss. Only CR2-CR1 and
  CR3-CR2 isolate the proposed additions; CR1-TF3 changes several conditions.
- Record fixed normal16/seed0/backbone/reference pipeline, future32-run full8
  comparison, no-query-fit boundaries and compact-descriptor memory budget in
  analysis/2026-09-14_cross_image_reference_design.md. Primary/secondary joint
  gain is a target, not an expected numeric result or a novelty claim.
- Documentation only: no source/YAML change, training/evaluation, remote access
  or cumulative-results update. No experiment scheduled by this design.

## 2026-09-14 — Compact comparison index and bounded collection reports

- Reorganized comparisons.md from1680 lines into a compact latest summary,
  one CSV-macro table and scope/detail links. All548 CSV rows remain unchanged.
  Preserved the complete old narrative in
  results/comparisons/details/history_through_20260914.md; only the archive
  notice and relative Markdown link locations changed, not result content.
- Generic experiments.py collect now replaces one latest-summary block and
  renders each macro once. Per-run class tables/audits are saved beside the
  run in comparison.md; repeated historical tables are no longer appended.
  Preserve comparison guards, unknown-vs-zero, scopes, seeds and user notes.
- Routine lookup starts at the summary; older detail is searched on demand.
  No method/config change, experiment, evaluation, remote sync or data cleanup.

## 2026-09-14 — TBF TF0--TF3 full8 final collection

- All32 runs complete without recorded failures: local8 ended10:29:39 KST;
  remote24, last completion10:58:09 KST. Four remote GPUs idle at11:02:49.
  No experiment active or scheduled by this collection; containers retained.
- Full8 AUPRO x100 TF0/1/2/3:52.47033/52.70446/52.61031/52.76262;
  pAUROC:75.99215/75.79109/75.79416/75.51276;
  oracle F1:36.65951/36.68387/36.62025/35.96161.
  TF3-TF0 AUPRO +0.29229pp but pAUROC -0.47940pp: no targeted joint win.
  TF2-TF1 -0.09415pp/+0.00307pp; TF3-TF2 +0.15231pp/-0.28141pp
  (AUPRO/pAUROC). Normal-tail selection shows no useful aggregate benefit.
  TF3 AUPRO gain concentrates in wallplugs +6.76877pp; sheet_metal -4.31818pp.
  Posthoc other-seven macro -0.63292pp. Single seen seed, no significance claim.
- All32 metrics match summaries/manifests with complete finite/nonconstant,
  shape-correct coverage. Original logs have expected0/64 sequential steps,
  finite values and fit_finished. All8 quartets match code/common conditions,
  support/query/GT, backbone/environment and MAD scales. Every TF1/2/3 triplet
  has identical64 actual episode records, not just identical seed settings.
- TF0 reproduces all8 previous R1 metrics/coverage and scales exactly. Six
  remote old/new absolute support-path prefixes differ; content/identity/GT
  hashes and ordered relative paths match. Record aliases, preserve manifests.
- Generic pull encountered host read permissions after16 summaries; read-only
  transfer from the existing container recovered compact missing files. All92
  remote compact files are SHA256-identical. No permission change or cleanup;
  original full logs remain remote, bounded audit summaries retained locally.
- Generic collect processed5 roots. Canonical Markdown/CSV replace52 interim
  lane rows/5 blocks with36 full8 rows/one block, preserving512 historical
  rows (548 total). Macro means weight classes, not unequal GPU lane averages.
- Evidence: results/local_runs/bcma_docker_20260911/tail_aware_fusion_20260914/
  collection_audit_20260914.json and collection_remote_audit_20260914.json;
  compact remote results/remote_runs/tail_aware_fusion_20260914_dsba3.
  No source/config edit, training/evaluation, container restart/stop, dense
  map/checkpoint/full-log pull or new method-specific helper during collection.

## 2026-09-14 — TBF TF0--TF3 implementation, validation and launch

- User authorized the designed fusion ablation on local and dsba3 GPUs.
  Implemented only tail_aware_fusion.py, lazy registration, four YAML configs,
  focused tests and TAIL_AWARE_FUSION.md; existing runner/CLI/metrics unchanged.
- TF0 exact R1; TF1 learns a global scalar; TF2 changes normal sampling to
  current top5%; TF3 changes the gate to a shared2->16->1 MLP (65 parameters).
  Fixed rho.5,64 steps,lr.01,margin.5,pair weight.1. R1 backbone/banks/MAD fixed.
  All64 episode inputs/masks/native coordinates are reproducibly keyed;
  source-excluded fitting, no external donor/query GT, no query updates.
- Sampling implementation explicitly uses at most4096 native normal pixels
  per view and128 positive pixels; this is a bounded-memory tail surrogate,
  not exact dataset-wide pAUROC optimization. Native grid sampling matches
  bilinear interpolation. Only sampled distances survive within an episode.
- Both existing containers passed135 focused tests (4 skipped) and a separate
  real cached DINOv2 CUDA smoke at448/672:TF0 parity, TF1--3 two finite updates,
  finite/nonconstant predictions, matched scales/episodes. No smoke benchmark
  metrics. Backbone state remains frozen, gate state unchanged at prediction.
- Code SHA668d54ddd9cb33b975e67c1846bc5ea70fe811f389c1dee7714f10885f0b1654;
  remote local-only immutable snapshot1f36c150db82307f11045dc72856e419d756ce13d08c7ec0db7621836a710ef8.
  Torch2.6.0+cu124/CUDA12.4 identical. No git push/container replacement/download.
- Started32 class runs10:08:41 KST via one nohup+setsid generic Bash invocation,
  parent1249970. All five lanes running TF0 on correct devices at initial check:
  local0 can/fabric; dsba3 GPU0 fruit_jelly/wallplugs, GPU1 rice/walnuts,
  GPU2 sheet_metal, GPU3 vial. Source/configs frozen after launch.
- Roots: local /results/tail_aware_fusion_20260914/local/lane_0; remote
  /results/tail_aware_fusion_20260914/remote/lane_0..3. Host launch logs under
  results/experiment_launch/tail_aware_fusion_20260914. Original remote logs
  stay remote; later pull compact metrics/configs/manifests only. No automatic
  collector yet, and previous512 cumulative CSV rows remain unchanged.
- Primary AUPRO with targeted pAUROC recovery; compare to fresh TF0 on seed0
  normal16 full8 seen TESTpublic. No new performance claim at launch.
- Follow-up10:11 KST: local TF0 can/fabric saved metrics/coverage and complete
  support/query/GT/MAD metadata exactly match prior R1. Remote GPU3 TF1/vial
  step13/64 has finite loss/gradient and a gate weight changed from initialization.
  All five queues remain running without failures; no source edits after launch.

## 2026-09-14 — R/P full8 final collection

- All64 final class runs complete: local16 ended03:12:08 KST; remote48,
  last completion05:17:32 KST. Exclude5 original R1 CPU/CUDA failures and
  preserve their artifacts/history. No post-repair final-run failure.
- Full8 AUPRO x100 D3/L3C/R0/R1/R2/P0/P1/P2:
  48.60330/47.96204/46.66157/52.47033/42.92744/14.73310/16.01451/14.24413.
  R1-L3C +4.50830pp but pAUROC -1.16388pp. R1-R0 +5.80876pp,7/8 gains;
  R2-R1 -9.54289pp,7/8 declines. P1-P0 +1.28141pp; P2-P1 -1.77038pp.
  R1 is this batch's strongest primary method, not an all-metric/global-protocol
  best. Prefer R1 for subsequent design; no new experiment launched here.
- All64 final results match summary.tsv/manifests, with complete finite,
  nonconstant and shape-correct query predictions. All64 training logs have
  fit_finished, expected0/64/128 sequential steps and finite training metrics.
  P2/can AUPRO0 is a true zero, not missing or invalid prediction coverage.
- All8 class octets match support/query/GT protocol, within-host environment,
  backbone weights and metric contract. All8 P triplets share128 actual episode
  hash sequences and1568 fit calls; P1 uncertainty only, P2 synthetic only.
  All8 R1/R2 pairs share exact support MAD scales. Feature/compute changes
  remain declared; layer versus normalization and R2 subcomponents not isolated.
- Cross-repair code hashes preserved. Retained snapshot diff shows only capped
  query device transfer in R1/R2 calibration; no config changes. Generic
  comparison guards retained; full8 report explicitly audits method-level
  cross-code/feature contrasts instead of relabeling them strict matched runs.
- Generic pull recovered32 summaries but hit host permission errors. Read-only
  compact transfer from the same existing container recovered missing files;
  all184 remote compact artifacts verified byte-identical. Remote48 training
  logs summarized remotely, not copied. No permission change or cleanup.
- Generic collect processed all5 roots; replace104 per-lane rows/5 blocks with
  72 full8 rows/one canonical block. Preserve all440 historical CSV rows;
  total512. Full8 means weight classes, not five unequal lane means.
- Evidence: results/local_runs/bcma_docker_20260911/reference_prediction_20260914/
  collection_audit_20260914.json and collection_log_audit_20260914.json.
  Compact remote:results/remote_runs/reference_prediction_20260914_dsba3.
  No source/config edit, training/evaluation, container restart/stop, dense
  map/checkpoint/full-log pull or new helper during collection.

## 2026-09-14 — R1/R2 CUDA repair and five-lane resume

- Local status check found invalid_run at01:05:11 KST, R1/can; all four remote
  lanes also failed on their first R1 class. CPU-staged capped support records
  were passed to GPU bank matrix multiplication during _fit_scales. No GPU
  hardware failure or OOM was reported. Containers remained running/idle.
- Added actual CUDA R1/R2 fit/predict and shared-scale regression cases;
  all3 failed with the same device mismatch before repair. Fix moves only the
  current capped support query to the reference bank device. No objective,
  feature, sampling or config change. P0/P1/P2 fit/predict also now runs on CUDA
  in the test suite. Both actual containers passed97 focused tests.
- Real cached DINOv2 ViT-L14 R1/R2 fit/predict at448+672 with3 normal can
  supports passed finite/nonconstant checks and exact R1/R2 scale equality.
  This was a runtime smoke, not a new benchmark metric or method comparison.
- Code snapshot relocation changes resolved config paths on dsba3. Updated
  generic experiments.py resume to allow this only for byte-identical existing
  old/new config files with unchanged labels, objects and split. Retain both
  plans and config hashes in status.config_relocations. Changed/missing/reordered
  file tests reject without starting runs or changing status. No new helper.
- Remote audit found only methods/reference_matching.py changed within src/fmad;
  all method YAML configs remained byte-identical. Package code SHA changed
  from f22b0607728b985215e12022c3b5f1b6d1c4b6f9445f73adbe3c1d071b63b212 to
  27f1bbf62c96b44d471829ba91f9eeb95c77181a63e067a39054948d6253e0ce.
  Final activated snapshot:c2c129a5bcaee17dcb7b1882f0c9555928acd7d9f154b80b85e0c5ec3cc7e481.
  Do not rewrite old result hashes or remove strict collector comparability
  checks: later cross-repair comparisons require this explicit source audit.
- At02:11:18 KST launched existing Bash invocation plus --resume under
  nohup/setsid, parentPID583054. Preserved local6/remote18 D3/L3C/R0 completions
  and skipped all24, leaving40 planned class runs. Same5 GPUs/class placement,
  configs, seed, support policy,14400s per-class safety ceiling and output roots.
  At02:11:36 all GPUs active; all lane statuses running with their original
  failure preserved. Remote statuses each record one config relocation.
- Active host log:results/experiment_launch/reference_prediction_20260914/launcher.resume1.log.
  Remote execution log:/tmp/fmad-execution-logs/run.RcjRgYWa. Original R1 logs
  remain; resumed console logs use .attempt2.log and distinct run identities.
  No container creation, download, dense output, deletion or metric collection.
- At02:13:16 KST local R1/can completed and R1/fabric was active. All four
  remote initial R1 classes passed fit_finished; fruit_jelly/vial completed,
  proceeding to wallplugs/R2-vial. All five lane states remain running.

## 2026-09-14 — independent R/P implementation, verification and five-GPU launch

- User requested complete implementation, local plus all four dsba3 GPUs, and
  one Bash invocation that runs configs sequentially. Implemented reference
  matching R0/R1/R2 and blind-region normal prediction P0/P1/P2 with configs,
  existing method registry/CLI integration and regression tests. Bounded
  implementation and independent review used installed native agent roles.
- R0 parity, R1/R2 bank identity and fixed support-scale equality are tested.
  P masks prepared pixels before encoding with a one-token guard; same-shift
  normal targets, visible-only style/context, conditional variance and detached-
  variance synthetic margin are tested. Synthetic resize spillover is clipped
  to the scored tile. No label/dataset access or persistent dense-map outputs.
- Generic scripts/run_experiments.sh launches disjoint per-GPU lanes through
  experiments.py run and run_remote.sh. Remote lanes share one execution lock.
  Corrected nested runtime PATH, lane directory ownership and mixed-backend
  argument validation before launch. No method-specific queue/ledger helper.
- Both actual containers passed88 focused tests. Cached DINOv2 multilayer
  pixel-masking and full-width CUDA predictor forward/backward checks passed.
  Torch2.6.0+cu124/CUDA12.4 and package-source hash match. Reused existing
  containers; no provisioning, GPU exposure change, download, git push or cleanup.
- Code SHA256:f22b0607728b985215e12022c3b5f1b6d1c4b6f9445f73adbe3c1d071b63b212;
  remote snapshot:fd24389e551b54e2d606eb9f0a22afe4b043bf10dca50994ca7a7a8ee2c839fe.
- Initial00:43:55 KST nohup-only parent exited before any class run. Verified
  absent lane outputs/idle GPUs before retry. Preserved initial launcher log;
  nohup/setsid retry started00:45 KST, parentPID457405 remains active.
- Planned64 class runs: D3/L3C/R0/R1/R2/P0/P1/P2 on all8 classes, seed0 normal16,
  seen TESTpublic development. LocalGPU0 can/fabric; dsba3GPU0 fruit_jelly/
  wallplugs, GPU1 rice/walnuts, GPU2 sheet_metal, GPU3 vial. Each class stays on
  one device across methods. Five lanes parallel, one worker per lane.
- At00:49 KST all5 GPUs active. Local D3/can completed and D3/fabric had finite
  training loss/gradient; remote fruit_jelly/rice/vial fit_finished and sheet_metal
  had finite updates. This verifies execution, not final method performance.
- Locked design, caveats and exact invocation:configs/methods/REFERENCE_PREDICTION.md.
  Host launch root:results/experiment_launch/reference_prediction_20260914;
  active launcher.attempt2.log, local/lane_0.launch.log. Local results under
  results/local_runs/bcma_docker_20260911/reference_prediction_20260914/local/lane_0;
  remote originals hun_fsad_tta_012:/results/reference_prediction_20260914/remote.
  Full8 collection remains pending; no new metrics entered in the ledger here.
  No source/config/script edits after successful launch; maps/checkpoints off.

## 2026-09-13 — FBPA full8 final collection

- Local24/24 finished23:33:47 KST; remote8/8 finished00:10:26 KST. The user-
  interrupted sheet_metal attempt remains on disk and is excluded; resumed
  final run completed64 updates. No runtime failure among32 final runs.
- Full8 x100 L3C/M0/M1/M2 AUPRO:47.96117/47.35899/47.70336/47.73587;
  pAUROC:77.14863/76.69564/76.99095/76.93707;
  oracle F1:36.41526/36.01830/36.51869/36.40540.
- M1-M0 AUPRO +0.34437pp,4/8 gains, chiefly vial +3.06554pp and wallplugs
  +1.00584pp; local6 -0.21941pp. Family-balance evidence is scoped to full8
  seen seed0, not uniform gains or a proven gradient-conflict mechanism.
- M2-M1 +0.03251pp with lower pAUROC/F1 (-0.05388/-0.11330pp). M2 remains
  below same-batch L3C by0.22530pp; no new best-primary candidate. Historical
  L3C47.97341 is kept separate from current47.96117.
- All32 original logs:64 steps,fit_finished,finite losses/gradient norms.
  All final predictions:full finite/nonconstant coverage and correct shapes.
  All8 quartets match full support/query/GT protocol/common config/backbone/
  metric/code/within-host environment. All8 M0/M1/M2 triplets have identical
  128 episode manifests. Only family_balance then source_pairing switches differ.
- Trainable parameters100384 for all; frozen backbone. M0/M1/M2 fit extraction
  calls1856/1856/1600, main training views1024/1024/768. Summed final fit time
  L3C/M0/M1/M2:3241.02/3213.33/3215.48/2816.34s; M2-M1 -12.41%, excluding
  interrupted work. Not wall-time/inference/equal-hardware benchmark. Fit CUDA
  peaks essentially unchanged. Maps/checkpoints off; no dense artifact pull.
- Existing generic pull recovered4 summaries but host permissions blocked25
  compact files. Read-only docker cp from the existing stopped container
  recovered status/config/metrics/manifests with byte equality verified.
  Eight original remote logs summarized remotely, not copied locally.
  Compact transfer digest66f8f28eca4a901b78f7cba97ea2558240d06c395d5b48a515b2598a17bba1b3.
- Generic collect ran on both roots. Canonical comparisons.md and paired CSV
  replace40 interim rows with36 full8 rows, preserving404 historical rows.
  Audit: results/local_runs/bcma_docker_20260911/family_balanced_paired_20260912_local/collection_audit_20260913.json.
  Compact remote: results/remote_runs/family_balanced_paired_20260912_dsba3_gpu2.
  No code/config edit, evaluation, restart, cleanup, helper script or new run.

## 2026-09-13 — FBPA user interruption and same-condition resume

- At01:17:41 KST, honored the user's temporary-stop request: sent SIGTERM to
  verified local M2/sheet_metal PID896787. The runner exited to invalid_run,
  preserving22 local completed identities and unfinished artifacts. GPU memory
  returned to41MiB/0% utilization; container remained running at that check.
  dsba3 had already completed8/8 at00:10:26 KST. No files were deleted.
- The user later requested continuation. At23:13 KST both existing containers
  were found stopped. Started only fsad-tta-local-bcma; did not recreate it.
  Read dsba3 stored status via docker cp stdout without starting its container:
  complete8/8, no failures. No remote execution or source synchronization.
- Verified current package-source SHA256 against prior manifests for all four
  stages, current config values against saved effective configs, and identical
  Python/package versions. Same local GPU0 UUID and450W; real CUDA allocation
  succeeded and both pending class data directories are accessible.
  Code SHA: ce423a34f66f373427bc44539335af2a5a1206f0f79db1515c4b20bd98644910.
- At23:15:18 KST launched existing scripts/experiments.py run --resume with
  the original full plan/seed0/64 steps/one worker/3600s timeout. It skips22
  completed local identities and runs only M2/sheet_metal then M2/walnuts.
  The earlier SIGTERM is retained as a user-interruption history entry, not
  evidence against the method. No checkpoint: sheet_metal refits from step0.
  Original unfinished run and log remain; retry uses a new run identity and
  M2_sheet_metal.attempt2.log. No completed-run rerun or result overwrite.
- Detached local runner319785/training319818 verified on
  GPU-0070b668-714f-bcbb-6c46-f0b68c9b7f6b. Resume launch log:
  results/local_runs/bcma_docker_20260911/family_balanced_paired_20260912_local.resume_20260913.launch.log.
  Original output root/CLI/configs retained. No source/config/hardware changes;
  maps/checkpoints remain disabled. Source/configs frozen while queue runs.
- Restarted the existing generic collector PID320193, log:
  results/family_balanced_paired_20260912_collect.resume_20260913.log.
  Paired canonical Markdown/CSV returns to running22/24 status. Remote compact
  pull, final full8 merge and method verdict remain pending; no new claim.

## 2026-09-12 — FBPA implemented, validated and launched

- User authorized experiments after the design-only stage. Added the method
  family_balanced_paired.py, registry entry, four L3C/M0/M1/M2 configs and
  focused tests. Common CLI, data, support, backbone and evaluation unchanged.
- Off delegates exactly to L3C, including weak-synthetic behavior. Enabled
  M0/M1/M2 share128 keyed episode/input/mask/teacher/reference identities,
  fixed14-source banks and64 optimizer updates. Two same-theta half-loss
  backwards precede one clip/Adam step. M1 balances families; M2 shares the
  normal source/shift within each pair. No new loss or novelty claim.
- Per-step clean feature reuse for M2 is bounded and discarded after the
  update; only scalars and128 compact hash/provenance entries persist.
  Maps/checkpoints remain disabled. Metadata includes view extraction counts
  and fit CUDA peak allocation. No extra model evaluation pilot was added.
- Both existing containers passed173 focused tests each,4 skipped (one opt-in
  full-shape CUDA test and three unavailable-rsync checks). Host52 workflow/
  security tests passed, including real-rsync checks. Real CUDA allocations
  on both assigned GPUs succeeded. All8 prior normal support sets contain16
  accessible distinct file contents. Target output roots were absent.
- Activated the validated immutable local snapshot on dsba3:
  7ed14b60b67e9177ac01a8fc8b71f4e7ae9ff0fc8c07ac2770d92aba79676259.
  Used existing update-code local-snapshot mode, no Git push or provisioning.
- Started L3C -> M0 -> M1 -> M2 at22:47:54/55 KST via existing
  scripts/experiments.py run and run.py/CLI. Local24 runs (six classes) in
  fsad-tta-local-bcma on GPU0/450W; remote8 (vial/wallplugs) in
  hun_fsad_tta_012 on dsba3 physicalGPU2. Seed0,normal16,448+672,64 updates,
  seen TESTpublic development; one worker/GPU,3600s/class,32 runs maximum.
- At22:48 KST: local L3C/can steps0..4, remote L3C/vial step0 recorded;
  losses/gradient values finite and CUDA process UUIDs match assigned GPUs.
  Both statuses running,0 completed,no failures. New component benchmark
  execution/performance remains pending; startup is not a performance result.
- Local runner687716 / training687758; remote runner2918264 /
  training2918269; local remote-launcher687724. Remote helper logs:
  /tmp/fmad-execution-logs/run.PIVQXPWi. Both launches detached.
- Local root: results/local_runs/bcma_docker_20260911/family_balanced_paired_20260912_local.
  Remote root: hun_fsad_tta_012:/results/family_balanced_paired_20260912_dsba3_gpu2.
  Launch logs: local root plus.launch.log, and
  results/remote_runs/family_balanced_paired_20260912_dsba3_gpu2.launch.log.
- Started the existing generic local collector688104; log
  results/family_balanced_paired_20260912_collect.log. It updates the paired
  canonical comparison Markdown/CSV without inventing incomplete metrics.
  Remote compact pull and full8 merge are pending. Source/configs frozen
  until both queues stop; no new methods, seeds or sweeps scheduled.

## 2026-09-12 — FBPA method design only, following L3C analysis

- Narrowed the target to update-level synthetic-family imbalance. K1 already
  balances families globally but uses16-step homogeneous blocks; L3C mixes
  both per update while also changing exposure and reference-bank exclusion.
  Historical gains therefore do not isolate the mechanism.
- Designed one incremental method: M0 exposure-matched blocked control,
  M1 balanced two-source updates, M2 same-source/shared-shift family pairing.
  Fixed128 episode identities,64 updates, per-episode RNG and14-source reference
  population avoid exposure/input/bank confounding within the three variants.
- The fixed-theta population objective is unchanged; claimed mechanism is
  packing and finite-step trajectory, not a new loss or invariance guarantee.
  Same-source pairing can increase source variance, and synthetic shortcuts
  may persist. Both are explicit failure cases rather than assumed gains.
- Integer-only schedule audit passed identical episode multiset, counts and
  exclusion mapping. No dataset/model/evaluation invocation. Proposed future
  L3C off anchor -> M0 -> M1 -> M2 compares final full8 metrics, not a pilot
  ladder. No implementation, executable config, remote sync or experiment.
- Saved configs/methods/FAMILY_BALANCED_PAIRED.md and live status/checklist.
  Existing results/metrics and canonical cumulative tables are unchanged.

## 2026-09-12 — L1/L2/L3 full8 collection and matched-control verdicts

- Local36/36 finished20:58:19 KST; dsba3 12/12 finished19:27:16 KST.
  Both queues started17:54:05 KST, no recorded failures. All48 final results
  valid: finite/nonconstant full coverage,64 completed steps,normal16,frozen
  backbone,maps/checkpoint off. Original logs confirm all steps and fit_finished.
- Full8 x100 K1/L1/L2C/L2/L3C/L3 AUPRO:
  47.71643/47.70081/31.93149/32.82129/47.97341/47.93244.
  pAUROC:76.74333/77.16674/70.16059/70.45137/77.16807/77.13435.
  Oracle F1:36.22386/36.88045/29.32723/29.61005/36.43473/36.43894.
- L1-K1 AUPRO -0.01562pp; secondary pAUROC +0.42341pp. Local6 +0.45701pp
  does not survive full8, particularly vial -2.61600pp. Four class gains.
- L2-L2C +0.88980pp, primarily vial/wallplugs; L2 remains -14.89514pp vsK1
  and worse on all8 classes. A target-choice effect is not restorer validation.
- L3C-K1 +0.25697pp is highest-primary in this batch; L3-K1 +0.21601pp,
  but L3-L3C -0.04096pp. Meta-learning has no added primary benefit here.
  L3C remains a simpler development candidate only; no significance/final claim.
- All8 sextets match support/query/GT protocol, common effective config and
  backbone. Same code SHA256 across48:
  91d308922d66b3ceacc0c09200c3c09e7323beedc967b2ad4b7d9c0dd4292312.
  Same Python/packages, host kernel/glibc differ; same GPU per class across
  controls (localGPU0 450W, remoteGPU2), mixed hardware not equal compute.
  L2C/L2 train32768 parameters vs100384 others; L3C/L3 use8 vs4 training views.
- Explicit pair switches verified. Read-only original-log aggregation confirms
  matched recorded source/family/synthetic-selection sequences across8 pairs
  of K1/L1,L2C/L2,L3C/L3. All512 L3 virtual steps nonzero, L3C all zero.
  Sum class fit seconds K1/L1/L2C/L2/L3C/L3:
  1658.70/1671.65/1609.86/1611.87/3234.90/3337.96. Not queue wall time.
- Generic pull recovered summaries, while host permissions blocked compact
  files. Existing-container read-only tar transfer recovered37 status/config/
  metrics/manifests; all bytes verified. Aggregate digest:
  e29204b28398b237996bbd601aa8a8c1cfdbfb8fe6a1cf93fa2737cc2124ffad.
  Full remote training logs stayed remote. No permission or container change.
- Generic collect ran for local and compact remote roots; canonical Markdown/
  CSV replace60 interim local6/remote2 rows with54 full8 rows and preserve350
  historical rows. No duplicate cumulative table, new helper, source/config
  edit, new experiment/evaluation, restart, GPU change or artifact deletion.
  Local root: results/local_runs/bcma_docker_20260911/operating_restore_transfer_20260912_local.
  Compact remote: results/remote_runs/operating_restore_transfer_20260912_dsba3_gpu2.

## 2026-09-12 — L1/L2/L3 implementation and local validation

- User requested all three proposed methods in a sequential experiment queue.
  Added operating-region objective, same-shift restoration, and exact
  cross-environment meta-update with canonical/pooled-ERM controls.
- Existing CLI dispatch and method-agnostic experiments.py are retained; no
  method-specific launcher/ledger scripts. Six configs including K1 control,
  same full8/seed0/normal16/64 updates/448+672/seen-development evaluator.
- Corrected a parent/child switch collision during review; forced weak-selector
  tests now prove disabled variants retain K1 rather than silently reverting H1.
-149 local focused tests pass; exact meta-gradient/HVP tests included.
  Full-shape random-feature CUDA execution test passes at7.459GiB peak graph
  allocation, without dataset evaluation or backbone loading. New metrics N/A.
- L3 requires one worker/GPU. Local GPU0 remains450W; dsba3GPU2 is free and
  existing container idle. No GPU setting, provisioning, or Git push performed.
  Remote149 tests also passed after activating immutable snapshot
  be30187a3e25dc5232b81977d796f29409cadb3ba49cb80c4afd3ea8709d0afd.
- Both queues launched17:54:05 KST via existing experiments.py/CLI. Local
  K1/can PID264875 and dsba3 K1/vial PID2868761 verified on assigned UUIDs;
  running/no failures, new completed results not yet available at launch.
- Local root: results/local_runs/bcma_docker_20260911/operating_restore_transfer_20260912_local.
  Remote root: hun_fsad_tta_012:/results/operating_restore_transfer_20260912_dsba3_gpu2.
  Remote host launcher264885 has its own session; local queue is container-detached.
  One generic local ledger collector was started; remote logs remain remote,
  compact artifacts will be recovered at result collection. No source edits
  after launch, no maps/weights or additional experiment scripts.

## 2026-09-12 — K1/K3 full8 collection, no primary improvement

- Verified local12/12 complete15:45:23 KST after authorized recovery and
  dsba3 4/4 complete14:59:07. Two archived partial pre-reboot attempts remain
  preserved and excluded; final outputs are valid. No new experiment/restart.
- All16 pass finite/coverage/64-step/normal16/100384-parameter/frozen-backbone/
  maps-and-checkpoint-off checks. All8 pairs match full support/query/GT
  protocol, common effective config, code/backbone and recorded environment.
  Local pairs450W, remote pairsGPU2; no cross-device pair or historical500W
  compute-match claim. Full16 code hash:
  a10803343ea5211db5fbfcbe868f3dbaaef7087a9a39dfc9786fb600cc028e42.
- Full8 x100 K1/K3: AUPRO47.69026/47.65616, pAUROC76.74015/76.70431,
  oracle F1 36.21609/36.11109. Deltas -0.03410/-0.03583/-0.10500pp.
  Five AUPRO class gains, three declines; vial -0.27628pp. No primary benefit
  or target-class recovery; no K3 promotion or statistical/generalization claim.
- Original training logs aggregated read-only in place:512 steps per method,
  K3 projection89 (17.38%), replay error0 and max linear excess1.0722e-11.
  All8 synthetic/source/family selection sequences match, weak345 each.
  Mean P/b0.82623 ->0.72348; vial1.47532 ->1.01179 despite worse AUPRO.
  Surrogate preservation improvement is insufficient here; root cause remains
  a hypothesis. Sum class fit seconds2302.36 ->2339.86 (+1.63%), not wall time.
- Generic pull recovered summaries; remote host permission errors on compact
  files were bypassed by read-only container tar transfer of exactly13 status/
  config/metrics/manifest files. All13 hashes match byte-for-byte. No permission
  changes, full log/map/weight pull or source update. Generic collect ran for
  both roots, then canonical full8 CSV/Markdown replaced20 interim rows with18,
  preserving332 historical rows. No duplicate cumulative table or new script.

## 2026-09-12 — K1/K3 launch, host reboot and authorized local resume

- Original launch14:41:17/18 KST: K1 -> K3, local GPU0 six classes, dsba3 GPU2
  vial/wallplugs, two workers/GPU, seed0/normal16/448+672/64 steps/seen TESTpublic.
  Full8 AUPRO@.05 primary, maps/weights off, 16 planned class results; no sweep.
- Both containers passed133 focused tests and CUDA allocation checks. Synced
  local source through existing update-code local-snapshot mode; remote release
  f99c089e2035d57fc849caef5887485925ae6c7cd5008ec5a835823ced85fbe5.
  No Git push, container replacement, GPU allocation expansion or driver edits.
- Local reboot confirmed14:42:57 KST, last can/fabric fitting logs14:42:10,
  no completed metrics. Normal shutdown was not recorded; exact cause unknown.
  Initial container and local collector stopped. Remote survived and finished
  K1/K3 four tasks at14:59:07 without recorded failures; metrics audit pending.
- User authorized restarting. Started SAME local container, verified CUDA,
  retained observed450W post-boot power (initial interrupted attempts500W).
  K1 and K3 are both rerun under current local450W conditions. No power change.
- Preserved original status in status.before_reboot_resume.json and moved only
  the incomplete can/fabric worker trees to interrupted/reboot_20260912_144257/
  within the existing local queue, outside active stages. No deletion. Marked
  confirmed host interruption invalid_run with provenance; used generic
  experiments.py run --resume, retaining an interruption record and original
  logs. Fresh class fitting is required because checkpoints are disabled.
- Resume15:07:05 KST, local GPU0 can/fabric processes verified. Same K1 -> K3
  config sequence and 12 pending valid local results; two aborted partial
  attempts are retained separately. No remote rerun or code resynchronization.
- Generic collect --watch restarted for the same local root; canonical
  Markdown/CSV retain historical results. Full8 result consolidation is pending.

## 2026-09-12 — K3 constructed and locally unit-validated; not launched

- Request: construct the next method after K2's analysis. Hypothesis: retain
  K1 synthetic discrimination while limiting preservation-increasing parameter
  displacement, not weaken candidate selection. This remains untested on data.
- Added projected_update.py and config/design. Minimal common optimizer-step
  hook defaults to the original AdamW step; K3 uses one extra clean scoring
  VJP from already-cached frozen features and projects the actual Adam proposal.
- Current support first-order slack only, zero-gradient no-op, retained Adam
  moments; no finite-step/all-support/representation guarantee or novelty claim.
  Actual displacement/correction diagnostics are scalar; maps/weights stay off.
- Existing local container was idle (sleep infinity only) before mounted-source
  edits. 133 focused CPU tests passed: exact off K1/H1 parity, Adam momentum
  counterexample, projection/slack/zero/tiny-gradient checks, matching K1
  synthesis and backbone calls, gradient/state preservation, common config,
  source-output-policy and generic-runner regressions.
- K1 -> K3 matched full8/seed0/64-step development comparison is documented,
  NOT launched. No benchmark/GPU experiment, remote sync, Git push, container
  replacement, dense outputs, or new performance/cumulative-ledger rows.

## 2026-09-12 — K2 full8 completion and collection

- Status check: local18/18 complete12:34:37 KST, dsba3 6/6 complete12:04:02 KST;
  no recorded failures; both assigned GPUs idle. No restart/new experiment.
- Full8 H1/K1/K2 x100 AUPRO47.41422/47.68475/47.55686,
  pAUROC76.54504/76.72671/76.62417, F1oracle35.81383/36.18544/35.97395.
- K2-K1 AUPRO -0.12789pp; onlyvial/sheet_metal improve. Vial recovers2.06683pp
  versusK1 and0.09190pp versusH1. Valid run, no primary improvement; no claim
  of statistical/generalization or causal mechanism success.
- All24 outputs finite/covered/64-step/normal16/100384 parameters/frozen backbone;
  all8 triplets matched full protocol/content/common config/code/runtime.
- Read-only K2 training log aggregation:512 steps, max preview/replay error0,
  feasible290, weak-strength177 (34.57%). Full remote logs remain remote.
- Scoped generic pull recovered summaries; permission-denied18 JSON/YAML files
  were read through the existing container plus terminal status. No remote
  permission changes; local JSON copies normalize terminal newlines only.
- Canonical Markdown/CSV replace21 local6 rows with27 full8 rows, preserving305
  historical rows. Current state/TODO updated; no next experiment scheduled.

## 2026-09-12 — K2 student boundary implemented and launched

- User authorized a method/experiment iteration on K1's diagnosed boundary
  mismatch. Added student_boundary.py, config K2 and STUDENT_BOUNDARY.md.
- K2 previews the current student, exact upcoming paired transform and sampled
  ranking pairs. Selects closest feasible worst-view quartile gap among the
  same three strengths; original-strength fallback, scalar diagnostics only.
  H1 losses/controller/inference fixed; off parity to K1 and H1 tested.
- 108 focused tests passed in each existing local/dsba3 container; CUDA tensor
  checks passed on local GPU0 and dsba3 GPU2. Local container NVML failure was
  recovered by restarting the SAME idle container; no power/driver edits.
- Source snapshot 17aaa3c579b3eded83be4a6c2a841f8985d6ba5eb497b9eb9307f2fce6ab2ffc
  activated with the existing local-snapshot remote helper. No Git push.
- Launched H1 -> K1 -> K2, seed0/normal16/448+672/64 steps, seen TESTpublic,
  24 class-run maximum, 3600s/class, two workers/GPU, no sweep. Local six classes
  under student_boundary_20260912_local; dsba3 GPU2 vial/wallplugs under
  student_boundary_20260912_dsba3_gpu2. Both started H1; no candidate metric yet.
- Fixed generic collector's parallel active-status rendering before launch.
  One local collector updates the same canonical CSV/Markdown; remote compact
  pull and full8 consolidation remain pending. Initial host background launch
  attempts exited before starting remote work/collector; separate-session
  launch succeeded. Local experiment was never duplicated or interrupted.

## 2026-09-12 — K1 full8 completion/status check

- Both queues complete without recorded failures: local12/12 at10:54:15 KST,
  dsba3 4/4 at10:34:01 KST. Both assigned GPUs idle at inspection.
- H1 -> K1 full8 x100: AUPRO47.40645 ->47.70115 (+0.29470pp), pAUROC76.52947
  ->76.73356 (+0.20410pp), oracle F1 35.78126 ->36.20219 (+0.42092pp).
  Six classes improve; vial -1.89786pp and sheet_metal -0.15140pp. Modest observed
  seed0 development gain; no causal, significance or sealed-final claim.
- All8 pairs match full recorded protocol, common effective config, code,
  backbone and runtime. All16 pass finite/coverage/step/support/parameter checks.
- Generic terminal collector had already populated local6 results. Normal
  pull recovered remote summaries but not root-owned compact JSON/YAML; read
  those12 files through the existing container without changing remote state.
- Updated canonical Markdown/CSV together: replaced14 local6 rows with18 full8
  rows, preserving287 historical rows. Kept remote logs, maps/checkpoints off.
- No training/evaluation/restart/source fix during status checking. The existing
  active parallel collector rendering bug remains a separate pending item.

## 2026-09-12 — K1 single-bottleneck implementation and validation

- User requested an improvement targeting only the most important problem.
  K1 changes H1 synthetic strength selection to a frozen support-normal boundary;
  it keeps absolute hard-normal competition, controller and inference fixed.
  This is a proxy-difficulty hypothesis, not a claim of realistic synthetic GT.
- Existing F2 weak blends/reliability differ: K1 uses frozen teacher candidate
  selection, not student reliability weighting. No novelty claim is made.
- Added boundary_synthetic.py, on/off config K1, bounded design and tests;
  common CLI/runner/evaluator unchanged. Both containers pass86 targeted tests.
- Source snapshot 1b9677fab5b80bd7a679aa01a23ee7398aba89f4cabc4a3d47a967d1491046d0;
  existing container local-snapshot update helper used, no Git push.
- Prepared16 class-run H1/K1 comparison, seed0/normal16/seen TESTpublic/full8;
  local GPU0 six classes, dsba3 GPU2 vial/wallplugs, 2workers/GPU, 3600s/class.
  Maps/checkpoints off; candidate adds teacher fitting forwards, no parameters
  or inference overhead. No K1 performance result is claimed at preparation.
- Launched both queues successfully under the assigned UUIDs. Local H1 uses
  two actual worker processes; stage status also lists queued classes. Remote
  H1 runs vial/wallplugs, with the execution lock held for the whole queue and
  immutable release as cwd. K1 follows its H1 stage automatically.
- Generic host collect watcher records local6 partial scope; combined full8
  collection remains pending remote completion and compact pullback.
- Watcher correction: its first active parallel-status update raised
  KeyError('object') before ledger writes. No training/evaluation failure.
  Left mounted script unchanged; switched to a bounded terminal-status waiter
  that then calls the existing collector. Parallel rendering fix is deferred.

## 2026-09-12 — J1 completed and collected

- All16 class runs completed without recorded failures; local 09:44:39 KST,
  remote 09:28:28 KST, both launch wrappers exit 0.
- Full8 AUPRO H1 47.40352 -> J1 47.53549 (+0.13197pp), pAUROC 76.53814
  -> 77.00359, oracle F1 35.80016 -> 36.49949 (x100). Six AUPRO gains,
  vial -3.03246pp and fabric -0.23413pp declines. Single-seed development only.
- Eight matched pairs and 16 final coverage/parameter/step checks passed.
  Remote metrics-only pull recovered summaries; the four root-owned metrics
  were read in the existing container without changing remote permissions.
- Replaced local6 temporary14 ledger rows with full8 18 rows; preserved269
  historical rows. No new method/evaluation/training or container changes.
  Detailed class results and provenance: results/comparisons.md, latest J1 section.
- Final J1 tail/mean ranking ratio is approximately4 in all classes; mining
  versus loss-scale credit remains unresolved. Vial final P/b 0.67770 ->
  1.41967 accompanies regression but does not prove its cause.

## 2026-09-12 — J1 hard-pair tail risk implemented and launched

- User requested the next proposed approach. Implemented `tail_risk.py` and
  `tail_risk_J1.yaml`; normal-tail competition, synthesis, H1 controller and
  common pipeline remain fixed. Ranking reduces the upper 25% of sampled
  pair hinges instead of averaging all of them. Novelty is not claimed;
  OHEM/CVaR related work and scale/selection confounds are in TAIL_RISK.md.
- Disabled J1 exactly reproduces H1 in fixture fitting. Full-mass ranking
  matches H1 loss/gradients within floating tolerance and consumes identical
  sampler state. Focused gradient/invalid-input/scalar-output tests plus the
  existing relevant suite: 96 passed locally and 96 remotely. Ruff was absent.
- Reused the stopped local container after observing a host reboot outside
  this task. Kernel/NVML match 580.178.04; local and remote CUDA smoke passed.
  dsba3 container was already running and GPU2 idle. No container recreation,
  driver modification, Git push, model/data download or GPU expansion.
- Synced code using run_remote.sh update-code while idle; release snapshot
  6e792a707969c2eb01ed78aa1173de8c222f86ced3705909c9f0171b9761e9c1.
  Local/remote launches use existing scripts/experiments.py and run.py/CLI,
  detached Docker execution with shared execution locks and no-clobber logs.
- H1 then J1: 16 class runs total, seed0/normal16/64 steps/448+672,
  seen-development full8 AUPRO primary. Local can/fabric/fruit_jelly/rice/
  sheet_metal/walnuts; dsba3 physical GPU2 vial/wallplugs. Two workers/device.
  Status roots: tail_risk_20260912_local and tail_risk_20260912_dsba3_gpu2.
  Maps/weights off; remote logs/provenance retained there. Performance pending.

## 2026-09-12 — Three independent H1 extensions: complete collection

- H1/I1/I2/I3 completed all 32 class runs, local24 and dsba3 GPU2 eight.
  This collection performed no fitting/evaluation, code change, container
  restart, GPU reset, or remote permission modification.
- Full8 AUPRO@0.05 x100: H1 47.41448, I1 selective preservation 47.30383,
  I2 paired margin 46.95461, I3 anchored retrieval 47.37585. All candidates
  remain below the new matched H1 control; changes are not cumulative.
- 24 candidate/control pairs match support/query/GT identities/content,
  code, common config, backbone and recorded environment. All32 final
  manifests/metrics pass coverage checks and record 64 steps/100,384 params.
- I2 ranking loss is zero on 306/512 logged steps, with mean uplift 0.33731
  versus margin 0.05584. Mean P/b is 0.21664 versus H1 0.60677 despite lower
  AUPRO/pAUROC/oracle F1. Synthetic-objective/real-metric mismatch is a
  post-hoc lead, not a proved causal explanation. I1 and I3 have class
  tradeoffs; their small negative deltas do not reject entire method families.
- Used scripts/experiments.py collect for the local results, then integrated
  remote metrics into the same canonical CSV/Markdown full8 scope. The
  metrics-only pull could read summaries but not root-owned metrics; the
  existing remote container supplied those eight metric files without chmod.
  No configs/manifests/full training logs/maps/weights were pulled back.
- Replaced only the new local6 interim rows (28) with 32 class + 4 macro
  rows, preserving 233 historical rows. Detailed analysis, class tradeoffs,
  provenance, baseline caveats and the local driver-update note are recorded
  at results/comparisons.md#three_directions_20260912_combined_status.
- Decision: no demonstrated performance winner among I1/I2/I3. A next
  design could retain hard-normal competition while revising the easy
  synthetic target; this is a suggestion, not a launch or implementation.

## 2026-09-08 — Persistent-container remote workflow

- User-selected workflow: local code edit/test, authorized commit/push, Git
  fast-forward update inside one designated existing container, explicitly
  assigned GPU execution, and compact quantitative result pullback.
- `scripts/run_remote.sh` now reuses a running container or starts the same
  stopped one; missing containers fail without provisioning. Explicit `stop`
  does not remove the container, and `setup` is disabled.
- `update-code` refuses a dirty checkout and uses `git pull --ff-only`.
  It shares a nonblocking container lock with `run`/`exec`. Command arguments
  retain their boundaries; GPU selection is explicit via
  `REMOTE_GPU_DEVICES`. Foreground experiment logs/exit status remain remote.
- `pull` allowlists metrics, summary, config, and run manifest; no map, weight,
  or training-log transfer and no automatic remote cleanup. Legacy sync/build
  are maintenance operations, not part of the experiment cycle.
- Validation: 44 offline remote-workflow/security tests, Bash syntax check,
  and focused Python lint passed. A real local rsync fixture confirmed the
  compact-result transfer set. No remote SSH, Git push, Docker operation, or
  benchmark was performed. Existing-container mounts, GPU exposure, Git
  credentials, and live runtime compatibility remain unverified.

## 2026-09-08 — Active guidance separated from historical evidence

- The live index and TODO now point to the current project contract and
  `skill_graph/current.md`; this log retains dated evidence rather than serving
  as an execution queue.
- Research iteration now prioritizes method implementation and matched
  development performance, followed by claim-focused validation and post-hoc
  analysis. Execution validity, performance evidence, and research decisions
  are distinct; historical family closures do not create new blanket bans.
- The global FMAD experiment and discovery-wrapper skills were updated to
  follow project-local conditions and budgeted iteration. Their former texts
  are preserved in [instruction backups](archive/instructions_2026-09-08/README.md).
  Hook registration, model settings, runtime code, and historical quantitative
  evidence were not changed by this guidance update. No experiment was run.
- The common `src/fmad` pipeline, identical-condition comparison contract, and
  default output policy are the current operating baseline. See the
  [refactor report](analysis/2026-09-08_common_pipeline_refactor/report.md),
  [results consolidation](analysis/2026-09-08_results_consolidation/README.md),
  and [current state](current.md).
- All text below this transition note is historical and intentionally unchanged.
  Experiment-specific bans, approvals, and termination conditions remain scoped
  to their named experiment and recorded conditions; they do not silently
  govern a new method.

## 2026-08-29 — PSAR-D1 closes invalid after repeat storage failure

- Integrated the completed refactor without overwriting its changes, applied
  only the preregistered exact-semantics equal-image q95 complexity repair, and
  resealed source generation `1c9314da...a703e0`. Focused tests passed `101/101`;
  remote preflight and bootstrap/CUDA/cell smokes all passed.
- In sole run `psar_d1_dsba3_q95_reseal_20260829_v1`, canonical `fabric`,
  `vial`, and `walnuts` packages completed, proving that q95 no longer stalls
  map generation. This is operational evidence, not a detector result.
- `fabric_repeat` then filled `/workspace` after 495 raw maps. Its summary write
  also failed (`Errno 28`, `launch_failed_rc=74`), leaving no repeat inventory
  and no complete 24-package barrier. GT/evaluation never opened and no
  scientific metric is reported.
- Ordered gate 1 gives `INVALID_FAIL_CLOSED / CLOSE PSAR-D1`. Readable maps had
  already crossed `revision_budget=0`, so no storage repair/rerun, all-eight
  confirmation, or LoRA is authorized.
- The 26-file compact bundle is retained locally and remotely with manifest
  `c3eb2332...00484764`. After verification, only the exact run's dense files
  and isolated snapshot were removed; the cleanup receipt is retained.
- Report:
  `skill_graph/experiments/2026-08-28_psar_d1_checkpoint/report.md`.

## 2026-08-28 — ECO-O1 invalid terminal closes the ECO family

- Ran the sole ECO-G2-authorized lifted-J0/G2 operator-sentinel generation in
  three separately sealed immutable invocations on the fixed dsba3 project
  container and authorized GPU scope only.
- Attempts 001 and 002 stopped before data/model-derived output on controller
  leaf-creation and duplicate-argument defects. The final attempt passed
  `58/58` remote P0 tests and byte-identical P1 synthetic runs, then imported
  the exact passed parent N0 hash
  `df8404a51aef2bb6148bb75d95b548b6debbf23c35d517a8da91016774b0932e`.
- The frozen G2 validator nevertheless requires full custody equality. Parent
  N0 records G2 snapshot `65b887c0...`, while the current child snapshot is
  `eedd3839...`; the exact parent artifact is therefore structurally
  inadmissible inside the separately sealed O1 snapshot. The run stopped before
  support image decoding, model opening, support seal, held shards, or any O1
  scientific metric.
- Final terminal is `INVALID_FAIL_CLOSED / END_ECO_FAMILY`. This is an invalid
  execution closure, not `VALID_BUT_NO_GAIN` and not a negative detector
  result. No fourth repair/run, detector bridge, three-object checkpoint,
  all-eight evaluation, support-only rescue, or LoRA is authorized.
- All three compact invalid pulls and invalid remote snapshots are retained.
  Final artifact SHA256 is
  `c332e2717731db11a028ae1bf2ce58e9c92ac05ea18775f8bdef9cc94ca4c8e5`.
- Report:
  `skill_graph/experiments/2026-08-28_eco_o1_lifted_j0_g2_operator_sentinel/report.md`.

## 2026-08-28 — ECO-G2 native observation passes; one sentinel only

- Completed the unchanged ECO-G2 N0/N1 contract in valid run
  `eco_g2_nativecore_dsba3_20260827T212434Z_g2v5` after recording four
  fail-closed, non-promotable runtime attempts.
- N0 passed all 20 committed masks; native strict-core counts were `76--247`.
  N1 passed all eight held/readout targets with `80--183` active core rows and
  energy coverage `0.967477--0.992872`; RAW/standardized decisions agreed and
  held-0 repeat bytes were identical.
- Verdict is `PROMISING_DIAGNOSTIC / OPERATOR_ONLY_SENTINEL_AUTHORIZED /
  NO_LORA`. The old operator confound remains unresolved, so benchmark,
  detector, and LoRA claims remain closed.
- Per the prospective portfolio policy, exactly one separately preregistered
  operator-only sentinel is next. It must lead directly to a pixel detector
  bridge or terminate ECO; another structural/support-only rescue is barred.
- Report:
  `skill_graph/experiments/2026-08-28_eco_g2_nativecore_stage_minus1/report.md`.

## 2026-08-28 — ECO-Audit v2 localizes two independent attribution blockers

- Completed immutable
  `eco_audit_v2_dsba3_20260828Tfrozen12Z_v12` on the fixed dsba3 project
  container, GPUs 0--3, after P0/P1/P2, four held shards, and one byte-identical
  held-0 repeat. No other user/container/process resource was touched.
- The v11 source-footprint equal-image Q10 repair made all 16 support images
  valid with 8--17 tokens and sealed raw/std thresholds
  `1.0152778625 / 0.0108348746` without held access.
- The aggregate is structurally valid, but all 32 cells are officially
  `UNRESOLVED_OR_INVALID`: `ACTIVE_INTERIOR` is empty in 24 and singleton in 8.
  Independently, all 32 cells are pixel-operator-confounded (`kappa_all` above
  `.10` and boundary commutator-error share above `.25`).
- Active-global, active-boundary, and N5 pass `32/32`; active coverage is
  `0.9636--0.9918`, and the exact repeat is byte-identical. This is a useful
  metric-localization signal but not a genuine local mismatch or LoRA gate.
- Verdict `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`: close the exact ECO
  geometry/operator configuration and allow at most one separately
  preregistered geometry-only feasibility diagnostic. AGIAS and LoRA remain
  closed.
- Compact evidence remains locally and remotely; no dense/anomaly-map artifact
  exists, and only the exact completed v12 immutable snapshot was removed after
  an exact-path absence/result-preservation check.
- Report:
  `skill_graph/experiments/2026-08-27_eco_audit_v2_stage_minus1/report.md`.

## 2026-08-27 — NCFC postmortem, literature, and future direction

- Local experiment lineage and NCFC-S1 were re-audited together with primary
  literature on representation metrics, equivariance/action learning,
  identifiability, frozen few-shot anomaly detection, synthetic anomalies, and
  residual adapters/LoRA.
- The design loop converged after four stages (`0.78 -> 0.82 -> 0.80 -> 0.80`).
  Best immediate direction: frozen `ECO-Audit v2`; conditional learned program:
  `AGIAS`.
- Direct single-path backbone LoRA remains closed. No GPU experiment or remote
  resource access occurred in this research-only turn.
- Report:
  `analysis/2026-08-27_ncfc_postmortem_literature_future_direction.md`.

# 2026-08-27 — NCFC-S1 finds a normal coordinate but no strict anomaly commutation

- Completed frozen DINOv2 block-11 `can x J0` NCFC-S1 on dsba3 GPUs 0--3
  with four held bases, both EV signs, raw/standardized readouts, and one exact
  held-0 repeat.
- Structural, normal, RN-safety, and hard-null gates passed in both readouts;
  clean-derived anomaly correction cosine was `0.956--0.993` and the repeat
  maximum metric difference was `0.0`.
- Strict residual commutation passed only `3/8` cells per composition order.
  Held base 1 was the dominant failure (`0.717--0.742` balanced cosine) while
  retaining `0.907--0.922` mask-inside cosine and near-identity RN geometry.
- Raw and standardized results agreed, so readout normalization does not
  explain the failure. Compact custody and source hashes passed, independent
  reaggregation was byte-identical, and no dense artifact was retained.
- Verdict `NORMAL_ONLY_NO_ANOMALY_SIGNAL / END_NCFC`; do not reopen LoRA.
- Report:
  `skill_graph/experiments/2026-08-27_ncfc_s1_stage_minus1/report.md`.

# 2026-08-25 — CRLP-Predict source-only all-eight diagnostic closed

- Implemented the isolated direction-decoding and cross-fitted shift-residual
  prediction gates with exact RN actionability, complete matched controls,
  fail-closed custody, and a validity-only GPU0 sentinel.
- A first immutable attempt exposed a pre-result one-dimensional magnitude
  control projection defect and stopped before sentinel output; after a direct
  regression fix, immutable run
  `crlp_predict_stage_minus1_dsba3_20260825T122151Z_v2` completed all eight
  objects on dsba3 GPUs 0--3.
- All `96/96` cells and `8/8` objects were valid, but Gate A/B passed `0/96`
  and the joint mechanism passed `0/8`. Generator identity remained strongly
  decodable at BA `0.7656--0.8740`; magnitude or residual-spectrum controls won
  87/96 Gate-A cells.
- Learned normal-tail q99 reduction was median `-4.375%`, met the 5% floor in
  `0/96`, preserved anomaly/RN geometry in only `12/96`, and beat matched
  controls in `0/96`. The paired-shift oracle passed preservation `93/96` and
  5% gain `40/96`, localizing the failure to safe shift estimation rather than
  the algebraic subtraction form.
- Compact hashes passed, local reaggregation was byte-identical, all forbidden
  flags were false, no dense maps remain, and GPUs 0--3 returned idle. Verdict
  `KILL_FOR_CLAIM / NO_CONTINUE`.
- Report:
  `experiments/2026-08-25_crlp_predict_stage_minus1/report.md`.

# 2026-08-14 — Transport-quotient RN TESTpublic completed without signal

- Completed and pulled all eight immutable v6 shards from dsba3/dsba5 GPU3;
  every shard passed custody, coverage, finite/nonconstant-map, and no-adaptation
  gates with no retained dense maps.
- The valid aggregate ranks full-support `B0_RN1` first at macro
  pAUROC/F1/AU-PRO `0.749400/0.336069/0.435772`.
- `M0_topk_orbit` scored `0.652313/0.184042/0.289897`, losing to B0 by
  `-.097088/-.152026/-.145875` and losing pAUROC on all eight objects.
- M0 only separated from `N0_shuffled_tangent` by
  `+.002303/-.000684/+.004648`; the local tangent orientation is not supported
  as the causal mechanism.
- Fixed an aggregation-only schema mismatch by reconstructing support identity
  from the SHA-pinned manifest and TESTpublic identity from identical arm-level
  `per_image_audit`; original experiment scores and snapshot hashes were unchanged.
- Protocol verdict `VALID_NO_SIGNAL`; research decision
  `KILL_FOR_CLAIM / NO_CONTINUE`.
- Report:
  `skill_graph/experiments/2026-08-13_transport_quotient_rn_testpublic/report.md`.

# 2026-08-14 — Optimized transport-quotient RN TESTpublic run submitted

- Initial v5 failed before any shard at the all-root tangent gather
  (52.73/33.04 GiB temporary allocation on can/fruit_jelly).
- Added bounded paired root/part gathering, preallocated output, and identity
  remap elision; 59 focused tests and a CUDA memory smoke passed.
- Submitted replacement immutable run
  `transport_quotient_rn_testpublic_v6_chunked_20260814` after both
  dsba3/dsba5 GPU3 preflights passed.
- Execution uses a resident exact RN index, one compact union tangent cache, and
  one resident CUDA quotient scorer without changing the five-arm method contract.
- Queue ownership is dsba3 `can/fabric/walnuts` and dsba5
  `fruit_jelly/rice/sheet_metal/vial/wallplugs`; aggregation remains pending all
  eight same-snapshot shards.
- Report:
  `skill_graph/experiments/2026-08-13_transport_quotient_rn_testpublic/report.md`.

# 2026-08-13 — Transport-quotient RN TESTpublic runner surface preregistered

- Added the local runner/aggregator contract for all-eight full-support
  TESTpublic transport-quotient RN diagnostics.
- Required arms are `B0_RN1`, `B1_exact_anchor_orbit`, `M0_topk_orbit`,
  `N0_shuffled_tangent`, and `C0_no_tangent`; runner primary is
  `B1_exact_anchor_orbit`.
- The score contract is no-gate/no-cap and requires RN candidate distance,
  orthogonal residual, and support-variance-regularized tangent-coordinate
  penalty.
- Report:
  `skill_graph/experiments/2026-08-13_transport_quotient_rn_testpublic/report.md`.

# 2026-08-12 — SPCT B1 full-support TESTpublic preregistered

- Frozen the new full-support TESTpublic launch package for `B0`, `B1`, `N0`,
  and `RN_FMLK_full`.
- Support custody is the full `train/good` population, sealed under canonical
  remote prefix `/home/hunim/Volume/DATA/mvtec_ad_2` at SHA-256
  `ac084290bd40d171f12e876b45c3baf9b2269010cc2e62a0e91d7c4b80badd53`.
- Test counts are fixed at `72/90`, `66/90`, `20/60`, `42/90`, `24/90`,
  `35/105`, `60/90`, and `60/90` across the eight objects, with
  `sheet_metal` at resolution `448` and the others at `672`.
- Verdict ceiling is `PROMISING_DIAGNOSTIC` only; the run is benchmark-
  incomparable because the support budget is not SuperAD16.

# 2026-08-10 — Cross-image-calibrated region completion does not recover missing components

- Fitted current/low-pass score calibrators from 16 true P16 leave-one-image-out folds per object,
  with 15 fit images per fold and zero held/fit overlap; all seven validity gates passed.
- Duplicated `can` across local RTX 4090 GPU 0 and dsba5 host GPU 3 agreed within
  `5.46e-7` over every shared continuous metric and arm.
- A5 lost AU-PRO/component recall versus A0 (`-.009657/-.083333`) and calibrated max
  (`-.004691/-.080729`); current-only morphology matched A5 component recall and had
  `+.004993` higher AU-PRO.
- `rice` F1 fell `-.053521`, `can` exceeded its normal-FPR guard, and medium AP fell
  `-.196325`; strict and continuation gates both failed.
- Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`. Do not tune completion thresholds, lift,
  connectivity, calibration knots, fusion, or object-specific policies.
- Report:
  `skill_graph/experiments/2026-08-10_cross_image_calibrated_region_completion_stage_minus1/report.md`.

# 2026-08-08 — DVT contains a useful low-frequency positional component, but pure low-pass loses medium gain

- Fixed Gaussian low-pass DVT repaired small top-1 coverage from `-.011532` to `+.001132` versus raw.
- It retained only `67.48%/71.71%` of current DVT's medium AP/top-5 gain, failing the strict 75% gate.
- Correct low-frequency position beat an equal-energy shuffled null by `+.056823/+0.063081` on medium
  AP/top-5 coverage, so the medium signal is not generic smoothing or norm reduction.
- Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`; next allow only a frozen low-frequency correction plus
  protected high-frequency defect-residual decomposition, not sigma/alpha/fusion tuning.
- Report: `skill_graph/experiments/2026-08-08_scale_selective_residual_dvt_diagnostic/report.md`.

# 2026-08-08 — DVT effect size survives K5, but object generality is scorer-sensitive

- Repeated the four-object DVT × Flow factorial with K1/K5 emitted from the same evaluation.
- K5 retained DVT pAUROC/F1 main effects (`+.030991/+.069351`) and DVT×Flow pAUROC interaction
  (`+.006424`), close to K1 (`+.034675/+.069455`, interaction `+.006278`).
- The strict gate failed because positive objects decreased `3/4 → 2/4`; AU-PRO main effect also
  shrank `+.009686 → +.000111`.
- Verdict: `SCORER_DEPENDENT_DVT_ATTRIBUTION / EFFECT_SIZE_MOSTLY_RETAINED`; keep K1 and prioritize
  scale-selective positional structure over k/Mahalanobis sweeps.
- Report:
  `skill_graph/experiments/2026-08-08_baseline1_dvt_representation_scorer_factorial/report.md`.

# 2026-08-08 — Exact-position retrieval is stable but overconstrains local defects

- Restricted full-feature top-5 retrieval to the same support coordinate and compared it with
  global retrieval, half-grid rolled position, and source-permuted position hard nulls.
- Correct position reduced shifted-normal tail `39.55%` and improved clean↔shift Jaccard `+.2356`
  versus global retrieval, but passed wrong-position mechanism separation on `0/4` objects.
- Local defect retention fell to `.8851` macro and `.6422` on fabric; broad retention was `1.2144`.
- Verdict: `POSITION_OVERCONSTRAINT / NO_CONTINUE`. DVT positional evidence does not transfer to
  a hard same-coordinate retrieval constraint.
- Report:
  `skill_graph/experiments/2026-08-08_positional_key_full_value_retrieval_stage_minus1k/report.md`.

# 2026-08-08 — Stable key preserves defect value but does not improve retrieval

- Split DINO retrieval into a stable-channel key and full-feature anomaly value, with exact `k=5`,
  full/full baseline, stable/stable absorption control, and shuffled-key hard null.
- A3 local/broad defect retention was `1.0572/1.0155`, whereas stable/stable A1 retained only
  `0.4901/0.5945`.
- A3 shifted-normal relevance passed `1/4`, retrieval mechanism `0/4`; Jaccard margin was only
  `+.00125/+.00611` versus A0/A2 and tail reduction versus A0 was `-4.74%` median.
- Verdict: `HARD_NULL_ABSORBED`. Keep full anomaly value, but close sensitivity-ranked stable key.
- Report:
  `skill_graph/experiments/2026-08-08_stable_key_full_value_retrieval_stage_minus1j/report.md`.

# 2026-08-08 — Shift-stable DINO channels also carry defect structure

- Fit a fold-local fixed readout from the lower half of DINOv3-H+ channels ranked by normal-support
  photometric sensitivity; compared it with identity and an equal-cardinality score-shuffled null.
- The readout passed invariance on `4/4` objects: held compound-shift displacement fell to `0.6078`
  versus `1.0052` for the shuffled null, so selective shift-sensitive channels do exist.
- The same selection retained only `0.4941/0.5949` of local/broad defect contrast and reduced clean
  NN identity/pair geometry to `0.6719/0.8039`; raw 1-NN relevance passed on only `1/4` objects.
- Verdict: `ANOMALY_ABSORPTION`. A fixed lower-50% channel rejection is not a safe invariant
  representation, and its trainable diagonal-gate continuation is closed.
- The support-only diagnostic used no test/public/private/GT or metadata signal and is not benchmark comparable.
- Report:
  `skill_graph/experiments/2026-08-08_dino_invariant_channel_readout_stage_minus1/report.md`.

# 2026-08-08 — Small-defect loss begins at DVT, not Flow

- Re-ran all eight objects and measured the identical query at pre-DVT feature, post-DVT feature,
  Flow latent 1-NN, and deployed-final score stages.
- Small-defect top-1% coverage changed from exact `0.0` before DVT to `-0.011667` immediately after
  DVT, satisfying the preregistered onset rule with four negative objects.
- Flow latent and deployed final both stayed at `-0.011532`, while medium AP/top-5 coverage gains
  were already `+.07412/+.07464` post-DVT and ended at `+.07571/+.07321`.
- Verdict: `LOSS_AT_DVT_SUBTRACTION`. Flow neither creates nor repairs the small-defect cost; it only
  slightly modulates a DVT-originated scale tradeoff.
- Report:
  `skill_graph/experiments/2026-08-08_dvt_small_defect_signal_path_attribution/report.md`.

# 2026-08-08 — DVT medium gain requires correct positional correspondence

- Compared current DVT against an equal-norm deterministic position shuffle and spatially constant global centering.
- Medium current-minus-shuffled AP/top-5 coverage was `+.09024/+.08731`; current-minus-global was
  `+.07571/+.07321`. All positional-specificity gates passed on six eligible objects.
- Shuffled/global controls did not reproduce the joint medium AP/coverage improvement over raw.
- Small current-minus-raw top-1 coverage remained `-.01153`; object collapse guards failed.
- Verdict: `POSITION_SPECIFIC_MEDIUM_GAIN_WITH_SMALL_DEFECT_COST` (diagnostic only).
- Interpretation: DVT is a scale-dependent object-position prior, not generic centering, arbitrary feature surgery,
  a universal denoiser, or a demonstrated distribution-shift corrector.
- Report:
  `skill_graph/experiments/2026-08-08_dvt_positional_causal_controls/report.md`.

# 2026-08-08 — DVT gain concentrates on aligned, medium-scale defects

- Measured leave-one-out support-only positional explainability before opening TESTpublic GT.
- Stratified GT components at fixed `<=1%`, `1--5%`, and `>5%` image-area boundaries.
- Medium defect AP/full AUROC/top-5% coverage changed by `+.07571/+.04317/+.07321`; scale gate passed.
- Alignment-high medium AP/coverage was `+.10908/+.10830` versus `+.00896/+.00302` in the
  eligible alignment-low subset; modulation gate passed, with a noted 4-vs-2 eligibility imbalance.
- Small defect top-1% coverage fell `-.01153`; `fruit_jelly` and `sheet_metal` AP collapse blocked
  a small-defect claim. Large was ineligible with only two objects.
- Local/dsba5 aggregate parity passed across 400 continuous values, worst delta `6.3310e-6`.
- Verdict: `ALIGNMENT_SCALE_ATTRIBUTION_SUPPORTED` (diagnostic only).
- Report:
  `skill_graph/experiments/2026-08-08_dvt_alignment_defect_scale_attribution/report.md`.

# 2026-08-08 — DVT localization prior survives all-8 threshold-free validation

- Compared fixed `fused+Flow` DVT off/on arms on all eight TESTpublic objects without condition,
  filename, or split metadata and without threshold-conditioned primary metrics.
- DVT changed macro bad-image pixel AP/full AUROC/top-5% defect coverage by
  `+.010637/+.008167/+.006349`; positive object counts were `7/8`, `7/8`, and `6/8`.
- Excluding fabric, AP and top-5% coverage remained positive at `+.008755/+.005540`.
- Top-1% coverage/component hit were slightly negative, so the supported mechanism is broad
  ranking/moderate-area coverage improvement, not universal peak sharpening.
- Local/dsba5 metric parity passed: 108 values, worst absolute delta `5.1280e-6`.
- Verdict: `GENERAL_LOCALIZATION_PRIOR_SUPPORTED` (diagnostic only).
- Report:
  `skill_graph/experiments/2026-08-08_baseline1_dvt_all8_rank_coverage/report.md`.

# 2026-08-08 — DVT gain attributed primarily to defect-pixel evidence

- Re-ran `fused+Flow` and `DVT+Flow` on the fixed four-object Baseline1
  diagnostic with condition suffixes used only after scoring.
- At each object/arm global oracle threshold, DVT changed defect-pixel TPR by
  `+.09616/+.09878/+.09577` for regular/photometric/geometric conditions.
- Normal good-pixel FPR changed by only `-.00019/-.00030/-.00058`; the strict
  shifted-normal suppression pattern held for `1/4` objects.
- Shifted-condition defect TPR improved for `3/4` objects, while component recall
  decreased macro, indicating stronger within-defect coverage/ranking rather
  than discovery of more defect instances.
- Verdict: `DEFECT_EVIDENCE_GAIN`. DVT remains an object-position localization
  prior, not a demonstrated shift estimator or metadata-conditioned module.
- Report:
  `skill_graph/experiments/2026-08-08_baseline1_dvt_condition_attribution/report.md`.

# 2026-08-08 — External auxiliary metadata branch killed; frozen image-only prior remains

- Audited whether the current MVTec AD2 single-image repo/data contains a deployable
  external auxiliary evidence source for nuisance-defect separation.
- Local evidence root was `/home/hun/Volume/DATA/mvtec_ad_2`; the protocol default
  remote root `/home/hunim/Volume/DATA/mvtec_ad_2` was not mounted in this session.
  All eight public AD2 objects were present locally.
- Only dataset-level `readme.txt` and `license.txt` were found as non-image files.
  No per-image manifest/CSV/JSON/XML/camera/exposure/sensor sidecar was found.
  Sampled PNGs across train/validation/public/private/private_mixed had the same
  generic `gamma=0.45455`, `dpi=(299.9994,299.9994)`, and empty EXIF.
- Public filenames expose `regular/overexposed/underexposed/shift_*`, while private
  splits expose `regular/mixed`. This is not usable metadata; it is a benchmark
  oracle channel. Verdict for metadata-token routing:
  `FORBIDDEN / DESIGN_KILL_METADATA_ORACLE`.
- Paired acquisition remains forbidden under the current single-image task unless
  the user explicitly reauthorizes a task switch. Metadata and paired capture are
  not unblock contracts.
- Overall verdict:
  `NO_METADATA / DESIGN_KILL_METADATA_ORACLE / NEXT_FROZEN_IMAGE_ONLY_PRIOR_FEASIBILITY`.
  The only allowed next branch is a named frozen general image-only prior audit
  with weights, license, output schema, and anomaly-preservation assay.
- Analysis:
  `skill_graph/analysis/2026-08-08_external_auxiliary_evidence_feasibility_audit.md`.
- Report:
  `skill_graph/experiments/2026-08-08_external_auxiliary_evidence_feasibility_audit/report.md`.

# 2026-08-08 — Nuisance-orbit anomaly persistence design killed as a closed branch

- FMAD negative-evidence intake stopped the proposed query nuisance-orbit
  anomaly-persistence experiment before code or GPU execution.
- The mechanism retreads `.omx/specs/.../stage-02` `Query Counterfactual
  Component Persistence`: multiple reversible query views, inverse-aligned
  score maps, and persistent component retention. The stored evaluator verdict
  was `reject` with score `0.32`, citing multiple query transforms, post-hoc
  map shaping, and persistent-nuisance/transform-sensitive-defect ambiguity.
- 2026-07-18 transformed support union and patchwise-min already showed only
  weak heterogeneous coverage (`+0.001790` pAUROC, `8.53%` tail reduction,
  `-0.000049` F1) with `NO_CONTINUE`; 2026-07-10 rotation-8 support
  augmentation was also `KILL_FOR_CLAIM / NO_CONTINUE`.
- WSNC/Augerino are retained only as literature context for transformation
  family misspecification and unsafe invariance; they are not direct proof of
  the design kill.
- Verdict: `DESIGN_KILL_CLOSED_BRANCH`. Do not run orbit sweeps, query-view
  persistence, or Flow Matching/Diffusion/DiT variants from this evidence.
  The next non-overlapping direction is an external auxiliary variable/evidence
  source feasibility audit, not a promoted experiment.
- Report:
  `skill_graph/experiments/2026-08-08_nuisance_orbit_anomaly_persistence_design_audit/report.md`.

# 2026-08-07 — Layer-7 / fused split interface hard-stopped

- Used layer 7 only to select a fixed 75% mask-free inlier patch set while retaining fused
  coordinates for condition transport and anomaly scoring.
- C7 reduced C0 local/broad condition drift by at most `21.8%/13.6%`, below the fixed 25% gate;
  `walnuts` local drift increased and shifted-normal tail worsened on three objects.
- Strict and weak-continuation passes were both `0/4`. Local/dsba5 validity and parity passed with
  mismatch 0 and no test/public/private/GT access.
- Verdict: `KILL_FOR_CLAIM / NO_CONTINUE_SPLIT_INTERFACE`. Stop layer/fraction/distance tuning and
  do not open Flow Matching/Diffusion/DiT or public evaluation from this evidence.
- Report:
  `skill_graph/experiments/2026-08-07_layer7_fused_split_interface_stage_minus1h/report.md`.

# 2026-08-07 — Layerwise defect spillover found localization without safe conditioning

- Fixed layers `[7,15,23,31]` and the fused DINO representation were audited on all fixed-16
  support sources without fitting, Flow, DVT, test data, or ground truth access.
- Layer 7 preserved local defect energy spatially on all four objects (far spill `.060--.088`),
  whereas later/fused representations were less local. Nevertheless its condition contamination
  was `.214--.688`, so conditioner feasibility passed `0/4`.
- Local GPU 0 and dsba5 GPU 3 completed with identical hashes, validity pass, and parity mismatch 0.
- Verdict: `KILL_FOR_CLAIM / CONTINUE_LAYER_INTERFACE_DIAGNOSTIC`. Permit only one support-only
  split-interface preflight (layer-7 conditioner input, existing fused anomaly scorer); do not infer
  layer selection or open Flow Matching/Diffusion/DiT/public benchmarking.
- Report:
  `skill_graph/experiments/2026-08-07_layerwise_defect_spillover_stage_minus1g/report.md`.

# 2026-08-07 — Score-aligned residual and kNN scorer sentinel stopped

- A fixed rank-4 support nuisance subspace captured more held NN residual energy than random,
  but margins `.053/.072` missed `.10`; primary 1NN did not improve and clean NN geometry failed.
- Fixed scorer controls showed small consistent 5NN averaging gains (~`2%`), 15NN regressions,
  and object-dependent soft-min behavior. This does not justify a k sweep or scorer claim.
- Two-object sentinel checksum/custody passed with no test/public/private/GT access. The frozen
  sentinel gate failed both objects, so remote/four-object scaling was blocked.
- Verdict: `KILL_FOR_CLAIM / NO_FOUR_OBJECT_CONTINUATION / NO_PUBLIC_ACCESS`.

# 2026-08-07 — Shared-direction intervention hard-stopped

- Stage -1c tested the only continuation licensed by Stage -1b: a fixed, untuned rank-1
  removal of the support-shared global photometric direction on a distinct unseen transform.
- Synthetic local/broad defect contrast and clean geometry were retained, but shared-direction
  q995 reduction was near zero (`-.00024`--`.00029`) versus the frozen `.02` gate and did not
  dominate the source-random direction null (`0/4`).
- Local compact checksums and custody passed. The remote lane was stopped after the local
  `0/4` hard-stop made parity unable to change the verdict; no public/private/test or GT path
  was accessed.
- Verdict: `KILL_FOR_CLAIM / NO_CONTINUE_SHARED_DIRECTION_INTERVENTION / NO_PUBLIC_ACCESS`.
  Cross-image direction reproducibility must not be used as a nuisance-removal selector.

# 2026-08-07 — Support-shared photometric direction diagnostic continued design-only

- Completed the separately preregistered Stage -1b evidence audit after Stage -1a1
  rejected spatial conditional-smoothness as a global nuisance selector.
- The held support image displacement direction for both shared global and smooth
  photometric transforms transferred better than a source-random photometric null
  on all four objects. Minimum margins were `.0829/.0918` on `vial`; all passed
  the frozen `.05` gate.
- Local GPU 0 and dsba5 GPU 3 hashes, access ledgers, compact checksums, and
  aggregate parity passed (maximum delta `1.41e-6`), with no test/public/private
  or ground-truth path accessed. The remote immutable source snapshot was removed.
- Verdict: `PROMISING_DIAGNOSTIC / CONTINUE_DESIGN_ONLY / NO_PUBLIC_ACCESS`.
  This permits one held-transform transfer/defect-retention diagnostic only; it
  does not authorize module implementation or public benchmarking.
- Report:
  `skill_graph/experiments/2026-08-07_support_shared_photometric_direction_stage_minus1/report.md`.

# 2026-08-07 — Mask-free conditional-smoothness audit stopped support-only

- Separately preregistered and ran the Stage -1a1 measurement that excludes
  unchanged local-defect patches from smoothness averaging by conditioning on
  DVT displacement magnitude, without using any defect mask in the metric.
- Global displacement coherence and energy effective coverage exceeded the
  defect aggregate on all four objects, but active-pair conditional-smoothness
  was lower for global shifts on all four (`0/4`, required `>=3/4`). Positive
  local/broad defect retention makes this a factorisation-signal failure, not
  a safety failure.
- Local GPU 0 and dsba5 GPU 3 used identical runner/support hashes and access
  ledgers; all aggregate metrics met parity tolerance (maximum delta
  `.0009765625`). No public/private/test or ground-truth path was accessed;
  the dsba5 immutable source snapshot was removed after pullback.
- Verdict: `INCONCLUSIVE_OR_NOT_SUPPORTED / NO_NEW_MODULE / NO_PUBLIC_ACCESS`.
  Do not retune this statistic's threshold or use coverage alone to open a
  Flow/Flow-Matching/Diffusion/DiT branch.
- Report:
  `skill_graph/experiments/2026-08-07_global_local_conditional_smoothness_stage_minus1/report.md`.

# 2026-08-07 — Few-shot AD under Distribution Shift 통합 실험 Ledger 추가

- 개별 experiment report를 유지한 채, Baseline1의 objective mismatch에서 시작해 paired loss,
  shift-conditioned Flow, semantic atlas, WTCF, latent geometry, global/local coherence audit으로
  이어지는 설계·실험·gate 흐름을 한 문서로 통합했다.
- 이후 mask-free conditional-smoothness audit도 별도 사전등록·완료됐지만 conditional gate가
  `0/4`로 실패했다. 따라서 `Baseline1` 유지 외 기존 branch retune, 새 module 구현, public
  benchmark 평가는 계속 차단 상태다.
- Document:
  `skill_graph/analysis/2026-08-07_fewshot_distribution_shift_integrated_experiment_ledger.md`.

# 2026-08-07 — Global/local factorisation assumption audit stopped support-only

- Completed the preregistered fixed-16, four-object global/local coherence
  Stage -1a0 audit on both local GPU 0 and dsba5 GPU 3 using an identical
  frozen snapshot, DINOv3 checkpoint, and support manifest.
- Global coherence exceeded local/broad defect coherence on all four objects,
  and shift+defect contrast retention stayed positive (`.819`--`.984` broad).
  The preregistered spatial-smoothness inequality held only for `fabric`;
  `fruit_jelly`, `walnuts`, and `vial` failed it, so the `>=3/4` factorisation
  gate was not met.
- All lane input ledgers matched and all aggregate metrics reproduced within
  `6.3e-7`; checksum manifests passed after relocating a local controller log
  that was appended after initial manifest creation. No public/private/test or
  ground-truth path was accessed.
- Verdict: `INCONCLUSIVE_OR_NOT_SUPPORTED / NO_NEW_MODULE / NO_PUBLIC_ACCESS`.
  후속 mask-free conditional-smoothness assay는 별도 기록에서 완료됐으며, 그 `0/4` 결과도
  Flow/Flow-Matching/Diffusion/DiT candidate 구현을 차단한다.
- Report: `skill_graph/experiments/2026-08-07_global_local_coherence_stage_minus1/report.md`.

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

# 2026-08-07 — Flow latent geometry blocked by a structural witness sentinel

# 2026-08-07 — Baseline1 k=5 operational scorer rejected

- Froze k=5 as a single operational candidate, not a method contribution or
  tunable sweep, and compared it with current K1 under identical H+ features,
  DVT, Flow, support, density term, and postprocessing.
- Completed all eight TESTpublic objects in two exit-0 local shards after the
  dsba5 shard was invalidated when orphaned v2/v3 experiment children overlapped
  the next run and caused OOM; terminated those children and removed the
  regenerable remote snapshot/result roots after verification.
- Added a float16 histogram-exact evaluator for F1/AP/pAUROC/AU-PRO and verified
  parity with the previous full-sort implementation on tied-score fixtures.
- K5-K1 macro pAUROC/F1/AU-PRO was
  `-0.004222/-0.002285/-0.003457`; `sheet_metal` AU-PRO fell `-0.030263`.
  The extreme good-pixel tail fell, but normal pooled FPR increased.
- Verdict: `RETAIN_K1 / K5_NOT_DEFAULT`; no further k sweep or per-object k.
- Report:
  `skill_graph/experiments/2026-08-07_baseline1_knn5_default_ab/report.md`.

# 2026-08-07 — Conditional normal-memory transport stopped

- Compared identity, the killed rank-1 projection control, query-median location
  normalization, and query-median/MAD diagonal transport on four fixed-support
  objects without test or label access.
- Validity and local GPU 0 / dsba5 GPU 3 parity passed with zero mismatches, but
  both conditional arms passed `0/4` objects.
- Location-only tail reduction was negative on three objects and only `+.054` on
  `fabric`; defect-induced conditioner drift was `.218--.807`, failing the `.10`
  limit everywhere.
- Verdict: `STOP_CONDITIONAL_ALIGNMENT_FAMILY` for naive global moment alignment;
  do not expand to Flow Matching/Diffusion/DiT until a defect-invariant condition
  identifiability gate exists.
- Report:
  `skill_graph/experiments/2026-08-07_conditional_normalization_transport_stage_minus1e/report.md`.

# 2026-08-07 — Spatial-consensus conditioner stopped

- Tested fixed `4x4` block consensus as a mask-free, query-only condition estimator
  against the Stage -1e naive median negative control.
- C0 remained strongly aligned with the photometric oracle direction, but C2 did
  not improve alignment and had `.51--.63` relative error.
- C2 local/broad defect contamination drift was `.65--3.95`; `walnuts/fabric`
  were worse than C0 and no object passed the joint gate (`0/4`).
- Local GPU 0/dsba5 GPU 3 parity passed with zero metric mismatches and zero
  public/private/GT access.
- Verdict: `KILL_FOR_CLAIM / NO_CONTINUE_SPATIAL_CONSENSUS`; do not tune block
  count/fraction. Isolate representation receptive-field spillover next.

- Ran a measurement-only Stage -1 JVP diagnostic (`identity` vs frozen `a6`,
  four direction families) over `can`, `fabric`, `fruit_jelly` x seeds `0,1,2`
  on dsba5 GPU 3, reusing the sealed WTCF manifests and folds unchanged.
- `can`/seed0 reproduced the recorded 2026-08-04 support-manifest SHA-256
  byte-for-byte; access custody, `identity_exact`, inverse error
  (`7.2e-07`-`9.5e-07`), and defect/witness orthogonality (`0.027`-`0.052`) all
  passed in every case.
- Witness scarcity proved structural, not object-specific: 5,832 candidates
  yielded six accepted witnesses, `8/9` cases below the minimum of three, so
  the separation margin `M` and hypothesis H1 are blocked.
- Exploratory observations, all without exception: `unsupported` exceeded
  `defect_proxy` in `3/3` measurable cases by `+0.034`-`+0.039`; `principal`
  contracted in `9/9` (`0.773`-`0.912`, at most `0.018` spread across seeds
  within an object) while every other family expanded; effective rank roughly
  doubled and latent RMS fell to `0.81`-`0.89`; principal curvature
  (`0.043`-`0.065`) exceeded defect curvature (`0.0057`-`0.0082`) by an order
  of magnitude.
- Four defects found and fixed during execution: a pre-existing container
  `torch 2.1.2` / `transformers 4.56.2` pytree regression that also breaks the
  WTCF preflight today; an uncapped `defect_proxy` family (~27,600 directions,
  ~10^7 JVP calls) that burned a 5h54m run producing no measurement
  (Amendment 2); a NaN serialization crash on the zero-witness sentinel that
  killed a nine-case run after three (Amendment 3); and a sentinel/schema
  mismatch caught pre-execution by direct indexing.
- Amendment 3 declared Stage B post-result and exploratory: the early-abort
  sentinel was suppressed so the witness distribution could be observed, and
  the remaining eight hashes were sealed before feature extraction. The H1
  verdict stays permanently blocked and cannot be revived by lowering the
  witness minimum.
- Report:
  `skill_graph/experiments/2026-08-06_flow_latent_geometry_stage_minus1/report.md`.
- Verdict: `INCONCLUSIVE_WITNESS_SENTINEL / H1_BLOCKED / EXPLORATORY_ONLY`.
  Next: measure the candidate pairwise-cosine distribution to separate a
  data property from the `0.85` criterion before relying on the interpretation;
  then preregister the witness-free `unsupported` vs `defect_proxy` contrast
  with `unsupported` decoupled from the witness count.

# 2026-08-08 — Frozen CLIP prior separability Stage -1i implemented

- Replaced the metadata auxiliary-variable branch with a support-only frozen
  visual-prior diagnostic. Metadata, filename conditions, split signals,
  `test_public`, `test_private`, and `ground_truth` are explicitly forbidden.
- Added a CLIP ViT-L/14@336 JIT patch-token extractor, DINOv3-H+ layer-7
  comparator, fixed `k=5` mean NN scoring, scale-normalized shift/core/selectivity
  metrics, geometry noncollapse audit, and strict/continuation gates.
- Local verification passed: py_compile, focused pytest (`6 passed`), and
  remote-controller `bash -n`.
- dsba5 CLIP cache was synchronized to
  `/workspace/.cache/clip/ViT-L-14-336px.pt` with SHA256
  `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`.
- Remote v3 completed with valid manifest/access/coverage/geometry and no
  metadata/filename/test/GT access. Strict and continuation gates were both
  `0/4`.
- CLIP reduced shift and improved defect gain on some texture objects
  (`walnuts`, `fabric`) but failed cross-object selectivity and worsened shift
  strongly on `fruit_jelly` and `vial`.
- Verdict: `KILL_FROZEN_CLIP_PRIOR / NO_CONTINUE`; no CLIP fusion,
  descriptor-swap, prompt-prior, or per-object prior follow-up.
- Report:
  `skill_graph/experiments/2026-08-08_frozen_clip_prior_separability_stage_minus1i/report.md`.
# 2026-08-08 — Baseline1 strength is representation-led, not Flow-led

- DINOv3-H+ `layer7/layer31/fused × raw/Flow` reduced-object factorial completed.
- Fused Flow added only `+.002099` pAUROC and reduced AU-PRO `-.003621`; the
  preregistered Flow and fusion×Flow synergy gates failed.
- Raw fused exceeded the raw single-layer mean, but raw layer 31 exceeded fused
  pAUROC/F1. Retain Baseline1 while attributing its strength primarily to the
  pretrained representation and the remaining DVT/scoring path, not Flow alone.
- Report: `skill_graph/experiments/2026-08-08_baseline1_mechanism_attribution/report.md`.
# 2026-08-08 — Baseline1 strength is DVT-led above DINOv3 representation

- Fused DINOv3-H+ `DVT off/on × raw/Flow` factorial completed on four objects.
- DVT main effect was `+.034675/+.069455/+.009686` on pAUROC/F1/AU-PRO;
  DVT×Flow pAUROC interaction was `+.006278`. Both preregistered gates passed.
- DVT+Flow nearly reproduced the same-four-object full Baseline1 K1 context,
  leaving density/scorer extras with negligible residual explanatory mass.
- Report: `skill_graph/experiments/2026-08-08_baseline1_dvt_flow_factorial/report.md`.

# 2026-08-09 — Two-band protected residual preserves medium signal but absorbs small anomalies

- Completed fixed B0/B1/B2 all-eight MVTec AD2 diagnostic with exact support-16, DINOv3-H+ fused
  features, three-epoch Flow, density off, and raw 1-NN.
- Validity, correction-energy/null, high-branch, coverage, custody, and local/dsba5 parity gates passed;
  4,740 duplicated-B0 metrics had maximum absolute delta `0.00027755`.
- B1 retained `96.44%/95.69%` of frozen current-DVT medium AP/top-5 gains and beat B2 on medium AP/top-5
  by `+.096389/+.106679`, confirming a position-specific medium-scale component.
- B1 nevertheless reduced small AP/top-1 by `-.019267/-.057738` versus B0 and AU-PRO by `-.060379`;
  `rice` medium AP and `sheet_metal` small top-1 triggered catastrophic no-harm failures.
- Verdict: `ANOMALY_ABSORPTION / NO_CONTINUE`. Do not retune the band split, sigma, scale, Flow, scorer,
  or per-object policy.
- Report: `skill_graph/experiments/2026-08-09_two_band_protected_residual_dvt/report.md`.

# 2026-08-09 — Baseline1-core official private anchor returned

- MVTec AD2 official server scored the frozen Baseline1-core DINOv3-H+/16 P16
  submission (8,180 files, checker `All checks successful`).
- private `AU-PRO .5440 / Class-F1 .7350 / Seg-F1 .4594`;
  private_mixed `AU-PRO .5314 / Class-F1 .7419 / Seg-F1 .4405`.
- Equal-budget reading: versus SuperAD-16 (the only other P16 row) Ours is
  `-.0124` private Seg-F1 and `+.0154` private-mixed Seg-F1, with a stronger
  backbone. No superiority claim at equal normal-data budget.
- Ours degrades least under mixed shift on all three metrics:
  Seg-F1 `-4.11%` versus SuperADD `-5.35%`, RoBiS `-8.78%`, SuperAD-16 `-9.90%`;
  AU-PRO `-2.32%` versus RoBiS `-11.30%`; Class-F1 does not degrade (`+.0069`).
  This is a two-point server observation, not yet a repeatable claim.
- Threshold transfer is not the bottleneck: public raw-oracle F1 `.530177` to
  private Seg-F1 `.4594` lost only `-.0708`, and the SuperADD F1 gap is nearly
  constant from public (`-.0959`) to private (`-.1148`). The `-0.318`
  normal-only calibration gap is specific to that separate path.
- Largest remaining absolute gap is now private AU-PRO versus RoBiS (`-.1285`),
  so region coherence supersedes thresholding as the improvement target.
- Verdict: `OFFICIAL_ANCHOR_OBTAINED / NOT_A_SUPERIORITY_CLAIM`.
- Report:
  `skill_graph/experiments/2026-08-09_baseline1_private_submission/report.md`.

# 2026-08-09 — `can` collapse is an operating-point failure, not signal absence

- Ran the preregistered three-way attribution: A1 support ladder `S16/S64/S128` (24 runs, local),
  A2 per-layer probe with DVT off and Flow identity (40 runs, dsba5 GPU3), A3 ordered stage path
  (8 objects, local), plus a duplicated A2 `fused` parity arm.
- All seven validity gates passed; parity max shared-metric delta `~3.6e-8`.
- Frozen rule in fixed order rejected all three hypotheses: H-B `can` best-layer AUROC `.8442`
  (needs `<.60`); H-C best-minus-deployed `+.0130` (needs `>=.10`); H-A S128-minus-S16 `-.0034`
  (needs `>=.10`). Verdict `INCONCLUSIVE`.
- Correction: `can` is not a no-signal object. Deployed full-range pixel AUROC is `.8312` and the
  best layer probe `.8442`. The earlier "near chance" reading used pAUROC@0.05 `.5734`.
  The collapse is confined to the low-FPR region and thresholded outputs
  (F1 `.0011`, component recall `.1458`).
- Eight times the normal references does not help and slightly hurts, consistent with `can` defects
  lying inside the normal manifold spanned by `train/good`.
- No pipeline stage loses the signal; `post_dvt_feature` gives `can` its best endpoints and cuts
  normal FPR `.00724 -> .00237`.
- `can` and `vial` hold the two highest normal FPRs and share the signature of adequate full-range
  ranking with low-FPR collapse. Association is descriptive, not causal.
- Custody deviations recorded: the 2026-08-08 A3 artifacts failed custody (`closefill_erode` versus
  contracted `none`) and were rerun; the preregistration's support hash was corrected from the
  DINOv2-L `same16` `7e93f1…` to the DINOv3-H+ Baseline1-core `9f223f7c…` before execution.
- Licenses no representation, support-budget, or signal-path work. Licenses only the separately
  preregistered measurement-only normal-tail protocol.
- Report: `skill_graph/experiments/2026-08-09_can_collapse_attribution/report.md`.

# 2026-08-10 — Region-completion map-level protocol invalidated by its own operator

- Ran anchors `raw`/`position_current`/`position_lowpass` plus `D0`/`D1`/`D2` locally and the
  `D3_rank_max_shuffled` hard null on dsba5 GPU3, all eight objects.
- Validity gates 1/2/5/6 passed; gates 3 and 4 failed and the verdict short-circuited to `INVALID`.
  Hard-null, strict, and continuation gates were never evaluated.
- Cause was a preregistration error, not an implementation defect. The protocol justified gate 3 with
  "rank normalization is strictly monotone per image", but every metric is pooled over an object's
  images via accumulated float16 histograms, so a per-image transform is not monotone on the pooled
  pixel set.
- Measured consequence: per-image rank normalization forces defect-free normal images to span the
  full range. `fabric` `seg_F1` fell `.668776 -> .057339` and `seg_AUROC_0.05` `.961722 -> .548660`,
  while `can` rose `+.0427` pAUROC and `+.0860` AU-PRO. `R` itself is correct and unit-tested.
- The preregistered "close the current-plus-low-pass family" clause does NOT fire: it is conditioned
  on a valid experiment reaching its endpoints, and this one never tested the hypothesis.
- `D0` is not a no-op control but a real per-image normalization arm. Its macro
  `gt_component_recall_at_oracle` rise `.4850 -> .6550` is an artifact of the same mechanism, not
  region recovery. Hypothesis-generating only; no confirmatory weight.
- Successor must normalize on a support/held-normal-fitted scale that stays comparable across images,
  and should be designed only after the normal-tail protocol reports.
- Report: `skill_graph/experiments/2026-08-09_region_completion_map_level/report.md`.

# 2026-08-10 — Normal tail is structured but not learnable from unshifted normals

- Ran the measurement-only normal-tail protocol: `public`/`fit` scoring and T1/T2/T3 statistics
  locally, permutation-null replication on dsba5 GPU3. Validity `6/6`; parity max delta `~1.4e-6`.
- G1 structure passed `8/8`. Six of eight objects concentrate `>70%` of tail mass in one decile of
  normal images; `can/fabric/vial/walnuts` beat their position-permutation null on `C_pos` by
  `+.3424/+.1789/+.4131/+.2794`.
- G2 predictability failed `1/8`. Only `vial` reached `rho>=.30` (`.4280`, null `-.0006`); every other
  object sat near zero. G3 was not evaluated per the stop-at-first-failure rule.
- Verdict `TAIL_NOT_IDENTIFIABLE_WITHOUT_TEST / NO_CONTINUE`.
- Likely cause is structural in the benchmark: `test_public` normals include `*_overexposed`,
  `*_shift_1`, `*_shift_2` variants while `train/good` contains none. The nuisance that lifts normal
  scores at test time is absent from the normals any module is fitted on.
- This supplies one coherent explanation for the eleven closed module branches: invariant channels,
  shared photometric direction, conditional normalization, spatial consensus, stable/positional
  retrieval keys, and band-split residuals were all fitted on data lacking the nuisance they targeted.
- Preregistration limitation recorded: no minimum-tail-mass eligibility criterion, so `rice` and
  `sheet_metal` (normal FPR `1e-5`) produced degenerate saturated statistics. Restricting to the three
  objects with a real tail (`can/fabric/vial`) still fails G2 at `1/3`; verdict unchanged and not re-read.
- Closes identify-then-suppress on normal data. Leaves open (a) constructions not requiring tail
  prediction, which is also the corrected-design requirement from the region-completion protocol, and
  (b) test-time adaptation, which the current R7 static-inference contract forbids.
- Report: `skill_graph/experiments/2026-08-09_normal_tail_structure_predictability/report.md`.

# 2026-08-10 — Research scope decision: retain the few-shot P16 constraint

- Decision: `P16` stays the headline claim. `Pfull` rows are additional scaling evidence, not a
  replacement, and no superiority claim is made at either budget.
- Grounding: Block A1 measured that 8x more real normals (`S16 -> S128`) moves full-range pixel
  AUROC by a control median of `+.0047` and `can` by `-.0034`. Few-shot is therefore not the binding
  constraint; the measured deficits are AU-PRO (`-.1285` vs RoBiS) and
  `gt_component_recall_at_oracle` (`.484`), which are region-completion problems and survive any
  budget change.
- Dropping the constraint would forfeit both distinctive results: public pAUROC parity with Pfull
  SuperADD (`.839561` vs `.839300`) and the smallest private-to-mixed degradation in the table
  (Seg-F1 `-4.11%` versus `-5.35%/-8.78%/-9.90%`). The current Pfull diagnostic pAUROC `.833807` is
  below the P16 result, so head-to-head at full budget has no supporting evidence.
- Recorded counter-reading to defend against: A1's null can be read as capacity limitation rather
  than efficiency. Mitigation is to fill the Main Table `Pfull` rows and show `Pfull ~= P16` for our
  method explicitly, rather than omitting them.
- Consequent priorities: (1) fill `Pfull` rows as scaling evidence; (2) define a public-split shift
  proxy so the robustness result becomes repeatable rather than a two-point server observation;
  (3) direct performance work at region completion.

# 2026-08-10 — The damaging AD2 nuisance is spatially non-uniform illumination, not global exposure

- Computed a public-split condition decomposition from `condition_attribution`, which every run
  already emits. No GPU was required.
- Normal false-positive rate at the global oracle threshold, macro over all eight objects:
  `regular .00126`, `photometric .00140` (`+11%`), `shift_* .00284` (`+125%`).
  Per object the `shift_*` inflation reaches `25x` (`walnuts`), `7.1x` (`fabric`), `2.8x` (`can`).
- Naming correction: the evaluator labels `_shift_*` as `geometric`, and that label is wrong.
  Phase-correlation against the matching `regular` image gives alignment offsets of `(0,0)` to `±1`
  pixel, i.e. noise level. The objects do not move.
- What actually changes is the spatial structure of illumination. Spatial coefficient of variation of
  the multiplicative field `img/regular`: `overexposed/underexposed` `.016--.061` (near uniform),
  `shift_1/2/3` `.044--.274`, typically `3--6x` higher. This matches the dataset's documented
  "additional light sources" condition.
- Consequence for the synthetic-shift bank experiment: `E1` augmented with a spatially **uniform**
  global exposure model `I * 2^lambda`. That covers the `+11%` benign axis and leaves the `+125%`
  damaging axis uncovered. The near-zero `E1` result (macro pAUROC `+.00036`) is therefore explained
  mechanistically rather than as a generic augmentation failure.
- The same reading applies to the RoBiS-style augmentation adopted from the literature (exposure plus
  Gaussian noise) and is consistent with its reported `+1.9%` F1-max.
- Adjacency noted, not treated as closure: Stage -1a1 conditional-smoothness gated `0/4`, but that
  attempted to identify a smooth nuisance field from data. Synthesising one is a different
  operationalization, per the identify-versus-synthesise distinction established by the normal-tail
  protocol.

# 2026-08-10 — Synthetic-shift bank expansion invalidated by gate scoping and a missing parity run

- Ran `E0` (bank 16) and `E1` (bank 112, global exposure ladder) locally and the energy-matched
  `E2` noise null on dsba5 GPU3, all eight objects. Flow was verifiably frozen: identical parameter
  checksum across arms and unchanged before versus after bank population.
- Aggregator returned `INVALID`. Validity gates 1/5/6 passed; 2/3/4/7/8 failed and the endpoint
  gates were never evaluated.
- Fault 1, mine: `S128_real` is a reused A1 ladder anchor whose Flow is fitted on 128 images and
  whose artifacts carry no `bank_size` / `flow_fit_source_view_count` / `flow_parameter_checksum`.
  The preregistration wrote gate 3 as "every arm reports a Flow fitted on exactly the 16 original
  support views", which that anchor cannot satisfy by construction. Every failure in gates 2/4/7
  names `S128_real` and nothing else. It should have been an external frozen comparator, custody
  checked only, as `raw`/`position_current`/`position_lowpass` were in the scale-selective protocol.
- Fault 2, mine: the preregistration required one arm duplicated across local and dsba5 for parity,
  and no duplicate was run. Gate 8 correctly failed.
- Descriptive measurement only, observed before the aggregator ran and carrying no verdict weight:
  macro `E1-E0` pAUROC `+.00036`, F1 `+.00054`, AU-PRO `+.00016`, normal FPR `-.00021`. `can`
  deployed normal FPR fell `.00562 -> .00405` (`-28%` relative) for `+.0014` pAUROC.
- Superseded regardless: the condition decomposition shows this protocol augmented the spatially
  uniform exposure axis (`+11%` normal-FPR inflation) and left the spatially structured `_shift_*`
  axis (`+125%`) uncovered.
- Corrective actions: the spatial-illumination successor was amended before its data was observed to
  scope arm-level gates correctly, and an `F1_spatial` parity duplicate was launched on the local
  RTX 4090 while the dsba3 arms were still running.
- Report: `skill_graph/experiments/2026-08-10_synthetic_shift_bank_expansion/report.md`.

# 2026-08-10 — Spatial-illumination bank expansion: INVALID on a gate formulation error, but the
# measurement is decisive

- Ran `F1_spatial` (dsba3 GPU2), `F2_highfreq_null` (dsba3 GPU3), and an `F1_spatial` parity
  duplicate on the local RTX 4090, all eight objects. Anchors `F0_baseline`, `E1_global`, `S128_real`
  reused after custody verification.
- Validity gates 1/2/4/5/6/7/8/9 passed. The arm-class amendment worked: `S128_real` cleared gate 7
  through the documented path-verification route with `train_good_only_verified=true`.
- Gate 3 failed and forced `INVALID`. Cause is mine: it demanded an identical Flow checksum across
  all contract arms, but the arms ran on two machines and the checksums partition exactly by machine
  (local `13c6209725bafd` for `can` across `F0`/`E1`/parity; dsba3 `9fe15d92cf1cd7` across `F1`/`F2`).
  Within every arm `checksum_before == checksum_after` and `flow_fit_source_view_count = 16`, so the
  substantive property the gate protects does hold, and gate 9 bounds the cross-machine difference at
  `7.17e-05` against a `1e-3` tolerance. Gate 3 should have been three conditions, not one byte-equality
  requirement. Corrected for future protocols; NOT amended retroactively, since endpoint data had
  already been seen.
- Measurement without verdict weight, from the parity-verified arm: macro shift-group
  `good_pixel_fpr_at_global_oracle` `.00284 -> .00274` = `-3.5%` against a `-20%` mechanism gate, and
  macro pAUROC `+.00069`. Among the three objects with a real tail: `can -9.6%`, `vial +7.3%` (worse),
  `fabric -4.3%`. `sheet_metal`'s `-41%` sits on `.00003 -> .00002` and carries no weight.
- Together with the predecessor's global exposure ladder (macro pAUROC `+.00036`), neither
  augmentation model moves the mechanism endpoint. The earlier null was not caused by the wrong
  augmentation model.
- Fourth independent normal-side manipulation that fails to transfer to deployed scores, after
  Stage -1b/-1c shared direction, the normal-tail predictability protocol, and the synthetic-shift
  bank. No closure clause fires (the protocol is INVALID), but the family is not worth continuing
  without a stated mechanism.
- Report: `skill_graph/experiments/2026-08-10_spatial_illumination_bank_expansion/report.md`.

# 2026-08-10 — Direct DINOv3 attention facets are spatially meaningful but not a safe hidden-feature replacement

- Ran the frozen Stage A assay on every GPU idle at submission: `can` on local GPU0, `fabric/rice`
  on dsba3 GPUs 2/3, and `walnuts` on dsba5 GPU3. All four immutable v2 shards completed, were
  checksum-verified after pull, and covered 600 full `test_public` images with fixed support-16.
- Validity passed with no private access, no query adaptation, GT opened only after all arm maps, and
  no dense maps retained. The first v1 submission was excluded because a reintroduced provisional
  `torchvision::nms` schema patch aborted before metrics; v2 used a new archive after regression repair.
- Every real facet failed. Best arm `V` changed macro pAUROC/F1/AU-PRO by
  `-.033298/-.059992/+.004786` versus the dimension-matched hidden control. It beat wrong-layer and
  spatial-roll nulls, showing real spatial signal, but this was insufficient for defect preservation.
- Object heterogeneity was decisive: V improved `can` and `walnuts`, while `fabric` lost
  `-.143666` pAUROC, `-.229968` F1, `-.055960` AU-PRO and `rice` lost `-.025233/-.011994/-.061032`.
  K/Q facets and the support-LOO equal ensemble were less stable.
- Verdict `KILL_FOR_CLAIM / NO_CONTINUE`. The sequential gate keeps Stage B V-V/QQ/KK propagation
  and Stage C mid-to-late routing closed. This adds another direct example that support-only evidence
  does not identify when a normal-side representation change will preserve defect evidence.
- Report: `skill_graph/experiments/2026-08-10_dino_attention_facets_stage_minus1/report.md`.

# 2026-08-11 — TCCR and DB-FSPC completed negative with dsba3 excluded

- Implemented compact TCCR v2 Stage -1 primitives and runner in new files only:
  `src/flow_tte/tccr.py`, `scripts/run_tccr_stage_minus1.py`, and tests.
- Implemented DB-FSPC Gate A parser/runner in new files only:
  `src/flow_tte/db_fspc.py`, `scripts/run_db_fspc_gate_a.py`, and tests.
- Post-run focused verification passed (`60 passed`) in addition to runner
  smoke phases, launch-time hashes, and `py_compile`.
- Allocation: local GPU 0 ran TCCR `can,fabric,vial`; dsba5 host GPU 3/container
  slot 0 ran TCCR `rice,can` parity and then DB-FSPC sequentially. `dsba3` was
  not accessed.
- TCCR completed `240/240` shard query evaluations. Its strict aggregate was
  `KILL_NO_CONTINUE`: phase shuffle matched TCCR on all four objects,
  preservation failed on `fabric/rice`, and duplicated-`can` parity max delta
  was `0.001957 > 0.001`. Descriptively, TCCR macro AU-PRO/pAUROC/F1 was
  `0.013620/0.491648/0.011798` versus P16
  `0.552158/0.806379/0.448470`.
- DB-FSPC completed all `336/336` Qwen3-VL-2B calls under vLLM 0.11.0. The
  runtime gates passed (`OOM=0`, peak `13,428 MiB`, p95 `27.731 s`), but only
  `29/272` non-null responses parsed and stability, hflip geometry,
  whole-object fallback, and text-only null controls failed. Verdict:
  `GATE_A_FAIL / NO_GATE_B`.
- Both pulled manifests preserve the intended custody boundaries. No dense
  anomaly map existed; immutable dsba5 source snapshots were removed after
  verified pull, and compact evidence was retained.
- Reports:
  `skill_graph/experiments/2026-08-11_tccr_stage_minus1/report.md` and
  `skill_graph/experiments/2026-08-11_db_fspc_gate_a/report.md`.

# 2026-08-11 — Support-Orbit Velocity Mismatch Stage -1 designed

- Reframed the failed SO-CFM field from a support-correction operator into a
  differential reference: squared disagreement between normal velocity and an
  affine support-to-observation geometric velocity is the candidate evidence.
- Froze a four-object, support-16, leave-one-image-out Stage M with all eight
  signed nuisance codes, true-code/same-source oracle ceilings, raw/orbit/flow
  endpoint baselines, and constant/velocity/code/spatial/time hard nulls.
- The primary score does not transform the query, transport the support, use a
  filename condition, or update parameters/memory at inference. It therefore
  does not re-enter the killed query-view persistence operator.
- Synthetic anomaly learning is sequential: only surviving Stage M evidence
  licenses Stage G's CutPaste/chromatic-blotch ranking loss, which is evaluated
  on disjoint thin-scratch/broad-blur families and must beat a mask-shuffle
  training null. A pre-fit activation rate blocks another zero-loss guard.
- Status `PRE-REGISTERED / NOT RUN`; no GPU, public/private/GT, or result access.
- Preregistration:
  `skill_graph/experiments/2026-08-11_support_orbit_velocity_mismatch_stage_minus1/preregistration.md`.

# 2026-08-11 — Support-Orbit Velocity Mismatch Stage -1 completed

- The merged aggregate is valid and final: `validity_ok=true`,
  `benchmark_comparable=false`, `strict=false`, and `continuation=false`.
- `M0_sovm_enum` reached macro pAUROC `0.674283` with local/broad
  `0.741302/0.607263` and normal FPR `0.045711`.
- `B1` stayed strictly negative
  (`[-0.005158,-0.000449]`); `N3_spatial_source_shuffle` was the only strong
  positive control at `+0.145896` with 4/4 wins.
- Engineering fixes that stabilized the run were exact parity benchmarking on
  GPU-resident data, clean cache handling, remote tar/stdin/same-file transfer,
  deterministic blur no-op fallback, and the aggregate ratio contract. These
  are execution enablers, not method evidence.
- Supplementary dsba3 GPU 3 `vial` and dsba5 GPU 3 `walnuts` parity runs stay
  in progress and are not needed for the primary gate.
- Report:
  `experiments/2026-08-11_support_orbit_velocity_mismatch_stage_minus1/report.md`.

# 2026-08-12 — SPCT confidence/calibration/TTA Stage -1 completed

- All eight paired Probe16/full-support shards completed valid on dsba3/dsba5
  GPU3 and were merged with 10,000-draw hierarchical bootstrap evidence.
- Protocol verdict `VALID_NO_SIGNAL`: LOO confidence collapsed to raw retrieval,
  hierarchical conformal was overconservative, the synthetic gate was matched
  by its nulls, and key-only TTA raised FPR while losing to a wrong-part null.
- Full support was the surviving diagnostic: B0/B1 pAUROC improved by
  `+0.064629/+0.080122`, both 4/4 wins with positive lower bounds.
- Research verdict `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`. Under the updated
  performance-first objective, full-support B1 is promoted directly to an
  all-eight TESTpublic run against a matched full-memory baseline; no support64
  monotonicity experiment is required.
- Report:
  `experiments/2026-08-12_spct_confidence_calibration_tta_stage_minus1/report.md`.

# 2026-08-12 — RN-anchor gated local tangent TESTpublic preregistered

- Frozen all-eight TESTpublic public-shadow aggregate contract before result
  intake. Required arms are `A0_RNFMLK_final_layer_1nn_exact_anchor`,
  `A1_RN_anchor_unconditional_local_tangent`,
  `A2_support_confidence_gated_tangent`, `A3_capped_gated_tangent`, and
  `N0_part_shuffled_tangent_control`; optional `O0_oracle` is excluded from
  claims.
- `A0_RNFMLK_final_layer_1nn_exact_anchor` is mandatory hard null.
  `A3_capped_gated_tangent` preserves exact RN anchor/distance and admits only
  support-LOO bounded multiplicative correction `d_rn*(1-0.25*gate)`, with
  tangent magnitude fusion forbidden.
- Strict claim requires `A3>A0` on macro pAUROC, F1, and AUPRO; continuation
  requires A1 over prior SPCT and A2/A3 closing the RN gap without object
  pAUROC harm worse than `0.02`.
- SuperAD16 remains budget-incomparable context unless support budget matches.
- Preregistration:
  `experiments/2026-08-12_rn_anchor_gated_local_tangent_testpublic/preregistration.md`.

# 2026-08-14 — RNC-PA design loop completed

- Ran a bounded five-stage deep-research/autoresearch loop using completed TQRN,
  SPCT, retrieval, attention-facet, and region-completion evidence plus primary
  dense-correspondence/partial-OT literature.
- Closed residual-witness rarity as the headline because of tangent/tail
  calibration proximity; reciprocal-only and field-smoothness remain controls.
- Promoted RNC-PA at design score `0.87`: exact RN top-5 edges, fixed 3x3 query
  region, source-grouped one-to-one partial assignment, +/-2 relative translation,
  and explicit support-LOO dustbin.
- Verdict `DESIGN_STAGE / CONDITIONAL_KEEP`; no implementation or GPU result yet.
- Analysis:
  `analysis/2026-08-14_rn_preserving_neighborhood_partial_assignment.md`.

# 2026-08-15 — CF-GC-RN Stage -1 launched

- Replaced the failed additive RNC-PA regional score with leave-center-out
  geometry-conditioned selection of one RN reference; the final score remains
  that selected candidate's normalized descriptor distance.
- Pre-registered `rice/vial`, full `train/good`, complete TESTpublic, same-run
  RN1, offset/neighborhood hard nulls, and reduced-object diagnostic gates.
- Verified 38 focused tests plus lint/compile/shell/smoke and launched immutable
  run `geometry_conditioned_rn_rice_vial_stage1_v1_20260815` on dsba5 GPU3
  (`rice`) and dsba3 GPU3 (`vial`).
- Report:
  `experiments/2026-08-15_cross_fitted_geometry_conditioned_rn_stage_minus1/report.md`.

# 2026-08-15 — Simple-RN representative + DVT TESTpublic prepared

- Re-centered the next experiment on the strongest simple exact RN direction:
  no TTA, no query updates, no score fusion.
- Canonical aggregate `scripts/aggregate_simple_rn_testpublic.py` now expects
  nine arms including representative K subsets, random64, full/R64 DVT, and
  `N_DVT_R64_position_shuffle_RN`.
- Added remote controller
  `scripts/run_simple_rn_representative_dvt_testpublic_remote.sh` using
  canonical runner `scripts/run_simple_rn_testpublic.py`.
- Experiment notes:
  `experiments/2026-08-15_simple_rn_representative_dvt_testpublic/report.md`.

# 2026-08-15 — Patch-Coverage RN TESTpublic ops prepared

- Promoted the next constrained diagnostic from image-level support selection
  to 16x16 spatial patch-memory coverage while keeping exact normalized RN
  scoring unchanged.
- Added aggregate/remote ops contracts for protocol
  `patch-coverage-rn-testpublic-20260815`, schema
  `patch-coverage-rn-object-v1`, and arms `B0/P25/P50/P75/N50`.
- Remote wrapper keeps dsba3 `can,fabric,walnuts`, dsba5
  `fruit_jelly,rice,sheet_metal,vial,wallplugs`, and host GPU3 to
  container `cuda:0`; `TARGET` preset routing is guarded for all remote modes.
- Experiment notes:
  `experiments/2026-08-15_patch_coverage_rn_testpublic/report.md`.
# 2026-08-16 — Protected Residual-Consensus RN Stage -1 launched

- Recovered CF-GC-RN v2 as `KILL_FOR_CLAIM / NO_CONTINUE` and froze a distinct
  feature-residual, rather than spatial-offset, nuisance-code diagnostic.
- Implemented and verified the eight-arm PRC-RN core/pipeline; 38 focused and
  adjacent tests passed with lint/compile/shell/diff checks.
- Submitted immutable rice/vial run
  `protected_residual_consensus_rn_rice_vial_stage1_v1_20260816` on dsba5/dsba3
  host GPU3 after exact GPU UUID and resource preflight.
- Both shards completed with compact-only custody. Final aggregate is
  `KILL_FOR_CLAIM / NO_CONTINUE`: M1-B0 macro pAUROC/F1/AU-PRO
  `-0.000028/-0.000042/-0.000005`, conditional excess AUC `0.487202`.
- Rice used exact RN1 fallback throughout; vial's residual-guided switches
  slightly harmed RN1. Close this selector and retain exact RN1.
# 2026-08-16 — Fundamental canonical transport design audit

- Promoted FACT-RN only to design-stage `REVISE` (`0.58`), not to a method claim.
- The architecture changes the comparison space before retrieval and requires offline intervention learning; support-only projection/gating remains closed.
- Analysis: `skill_graph/analysis/2026-08-16_fundamental_canonical_transport_architecture.md`.
# 2026-08-24 — ATAC-v1 dsba3 support-validity replication completed

- Ran immutable can support-16 only on dsba3 GPU0 in `hun_fsad_tta_012`.
- Exact support hash and no-peek custody passed; local excursion fatal replicated
  within `3.41e-7`, and strict fp32 linear-control identity also failed.
- Aggregate: `KILL / FATAL_SUPPORT_GATE`; research verdict
  `KILL_FOR_CLAIM / NO_CONTINUE`. No public metrics and no dense maps.
- Report: `experiments/2026-08-24_atac_stage_minus1/report.md`.

# 2026-08-24 — FACT-RN Stage -1 closed at support-only sentinel

- Revalidated the immutable `can` shard with exact SuperAD-16 custody and no
  TESTpublic/private/GT access.
- FACT-RN achieved only `0.6156%` q99 reduction and `1.7426%` nuisance
  correction; the part-frame shuffle matched the full arm within `3.13e-7` on
  nuisance correction.
- Applied the preregistered fatal shuffled-null gate before remote expansion.
  Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`; no dense maps or public run.
- Report:
  `experiments/2026-08-17_factorized_canonical_transport_rn_stage_minus1/report.md`.

# 2026-08-24 — CIDER-CF synthetic-counterfactual mechanism designed

- Ran a five-stage deep-research/adversarial loop after closing FACT-RN.
- Rejected direct synthetic-mask classification and pooled synthetic feature
  selection as shortcut-prone prior-art retreads.
- Promoted CIDER-CF at design score `0.88`: synthetic families act as
  counterfactual environments defining a nuisance-orthogonal covariance
  alternative; inference uses one cross-fitted conformalized LLR.
- Frozen leave-target/leave-generator custody, generator-ID, mask, boundary,
  normal-transform, source/part, pooled, and random-spectrum fatal nulls.
- Verdict: `DESIGN_STAGE / CONDITIONAL_KEEP`; only a four-object support-only
  Stage -1 is licensed next.
- Design: `analysis/2026-08-24_cider_synthetic_counterfactual_design.md`.

# 2026-08-24 — MR-JW Stage -1 mechanism refuted

- Froze and implemented a support-only joint multi-resolution witness assay
  with independent-scale, single-scale/raw, duplicate, spatial-pair shuffle,
  and support-identity shuffle controls.
- Completed immutable dsba3 run
  `mr_jw_stage_minus1_dsba3_all4_20260824_v1` on GPUs 0--3 with exact
  support-16 custody and no public/private/GT access.
- MR-JW lost to independent scale NN on all four objects (macro pAUC margin
  `-0.015474`) and to the rowwise best shuffle (`-0.004657`); clean LOO
  stability failed on can, vial, and walnuts.
- Corrected only preregistered verdict precedence in aggregate-v2; the result
  is `MECHANISM_REFUTED / NO_CONTINUE`. Compact artifacts are retained and no
  `anomaly_maps/` directory remains.
- Report: `experiments/2026-08-24_mr_jw_stage_minus1/report.md`.

# 2026-08-25 — CUG-RN-S1 support-only panel completed

- Implemented the shift-pair-only rank-8 PSD meta-update, exact hard-RN
  evaluation, matched nulls, fail-closed custody, and sequential dsba3
  controller under protocol `cug-rn-s1-stage-minus1-support-only-20260825`.
- Immutable v3 passed the GPU0 can validity sentinel, then ran
  can/fabric/vial/walnuts on GPUs 0--3. All four objects were valid and
  safe-positive, but none passed full causal activity.
- M0 failed matched-null normal-tail superiority on 41/48 rows and held-family
  superiority on 47/48; topology-null completion was 0/12 on can and walnuts.
- Verdict: `NEAR_MISS / ONE_BOUNDED_ITERATION`; all-eight/public/GT stayed
  locked. Local reaggregation and snapshot hashes matched, and no dense maps
  remain.
- Report: `experiments/2026-08-25_cug_rn_stage_minus1/report.md`.

# 2026-08-25 — CUG-RN-S2-MTP direct margin/topology retry closed

- Implemented the isolated global-RN control/candidate path, direct matched-null
  margin objective, raw-neighbor and log-scale constraints, independent
  duplicate gate, exact panel scope, complete custody validation, and pinned
  dsba3 host key. Adjacent S1/S2 verification passed 31 tests plus critical
  lint, compile, shell, and smoke gates; an independent verifier returned
  `READY` before launch.
- Immutable run `cug_rn_s2_mtp_stage_minus1_dsba3_seq_20260825_v1` passed the
  GPU0 can validity sentinel, then used dsba3 GPUs 0--3 for
  can/fabric/vial/walnuts and stopped after the support-only panel.
- Panel validity was `4/4` and `48/48` rows. Safe-positive held in `34/48`, but
  normal/anomaly matched margins were each `0/48`, topology was `0/48`, and
  full activity was `0/48`; source fits also satisfied none of the four joint
  constraints.
- Remote compact hashes passed, local reaggregation was byte-identical, all
  forbidden-access flags stayed false, no dense maps existed, and GPUs 0--3
  returned idle. Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`; close the learned
  CUG PSD-update branch and require source-only RN-isometric feasibility before
  any new adaptation design.
- Report:
  `experiments/2026-08-25_cug_rn_s2_mtp_stage_minus1/report.md`.

# 2026-08-25 — Source-only shift/anomaly residual separation literature audit

- Surveyed primary work on shared/private encoders, weak paired/group
  disentanglement, invariant/equivariant splits, linear concept erasure,
  covariance contrast, cross-environment objectives, paired clean restoration,
  and factorized synthetic anomaly generation.
- Ran a bounded adversarial design loop. CRLP, protected erasure, and a crossed
  interaction gate scored `0.64`, `0.59`, and `0.46`; the loop stopped after
  two non-improvements.
- No method was promoted. The only bounded next action is a source-only
  CRLP-Predict diagnostic with separate decodability and cross-fitted RN
  actionability gates under joint held object/shift/generator folds.
- Verdict: `DESIGN_STAGE / REVISE`; no dataset/public/test/private/GT access and
  no experiment metric claim.
- Analysis:
  `analysis/2026-08-25_source_only_shift_anomaly_residual_separation_literature.md`.

# 2026-08-26 — TARCD-LoRA-S1 single-seed preregistration

- Closed the support-internal split leakage risk: fit uses exact target
  support-16, while evaluation uses four support-disjoint `train/good` bases
  whose full descendant trees are sealed.
- Froze one seed, one held generator, three shift cells, fused-qkv rank-4 LoRA,
  canonical KD versus RN-plus-margin auxiliary training, matched controls, and
  a `can/fabric/vial/walnuts` support-only panel on dsba3 GPUs 0--3.
- TESTpublic/private/GT remain locked; even a full pass is diagnostic only.
- Preregistration:
  `experiments/2026-08-26_tarcd_lora_stage_minus1/preregistration.md`.

# 2026-08-26 — TARCD-LoRA-S1 single-seed result

- Completed the immutable support-adapted run
  `tarcd_lora_s1_stage_minus1_dsba3_20260825T161712Z_v4`: GPU0 `can`
  validity sentinel followed by `can/fabric/vial/walnuts` on dsba3 GPUs 0--3.
- All `4/4` objects and `12/12` held cells were valid, but paired-correction
  retention, complete preservation, and full cell success were each `0/12`;
  object mechanism success was `0/4`.
- M1 median correction cosine was `0.0667` (maximum `0.5125`) versus the
  frozen `0.70` floor. M1 normal-tail reduction was lower than canonical KD in
  all `12/12` cells; median object-level q99 reduction was only `3.381%`.
- Candidate M1 repeated byte-identically between the `can` sentinel and panel,
  but M0 and shuffled-control adapter hashes did not, adding a whole-control
  repeatability failure. No seed/hyperparameter sweep is permitted.
- Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`. All public/test/private/GT/query
  flags remained false, compact results were independently reaggregated, and
  zero dense `anomaly_maps/` were retained.
- Report:
  `experiments/2026-08-26_tarcd_lora_stage_minus1/report.md`.

# 2026-08-26 — TARCD-LoRA-S2-DCR single-seed preregistration

- Froze a separately named successor after S1 correction retention failed
  `0/12`; no S1 seed/weight/rank/threshold tuning is reopened.
- Replaced endpoint-plus-orthogonality optimization with direct paired clean
  and anomaly corrections, anomaly cross-recomposition, and slack-only RN
  topology guards. Added matched clean-only, synthetic, RN-guard, shuffled,
  random, static, and paired-oracle arms.
- Fixed one seed (`20260826`), exact support16 custody, four support-disjoint
  held bases, held generator, three shifts, and a dsba3 GPU0--3 four-object
  support-only panel. Public/test/private/GT remain locked.
- Preregistration:
  `experiments/2026-08-26_tarcd_lora_s2_dcr_stage_minus1/preregistration.md`.

# 2026-08-26 — TARCD-LoRA-S2-DCR single-seed result

- Completed immutable run
  `tarcd_lora_s2_dcr_stage_minus1_dsba3_20260826T013826Z_single_seed`: GPU0
  `can` validity sentinel followed by `can/fabric/vial/walnuts` on dsba3 GPUs
  0--3, with exactly seed `20260826`.
- All `4/4` objects and `12/12` held cells were valid, but candidate correction
  cosine and hard-null superiority each passed `0/12`; complete cells were
  `0/12` and object mechanism passes `0/4`.
- Direct correction improved S1 median correction cosine from `0.0667` to
  `0.3250` and norm ratio from `0.3435` to `0.7886`, but still missed the
  absolute direction gate. Synthetic recomposition and RN-guard contribution
  passed only `4/12` and `7/12`; the static-mean correction was the best hard
  null in `10/12` cells.
- Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`. Compact hashes passed, local
  reaggregation was byte-identical, forbidden-access flags remained false,
  no dense maps were retained, and GPUs 0--3 returned idle.
- Report:
  `experiments/2026-08-26_tarcd_lora_s2_dcr_stage_minus1/report.md`.

# 2026-08-26 — TARCD-LoRA-S3-VR sequential repair preregistration

- Froze one change only after S2: target-norm normalized vector Huber replaces
  the zero-start singular direction/log-scale correction objective.
- Preserved S2 randomization, LoRA, optimizer, RN, synthetic generators,
  controls, evaluation gates, one seed, and exact support/held custody.
- Locked public/test/private/GT access. GPU0 `can` may authorize only a valid
  four-object support-only panel on dsba3 GPUs 0--3.
- Preregistration:
  `experiments/2026-08-26_tarcd_lora_s3_vr_stage_minus1/preregistration.md`.

# 2026-08-26 — TARCD-LoRA-S3-VR result

- The preregistered sequential GPU0 `can` run stopped before panel expansion:
  scientific metrics repeated exactly, but D0/D2/M0 adapter bytes did not.
- A separately marked `observational_nonclaim` run actively used dsba3 GPUs
  0--3 for the fixed `can/fabric/vial/walnuts` panel from the same immutable
  snapshot. All `4/4` objects and `12/12` cells were valid.
- Normalized vector Huber fixed S2's conditioning pathology: the maximum first
  D2 gradient ratio versus S2 was `7.59e-7`, below the `0.01` ceiling.
- Held behavior still failed. Correction cosine improved from `0.3250` to
  `0.4187` but missed the `0.4250` continuation floor; the median hard-null
  margin was `-0.1440`, complete cells were `0/12`, synthetic contribution was
  `0/12`, and object mechanism passes were `0/4`.
- Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`. Independent local reaggregation was
  byte-identical to the remote aggregate; forbidden-access flags remained
  false, no dense maps were retained, and GPUs 0--3 returned idle.
- Report:
  `experiments/2026-08-26_tarcd_lora_s3_vr_stage_minus1/report.md`.

# 2026-08-26 — SCOPE-LoRA Cycle 1 projector feasibility result

- Completed immutable remote run
  `scope_lora_cycle1_dsba3_20260826T1442Z_v3` on dsba3 GPUs 0--3 after two
  result-free launch-path failures were isolated and repaired.
- All `4/4` objects and `12/12` cells passed structural/no-op/custody checks.
  Rank-4 held explained energy had aggregate median `0.387690`, and PT-P
  reduced the normal q99 by `+2.6555%` in aggregate.
- The preregistered anomaly gate failed on `can/vial`, so P1 LoRA was not
  launched.  Algebraic audit showed that the same projected correction cancels
  from that residual, making the failure a frozen shift/anomaly commutation
  failure rather than evidence of projector-induced harm.
- Verdict: `INSUFFICIENT_PROJECTOR_SIGNAL / CONTINUE_ONE_DIAGNOSTIC`;
  benchmark comparison and public/private/test/GT access remain closed.
- Report:
  `experiments/2026-08-26_scope_lora_cycle1/report.md`.

# 2026-08-26 — SCOPE-LoRA Cycle 2 CEB preregistration

- Froze one sequential repair before Cycle-2 results: a calibrated excess
  bypass `S_CEB=max(S_PT,S_frozen-tau_family)` fitted only on the four
  calibration-support shifted-normal descendants.
- Kept the Cycle-1 rank-4 projector, paired teacher, exact RN memory, held G2
  generator, four objects, three shifts, seed, and custody unchanged.
- Added score-level held-mask pAUROC, normal-tail/activation guards, spatial
  shuffle, max-stream, and projected static-mean hard nulls.  P1 remains closed
  unless every preregistered protection, normal-utility, and static-value gate
  passes.
- Preregistration:
  `experiments/2026-08-26_scope_lora_cycle2_ceb/preregistration.md`.

# 2026-08-27 — SCOPE-LoRA Cycle 2 CEB result

- Completed all four support-only objects and twelve held cells on dsba3 GPUs
  0--3 with structural, custody, normal-utility, and synthetic-protection gates
  intact.
- The calibrated bypass reduced shifted-normal q99 by `+0.0265548`, but added
  exactly `0.0` pAUROC over PT-P, did not beat the spatial null, and did not
  establish value over the static/max-stream controls.
- Verdict: `INSUFFICIENT_PROTECTED_SCORE_SIGNAL / END_OUTER_LOOP`. The q95
  threshold retry was not opened because two science gates failed together.
- Report:
  `experiments/2026-08-26_scope_lora_cycle2_ceb/report.md`.

# 2026-08-27 — Oracle-Family-Expert LoRA Stage -1

- Preregistered a separately named oracle upper-bound diagnostic: three
  independent rank-4 DINOv2 fused-QKV LoRA experts for J0/J1/J2, each trained
  for exactly 48 updates with S3/S2 objectives, randomization, custody, and
  controls retained.
- Completed immutable run
  `oracle_family_expert_lora_stage_minus1_dsba3_20260826T1617Z_v2` on the
  project-owned dsba3 container and GPUs 0--3. Remote focused tests, offline
  DINO loading, actual CUDA LoRA smoke, all 12 task statuses, hashes, and
  forbidden-access checks passed.
- Global hard-null margin improved from S3's `-0.1439931` to `-0.0832106`,
  q99 reduction was `+0.0102168`, RN contribution passed `8/12`, positive
  hard-null margin appeared in `3/12`, and strict synthetic contribution
  appeared for the first time in `1/12`.
- The anomaly object gate failed on `can/vial`; only `7/12` cells retained the
  complete anomaly guard, and synthetic contribution missed the progressive
  `3/12` floor. Verdict:
  `INSUFFICIENT_ORACLE_FAMILY_SIGNAL / END_FAMILY_ROUTING_BRANCH`.
- This kills family-routed/hypernetwork LoRA and any family/rank/LR/threshold/
  loss-weight/seed sweep as the next mechanism. A separately preregistered
  deployable change must protect frozen anomaly evidence during actual QKV
  adaptation rather than add another score bypass.
- Report:
  `experiments/2026-08-27_oracle_family_expert_lora_stage_minus1/report.md`.

# 2026-08-27 — Protected Asymmetric-PCGrad QKV-LoRA Stage -1

- Completed immutable dsba3 run
  `protected_pcgrad_qkv_lora_stage_minus1_dsba3_20260827Tpcgrad01Z_v2` on the
  fixed project container with `can/fabric/vial/walnuts -> GPU0/1/2/3`.
- Preflight, actual CUDA QKV-LoRA smoke, all four object summaries, exact
  `12/12` cells, custody hashes, fixed 144-step traces, and support-only
  aggregation passed. No other user's resource was inspected or touched.
- P2 had zero conflict steps on every object, so protected PCGrad collapsed to
  the D2 scalar-sum baseline. The shuffled control produced
  `88/94/52/85` conflicts and protected all actual AdamW updates, confirming
  that the implementation worked but the treatment was inactive.
- Global correction cosine/norm were `0.3664246/0.3416666`, hard-null margin
  was `-0.1441973`, q99 reduction was `+0.0187851`, anomaly guards were
  `7/12`, and synthetic/PCGrad contribution were both `0/12`.
- Verdict:
  `INSUFFICIENT_PROTECTED_PCGRAD_SIGNAL / END_SYNTHETIC_QKV_LORA_BRANCH`.
  Close QKV-LoRA optimizer/update/rank/block/LR/loss/seed sweeps and require a
  separately named representation architecture for any continuation.
- Report:
  `experiments/2026-08-27_protected_pcgrad_qkv_lora_stage_minus1/report.md`.

# 2026-08-27 — CRFR-S1 crossed factorization result

- Completed immutable run
  `crfr_s1_stage_minus1_dsba3_20260827Tcrfr01Z_v3` in the fixed project
  container with an exact-repeat can gate followed by the four-object dsba3
  GPU0--3 panel. Scientific metrics repeated with maximum absolute difference
  `0.0` and the M1 model-state SHA-256 matched exactly.
- All `12/12` held cells and custody checks passed structurally, but causal
  factorization passed `0/12`, anomaly isolation `2/12`, synthetic
  contribution `1/12`, and positive hard-null margin `0/12`.
- Shuffling the nuisance code changed held reconstruction error by only about
  `-3.4e-5` to `+3.6e-5` relative, showing that the residual recomposer ignored
  code identity. Global correction cosine/norm were `0.05855/0.39573`, global
  hard-null margin was `-0.39564`, and q99 reduction was `+0.04609`.
- Verdict:
  `INSUFFICIENT_CRFR_SIGNAL / END_SUPPORT_ONLY_SYNTHETIC_REPRESENTATION_BRANCH`.
  Close CRFR architecture/loss/hyperparameter tuning; any continuation must
  be a separately named architecture with structurally identifiable nuisance
  action and unchanged support-only gates.
- Report:
  `experiments/2026-08-27_crfr_s1_stage_minus1/report.md`.

# 2026-08-27 — SIAI-S1 J0 true-action oracle result

- Completed support-only `can x J0` sentinel
  `siai_s1_j0_oracle_stage_minus1_dsba3_20260827_siai03_v1` on the fixed
  project dsba3 container and GPU0 only; `20/20` remote tests and all source,
  controller, custody, compact-output, and exact-repeat checks passed.
- The analytic inverse beat the stronger static/wrong-sign null by correction
  cosine `+0.07628`, improved pixel MAE over wrong sign by `81.24%`, and
  preserved RN rank/neighbor/scale at `0.98859/0.93652/1.01385`.
- It nevertheless missed the core gates: correction cosine `0.55529` and
  anomaly cosine `0.79451` versus `0.90`. Positive-EV bases saturated about
  `30%` of pixels with MAE `6.7--6.8`; negative-EV bases had no saturation and
  MAE about `0.35`, isolating irreversible clipping as the next diagnostic.
- Verdict: `INSUFFICIENT_SIAI_SIGNAL / END_SIAI_S1`. Do not run SIAI-S1 J2 or
  fit its estimator; any continuation must be a separately named posthoc
  sign/saturation diagnostic before backbone training.
- Report:
  `experiments/2026-08-27_siai_s1_j0_oracle_stage_minus1/report.md`.

# 2026-08-27 — SIAI-D1 J0 sign/saturation diagnostic result

- Completed exact-repeat support-only `can x J0` diagnostic
  `siai_d1_j0_sign_diagnostic_stage_minus1_dsba3_20260827_d1sign01_v2` on
  dsba3's fixed project container and GPU0 after `17/17` remote tests.
- Negative EV restored normal-feature correction and topology: correction
  cosine/norm `0.90997/0.96159`, RN rank/neighbor/scale
  `0.99859/0.99609/1.00142`, and hard-null margin `+0.26792`.
- The clipping mechanism was strongly sign-asymmetric: negative/positive
  median reconstruction MAE `0.34973/6.76342`, zero versus about `30%`
  saturation, and correction-cosine gap `+0.72696`.
- Negative-EV anomaly cosine still reached only `0.80972` versus `0.90`, the
  sole failed reversible-signal component. Gates:
  `SATURATION_ASYMMETRY=true`, `NEGATIVE_EV_REVERSIBLE_SIGNAL=false`.
- Verdict: `NO_REVERSIBLE_SIGN_SIGNAL`. Stop explicit pixel inversion; the
  next separately named sentinel must test anomaly-preserving feature
  commutation without reviving closed LoRA/CRFR/score-bypass branches.
- Scientific repeat max diff `0.0`; signature
  `7b4da4230ab33097afb8bab3aa3cffc517c4ff3ecc80d27854c7734946a1e001`.
  The immutable snapshot was removed and GPU0 returned idle; no other user's
  resource was enumerated or touched.
- Report:
  `experiments/2026-08-27_siai_d1_j0_sign_diagnostic_stage_minus1/report.md`.

# 2026-08-27 — SAEC-LoRA-S1 shared-action commutation result

- Completed immutable support-only run
  `saec_lora_stage_minus1_dsba3_20260827T1549KST_v2` in the fixed dsba3
  project container. The exact `can` repeat had scientific metric max
  difference `0.0` and identical combined model-state SHA-256 before the
  `can/fabric/vial/walnuts` GPU0--3 panel expanded.
- The LoRA path was active: all effective adapter norms were nonzero and M1
  reduced training loss on three objects. M1 also beat the shuffled N1 held
  relation error on `4/4`, by `4.24%--33.47%`.
- The treatment was not identifiable. Matched N2 no-commutation beat M1 on
  relation error on all four objects; macro relation gain was `-0.003369`,
  M1-minus-N2 commutation improvement never exceeded `0.000426`, action
  identification was `0/4`, and synthetic contribution was `0/16`.
- Base-ID shortcut accuracy exceeded the `0.35` ceiling on all four objects.
  Anomaly protection passed only `fabric/walnuts`; `can/vial` had severe
  q99/RN harm under several actions.
- Verdict: `NO_IDENTIFIABLE_ACTION / KILL_FOR_CLAIM / NO_CONTINUE`. Close the
  SAEC relation without rank/LR/block/seed/loss/action sweeps. Any successor
  must be a distinct separated architecture with untouched frozen anomaly
  evidence, not a retuned shared commutation arm.
- Remote and independent aggregate SHA-256 matched exactly at
  `f85b2d36d9d7d127c8aac0ee569234ad65385cd3d2fd35f5229b13ff526011c6`;
  forbidden ledgers were empty, no dense maps/checkpoints were retained, and
  dsba3 GPUs 0--3 returned idle.
- Report:
  `experiments/2026-08-27_saec_lora_stage_minus1/report.md`.

## 2026-09-08 — SRA-AD2-ALL8-SAME16-S3-v1

Design frozen for one cumulative shift-residual matrix:
A0 normalized full residual, A1 residual branch, A2 tangent coordinates,
A3 gate auxiliary. Scope is all eight MVTec AD2 objects and 1084
test_public good/bad queries, three paired seeds, exact HAS16 references,
240 logical feature-view slots / 192 unique views, and 576 updates per arm.
Same-condition SuperAD-16 and last-layer same-HAS16 RN16 are independent
comparators. Documentation status is IMPLEMENTING; source checks, training,
metrics, and remote launch are pending. Runtime auditor owns dsba1 host
GPU [0,2,4] worker/UUID mapping.

## SRA latest execution binding — 2026-09-08

The initial matrix is single-seed 20260906 only: A0-A3, 32 heads,
all-eight TESTpublic, 1084 queries. Multi-seed work is optional follow-up.
Reuse existing SuperAD-16 and RN-FMLK artifacts first; no automatic baseline
rerun. Reference/split/preprocessing/metric mismatches are comparable=false
and do not block candidate execution. Single-seed SD is null, not zero.
GPU hosts remain dsba1 [0,2,4].

### 2026-09-08 — shift-residual ablation integration check

Documentation state is INTEGRATION_CHECK. The frozen initial matrix remains
single seed 20260906, A0-A3, 32 heads, eight objects, and 1084 TESTpublic
queries. Existing SuperAD-16 and RN-FMLK outputs are reused from
baseline_reuse_manifest.json; baseline rerun is not automatic. Because
historic baselines are FP32 and candidate streaming is FP16, candidate versus
historic rows are comparable=false; historical_pair_valid is tracked
separately for a provenance-matched historical pair. RUNNING awaits explicit
READY/launch evidence. No training or evaluation has started according to this
log entry.

### 2026-09-08 — deployment checkpoint

State is DEPLOYING after critic READY and successful local preflight. Remote
approval exists, but no launch receipt/PID exists yet; RUNNING and started
training/evaluation are intentionally not recorded. The evaluator contract is
querywise: prediction -> GT metric update -> immediate feature/map discard.
Inventory may be complete for scheduling, but there is no global prediction
barrier followed by a second metric pass. Training and selection remain
ground-truth-free.

### 2026-09-08 — run_v2 launched

Runtime status is RUNNING for feature extraction only; metrics and finished
heads are not yet present. Receipt:
local results/remote_runs/dsba1/sra_ablation_20260908/run_v2/launch_receipt.json,
SHA-256
d6a24e01806e80bf76df6550ded4b8bde230e81b162407047af1252982838fc0.
Remote root is /workspace/results_remote/sra_ablation_20260908/run_v2.
Immutable source snapshot:
/workspace/source_snapshots/sra_ablation_20260908_v2_final,
SHA-256
4e26bf2ae6536ed970257d0322317e82bf867b4c61e94f128c1f29000d9bb8db.
Controller 2196, workers 2199/2200/2201, and Python workers 2206/2207/2208
are alive; can/fabric/fruit_jelly are assigned to host GPUs 0/2/4. Local
tests, remote tiny calls, and strict preflight passed. v1 exited because
scripts/run_cider_cf_stage_minus1.py was missing from packaging; logs remain.
v2 includes canonical closure/source manifest. Baseline reuse and the
FP32-historic/FP16-streaming caveat remain; dense map accumulation is false.

### 2026-09-12 — GARD completion and dsba3 environment-only setup

G1/G2 completed 16/16 local class runs, 23:55:01–00:27:03 KST (32.04 min),
no recorded failures. AUPRO@0.05 x100: G1 46.71704, G2 46.32650; G2-G1
-0.39054pp. Neither improves D3 or F3. Updated the canonical comparison CSV
and `results/comparisons.md` using the existing collector; no reevaluation.
F1/G1/G2 support/query/GT content, backbone provenance and environment match
on all 8 classes. Interpretations remain seed0 seen-development findings.

SSH recovered. Reused the existing dsba3 container `hun_fsad_tta_012` and
retained all legacy code/results/environment. Copied local `/usr/local` into
`/workspace/.local-runtime/python311-cu124-036828932804` (original local image
036828932804...), matching Python 3.11.16 and all ten pinned package versions.
Remote CUDA tensor smoke passed on assigned GPU 2, all four GPU UUIDs are
visible, and 56 method/CLI/output/runner tests passed. No remote experiment.
The OS and GPU remain Ubuntu 22.04/A6000 rather than Debian/4090.

Local-authoritative `update-code` mode now activates separate SHA-256 snapshots
under the existing execution lock and skips unchanged transfers. At validation:
source snapshot 7598eb3889c781babd0ba456d9fb9cbb37dbc530bd4228453d981fe02521b8f3;
FMAD package hash 061247e5673dd6d9665d2e89fe816df55641d5bf439d57d1377f784c76111730.
52 local remote-helper regression tests passed, including no-op updates,
secret exclusions, removed-file isolation, archive corruption and lock refusal.

Data relative-name/size inventory SHA-256 matched:
c9956b2713f18f67218503838e9b095191a6a0fe5b65f8c07c4f3b46126961d5
(not a full byte-content audit). DINOv2 repository SHA-256 matched:
8e82773220ebd38c121c4b5d0f5c57b2ee009c82cf76c26efe306513379d2a18;
ViT-L/14 checkpoint file SHA-256 matched:
d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428.
Existing cache/data were reused without downloads. `/data/mvtec_ad_2`,
`/model-cache`, `/results` aliases provide local-compatible paths; output goes
to the separately mounted `results_local_compatible` tree. Private dsba3
preset selects the isolated runtime and four UUIDs. Metrics-only pull is ready;
logs/provenance remain remote. No checkpoint/map export, cleanup or Git push.

### 2026-09-12 — single-focus H1 controller correction

User narrowed work to one core issue. Selected G1's delayed/zero preservation
coefficient, not synthetic confidence, contrast weighting, or other failure
hypotheses. Added opt-in `immediate_preservation` to the existing relational
method; H1 config differs from G1 only by this flag. It computes the detached
current-step coefficient as clip(P/b, 0.1, 1.0), with no EMA-dual history.
Default G1/G2/F1–F3 execution paths remain unchanged. No new parameters,
forward passes, output maps or checkpoints.

Local dedicated container: 50 tests passed across relational-tail,
tail-contrast, and output-policy tests. Tests cover zero-lambda recovery,
current-step response, detached controller gradients, one-key config matching,
incompatible settings, and tiny-backbone integration. Ruff is unavailable in
the local venv. This verifies the control behavior, not anomaly-detection
performance or strict budget satisfaction. H1 benchmark not launched; the
next bounded comparison is G1 versus H1 under the same seen-development
conditions, without new components or a hyperparameter sweep.

### 2026-09-12 — G1/H1 launched on local + dsba3 GPU 2 only

User authorized experiment execution using the local GPU and only dsba3
physical GPU 2, superseding the earlier four-GPU allocation. Both target
devices were idle before launch. Kept the existing containers, code and
method configuration; `update-code` verified the unchanged snapshot
`87244c8a01bac3fa87eea04dec32deed05f4f8672033b18627bf0adcf743d08d` without transfer.
Preflight confirmed exactly one CUDA-visible device in each selected process.

Launched two detached instances of `scripts/experiments.py run`, each with
`--configs configs/methods/relational_tail_G1.yaml configs/methods/relational_tail_H1.yaml
--labels G1 H1 --data-root /data/mvtec_ad_2 --device cuda:0 --workers 2
--split test_public_seen_development --timeout 3600`.

- Local GPU UUID `GPU-0070b668-714f-bcbb-6c46-f0b68c9b7f6b`, container
  `fsad-tta-local-bcma`: `--objects can fabric fruit_jelly rice`,
  `--output-dir /results/preservation_h1_20260912_local`.
  Controller host PID 941299; initial CLI PIDs 941327/941328.
  Host output root: `results/local_runs/bcma_docker_20260911/preservation_h1_20260912_local`.
- dsba3 GPU UUID `GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639` (physical index 2),
  container `hun_fsad_tta_012`: `--objects sheet_metal vial wallplugs walnuts`,
  `--output-dir /results/preservation_h1_20260912_dsba3_gpu2`.
  Controller host PID 2695702; initial CLI PIDs 2695705/2695706.
  Source is the immutable release with the snapshot hash above; the copied
  local Python 3.11/cu124 runtime is selected explicitly. The wrapper holds
  the remote experiment/update lock for its entire lifetime.

Both status files reported running after launch (start epoch
1789142374.9071796 local / 1789142375.3299682 remote). Each method pair is on
the same host/GPU per class; aggregate full8 is a declared mixed-hardware
comparison. Seed0/normal16/448+672/64-step settings and streaming evaluator
unchanged. Max 16 class runs total; no sweep. Primary AUPRO@0.05, with vial
and preservation behavior diagnostic only. Verify completed artifact and
support/evaluator matches before metric claims. Maps and weights remain off.
Remote logs/configs/manifests stay remote; retrieve metrics only. No unrelated
GPU jobs, legacy results, or containers were modified or stopped.

Post-launch GPU verification: local GPU utilization 96%, 4946 MiB allocated;
dsba3 physical GPU 2 utilization 100%, 4665 MiB allocated. Remote CLI PIDs
2695705/2695706 were both listed on the authorized UUID. Initial per-class
logs contained the existing nonfatal xFormers-unavailable warnings; no final
metrics or completed class runs had been reported at this check.

### 2026-09-12 — H1 partial result collection (01:20:52 KST)

Read-only result check; no experiment launch, restart, stop or code change.
Local queue completed 8/8 at 01:12:57 KST (13.37145 minutes), no recorded
failures. Four-class G1/H1 AUPRO@0.05 x100 = 40.38215/39.72151
(-0.66064pp); pAUROC = 76.36384/75.25288; oracle F1 = 35.13830/33.72991.
H1-G1 class deltas: can -2.12269pp, fabric +0.58752pp,
fruit_jelly -0.95248pp, rice -0.15492pp. All four pairs match code,
support/query/GT content, backbone, environment and trainable parameters;
only the intended immediate-preservation flag and output path differ.

dsba3 GPU 2 still runs G1 sheet_metal/walnuts; G1 vial/wallplugs have final
manifest/metrics, H1 is pending. Artifact completion count 10/16 overall;
parallel status completes its list at the stage barrier, not per finished
worker. Remote vial/wallplugs AUPRO x100 = 62.50249/23.68718. No paired
remote or full8 claim. Final check confirmed only these two remote manifests
and active GPU 2 utilization (77%, 5269 MiB).

Used the existing collector for the local completed scope and updated the
same canonical Markdown/CSV with the remote unpaired partial rows. Normal
metrics-only rsync encountered host permission denied on container-owned
metrics files; read those completed metric JSONs through docker exec instead.
No chmod, full-log/config/manifest download, map/weight copy or experiment
interruption. Verified 12 unique new ledger rows: 10 class metrics match
their saved artifacts and two local4 macros match arithmetic aggregation.
No remote macro or full8 number created. The local subset shows no gain;
vial repair and overall H1 verdict remain unresolved pending the remote run.

### 2026-09-12 — H1 full8 completion, three independent methods, new launch

Recovered all remote H1/G1 completed metrics through the existing container
(host metrics-only rsync cannot read some container-owned files). No full
logs/configs/manifests downloaded. Original local queue ended 01:12:57 KST;
remote ended 01:42:56 KST, exit 0. 16/16 class runs, no recorded failures.
Full8 AUPRO@0.05 x100 G1 46.71464 -> H1 47.42062 (+0.70598pp),
pAUROC 76.75832 -> 76.54791. H1 improves 3/8 and declines on 5/8; vial
+8.25242pp dominates the aggregate gain. Vial final support P/b falls from
4.16909 to 0.67699. All eight pairs match content/code/backbone/environment
and parameters within class. Canonical CSV partial rows replaced by 16
completed class rows and 2 full8 macros; metrics/means/identities verified.
The old local4-only negative result was not the final method verdict.

User requested three distinct designs, implementation and experiments.
Added I1 selective-preservation (normal bulk only, teacher tail free),
I2 paired-margin (same-location worst-view synthetic uplift with bounded
teacher-shift margin), I3 anchored-retrieval (frozen cosine reference index,
adapted distance). All independently extend H1, with config switches that
recover H1 when disabled. Added one preservation-term method hook, registry
entries, three method files/configs and a shared targeted test module; no
new queue/ledger script or shared fitting/evaluation change. Brief published
prior-work sanity check and failure predictions are recorded in
`configs/methods/THREE_DIRECTIONS.md`; no novelty claim.

85 targeted tests passed locally (62 method/output + 23 runner/CLI/pipeline/
results); the same 85 passed remotely in 14.07s. Exact replay vs full-graph
gradients, enabled/disabled tiny-backbone integration, frozen backbone,
parameter parity, target detach/bounds, tail selection, anchor selection,
registry and config differences verified. Existing containers reused.
Activated source snapshot:
bedcfcda2cf3963ff5105eac9f81bf4faae721176dc9deea59266f523bd5b960.
No Git commit/push, model/data download or container recreation.

At 06:22:21 KST launched `scripts/experiments.py run` with labels H1 I1 I2 I3
and corresponding relational_tail_H1/selective_preservation_I1/
paired_margin_I2/anchored_retrieval_I3 YAML configs. Common arguments:
`--data-root /data/mvtec_ad_2 --device cuda:0 --workers 2
--split test_public_seen_development --timeout 3600`.

- Local `fsad-tta-local-bcma`, GPU-0070b668-714f-bcbb-6c46-f0b68c9b7f6b:
  can fabric fruit_jelly rice sheet_metal walnuts;
  `/results/three_directions_20260912_local`, controller PID 1394801,
  initial CLI PIDs 1394829/1394830. Host artifact parent remains
  `results/local_runs/bcma_docker_20260911/`.
- dsba3 `hun_fsad_tta_012`, physical GPU 2 only,
  GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639: vial wallplugs;
  `/results/three_directions_20260912_dsba3_gpu2`, controller PID 2748298,
  initial CLI PIDs 2748301/2748302. Immutable release above; isolated copied
  local Python/cu124 runtime. Holds remote experiment/update lock throughout.

Both status files report running and both CLI pairs have GPU allocations
on their assigned UUIDs. H1 control runs first; all three candidates follow
automatically, not concurrently with their same-class baseline. 32 total
class runs, seed0/normal16/64 steps/448+672. Primary full8 AUPRO@0.05;
mixed hardware declared, same-class methods stay on one device. No new
candidate metric or performance claim yet. Dense maps/weights off; logs and
provenance retained remotely, metric-only recovery. Unrelated jobs untouched.
