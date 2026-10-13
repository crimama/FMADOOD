# Execution Log

- 2026-08-12: SPCT result and adjacent negative evidence reviewed.
- 2026-08-12: Four-lane continuation frozen: confidence gate, hierarchical
  conformal calibration, synthetic activation gate, and isolated key-only TTA.
- 2026-08-12: Both support budgets and all arms remain mandatory regardless of
  early gate outcomes.
- 2026-08-12 00:38 KST: Local CUDA preflight passed on GPU0. Support-only and
  no-test/no-GT custody, manifest hashes, runner/aggregator/controller hashes,
  and compact-output policy were recorded.
- 2026-08-12 00:40 KST: Final verification passed: 47 focused tests, Ruff,
  py_compile, bash syntax, CUDA smoke, and independent read-only audit.
- 2026-08-12 00:40 KST: Submitted immutable run
  `spct_confidence_stage_minus1_all14_v1`. dsba3 controller PID `321739`
  owns `vial -> walnuts`; dsba5 controller PID `71846` owns
  `fabric -> can`. Each object runs `support16 -> full_train_good` and every
  shard evaluates the frozen 14-arm panel.
- 2026-08-12 00:41 KST: Both controllers report `running_or_pending`, with no
  failure marker. dsba3 host GPU3 was idle at submission; dsba5 host GPU3 is
  active in first-shard feature extraction.
- 2026-08-12 08:43 KST: v1 audit found zero completed shards after 8 hours.
  Both exact child processes were alive at approximately 100% CPU, GPU
  utilization was 0%, and no log progress existed after initialization.
- 2026-08-12 08:47 KST: Root cause isolated to per-query O(N^2) conformal
  self-coverage recomputation and TN0 per-part scalar CUDA synchronization.
- 2026-08-12 08:57 KST: Semantics-preserving repair completed: fold-prepared
  conformal references, searchsorted tail counts, vectorized TN0 energy
  matching, T0-energy reuse, calibration-pass reuse, capped synthetic-view
  precompute, and flushed progress markers. Verification: 53 tests, lint,
  compile, shell syntax, local smoke, slow-reference parity, and independent
  review all passed.
- 2026-08-12 08:59 KST: Stopped only the two v1 runner/controller pairs after
  command-line identity verification; retained their logs and wrote
  `aborted_cpu_hotspot.txt`. Submitted immutable replacement run
  `spct_confidence_stage_minus1_all14_v2` with identical queues, data,
  manifests, arms, metrics, and gates.
- 2026-08-12 09:08 KST: First v2 shard `fabric/support16` completed valid with
  held-source exclusion and zero query/test access. B0/B1/G0/S0 pAUROC were
  `0.638016/0.638216/0.638016/0.638016`; C0 was `0.597213` with zero FPR;
  T0/TN0 were `0.637646/0.636746` with FPR `0.089568/0.091060`; forbidden O0
  was `0.645708`. These are one-object intermediate diagnostics only. The
  controller advanced to `fabric/full_train_good`.
- 2026-08-12: All eight v2 shards completed and were checksum-verified after
  pull. Both remote GPU3 queues returned idle with no failure marker.
- 2026-08-12: The first strict merge rejected 657 constant per-image maps,
  all from the discrete `CN0_global_conformal` null. Each CN0 arm remained
  globally finite/nonconstant (13–65 distinct scores). The validator was
  corrected to enforce non-collapse at arm/shard scope while preserving every
  row-level finite and custody check; 35 focused tests, Ruff, compile, and
  shell checks passed.
- 2026-08-12: Final merge passed `validity_ok=true` with protocol verdict
  `VALID_NO_SIGNAL`. G0, C0, S0, and T0 all failed their independent gates.
  Full support nevertheless improved B0/B1 by `+0.064629/+0.080122` pAUROC,
  both with 4/4 wins and positive 10,000-draw bootstrap lower bounds.
- 2026-08-12: Removed both regenerable remote source snapshots after verified
  pull. No dense result tree existed; compact shards, logs, checksums, and the
  merged summary were retained.
