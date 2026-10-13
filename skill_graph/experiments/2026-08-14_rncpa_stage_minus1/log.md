# RNC-PA Stage -1 Log

## 2026-08-14

- Converted the promoted RNC-PA design into a frozen reduced-object TESTpublic
  mechanism protocol.
- Fixed the RN lower-bound, regional partial-assignment operator, four
  structural controls, support-only dustbin calibration, custody barrier, and
  strict/continuation/kill gates before metric execution.
- Implemented the exact-RN lower-bound, coherent regional assignment core,
  seven-arm runner, fail-closed aggregator, and GPU3 remote controller.
- Verification passed: 40 focused tests, Ruff, basedpyright, py_compile,
  shell syntax, scoped diff check, local manifest smoke, and CUDA parity.
- `rnc_pa_stage_minus1_v1_20260814` stopped before any shard because the
  controller forwarded the obsolete `mechanism` phase. No result was retained.
- `rnc_pa_stage_minus1_v2_20260814` stopped before any shard because the shared
  backbone returned layers 5/11/23 while the runner expected a single layer.
  The runner now selects frozen native layer 23 explicitly. No result was
  retained.
- Submitted immutable execution `rnc_pa_stage_minus1_v3_20260814`:
  dsba3 GPU3 runs `can -> walnuts`, and dsba5 GPU3 runs `rice -> vial`.
  Both host/container GPU UUID checks passed. Initial logs reached
  `can: support final features 1/412` and
  `rice: support final features 1/313`; both shards are running.
- Partial immutable v3 completion at 2026-08-14 11:30:30 UTC: dsba5
  completed valid compact shards for `rice` and `vial` and returned GPU3.
  `rice` took 8109.97 s and `vial` 7512.24 s. The dsba3 `can` runner
  remained active after 4 h 35 min with all 412 support features extracted;
  `walnuts` had not started. No OOM, traceback, or failed marker was present.
- Provisional metrics, not an aggregate verdict: `rice` M0 versus B0 was
  0.724881 versus 0.914328 pAUROC and 0.106183 versus 0.624266 F1;
  `vial` was 0.637201 versus 0.656234 pAUROC and 0.256179 versus 0.280986 F1.
  M0 conditional invalidity AUC was 0.531637 for `rice` and 0.727172 for
  `vial`. Full keep/kill evaluation remains blocked on `can` and `walnuts`.
- Partial immutable v3 completion at 2026-08-14 12:36:41 UTC: dsba3
  completed a valid `can` shard in 19599.21 s. M0 versus B0 was 0.532220
  versus 0.589333 pAUROC and 0.000379 versus 0.001415 F1; M0 conditional
  invalidity AUC was 0.608762. `walnuts` then started and completed all 432
  support feature extractions; after 3 h 28 min it remained actively scoring
  with no failed marker. Three of four immutable shards are complete.
