# CF-GC-RN Stage -1 Log

- 2026-08-15: Pre-registered the rice/vial reduced-object diagnostic.
- 2026-08-15: Confirmed dsba3 GPU3 and dsba5 GPU3 idle before implementation.
- 2026-08-15: Added vectorized CF-GC-RN core, compact TESTpublic runner,
  fail-closed aggregate, and two-host GPU3 execution wrapper.
- 2026-08-15: Passed 38 focused tests, lint, compile, shell syntax, local
  sealed-manifest smoke, and both remote preflights.
- 2026-08-15: Launched immutable run
  `geometry_conditioned_rn_rice_vial_stage1_v1_20260815`: dsba5 GPU3 rice,
  dsba3 GPU3 vial.
- 2026-08-15: The v1 computation reached result emission on both hosts, then
  failed strict JSON serialization because an expected no-consensus diagnostic
  used an infinity sentinel. Replaced the sentinel with JSON `null`, added
  regression coverage, and passed 43 focused tests plus lint/compile/shell
  checks.
- 2026-08-15: Launched corrected immutable run
  `geometry_conditioned_rn_rice_vial_stage1_v2_20260815`: dsba5 GPU3 rice,
  dsba3 GPU3 vial.
- 2026-08-15: Deleted 96 regenerable `anomaly_maps/` directories from old
  2026-07-10--14 dsba3/dsba5 result roots after confirming the active run was
  excluded. Reclaimed 34.962 GiB; compact metrics/manifests/summaries, datasets,
  model caches, and private-submission artifacts were preserved. Local free
  space increased from 78 GiB to 113 GiB.
