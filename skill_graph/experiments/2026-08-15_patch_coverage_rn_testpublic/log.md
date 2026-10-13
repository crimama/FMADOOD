# Patch-Coverage RN TESTpublic Log

## 2026-08-15

- Registered protocol `patch-coverage-rn-testpublic-20260815`.
- Fixed arm panel:
  `B0_full_RN`, `P25_cell_kcenter_RN`, `P50_cell_kcenter_RN`,
  `P75_cell_kcenter_RN`, `N50_cell_random_RN`.
- Fixed selector grid `16x16`.
- Pinned full support manifest
  `ac084290bd40d171f12e876b45c3baf9b2269010cc2e62a0e91d7c4b80badd53`.
- Remote ownership: dsba3 `can,fabric,walnuts`; dsba5
  `fruit_jelly,rice,sheet_metal,vial,wallplugs`; host GPU3 maps to
  container `cuda:0`.
- Added wrapper regression coverage for `TARGET=dsba3|dsba5` preset routing
  across all remote modes.
- Integrated the source-per-cell selector into the runner. Incremental
  farthest-first updates preserve exact ordering while avoiding cubic repeated
  distance computation; selected memories contain only native patch rows.
- Passed 37 focused/adjacent tests, scoped ruff, compile, shell syntax, diff
  checks, and a sealed-manifest smoke.
- Stopped a zero-shard duplicate invocation that was incorrectly using the old
  Simple-RN+DVT runner under a patch-memory run name.
- A concurrent read-only verification lane incorrectly submitted duplicate
  `patchcoverage` and `spatial_patch_memory_v3` runs. No duplicate completed a
  shard. Their exact snapshot-scoped Python processes were terminated.
- Fixed remote `stop` so it terminates the controller and any object runner
  rooted in that immutable run snapshot instead of orphaning the child.
- Passed 24 focused tests after the stop-path regression update.
- Launched immutable run `patch_coverage_rn_testpublic_20260815_v2` on both
  canonical GPU3 queues after fresh idle, UUID, memory, and manifest preflight.
  Both controllers later received external SIGTERM at the same second while
  the client session was closed; no shard completed and no traceback/OOM was
  present.
- Added `setsid` detachment to the remote launch path and a regression test;
  the focused suite now passes 25 tests.
- Launched replacement immutable run
  `patch_coverage_rn_testpublic_20260815_v4_detached`. On both servers the
  controller has `PPID=1` and its own SID/PGID, with exactly one first-object
  runner and no failure marker.
- Completed and pulled all eight shards: dsba3 3/3 and dsba5 5/5.
- Corrected an aggregate-only contract bug: P50 and N50 intentionally have
  equal per-cell counts, so the hard-null check now compares selection identity
  SHA rather than rejecting equal quotas. Aggregate tests pass 14/14.
- Valid aggregate: full RN ranks first at macro pAUROC `0.749400`; random N50
  is second at `0.749247`; P50 k-center is third at `0.749121`.
- Final verdict: `NO_SIGNAL_DIAGNOSTIC`. K-center does not beat its equal-budget
  random null; retain full RN for performance and random N50 only as an
  efficiency option.
