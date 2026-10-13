# Protected Residual-Consensus RN Stage -1 Log

- 2026-08-16: Recovered the completed CF-GC-RN v2 aggregate.  It lost to RN1
  by `-0.000180` pAUROC and `-0.001261` AU-PRO; conditional excess AUC was
  `0.518845`, so the branch is `KILL_FOR_CLAIM / NO_CONTINUE`.
- 2026-08-16: Frozen the distinct feature-residual consensus diagnostic before
  implementation.
- 2026-08-16: Implemented the core, runner, strict aggregate, remote controller,
  and focused tests. Removed an import-time inherited-module mutation exposed
  by the adjacent regression suite; final local verification was 38 passed.
- 2026-08-16: Submitted immutable v1 after both GPU3 UUID/data/hash/resource
  preflights: dsba5 rice PID 126583, dsba3 vial PID 383230.
- 2026-08-16: Both shards completed and were pulled compact-only.  Fixed two
  aggregate-only contract defects (retired geometry gate reuse and an OR/AND
  continuation error) with regression coverage; no experiment artifact was
  modified.  Final verification is 40 focused/adjacent tests passed.
- 2026-08-16: Final aggregate verdict `KILL_FOR_CLAIM / NO_CONTINUE`.
  M1-B0 macro pAUROC/F1/AU-PRO is
  `-0.000028/-0.000042/-0.000005`; conditional excess AUC is `0.487202`.
  Rice exactly falls back to RN1 and vial is slightly worse than RN1.
- 2026-08-16: Verified zero dense-map persistence, retained compact local and
  remote results, and removed both regenerable remote isolated code snapshots.
