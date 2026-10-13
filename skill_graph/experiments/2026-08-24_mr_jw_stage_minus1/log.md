# MR-JW Stage -1 log

- 2026-08-24: froze protocol `mr-jw-stage-minus1-20260824` before code or
  results. Public/private/test/GT access remains closed; synthetics are sealed
  evaluation-only interventions.
- 2026-08-24: froze dsba3 mapping `can:0`, `fabric:1`, `vial:2`, `walnuts:3`
  and the fixed container `hun_fsad_tta_012`.
- 2026-08-24: first remote preflight stopped before launch because the cache
  probe lacked snapshot `PYTHONPATH`. Added only the missing preflight
  environment; no shard, model result, or gate summary had started.
- 2026-08-24: launched immutable run
  `mr_jw_stage_minus1_dsba3_all4_20260824_v1` after successful second
  preflight. Repaired a local read-only status-format string after launch; the
  detached snapshot, controller, scorer, and results were unchanged.
- 2026-08-24: all four shards and initial aggregate completed. The initial
  aggregate labeled the result `KILL_FOR_CLAIM` because clean-LOO stability
  was fatal, but both primary margins were also non-positive. Corrected only
  the preregistered verdict precedence in aggregate schema v2 so direct
  refutation is labeled `MECHANISM_REFUTED`; rows, gates, and metrics are
  unchanged, and the original remote aggregate remains retained.
- 2026-08-24: local aggregate-v2 replay recorded
  M0-independent `-0.015474`, M0-best-shuffle `-0.004657`, zero object
  direction wins, and clean-LOO stability failure on three objects. Final
  verdict `MECHANISM_REFUTED / NO_CONTINUE`; remote and local dense-map audits
  returned no paths.
