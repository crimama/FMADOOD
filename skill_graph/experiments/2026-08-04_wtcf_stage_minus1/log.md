# WTCF Stage -1 Execution Log

## 2026-08-04

- Froze the support-only five-object, three-seed, nine-arm contract before the
  valid run.
- Implemented WTCF core, focused tests, single-case runner, validity-first
  aggregation, and dsba5 GPU-3 fixed-container controller.
- Invalidated v1--v5 for infrastructure or pre-result implementation defects;
  none is used as scientific evidence.
- Completed valid v6 `can/seed0` sentinel in `hun_fsad_tta_012`, with host GPU
  3 exposed as container `cuda:0`.
- Passed access, finite-score, proxy, geometry, witness, and training checks;
  failed directional JVP because witness ratios remained `.9236--.9493`
  instead of reaching `<=.85`.
- Stopped automatically before the remaining 14 cases and before any
  public/private/test/ground-truth access.
- Pulled compact v2--v6 evidence, removed all isolated v1--v6 snapshots and
  invalid v1--v5 remote results, and verified GPU 3 idle.
- Final verdict: `EARLY_VALIDITY_KILL / NO_FULL_STAGE / NO_PUBLIC_ACCESS`.

