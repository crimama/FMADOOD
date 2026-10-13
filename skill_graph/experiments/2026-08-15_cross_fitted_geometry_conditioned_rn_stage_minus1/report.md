# CF-GC-RN Stage -1 Report

Date: 2026-08-15
Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`

## Negative Evidence Intake

RNC-PA's additive regional penalty lost to RN1 on all three completed objects
and inflated the normal tail.  Offset and neighborhood nulls nevertheless
showed that local geometry retained a small conditional signal.  CF-GC-RN
tests that signal as a reference-selection mechanism rather than another score
fusion.

## Motivation

Determine whether neighbor-only spatial coherence can select a better normal
reference while retaining RN1's descriptor discrimination.

## Implementable Design

See `preregistration.md`.  The run is limited to rice and vial and includes
same-run RN1 plus five causal controls.

## Evaluation Alignment

Candidate and RN1 are matched on support, backbone, split, and evaluator.
SuperAD-16 is budget-incomparable context only.

## Code Modification / Creation

Implemented a new vectorized CF-GC-RN core, TESTpublic runner, strict aggregate,
two-host remote wrapper, and focused tests without modifying the killed RNC-PA
implementation.

## Added Code Evaluation

Passed 38 focused tests, ruff, Python compilation, shell syntax, diff checks,
the sealed full-support smoke, and dsba3/dsba5 remote preflights.

## Remote Execution

Corrected immutable run
`geometry_conditioned_rn_rice_vial_stage1_v2_20260815` completed: dsba5 host
GPU3 ran rice and dsba3 host GPU3 ran vial. Both host/container GPU UUID checks
passed. The compact aggregate is stored under
`results/remote_runs/geometry_conditioned_rn_rice_vial_stage1/aggregate/geometry_conditioned_rn_rice_vial_stage1_v2_20260815/aggregate.json`.

## SuperAD Baseline and Unified Metrics

Same-condition RN1 is the hard baseline.  `superad_*` fields are N/A with
`comparable=false` for this reduced-object full-support diagnostic.

## Results and Analysis

The valid reduced-object macro result was:

| Arm | pAUROC@0.05 | F1 | AU-PRO@0.05 | Normal FPR |
| --- | ---: | ---: | ---: | ---: |
| `B0_exact` | **0.785281** | **0.452626** | **0.594082** | **0.005430** |
| `M1_CF_GC_RN` | 0.785101 | 0.452613 | 0.592822 | 0.005551 |

M1-B0 was `-0.000180/-0.000014/-0.001261` on pAUROC/F1/AU-PRO.
The conditional excess AUC was only `0.518845`; rice/vial were
`0.500364/0.537326`, below the continuation gate. M1 also failed to dominate
the offset and neighborhood nulls. Candidate coverage was nonzero, so this was
not an inactive-path failure.

## Continuation Assessment

The local neighbor/offset selector does not add anomaly-aligned evidence beyond
exact RN. Do not scale or retune CF-GC-RN. A distinct feature-residual
test-time inference diagnostic may proceed only while preserving the selected
raw RN distance and exact no-op fallback.

## Conclusion

`KILL_FOR_CLAIM / NO_CONTINUE` for CF-GC-RN itself. Exact RN remains the
stronger method.

## Post-Conclusion Storage Cleanup

The local pullback contains zero `anomaly_maps/` directories. Compact metrics,
manifests, logs, and aggregate SHA-256
`db76d775229677b435570d8e7ed045338730f68df15e50be28e5d21b6de4c967` are
preserved.
