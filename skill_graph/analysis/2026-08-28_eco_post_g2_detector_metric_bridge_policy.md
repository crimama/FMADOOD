# ECO post-G2 detector/metric bridge policy

Date: 2026-08-28  
Status: `RESULT-FREE / PROSPECTIVE / DOES_NOT_AMEND_ECO-G2`  
Applies after: `eco-g2-nativecore-20260828-v1` terminal result

## Boundary

This policy does not change the frozen ECO-G2 N0/N1 preregistration, inputs,
thresholds, verdicts, or execution order. ECO-G2 remains a support-only
mechanism diagnostic with no detector, benchmark, LoRA, query, test, or ground
truth access. Its current run must finish under its existing contract.

This document governs only the branch that becomes available after the
ECO-G2 terminal result. Its purpose is to stop an unlimited chain of
support-only diagnostics and force either an anomaly-detector connection or
closure of the ECO family.

## Prospective state machine

1. `N0_FAIL_CLOSE_EXISTING_NATIVE_ROUTE/NO_GPU` closes the existing ECO native
   line. No benchmark rescue, support-only rescue, or replacement mechanism is
   authorized.
2. `N1_FAIL_CLOSE_NATIVE_OBSERVATION_LINE` also closes the existing ECO native
   line. No further ECO structural/support-only diagnostic is authorized.
3. `PROMISING_DIAGNOSTIC/OPERATOR_ONLY_SENTINEL_AUTHORIZED/NO_LORA` authorizes
   at most one separately named, result-free-preregistered operator-only
   sentinel. It does not authorize LoRA or detector claims.
4. After that sentinel, the next step must be either a detector bridge or
   `END_ECO_FAMILY`. A second structural/support-only sentinel is forbidden.
5. A detector bridge that passes its validity gate must immediately enter one
   small preregistered metric checkpoint. A bridge that cannot produce valid
   query anomaly outputs closes the ECO family.
6. Only a `PROMISING_DIAGNOSTIC` checkpoint authorizes an unchanged all-eight
   confirmation. Any method change after checkpoint exposure starts a new
   generation and requires a new preregistration.

## One-sentinel budget

If N1 passes, the child sentinel must freeze its operator, nulls, observations,
thresholds, and terminal rule before opening its inputs. Its budget is exactly
one run family and one result-dependent interpretation. It may determine
whether the native-core mechanism is genuine; it may not add a detector,
benchmark, adapter, or another support-only rescue inside the sentinel.

## Mandatory detector bridge

The detector child preregistration must define, before query access:

- the exact formula producing one anomaly score per query or one pixel anomaly
  map per query;
- output shape, resizing, normalization, and original-resolution evaluator
  connection;
- finite and nonconstant-output checks and evaluated-map coverage;
- normal/anomaly directionality, including the predesignated statistic and
  pass threshold;
- a separately named identity/no-op arm;
- separately named mechanism-matched hard-null arms and their frozen envelope;
- `delta_vs_identity`, every per-null delta, and
  `delta_vs_hard_null_envelope`;
- exact method, source, configuration, support-manifest, preprocessing, and
  evaluator hashes.

Identity and mechanism-matched hard nulls are distinct controls. If output is
nonfinite, constant, incompletely covered, directionally reversed, or cannot
be evaluated through the frozen detector path, the verdict is
`END_ECO_FAMILY / NO_DETECTOR_BRIDGE`.

## Immediate three-object metric checkpoint

The detector child preregistration must freeze the exact weak, strong, and
control object names, their roles and order, and the selection rule before any
of those query results are opened. The checkpoint has `revision_budget=0`.
The object identities cannot be inferred from ECO-G2, which only opens
`can/train/good`; they must be selected by a result-independent rule and
sealed in the child preregistration.

All arms run in the same checkpoint with the exact SuperAD-selected 16
references, identical preprocessing/evaluator, and identical object/query
population. Required comparators are:

- candidate detector bridge;
- same-condition `SuperAD-16`;
- same-condition `RN-FMLK`;
- identity/no-op;
- every mechanism-matched hard null and their envelope.

For every object, arm, and macro summary, the compact output must include:

- `seg_AUROC_0.05`;
- `seg_F1`;
- `AU-PRO@0.05` (canonical machine field: `seg_AUPRO_0.05`);
- finite/nonconstant map count and map coverage;
- normal FPR;
- runtime and peak memory;
- `delta_vs_superad`, `delta_vs_rn_fmlk`, `delta_vs_identity`, every
  `delta_vs_hard_null`, and `delta_vs_hard_null_envelope`.

The child preregistration must designate the primary metric or a closed metric
disjunction and numerical F1, coverage, and strong/control no-harm boundaries.
The checkpoint does not require paper-level significance. It returns
`PROMISING_DIAGNOSTIC` only when at least one predesignated core metric moves in
the intended direction, F1 and coverage do not collapse, the signal is not
fully absorbed by the hard-null envelope, and strong/control no-harm is not
fatally violated.

It returns `VALID_BUT_NO_GAIN / CLOSE_FAMILY` when the candidate is inferior to
both SuperAD-16 and RN-FMLK on the preregistered decision rule, a hard null
absorbs the signal, or F1/coverage collapses. All other valid nonpromotions are
`CLOSE_FAMILY`; there is no same-generation repair.

## All-eight confirmation and final claim

`PROMISING_DIAGNOSTIC` authorizes one byte-identical all-eight run only. The
method/configuration hashes, detector formula, selected-16 references,
preprocessing, evaluator, comparators, hard-null envelope, and thresholds must
be inherited unchanged.

A final claim requires all eight `TESTpublic` objects, exact SuperAD-selected
16 references, same-condition SuperAD-16 and RN-FMLK, the hard-null envelope,
joint preservation of `seg_AUROC_0.05`, `seg_F1`, and coverage, object-level
no-harm, and preregistered repeats and confidence intervals. Results observed
before a method change cannot be reused as confirmatory evidence for the
changed method.

## Ledger and selection-bias control

Every success, failure, invalid run, and abandoned bridge is recorded with:

- `generation_id` and `parent_generation_id`;
- method/source/config/evaluator/support hashes;
- exposed objects and phases;
- checkpoint object identities and `revision_budget`;
- every arm result and terminal verdict;
- whether promotion, closure, or a new generation followed.

Checkpoint exposure consumes that generation's revision budget. Any change
after seeing checkpoint or all-eight results creates a new generation with a
new result-free preregistration. This policy cannot be relaxed after an ECO-G2,
sentinel, checkpoint, or all-eight result.

## Portfolio conclusion

Support-only studies do not each need benchmark metrics, but a single method
family cannot chain them indefinitely. The current lineage is
`NCFC-S1 -> ECO-Audit v2 -> ECO-G2`; therefore ECO-G2 is the final unrestricted
support-only diagnostic in this family. Its only prospective exits are the
state-machine branches above: closure, or one sentinel followed immediately
by a valid detector/metric bridge.
