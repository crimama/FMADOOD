# Same16 Baseline Metric-Contract Closure

Date: 2026-07-24  
Verdict: `KILL_FOR_CLAIM / NO_SHOT_SCALING`

## Negative Evidence Intake

Historical rows were rejected because they did not share the exact reference,
backbone, preprocessing, population, and metric contract. This run uses only
same-run Basic, SuperAD-16, and RN-FMLK evidence.

## Motivation

Close the all-eight same-condition baseline gate before any Basic/A6 promotion
or shot-scaling run.

## Implementable Design

All branches use full MVTec AD2 `test_public/good,bad`, the exact fixed-16
support SHA `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`,
DINOv2-L/14, and the frozen resolution contract.

## Evaluation Alignment

The common evaluator reports original-resolution standardized pixel
pAUROC@0.05, raw oracle pixel F1, AU-PRO@0.05, and complete map/shape/finite
coverage. These are TESTpublic public-shadow metrics, not private-server claims.

## Code Modification / Creation

- Fixed the remote runner so Basic and mechanism branches import `fmad` and
  `src.post_eval` from the isolated snapshot instead of stale `/workspace`.
- Added a regression test for the snapshot import contract.
- Added a recovery evaluator that verifies module provenance, preserves stale
  metrics as legacy evidence, validates coverage, and atomically replaces
  metrics from retained dense maps.

## Added Code Evaluation

Remote `py_compile`, shell syntax, and five focused recovery/runner tests pass.
The recovery verified snapshot module origins before evaluation. All 40 stale
Basic/mechanism object rows passed the new metric schema and coverage checks;
40 legacy metric backups were retained.

## Remote Execution

Run `same16_baseline_metric_contract_20260723_v6` completed in fixed dsba3
container `hun_fsad_tta_012`. The original controller generated all 56
object/arm map sets but failed closed at aggregation because Basic/mechanism
had imported the stale evaluator. The preserved maps were re-evaluated with
the isolated snapshot, then both aggregators and the independent adjudicator
completed.

Remote compact root:
`/var/tmp/fsad_tta/results_remote/same16_baseline_metric_contract_20260723_v6`

Local compact pull:
`results/remote_runs/dsba3/same16_baseline_metric_contract_20260723_v6`

## SuperAD Baseline and Unified Metrics

| Method | pAUROC@0.05 | Raw F1 | AU-PRO@0.05 |
|---|---:|---:|---:|
| Basic | 0.772635045 | 0.355675500 | 0.533407769 |
| SuperAD-16 | 0.761857074 | 0.368423110 | 0.534924081 |
| RN-FMLK | 0.745312086 | 0.331950199 | 0.425132995 |

Basic minus SuperAD is `+0.010777972` pAUROC,
`-0.012747610` F1, and `-0.001516312` AU-PRO. Basic therefore does not
dominate SuperAD across the preregistered metrics. It also fails the
object-collapse/directional gate against RN-FMLK despite positive macro deltas.

## Results and Analysis

The independent decision is `A6_NOT_PROMOTED_RETAIN_BASIC`. Basic remains the
selected detector, but it is not a claim-positive replacement for SuperAD.
The machine decision is `KILL_FOR_CLAIM_NO_SHOT_SCALING`; the blocked marker
is `shot_scaling_blocked.txt`.

## Continuation Assessment

No shot curve or scaling continuation is authorized. The retained asset is a
fully comparable same-condition baseline table, corrected evaluator custody,
and reusable recovery tooling.

## Conclusion

`KILL_FOR_CLAIM / NO_SHOT_SCALING`. Basic has the best pAUROC, but the F1 and
AU-PRO losses to SuperAD prevent promotion.

## Post-Conclusion Storage Cleanup

All 7,588 regenerable TIFF maps and all anomaly/calibration map directories
were removed after aggregation and adjudication. Compact metrics, manifests,
logs, 40 legacy backups, 28 target recovery markers, decision artifacts, and
cleanup evidence were pulled locally. The isolated remote source snapshot was
removed.
