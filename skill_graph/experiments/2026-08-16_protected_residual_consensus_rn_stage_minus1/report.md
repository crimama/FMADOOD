# Protected Residual-Consensus RN Stage -1

Date: 2026-08-16
Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`

## Negative Evidence Intake

See `preregistration.md`.  The method preserves exact RN distance and tests a
feature-residual consensus, not a spatial-offset, tangent, flow, or additive
score branch.

## Motivation

Test whether protected test-time nuisance inference can improve which frozen
RN top-5 reference is used without changing the anomaly evidence itself.

## Implementable Design

Frozen in `preregistration.md` before implementation.

## Evaluation Alignment

MVTec AD2 single-image rice/vial full-support TESTpublic diagnostic with a
same-run exact RN1 comparator.  SuperAD-16 is budget- and panel-incomparable.

## Code Modification / Creation

- Added `src/flow_tte/protected_residual_consensus.py`.
- Added the PRC runner, fail-closed aggregate, two-host remote controller, and
  focused core/pipeline tests without modifying the killed CF-GC implementation.

## Added Code Evaluation

Focused PRC plus adjacent CF-GC regressions: `38 passed`. Ruff, Python
compilation, shell syntax, CLI help, and scoped diff checks passed. An
import-time base-module mutation found by the adjacent suite was removed before
the remote snapshot.

## Remote Execution

Immutable run `protected_residual_consensus_rn_rice_vial_stage1_v1_20260816`
was submitted after manifest/data/RAM/disk and host/container GPU UUID
preflight. dsba5 host GPU3 runs rice; dsba3 host GPU3 runs vial. Both runners
are active in the fixed project container.

## SuperAD Baseline and Unified Metrics

`superad_seg_AUROC_0.05=N/A`, `comparable=false`.  The primary immediate
baseline is same-run `B0_exact_RN1`.

## Results and Analysis

Immutable run `protected_residual_consensus_rn_rice_vial_stage1_v1_20260816`
completed on both hosts and passed support/test path custody, complete map
coverage, finite/nonconstant score, GT-after-map, no-update, and compact-only
validation.

| arm | macro pAUROC | macro F1 | macro AU-PRO | normal FPR |
| --- | ---: | ---: | ---: | ---: |
| B0 exact RN1 | 0.785281 | 0.452626 | 0.594082 | 0.005430 |
| M1 protected residual consensus | 0.785253 | 0.452584 | 0.594077 | 0.005339 |
| N1 code-dimension shuffle | 0.785272 | 0.452627 | 0.594039 | 0.005432 |
| N2 patch-pair shuffle | 0.784713 | 0.452486 | 0.592560 | 0.005453 |
| N3 untrimmed mean | 0.785119 | 0.452475 | 0.593708 | 0.005349 |
| N4 confidence shuffle | 0.785159 | 0.452534 | 0.593827 | 0.005347 |

`M1-B0` was pAUROC `-0.000028`, F1 `-0.000042`, AU-PRO `-0.000005`,
and normal FPR `-0.000091`.  Rice fell back on every query and reproduced B0
exactly (`0.914328/0.624266/0.547475`).  Vial activated M1 on part of the
panel but slightly lost to B0: pAUROC `0.656177` versus `0.656234`, F1
`0.280901` versus `0.280986`, and AU-PRO `0.640679` versus `0.640690`.

The macro conditional excess AUC was `0.487202`, below both chance-oriented
diagnostic evidence and the preregistered `0.55` continuation threshold.  M1
also failed to beat the code-dimension shuffle on pAUROC (`-0.000019`).  It did
beat the patch-pair, untrimmed-mean, and confidence-shuffle controls, but those
small margins cannot authorize continuation without either a B0 improvement
or conditional AUC evidence.  The corrected fail-closed aggregate therefore
returns `KILL_FOR_CLAIM / NO_CONTINUE`.

Relative to the preceding CF-GC-RN branch, PRC is less harmful (CF-GC-RN
`M1-B0=-0.000180` pAUROC and `-0.001261` AU-PRO), but it still does not improve
the same exact RN1 baseline.  Both experiments independently point to the same
result: top-5 reference reselection from query-derived consensus is mostly
absorbed by RN1 or introduces weak, non-causal rank changes.

## Continuation Assessment

Do not tune the `0.10` fallback threshold, robust inlier fraction, or current
residual code.  Rice provides no active mechanism evidence, vial moves in the
wrong direction, conditional AUC is below `0.5`, and the strongest hard null
slightly exceeds M1.  Exact RN1 remains the method to preserve.

The next TTA investigation should target a genuinely new evidence source that
can predict *when RN1 is wrong* from support-only calibration, rather than a
new rule for choosing among the same RN top-5.  Any such branch must first
demonstrate support-LOO error prediction or held-nuisance risk separation and
must leave RN scores unchanged when that evidence is absent.

## Conclusion

Protected residual consensus is safe but ineffective: its protection works,
yet the residual code does not identify beneficial RN candidate switches.
Close this formulation and retain exact full-support RN1 as the current
strongest simple comparison rule.

## Post-Conclusion Storage Cleanup

Both pulled shards report `dense_maps_persisted=false`; no regenerable anomaly
maps were transferred.  Compact summaries and the final aggregate are retained.
Both regenerable remote isolated code snapshots were removed only after the
remote `complete.txt` markers and local pullbacks were verified; remote result
directories remain intact.
The final aggregate SHA256 is
`17dca442d6aaa6b09c7d619e86d24a7f711be9bfe3bc65c6b41ec305af8de89a`.
