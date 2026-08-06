# A6 and Dominant-L17 Public-Shadow Confirmation

Date: 2026-07-24  
Verdict: `KILL_FOR_CLAIM / RETAIN_BASIC / NO_SHOT_SCALING`

## Negative Evidence Intake

This bounded confirmation tests the previously observed Flow-without-DVT and
layer-17 signals without reopening broad DVT, density, routing, or fusion
searches.

## Motivation

Test whether the frozen A6 and dominant-layer signs transfer to full
original-resolution MVTec AD2 TESTpublic while retaining A7/L17 hard controls.

## Implementable Design

The frozen arms are Basic, A6, A7, A6-L17, and L17. All use the exact same 16
references, static memory, no query updates, and the common raw
pAUROC/F1/AU-PRO evaluator.

## Evaluation Alignment

All eight objects, support identities/hash, raw-map coverage, three metrics,
2,000 object bootstraps, direction counts, and collapse gates were validated.

## Code Modification / Creation

The isolated-import correction and dense-map recovery are documented in the
joint Same16 report. No anomaly maps were regenerated.

## Added Code Evaluation

All 40 Basic/mechanism object rows passed schema, population, shape, and finite
coverage checks before aggregation.

## Remote Execution

Joint run `same16_baseline_metric_contract_20260723_v6` completed on dsba3.
Compact evidence is at
`results/remote_runs/dsba3/same16_baseline_metric_contract_20260723_v6`.

## SuperAD Baseline and Unified Metrics

| Arm | pAUROC@0.05 | Raw F1 | AU-PRO@0.05 |
|---|---:|---:|---:|
| Basic | 0.772635045 | 0.355675500 | 0.533407769 |
| A6 | 0.759283906 | 0.350215920 | 0.538267291 |
| A7 | 0.718392943 | 0.296003459 | 0.499769220 |
| A6-L17 | 0.722623630 | 0.290316333 | 0.455608697 |
| L17 | 0.706543079 | 0.276693039 | 0.439252714 |

## Results and Analysis

A6 beats A7 in macro terms but fails the preregistered no-collapse and
bootstrap support gates. Relative to Basic, A6 is `-0.013351139` pAUROC,
`-0.005459580` F1, and `+0.004859522` AU-PRO; none of the supported-metric
promotion gates pass. A6-L17 is substantially worse than A6 on all three
metrics, so dominant-L17 repair is rejected.

## Continuation Assessment

The mechanism signal is diagnostic only. There is no authorized shot-scaling
or method-promotion continuation.

## Conclusion

`KILL_FOR_CLAIM / RETAIN_BASIC / NO_SHOT_SCALING`.

## Post-Conclusion Storage Cleanup

All 7,588 dense maps and the isolated snapshot were removed after the compact
artifacts and blocked decision were verified and pulled.
