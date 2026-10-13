# CIDER-CF Stage -1 report

Status: `KILL_FOR_CLAIM` (fatal support-only mechanism failure).

## Execution

Final immutable run: `cider_cf_stage_minus1_dsba3_all4_20260824_v5` in fixed
container `hun_fsad_tta_012`. GPU mapping was can:0, fabric:1, vial:2,
walnuts:3. All four shards completed and compact artifacts were pulled to
`results/remote_runs/dsba3/`. No anomaly-map directory remains.

Custody passed: exact manifest hash, support16, leave-target-out source, and
zero public/private/test/ground-truth access. Target synthetics were evaluation
only and query memory was never updated.

## Result

| object | held-family macro pAUC@.05 | min family | nuisance FPR | generator probe | min hard-null margin |
|---|---:|---:|---:|---:|---:|
| can | 0.9432 | 0.8055 | 0.0371 | 0.4947 | -0.0125 |
| fabric | 0.9732 | 0.9054 | 0.0322 | 0.4998 | -0.0046 |
| vial | 0.9249 | 0.8445 | 0.0562 | 0.5749 | -0.0472 |
| walnuts | 0.9289 | 0.8688 | 0.3540 | 0.5069 | -0.0598 |

Macro pAUC was 0.9426 and the weakest family was 0.8055, so the instruments
produce strong synthetic separation. That evidence is not mechanism-valid:
generator identity is easily decoded (maximum balanced accuracy 0.5749 versus
chance 0.20 and fatal threshold 0.35), no object beats the hard nulls, and
walnuts has catastrophic normal-nuisance FPR. Macro nuisance FPR is 0.1199.

## Verdict and next direction

The primary claim is killed. Synthetic anomalies are useful here only as a
stress-test/evaluation battery, not as a learned covariance alternative. Do
not open TESTpublic, do not tune this method on target synthetics, and do not
promote CIDER-CF to SuperAD/RN comparison. Any successor must remove explicit
generator coordinates before fitting (for example, paired family-contrast
orthogonalization learned source-only) and must first pass walnuts normal-orbit
specificity; that is a separately preregistered mechanism, not a CIDER-CF tweak.
