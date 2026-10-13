# SO-CFM Synthetic-Guard Stage -1 Report

Verdict: `KILL_FOR_CLAIM_CONTINUE_DIAGNOSTIC`.

This is a support-only mechanism diagnostic. SuperAD/RN-FMLK fields are `N/A`, and `benchmark_comparable=false`.

## Conclusion

The tested shared conditional velocity field does not provide a viable shift-correction claim. Guarded
SO-CFM (A3) reached macro displacement coverage `0.520364`, but its macro 95th-percentile tail reduction
was `-0.003347` against the frozen `0.20` gate. The simpler discrete support-orbit bank (A1) was stronger
on both coverage (`0.732499`) and tail reduction (`0.012623`); A3 beat A1 tail reduction on `0/4` objects.

The synthetic-anomaly guard also did not establish causal value. A3 was no worse than normal SO-CFM (A2)
on defect retention on only `1/4` objects, separated from the mask-shuffle null on `0/4`, and its terminal
hinge loss was zero in all 16 object-fold fits. Thus the present hinge was inactive. This kills the current
guard formulation, but does not rule out a redesigned active-margin/Jacobian constraint.

The bounded continuation is to retain the discrete orbit bank as the empirical upper baseline and test a
residual model that predicts the exact A1 transport displacement, with an explicitly active anomaly
Jacobian/contrast preservation loss and a pre-fit activation-rate gate. Do not scale the current SO-CFM or
open benchmark/query data from this result.

## Execution and Custody

| Resource | Process-visible device | Objects |
| --- | --- | --- |
| local GPU0 | `cuda:0` | `can`, `fabric` |
| dsba3 host GPU3 | `cuda:0` | `vial` |
| dsba5 host GPU3 | container device 0 / `cuda:0` | `walnuts` |

All four results record the frozen support manifest SHA-256
`7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`, with
`query_or_test_access=false`, `test_public_accessed=false`, and `test_private_accessed=false`.

## Gate Counts

| Gate | Pass objects | Required |
| --- | ---: | ---: |
| `displacement_coverage_ge_0p50` | 2 | 4 |
| `velocity_shuffle_coverage_margin_ge_0p20` | 2 | 4 |
| `shifted_normal_tail_reduction_ge_0p20` | 0 | 4 |
| `held_local_defect_retention_ge_0p95` | 4 | 4 |
| `held_broad_defect_retention_ge_0p95` | 3 | 4 |
| `guard_retention_not_worse_than_normal` | 1 | 4 |
| `guard_tail_no_more_than_2pp_worse_than_normal` | 4 | 4 |
| `mask_shuffle_retention_margin_ge_0p03` | 0 | 4 |
| `beats_discrete_orbit_bank_tail` | 0 | 4 |
| `beats_discrete_orbit_bank_retention` | 4 | 4 |
| `anomaly_positive_absorption_stress_visible` | 0 | 4 |

## Object Summary

| Object | Pass | A3 coverage | A3 tail reduction | A3 min retention | Skipped | Source |
| --- | --- | ---: | ---: | ---: | --- | --- |
| can | no | 0.422728 | -0.000822 | 0.995519 | N3_patchwise_code, clean_geometry_and_code_drift | `results/socfm_synthetic_guard_stage_minus1_20260811/local/mechanism/can/socfm_object_summary.json` |
| fabric | no | 0.637904 | 0.002905 | 0.997321 | N3_patchwise_code, clean_geometry_and_code_drift | `results/socfm_synthetic_guard_stage_minus1_20260811/local/mechanism/fabric/socfm_object_summary.json` |
| vial | no | 0.365247 | -0.019148 | 0.921188 | N3_patchwise_code, clean_geometry_and_code_drift | `results/remote_runs/dsba3/socfm_synthetic_guard_stage_minus1_20260811_dsba3_v3/mechanism/vial/socfm_object_summary.json` |
| walnuts | no | 0.655576 | 0.003677 | 0.999357 | N3_patchwise_code, clean_geometry_and_code_drift | `results/remote_runs/dsba5/socfm_synthetic_guard_stage_minus1_20260811_dsba5_v3/mechanism/walnuts/socfm_object_summary.json` |
