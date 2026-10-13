# Spatial-Illumination Memory-Bank Expansion Report

Date: 2026-08-10

Protocol: `spatial-illumination-bank-expansion-all8-20260810`

Verdict: `INVALID` — validity gate 3 as written cannot hold for arms executed on
different machines. My error. Endpoint gates were not read.

Scope: `superad_seg_AUROC_0.05=N/A`, `superad_seg_F1=N/A`,
`delta_vs_superad=N/A`, `comparable=false`.

## Outcome

Validity gates 1, 2, 4, 5, 6, 7, 8, 9 passed. Gate 3 failed on all eight
objects, so the verdict short-circuits to `INVALID` and the mechanism,
hard-null, model-correction, real-normal, anti-absorption, strict, and
continuation gates were not evaluated.

The `2026-08-10` arm-class amendment worked as intended: gates 2, 4, and 7 now
pass, and `S128_real` was verified through the documented path-verification
route (`arm_classes.external_gate7.S128_real.gate7_route = "path_verification"`,
`train_good_only_verified = true`).

## Root Cause

Gate 3 required an "identical Flow parameter checksum across the contract arms".
The contract arms were executed on two different machines, and the checksums
partition exactly by machine:

| arm | machine | `can` checksum | `vial` checksum |
| --- | --- | --- | --- |
| `F0_baseline` | local RTX 4090 | `13c6209725bafd` | `f6b86cd799a6e1` |
| `E1_global` | local RTX 4090 | `13c6209725bafd` | `f6b86cd799a6e1` |
| `F1_spatial` parity | local RTX 4090 | `13c6209725bafd` | `f6b86cd799a6e1` |
| `F1_spatial` | dsba3 GPU 2 | `9fe15d92cf1cd7` | `04d985ce459f06` |
| `F2_highfreq_null` | dsba3 GPU 3 | `9fe15d92cf1cd7` | `04d985ce459f06` |

Within a machine the Flow fit is bit-reproducible; across machines it is not,
which is ordinary floating-point architecture dependence, not a defect.

The substantive property gate 3 exists to protect — that bank augmentation does
not alter the Flow — **holds in every arm**: `flow_checksum_before_bank_population`
equals `flow_parameter_checksum` everywhere, and every arm reports
`flow_fit_source_view_count = 16`.

Gate 9 independently shows the cross-machine difference is numerically
irrelevant: the local and dsba3 `F1_spatial` arms agree to a worst absolute
delta of `7.17e-05` against a `1e-3` tolerance.

Gate 3 should have been written as three separate conditions — checksum
unchanged before versus after bank population within each arm; checksum
identical across arms sharing a machine; cross-machine agreement bounded by the
parity tolerance — rather than as a single byte-equality requirement across all
arms. That is a formulation error I will apply to future protocols; I am not
amending it retroactively, because the endpoint data has now been seen and
rewriting a gate at this point would be post-hoc rescue.

## Measurement Recorded Without Verdict Weight

The `F1_spatial` parity arm completed all eight objects on the local RTX 4090
and agrees with the dsba3 arm within `7.17e-05`, so the following preregistered
mechanism quantity is well determined. It is reported as a measurement. It is
**not** a verdict, and it does not fire any preregistered closure clause.

Shift-group (`_shift_*`) `good_pixel_fpr_at_global_oracle`, `F1` versus `F0`:

| object | F0 | F1 | relative |
| --- | ---: | ---: | ---: |
| can | .01018 | .00921 | -9.6% |
| vial | .00643 | .00690 | **+7.3%** |
| fabric | .00320 | .00306 | -4.3% |
| fruit_jelly | .00121 | .00118 | -2.0% |
| wallplugs | .00093 | .00092 | -0.5% |
| walnuts | .00074 | .00063 | -15.0% |
| rice | .00002 | .00001 | -9.6% |
| sheet_metal | .00003 | .00002 | -41.0% |
| **macro** | **.00284** | **.00274** | **-3.5%** |

Macro `seg_AUROC_0.05` moved `+0.00069`.

The preregistered mechanism gate required `<= -20%`. The observed `-3.5%` is far
short of it. `sheet_metal`'s `-41%` is on an absolute rate of `.00003 -> .00002`
and carries no weight; among the three objects that actually exhibit the tail
problem the values are `-9.6%` (`can`), `+7.3%` (`vial`, worse), and `-4.3%`
(`fabric`).

## Interpretation

Spatially structured illumination augmentation of the support bank does not
materially reduce the deployed normal false-positive inflation under the real
`_shift_*` conditions, and does not move headline metrics.

Read together with the predecessor protocol, whose global exposure ladder gave
macro pAUROC `+0.00036`, the question that motivated this experiment — whether
the earlier null result was caused by the wrong augmentation model — now has an
answer from the measurements even though its gate was never formally read:
**neither augmentation model moves the mechanism endpoint.** The exposure model
targeted the benign axis and the spatial model targets the damaging axis, and
both leave the deployed normal tail essentially where it was.

This is the fourth independent manipulation of the normal side that fails to
transfer to deployed scores:

| protocol | established on the normal side | transfer to deployed score |
| --- | --- | --- |
| Stage -1b / -1c | shared photometric direction reproduces `4/4` | tail reduction `.0002` against a `.02` gate |
| normal-tail structure | tail is structured `8/8` | not identifiable from `train/good`, `1/8` |
| synthetic-shift bank | `can` normal FPR `-28%` | pAUROC `+.0014` |
| this protocol | spatial illumination augmentation applied | shift-group FPR `-3.5%`, pAUROC `+.0007` |

## Continuation Assessment

- No preregistered closure clause fires, because the protocol is `INVALID`. The
  spatial-illumination family is not formally closed by verdict.
- It is nevertheless not worth continuing on this evidence. Any successor in
  this family must first state how it would move the shift-group normal FPR,
  given that two augmentation models moved it by `-3.5%` and effectively zero.
- Gate 3's formulation is corrected for future protocols as described above.
- The accumulated pattern argues for leaving the normal side entirely. The
  remaining measured deficit is on the defect side: macro
  `gt_component_recall_at_oracle` `.484` and private AU-PRO `-.1285` against
  RoBiS, whose published ablation attributes `+12.5%` SegF1 to mask refinement
  versus `+1.9%` to shift augmentation.

## Execution Record

- `F1_spatial` on dsba3 container `hun_fsad_tta_012`, host GPU 2;
  `F2_highfreq_null` on host GPU 3; both immutable snapshots.
- `F1_spatial` parity duplicate on the local RTX 4090, all eight objects,
  worst shared-metric delta `7.17e-05`.
- `F0_baseline`, `E1_global`, `S128_real` reused after custody verification.
- Support manifest custody verified against SHA-256
  `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`.
- No `anomaly_maps/` retained locally or in either pulled remote run.

## Conclusion

`INVALID` on a gate-formulation error of mine. The experiment nevertheless
produced a well-determined, parity-verified measurement: spatially structured
illumination augmentation of the normal bank reduces the shift-group normal
false-positive rate by `3.5%` against a required `20%`, and moves macro pAUROC
by `+0.0007`. Combined with the predecessor, neither augmentation model reaches
the mechanism it was designed to move.

Primary artifact: `results/spatial_illumination_20260810/summary.json`.
