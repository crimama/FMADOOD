# Normal-Tail Structure and Predictability Report

Date: 2026-08-10

Protocol: `normal-tail-structure-predictability-all8-20260809`

Verdict: `TAIL_NOT_IDENTIFIABLE_WITHOUT_TEST / NO_CONTINUE`

Scope: measurement-only. `superad_seg_AUROC_0.05=N/A`, `superad_seg_F1=N/A`,
`delta_vs_superad=N/A`, `comparable=false`. No intervention is licensed.

## Outcome

All six validity gates passed. G1 passed, G2 failed, and per the preregistered
ordering G3 was **not evaluated**. Local/dsba5 parity on the permutation-null
arm was clean (largest observed delta `~1.4e-6`, bound `1e-3`).

The normal-side tail is strongly structured, and it is **not** identifiable
from clean training normals.

## Results

| object | top-decile share | Gini | `C_pos` | `C_pos_null` | `rho` | `rho_null` | `V` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| can | 0.3466 | 0.6106 | 0.4167 | 0.0743 | 0.1153 | -0.0002 | 0.0016 |
| fabric | 0.8532 | 0.8703 | 0.3291 | 0.1502 | 0.0213 | 0.0003 | 0.0065 |
| fruit_jelly | 0.2824 | 0.5299 | 1.0000 | 0.7955 | 0.0994 | 0.0003 | 0.0286 |
| rice | 0.9857 | 0.9344 | 1.0000 | 1.0000 | -0.0004 | -0.0005 | 0.0045 |
| sheet_metal | 0.9197 | 0.8744 | 1.0000 | 1.0000 | -0.0002 | -0.0002 | 0.0020 |
| vial | 0.2542 | 0.4316 | 0.5464 | 0.1333 | **0.4280** | -0.0006 | 0.0664 |
| wallplugs | 0.8670 | 0.9212 | 0.5919 | 0.5480 | 0.0000 | 0.0000 | 0.0000 |
| walnuts | 0.7094 | 0.8449 | 1.0000 | 0.7206 | -0.0046 | 0.0005 | 0.0000 |

**G1 structure: pass, 8/8.** Every object satisfied at least one structure
criterion. The tail is highly concentrated: six of eight objects put more than
`70%` of tail mass in a single decile of normal images, and `can`, `fabric`,
`vial`, and `walnuts` beat their own position-permutation null on `C_pos` by
`+0.3424`, `+0.1789`, `+0.4131`, and `+0.2794`.

**G2 predictability: fail, 1/8.** Only `vial` reached `rho >= 0.30`
(`0.4280`, null `-0.0006`). Every other object sat near zero. The positions
that inflate on test normals are not the positions that inflate on training
normals.

**G3 anti-absorption: not evaluated.** The preregistration stops at the first
failed gate. The `V` column above is reported for the record only and carries
no verdict weight.

## Interpretation

The tail exists and is structured, so it is not noise. But it does not transfer
from `train/good` to `test_public/good`.

The likely reason is visible in the data itself: MVTec AD2's public test normals
include explicitly shifted variants (`*_overexposed`, `*_shift_1`, `*_shift_2`),
while `train/good` contains none. The nuisance that lifts normal scores at test
time is therefore a condition that is **absent from the training normals by
construction**.

This has a direct consequence for the eleven previously closed module branches.
Every one of them attempted to learn a support-relative transform — invariant
channels, shared photometric directions, conditional normalization, spatial
consensus, stable or positional retrieval keys, band-split residuals — from
unshifted normal data. This measurement says the signal those branches needed to
learn was not present in the data they were fitted on. Their failures are
consistent with a single cause rather than eleven separate design faults.

`vial` is the informative exception. It is the object with the highest normal
FPR (`0.00585`) and the only one whose tail transfers. Its tail is therefore
plausibly a stable geometric property rather than a shift artifact. One object
does not support a claim.

## Preregistration Limitation (recorded, not used to rescue the result)

The protocol set no minimum-tail-mass eligibility criterion, unlike the
`eligible_object_count` convention used in the earlier defect-scale work.
`rice` and `sheet_metal` have normal FPR `0.00001`, so their tails are nearly
empty; `C_pos` saturates at `1.0000` (equal to its own null) and `rho` is
effectively undefined. Those two objects should not have been scored as
informative rows in either direction.

Restricting attention to the three objects that actually exhibit the tail
problem — `can`, `fabric`, `vial` — G2 still fails at `1/3`. The verdict is
unchanged, so no re-reading is performed. The limitation is recorded for the
successor protocol.

## What This Closes

- Identifying the normal tail from clean normal data, then suppressing it, is
  closed. Position-recurrence priors, per-position normalization fitted on
  `train/good`, and any successor that requires knowing *which* positions will
  inflate at test time are not supported by evidence.
- G3 was never reached, so nothing is established about whether suppression
  would have been safe. That question is now moot for this family.

## What Remains Open

The tail is real and structured. What fails is predicting it from unshifted
normals. Two directions survive, and both are outside this protocol's licence:

1. Constructions that do not require predicting the tail — for example a
   normalization fitted on statistics that are themselves shift-stable, or a
   score scale made comparable across images without per-position knowledge.
   This is also the corrected-design requirement left open by
   `region-completion-map-level-all8-20260809`.
2. Test-time adaptation, which the current contract forbids under R7 static
   inference. Any move here changes the method's claim class and must be
   preregistered as such.

## Execution Record

- `public` and `fit` scoring plus T1/T2/T3 statistics: local RTX 4090 container,
  all eight objects.
- Permutation-null replication: dsba5 container `hun_fsad_tta_012`, host GPU 3,
  immutable snapshot; parity max delta `~1.4e-6`.
- Fit-normal set verified as exactly `S128 \ S16`, 112 images per object, drawn
  only from `train/good`, zero intersection with support-16, `test_public`,
  `validation`, or private splits.
- Permutation audit: multiset preserved on every image
  (`max_sorted_value_difference = 0.0`), no identity permutation.
- Access ledger: no query-time update, no metadata/condition input, no private
  access, no new fitting on `test_public`.
- Spearman backend: `scipy`.
- Two runtime defects were found and fixed before the accepted run: the
  fit-view support manifest was not resolvable by absolute path across data
  roots (fixed by basename re-rooting with set-equality custody), and
  `expected_fit` compared extension-bearing ladder basenames against
  extension-free map stems. Neither altered the scoring pipeline.
- No `anomaly_maps/` retained locally or in the pulled remote run.

## Conclusion

`TAIL_NOT_IDENTIFIABLE_WITHOUT_TEST / NO_CONTINUE`. The normal-side tail that
destroys low-FPR performance is strongly structured but does not transfer from
training normals to shifted test normals. This closes the identify-then-suppress
family and offers a single coherent explanation for the eleven previously closed
branches: they were fitted on data that does not contain the nuisance they were
meant to remove.

Primary artifact: `results/normal_tail_structure_20260810/summary.json`.
