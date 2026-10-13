# Synthetic-Shift Memory-Bank Expansion Report

Date: 2026-08-10

Protocol: `synthetic-shift-bank-expansion-all8-20260810`

Verdict: `INVALID` — two preregistration/execution faults, both mine. The
endpoint gates were never read.

Scope: `superad_seg_AUROC_0.05=N/A`, `superad_seg_F1=N/A`,
`delta_vs_superad=N/A`, `comparable=false`.

## Outcome

Validity gates 1, 5, 6 passed. Gates 2, 3, 4, 7, 8 failed, so the verdict
short-circuits to `INVALID` and the hard-null, real-normal, anti-absorption,
strict, and continuation gates were not evaluated.

## Root Causes

**Fault 1 — the reused anchor was gated as if it were an arm.**

`S128_real` was reused from `can-collapse-attribution-all8-20260809` Block A1. It
is a real-normal ladder arm from a different protocol: its Flow is fitted on 128
images, its bank is the 128-image bank, and its artifacts carry no
`bank_size`, `flow_fit_source_view_count`, or `flow_parameter_checksum`
diagnostics at all (`initial_memory_size=489216` is all it records).

The preregistration nonetheless wrote validity gate 3 as "**every arm** reports a
Flow fitted on exactly the 16 original support views". `S128_real` cannot satisfy
that by construction — being fitted on more normals is what it *is*. Gate 3
therefore produced 9 failures: one direct (`S128_real: flow was not fitted on
original support-16`) and eight cascaded, because the cross-arm checksum
comparison included `S128_real`'s `None`. Gates 2, 4, and 7 failed for the same
reason, and every one of their failures names `S128_real` and nothing else.

`S128_real` should have been declared an **external frozen comparator** — exempt
from arm-level Flow and bank gates, subject only to its own custody check —
exactly as `raw` / `position_current` / `position_lowpass` were handled in the
scale-selective protocol.

**Fault 2 — the required parity replication was never executed.**

The preregistration states "One arm is deliberately duplicated across local and
dsba5 for parity." No duplicated arm was run. Gate 8 correctly reports
`duplicated local/dsba5 parity exceeded tolerance`. This is not a scoping error;
it is a requirement I simply did not carry out.

## What Was Measured Before Any Gate Manipulation

The `E0`/`E1` contrast was computed and observed before the aggregator was run,
and is reported here as a descriptive measurement only. It carries no verdict
weight and no gate was read to obtain it.

| macro contrast | value | strict gate |
| --- | ---: | ---: |
| `E1 - E0` `seg_AUROC_0.05` | +0.00036 | `>= +0.010` |
| `E1 - E0` `seg_F1` | +0.00054 | — |
| `E1 - E0` `seg_AUPRO_0.05` | +0.00016 | — |
| `E1 - E0` `normal_image_pooled_fpr_at_oracle` | -0.00021 | `< 0` |

Per object, the largest single movement was `can`'s deployed normal FPR falling
`0.00562 -> 0.00405`, a `28%` relative reduction, against a pAUROC change of
only `+0.0014`. Both `E2` objects visible during execution (`fabric` raw F1
`0.66865`, `rice` `0.71426`) were indistinguishable from `E0` and `E1`.

No verdict is claimed from any of this.

## Why This Experiment Is Superseded Regardless

After these runs, a public-split condition decomposition — computed from the
`condition_attribution` block that every run already emits, with no GPU — showed
that this protocol's augmentation model targeted the wrong axis.

Normal FPR at the global oracle, macro over eight objects: `regular .00126`,
`photometric .00140` (`+11%`), `_shift_* .00284` (`+125%`). Phase correlation
against the matching `regular` image gives alignment offsets of `(0,0)` to `±1`
pixel, so `_shift_*` is not spatial displacement despite the evaluator's
`geometric` label; the spatial coefficient of variation of the multiplicative
field `img/regular` is `.016--.061` for exposure conditions and `.044--.274` for
`_shift_*`.

This protocol augmented with a spatially **uniform** global exposure ladder
`I * 2^lambda`. That covers the benign `+11%` axis and leaves the damaging
`+125%` axis untouched. Even had the gates been correctly scoped and parity
executed, the augmentation model was wrong for the nuisance that actually
inflates normal scores.

## Corrective Actions Taken

1. The successor protocol
   `spatial-illumination-bank-expansion-all8-20260810` is amended **before its
   data is observed** so that arm-level Flow and bank gates apply only to arms
   generated under this contract, with external comparators checked for custody
   only.
2. A parity duplicate of `F1_spatial` was launched on the local RTX 4090 while
   the dsba3 arms were still running, so the successor's parity gate has the
   replication it requires.

## Conclusion

`INVALID`. The protocol failed on its own construction and execution, not on its
hypothesis: an external comparator was gated as an arm, and the required parity
replication was not run. The measured `E1 - E0` contrast is near zero, and an
independent condition decomposition shows the augmentation model targeted the
benign axis, so the question this protocol asked is carried forward by the
spatial-illumination successor rather than retried here.

Primary artifact: `results/synthetic_shift_bank_20260810/summary.json`.
