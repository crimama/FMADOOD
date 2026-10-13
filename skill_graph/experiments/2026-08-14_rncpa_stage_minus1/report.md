# RNC-PA Stage -1 Report

Date: 2026-08-14
Status: `RUNNING / IMMUTABLE V3`
Verdict ceiling: `PROMISING_DIAGNOSTIC`

## Purpose

Test whether exact RN correspondence becomes more discriminative when a 3x3
query region must admit a coherent partial assignment to one normal support
source under one bounded relative translation. Exact RN remains the hard score
floor, so the experiment tests additive structural invalidity rather than
replacing the strongest surviving baseline.

## Claim Boundary

This reduced-object full-support TESTpublic diagnostic is not comparable to
SuperAD-16 and cannot support a benchmark claim. Results, validity checks,
execution provenance, and the final keep/kill decision will be recorded after
all four immutable shards aggregate successfully.

## Results

Execution `rnc_pa_stage_minus1_v3_20260814` is running on dsba3 GPU3
(`can -> walnuts`) and dsba5 GPU3 (`rice -> vial`). Results remain pending until
all four compact shards pass fail-closed aggregation. Superseded v1/v2 attempts
produced zero valid shards and are excluded from every comparison.

Partial status: dsba5 completed valid `rice` and `vial` shards. Their
provisional M0 deltas versus B0 RN1 are negative on both pAUROC and F1:
`rice` −0.189447/−0.518083 and `vial` −0.019033/−0.024808. The `vial`
conditional invalidity AUC is positive at 0.727172, but this has not translated
into an end-metric gain. These values are not a verdict; dsba3 is still scoring
`can`, with `walnuts` queued after it.

Subsequently, `can` completed validly. Its M0 versus B0 pAUROC/F1 was
0.532220/0.000379 versus 0.589333/0.001415, with conditional invalidity AUC
0.608762. Therefore all three completed objects currently lose to the exact
RN1 floor on both primary metrics. `walnuts` is the sole remaining shard and
is actively scoring; fail-closed aggregation remains pending.
