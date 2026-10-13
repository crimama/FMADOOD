# DB-FSPC Gate A Preregistration

Date: 2026-08-11

Status: `PREREGISTERED / PENDING`

## Negative Evidence Intake

Gate A is a support-only label-free checkpoint. It is intentionally not an
anomaly-scoring experiment and must not be used to claim anomaly improvement.

The lane exists to validate structural custody, runtime stability, and output
consistency for the revised VLM path before any later stage tries to use the
result as evidence for localization or scoring.

## Frozen Revision

- actual VLM: `Qwen/Qwen3-VL-2B-Instruct`
- design-stage 8B variant: frozen out for this lane
- model revision: `8964489`
- runtime: `vLLM 0.11.0`
- runtime reason: coordinate-grounding regression in `0.12.0`
- transformers: `4.57.1`
- qwen-vl-utils: `0.0.14`
- numpy: `1.26.4`
- hardware reason: TITAN RTX 24 GB requires an FP16-safe budget

This revision is part of the preregistration. It is not a reaction to the
result of Gate A.

## Frozen Inputs

- support-only
- label-free
- sentinel indices: `[0, 5, 10, 15]`
- objects: `can,vial,fabric,rice`
- dsba5 only
- no anomaly metrics
- no GT opening for decision making

## Gate A Checks

Gate A is limited to the following checks:

1. schema and parsing stability
2. repeated-run output stability
3. structural consistency across the frozen sentinel subset
4. runtime sanity on the frozen dsba5 GPU allocation

If Gate A passes, it only licenses the next step of the support-only lane.
It does not imply anomaly performance or defect localization quality.

Gate A can run only after the remote TCCR shard completes. It cannot license
Gate B unless TCCR earns `PROMISING_DIAGNOSTIC`.

## Reporting Contract

The report must stay `PENDING` until the run finishes. No score, ranking, or
qualitative conclusion may be invented in advance.
