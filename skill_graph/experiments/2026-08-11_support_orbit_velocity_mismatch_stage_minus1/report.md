# Support-Orbit Velocity Mismatch Stage -1 Report

Date: 2026-08-11
Status: `COMPLETED`
Verdict: `ORACLE_ONLY / NO_CONTINUE`

## Negative Evidence Intake

See `analysis.md` and `preregistration.md`. The merged Stage M aggregate is
valid, benchmark-incomparable, and stops at `ORACLE_ONLY / NO_CONTINUE`.

## Motivation

Test normal-flow velocity disagreement as anomaly evidence without transporting
support or query features.

## Implementable Design

Frozen in `preregistration.md`.

## Evaluation Alignment

Support-only synthetic diagnostic; `benchmark_comparable=false`; SuperAD,
RN-FMLK, TESTpublic, and real segmentation metrics are not applicable.

## Code Modification / Creation

Documentation update only in this pass. The execution-side fixes recorded in the
run artifacts - GPU-resident exact parity benchmarking, clean cache handling,
remote tar/stdin/same-file transfer, deterministic blur no-op fallback, and the
aggregate ratio contract - are engineering enablers, not method evidence.

## Added Code Evaluation

The authoritative merged summary and object summaries were inspected. The
primary aggregate reports `validity_ok=true`, `strict=false`,
`continuation=false`, and `benchmark_comparable=false`.

## Remote Execution

Primary distributed lanes were local GPU 0 `can/vial/walnuts` and dsba5 GPU 3
`fabric`. Supplementary parity runs are complete.

| Object | Primary M0 | Parity M0 | Max pAUROC diff | Max gap diff | Artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| can | 0.6625654754825345 | 0.6625652760754899 | 2.53229115332e-6 | 1.01340554559e-6 | `results/remote_runs/dsba5/sovm_stage_minus1_dsba5_20260811_v9/shards/can/summary.json` |
| fabric | 0.6442480186358903 | 0.644247890417822 | 3.70415149442e-6 | 3.15085349778e-6 | `results/remote_runs/dsba3/sovm_stage_minus1_dsba3_20260811_v2/shards/fabric/summary.json` |
| vial | 0.6903429565503281 | 0.6903429565503281 | 1.29239040536e-6 | 6.46195202569e-7 | `results/remote_runs/dsba3/sovm_stage_minus1_dsba3_20260811_v1/shards/vial/summary.json` |
| walnuts | 0.6999743177947801 | 0.6999743177947801 | 1.47511423276e-6 | 7.37557116381e-7 | `results/remote_runs/dsba5/sovm_stage_minus1_dsba5_20260811_v8/shards/walnuts/summary.json` |

All four parity summaries report `validity=true`. Support hash is unchanged.
Query access is `false`, test access is `false`, and the global max absolute
`M0` difference is `1.99407044654e-7`. The maximum pAUROC gap observed across
parity summaries is `3.70415149442e-6`, and the maximum gap difference is
`3.15085349778e-6`.

The single-object controller exit code `2` was expected missing-object aggregate
postprocessing, not shard failure.

Primary artifacts:

- [Merged summary](../../../results/remote_runs/merged/sovm_stage_minus1_distributed_20260811_v1/summary.json)
- [can shard summary](../../../results/remote_runs/merged/sovm_stage_minus1_distributed_20260811_v1/shards/can/summary.json)
- [fabric shard summary](../../../results/remote_runs/merged/sovm_stage_minus1_distributed_20260811_v1/shards/fabric/summary.json)
- [vial shard summary](../../../results/remote_runs/merged/sovm_stage_minus1_distributed_20260811_v1/shards/vial/summary.json)
- [walnuts shard summary](../../../results/remote_runs/merged/sovm_stage_minus1_distributed_20260811_v1/shards/walnuts/summary.json)

## SuperAD Baseline and Unified Metrics

All benchmark fields remain `N/A`/`null`. The merged aggregate is explicitly not
benchmark comparable.

## Results and Analysis

| Quantity | Value |
| --- | ---: |
| M0 macro pAUROC | 0.674283 |
| M0 local / broad pAUROC | 0.741302 / 0.607263 |
| M0 normal FPR | 0.045711 |
| B0 / B1 / B2 deltas | -0.000816 / -0.002893 / -0.002655 |
| N0 / N1 / N2 / N3 / N4 deltas | -0.001516 / -0.000474 / +0.000236 / +0.145896 / -0.000057 |
| B1 95% CI | [-0.005158, -0.000449] |
| N3 95% CI | [0.114077, 0.177550] |
| N3 wins | 4 / 4 |
| Selected-code entropy | 0.599870 |
| Max selected-code fraction | 0.713889 |

Interpretation:

- `M0_sovm_enum` is materially below the preregistered strict threshold and
  below the same-family oracle ceilings, so the learned velocity/code/time
  formulation is not supported here.
- `N3_spatial_source_shuffle` is the only strong positive control. Its strictly
  positive bootstrap interval and 4/4 wins show that spatial/source
  correspondence is a real mechanism signal.
- `N0`, `N1`, `N2`, and `N4` are non-positive or effectively null. The score is
  not rescued by constant, velocity, code-fit, or time shuffles.
- The low entropy and high max-code fraction indicate collapse toward a small
  subset of codes, not broad nuisance disentanglement.
- `O1_same_source` and `O2_full_oracle` are positive, but they are excluded from
  candidate gates by design and only explain the `ORACLE_ONLY` negative verdict.
- This does not imply that all transport or flow-style mechanisms are
  impossible. It does mean this particular learned velocity/code/time
  formulation is not yet the right pivot; spatial/source correspondence is the
  strongest next diagnostic direction.

## Continuation Assessment

- Strict gate: fail.
- Continuation gate: fail.
- Oracle-only condition: true.
- B1 bootstrap interval is strictly negative.
- The spatial/source shuffle control dominates the learned field, so Stage G
  remains closed.

## Conclusion

`ORACLE_ONLY / NO_CONTINUE`. Stage M is complete, but it does not authorize a
Stage G continuation or a claim promotion.

## Post-Conclusion Storage Cleanup

The merged result root is compact and `dense_maps_persisted=false`. The compact
evidence is retained in the merged summary and shard object summaries listed
above.
