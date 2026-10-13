# Defect-Side Cross-Image-Calibrated Region Completion Stage -1

Date: 2026-08-11  
Strict verdict: `KILL_NO_CONTINUE`  
Aggregate validity: `FAIL`

## Negative Evidence Intake

This compact diagnostic tested whether defect-side cross-scale excess could
recover anomaly seeds that the fixed support-only P16 H+DVT anchor misses. It
was designed after normal-side calibration and direct DINO attention variants
failed to predict whether query defect evidence would be preserved.

The result is negative. Two-phase TCCR collapses close to chance localization,
is matched or exceeded by phase/null controls, and severely degrades every
canonical object relative to P16. The frozen aggregate is additionally invalid
because three duplicated-`can` scientific scalars exceeded the preregistered
cross-machine tolerance.

## Method and Frozen Scope

- Dataset: MVTec AD2 single-image.
- Support: fixed SuperAD-16 manifest, SHA-256
  `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`.
- Query selection: deterministic `16 good + 32 bad` public images per object.
- Backbone: frozen DINOv3 H+/16 revision
  `c807c9eeea853df70aec4069e6f56b28ddc82acc`.
- Required anchor: exact `P16_H+DVT_anchor`, including the full frozen
  H+DVT feature fusion and raw latent 1-NN score path.
- Coarse observation: 672 full image, fused layers `[7,15,23,31]`.
- Fine observation: layer 7 at 1344 through overlapping 672 crops, with two
  deterministic phases.
- No query-time parameter or memory update; no private split access.
- All dense maps stayed in RAM. Public ground truth was opened only after all
  arms for a query had been staged.

Implemented arms were the exact preregistered 13 arms: P16, raw G0, raw tiled
fine kNN, lift-only, calibrated max/mean, matched coarse-to-fine predictor,
one-phase excess, two-phase TCCR, geometry shuffle, phase shuffle, RGB edge,
and equal-compute duplicate tiles. No arm or gate was omitted or retuned.

## Protocol Stage Ledger

| Stage | Result |
|---|---|
| 0. Negative-evidence intake | Prior normal-side and direct attention interventions did not identify safe defect preservation. |
| 1. Mechanism definition | Cross-scale counterfactual excess was isolated from lifting, calibration, predictor, phase, geometry, RGB, and compute effects. |
| 2. Feasibility | Exact DINO snapshot, support custody, RAM-only maps, and public-GT ordering were implementable. |
| 3. Preregistration | Objects, shards, hashes, 13 arms, parity tolerance `1e-3`, and continuation/kill rules were frozen. |
| 4. Implementation | Exact runner and strict aggregator were implemented without changing the frozen P16 score path. |
| 5. Verification | Focused post-run suite: `60 passed`; launch-time hashes and smoke checks passed. |
| 6. Execution | Local and dsba5 shards completed all `240/240` query evaluations; `dsba3` was not used. |
| 7. Aggregation | Duplicated-`can` parity was checked and the four canonical objects were assembled without private data. |
| 8. Decision | Strict verdict `KILL_NO_CONTINUE`; no replication or claim-tier stage is licensed. |

## Execution and Resource Allocation

| Shard | Host / GPU | Objects | Query evaluations | UTC interval | Elapsed |
|---|---|---|---:|---|---:|
| `run_local_gpu0_exact_v1` | local GPU 0, RTX 4090 | can, fabric, vial | 144 | 18:31:08–22:25:16 | 3 h 54 m |
| `tccr_stage_minus1_exact_20260811_v1` | dsba5 host GPU 3 / container CUDA 0, TITAN RTX | rice, can parity | 96 | 18:31:21–00:11:22 | 5 h 40 m |

Runner SHA-256:
`5f998f1e2cfb36bd706ca0d5e5d2e36439a7f9c2fa3a02d689c27af2e90bfec7`.
The local and remote manifests both record `test_private_accessed=false`,
`dense_maps_persisted=false`, and
`ground_truth_accessed_after_maps_staged=true`.

## Strict Validity Result

The aggregate failed before positive gates could be evaluated.

- Phase shuffle matched the TCCR gain on `can`, `fabric`, `vial`, and `rice`.
- Two-phase preservation was `0.0` on `fabric` and `rice`, below the `0.90`
  floor.
- `vial` good-tail FPR reduction was `0.031763`, below the `0.20` floor.
- Duplicated-`can` maximum absolute scientific-scalar drift was `0.001957`,
  above the frozen `0.001` tolerance.

The three parity failures were narrow:

| Duplicated-`can` scalar | Local | dsba5 | Absolute delta |
|---|---:|---:|---:|
| geometry-shuffle image AP | 0.614417 | 0.612759 | 0.001657 |
| geometry-shuffle image AUROC | 0.425781 | 0.423828 | 0.001953 |
| good-tail FPR reduction | 0.630654 | 0.632611 | 0.001957 |

The principal P16, TCCR, and phase-shuffle AU-PRO parity deltas were all below
`8e-7`, but the preregistered tolerance is conjunctive; the aggregate remains
invalid and cannot support a positive claim.

## Descriptive Results

The following canonical macro values use local `can/fabric/vial` and remote
`rice`. They explain the failure but are not a substitute for the failed
strict aggregate gate.

| Arm | Macro AU-PRO | Macro pAUROC@0.05 | Macro pixel F1 | Component-seed recall |
|---|---:|---:|---:|---:|
| calibrated coarse/fine max | 0.598150 | 0.835293 | 0.452307 | 0.550582 |
| calibrated coarse/fine mean | 0.575218 | 0.807352 | 0.391833 | 0.543836 |
| P16 H+DVT anchor | 0.552158 | 0.806379 | 0.448470 | 0.521904 |
| raw G0 layer 7 | 0.387208 | 0.625129 | 0.126091 | 0.447631 |
| raw tiled fine kNN | 0.311356 | 0.604053 | 0.094352 | 0.369603 |
| matched predictor residual | 0.156367 | 0.538818 | 0.059909 | 0.234338 |
| phase shuffle | 0.033058 | 0.492101 | 0.010888 | 0.116870 |
| equal-compute duplicate | 0.025643 | 0.493025 | 0.012526 | 0.115441 |
| one-phase excess | 0.025643 | 0.493025 | 0.012526 | 0.115441 |
| **two-phase TCCR** | **0.013620** | **0.491648** | **0.011798** | **0.045195** |

TCCR degrades every object relative to P16:

| Object | AU-PRO: TCCR / P16 | pAUROC: TCCR / P16 | Pixel F1: TCCR / P16 | Seed recall: TCCR / P16 |
|---|---|---|---|---|
| can | 0.000379 / 0.117426 | 0.487296 / 0.555343 | 0.000187 / 0.000821 | 0.000000 / 0.116279 |
| fabric | 0.024947 / 0.663172 | 0.496659 / 0.965888 | 0.009984 / 0.627618 | 0.027523 / 0.550459 |
| vial | 0.017689 / 0.825688 | 0.493428 / 0.772458 | 0.033661 / 0.462075 | 0.046875 / 0.718750 |
| rice | 0.011467 / 0.602347 | 0.489209 / 0.931825 | 0.003362 / 0.703369 | 0.106383 / 0.702128 |

Support LOO q99/q999 tail coverage passed for every object, and the fixed
synthetic microdefect retention ratio was `1.0` for all four. The collapse is
therefore not explained by a missing support-tail calibration check or by the
synthetic retention guard. The excess/residual construction itself removes
defect evidence, while a phase shuffle can reproduce its small apparent gain.

The calibrated max arm is descriptively strongest, but it is a comparator in
an aggregate that failed validity. It is not promoted or retuned from this
result.

## Conclusion and Continuation Boundary

`KILL_NO_CONTINUE`.

Defect-side two-phase cross-image-calibrated residual completion is not a
nuisance-invariant, defect-preserving replacement for P16. It turns strong
object-level localization into near-chance maps and does not beat its phase or
equal-compute controls. No TCCR replication, Gate B, benchmark claim, or
deployment claim is licensed.

DB-FSPC Gate A remains allowed only as the already-preregistered support-only
VLM structural/runtime audit. Even if Gate A passes, it cannot rescue TCCR or
authorize DB-FSPC Gate B because TCCR did not earn `PROMISING_DIAGNOSTIC`.

## Artifacts and Cleanup

- Local summary SHA-256:
  `3f02e14b6a21a08f19779839acf4a82c7a90b869dbee93affefa8a590a07fc15`.
- Remote summary SHA-256:
  `78c3d96fbf369a77b598116db328a61604c18c5e5833b07c22269086e17afeca`.
- Aggregate summary SHA-256:
  `22164ffcee14719a32c402a963a2c71fd3d34dd65bc1484ef81faf00e8be79c8`.
- Aggregate verdict SHA-256:
  `09dfad9af0905c541cb649bc14749253bfc17ad2642578637c05a9cacd4f62aa`.

No `anomaly_maps/` directory exists in the local, pulled remote, or aggregate
artifacts. The checksum-verified dsba5 TCCR source snapshot was removed after
pull; compact remote results and local evidence were retained.
