# ECO-Audit v2 Stage -1 report

Date: 2026-08-28  
Protocol: `eco-audit-v2-ncfc-attribution-20260827-v1`  
Status: `COMPLETED / SUPPORT-ONLY / EXACT-REPEAT`  
Official aggregate verdict: `UNRESOLVED_OR_INVALID`  
Research verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`

## Negative Evidence Intake

NCFC-S1 previously found a stable normal J0 correction coordinate but failed
strict synthetic-anomaly commutation, with the dominant deficit on held base 1
outside the anomaly mask. Its immutable verdict remains
`NORMAL_ONLY_NO_ANOMALY_SIGNAL / END_NCFC`; ECO does not reinterpret or promote
that result.

ECO is not a new detector, score fusion, threshold change, adapter retune, or
backbone-training branch. It is a separately named frozen measurement audit
that asks whether the NCFC deficit came from inactive residual rows, spatial
boundary effects, the image operator, coordinate drift, or a genuine active
local mismatch. The likely failure basins were therefore metric invalidity and
operator confounding, not optimization failure. No LoRA, optimizer, gradient,
checkpoint, query/test sample, ground truth, or query-memory update was allowed.

## Motivation

The prior all-token cosine mixed high-energy anomaly-response tokens with many
nearly inactive rows. A backbone adapter would be scientifically premature if
the apparent commutation deficit could instead be explained by an unstable
low-activity component or by noncommutation already present in image space.
ECO fixed one support-derived activity threshold, separated active/inactive and
interior/boundary/outside strata, added a pixel-space operator audit, and kept
the original NCFC measurements and hard nulls.

The intended outcome was not a one-shot performance gain. It was a fail-closed
answer about whether there was enough valid evidence to justify the next small
mechanism experiment before any further backbone training.

## Implementable Design

- Dataset boundary: MVTec AD2 single-image, `can/train/good` only, under the
  fixed dsba3 project container data root.
- Evidence: immutable SuperAD-selected support 16 plus four support-disjoint
  held normal bases; no public/private test or GT access.
- Representation: offline frozen DINOv2 ViT-L/14 block 11, native-grid forward
  followed by deterministic `16x16` pooling; raw and standardized/L2 readouts.
- Construction: frozen `G2_structural_scratch`, J0 exposure at exactly
  `-0.1333` and `+0.1333` EV, both `T(A(x))` and `A(T(x))` orders.
- Activity: support-only equal-image rational empirical Q10. The final v11
  estimand uses the tokenized source footprint with occupancy `>=0.5`, giving
  every support image exactly mass `1/16`.
- Spatial attribution: 8-neighbor eroded interior, boundary, and outside at
  `16x16`; required `ACTIVE_INTERIOR` and `ACTIVE_BOUNDARY` strata fail closed.
- Operator gate: exact J0 uint8 reproduction plus fixed all-region, boundary,
  preclip, and boundary-error thresholds.
- Nulls and guards: original N1--N4, activity-preserving N5, RN geometry,
  scalar-only S90, exact held-0 repeat, 32-cell fail-closed reducer.
- Claim gate: only a valid, repeatable `GENUINE_LOCAL_MISMATCH_CANDIDATE` under
  a bounded pixel operator and raw/standardized plus order agreement could open
  the separately preregistered AGIAS Gate B. It could not directly authorize
  LoRA.

## Evaluation Alignment

The design directly tests the motivating attribution question by checking the
feature coordinate only where the frozen synthetic anomaly produces measurable
activity and by independently measuring whether J0 and G2 commute before the
backbone. Raw/standardized and TA/AT agreement prevent a favorable readout or
composition order from being selected after observation.

This is a Stage -1 frozen support-only diagnostic, not a segmentation method or
benchmark evaluation. It has no anomaly maps, TESTpublic population, or
segmentation predictions, so it cannot be compared to same-condition SuperAD,
RN-FMLK, `seg_AUROC_0.05`, or `seg_F1`. The frozen unified fields are therefore
`benchmark_comparable=false`, `support_only=true`, and `SuperAD=N/A`. No method
claim or paper-level keep decision is possible regardless of the diagnostic
label.

## Code Modification / Creation

The completed implementation comprises:

- `src/flow_tte/eco_audit_v2.py`: frozen statuses, stratified metrics, N5, and
  fail-closed contracts.
- `scripts/run_eco_audit_v2_stage_minus1.py`: P1 fixture, P2 support seal, held
  shard, custody, and compact serializer.
- `scripts/aggregate_eco_audit_v2_stage_minus1.py`: exact 32-cell validation,
  repeat binding, reducer, and compact aggregate.
- `scripts/verify_eco_audit_v2_preflight.py`: P0/P1/P2 hash-chain and pre-held
  authorization verifier.
- `scripts/run_eco_audit_v2_stage_minus1_remote.sh`: fixed-container dsba3
  controller with immutable snapshot, exact GPU mapping, shard/repeat/pull, and
  exact-snapshot cleanup.
- `tests/test_eco_audit_v2_core.py`,
  `tests/test_run_eco_audit_v2_stage_minus1.py`,
  `tests/test_aggregate_eco_audit_v2_stage_minus1.py`,
  `tests/test_verify_eco_audit_v2_preflight.py`, and
  `tests/test_run_eco_audit_v2_stage_minus1_remote.py`.
- The static contract, support commitment, held commitment, source manifest,
  and preregistration amendments under this experiment directory.

The v11 amendment changed one scientific definition before v11 execution:
support Q10 calibration moved from eroded interior to the raw tokenized source
footprint. The v12 change was controller-only: explicit shard indices replaced
a single-quoted positional path that made v11 shards collide. Held scientific
definitions and all frozen thresholds were unchanged in v12.

## Added Code Evaluation

- Remote P0 passed all 58 focused pure/static tests inside the immutable
  snapshot with `CUDA_VISIBLE_DEVICES=""`; its ledger records zero data, model,
  support, held, or GPU-result access.
- P1 emitted the exact eight target records and 32 cells twice with byte-
  identical fixture bytes and zero result-bearing access.
- P2 passed the strict source/custody chain, encoded exactly 16 support images,
  sealed and byte-identically reloaded the pre-held manifest, and recorded
  `held_opened=0`.
- The final source-manifest SHA-256 is
  `d4a3e5e94bf4b00f2b07e9c52feae10ea64b42ca1a38b1366606a042f90d8a7e`;
  runner SHA-256 is
  `3fc26e931dc0a28a4187dc1b5b1f17b7c1aaf0b9362d82f3c6c145214f508634`;
  v12 controller SHA-256 is
  `66ca7f8e09b833eda20a72ee80962bb9ede50f40f288413088da058f8ac8d8a8`.
- Independent verification recomputed all retained file hashes, shard custody,
  repeat identity, schema/counts, forbidden flags, and dense-artifact absence.
  The only verification gap is that the saved chain names its original remote
  artifact root, so the complete verifier cannot be replayed verbatim against
  the pulled local directory. Direct hash and semantic checks of the pulled
  files passed.

## Remote Execution

The completed immutable run is
`eco_audit_v2_dsba3_20260828Tfrozen12Z_v12` in fixed project container
`hun_fsad_tta_012` on dsba3 GPUs 0--3. Each held index ran on its matching GPU,
followed by an exact held-0 repeat on GPU0. dsba2 GPU6 was not used because the
frozen controller and preregistered resource boundary selected dsba3 0--3.
No other container, process, PID, user, or resource was enumerated or modified.

Run lineage:

1. v10 passed P0/P1 but stopped in support-only P2 because 11/16 G2 masks had
   no eroded interior at `16x16`; no held access occurred.
2. v11 passed P0/P1/P2 after the source-footprint Q10 amendment. Its held
   controller sent all shards to a literal `held_$1` path, so the immutable run
   was abandoned; no held metric was opened for decision or aggregation.
3. v12 repaired only explicit shard path expansion, reran the full chain, wrote
   four distinct held shards, reproduced held 0 byte-for-byte, and aggregated.

The compact pull is retained at
`results/remote_runs/dsba3/eco_audit_v2_dsba3_20260828Tfrozen12Z_v12`
(27 files, 1,311,462 bytes). The exact held-0 and repeat artifact SHA-256 is
`2966de6def511bd21278e7ce247d97a87bed954435f5760fa7cc6b6d0d1af8dc`.
The aggregate SHA-256 is
`e5c70cdcef737fe37bfe6f4dd14dd4971bb7f929d652c622ea864e032534e78e`.

## SuperAD Baseline and Unified Metrics

| field | value | interpretation |
| --- | --- | --- |
| `superad` | `N/A` | no benchmark or detector comparison |
| `support_only` | `true` | support plus committed held-normal audit only |
| `benchmark_comparable` | `false` | no TESTpublic/GT or segmentation maps |
| `superad_seg_AUROC_0.05` | `N/A` | not measured |
| `method_seg_AUROC_0.05` | `N/A` | not measured |
| `delta_vs_superad` | `N/A` | comparison prohibited |
| `seg_F1` / map coverage | `N/A` | no result-bearing anomaly map |
| RN-FMLK/full-memory baseline | `N/A` | mechanism audit; original exact-RN guard retained |

The aggregate is structurally valid with eight target records and 32 cells,
but its sole label is `UNRESOLVED_OR_INVALID=32`. This structural validity is
execution evidence, not method-performance evidence.

## Results and Analysis

### Support calibration succeeded

The source-footprint amendment solved the P2 calibration blocker without
drops, fallback masks, per-image thresholds, or enlarged masks. All 16 support
images contributed 8--17 source tokens, every image retained mass `1/16`, and
the total mass was exactly one. The sealed thresholds were:

- raw `tau=1.0152778625488281`;
- standardized `tau=0.010834874585270882`.

This is a real positive engineering and measurement result: activity is now
calibrated reproducibly from every support image. It does not by itself validate
the held spatial estimand.

### Blocker 1: held interior geometry is invalid in every cell

| held base | readout | interior / boundary / outside tokens | active tokens | active coverage |
| ---: | --- | ---: | ---: | ---: |
| 0 | raw | `0 / 29 / 227` | `43` | `0.98717` |
| 0 | standardized | `0 / 29 / 227` | `45` | `0.98840` |
| 1 | raw | `0 / 40 / 216` | `12` | `0.96359` |
| 1 | standardized | `0 / 40 / 216` | `12` | `0.96737` |
| 2 | raw | `1 / 35 / 220` | `77` | `0.99184` |
| 2 | standardized | `1 / 35 / 220` | `70` | `0.98959` |
| 3 | raw | `0 / 31 / 225` | `30` | `0.98153` |
| 3 | standardized | `0 / 31 / 225` | `29` | `0.97970` |

Six of eight target records have zero eroded-interior tokens and two have one.
Consequently `ACTIVE_INTERIOR` is `EMPTY_STRATUM` in 24/32 cells and
`INVALID_SMALL_STRATUM` in 8/32. The required status fails in 32/32 cells even
though every active-global and active-boundary coordinate test passes. The
dominant issue is therefore the mismatch between small G2 structures,
`16x16` tokenization, and 8-neighbor erosion—not failed DINO encoding or an
unstable activity threshold.

### Blocker 2: the image-space operator is independently confounded

| operator check | observed | frozen gate | pass count |
| --- | ---: | ---: | ---: |
| exact uint8 J0 reproduction | bitwise true | required | `32/32` |
| `kappa_all` | `0.12193--0.17873` | `<=0.10` | `0/32` |
| `kappa_boundary` | `0.12193--0.17873` | `<=0.20` | `32/32` |
| preclip violation | `0--0.30382` | `<=0.01` | `16/32` |
| boundary commutator-error share | `0.78802--1.00000` | `<=0.25` | `0/32` |
| bounded for attribution | false | required | `0/32` |

All negative-exposure cells satisfy the preclip limit and all positive-exposure
cells fail it, but every sign/order/readout cell exceeds the all-region
commutator threshold and concentrates most commutator error at the boundary.
Thus fixing only the interior geometry would still leave all cells
`OPERATOR_CONFOUNDED`. Conversely, changing only J0/G2 rendering would not make
the current eroded-interior evidence valid. These are two independent barriers.

### Surviving positive diagnostic signal

The result is not a feature collapse:

- active coordinate gate: `32/32` pass;
- active-boundary coordinate gate: `32/32` pass;
- N5 hard comparison: `32/32` pass;
- active coverage: `0.96359--0.99184`;
- active CKA: `0.99948--0.99998`;
- active token-geometry Spearman: `0.99712--0.99960`;
- exact held-0 repeat: byte-identical.

The inactive component is much less aligned than the active component: active
cosine is `0.99525--0.99945`, whereas inactive cosine is
`0.65830--0.87125`. Boundary target energy and interaction error are also
dominant. This supports the motivation for activity-aware measurement, but the
preregistered fail-closed geometry and operator gates correctly prevent it from
being promoted as genuine local mismatch evidence.

For diagnosis only, removing both invalidity and operator precedence after the
fact would yield 12 `NO_LOCAL_FAILURE` and 20
`LOW_ACTIVITY_METRIC_COMPONENT` cells. Those are counterfactual labels, not
official results, and they are not used for a claim, gate, or method decision.

## Continuation Assessment

This experiment does not support a strict method claim and keeps AGIAS Gate B,
all sidecars, and all backbone LoRA closed. It does justify exactly one more
small diagnostic because support calibration, active-global geometry, N5, and
repeatability all succeeded while the failure was localized to a specific
required stratum.

The single next experiment should be separately named `ECO-G1` and change only
the held spatial estimand: make the already defined `16x16` tokenized source
footprint (`occupancy >=0.5`) the required active source stratum, while retaining
the current eroded interior/boundary/outside quantities as descriptive fields.
Keep the same eight target records, model/readouts, G2/J0 construction,
thresholds, operator audit, N1--N5, custody, and no-training boundary.

The weak continuation signal is successful geometry feasibility: all eight
target records must contain at least four required active-source rows and all
raw/standardized plus TA/AT statuses must be valid. Any target below four rows,
any readout/order disagreement, or any N5 failure hard-stops the geometry line.
Even if ECO-G1 passes, the existing pixel-operator failure remains a separate
blocker; it may authorize only a later, separately preregistered one-change
operator sentinel, never LoRA or benchmark access.

## Conclusion

`KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.

ECO v12 is a valid, reproducible experiment with an invalid scientific
attribution result. It successfully repaired support activity calibration and
showed that the active feature coordinate is highly stable, but it also proved
that the current `G2 -> 16x16 -> erosion` required interior and the current
J0/G2 image composition are independently unsuitable for causal attribution.
The exact ECO configuration is closed; LoRA remains unauthorized. The retained
assets are a tested fail-closed audit stack, immutable custody chain, compact
repeatable evidence, and a one-change geometry feasibility hypothesis.

## Post-Conclusion Storage Cleanup

No `anomaly_maps/`, `.npy`, `.npz`, `.pt`, `.pth`, `.pkl`, or other dense
result artifact is present in the compact pull. After this report was recorded,
the controller removed only the exact completed v12 immutable snapshot at
`/workspace/isolated_runs/eco_audit_v2_dsba3_20260828Tfrozen12Z_v12`.
An exact-path remote check confirmed that the snapshot is absent while
`/workspace/results_remote/eco_audit_v2_dsba3_20260828Tfrozen12Z_v12/held/complete.txt`
remains present. The compact remote result and local pull are preserved.
