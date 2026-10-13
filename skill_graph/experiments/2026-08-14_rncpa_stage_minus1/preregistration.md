# RNC-PA Stage -1 Preregistration

Date: 2026-08-14
Status: `PRE-REGISTERED / NOT RUN`
Protocol: `rnc-pa-stage-minus1-20260814`

## 1. Negative-Evidence Intake

This experiment begins from three closed or constrained branches.

- Transport-Quotient RN lost to exact final-layer RN on all eight MVTec AD2
  objects (`M0-B0=-0.097088` macro pAUROC) and was nearly absorbed by its
  shuffled-tangent null (`M0-N0=+0.002303`).
- SPCT suggested that source/part correspondence may be useful, but its
  tangent direction was not independently causal enough for promotion.
- Map-level region completion was absorbed by simpler spatial processing.

Therefore this protocol removes tangent projection, flow fitting, learned
deformation, score gating, synthetic anomaly training, and morphology. It asks
one narrower question: do exact RN matches become more diagnostic when nearby
query patches must agree on one normal source and one relative translation?

## 2. Claim and Dataset Boundary

- Dataset: MVTec AD2 single-image.
- Support: the complete per-object `train/good` population from the sealed
  all-eight manifest.
- Evaluation: complete `test_public/good,bad` for `can`, `rice`, `vial`, and
  `walnuts`; ground truth is opened only after all arm maps exist.
- Backbone: frozen `dinov2_vitl14`, normalized final native patch layer 23.
- Objects and support counts: `can=412`, `rice=313`, `vial=291`,
  `walnuts=432`.
- This is a reduced-object, full-support mechanism diagnostic. It is
  `benchmark_comparable=false`, has no same-condition SuperAD comparison, and
  cannot support a benchmark or superiority claim.
- No `test_private`, query adaptation, memory update, or parameter update is
  permitted.

## 3. Frozen Method

For every query patch, exact normalized RN top-5 matches are computed against
the full final-layer support bank. The exact top-1 distance is `B0_RN1` and is
an invariant lower bound.

`M0_RNCPA` uses a clipped 3x3 query neighborhood. Exact RN center matches
nominate at most four support sources. For each source and each global relative
translation `(dy,dx)` in `[-2,2]^2`, the query neighborhood is mapped
one-to-one into that same support source. Every row may instead choose a
dustbin with cost `B0(row)+tau`. The lowest regional energy supplies the
center-row structured cost, and the emitted center score is
`max(B0(center), structured_cost)`.

The implementation must not learn or fit a deformation field. Translation is
a finite enumeration used only to test local correspondence consistency.

## 4. Support-Only Calibration

`tau` is fixed before TESTpublic scoring. At most 16 support sources are chosen
by SHA(relative-path) order and at most 256 part positions are selected by
deterministic even spacing. Cross-source same-part normalized distances define
the q95 dustbin increment. Calibration paths, counts, quantile, and hash are
recorded. TESTpublic images and labels never influence `tau`.

## 5. Arms and Causal Controls

- `B0_RN1`: exact normalized final-layer top-1 RN.
- `C0_independent`: the same candidate evidence without regional coupling.
- `M0_RNCPA`: source-consistent, translation-consistent partial assignment.
- `N1_source_shuffle`: preserves unary distances/ranks while deranging source
  identities.
- `N2_offset_shuffle`: preserves sources/distances while deranging relative
  translations.
- `N3_neighborhood_shuffle`: preserves the center evidence but replaces its
  surrounding neighborhood deterministically.
- `N4_no_dustbin`: forces every regional row onto normal memory.

All nulls must preserve row count and exact unary evidence as closely as their
intervention permits. Randomness is fixed by protocol/object/query identity.

## 6. Metrics and Barrier

Primary metrics are pixel `pAUROC@0.05`, F1, and `AU-PRO@0.05`, computed by the
existing native evaluation path. All arm maps are completed before public GT is
read. Only compact scalar summaries and provenance hashes may persist.

The mechanism metric is conditional invalidity. Pixels are stratified into
fixed B0-distance deciles; within eligible deciles, the AUC of
`M0_RNCPA-B0_RN1` for bad versus good pixels is computed and macro-averaged.
The same statistic is computed for structural nulls. It is diagnostic only and
is never used to tune the method.

## 7. Gates

Strict mechanism pass requires all of:

- macro `M0-B0 >= +0.01` on pAUROC and F1;
- macro `M0-null >= +0.01` pAUROC for each of N1, N2, and N3;
- conditional invalidity AUC `>=0.55` and at least `+0.02` above each
  structural null;
- no object pAUROC loss worse than `-0.02` versus B0;
- complete coverage, finite nonconstant scores, valid custody, and no leakage.

A continuation-only result may be retained when M0 improves B0 and at least
two structural-null comparisons but misses one strict magnitude threshold.
The branch is killed if a structural null absorbs M0, conditional AUC is at
chance, custody fails, or improvement is explained only by forcing rows without
the dustbin.

## 8. Execution Plan

- dsba3 host GPU 3 / container logical CUDA 0: `can -> walnuts`.
- dsba5 host GPU 3 / container logical CUDA 0: `rice -> vial`.
- Each server receives an immutable snapshot with pinned runner, core,
  aggregator, and manifest hashes.
- Preflight proves data counts, host/container GPU UUID identity, free GPU
  memory, available RAM, and absence of an active conflicting process.
- Aggregation is fail-closed until all four compact object shards from the same
  snapshot are present.

## 9. Retention and Cleanup

Keep the preregistration, compact JSON shards, checksum manifests, aggregate
summary, report, and log. Delete transient maps, features, tensors, NumPy
arrays, and superseded retry artifacts after checksum verification. Update the
experiment report, global skill graph log/index, and task ledgers with the
result and decision.
