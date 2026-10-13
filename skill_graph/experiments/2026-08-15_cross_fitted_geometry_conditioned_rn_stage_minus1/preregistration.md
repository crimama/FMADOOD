# CF-GC-RN Stage -1 Preregistration

Date: 2026-08-15
Status: `PRE-REGISTERED / NOT RUN`
Protocol: `cf-gc-rn-stage-minus1-20260815`

## Negative-Evidence Intake

The completed RNC-PA shards for rice, vial, and can all lost to same-run exact
RN1.  Its same-source null nearly absorbed the conditional signal, while offset
and neighborhood shuffles removed more signal.  The failed branch directly
converted regional mismatch into an anomaly penalty and inflated the normal
tail.  This experiment does not tune that score, threshold it, or fuse it with
RN1.  Geometry is used only to choose the normal reference whose single
descriptor residual becomes the final score.

## Dataset and Claim Boundary

- Dataset: MVTec AD2 single-image.
- Objects: `rice`, `vial`.
- Support: complete sealed `train/good` population (`rice=313`, `vial=291`).
- Evaluation: complete `test_public/good,bad` (`rice=42/90`, `vial=35/105`).
- Backbone: frozen DINOv2 ViT-L/14, normalized final native patch layer 23.
- Candidate and RN1 comparator use the identical support path set, feature
  cache, resolution, query population, and evaluator.
- No query adaptation, parameter update, public-label fitting, private-test
  access, or dense-map persistence is permitted.
- This is a reduced-object, full-support mechanism diagnostic.  It is not a
  same-condition SuperAD-16 comparison and cannot support a benchmark claim.

## Method

For every query patch, exact normalized RN top-5 candidates are retrieved from
the full support bank.  For a center patch `p`, the method excludes the center
and uses only valid rows in its 3x3 neighborhood.  Neighbor matches vote for a
bounded support-coordinate offset in `[-2,2]^2`.  A candidate must have at
least five valid neighbor rows.  Source identity is not a primary constraint.

The geometry consensus only changes reference selection.  It is never added
to the anomaly score.  The primary score is

```text
M1(p) = ||q(p) - z(selected geometry-consistent RN candidate)||_2
```

If no candidate has sufficient evidence, the method falls back to exact RN1.
Ties are resolved by the frozen manifest source index, patch index, and offset.
Here `cross-fitted` means leave-center-out prediction: the center descriptor
nominates its frozen RN candidates, but only neighboring rows may select among
them.  There is no learned support-fold threshold or query-derived fitting.

## Arms

- `B0_RN1`: exact normalized full-support RN1.
- `M1_CF_GC_RN`: neighbor-only, source-free geometry-conditioned reference.
- `C0_TOPK_NO_GEOM`: top-k without geometry; must equal B0.
- `N1_SOURCE_SHUFFLE`: deranged candidate source identities.
- `N2_OFFSET_SHUFFLE`: deranged spatial offsets.
- `N3_NEIGHBOR_SHUFFLE`: wrong neighbor rows with the center list preserved.
- `N4_INCLUDE_CENTER`: includes the center in consensus, testing anomaly
  self-absorption.
- `N5_SAME_SOURCE`: forces a single support source, testing the unsupported
  source-identity claim.

## Metrics and Gates

Primary endpoints are pixel pAUROC@0.05, best-threshold F1, and AU-PRO@0.05.
Diagnostics include candidate switch rate, conditional switch/excess AUC in
fixed B0 deciles, good-pixel p99.9 and FPR, valid-neighbor coverage, border
correlation, selected-source entropy, and compact provenance hashes.

Strict reduced-object pass requires:

- complete finite nonconstant maps and valid custody;
- no object pAUROC loss worse than -0.005 versus B0;
- macro M1-B0 >= +0.005 on pAUROC or F1 with the other non-negative;
- conditional switch/excess AUC >= 0.58 on rice and vial;
- M1 mechanism AUC exceeds offset and neighborhood nulls by >= 0.02;
- normal FPR and p99.9 do not materially inflate;
- the result is not explained by border coverage or the same-source arm.

Continuation-only requires non-negative endpoint movement on both objects,
conditional AUC >= 0.55, and dominance over both offset and neighborhood
nulls.  The branch hard-stops if either object loses more than 0.005 pAUROC,
the nulls absorb the effect, switches are a no-op, or good-pixel switching
drives normal-tail inflation.

## Execution

- dsba5 host GPU3 / container logical CUDA0: `rice`.
- dsba3 host GPU3 / container logical CUDA0: `vial`.
- Each shard runs from an immutable snapshot with pinned runner, core,
  aggregator, and support-manifest hashes.
- All maps are staged before GT is opened; only compact JSON evidence remains.
