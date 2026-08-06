# DSCR Paired Acquisition Interaction Audit

Date sealed: 2026-07-16, before PAIA maps or condition metrics are generated  
Status: pre-registered forensic diagnostic  
Maximum verdict: `KILL_FOR_CLAIM / PROMISING_DIAGNOSTIC`

## Negative Evidence Intake

The global high-resolution density branch remains closed for method claims. The
preceding compact DSCR cell showed that raw density NLL (D0) and its normal-tail
calibration (D0c) both lost to the same-memory global cosine 1-NN hard null (G0),
with class-harm violations on `can` and `vial`. This audit does not modify,
calibrate, tune, fuse, or promote that method.

The only unresolved causal question is whether the D0-G0 loss is already present
under the matched `regular` acquisition, or is selectively induced by the paired
`overexposed/shift_1/shift_2` acquisition changes. The earlier maps were removed
after evaluation, so this interaction cannot be reconstructed from retained
compact metrics.

This is not a retread of conditional flow, coordinate conditioning, foreground
mixtures, threshold tuning, morphology, whitening, Mahalanobis scoring, capacity
sweeps, registration, reconstruction, or hard-local correspondence. It freezes
the already-tested G0 and D0 arms and uses them only as causal probes.

## Motivation and Identification Claim

For object `o`, condition `c`, and metric `M`, define

```text
delta_M(o,c) = M(D0 | o,c) - M(G0 | o,c)
interaction_M(o) = mean_{c in shifted} delta_M(o,c) - delta_M(o,regular)
shifted = {overexposed, shift_1, shift_2}
```

- If `delta(regular)` is already harmful, unconditional global likelihood is
  structurally misaligned even without acquisition shift.
- If `regular` is retained but `interaction` is strongly negative, acquisition
  shift is the discriminating failure mechanism and only a genuinely new
  acquisition-aware evidence source remains conceptually admissible.
- If both are harmful, the branch stays closed; conditioning a scorer that already
  fails on `regular` would only complicate a killed family.

## Implementable Design

- `target_dataset=MVTec AD2 single-image`
- dsba3 data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- Objects: `fabric`, `rice`, `vial`.
  - `rice`: sole positive D0 pAUROC object.
  - `vial`: largest negative-transfer object.
  - `fabric`: mixed control with lower pAUROC but higher oracle F1.
- Population per object: exactly one good and one bad source identity, each under
  `regular`, `overexposed`, `shift_1`, and `shift_2`.
- Support: byte-identical P16-random seed-0/fold-0 paths from the prior manifests;
  12 regular memory normals plus four disjoint regular held-out normals.
- Feature geometry: frozen DINOv3 H+ layer 7, native 512 crops, 1024 crop input,
  identical tiling and stitching.
- Arms used for the gate:
  - `G0`: unrestricted global cosine 1-NN residual.
  - `D0`: frozen two-coupling-layer, three-epoch density NLL with 32,768-token cap.
- The retained runner may emit D0c for code-path parity, but PAIA ignores it.
- All three objects run on the same dsba3 retained snapshot to remove cross-host
  implementation ambiguity.

No new method code is allowed. The only new code is an offline evaluator that
groups existing maps by acquisition condition and computes condition metrics.

## Baseline and Comparability Contract

G0 is the paired immediate hard null and the only valid comparator for this causal
audit. A same-condition SuperAD run does not exist for this three-object compact
population, so `comparable=false`; this cell cannot produce a method-superiority
claim, `KEEP`, or `CONDITIONAL KEEP`. Historical SuperAD/RN-FMLK values remain
context only and are not subtracted from PAIA metrics.

## Metrics and Output Contract

Primary metrics, computed separately for each object-condition pair:

1. pixel `seg_AUROC_0.05`;
2. pixel oracle segmentation F1.

Pixel AP is explanatory only. The evaluator reports both native float32 TIFF
metrics and the prior float16 evaluation path. Float32 is primary for the causal
interaction; float16 is a parity/quantization diagnostic.

Required output fields include per-condition G0/D0 metrics, `delta_vs_g0`, object
`interaction`, arm-ID parity, finite/nonconstant status, query coverage, and
`comparable=false`. Each condition must contain exactly one good and one bad map
for both arms.

The millions of pixels are spatially dependent and are not treated as independent
replicates. With only one good and one bad source identity per object, this is a
mechanism preflight without a confidence-interval or generalization claim.

## Pre-registered Decision Gate

The acquisition-conditioning hypothesis remains admissible for exactly one later
design diagnostic only if all conditions pass:

1. Three-object macro `delta_pAUROC(regular) >= 0` and
   `delta_F1(regular) >= 0`.
2. No object at `regular` loses more than `0.03` pAUROC or `0.05` F1.
3. Macro `interaction_pAUROC <= -0.03`, with a negative interaction on at least
   two of three objects.
4. Macro `interaction_F1 < 0`.
5. Complete, finite, nonconstant G0/D0 coverage for all 24 logical queries.

Passing yields `KILL_FOR_CLAIM / PROMISING_DIAGNOSTIC`: D0 remains rejected, but
the acquisition-specific causal hypothesis survives. It does not authorize a
broad sweep.

Any of the following yields `KILL_FOR_CLAIM / NO_CONTINUE`:

- regular retention fails and a per-object harm bound is violated;
- macro `interaction_pAUROC > -0.03`;
- fewer than two objects have negative pAUROC interaction;
- D0 loss is comparable under regular and shifted conditions;
- validity or coverage fails.

## Verification and Remote Execution Contract

Before execution, verify the exact support/query paths, runner and method hashes,
dsba3 container/data/model availability, idle host GPU allocation, evaluator unit
tests, and CLI preflight. Execution uses dsba3 GPUs 0--2 with one object per GPU.
No method files are synchronized over the retained remote snapshot.

After metrics, manifests, logs, and the report are pulled locally, all regenerable
`anomaly_maps/` directories from this PAIA run are deleted remotely and locally.

