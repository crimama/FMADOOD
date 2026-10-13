# Shift-Residual Full-Feature Ablation

Date: 2026-09-08  
Protocol: SRA-AD2-ALL8-SAME16-S3-v1  
Status: DESIGN_READY_FOR_ONE_REVIEW  
Verdict: PENDING_IMPLEMENTATION_AND_EXECUTION

## Claim and boundary

This protocol tests one frozen method matrix on MVTec AD2 single-image:
whether a learned full-feature support-residual scorer becomes shift-robust
when it (a) models a bounded normal-shift orbit, (b) exposes the shift-tangent
coordinates to the scorer, and (c) receives synthetic-anomaly
anti-absorption supervision. The final arm is A3.

The strict benchmark population is all eight public objects:
can, fabric, fruit_jelly, rice, vial, wallplugs, walnuts, and sheet_metal,
using the full TESTpublic/good,bad population. The primary metric is
seg_AUROC_0.05; oracleF1 and AUPRO.05 are reported with it. No test GT is
used for fitting, calibration, arm selection, or redesign.

This is a single production matrix, not a performance-triggered pilot loop.
All four arms run to completion unless a runtime, data-custody, or
comparability failure invalidates the run. Cumulative deltas are
order-dependent component contributions, not universal causal effects.

## Negative-evidence intake and motivation

The previous C1 result is held-only and conditional; public outcome remains
unknown. Its RN representation is four scalar layerwise nearest-neighbor
distances. A head cannot recover distinctions discarded by that map. The
candidate therefore changes the evidence source to learned full-vector support
residuals, rather than retuning a threshold, passive feature projection, or
RN-FMLK/kNN score.

The candidate also addresses a specific over-invariance risk: removing a
normal shift component can remove a real defect when the defect lies in the
same feature direction. A gate trained on synthetic anomaly masks is used to
retain anomaly evidence in the residual branch. This does not solve
identifiability when nuisance and anomaly effects are observationally
collinear.

## Dataset and reference custody

- Dataset: MVTec AD2 single-image.
- Data root: /home/hunim/Volume/DATA/mvtec_ad_2.
- Objects: exactly the eight objects listed above.
- Public population: all test_public/good and test_public/bad images.
- Reference manifest: the exact same 16 selected normal references per object
  used by same-run SuperAD-16 (HAS16, all eight objects). The complete
  manifest and SHA-256 are recorded in the run manifest; a truncated hash is
  not custody evidence.
- Train-only reference split: manifest entries 0..11 are R_train; entries
  12..15 are R_hold. R_hold is not decoded for fitting, nuisance-basis
  construction, or arm selection. It is released only after all heads are
  sealed for post-seal normal diagnostics/calibration.
- Public inference support: all 16 same-manifest references are used by every
  candidate arm and by SuperAD-16. The support-budget match is explicit even
  though the nuisance model is fit on R_train.
- For a training source index i, the support bank excludes
  i,(i+1)%12,(i+2)%12,(i-2)%12; this prevents self/donor role leakage.
  The new protocol regenerates these features and hashes and inherits no C1
  cache or checkpoint lineage.

Backbone and preprocessing are frozen DINOv2 ViT-L/14, layers
[5,11,17,23] (zero-indexed [5,11,17,23]), with the canonical
prepare_image -> extract_features path. Use short edge 672 for every object
except sheet_metal at 448, matching the paper-aligned contract. The same
preprocessing and masking policy is used for candidate and comparators where
applicable.

## Augmentation scope

Training synthetic anomalies are exactly the existing:

- G0_poisson_patch
- G1_material_blotch
- G4_frequency_local

G2_structural_scratch is not a training generator. Each synthetic mask is
used only as a weak training label; its semantic validity is an assumption,
not a measured fact.

Training shift is exactly the four global exposure gains
e={0,-0.0667,-0.1333,-0.2} with:

~~~
PIL bicubic resize -> ToTensor -> float32 gain 2^e -> Normalize
~~~

There is no clipping, rounding, uint8 conversion, or extra blur/field
augmentation. The shift basis is fit only from clean R_train pairs
(x,T_e x). The held public data are not transformed and are not used to
choose the augmentation support. The method therefore claims robustness only
to the bounded normal-shift model, not arbitrary domain invariance.

## Cumulative method matrix

Every arm instantiates the same trainable parameter allocation, initialization
family, optimizer, update count, and data order. Disabled branches receive
zero inputs and zero loss coefficient; they are not deleted. The matrix is:

| Arm | Enabled component added at this step | Route |
|---|---|---|
| A0 | Raw learned full-vector support-residual branch | q_raw only |
| A1 | Bounded normal-shift residual branch | q_raw + q_perp |
| A2 | Explicit shift-tangent coordinates | q_raw + q_perp + c |
| A3 | Synthetic-anomaly anti-absorption gate loss | A2 + L_AA |

A0 is not RN-FMLK: it learns a shared full-vector projection and local
residual head. RN-FMLK is an independent full-memory DINO kNN baseline.
A1 is not passive feature surgery: the residual branch and projection are
learned jointly with anomaly-map supervision. A3 does not add C1's
already-present scalar LATcons; its new loss supervises a gate on the
full-feature residual branch.

The final method is A3. A0->A1->A2->A3 is the only cumulative order.
No alternative ordering, coefficient sweep, early stopping, or post-hoc
method change is permitted.

## Fixed optimization

Use the same schedule for all four arms and all three paired seeds
20260906, 20260907, and 20260908:

- 576 updates per object/seed/arm.
- AdamW, lr=3e-4, weight_decay=1e-4.
- Betas (0.9,0.999), epsilon 1e-8.
- Batch size 1, gradient clip 1, no scheduler.
- Backbone forward bf16; feature algebra, projection, maps, head, and loss
  fp32.
- Identical parameter initialization per seed across A0..A3.
- Deterministic 144-cell order (base i, generator G0/G1/G4, gain e),
  repeated four times to form 576 updates.

The exact architecture, support residual, nuisance basis, and losses are in
implementable_design.md. All constants are fixed before implementation.

## Baselines and metric alignment

Run independent same-condition baselines:

1. SuperAD-16, using the exact 16-reference manifest and canonical
   same-condition evaluator.
2. RN-FMLK, the full-memory DINO kNN hard baseline/null under the same
   object population, split, backbone, preprocessing, and metric evaluator.

Baseline forwards may use their canonical implementations; candidate arms
share one query backbone forward. A custom policy used for all candidate arms
must be labeled as such and must not be called the exact paper Table-1
SuperAD implementation unless it is identical.

Report, in this order:

1. each object and seed separately;
2. per-object macro mean and standard deviation over the three paired seeds;
3. the all-eight-object macro mean over object means.

Never pool pixels or seeds across arms. Use:

- seg_AUROC_0.05;
- best-threshold oracleF1 as a labeled diagnostic;
- AUPRO.05.

The test GT stream is opened only after prediction hashes and shapes pass the
GT-free inventory. Metrics are streamed with truthfully labeled FP16 score
histograms/accumulators; no dense native prediction maps are persisted.

The method-claim gate requires comparable same-run SuperAD-16 and RN-FMLK
artifacts, all eight objects, all three metrics, no strong/control-category
collapse, and no claim-coverage or map-coverage failure. A weak cumulative
signal is reported as diagnostic only; it never triggers redesign in this
matrix.

## Execution and storage boundary

- Remote host: dsba1, host GPU 6 only, container CUDA slot 0.
- Verify remote availability in parallel, but do not launch from this design
  step. A busy/unavailable GPU defers execution; there is no fallback GPU.
- Run one production-route verification, then launch the complete matrix.
- Feature tensors are shared across A0..A3; one training cell is resident at
  a time. The cache is bounded and records exact bytes and manifest hashes.
- Query one image at a time, share one candidate backbone forward across all
  four heads, and retain only streaming metric state.
- Do not write dense native maps, NPY/TIFF map trees, or full-resolution map
  lists. Preserve compact metrics, configs, hashes, logs, and manifests.

## Output and decision labels

Each arm emits per-object/per-seed rows with
method_<metric>, superad_<metric>, delta_vs_superad, comparable,
rn_fmlk_<metric>, and coverage/finite/nonconstant diagnostics. The complete
schema is config_manifest.schema.json plus the output fields in
implementable_design.md.

Use KEEP or CONDITIONAL KEEP only if the full FMAD gates pass. Otherwise use
PROMISING_DIAGNOSTIC, CONTINUE_DIAGNOSTIC,
KILL_FOR_CLAIM / NO_CONTINUE, BLOCKED_BASELINE, BLOCKED_DATA, or
DESIGN_KILL_CLOSED_BRANCH as appropriate. This document authorizes no
remote run and no source implementation.
