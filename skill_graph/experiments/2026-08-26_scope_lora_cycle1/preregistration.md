# SCOPE-LoRA Cycle 1 preregistration

Date: 2026-08-26  
Protocol: `scope-lora-cycle1-support-intervention-20260826`  
Status: `PREREGISTERED / NO CYCLE-1 RESULTS INSPECTED`  
Seed: exactly one, `20260826`

## Objective

This is the first bounded experiment after TARCD-LoRA-S3-VR.  S3 repaired its
zero-start gradient pathology but retained a shared image-only adapter, lost
correction magnitude, received no strict synthetic contribution, and lost to a
static mean in `10/12` cells.  Cycle 1 asks two ordered questions:

1. **P0 projector feasibility:** does a train-fit rank-4 span retain a useful
   part of the paired shift correction on support-disjoint held normal and
   held-generator counterfactual samples?
2. **P1 oracle routing:** conditional on P0 being useful, can an oracle-coded
   support-carrier QKV-LoRA sidecar produce a better projected correction than
   matched projected static and output-side controls?

The experiment seeks a positive trend and a single next failure mode.  It is
not required to reach SuperAD performance in this cycle.

## Frozen custody

- Dataset: `/home/hunim/Volume/DATA/mvtec_ad_2`.
- Objects: `can`, `fabric`, `vial`, `walnuts`.
- Support: exact 16 `train/good` paths per object from the frozen manifest with
  SHA-256
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Deterministic primary split: 12 fit and 4 calibration bases.  Existing
  support-disjoint held bases and held synthetic generator are preserved.
- Inputs may contain only support `train/good` images and deterministic
  descendants `x,T(x),A(x),T(A(x))`.  Public/private test, GT, query-memory
  update, and benchmark threshold selection remain forbidden.
- Backbone: frozen DINOv2 ViT-L/14.  RN memory and final token normalization
  remain identical to TARCD S2/S3.

## P0: projector feasibility

For frozen normalized tokens `Z0`, fit correction rows are
`d=Z0(x)-Z0(T(x))`.  Stack all fit rows across the three frozen shift families,
compute deterministic rank-4 SVD with sign canonicalization and QR, and set
`P=UU^T`.  No held row contributes to `U`.

Evaluate on held rows:

- explained energy `EV=||dP||^2/||d||^2`;
- paired projected teacher `PT-P`, which adds `dP` to the shifted residual;
- shift-family projected fit mean `N3mu-P` using the same `U`;
- exact complement error, normal q99 reduction, correction norm, and whole /
  P-parallel anomaly residual retention on the held generator.

The previous design's simultaneous `EV >= 0.30` and full-target correction
cosine `>= 0.90` are not used: for an orthogonal projection,
`cos(dP,d)=sqrt(EV)`, so those thresholds are mutually inconsistent unless
`EV >= 0.81`.  Cycle 1 reports this identity explicitly and gates on energy
plus downstream behavior instead.

P0 is a `POSITIVE_PROJECTOR_SIGNAL` only if all structural checks pass,
object-median EV is positive for all four objects, aggregate median EV is at
least `0.20`, PT-P aggregate normal q99 reduction is positive, and PT-P does
not worsen whole anomaly cosine by more than `0.05` or anomaly norm outside
`[0.70,1.40]`.  `EV < 0.20` diagnoses an insufficient span and stops P1; it
licenses exactly one rank/basis repair in Cycle 2, not a sweep.

## P1: oracle-routed support-carrier LoRA

P1 runs only after `POSITIVE_PROJECTOR_SIGNAL`.

- Frozen prefix: blocks 0--7.
- Support carrier: robust aggregate of fit-support block-7 tokens.  The LoRA
  sidecar receives this carrier only, never query tokens.
- Adapter: fused-QKV LoRA in blocks 8--11, rank 4, alpha 8.  Original backbone
  parameters remain frozen.
- Oracle code: deterministic PCA-4 coordinates of the query-specific paired
  clean projected correction, fit and standardized on `S_fit` only.  It is
  privileged training/diagnostic information and is not a deployment claim.
- Readout: adapted-minus-frozen support-carrier suffix output, followed by the
  exact rank-4 projection `P`.  No normalization occurs after adding the
  correction to the frozen query residual.
- Cycle 1 uses the raw projected adapted-minus-frozen sidecar output and
  monitors direction and norm separately.  It deliberately omits a radius
  head: if direction improves while norm remains below `0.70`, adding only the
  radius head is the isolated Cycle-2 repair.  At code zero or abstention, the
  sidecar is skipped and frozen features, residuals, neighbor indices, and
  scores must agree within `1e-7`.
- Training: 144 AdamW updates, learning rate `1e-4`, zero weight decay, global
  clipping `1.0`, fixed schedule and seed.  Monitoring snapshots are written
  at steps `0,36,72,108,144`.

Matched controls use the same `U`, oracle code, targets, schedule, optimizer,
and evaluation rows: `N3mu-P`, a same-code projected tensor decoder (`N3cap`),
and a support-carrier output-LoRA placement control.  Candidate/control
parameter counts and actual trainable names are recorded.

P1 is a `PROMISING_DIAGNOSTIC` only if the candidate has finite nonzero
gradients, avoids correction-norm collapse, improves held correction cosine by
at least `0.02` over the best matched learned control, has positive normal q99
margin, whole anomaly cosine at least `0.85`, anomaly norm in `[0.70,1.40]`,
and exact no-op/complement errors.  Otherwise the best checkpoint trend and
the first isolated failure become the sole Cycle-2 repair target.

## Synthetic and monitoring contract

Clean and anomalous members of every quadruplet share the exact transform
metadata and realized seed.  Generator identity is audit-only and cannot enter
the model.  Every checkpoint records total/component loss, pre-clip gradient
norm, correction cosine/norm, normal q99, code anomaly drift, relative
commutator error, whole/P-parallel anomaly retention, route coverage, no-op and
projection errors, trainable parameter names, and adapter-state SHA-256.

## Remote execution and baselines

Run in fixed container `hun_fsad_tta_012` on idle host GPUs 0--3 using an
immutable source snapshot.  Dense maps are neither generated nor retained;
this is required because the current remote result volume has limited free
space.  Compact artifacts are pulled and independently reaggregated.

The existing same-condition baseline artifact is valid and is referenced, not
rerun: SuperAD AUROC/F1/AU-PRO `0.761857/0.368423/0.534924`, RN-FMLK
`0.745312/0.331950/0.425133`.  Cycle 1 remains support-only and therefore has
`benchmark_comparable=false`; these values are context, not a pass gate.
