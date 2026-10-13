# C1 Factorial: Synthetic and Shift Augmentation on dsba1

Date: 2026-09-06  
Protocol: `C1-DIAG-AT-P16-can-vial-s3-v1`  
Status: `SMOKE_PASS / MAIN_RUN_NOT_STARTED`  
Verdict: `PENDING_EXECUTION`

This preregistration records the final architect specification. No result is
available yet.

## Negative evidence intake

C1 is a same-head factorial diagnostic. It separates synthetic anomaly
augmentation and shift-assumed augmentation/consistency in the existing PSAR
head rather than introducing a new backbone architecture. Prior PSAR
operational-invalid evidence is not treated as a method refutation. Historical
DVT/private threshold settings, including the `0.25` versus `0` density-weight
mismatch, are not same-condition comparators.

## Motivation and claim boundary

The primary decision claim is utility on the public can/vial identity
condition: whether the factorial training arms improve the identity metric
under the predeclared public protocol. The primary claim metric is
segmentation pixel `AUROC_0.05`; oracle F1 and `AU-PRO_0.05` are secondary
reported metrics. The held `G2 × cos_1_1` branch is a secondary source-held
mechanism/transfer diagnostic. It does not enter promotion or kill decisions,
does not establish baseline superiority, and does not support a real-shift
robustness claim.

## Dataset, objects, and feature contract

- Dataset: MVTec AD2 single-image.
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`.
- Objects: `can`, `vial`.
- Backbone: frozen DINOv2 ViT-L/14.
- Feature layers: layers `5, 11, 17, 23`.
- Short-edge resolution: `672`.
- Head: PSAR same-head C1, with a runtime assertion of exactly `241` head
  parameters.
- Reference budget: exact SuperAD P16 support, with the full Same16 manifest
  hash verified at runtime. A historical truncated hash is not considered a
  full verification.

## Arms, seeds, and fit count

| Arm | Factors |
|---|---|
| `C00` | identity |
| `C10_A` | synthetic anomaly augmentation |
| `C01_T` | shift-assumed augmentation with consistency |
| `C11_AT` | synthetic anomaly plus shift-assumed augmentation with consistency |

Seeds are `20260906`, `20260907`, and `20260908`. The design contains 24
head fits: 2 objects x 4 arms x 3 seeds, with 576 updates per head and
13,824 planned updates in total. Same-condition `SuperAD-P16` and `RN-P16`
runs are required controls and are new dsba1 reruns with their own config
hashes. Old private results and historical values are not comparators.

## Split and custody

The strict split uses new P8 bases with train indices `0..11` and excludes the
following deterministic held descendants:

```text
i = 0..11
d = (i + 1) % 12
k+ = (i + 2) % 12
k- = (i - 2) % 12
```

Held bases `12..15` are image features and descendants. They must not be
decoded until all 24 heads are sealed. After sealing, `P15 = P16` excluding
the held base; held calibration uses the specified post-seal arrangement.
The held synthetic donor uses the held cycle, and `P14 = P16` excluding the
base and donor. Public evaluation uses P16. The old P12 fit is discarded and
cannot be used for this protocol.

## Augmentation contract

Training synthetic anomalies use the existing generators:

- `G0_poisson_patch`
- `G1_material_blotch`
- `G4_frequency_local`

`G2_structural_scratch` is held out. The synthetic anomaly and shifted
synthetic paths must retain their predeclared masks and must not open GT.

Training T applies, in this exact order:

```text
PIL bicubic resize -> ToTensor -> float32 gain 2^e -> Normalize
```

where `e ∈ {0, -0.0667, -0.1333, -0.2}`. There is no clipping or rounding;
`W=I`, and the operation is a float-pixel transformation. This is a
shift-assumption diagnostic and is not a guarantee of DINO defect preservation.
The held T condition uses only `cos_1_1`, with `0.15` divided-field maximum,
float multiplication, and no clipping. Its coordinates are fixed as:

```text
native G2 synthesis
-> canonical PIL short-edge-672 resize
-> ToTensor
-> float32 resized-H/W cos_1_1 exponential multiplicative field (CV.15, field/max)
-> Normalize
-> patch-14 crop
```

After spatial multiplication, clipping, rounding, uint8 conversion, and PIL
re-entry are forbidden. The native mask is not warped; use the head-map native
`INTER_LINEAR` restoration before applying it. Other cosine cells, blur,
clipping, and public transformed stress are outside the initial scope. The
identity path must match `prepare_image` exactly.

## Exact objective

Let `r` be the normalized layer-23 P8 nearest-neighbor teacher. Compute `q50`
and `q95` from equal-image 12-fit-base teacher maps, then define

```text
y = clip((r - q95) / max(q95 - q50, 1e-6), -4, 4)
```

Use 1024 deterministic non-tie patch pairs. For head outputs `lx`, `lA`,
`lTx`, and `lTA`, and mask `m`, define

```text
B(s,m) = 0.5 * (mean_m softplus(-s) + mean_not_m softplus(s))

L00 = mean Huber_delta=0.5(lx - y)
      + 0.25 * mean softplus[-sign(rp-rq)(lp-lq)]

LA = B(lA,m) + mean softplus(lx)
     + mean_m softplus[0.5 - (lA-lx)]

LT = mean Huber_delta=0.5(lTx-lx)

LATabs = B(lTA,m) + mean softplus(lTx)

LATcons = 0.5 * (mean_m + mean_not_m)
          Huber_delta=0.5[(lTA-lTx) - (lA-lx)]

L(a,t) = L00 + a*LA + 0.25*t*LT
         + a*t*(LATabs + 0.25*LATcons)
```

Here `a,t ∈ {0,1}` select the factorial arm. Empty masks, empty background,
and nonfinite values are fail-closed errors. GT is never opened for training.

## Optimization and execution

- Updates: exactly `576` per head.
- Optimizer: AdamW.
- Learning rate: `3e-4`.
- Weight decay: `1e-4`.
- Betas: `(0.9, 0.999)`.
- Epsilon: `1e-8`.
- Batch size: `1`.
- Gradient clip: `1`.
- Scheduler: none.
- Hyperparameter search: none.
- Frozen DINO backbone forward: bf16.
- RN, maps, head, and loss: fp32 on CUDA 0.
- Paired seeds use identical head initialization across arms.

## Storage and remote boundary

- Host: dsba1.
- Host GPU: GPU `6` only.
- Container CUDA slot: `0`, with UUID equality asserted.
- Credential path: `configs/remote/dsba1.env`; secret contents are not copied
  into artifacts.
- Reuse the validated dsba3 Docker environment and dataset through a dedicated
  dsba1 container plan. Never delete or modify the old dsba3 container, image,
  data, cache, or results.
- Use one object/seed training RN grid cache shared by the four arms.
- Deduplicate 12 bases, 48 shift cells, and 576 A/TA cells per object/seed
  group; seal the group before removing only its new run-owned cache.
- Estimate raw cache bytes as `Craw = 20016 × H × W`; cap storage at
  `ceil(1.15*Craw)` and
  require at least `2*cap + 5 GiB` free.
- After each four-head seal, remove only new run-owned regenerable caches;
  retain manifests. Never remove old results, data, or caches.
- Query one at a time. Share one backbone forward across 12 heads plus B0/B1
  per object and retain only 14 stream histogram accumulators.
- Do not write NPY, TIFF, or full-resolution map lists.

All cells run sequentially. A busy GPU defers the run; there is no fallback
GPU.

## Planned custody sequence

1. CPU tests, then a separate `can`, seed-0, first-cell, one-update smoke for
   all four arms. This smoke is diagnostic only and invalid for science.
2. Run all 24 full fits without held-base peeking or checkpoint selection.
3. Seal all 24 checkpoints.
4. Pass 1: map/hash/shape inventory without GT; seal the inventory and release
   dense prediction maps.
5. Pass 2: regenerate maps, require exact hash match, then open the GT stream
   for histogram metrics; retain no dense prediction map tree.

The process stops if custody, finite-value, coverage, or hash gates fail.

## Canonical parity audit

After global 24-head seal, calibration, and held completion, and before the
global public inventory, run the canonical parity audit. B0/B1 must be
shape-exact, finite, and canonical-nonconstant, with
`max_abs(FP32_native-FP32_canonical) <= 1e-5` and `rtol=0`. For each object,
use the lexical-first `train/good` image excluded from its Same16 set (one
image per object, two total); record its path and file SHA. P16 self-query is
forbidden because it degenerates to self-NN. This source is for implementation
audit only and is excluded from training, calibration, thresholding, model
selection, and performance metrics. Missing source fails closed, with no
public or held fallback. Record shape, max/mean absolute error, and pass
status only; store no map. The precise implementation location and fixed
tolerance rule remain pending implementation binding.

## Post-fit population and ordering binding

The public can/vial branch uses the identity condition only. All
`PROMISING`, `COMPONENT_SIGNAL_ONLY`, and `VALID_NO_UTILITY` decisions, and
the B0/B1 comparison gate, use only these public identity metrics; no public
transformed-stress population is added.

The held `G2_structural_scratch × cos_1_1` condition is a source-held
diagnostic for the four paired C1 arms only. It contains
`2 objects × 3 seeds × 4 held bases = 24` synthetic cells and `4 arms × 24 =
96` maps. No held B0/B1 maps are generated; the corresponding 48 maps are
outside the population.

Held mask and synthesis seeds are derived exactly as follows, using the
existing `0x1f` eight-field encoding and deterministic shift:

```text
psar_d1.deterministic_seed(
    PROTOCOL_ID, object, scientific_seed, base, donor,
    'T_H_COS_1_1_CV015', 'G2_structural_scratch', role
)
```

`role` is `mask` or `synthesize`. The seed is never replaced by a runtime or
global random seed.

The post-fit order is fixed:

```text
global 24-head seal
-> calibration
-> held source diagnostic
-> public global GT-free inventory
-> exact-hash-matched GT pass 2
```

Calibration uses bases `12..15 × 4 train-T` on P15 with exact native float32
q95. Within each object, 16 compact RN grids are shared by 12 heads. A single
head's 16-native calibration data may exist temporarily in RAM only; it is not
persisted.

The cache budget uses `Craw = 20016 × H × W` raw-grid sizing, with cap
`ceil(1.15*Craw)`. The teacher is computed once per base; `y` and rank are
regenerated at read time. Actual NPZ size and manifest bytes are checked
against the budget before proceeding.

## Metrics and gates

Primary metric: `seg_AUROC_0.05`. Also report oracle `seg_F1` and
`AU-PRO_0.05`, all using matched float16-order histograms across branches.
These histograms are not FP32-exact. Record counts, finite/nonconstant checks,
and seed metadata.

The q95 activity threshold is computed from four post-seal held-normal bases x
four train-T conditions using the P15 equal-image ECDF. Activity and FPR are
diagnostics only; oracle segmentation F1 remains independent.

Define arm deltas descriptively as

```text
DeltaA = 0.5 * [(C10_A-C00) + (C11_AT-C01_T)]
DeltaT = 0.5 * [(C01_T-C00) + (C11_AT-C10_A)]
I_AT   = C11_AT - C10_A - C01_T + C00
```

These three-seed summaries are not causal proof. The architect-proposed
diagnostic gate is `PROMISING` when C11 improves C00 macro pAUROC by at least
`0.005`, every object delta is at least `-0.010`, macro F1 and AU-PRO deltas
are each at least `-0.020`, at least one of `DeltaA` or `DeltaT` is at least
`0.003`, and C11 beats both B0 and B1 macro by at least `0.005` without
no-harm failure. These are proposed gates, not observed results.

Component-only baseline failure is `COMPONENT_SIGNAL_ONLY`; a valid run with
no utility is `VALID_NO_UTILITY`. No all-eight-object claim is possible from
this two-object diagnostic.

`LATcons` follow-up ablation is outside the initial scope and may be considered
only if the initial result preserves the same inputs and removes only the
consistency term. C2 and C3 are not run in this protocol.

## Current status

Implementation preflight and the diagnostic smoke have passed. The latest
five-suite receipt reports 40 passed, 11 warnings, and 12.81 seconds after
small fixes. The scientific 24-fit training run, calibration, held diagnostic,
public inventory, GT metrics, and conclusion are still `N/A`. No result or
scientific claim is reported here.
