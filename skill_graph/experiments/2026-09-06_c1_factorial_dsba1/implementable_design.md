# C1-DIAG-AT-P16-can-vial-s3-v1: Implementable Design

Date: 2026-09-06  
Protocol: `C1-DIAG-AT-P16-can-vial-s3-v1`  
Status: `SMOKE_PASS / MAIN_RUN_NOT_STARTED`

## Scope

This is a two-object, support-16 C1 diagnostic on MVTec AD2 single-image. It
uses the PSAR same head, frozen DINOv2 ViT-L/14 features from layers
`5,11,17,23`, short edge `672`, and an asserted head parameter count of `241`.
The factors are synthetic anomaly augmentation (`A`) and shift-assumed
augmentation with consistency (`T`).

The primary decision claim is public can/vial identity utility. The held
`G2 × cos_1_1` population is a secondary source-held mechanism/transfer
diagnostic only; it is excluded from promotion/kill decisions, baseline
superiority claims, and real-shift robustness claims.

Objects are `can` and `vial`; seeds are `20260906/07/08`; arms are
`C00`, `C10_A`, `C01_T`, and `C11_AT`. There are 24 head fits, 576 updates per
fit, and 13,824 planned updates. Same-condition `SuperAD-P16` and `RN-P16`
are required new dsba1 reruns, with config hashes recorded at runtime.

## Split and reference custody

Use new P8 bases with training indices `0..11`. For each `i`, exclude
`d=(i+1)%12`, `k+=(i+2)%12`, and `k-=(i-2)%12`. Held bases `12..15` and their
descendants are image features only and remain unopened until every head is
sealed. After the seal, use `P15=P16\heldbase` for calibration and the held
synthetic donor held cycle; use `P14=P16\base,donor` for the corresponding
held arrangement. Public evaluation uses P16. The previous P12 fit is
discarded.

The exact Same16 manifest must be fully hash-verified at runtime. A truncated
historical hash cannot satisfy custody.

## Augmentation and held field

Train A uses `G0_poisson_patch`, `G1_material_blotch`, and
`G4_frequency_local`; `G2_structural_scratch` is held. Train T is exactly:

```text
PIL bicubic resize -> ToTensor -> float32 gain 2^e -> Normalize
```

with `e={0,-0.0667,-0.1333,-0.2}`, no clipping or rounding, and `W=I`.
The identity path must be byte/semantic-equivalent to `prepare_image`.
Held T uses only `cos_1_1`, with `0.15` divided-field maximum and float
multiplication without clipping. Its fixed coordinate path is:

```text
native G2 synthesis -> canonical PIL short-edge-672 resize -> ToTensor
-> float32 resized-H/W cos_1_1 exponential multiplicative field
   (CV.15, field/max) -> Normalize -> patch-14 crop
```

After spatial multiplication, clipping, rounding, uint8 conversion, and PIL
re-entry are forbidden. The native mask is not warped; apply it after
head-map native `INTER_LINEAR` restoration. Other cosine cells, blur, clipping,
and public transformed stress are excluded from this initial run.

## Objective

The normalized layer-23 P8 nearest-neighbor teacher is `r`. Compute equal-image
12-fit-base teacher map quantiles `q50` and `q95`, then

```text
y = clip((r-q95)/max(q95-q50,1e-6), -4, 4)
```

Use 1024 deterministic non-tie patch pairs. With head outputs `lx`, `lA`,
`lTx`, `lTA` and mask `m`, use exactly:

```text
B(s,m) = .5*(mean_m softplus(-s) + mean_not_m softplus(s))

L00 = mean Huber_delta=.5(lx-y)
      + .25*mean softplus[-sign(rp-rq)(lp-lq)]

LA = B(lA,m) + mean softplus(lx)
     + mean_m softplus[.5-(lA-lx)]

LT = mean Huber_delta=.5(lTx-lx)

LATabs = B(lTA,m) + mean softplus(lTx)

LATcons = .5*(mean_m+mean_not_m)
          Huber_delta=.5[(lTA-lTx)-(lA-lx)]

L(a,t) = L00 + a*LA + .25*t*LT
         + a*t*(LATabs + .25*LATcons)
```

Here `(a,t)` selects the arm. Empty mask, empty background, and nonfinite
values fail closed. No GT is opened during fitting.

## Optimization

Use exactly 576 updates per head, AdamW with `lr=3e-4`, `weight_decay=1e-4`,
betas `(0.9,0.999)`, epsilon `1e-8`, batch size `1`, and gradient clip `1`.
There is no scheduler or hyperparameter search. Use bf16 for backbone forward
and fp32 for RN, maps, head, and loss on CUDA 0. Paired arms use identical
head initialization per seed.

## Resource and storage contract

Run on dsba1 host GPU `6` only, exposed as in-container CUDA slot `0`, with
UUID equality asserted. Source credentials from `configs/remote/dsba1.env`
without copying secret values into artifacts. Reuse the dsba3 Docker
environment and dataset after read-only validation through a dedicated dsba1
container plan; do not touch the existing dsba3 container/image/data/cache or
results.

Use one object/seed training RN grid cache shared by all four arms; deduplicate
12 bases, 48 shift cells, and 576 A/TA cells per object/seed group. Use
`Craw = 20016 × H × W` raw-grid sizing,
cap `ceil(1.15*Craw)`, and require free space of at least `2*cap+5 GiB`.
Check actual NPZ and manifest bytes against the cap. The teacher is computed
once per base; `y` and rank are regenerated at read time. After each four-head
seal, remove only new run-owned regenerable cache and retain its manifest. Do
not write NPY/TIFF/full-resolution map lists. Query one at a time, sharing one
backbone forward across 12 heads plus B0/B1 per object and retaining only 14
stream histogram accumulators.

Run sequentially with no GPU fallback. If GPU 6 is busy, defer.

The 12-base, 48-shift, and 576-cell cache accounting is a training-cache
operations budget and is separate from the 16-shot support budget used for
learning and evaluation.

## Execution gates

1. Run CPU tests.
2. Run a separate `can`, seed-0, first-cell, one-update smoke for all four
   arms. Mark it invalid for scientific aggregation.
3. Run all 24 fits without held-base peek or checkpoint selection.
4. Seal all 24 checkpoints.
5. Perform pass-1 map/hash/shape inventory with GT closed, seal the inventory,
   and release dense prediction maps.
6. Regenerate maps in pass 2, require exact hash match, then open the GT
   stream for histogram metrics; retain no dense prediction map tree.

The primary metric is `seg_AUROC_0.05`; report oracle F1 and `AU-PRO_0.05`
with matched float16-order histograms (not FP32-exact), counts,
finite/nonconstant checks, and seed metadata. The q95 activity threshold uses
four post-seal held-normal
bases x four train-T conditions and is diagnostic for activity/FPR only.

## Post-fit population and ordering binding

The public can/vial branch uses the identity condition only. All
`PROMISING`, `COMPONENT_SIGNAL_ONLY`, and `VALID_NO_UTILITY` decisions, and
the B0/B1 comparison gate, use only these public identity metrics. No public
transformed-stress population is added.

The held `G2_structural_scratch × cos_1_1` condition is a source-held
diagnostic for the four paired C1 arms only: 2 objects x 3 seeds x 4 held
bases = 24 synthetic cells, producing 96 maps across four arms. Held B0/B1
maps are not generated; the corresponding 48 maps are outside the population.

Held mask and synthesis seeds use the existing `0x1f` eight-field encoding:

```text
psar_d1.deterministic_seed(
    PROTOCOL_ID, object, scientific_seed, base, donor,
    'T_H_COS_1_1_CV015', 'G2_structural_scratch', role
)
```

`role` is `mask` or `synthesize`; the held shift is deterministic. The exact
post-fit order is:

```text
global 24-head seal
-> calibration
-> held source diagnostic
-> public global GT-free inventory
-> exact-hash-matched GT pass 2
```

Calibration uses bases `12..15 x 4 train-T` on P15 with exact native float32
q95. Per object, 16 compact RN grids are shared by 12 heads; one head's
16-native calibration data exists in RAM temporarily only. No calibration
maps are persisted.

## Predeclared diagnostic gates

Define

```text
DeltaA = .5[(C10_A-C00)+(C11_AT-C01_T)]
DeltaT = .5[(C01_T-C00)+(C11_AT-C10_A)]
I_AT   = C11_AT-C10_A-C01_T+C00
```

The proposed `PROMISING` gate requires C11-C00 macro pAUROC `>=.005`, each
object delta `>=-.010`, macro F1 and AU-PRO deltas `>=-.020`, one of DeltaA
or DeltaT `>=.003`, and C11 better than both B0 and B1 macro by `>=.005` with
no harm. These gates are prospective. Three-seed summaries do not establish
causality. A component-only baseline failure is `COMPONENT_SIGNAL_ONLY`; a
valid nonsignal outcome is `VALID_NO_UTILITY`.

No all-eight-object claim is licensed. A `LATcons` ablation is a future,
single controlled follow-up only if initial evidence warrants it, preserving
the same inputs and removing only the consistency term. C2 and C3 are outside
this protocol.

## Pending implementation checks

The implementation must assert the 241-parameter head, exact identity image
path, full manifest hash, split custody, no-peek ordering, finite/nonconstant
outputs, sequential GPU binding, compact output schema, and the two-pass hash
barrier before any scientific result is accepted.

The canonical parity audit occurs only after the global 24-head seal,
calibration, and held diagnostic, before the global public inventory. B0/B1
must be shape-exact, finite, canonical-nonconstant, and satisfy
`max_abs(FP32_native-FP32_canonical) <= 1e-5` with `rtol=0`. The parity source
query is the lexical-first `train/good` image excluded from the object’s
Same16 set (one source image per object, two total), with path and file SHA
recorded. P16 self-query is forbidden because it degenerates to self-NN. The
source is for implementation audit only and cannot enter training, calibration,
thresholding, model selection, or performance metrics. Missing source fails
closed; there is no public or held fallback. The audit records shape,
max/mean absolute error, and pass status only, with no map storage. The exact
source-query location and fixed rule are pending implementation binding.
