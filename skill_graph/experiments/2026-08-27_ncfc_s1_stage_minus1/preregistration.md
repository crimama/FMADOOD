# NCFC-S1 Stage -1 preregistration

Date: 2026-08-27  
Protocol: `ncfc-s1-normal-coordinate-frozen-commutation-20260827-v2`  
Status: `COMPLETED / NORMAL_ONLY_NO_ANOMALY_SIGNAL / END_NCFC`  
Claim tier: support-only representation/readout diagnostic; not benchmark-comparable

Revision note: the v1 remote model-smoke stopped before any scientific shard
because the 672-pixel smaller-edge preprocessing preserved the can aspect ratio
and produced a native `48x104` grid rather than the incorrectly assumed
`48x48` grid. V2 changes only that result-free preprocessing contract: every
runtime aspect-preserving native grid with both dimensions at least 16 is
recorded and deterministically pooled to 16x16. No v1 scientific result exists.

## Question and boundary

The closed SIAI sign/saturation diagnostic showed that a known J0 exposure
coordinate can recover part of the normal frozen-feature geometry, but that the
synthetic anomaly residual does not commute sufficiently with the coordinate.
SAEC-LoRA then showed that shared action relations can be fitted, while its
commutation term had no identifiable held contribution.  NCFC-S1 makes one
separate, no-training observation:

> Does a clean-derived normal translation coordinate predict the corresponding
> shifted synthetic-anomaly displacement in frozen DINOv2, and does that answer
> depend on the feature readout rather than the assumed action itself?

This is a feasibility assay. It does not learn a translation, modify an image,
remove a shift, update a backbone, or establish causal evidence that a feature
translation exists. In particular, the residual

`[R(T(A(x)))-R(A(x))] - [R(T(x))-R(x)]`

is a mixed finite difference. Clean-derived correction terms can telescope or
cancel even when the underlying image transformation is not a valid feature
translation. Any positive result is therefore evidence about a frozen
representation/readout assay only, not a causal translation claim.

The diagnostic is `can` only and support-only. It never opens TESTpublic,
TESTprivate, test images, ground truth, query adaptation, or a SuperAD
comparison. `SuperAD=N/A` and `benchmark_comparable=false` are fixed.

## Data custody and exact split

- Dataset: `/home/hunim/Volume/DATA/mvtec_ad_2` on the project-owned dsba3
  container.
- Object: `can` only.
- Fit/support memory: the immutable 16-image SuperAD-selected `train/good`
  support set, used only as frozen RN memory and for the support static control.
  Support manifest SHA-256:
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Held bases: exactly four support-disjoint `can/train/good` bases. Every held
  base receives both signed J0 actions, `-0.1333 EV` and `+0.1333 EV`; there
  are exactly eight held base/sign cells.
- Held anomaly: `G2_structural_scratch` only. Its anomaly donor, mask, and
  synthesis seed are fixed per held base and are never used to fit a control.
- No held base, held anomaly, held mask, or held generator label enters a
  statistic, mean, target, threshold, or selection rule before evaluation.
- Only compact JSON, text logs, hashes, and scalar summaries may remain.
  Images, dense feature tensors, score maps, anomaly maps, and checkpoints
  are not retained.

The runner constructs, for every held base `x` and sign `s`,

`x`, `T_s(x)`, `A(x)`, `T_s(A(x))`, and `A(T_s(x))`.

`T_s(A(x))` and `A(T_s(x))` use the same base, donor, mask, synthesis seed,
and signed action specification, but are rendered independently. Equality is
not assumed. The action is the existing deterministic J0 exposure transform;
the run records its signed value, clipping, and pixel discrepancy
only as diagnostics. No inverse or pixel correction is applied.

## Frozen backbone and readouts

The only backbone is the frozen DINOv2 ViT-L/14 at zero-based block 11 on the
fixed 672-pixel smaller-edge canvas. Its runtime aspect-preserving block-11
patch grid is recorded and deterministically spatially pooled to 16x16 for
both readouts. No parameter receives a gradient.
No LoRA, classifier, projector, router, decoder, learned normalization, or
feature fitting is permitted.

For each input `v`, report both readouts from the same forward pass:

1. `R_raw(v)`: the raw `model.norm` patch-token output at block 11, before the
   existing channel standardization and row-L2 normalization;
2. `R_std(v)`: the existing readout, namely population channel standardization
   with the fixed `1e-6` floor followed by token-wise row-L2 normalization with
   the fixed `1e-8` floor.

The two readouts are never mixed within a metric. Raw-RN uses cosine distance
on raw tokens with the same raw support memory; standardized-RN uses the
existing exact cosine RN path. No raw result may be post-normalized into the
standardized result and no readout may be selected after observing metrics.

## Frozen quantities and controls

For readout `R`, held base `b`, and signed action `s`, define

`d_R(b,s) = R(T_s(x_b)) - R(x_b)`  (forward clean translation)

`c_R(b,s) = -d_R(b,s) = R(x_b) - R(T_s(x_b))`  (applied correction)

`r_0_R(b) = R(A_b(x_b)) - R(x_b)`

`r_TA_R(b,s) = R(T_s(A_b(x_b))) - R(T_s(x_b))`

`r_AT_R(b,s) = R(A_b(T_s(x_b))) - R(T_s(x_b))`.

The primary commutation assay compares `r_TA_R` and `r_AT_R` with `r_0_R`.
Because adding the same clean correction to the shifted normal and anomaly
representations cancels in these residual differences, this residual assay is
reported separately from correction-vector quality. Correction-vector metrics
compare `c_R` with the ideal clean correction and with the two diagnostic ideal
anomaly corrections `R(A(x))-R(T(A(x)))` and `R(A(x))-R(A(T(x)))`.

The exact arms are evaluated independently for both readouts:

- `B0_SHIFTED_FROZEN`: the uncorrected shifted representation;
- `M1_PAIRED_CLEAN_TRANSLATION`: the held base's clean correction `c_R(b,s)`; this is the
  candidate assay and uses no anomaly-derived quantity;
- `N1_STATIC_MEAN_TRANSLATION`: the mean clean translation from the 16 support
  images at the same signed action, excluding all held bases;
- `N2_WRONG_SIGN_TRANSLATION`: the support/held clean translation for the
  opposite signed J0 action, preserving magnitude and all other construction;
- `N3_BASE_SHUFFLE_TRANSLATION`: the clean translation from a deterministic
  cyclically different base at the same sign;
- `N4_RESIDUAL_SHUFFLE_TARGET`: the cyclic next-held-base anomaly residual with
  token rows deterministically rolled by one within the same shape. This is a
  residual-target null, not a translation candidate and not a correction arm.

All controls preserve the support, tensor shape, action magnitude, and RN
scoring contract. Static, wrong-sign, and base-shuffle controls are clean-only.
The held anomaly is used only to evaluate the candidate and controls.

## Required per-cell metrics

For each of the 8 base/sign cells and each readout, record:

- clean correction cosine and Frobenius norm ratio;
- `T(A)` and `A(T)` anomaly correction cosine and norm ratio;
- `T(A)` and `A(T)` residual-commutation cosine against `r_0_R`, mask-inside
  cosine, norm ratio, top-10 residual-support IoU, and interaction norm;
- correction margin over `N1`, `N2`, and `N3`, and residual-commutation margin
  over `N4`;
- exact-RN shifted-normal q99 and q99 reduction versus `B0`;
- exact-RN anomaly rank correlation, nearest-support identity retention, and
  `(Q99-Q50)` scale ratio against the paired clean RN result;
- `T(A)` versus `A(T)` pixel MAE, maximum error, and clipping rate;
- finite/nonconstant flags, tensor shapes, support/held path hashes, and source
  and snapshot hashes;
- compact construction custody: base/donor relative paths, mask and synthesis
  seeds, mask hash, signed EV, composition-order labels, all five input-array
  hashes, and shard GPU provenance.

The primary numerical summaries are medians over the four held bases within
each sign and the unweighted median over both signs. Positive and negative
sign rows may not be dropped, reweighted, or replaced.

## Exact gates

All gates are applied separately to `R_raw` and `R_std`.

### Structural gate: `NCFC_STRUCTURAL_VALID`

The structural gate requires all of the following:

- exactly 4 held bases x 2 signs x 2 readouts, with no missing, duplicated, or
  extra cell;
- all five required inputs are present for every cell and both composition
  orders use the declared base, donor, mask, seed, and sign;
- raw and standardized tokens both have the frozen `[256,1024]` pooled shape,
  derived from a recorded runtime native grid whose dimensions are both at
  least 16; all values and compact metrics are finite;
- every RN score vector is nonconstant, every support/held path hash matches,
  and no forbidden data or target statistic was accessed;
- parameters, buffers, support memory, and readout definitions are unchanged;
- an exact repeated run has maximum scientific-metric absolute difference
  `<=1e-7` and identical canonical scientific-signature SHA-256;

Any structural failure is `INVALID_CUSTODY_OR_READOUT / NO_SCIENTIFIC_RESULT`.

### Normal gate: `NCFC_NORMAL_TRANSLATION_SIGNAL`

For `M1_PAIRED_CLEAN_TRANSLATION`, every one of the 8 base/sign cells must
have, for the clean shifted representation:

- correction cosine `>=0.90`;
- correction norm ratio in `[0.80,1.25]`;
- q99 reduction versus `B0` `>=-0.005`;
- RN rank and neighbor retention `>=0.90` and RN scale ratio in `[0.80,1.25]`.

The aggregate normal gate additionally requires both sign strata to pass and
the median candidate margin to be positive for both readouts.

### Anomaly gate: `NCFC_ANOMALY_COMMUTATION_SIGNAL`

For both `T(A)` and `A(T)` in every base/sign cell, `M1` must satisfy:

- residual-commutation cosine `>=0.90` against `r_0_R(b)`;
- anomaly correction cosine `>=0.90` and norm ratio in `[0.80,1.25]`;
- no anomaly residual norm reduction greater than 10% relative to the paired
  clean anomaly residual;
- no cell may have anomaly cosine below `0.80`, even in the weaker continuation
  assessment.

This gate is explicitly a representation/readout commutation gate. Passing it
does not authorize the statement that J0 translation is causal, invertible, or
deployable.

### RN safety gate: `NCFC_RN_SAFE`

For shifted normal and both shifted-anomaly composition orders in every cell:

- RN rank correlation `>=0.90`;
- nearest-support retention `>=0.90`;
- RN scale ratio in `[0.80,1.25]`;
- shifted-normal q99 reduction versus `B0` `>=-0.005`;
- all anomaly residual and RN metrics remain finite and nonconstant.

### Readout gate: `NCFC_READOUT_CROSSCHECK`

Both raw and standardized readouts must pass `NCFC_STRUCTURAL_VALID`. A strict
NCFC signal additionally requires that both readouts pass the normal, anomaly,
and RN gates. Hard-null distinctness additionally requires normal correction
margin over the strongest of `N1`--`N3` and both residual-commutation margins
over `N4` to be `>=+0.01` in at least 6 of 8 cells in each readout, including
at least 3 cells of each sign.
If only one readout passes, the result is `READOUT_DEPENDENT_SIGNAL`; it is
not a method or translation claim and does not authorize training.

## Sequential verdicts and stop rules

The single can-only run has the following fixed verdicts:

1. `INVALID_CUSTODY_OR_READOUT / NO_SCIENTIFIC_RESULT` if the structural gate
   or exact repeat fails.
2. `NO_NORMAL_COORDINATE_SIGNAL / END_NCFC` if structural validity passes but
   either readout fails the normal gate.
3. `NORMAL_ONLY_NO_ANOMALY_SIGNAL / END_NCFC` if normal and RN pass but the
   anomaly gate fails. This is the expected stop for a telescoping-only or
   readout-specific effect.
4. `READOUT_DEPENDENT_SIGNAL / NO_CONTINUE` if exactly one readout passes all
   scientific gates.
5. `NCFC_REPRESENTATION_SIGNAL / CONTINUE_ONE_PREREGISTERED_DIAGNOSTIC` only
   if both readouts pass every strict gate, both signs pass, all eight cells
   are valid, and the candidate beats every hard null as specified. This
   licenses one separately preregistered diagnostic only; it does not license
   LoRA, CRFR, score correction, pixel inversion, J2, other objects, or public
   data.

No result licenses a rank, block, learning-rate, loss-weight, seed, readout,
or action-strength sweep. A failed anomaly gate closes the anomaly-commutation
sentinel even when clean translation or RN metrics are positive.

## Distinctness from closed branches

NCFC-S1 is deliberately not a continuation of:

- closed SAEC/protected-PCGrad/oracle-family/CRFR QKV or backbone LoRA: there
  is no training and no learned parameter;
- closed score-bypass or RN correction branches: the exact RN score is not
  fused, rescaled, thresholded, reranked, or conditionally bypassed;
- closed pixel-inverse/SIAI branches: NCFC does not invert, modify, or recover
  exposure pixels and uses the supplied J0 transform only as an input pair;
- closed layerwise-DINOv3 branches: the backbone is fixed DINOv2 ViT-L/14 and
  only block 11 is read.

## Remote execution, custody, and retained artifacts

- Use only the project-owned dsba3 container and project GPUs `0--3`; do not
  use dsba2 GPU6 for this protocol.
- Preflight may inspect only the selected project GPU's availability/UUID and
  the exact project container. Never enumerate, enter, stop, reset, clean, or
  modify another user's container, process, job, file, volume, or GPU.
- Use one immutable remote snapshot and one exact-repeat invocation. The
  snapshot, runner, readout bundle, support manifest, and aggregator hashes
  are frozen before result-bearing execution and copied into every compact
  object summary.
- Pull only compact JSON/log/hash artifacts. Remove only this run's exact
  immutable snapshot after custody verification; retain the compact aggregate,
  repeat signature, manifest, and custody ledger.

## Frozen source identity

Result-bearing execution is forbidden until the runner, aggregator, focused
tests, readout bundle, remote controller, immutable support manifest, and
configuration are complete and hashed. The prelaunch freeze is:

- runner SHA-256: `f52fc1582bb98bd7223ec6465b34f0c1bef277c1a957f4fe80e301a58cd3b141`
- aggregator SHA-256: `3ce56edf96612a9358e280b98a9c25b02042f1652cdb1b7b1b6c98d98d0864b6`
- frozen DINOv2/readout bundle SHA-256: `328aa50d8a61566630aa788559ff1ac96b16f61dc145e7ed619f44724f99385e`
- remote controller SHA-256: `0cd8c9bfe889557d4da3fd48a6e49e97026af52bb3906d04a80fb07881748b14`
- source-manifest SHA-256: `0d3585973579fd82d82b4a899812bffbdaa78c313aa245b349c4090eacf2f126`
- frozen configuration SHA-256: `5e242cfe60facd920b377c06cce3a75235e0a79ff6be1c213c01177e3629c57a`

The prelaunch manifest must bind these values and the fixed support manifest
above. Any source change after that freeze requires a new protocol/run name and
cannot be reported as this NCFC-S1 result.
