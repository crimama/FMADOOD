# NCFC-S1 Stage -1 report

Date: 2026-08-27  
Protocol: `ncfc-s1-normal-coordinate-frozen-commutation-20260827-v2`  
Status: `COMPLETED / SUPPORT-ONLY / EXACT-REPEAT`  
Verdict: `NORMAL_ONLY_NO_ANOMALY_SIGNAL / END_NCFC`

## Outcome

The frozen block-11 representation contains a strong paired clean J0
correction coordinate. Both readouts passed the clean normal gate, RN safety,
and every hard-null margin. The same clean correction also matched the ideal
anomaly correction in every cell: the raw and standardized `T(A)`/`A(T)`
correction cosines ranged from `0.955997` to `0.992563`, with Frobenius norm
ratios close to one. The exact `can` repeat reproduced every scientific metric
with maximum absolute difference `0.0` and an identical scientific signature.

That positive signal did not satisfy the preregistered anomaly-commutation
gate. Only `3/8` cells per composition order reached the required residual
cosine `>=0.90` in either readout; `6/8` reached `>=0.88`. The main failure was
held base 1, whose mask-balanced cosine was only `0.717--0.742`, even though
its mask-inside cosine remained `0.907--0.922`, correction cosine was
`0.974--0.993`, residual magnitude remained near one, and exact-RN geometry
was nearly unchanged. Raw and standardized results were almost identical, so
the failure is not attributable to the existing channel standardization/L2
readout.

The experiment therefore stops at a frozen-representation diagnostic. It does
not authorize LoRA training, public/test access, score correction, or a claim
that J0 is a deployable nuisance coordinate.

## Frozen experiment

- Backbone: offline frozen DINOv2 ViT-L/14, zero-based block 11.
- Readouts: raw `model.norm` patch tokens and the existing standardized/L2
  tokens, each pooled from the recorded native `48x104` grid to `16x16`.
- Data: immutable `can` support-16 memory plus four support-disjoint held
  `train/good` bases; both `J0_NEG=-0.1333 EV` and `J0_POS=+0.1333 EV`.
- Anomaly: fixed `G2_structural_scratch` construction per held base.
- Candidate: held-base paired clean correction
  `c*=R(x)-R(T(x))`.
- Controls: uncorrected B0, support mean N1, opposite sign N2, next-base N3,
  and next-held token-rolled anomaly-residual target N4.
- Locked exclusions: training, gradients, TESTpublic/private, GT, query images,
  query-memory update, dense persistence, and all hyperparameter sweeps.

## Aggregate gates and metrics

| gate / median metric | raw model.norm | standardized/L2 |
| --- | ---: | ---: |
| structural validity | pass | pass |
| normal translation | `8/8`, pass | `8/8`, pass |
| anomaly commutation | fail | fail |
| RN safety | `8/8`, pass | `8/8`, pass |
| hard null | `8/8` (`4+4` signs), pass | `8/8` (`4+4` signs), pass |
| clean correction cosine / norm | `1.000000 / 1.000000` | `1.000000 / 1.000000` |
| clean margin over N1--N3 | `+0.197509` | `+0.199989` |
| shifted-normal q99 reduction | `+0.004567` | `+0.004483` |
| `T(A)` correction cosine | `0.981885` | `0.981462` |
| `A(T)` correction cosine | `0.977838` | `0.977305` |
| `T(A)` commutation cosine | `0.893708` | `0.893822` |
| `A(T)` commutation cosine | `0.892075` | `0.892241` |
| `T(A)` / `A(T)` null margin | `+0.5429 / +0.5476` | `+0.5408 / +0.5468` |
| `T(A)` / `A(T)` residual norm ratio | `1.0091 / 1.0118` | `1.0100 / 1.0122` |
| median top-10 residual IoU | `1.0 / 1.0` | `1.0 / 1.0` |

All eight cells beat N4 by a large margin and retained residual magnitude, but
the strict gate intentionally requires absolute commutation as well as null
separation. A candidate can beat a poor shuffled target without being
sufficiently commutative, so the hard-null pass does not override the anomaly
gate.

## Per-cell localization

| held base | sign | raw `T(A)` / `A(T)` | standardized `T(A)` / `A(T)` | mask-inside range |
| ---: | --- | ---: | ---: | ---: |
| 0 | negative | `0.9137 / 0.9114` | `0.9142 / 0.9119` | `0.9977--0.9977` |
| 0 | positive | `0.8942 / 0.8919` | `0.8949 / 0.8924` | `0.9974--0.9975` |
| 1 | negative | `0.7376 / 0.7418` | `0.7206 / 0.7269` | `0.9070--0.9088` |
| 1 | positive | `0.7340 / 0.7341` | `0.7200 / 0.7169` | `0.9128--0.9218` |
| 2 | negative | `0.9238 / 0.9260` | `0.9237 / 0.9260` | `0.9970--0.9974` |
| 2 | positive | `0.9229 / 0.9155` | `0.9229 / 0.9154` | `0.9967--0.9974` |
| 3 | negative | `0.8836 / 0.8858` | `0.8837 / 0.8857` | `0.9906--0.9928` |
| 3 | positive | `0.8932 / 0.8923` | `0.8927 / 0.8921` | `0.9921--0.9938` |

The reported commutation cosine is the equal-weight mean of mask-inside and
mask-outside token cosines. Therefore the outside component can be recovered
algebraically as `2*balanced-inside`. Its range was `0.518--0.575` for held 1,
versus `0.776--0.855` for the other bases. This localizes the strict failure
primarily outside the anomaly mask, with a smaller held-1 degradation inside
the mask.

The failure is not explained by composition order or image-rendering error.
For held 1, `T(A)` and `A(T)` balanced cosines differ by at most about `0.0069`
within a readout/sign cell, and its pixel MAE (`0.0208`) is below held 2
(`0.0380`) and held 3 (`0.0268`). At the same time, held 1's mask-inside cosine
is genuinely lower than the other bases (`0.907--0.922` versus
`0.991--0.998`), and its interaction norm is higher (raw `0.157--0.184`,
standardized `0.111--0.132`). Thus an outside-mask metric effect is likely a
large component, but a base-specific representation mismatch may coexist.

The retained compact metrics do not include an activity-conditioned histogram
of target residual norms. Consequently they cannot distinguish between two
remaining explanations: unstable cosine direction where the outside-mask
target residual is nearly inactive, or a real background/spillover interaction
between J0 and the synthetic anomaly. The former is plausible but is an
inference, not a result of this experiment.

## Positive signal and scientific boundary

The useful result is narrower than the original training direction:

1. A clean-derived paired correction is exact on clean input and remains a
   strong predictor of the whole-image ideal anomaly correction (`8/8`).
2. Exact-RN rank, neighbor, and scale geometry is preserved in every cell.
3. Residual magnitude and strongest-token support are preserved even where
   the spatial direction gate fails.
4. The same pattern appears before and after the existing normalization, so a
   readout-only repair is not supported.

These facts say the normal coordinate is real and largely anomaly-compatible
at a global vector/topology level. They do not show uniform tokenwise
commutation, which was the prerequisite for reopening backbone adaptation.
The completed verdict therefore remains `END_NCFC`.

## Remote execution and reproducibility

- Completed run:
  `ncfc_s1_stage_minus1_dsba3_20260827T211122KST_v2`.
- Fixed project container: `hun_fsad_tta_012` on dsba3 GPUs 0--3; one held
  shard per GPU and one exact held-0 repeat on GPU0.
- Snapshot SHA-256:
  `efe549bf4da050242af1cff681875c0932d52d5fe8d2704532bff1908a6fb1b2`.
- Host/container GPU-map SHA-256:
  `a27a8597ba2d435fc05d059d2f9f28bdb323881bda68ad9bd9574c145492595d`.
- Model-state SHA-256:
  `59a755e385188971c57f52a38773028b508722c804ce6c9279d1dc2ac14ccbe8`.
- Scientific-signature SHA-256:
  `862099f411b543853b7b3374e509ba3936a103243d18f922a548212fd69c789a`.
- Aggregate SHA-256:
  `cda9eddaef3ae05644bab60254850618a5e5dd8ff312cfb59a0182f1dd6460da`.
- Exact repeat: metric maximum absolute difference `0.0`; signature identity
  pass.
- Independent local reaggregation was byte-identical to the remote aggregate.
- Retained pull: 18 files, 329,335 bytes total, largest file 77,703 bytes; no
  `.npy`, `.npz`, `.pt`, `.pth`, `.ckpt`, `.bin`, `.pkl`, or `.pickle` file.
- All forbidden-access flags are false. No test/public/private, GT, query, or
  dense artifact was accessed or retained.
- After compact pull and verification, only this run's exact immutable code
  snapshot was removed. The compact remote result remains. dsba3 GPUs 0--3
  were `1 MiB / 0%` at the final status check.

The preceding v1 attempt stopped in a result-free CUDA model smoke because the
native aspect-preserving patch grid was `48x104`, not the assumed `48x48`.
V2 only generalized deterministic pooling to recorded native grids with both
dimensions at least 16. No v1 scientific shard or result exists.

## Stop and next boundary

Close NCFC-S1 and do not sweep rank, layer, learning rate, action strength,
readout, or loss. In particular, do not attach a new LoRA on the strength of
the clean correction result alone.

If the measurement ambiguity is investigated, the only defensible next step
is a separately named, preregistered, frozen-backbone activity audit. It should
keep the identical bases, signs, compositions, readouts, and hard nulls, change
only the commutation summary by fixing a target-residual activity threshold
from support/synthetic construction before evaluation, and report active and
inactive token strata separately without retaining dense maps. That audit may
validate or reject the present cosine metric; it cannot revise this completed
verdict or authorize training by itself.
