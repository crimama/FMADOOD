# SIAI-D1 J0 sign/saturation diagnostic Stage -1 preregistration

Date: 2026-08-27  
Protocol: `siai-d1-j0-sign-diagnostic-20260827`  
Status: `PREREGISTERED / POSTHOC EXPLORATORY / SUPPORT-ONLY`

## Question and non-claim boundary

The completed SIAI-S1 oracle failed its confirmatory correction/anomaly gates
and is closed.  Its compact diagnostics exposed a large exposure-sign
asymmetry: positive-EV bases saturated about 30% of pixels with inversion MAE
near 6.8, while negative-EV bases had zero saturation and MAE near 0.35.
SIAI-D1 asks only whether that pixel-space asymmetry also explains the frozen
DINO feature failure.

This is a posthoc mechanism diagnostic, not an independent confirmation and
not a performance estimate.  All four previously selected held bases remain;
positive-EV cases may not be removed, reweighted, replaced, or hidden.  A
positive result cannot promote J2, an estimator, LoRA, a four-object panel, or
public/test evaluation.  A negative result ends explicit image-space exposure
inversion as the next architecture basis.

## Frozen data and arms

- Object: `can` only.
- Support: the immutable SuperAD 16-shot DINOv2 manifest with SHA-256
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Held normals: the same deterministic four support-disjoint `train/good`
  bases used by SIAI-S1; exactly two must receive `+0.1333 EV` and two
  `-0.1333 EV`, otherwise the run is invalid.
- Held anomaly generator: `G2_structural_scratch`; shift `J0` only.
- Frozen DINOv2 ViT-L/14 block-11 readout, `16x16` pooled tokens, exact 1-NN
  RN.  No learning, estimator, adapter, router, retrieval, score fusion, or
  query update.
- Exact arms for every base and sign group:
  `B0_SHIFTED_FROZEN_RN`, `N4_STATIC_MEAN`, `N1_WRONG_SIGN`,
  `O_TRUE_ACTION`, `O_PAIRED_CLEAN`.

The runner must emit the same correction, anomaly-residual, normal-q99, RN
rank/neighbor/scale, pixel reconstruction, wrong-sign, identity no-op, and
custody measures as SIAI-S1, now for each held base and for fixed positive- and
negative-EV groups.  Only compact scalar JSON/log/hash evidence may persist.

## Frozen diagnostic gates

All structural/custody/source/output checks and exact repeatability
(`scientific_metrics_max_abs_diff <= 1e-7` plus identical scientific-signature
SHA-256) are mandatory.

`NEGATIVE_EV_REVERSIBLE_SIGNAL` requires the negative-EV `O_TRUE_ACTION` group
to satisfy all of:

- correction cosine `>=0.90`, correction norm ratio in `[0.80,1.25]`;
- anomaly cosine `>=0.90`, anomaly norm ratio in `[0.80,1.25]`;
- RN rank correlation and neighbor retention `>=0.90`, RN scale ratio in
  `[0.80,1.25]`;
- correction cosine margin over `max(N4_STATIC_MEAN,N1_WRONG_SIGN) >=+0.01`;
- paired-normal pixel-MAE gain over wrong sign `>=+0.10`.

`SATURATION_ASYMMETRY` requires both:

- median negative-EV pixel MAE is at least 80% lower than positive-EV MAE;
- negative-EV correction cosine exceeds positive-EV correction cosine by
  `>=0.10`.

For both signs, the `O_PAIRED_CLEAN` instrumentation ceiling must have
correction and anomaly cosine `>=0.99`, RN rank and neighbor retention
`>=0.99`, and RN scale ratio in `[0.99,1.01]`.

The only positive diagnostic verdict is `CLIPPING_ASYMMETRY_CONFIRMED`, which
requires both diagnostic gates and both paired-clean ceilings.  Otherwise the
verdict is `NO_REVERSIBLE_SIGN_SIGNAL`.  Neither verdict changes the closed
SIAI-S1 result.

## Stop and continuation rule

If `CLIPPING_ASYMMETRY_CONFIRMED` is false, stop explicit image-space exposure
inverse work.  If true, it licenses only design work for a separately named
clipping-aware positive-EV recovery sentinel; it does not license training or
benchmark evaluation.  Thresholds, held bases, sign assignment, transform
magnitude, backbone/readout, or arms may not be changed after launch.

## Frozen source identity

Result-bearing launch is forbidden until runner, aggregator, remote controller,
tests, and machine-readable source manifest are complete and reviewed.  Before
launch, freeze runner, aggregator, inherited SIAI action/readout bundle,
normalized remote-controller, source-manifest, complete controller-file, and
configuration SHA-256 values here.  Remote preflight must recompute the exact
manifest and normalized-controller identity from the immutable snapshot.

Frozen before result-bearing launch:

- runner SHA-256: `92f36f2ca8cc6c6d108bb89af4e4a0d755bb32bef4ed51a34a7215dc9b4b846f`
- aggregator SHA-256: `992df4e3cba42f6c24aec036dea2c0f6afa57f26bc28f9df9abef4fdee616d29`
- inherited SIAI-S1 runner SHA-256:
  `190c20ceb4e46b4cd9bc2aae09c2d3ab37cf8fb1b8653151262cf425b7579283`
- SIAI action-core SHA-256:
  `27e39ce7f5c6de1c27d34a958dadca23bb19ffff64e3feb36264760a71772dc2`
- frozen readout-bundle SHA-256:
  `212fe5ce91cd6c5ae2cb43e921a70a051639d015eb788960de5c311b3419f033`
- normalized remote-controller SHA-256:
  `41fbf3f313a8c62e0f355b17a22c68a4c69bab1e5b4f90fef2b4d25384c44154`
- source-manifest SHA-256:
  `35a307cd213b1b7633876379ece6666b00a394fd1817d5215352586259aa226b`
- complete controller-file SHA-256:
  `59af91b28bd64730c33f0d987b11554acb33fba606b46e7f6582464e622d3e15`
- frozen configuration SHA-256:
  `fe2459fba1811094c2d38d71d62351ea522230e73a0250257a0f189ab4d5d233`

### Pre-result launch amendment 1

The first launch passed remote tests and DINO cache preflight but failed before
support loading, feature extraction, or result creation because the diagnostic
runner referenced `CIDER` on the inherited SIAI-S1 module instead of its
already-frozen inner runtime.  The failed snapshot was removed by its exact
run name.  The sole scientific-code change aliases that inner runtime and
redirects existing data/shift utility references to it; gates, data, signs,
arms, metrics, and configuration are unchanged.  These identities supersede
the corresponding launch values above:

- amended runner SHA-256:
  `6dd826a49e27d66996e0083f73e34205cfb30c2a8317a317e193880bd29539ca`
- amended source-manifest SHA-256:
  `1d83713d8f325a992c45ed73db95ed972681e361c494d6b814abee31dc195642`
- amended complete controller-file SHA-256:
  `326ca4cb7ac8556a7e5d43cf348315ce07db45d7bd9615f31b531e24b69f23bd`

The normalized remote-controller, aggregator, inherited SIAI-S1 runner,
action core, frozen readout bundle, and frozen configuration hashes are
unchanged.

## Remote execution boundary

All result-bearing work runs only in the project-owned dsba3 container
`hun_fsad_tta_012` on GPU0.  Preflight may query only selected GPU0's index,
UUID, memory, utilization, and compute UUID and must abort when occupied.  It
must not enumerate, inspect, modify, stop, or delete any other user's
container, process, file, volume, GPU, or job.  Cleanup may remove only the
exact immutable SIAI-D1 snapshot named by this run; compact results remain.
