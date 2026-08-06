# SDIG-NODyR Stage A Preregistration

Date: 2026-07-17
Status: FROZEN BEFORE REMOTE METRICS / TESTPUBLIC SHADOW ONLY

## Motivation and Boundary

The prior position, hard-local, density, support-consensus, and condition-aware
branches all modify a score field that already entangles acquisition nuisance
with defect evidence. This experiment instead tests a new cross-layer relation:
whether normal patches have a predictable depth-transition direction whose
violation remains localized under MVTec AD2 acquisition shift.

This is Stage A raw evidence only. It excludes score calibration, fusion,
query-dependent updates, support routing, density, hard-local matching,
foreground priors, RGB guidance, and morphology.

## Dataset and Compute Contract

- `target_dataset=MVTec AD2 single-image`
- `data_root=/home/hunim/Volume/DATA/mvtec_ad_2`
- objects: `can,fabric,fruit_jelly,rice,sheet_metal,vial,wallplugs,walnuts`
- split: full `test_public/good,bad`
- backbone: frozen `dinov2_vitl14`
- resolution: 672 except `sheet_metal` 448
- support: exact recorded SuperAD-selected 16 paths per object
- host: dsba3, GPUs `0,1,2,3`; host IDs and container CUDA slots recorded separately
- primary metrics: pixel `seg_AUROC_0.05` and best-threshold raw `seg_F1`
- output requirement: 100% finite full-image map coverage

## Frozen Candidate

For DINOv2 layers 15 and 23, with per-token layer normalization:

```text
x = LN(F15)
d = unit(LN(F23) - x)
d_hat = unit(Wx + b)
E = 1 - cosine(d_hat, d)
```

`W,b` are fitted per object on all patches from the same 16 normal support
images using centered dense isotropic ridge regression with fixed relative
ridge `0.01`. There is one original-image forward per query and bilinear
upsampling only.

## Intrinsic Controls

- `state_angle`: predict normalized `F23` state and score angular state error.
- `state_l2`: predict `F23` state and score mean squared residual (RD-style null).
- `raw_delta_l2`: predict the unnormalized depth delta and score mean squared residual.
- `shuffle_nodyr`: train the same direction predictor with cyclic cross-image
  pairing and a SHA256-seeded toroidal spatial offset, preserving image and
  spatial marginals while breaking true pairing.
- `unit_delta_l2_equiv`: numerical audit only. For unit vectors it is exactly
  `2 * nodyr`; it is excluded from the independent hard-null envelope and
  cannot support a novelty claim.

The independent envelope is the per-metric maximum over the first four nulls.
If `state_l2` or another null ties/beats NODyR, the transition-direction method
claim is killed as residual prediction under reparameterization.

## Run Order

1. Local unit and CLI tests.
2. Four-object source-`000` full-condition-orbit smoke, one object on each host
   GPU, for shape, finiteness, support identity, map inventory, and
   algebraic-equivalence checks. Smoke metrics are not used for decisions.
3. If smoke is valid, full all-eight run with two objects per host GPU.
4. Aggregate per-object, macro, split-condition upper-tail, transition-norm
   correlation, map coverage, and candidate-minus-null-envelope metrics.

## Gates

Shadow causal-signal audit (not a promotion decision):

- NODyR macro margin over the independent envelope is at least `+0.005` for
  both `seg_AUROC_0.05` and `seg_F1`;
- all eight regular object cells satisfy no-harm;
- shifted macro gain is at least `+0.005` on both metrics with no catastrophic
  cell below `-0.02/-0.03`;
- score reduction is not explained by bad-pixel recall collapse;
- full finite map coverage is 100%;
- source-orbit bootstrap lower bounds are positive.

This run does not implement support-only fixed calibration. `test_public` is a
shadow population and cannot select layers, ridge, continuation, or method
promotion. If source-orbit bootstrap or same-run SuperAD/RN-FMLK evidence is
absent, no claim-tier verdict is allowed even if the shadow audit is positive.

The shadow result is reported descriptively as positive, mixed, or negative.
It does not authorize a parameter change. A separate independent-development
preregistration is required before any continuation decision.

## Baseline Contract

Same-reference recorded SuperAD-16, RN-FMLK/full-memory DINO kNN, and H+ rows
are context until exact same-run artifacts are attached. They cannot produce a
`KEEP` or `CONDITIONAL KEEP` verdict for this run. The first decision is the
candidate-versus-intrinsic-envelope causal diagnostic.
