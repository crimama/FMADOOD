# CIDER-CF: Synthetic Counterfactual Design

Date: 2026-08-24  
Status: `DESIGN_STAGE / CONDITIONAL_KEEP`  
Research score: `0.88` after five stages  
Target: MVTec AD2 single-image

## Executive decision

The next admissible mechanism is **CIDER-CF**: Cross-fitted Cross-synthesis
Invariant Defect-Evidence Ratio. It does not train a network to recognize
synthetic masks. Instead, heterogeneous synthesis families act as independent
counterfactual instruments. Only residual directions that are (a) unexplained
by normal nuisance and (b) shared across synthesis environments define the
anomaly alternative. Inference is one calibrated covariance likelihood-ratio
test.

This is a design-stage candidate, not a positive result. Its first admissible
experiment is support-only; TESTpublic remains closed.

## Negative evidence intake

- FACT-RN was nearly identity-like and matched by its part-frame shuffle.
- SO-CFM's synthetic guard and SPCT's synthetic margin were exactly inactive.
- Shift-conditioned Flow learned its augmentation proxy but worsened normal
  shift and suppressed defects on fragile objects.
- Direct feature surgery, RN reranking/gating/fusion, support admission, and
  synthetic normal-memory expansion are closed families.

Therefore a successor must make synthetic evidence active, causally testable,
and independent of both renderer identity and RN score manipulation.

## Literature boundary

- CutPaste, DRAEM, NSA, and DeSTSeg learn representations or segmentation from
  synthetic positives and masks.
- SimpleNet separates projected normal features from Gaussian synthetic
  anomalies.
- RealNet uses diffusion synthesis for anomaly-aware feature/residual selection.
- GLASS broadens image- and feature-level anomaly synthesis, including
  near-manifold feature perturbations.
- Outlier Exposure motivates diverse surrogate anomalies; Fishr motivates
  environment-invariant evidence rather than pooled ERM.

CIDER's contribution is not another generator. It changes the role of synthesis
from positive-label generation to **instrumentation of an explicit alternative
hypothesis**, with leave-generator-out and generator-confusion falsification.

## Frozen Stage -1 representation

- Backbone: frozen DINOv2 ViT-L/14.
- Descriptor: layer 12 (`0`-indexed layer `11`) patch tokens after token-wise
  LayerNorm and L2 normalization.
- Canvas: shorter side `672`, then deterministic adaptive pooling to `16x16`.
- Support: exact SuperAD-16 paths, manifest SHA-256
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Targets: `can`, `fabric`, `vial`, `walnuts`.
- Offline source for target `t`: only the other seven objects' `train/good`.
- Target synthetic anomalies: evaluation-only after every parameter and rank is
  frozen.
- Forbidden: public/private/test/GT enumeration or access, filename condition
  tokens, query adaptation, score fusion, and target-synthetic model selection.

## Counterfactual environments

Each source normal image receives paired masks, areas, and strengths across five
families so the family changes while intervention support is controlled:

1. `G0_poisson_patch`: Poisson-blended normal patch from a different source.
2. `G1_material_blotch`: smooth local chromatic/texture/material perturbation.
3. `G2_structural_scratch`: thin spline scratch, erase, or duplicated edge.
4. `G3_feature_ascent`: bounded near-manifold feature perturbation without an
   image-space alpha boundary.
5. `G4_frequency_local`: localized Fourier amplitude/phase perturbation.

The training environments rotate: one family is held out in each fold. Metrics
use the eroded mask interior, boundary ring, and broad interior separately.

Normal nuisance instruments are separately generated from exposure/gamma,
white balance, smooth illumination fields, mild blur/compression, and bounded
subpixel affine transforms. They define `H0`, never anomaly positives.

## Cross-fitted mechanism

For frozen feature map `phi` and intervention `a_g`, form patch residuals

`delta_g(x,m,s) = phi(a_g(x;m,s)) - phi(x)`.

For target `t`, use only source objects `o != t`.

1. Estimate source nuisance covariance `Sigma_N` from normal-preserving paired
   residuals.
2. Whiten synthetic residuals: `z_g = Sigma_N^(-1/2) delta_g`.
3. Fit rank `r=16` orthonormal `U` and nonnegative `Lambda` by maximizing the
   minimum standardized interior energy across training generators while
   penalizing between-generator energy variance and outside-mask energy.
4. Reject the fold if a linear probe predicts generator identity from
   `U^T z_g` above the frozen chance margin.
5. Freeze `U`, `Lambda`, rank, and all regularization constants before target
   support is opened.

Map the whitened defect basis back to descriptor coordinates as
`B = Sigma_N^(1/2) U`. Target support16 estimates only the normal hypothesis.
For part `p`, freeze the normal center as the coordinate-wise median of its 16
normalized descriptors and use fixed
shrinkage

`Sigma_0,p = 0.9 Sigma_N + 0.1 diag(Sigma_target,p) + 1e-4 I`,

where target statistics come from support images and their normal nuisance
derivatives. The anomaly alternative is

`Sigma_1,p = Sigma_0,p + B Lambda B^T`.

For query residual `r_p = phi(q)_p - median_s(phi(s)_p)` from the robust support
part center, deploy exactly one
statistic:

`LLR_p = 0.5 [r_p^T(Sigma_0,p^-1 - Sigma_1,p^-1)r_p
              - log det(Sigma_1,p Sigma_0,p^-1)]`.

The final map is the monotone empirical upper-tail transform of `LLR_p` under
source-excluded support LOO normals. This calibration cannot change ordering and
is not score fusion. Query images never update means, covariances, calibration,
or the defect basis.

## Why this is not a closed-branch retread

- Not FACT-RN: no learned canonical transport and no query feature change.
- Not RN-FMLK: no nearest-neighbor distance is the deployed statistic.
- Not score fusion/gating: one LLR is emitted.
- Not RealNet/SimpleNet: no synthetic-positive classifier or synthetic-driven
  feature selector is deployed.
- Not SO-CFM/SPCT margin: synthetic evidence defines an active covariance
  alternative and must pass a pre-fit energy gate.
- Not an oracle: all source training is leave-target-out; target synthetics are
  evaluation-only; no benchmark query or condition token enters fitting.

## Stage -1 arms

- `B0_H0_local_covariance`: normal covariance statistic without defect expansion.
- `B1_pooled_synthetic_covariance`: pooled-family alternative.
- `M0_CIDER_CF`: full worst-family cross-fitted method.
- `A0_no_nuisance_whitening`.
- `A1_mean_family_not_worst`.
- `A2_no_part_conditioning`.
- `N0_family_label_shuffle`.
- `N1_mask_shuffle`.
- `N2_boundary_only`.
- `N3_source_part_shuffle`.
- `N4_normal_transform_alternative`.
- `N5_random_same_rank_spectrum`.
- `N6_single_best_generator`.

RN-FMLK is not needed to interpret synthetic Stage -1, but becomes mandatory
with same-run SuperAD-16 if a real anomaly benchmark is authorized.

## Stage -1 gates

Validity requires exact custody, equal arm/family coverage, deterministic repeat,
finite/nonconstant maps, covariance eigenvalue floor `>=1e-6`, condition number
`<=1e6`, and support LOO tail max/min `<=2`.

Mechanism pass requires all of:

- rotating held-family interior pAUROC_0.05 `>=0.75` macro and `>=0.65` each;
- local and broad held-family coverage `>=0.95`;
- held normal-nuisance FPR `<=0.05` macro and `<=0.08` per object;
- generator-ID balanced accuracy `<= chance+0.10`;
- `M0` beats `B1` and every hard null by `>=0.03` macro;
- `M0` wins against mask, boundary, source/part, and random-spectrum nulls on
  at least `3/4` objects;
- target-synthetic activation rate in the frozen `U` space lies in `[0.20,0.90]`
  so the test is neither inactive nor universal.

Fatal stop applies if any forbidden path is touched, target synthetics influence
fitting, any held family is below `0.60`, generator accuracy exceeds
`chance+0.15`, mask/boundary null is within `0.01`, three objects are null-matched,
or nonconstant coverage falls below `0.95`.

Soft continuation requires held-family macro `>=0.70`, nuisance FPR `<=0.08`,
generator accuracy `<=chance+0.15`, and positive margin over both pooled
covariance and mask shuffle on at least `2/4` objects. It authorizes one revision,
not public evaluation.

## Later real-anomaly promotion contract

Only a strict Stage -1 pass authorizes all-eight full TESTpublic. The benchmark
must use the exact same 16 references for CIDER and same-run SuperAD, plus
full-memory DINO RN-FMLK. Unified metrics are `seg_AUROC_0.05`, best-threshold
`seg_F1`, positive map coverage, object wins, hard-null margin, and no-harm.

Strict claim requires positive CIDER deltas versus both SuperAD and RN-FMLK on
macro pAUROC and F1, at least `6/8` object wins, coverage `>=0.90`, every hard
null beaten, and no strong/control object below `-0.01` on either primary metric.

## Risks and interpretation

- Cross-generator consensus can still encode shared mask geometry; boundary and
  feature-space environments are therefore fatal controls.
- Cross-object directions may not transfer to a held target.
- Gaussian LLR is an approximation; empirical LOO calibration and condition
  diagnostics carry validity.
- Consensus favors recurring defect families and may miss genuinely novel
  orthogonal defects.
- A synthetic pass proves only instrument validity. Real anomaly alignment is
  unknown until the separately gated benchmark.

## Recommended next action

Implement only the deterministic synthetic instrument generator, covariance
core, support-only runner, fail-closed aggregator, and synthetic dry-run. Do not
open TESTpublic or start an all-eight run during this implementation stage.
