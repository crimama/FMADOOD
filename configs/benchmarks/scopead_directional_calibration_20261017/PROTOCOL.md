# Directional likelihood calibration — 2026-10-17

## Motivation and research objective

User requests an unsupervised method approaching oracle performance. Matched
reference alone completed24/24: baseline AUPRO58.49834 vs matched58.35434;
pooled-excluded58.13278. It does not improve over full-bank Q95. Existing oracle
on same seen AD2 2-shot is63.81953, a5.32119pp observed gap. These are finite
same-GT optimization diagnostics, not an achievable unsupervised guarantee.
We will measure candidate recovery=(candidate-Q95)/(oracle-Q95), not assume it.
A provisional research milestone is recovering at least half this observed gap
(~61.16 AUPRO at2-shot); it is an aspiration, not a claimed result or automated
selection threshold. Single-seed development gains require later repeated seeds.

## Statistical model and novelty sanity check

Unit DINO vectors live on S^(D-1). A normal-reference token is a vMF kernel
center; the bank is an equal-weight vMF mixture. For each resolution/layer,
choose concentration kappa by maximizing the mean source-LOO log likelihood,
equally weighting normal support sources and original random bags. Pool kappa
across bags, giving12 fitted parameters, not48 bag-specific fitted scales.
Source-own tokens are fully excluded for fitting; reference identities and
normal support budget are unchanged. No query features/labels, anomaly synthesis,
external normal training images, supervised oracle weight transfer or backbone
training. This is an adaptation of established directional statistics:
Banerjee et al., 'Clustering on the Unit Hypersphere using von Mises-Fisher
Distributions', JMLR6:1345–1382,2005.
https://www.jmlr.org/papers/v6/banerjee05a.html
That work supports the normalized-feature likelihood model, not a guarantee of
anomaly localization performance or novelty of vMF mixtures.

For dimension d and concentration k:
log(C_d(k)/C_d(0)) = (d/2-1)log(k/2)-logGamma(d/2)-log I_(d/2-1)(k).
log likelihood ratio to uniform sphere for q and bank b_j:
log(C_d(k)/C_d(0))+logmeanexp_j(k*q dot b_j).
Use exact sphere normalizer with scaled Bessel (positive power series fallback
for high-order underflow); kappa=0 uniform is a feasible model. Bounded scalar
log-kappa optimization over[1e-3,1e7] is a numerical range; boundary hits recorded.
Equal source/bag likelihood weights; spatial tokens remain dependent. No iid
confidence intervals or conformal/FPR coverage claim.

## Controlled arms and limitations

1. Q95 unchanged baseline.
2. concentration_nn: same source-LOO maximum-likelihood kappa; score k*(1-max cos).
   Keeps cosine1NN matching and changes only per-bank scale. This score is not
   itself the exact mixture likelihood ratio; it isolates concentration scaling.
3. mixture: negative normal-mixture log-density ratio to uniform sphere. This
   changes matching to logmeanexp over the fixed bank. Exact normalizer retained;
   negative anomaly scores are valid and supported by the unchanged evaluator.

Q95 divisions remain in metadata solely for audit and are not used by arms2/3.
Metadata explicitly declares changed scoring and the role of those audit scales.
All arms keep bank features, tokens, support manifest, backbone, noise routing,
interpolation, layer/resolution/bag fusion and data fixed. Query bank uses all S
sources. LOO reference-size effects on density estimate uncertainty may remain;
this model does not claim to remove every support/query distribution mismatch.
An isotropic kernel per resolution/layer can be misspecified; outcome uncertain.
No learned/fixed global layer multiplier is copied from oracle. No automatic
rejection of shallow; linear0125 is retained.

## Execution and budget

AD2 test_public seen development,2-shot,seed0,8classes. VOID_support_2.json.
FP16 DINOv2-L layers5/11/23;448/672/896/1120;4bags cap256/source;
linear0125, original noise guard,CQoff. Primary AUPRO@.05 (higher);secondary
pAUROC@.05. Evaluation labels only used in common metrics.
24cells=3arms x8classes; four assigned dsba3GPU0/1/2/3; existing
hun_fsad_tta_012; immutable stage-only release. Timeout10800sec/cell.
Stop after24cells; no extra parameter sweep, private upload or periodic monitor.
Output: compact metrics/config/manifests/logs, fitted kappa/audits. Maps and
weights off. Use existing run_experiments.sh/experiments.py and collector.
Validate sphere normalizer against analytic3D, known-distribution MLE, bank
identity against Q95, exact brute-force mixture/NN scores and no query updates.
