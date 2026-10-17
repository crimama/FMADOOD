# Joint normal-score statistical calibration — 2026-10-17

Hypothesis: scale-only marginal normalizers omit covariance and heterogeneous
normal fluctuations. A Gaussian positive-mean subset likelihood ratio may
retain sparse defect evidence while accounting for normal score dependence.
This is a development hypothesis, not a claimed improvement.

## Fixed comparison and budget

- dsba3 existing hun_fsad_tta_012 container; the same four explicitly assigned UUIDs.
- AD2 test_public, already seen during development; seed0, VOID_support_2.json,
  all eight classes, FP16 DINOv2 ViT-L/14 layers 5/11/23, 448/672/896/1120.
- Same random cap256/source/bag, four bags, cosine 1NN, no CQ/postprocessing.
- Deep mean + 0.125 shallow, normal source LOO Q95 where declared.
- Primary higher-is-better AUPRO@0.05; secondary pAUROC@0.05; F1 oracle diagnostic.
- Six arms x eight classes = 48 cells. No anomaly maps or weights.
- Controls: none+mean, none+order, Q95+order. New: normal scale zscore+order,
  diagonal Gaussian subset GLRT, covariance-shrunk Gaussian subset GLRT.
- Stop after the complete declared comparison; diagnose execution failures
  separately. Extend shots/seeds only after the complete comparison merits it.

## Statistical model

At fixed normalized image coordinates collect 16x16 normal support probes,
interpolating *scores* bilinearly after exact source-excluded matching.
The per-resolution vector has the same fixed layer fusion and bag mean as
prediction. The resolution-score mean and population standard deviation are
estimated with equal weight per support image.

For covariance: C=(1-alpha) empirical_correlation + alpha I + 1e-6 I.
Alpha minimizes equal-source-block normal Gaussian negative log likelihood
with scalar bounded optimization and both endpoints included. The diagonal
arm fixes alpha=1. Numerical floors and tolerance are not label-tuned.

H0: z~N(0,C). H1: z~N(a 1_S,C), a>=0, for every nonempty resolution subset S.
Report sqrt(2 GLR)=max_S max(0, 1_S^T C^-1 z)/sqrt(1_S^T C^-1 1_S).
The maximum includes all 15 subsets; there are no learned defect weights or
hand-selected resolution subsets. Normalization and dependence estimation
are essential parts of this specified model. Gaussian likelihood ratios and
shrinkage covariance are established statistics, not a new statistical family.

## Limits and validity

Spatial probes are dependent. Source-LOO score groups also share reference
banks; source-block selection is a predictive criterion and does not establish
independent CV folds, conformal coverage, or calibrated p values.
Support source-LOO banks use n-1 images while prediction uses n; this existing
calibration/reference mismatch is measured in prior diagnostics and remains
uncorrected here. Full covariance on only two independent support sources
is exploratory; report alpha, source means and correlation, not nominal
significance. A Gaussian normal model can also be misspecified for multimodal
objects or heavy tails. Test labels enter evaluation only, never fit/predict.
The seen-public scores may guide further development and are not sealed evidence.

The none+order control isolates Q95's contribution to the existing order-pooling
gain; comparison only to none+mean cannot attribute that gain to calibration.
