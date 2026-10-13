# Direction A Literature Survey — Shift-Stable Construction Without Tail Prediction

Date: 2026-08-10

Scope: literature grounding for Direction A, defined after
`normal-tail-structure-predictability-all8-20260809` closed the
identify-then-suppress family. Direction A asks for constructions that do
**not** require predicting which positions inflate at test time, while staying
inside the R7 static-inference contract.

Status: survey only. No experiment is preregistered or licensed by this
document.

## 1. The Shift Is the Benchmark's Design, Not a Nuisance We Mismeasured

MVTec AD 2's test splits deliberately contain lighting conditions that are "not
necessarily contained in the training data" — at least four per scene (regular,
underexposed, overexposed, additional light sources), explicitly simulating
device-variation distribution shift.

This independently confirms the mechanism our G2 failure identified: the
nuisance that lifts normal scores at test time is **absent from `train/good` by
construction**. Our `rho ≈ 0` on 7 of 8 objects is the expected consequence,
not an artifact.

Reported robustness spread on this benchmark: EfficientAD and MSFlow lose more
than 10 pp AU-PRO under the shift, while RD and PatchCore lose ≤ 3 pp; state of
the art remains below 60% average AU-PRO.

Our own official private result sits at the robust end — Seg-F1 `-4.11%` and
AU-PRO `-2.32%` relative from private to private-mixed, the smallest
degradation among the four Main Table rows. The literature context strengthens
that this is a real property of the static support-bank design rather than
noise.

## 2. The Most Actionable Finding: RoBiS's Own Ablation

RoBiS is the DINOv2-B/14-reg Pfull row we trail by `-0.1285` private AU-PRO. Its
published ablation attributes its gains as:

| Module | Reported gain |
| --- | ---: |
| SAM-Finer (mask refinement) | **+12.5% SegF1** |
| Swin-Cropping (high-res tiling) | +3.7% F1-max |
| Data augmentation (noise + lighting) | +1.9% F1-max |
| MEBin (adaptive binarization) | +0.4% SegF1 |

The dominant term is **region refinement**, not shift handling and not
thresholding. Shift augmentation contributes `+1.9%` and adaptive binarization
`+0.4%`.

This redirects priority. Our largest measured gap against RoBiS is AU-PRO, and
our own diagnostics say `gt_component_recall_at_oracle` is `0.484` macro —
more than half of ground-truth components are never touched at the best
possible threshold. The literature's biggest lever and our biggest deficit are
the same thing: region completion.

Note the implication for the invalidated `region-completion-map-level`
protocol. Its target was correct; only its operator was wrong. The evidence now
says that target is where the comparable state of the art gets most of its
margin.

## 3. Cross-Image Score Comparability — How the Field Actually Handles It

Our region-completion failure left a specific requirement: a score scale that
stays comparable **across** images. The literature solves this at the
**threshold** stage rather than the score stage.

- **MEBin** (CVPR 2025, used by RoBiS) adaptively determines a per-image
  threshold via stable connected-component analysis, reporting lower FPR/FNR
  than fixed thresholds and Otsu and `+7.2%` F1 over fixed thresholding in its
  own setting. RoBiS combines it by logical OR with a `mean + 3*std` statistical
  threshold.
- General practice in tooling is the opposite of what our `R` did: normalize the
  display/decision range using statistics from a **calibration set spanning
  images**, explicitly warning against scaling each image to its own min/max.

That second point is the direct literature statement of our experiment-2 defect.
Per-image self-normalization is a known anti-pattern; our `fabric` collapse
(`seg_F1 0.6688 -> 0.0573`) is the textbook symptom.

Calibration is nonetheless not our main lever: our public-oracle to private
transfer lost only `-0.0708`, and MEBin buys RoBiS only `+0.4%`.

## 4. Synthesizing the Missing Shift — The Direct Answer to Our G2 Failure

Our G2 result says the shift cannot be *learned* from `train/good`. The
practical response in this literature is to *put it there*.

RoBiS applies to 50% of training samples:

- Gaussian noise, zero mean, `sigma = 15`
- Exposure change `I' = clamp(I * 2^lambda)`, `lambda ~ U(-0.2, 0.2)`

Few-shot memory-bank work makes the same argument from the other side: memory
banks are enriched with flips, rotations and translations because "the
variability at test time is likely to be significantly greater than in the
reference data".

This is fully compatible with our R7 static-inference contract — augmenting the
16 support images happens before inference and involves no test-time update.

Two cautions. First, the measured payoff is modest (`+1.9%`). Second, and
specific to us: our runner already carries `--support-brightness-range`
(currently `0.80,1.20`) and `--support-transforms`, and the deployed
Baseline1-core runs with `support_transforms=['identity']`. A brightness
jitter path therefore already exists in `superadd_bank.py` /
`superadd_patching.py` and is **unused** in the deployed configuration. Any
augmentation experiment must first establish what that existing hook actually
does, rather than adding a second mechanism.

## 5. Distribution-Shift-Robust AD — Mostly Outside Our Contract

The dedicated shift-robust AD line is active but assumes capabilities we have
foreclosed:

- **FiCo** (AAAI 2025) — `DiSCo` compensates distribution-specific information
  and `DiIFi` filters for distribution-invariant normality, in a teacher-student
  framework. Requires training.
- **Anomaly Detection under Distribution Shift** (arXiv 2303.13845) — minimizes
  the in-distribution/OOD normal gap "in both the training and inference
  stages". The inference-stage half is test-time adaptation.
- **Invariant Anomaly Detection under Distribution Shifts: A Causal
  Perspective** (NeurIPS 2023) — derives statistical conditions for invariant
  representations; again a learned-representation route.

Our contract is training-free with a frozen backbone and static inference.
These methods are therefore reference framing, not drop-in candidates. Their
existence does say that the invariant-representation route is considered to
require *learning*, which is consistent with our eleven closed branches failing
to obtain invariance from frozen features and unshifted normals alone.

## 6. What This Survey Changes

1. **Region completion is promoted above shift handling.** The comparable state
   of the art gets `+12.5%` from mask refinement versus `+1.9%` from shift
   augmentation, and region recall is our measured weakest link. The
   region-completion question should be reopened with a corrected operator.
2. **Per-image normalization is confirmed as an anti-pattern**, independently of
   our own failure. Any successor must normalize against cross-image statistics.
3. **Support augmentation is a legitimate, contract-compatible response to the
   G2 failure**, but with modest expected payoff and an existing unused code
   path that must be audited before anything new is built.
4. **The shift-robust AD literature does not offer a training-free static-
   inference method**, so Direction A cannot simply adopt one. Direction B
   (test-time adaptation) remains the route those papers take, and remains a
   claim-class decision rather than a technical one.

## 7. Open Questions Before Any Preregistration

- What exactly does the existing `support_transforms` / `support_brightness_range`
  path do, and why is the deployed configuration `identity`? Was it evaluated
  and rejected earlier, or never exercised?
- Is our AU-PRO gap to RoBiS attributable to refinement, to its Pfull budget, or
  to Swin-Cropping's high-resolution tiling? These are confounded in the
  published comparison and we cannot separate them from reported numbers alone.
- MEBin's `+0.4%` is measured on top of a `mean + 3*std` threshold within
  RoBiS's pipeline; it is not evidence about our raw-oracle threshold contract.

## Sources

- MVTec AD 2 dataset paper — https://arxiv.org/abs/2503.21622
- MVTec AD 2 dataset page — https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2
- RoBiS — https://arxiv.org/abs/2505.21152 , https://arxiv.org/html/2505.21152 ,
  https://github.com/xrli-U/RoBiS
- SuperAD — https://arxiv.org/abs/2505.19750
- AnomalyNCD / MEBin — https://arxiv.org/html/2410.14379
- FiCo — https://arxiv.org/abs/2412.10115 , https://github.com/znchen666/FiCo
- Anomaly Detection under Distribution Shift — https://arxiv.org/abs/2303.13845
- Invariant Anomaly Detection under Distribution Shifts —
  https://arxiv.org/abs/2312.14329
- AnomalyDINO — https://arxiv.org/pdf/2405.14529
- VAND 3.0 challenge — https://sites.google.com/view/vand30cvpr2025/challenge
