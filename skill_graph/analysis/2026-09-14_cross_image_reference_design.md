# CRR: Cross-image Reference-aware Residual Ranking

Original status: DESIGN ONLY, 2026-09-14. This document itself does not authorize
implementation, executable configuration, training, evaluation or remote sync.
Subsequent user instruction authorized implementation, review, local/dsba2
setup and execution; see the [implementation protocol](../../configs/methods/CROSS_IMAGE_REFERENCE.md)
and current.md for verified current status. The design below is preserved.
This is a post-hoc continuation of TBF, not a preregistered explanation of TBF.

## 1. Decision and evidence

Keep R1 as the detector and learn a small score residual. Test two additions
sequentially: cross-image normal-tail comparison, then reference-distribution
descriptors. Do not add another backbone, adaptation loop or auxiliary loss.

- Verified outcome: TF3 versus TF0/R1 gives +0.29229pp AUPRO@.05 but
  -0.47940pp pAUROC@.05. The intended joint improvement was not achieved.
- Structural observation: TBF ranks positives against normals from the same
  episode; its pair term preserves synthetic contrast across a shift, not
  absolute normal-score invariance. Evaluation pools pixels across query images.
- Structural observation: the TBF local gate observes two distances only.
  Its correction is `(w - .5) * (d11 - d23)` in normalized units; it cannot
  correct a point where the layer distances agree.
- Hypotheses, not established causes: cross-image ordering is underconstrained;
  two distances cannot distinguish a rare normal match from an anomalous match.
  Synthetic-to-real mismatch remains a competing explanation.

Evidence: [compact results](../../results/comparisons.md),
[TBF protocol](../../configs/methods/TAIL_AWARE_FUSION.md),
[TBF implementation](../../src/fmad/methods/tail_aware_fusion.py),
[R1 implementation](../../src/fmad/methods/reference_matching.py), and
[pooled evaluator](../../src/fmad/evaluation/streaming.py).

## 2. Fixed conditions and scope

Retain normal16, seed0, the exact support manifest, frozen DINOv2 ViT-L14,
layers11/23, resolutions448/672, R1 token selection and leave-source-out MAD
scales, native interpolation, preprocessing and the existing metric contract.
Fit a separate small head per class, shared between its two resolutions.
No query fitting, ground truth in the method, test-label calibration, backbone
updates, altered reference-bank contents or test-time augmentation.

The target is pixel-score ranking, not image-level detection or representation
learning. Normal support data and synthetic masks are the only fit supervision.

## 3. Reference-aware descriptors

Let B_l contain the existing normal reference tokens at layer l and one
resolution. Let owner(j) identify the source support image of token j.
Use cosine distance delta, with numerical negative distances clamped to zero
for descriptor construction. Preserve the exact existing R1 score path.

For query token q, define independently at each layer:

```
d_l(q)       = min_{j in B_l} delta(q, j)
j*_l(q)     = argmin_{j in B_l} delta(q, j)
D_l(q, s)   = min_{j in B_l: owner(j)=s} delta(q, j)
m_l(q)      = mean of the three smallest D_l(q, s), over distinct sources s
rho_l(j*)   = mean of the three smallest D_l(j*, s), s != owner(j*)
```

The last two quantities describe support from several normal images and the
normal spread around the selected reference. They are evidence features, not
calibrated probabilities or a rule that multi-source agreement is always normal.
The head may learn to suppress or increase a score depending on this evidence.
Do not force both layers to select one shared reference identity, as R2 did.

During a training episode from source i, exclude i from ALL reference searches,
including the construction of rho. Also exclude the selected reference's own
source when computing rho. At inference use all16 sources for matching and
exclude only the reference owner's source for rho. This preserves the existing
15-source fit versus16-source inference difference; it does not eliminate it.

Precompute D_l(j, s) for reference tokens using chunked distance products;
retain only the per-reference/per-source minima, never the full pairwise matrix.
Query-to-bank chunks can reduce the same distance product into both the R1
minimum and the per-source minima. Fix the number of distinct sources at3.

Using the unchanged scale sigma_l, construct six scalar feature grids:

```
u(q) = [d11/sigma11, d23/sigma23,
        rho11/sigma11, rho23/sigma23, m11/sigma11, m23/sigma23]
```

Interpolate these raw scalar grids to the requested native pixel coordinates
using the existing interpolation convention, THEN apply log1p(max(u, 0)) for
the head. Use identical ordering in fit and predict. The baseline score uses
the original R1 distances, not distances reconstructed from log features.

## 4. Bounded additive readout

At resolution r and native pixel p:

```
s0_r(p)    = existing R1 score = (d11/sigma11 + d23/sigma23) / 2
s_theta(p) = mean_r [s0_r(p) + alpha * tanh(h_theta(z_r(p)))]
alpha      = 0.5, in R1 normalized score units
h_theta    = Linear(6,16) -> GELU -> Linear(16,1)
```

There are129 nominal trainable parameters. Zero-initialize the last layer;
all trained stages start with the same tensors and an initially zero residual.
CR1/CR2 set the four geometry input slots to zero; CR3 exposes them. Nominal
parameter counts match, but effective input capacity changes and must be reported.
This readout can act when d11 equals d23. Its magnitude bound is not an AUPRO,
pAUROC or ranking guarantee. A shared monotone rescaling alone cannot improve
rank metrics; useful changes require differential corrections across pixels.

## 5. Cross-image normal-tail ranking

Reuse the64-episode source/family/shift deck from TBF, including four views
`x, T(x), A(x), T(A(x))`, actual-change masks and sampled native coordinates.
Keep rectangle_cutpaste/scratch_noise and the same photometric shift parameters.
Cache only frozen sampled scalar descriptors before optimization.

- N: sampled clean and shifted normal pixels from all episodes of this class.
- P: actual-change pixels in synthetic and shifted-synthetic views.
- Shifted normals are negatives; shifted synthetic anomalies are positives.
  There is NO separate shift-pair consistency loss in any trained CR stage.
- No dataset objects, query samples, evaluation masks or labels enter this pool.

For sampled normal pixels use mass weight H*W / n_sampled in their view. For
sampled positive pixels use actual_changed_area / n_sampled in their view.
Each of the64 declared episodes contributes its two views to the corresponding
virtual population. This is a pixel-weighted SYNTHETIC population, not an
estimate with guaranteed transfer to the real query distribution.

At each update, rescore cached descriptors with the current head in bounded
chunks. Mine the highest-scoring5% of normal MASS, not5% of unweighted sample
count. Fractionally include the boundary mass; use deterministic tie ordering.
Selection is stop-gradient; recompute selected scores with gradients. Do not
reuse stale cached learned scores between updates.

```
L_rank = E_{p+ in P, n- in Tail_0.05(N)}
         [0.5 - s_theta(p+) + s_theta(n-)]_+
```

CR1 restricts each positive's negative tail to its own episode's two normal
views. CR2/CR3 use the entire within-class pool. Do not mix categories, enforce
source-balanced tail quotas, or exclude same-source pairs from the global pool.
Log the resulting cross-source pair fraction and tail source coverage.

Why this addresses one blind spot: adding an episode-specific constant c_i to
all its scores cancels in a same-episode ranking term. A cross-image term has
an additional c_negative - c_positive and generally changes. This removes an
objective invariance; it does not prove that real-data offsets caused TBF's loss.

Fix64 AdamW updates, learning rate.01, weight decay0, gradient clip5 and
margin.5. At each update draw128 positives with their mass weights and128
tail negatives per positive:16384 pair terms. Use separate deterministic RNG
streams for episode generation, positive sampling and negative sampling.
Positive identities and candidate descriptors match across trained stages;
selected negatives may differ by the declared ablation. No additional loss.

## 6. Sequential ablation, not a pilot program

| Stage | learn_residual | global_tail_pool | reference_geometry | Question |
|---|---|---|---|---|
| CR0 | false | false | false | Exact same-code R1 reference |
| CR1 | true | false | false | New residual-learning control |
| CR2 | true | true | false | Does cross-image tail comparison help? |
| CR3 | true | true | true | Does reference-distribution evidence add value? |

CR0 must delegate the exact R1 path. CR1--CR3 compute/cache all six descriptors
for matching extraction exposure, then mask the geometry inputs as declared.
Compute beyond the backbone still differs: record it instead of asserting
equal runtime. Independent RNG streams prevent descriptor computation flags
from changing the episode deck or initialization.

The isolated comparisons are CR2-CR1 and CR3-CR2. CR1 versus TF3 changes the
head, removes L_pair and changes sampling details; it does NOT isolate removal
of the pair loss. No TF3 rerun or separate loss-removal pilot is proposed now.

Future implementation belongs in one method module under src/fmad/methods,
with these three config switches and four method YAMLs, registered through the
existing CLI. Use generic experiments.py run/collect and run_experiments.sh;
do not create a method-specific launcher, result ledger or collection script.

## 7. Cost, outputs and bounded future evaluation

With the planned maxima,64*(2*4096 + 2*128) =540672 sampled rows/class.
Two resolutions times six float32 scalars require24.75MiB; four reference-source
tables of4096*16 float32 entries require1MiB. These are analytic storage counts,
not measured RAM/VRAM. Metadata, weights, temporary chunks and the unchanged
R1 feature banks are additional. Cap the compact descriptor/provenance pool
at128MiB host RAM per class; fail explicitly if exceeded, never silently alter
the sampling protocol. Release the pool after fitting.

Planned encoder calls/class remain32 for CR0 and544 for CR1--CR3; query uses
two calls. Source-distance reductions, reference geometry preprocessing and
fresh head rescoring add compute. No wall-clock equivalence claim is justified.
During prediction retain small scalar feature grids and one required output
map; evaluate native head inputs in4096-pixel chunks, not six complete native
descriptor maps. Do not accumulate maps across images. Save compact metrics,
config/provenance and training logs; dense maps and fitted weights remain off.

When execution is separately requested: four stages times eight classes equals
32 class runs, one seed, same seen TESTpublic development split. No automatic
multi-seed, sweep, checkpoint selection or follow-up run. Record fit/query time,
peak memory, finite outputs, support identities and complete class coverage.
Before launch require exact CR0/R1 parity, source-exclusion checks and matching
fit/predict descriptor semantics. No such runtime checks have been run here.

Decision metrics: full8 equal-class macro AUPRO@.05 and pAUROC@.05; oracle F1
diagnostic. Joint success requires AUPRO no lower than fresh CR0 AND pAUROC
higher than fresh CR0. Also inspect per-class changes so one class cannot be
presented as broad improvement. A single seen seed is development evidence,
not statistical significance or untouched-final validation. Update only the
existing cumulative Markdown/CSV when actual results exist.

## 8. Falsifiable expectations and failure boundaries

| Contrast/observation | Expected if useful | If absent or reversed |
|---|---|---|
| CR2 vs CR1 | Better real pAUROC with preserved AUPRO; global ordering objective supplies useful training signal | Cross-image ranking alone is insufficient under this score head and synthetic distribution |
| CR3 vs CR2 | Added geometry yields incremental joint metric benefit | These descriptors do not justify their cost under the fixed training setup |
| Fit ordering improves, real metrics do not | Not a success condition | Synthetic/shift transfer or fit/inference bank mismatch remains plausible |
| Lower average training loss only | No necessary metric prediction | Do not promote the method from training loss alone |

Optional fixed-pool before/after ranking summaries use the same cached training
descriptors only. They are fit diagnostics, not independent validation and do
not require another encoder pass. Report them alongside, not instead of,
the32-run ablation metrics. No separate mechanism pilot is needed for the first
method comparison.

Important counterexamples: a legitimate normal mode may occur in only one
support image; an anomaly may resemble common normal patches; all six features
may coincide despite different labels; residual amplitude may be insufficient.
The15/16-source bank mismatch remains. Photometric shifts may not preserve
normality after clipping, and the two synthetic families do not cover real
defects. Pixel-mass ranking may favor large synthetic defects while AUPRO weights
regions differently. The head has no explicit spatial context. Frozen features
cannot reveal information discarded by the backbone. Hard/easy synthetic
separation can still yield zero hinge gradients. Do not respond to a failed
suite automatically with more losses, a larger head or longer training.

## 9. Literature positioning and novelty limits

- Narasimhan and Agarwal, ICML2013,
  [A Structural SVM Based Approach for Optimizing Partial AUC](https://proceedings.mlr.press/v28/narasimhan13.html):
  establishes restricted-FPR ranking as an optimization target. Our sampled
  weighted-tail neural hinge is a related design, not their exact optimizer,
  theorem or a guarantee of real-data pAUROC improvement.
- Roth et al., CVPR2022,
  [Towards Total Recall in Industrial Anomaly Detection](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html):
  normal patch-feature memory is established prior art. Memory matching itself
  is not a contribution of CRR.

Local-density methods are another overlap risk; a full formula-level audit has
not been completed. This bounded check does not establish novelty. The current
proposal is a testable project continuation: cross-image tail training of an
R1-preserving residual using source-separated reference geometry. Research value
depends first on the ablation results; a paper claim would additionally require
closer prior-art comparison, broader validation and non-post-hoc evidence.
