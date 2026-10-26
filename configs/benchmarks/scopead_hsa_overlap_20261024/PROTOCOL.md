# Direct High-Score Ambiguity analysis

Date: 2026-10-24

## Purpose and status

This post-hoc diagnostic tests whether high-scoring normal-background pixels
and defect pixels that receive nearly equal baseline scores become more
separable under multi-resolution aggregation. The data are MVTec AD 2
test_public labels already inspected during development. Results therefore
explain behavior on this development population; they are not an untouched-test
estimate or evidence of generalization to unseen shifts.

## Fixed source population

- Eight MVTec AD 2 classes: can, fabric, fruit_jelly, rice, sheet_metal, vial,
  wallplugs, and walnuts.
- Support settings: 2-shot and 16-shot; support seeds 0, 1, and 2.
- Use the existing 2026-10-20 Q95 mechanism caches. No feature extraction,
  memory-bank fitting, or nearest-neighbor computation is repeated.
- The cached score maps use frozen DINOv2-L/14 features from layers 5, 11, and
  23, resolutions 448/672/896/1120, four bags, Q95 normalization, and fixed
  layer fusion (deep-layer mean plus 0.125 times the shallow-layer score).
- Reconstruct each resolution map by layer fusion, bag averaging, and bilinear
  upsampling, then evaluate the individual resolutions, their arithmetic mean,
  and their pixelwise Rényi rank aggregate. The replay must match the cached
  Q95+Rényi map within 2e-5 maximum absolute error for every source image.
- No CQ or other spatial postprocessing is applied. The diagnostic does not
  invoke any query-time noise-guard action; it replays the cached component
  scores with all four resolutions.

## Pair construction and outcomes

For each bad query image, use its ground-truth mask only after cached score
reconstruction. Transform scores with the natural logarithm after clipping at
1e-8. Using the 1120-only Q95 map as the reference, select the highest-scoring
5% of pixels outside the defect mask. Match these candidate normal pixels to
all defect pixels by maximum-cardinality, order-preserving one-to-one matching,
with an absolute log-score difference no larger than 0.02. A pair is concordant
when the defect pixel has a higher score, discordant when the normal pixel does,
and tied pairs receive 0.5. Every aggregation arm is evaluated on the exact
same matched pixel pairs.

The primary outcome is class-macro paired concordance for the four-resolution
Rényi map versus 1120-only. The mechanism comparison is Rényi versus the
four-resolution arithmetic mean. Single-resolution arms (448, 672, and 896)
are diagnostic comparators. Repeat the complete pairing analysis with log-score
calipers 0.01 and 0.05 as sensitivity checks; 0.02 remains primary regardless
of the results. Report matched-image coverage, pair counts, defect coverage,
log-score margins, and per-class values alongside concordance.

Report regular, illumination (overexposed/underexposed), acquisition-shift
(shift_1 through shift_4), and pooled nonregular conditions. Missing
class-condition combinations remain missing. Average images within each
class/support-seed cell, average the three seeds within class, then macro-average
classes. Compute descriptive 95% bootstrap intervals by resampling the eight
class clusters; do not treat pixels or images as independent inferential units.

## Interpretation limits

Pair selection uses the baseline map and defect labels, so the analysis directly
probes score ordering among baseline-ambiguous locations. It does not estimate
deployment false-positive rates, select a threshold, or prove causal mechanism
or generalization. A positive Rényi-minus-1120 difference supports improved
separation in this selected population; a positive Rényi-minus-mean difference
is additionally needed to attribute a gain to rank aggregation rather than
multi-resolution averaging alone. Sensitivity to the fixed pairing caliper and
class-level consistency must be reported.

## Outputs

Store four per-image shards, the combined hsa_overlap_analysis.json, and
comparison.md under
results/remote_runs/dsba3/scopead_hsa_overlap_20261024/. Preserve remote
execution logs and source-index hashes with the results. Do not register this
post-hoc diagnostic as a standard benchmark macro in the cumulative comparison
ledger.

## 2026-10-24 implementation and matching audit

The original four shards and summary are retained unchanged. A second summary
under audit_v2/ reaggregates the saved per-image statistics; it does not replay
pixel maps or change the matching rule, calipers, images, or comparison arms.
The summary records its script hash and all four input-shard hashes, validates
source image counts, and rejects inconsistent stored concordance differences.

The implementation now assigns every tie exactly 0.5 and retains zero-valued
coverage cells. Future pixel replay uses the same tie convention for strict
rescue/regression transitions. Historical rescue/regression fields use exact
signs and remain explicitly labeled as legacy statistics. Regression checks
cover near-zero positive and negative differences, zero coverage, and strict
ordering reversals. Reaggregation changed none of the original concordance,
paired-difference, or coverage summaries.

The audit reports signed baseline log-score gaps, absolute matching gaps,
per-arm signed gaps, and rescue/regression fractions using the same hierarchy
of image, support seed, and class averaging. Rescue/regression denominators
are all matched pairs within an image, not only the initially incorrect or
correct pairs. There are 705 unique bad query images and 4,230 evaluations
across the two shot settings and three support seeds.

Maximum-cardinality matching neither minimizes score differences nor balances
their signs. A synthetic check using uniform scores on [0, 1] and a 0.02
caliper gives approximately 99.5%, 0.5%, and 50% concordance when normal/defect
candidate counts are 10001/101, 101/10001, and 101/101 respectively. This
demonstrates sensitivity to sampling density; it does not establish how much
of the observed real-data difference is attributable to matching.

For nonregular conditions, the class-macro baseline absolute gap is about
0.0176--0.0177, near the 0.02 caliper. Signed baseline gaps are about +0.013.
Rényi increases the average signed gap but rescues fewer pairs than it reverses
incorrectly. Thus average separation and correct ordering give different
conclusions on this selected population. Preserve both outcomes; do not use
the positive average margin to replace the original primary endpoint.

This audit cannot establish population-wide HSA improvement or deterioration.
A subsequent direct validation must distinguish observation complementarity
from the effectiveness of the chosen fusion rule and use complete-population
low-FPR outcomes alongside any selected-pair diagnostic. Calibration attribution
requires none/Q95 crossed with mean/Rényi under the same pipeline. All such
analyses remain development evidence on the already-seen public split.
