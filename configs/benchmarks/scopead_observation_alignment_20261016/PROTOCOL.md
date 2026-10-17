# Support-only cross-observation alignment, public development comparison

Hypothesis (post-hoc, 2026-10-16): after Q95 division, normal scores from
resolution × bag × layer observations retain different distribution shapes.
Matching their normal-reference quantiles before the unchanged fusion may
improve detection. This is an exploratory test on a previously seen population,
not an untouched final evaluation or a guarantee of better calibration.

## Locked comparison

- MVTec AD 2 test_public, all eight classes, seed 0, fixed VOID 2/16-shot normal
  support manifests. Four arms × two shot counts × eight classes = 64 cells.
- Frozen DINOv2 ViT-L/14, layers 5/11/23, resolutions 448/672/896/1120,
  four bags, 256 tokens/source, cosine 1NN, deep mean + 0.125 shallow,
  CQ off, existing noise guard and evaluator. No backbone or bank changes.
- Comparator Q95; candidates Gaussian survival, three-anchor transport, and
  nine-anchor transport. Primary AUPRO@0.05, secondary pixel pAUROC@0.05,
  higher is better. Native-grid normal score distribution is a separate
  mechanism endpoint. Report all class results and fallbacks, including losses.
- All fitted inputs are normal-support source-LOO distances. Each transport
  first divides by its own observation Q95. It computes its source quantiles
  and the arithmetic mean of matching quantiles across all 48 observations
  in that class (4 resolutions × 4 bags × 3 layers). A monotone piecewise
  linear map sends source to mean anchors. Above Q99, extrapolate using the
  last slope capped at 10; below the first knot use the first slope. A source
  with duplicate knots falls back to Q95. Three-anchor grid: .50/.95/.99;
  nine-anchor grid: .01/.05/.10/.25/.50/.75/.90/.95/.99. These constants
  are fixed before outcomes. Gaussian survival maps the existing empirical
  body and exponential Q95 tail to a standard-normal upper-tail score.
- Source-LOO references and query full-bank distances have different bank
  sizes; support tokens are dependent. This design aligns fitted reference
  quantiles exactly where slopes do not hit guards. It does not guarantee
  native normal-query alignment or superior anomaly ranking.

Run through `scripts/experiments.py run` via `lane.sh` on four existing dsba3
GPU UUIDs, existing container `hun_fsad_tta_012`, immutable stage-only source
release. Retain remote logs, pull compact artifacts, and collect with the
existing collector. Timeout 10800 s per class, no adaptive expansion.
Before paired differences, verify support/query/GT/bank hashes. Compare both
performance and native-grid normal-query distributions. No query, masks or
labels enter fitting; masks are used only for evaluation/diagnostics. The
calibration audit's old normalized>1 counter is not a common threshold for
nonlinear arms; use each arm's fitted transformed Q95.
