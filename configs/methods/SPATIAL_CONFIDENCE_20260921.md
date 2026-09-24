# Spatial agreement of the QR1 correction — sequential development trial

This is the second candidate stage authorized by the user's sequential-experiment
request, after collecting the trace-matched geometry control. Do not start its
evaluation before that collection. This document locks one rule, not a sweep.

## Evidence and hypothesis

Completed PG_B (QR1) versus PG_A (original features) improves shift_1 macro
AUPRO by 0.95975/0.55996 pp at 1/16 shot, but loses in two of eight classes at
each shot. This motivates testing whether the magnitude of its global correction
should depend on spatial consistency. Existing aggregate translation diagnostics
do NOT establish that consistency predicts useful corrections. This trial tests
that hypothesis rather than claiming an already validated confidence measure.

Repository sanity check: ReferenceConsensus weights individual residuals by
agreement across support banks; ValidatedReference validates additional
mode-specific corrections using spatial witness blocks, leaving global QR1
unchanged. Here only the magnitude of the global QR1 translation is attenuated.
No novelty claim or calibrated probability-of-normality claim is made.

## Locked rule

Use original R1 nearest-neighbour residuals, separately per layer/resolution.
Compute the ordinary bounded QR1 translation delta (ridge 1, norm cap 0.1).
Split the patch grid into top/bottom and left/right regions. Exclude one patch
on each side of each midpoint; no alternate checkerboard or split search.
For each axis, let a and b be the two residual means BEFORE norm clipping.
Define c_axis = clamp(2<a,b> / (||a||^2 + ||b||^2), 0, 1), with denominator
floor 1e-12. Fewer than 32 tokens in either region gives c_axis=0.
The single confidence c is min(c_vertical, c_horizontal); apply c*delta.
Zero confidence returns the original bank exactly. Otherwise use the existing
unit-normalized shifted bank, cosine NN scoring, fixed support MAD and mean
fusion. Confidence off delegates exactly to QR1 for parity tests.

This compares direction AND magnitude consistency, not only cosine direction.
It is threshold-free apart from existing token sufficiency/numerical guards.
No classifier, new trainable parameter, support transform or online persistent
adaptation. Fit only on the same normal supports; query correction sees only
the current unlabelled query. Global attention means regions are not statistically
independent. Repeated defects and systematic reference mismatch can agree;
legitimate spatially varying nuisance can disagree. No defect-preservation proof.

Record compact per-query confidence/norm diagnostics, including warmup and
evaluator calibration calls; distinguish those from public queries in analysis.
These are scalar records, not cached features/maps. Confirm finite output,
bounded translations, unchanged banks/scales/backbone and no cross-query state.

## Comparison, budget, and stop

16 evaluations: SC1 at full8 x nested 1/16 shot, seed0, same SD_support manifests,
frozen DINOv2-L layers11/23, resolutions448/672, original preprocessing/pipeline.
Reuse completed PG_A/PG_B metrics only after checking support/query/GT hashes,
backbone hashes, evaluator contract and exact geometry-off QR0/QR1 parity.
Reuse the per-class local/dsba3 GPU/runtime assignment of PG and trace matching.
Actual split: test_public_seen_development, repeatedly seen, NOT sealed final.
Evaluator-only normal calibration images stay evaluator-only. No maps/weights.

At BOTH shots require SC1 minus PG_B shift_1 macro AUPRO@0.05 >= +0.5 pp,
positive in >=5/8 classes and positive after removing the largest-gain class.
Require regular AUPRO no worse than either PG_A or PG_B by >0.5 pp, and shift_1
pAUROC no worse than PG_B by >0.5 pp. Check confidence is not merely a constant
identity/QR1 fallback. Diagnostics are explanatory, not tuning targets.
If a gate fails, stop this fixed rule; do not adjust splits, weights, thresholds,
shots or seeds on these outcomes. Passing prompts a separate replication
decision, not an automatic sweep or a paper-ready claim.

## Pre-evaluation execution checks

Local and dsba3 targeted suites: 55 passed, 1 opt-in backbone test skipped.
Separate cached-DINOv2-L normal-only smoke used the fixed can support1 and
`can/train/good/253_regular.png` (another support16 image, not a holdout claim).
All four layer/resolution outputs were finite; confidence was
0.19528, 0.49455, 0.17455, 0.59478, and applied norms were bounded below QR1.
This checks nondegenerate execution only, not correlation with useful correction.
No rule/threshold was changed after seeing the smoke. Maps were not saved.
Execution snapshot on dsba3:
`6914f8ae3233c3e943a7e3c60161e9069979d843c9ff14c0238ce9e00ab5e363`.
