# FLRS Stage -1 Support-Only Identifiability Report

Date: 2026-07-18
Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`

## Negative Evidence Intake

See `preregistration.md`. FLRS is a genuinely non-kNN same-layer relation
probe; it does not reopen projection, routing, density, hard-local matching,
cross-layer prediction, TTA, or score fusion.

## Motivation

Test whether local spatial relations create anomaly-aligned evidence that is
less shift-sensitive than absolute feature state.

## Implementable Design

Frozen in `preregistration.md` and Stage 5 of the deep-research artifact.

## Evaluation Alignment

Support-only; no TESTpublic/private metric claim is available. Exact 16-path
parity and all-eight object coverage are mandatory.

## Code Modification / Creation

Created `src/flow_tte/flrs.py`, the compact support runner, deterministic
all-eight aggregator, dsba3 controller, and focused regression tests.  The
first remote attempt exposed a CPU-only relation bottleneck; the final runner
uses equation-equivalent batched Torch kernels.  A second attempt exposed a
sealed no-op synthetic stimulus and was replaced by a metric-blind,
acquisition-visible deterministic retry rule.  Neither aborted attempt
produced a method verdict.

## Added Code Evaluation

`ruff`, Python compilation, and 13 focused tests passed.  Tests include
Torch-versus-NumPy signature, median/MAD, and score equivalence.  All eight
final object summaries have exact 16-support, 4x4 fold, shift-pair, finite,
nonconstant, access, and mask-alignment parity.

## Remote Execution

Completed as `flrs_stage_minus1_20260718_c1_v3` in fixed container
`hun_fsad_tta_012`, with GPU 0=`can,fruit_jelly`, GPU 1=`fabric,sheet_metal`,
GPU 2=`rice,wallplugs`, and GPU 3=`vial,walnuts`.  The final aggregation used
a hash-recorded serialization-only hotfix after the original aggregator tried
to JSON-encode a `numpy.bool_`; the eight object summaries were not rerun or
changed.

## SuperAD Baseline and Unified Metrics

SuperAD/RN-FMLK/H+ public baselines are intentionally deferred unless Stage -1
passes. Current result label cannot exceed `PROMISING_DIAGNOSTIC`.

## Results and Analysis

FLRS failed every performance gate while passing both validity gates:

- raw-state versus FLRS macro positive shift p95 inflation was
  `0.052037 -> 6.691346`; shift reduction was `-127.588035` rather than the
  required `+0.50`;
- clean defect pAUROC/F1 changed by `-0.030224/-0.054076` relative to raw;
- shifted defect pAUROC/F1 changed by `-0.029233/-0.052211`;
- the hard-null margins were negative (`-0.032626/-0.072230` clean and
  `-0.031560/-0.063687` shifted), with zero objects beating all nulls;
- bootstrap lower bounds were `-225.717305` for shift reduction,
  `-0.058786` for pAUROC null margin, and `-0.144046` for F1 null margin;
- G1--G4 failed; G5 coverage and G6 support/access audit passed.

The failure is structural.  Local cosine statistics have extremely narrow
clean-support MADs, so small acquisition changes become large robust z scores.
They also discard absolute channel evidence that raw state uses to detect
chromatic blotches and scratches.  Radius-1-only was worse still, while the
random-neighbor and spatial-permutation nulls remained near chance.  The one
complementary result—RGB gradients on Gaussian blur—came with catastrophic
shift inflation and cannot support a generic relation score.

## Continuation Assessment

Exact FLRS, radius-only relation energy, and simple relation-statistic
calibration are closed.  Public evaluation is not authorized.  The next
independent seed is a support-only factorial shift-versus-defect discriminator
with held-out defect-family evaluation; it must learn a boundary from crossed
shift/defect stimuli rather than assume that locality is invariant.

## Conclusion

FLRS does not improve the default and materially worsens both distribution
shift robustness and anomaly localization.  This negative result narrows the
remaining structural hypothesis: nuisance and defect evidence must be learned
jointly as a discrimination problem, not removed or normalized by a fixed
local relation statistic.

## Post-Conclusion Storage Cleanup

Pulled compact JSON/TSV/log/hash artifacts to
`results/remote_runs/dsba3/flrs_stage_minus1_20260718_c1_v3`.  The completed
v3 isolated snapshot was removed; aborted v1/v2 snapshots and result roots
were removed after process termination.  No dense feature, transformed image,
score-map, or anomaly-map directory was retained.
