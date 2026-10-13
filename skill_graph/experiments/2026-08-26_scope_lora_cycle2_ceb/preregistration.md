# SCOPE-LoRA progressive Cycle 2 preregistration

Date: 2026-08-26  
Protocol: `scope-lora-cycle2-calibrated-excess-bypass-20260826`  
Status: `PREREGISTERED / NO CYCLE-2 RESULTS INSPECTED`  
Seed: exactly one, `20260826`

## Sole question

Cycle 1 showed a positive rank-4 normal-correction signal but failed its raw
anomaly-residual commutation gate.  Because the same paired correction was
added to both terms of that residual, the correction canceled exactly and the
failed metric measured the frozen backbone's shift/anomaly interaction rather
than score harm caused by the projector.

Cycle 2 changes one thing only: it adds a calibration-gated frozen-score
evidence bypass and evaluates held synthetic masks at score level.  The rank,
projector fit, paired projected teacher, data custody, memory, feature
normalization, shifts, held generator, objects, and seed remain fixed.  No
LoRA optimization runs in Cycle 2.

## Frozen custody and split

- Dataset: `/home/hunim/Volume/DATA/mvtec_ad_2`.
- Objects: `can`, `fabric`, `vial`, `walnuts`.
- Exact support-16 manifest SHA-256:
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- `S_fit`: first 12 frozen support paths, used to fit the rank-4 projector and
  shift-family projected static mean.
- `S_cal`: remaining four frozen support paths.  Only deterministic
  shifted-normal descendants are used to calibrate the bypass.
- Evaluation: the same four support-disjoint target `train/good` bases and
  held generator `G2_structural_scratch` used in Cycle 1.
- Shifts: the frozen J0 exposure, J1 Gaussian RMS control, and J2 spatial
  illumination families, with identical seed derivation and composition order.
- `TESTpublic`, private test, GT, query-memory update, benchmark thresholds,
  and target held masks outside the deterministic G2 instrument remain sealed.

## Scores and the sole repair

All patch scores are exact Euclidean 1NN distances to the unchanged support-16
normal memory.  For a held shifted query, let `S_B0` be the frozen score and
`S_PT` the Cycle-1 paired projected-teacher score.  Define the positive score
gap

`D = max(S_B0 - S_PT, 0)`.

For each object and shift family, each of the four `S_cal` shifted-normal
descendants yields an image-level linear-interpolation q99 of `D`.  The frozen
threshold is

`tau_f = max(q99_image_1, ..., q99_image_4)`.

The calibrated excess bypass is

`S_CEB = S_PT + max(D - tau_f, 0)`.

Thus `S_CEB >= S_PT` and `S_CEB <= max(S_B0,S_PT)` pixelwise.  Evidence that
the projector removes within the calibrated normal gap stays removed; only a
larger frozen-over-corrected excess is restored.  If all calibration gaps are
zero, `tau_f=0` and strict `D>tau_f` activation is required.  No threshold is
re-estimated on held rows.

## Fixed arms and nulls

- `U0_UNSHIFTED`: unshifted clean/synthetic score, descriptive ceiling.
- `B0_FROZEN_SHIFT`: shifted frozen score.
- `PT_P`: paired rank-4 projected teacher without bypass.
- `PT_CEB`: the sole Cycle-2 candidate.
- `N0_NO_BYPASS`: `tau=+inf`, required to reproduce `PT_P`.
- `N1_FAMILY_SHUFFLE`: deterministic cyclic reassignment of the three frozen
  `tau_f` values without recomputing them.
- `N2_SPATIAL_SHUFFLE`: 64 deterministic within-image permutations of `D`,
  preserving its marginal values while breaking spatial alignment.
- `N3MU_P_CEB`: the Cycle-1 shift-family projected static mean with its own
  `S_cal` threshold; descriptive hard null for later P1.
- `N4_MAX_STREAM`: `max(S_B0,S_PT)`, an uncalibrated evidence-restoration null.

The same score maps, G2 masks, aggregation, and numerical precision are used
for every arm.  G2 masks are evaluator-only and never enter `tau`, projector,
memory, or correction construction.

## Metrics

Per cell and arm, record:

- held shifted-normal q95/q99 and reduction versus `B0_FROZEN_SHIFT`;
- bypass activation on `S_cal`, held normal, G2-inside, and G2-outside tokens;
- tie-aware patch AUROC and raw normalized pAUROC over `FPR <= 0.05` on held
  G2 masks;
- paired inside anomaly score lift over the corresponding shifted clean token;
- pAUROC margin versus `PT_P`, frozen, unshifted, family-shuffle, and the 95th
  percentile of the 64 spatial-shuffle margins;
- exact no-op, projection/complement, CEB lower/upper-bound, source-map hash,
  finite/nonconstant, and calibration/held-disjointness checks.

## Preregistered gate

Aggregation order is cell, then three-family object median, then four-object
median.  `PROMISING_PROTECTED_SCORE_SIGNAL` requires every condition below.

1. Structural/custody: all maps are finite and nonconstant; exact no-op,
   projection/complement, CEB identity/bounds, source-map immutability, and
   metric replay errors are at most `1e-7`; calibration and held rows are
   disjoint and every forbidden-access flag is false.
2. Normal utility: the global `PT_CEB` q99 reduction is at least half the
   immutable Cycle-1 PT-P value, namely `0.0132773854`; every object is at least
   `-0.01`; and every object's held-normal bypass activation is at most `0.05`.
3. Synthetic protection: for every object and for the global median,
   `PT_CEB` pAUROC is no more than `0.01` below the best of
   `B0_FROZEN_SHIFT`, `PT_P`, `N3MU_P_CEB`, and `N4_MAX_STREAM`.
4. Spatial specificity: the global `PT_CEB - PT_P` pAUROC margin is strictly
   above the global 95th percentile from the 64 deterministic spatial-shuffle
   nulls.
5. Query-specific value over the static hard null: `PT_CEB` pAUROC is at least
   `N3MU_P_CEB - 0.01`, its q99 reduction is at least
   `N3MU_P_CEB - 0.005`, and at least one strict margin is present:
   pAUROC `>=N3+0.01` or q99 reduction `>=N3+0.01`.

This gate licenses only the already designed P1 oracle-coded support-carrier
LoRA with the CEB scoring firewall.  It does not authorize a benchmark/public
branch.  If structural checks and synthetic protection pass but exactly one
of normal utility, spatial specificity, or static-null value narrowly fails,
the verdict is `NEAR_MISS_ONE_REPAIR`; Cycle 3 may change that one component
and remains support-only.  Any broader failure is
`INSUFFICIENT_PROTECTED_SCORE_SIGNAL`, keeps P1 closed, and ends this
three-cycle outer loop unless the failure is operational and result-free.

Pre-result clarification (recorded before any Cycle-2 scientific cell was
run): “narrowly” is fail-closed and quantitative. The sole failed scientific
gate must satisfy its relaxed boundary while the other two scientific gates,
structural/custody, and synthetic protection pass. Normal utility requires
global q99 reduction `>=0.0082773854`, every object `>=-0.015`, and every
object held-normal activation `<=0.06`. Spatial specificity requires the
candidate margin to be no more than `0.005` below the global spatial-null q95.
Static-null value requires pAUROC `>=N3-0.015`, q99 reduction `>=N3-0.010`,
and a reduced strict margin of at least `0.005` in either metric. These bounds
only classify a one-repair Cycle-3 license; they do not relax any Cycle-2 pass
gate.

## Execution and artifacts

Run the fixed four-object panel in `hun_fsad_tta_012` on dsba3 GPUs 0--3 from
one immutable snapshot.  Preflight must verify exact host/container GPU UUID
mapping, offline DINO cache, support hash, CUDA model path, and focused tests.
Only compact JSON summaries are retained; dense feature/score/anomaly maps are
forbidden.  Failed launch attempts cannot be promoted to results.
