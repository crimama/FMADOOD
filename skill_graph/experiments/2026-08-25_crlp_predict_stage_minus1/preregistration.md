# CRLP-Predict Stage -1 preregistration

Date frozen: 2026-08-25, before any CRLP-Predict result was observed.  This is
a source-only mechanism diagnostic, not a benchmark comparison or method
evaluation.

## Identity and custody

- Method: `CRLP-Predict-S1`.
- Protocol: `crlp-predict-stage-minus1-support-only-20260825`.
- Object schema: `crlp-predict-stage-minus1-object-v1`.
- Aggregate schema: `crlp-predict-stage-minus1-aggregate-v1`.
- Dataset interface: MVTec AD2 `train/good` only, plus deterministic
  descendants of those images.
- Frozen support: exact SuperAD16 manifest
  `skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/superad16_dinov2_reference_paths.json`,
  SHA-256 `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- `TESTpublic`, private test, `ground_truth`, query-time memory updates, and
  persistent dense maps are forbidden.  `benchmark_comparable=false` and
  SuperAD metrics are `N/A`.

Each outer shard holds out one complete AD2 object.  The fit set is one
deterministically hash-ranked `train/good` base from each of the other seven
objects.  The held-object evaluation set is four deterministically hash-ranked
`train/good` bases disjoint from SuperAD16.  All descendants of those bases are
sealed from fitting.  SuperAD16 is used only as an immutable exact-RN memory.

## Frozen representation and interventions

- DINOv2 ViT-L/14, layer 11, 672-pixel smaller edge.
- Realized patch tokens are adaptive-average-pooled to `16 x 16`, tokenwise
  population z-scored, then row-L2 normalized.
- Shift families:
  `J0_exposure_ev01333`, `J1_gaussian_rms_exposure_control`, and
  `J2_spatial_illumination_cv015`.
- Image anomaly families:
  `G0_poisson_patch`, `G1_material_blotch`, `G2_structural_scratch`, and
  `G4_frequency_local`.  Feature-ascent is excluded because image-space
  composition order is undefined.
- `T(A(x))` uses the identical exposure sign, Gaussian field/calibrated sigma,
  or spatial field as `T(x)` for the same base.

## Gate A — confound-matched direction decodability

For every held shift x anomaly cell, fit a rank-64 signed-projection ridge
probe using only the other two shift families, other three anomaly families,
and seven fit objects.  Evaluate only the held shift, held anomaly, and held
object.  Shift and anomaly residuals use the same base, anomaly mask, realized
patch grid, and patch coordinates.  At each paired coordinate both raw
residuals are scaled to their geometric-mean norm before row normalization;
the maximum paired norm mismatch must be `<=1e-6`.

Controls are deterministic family-label shuffle, cyclic source-base
reassignment at the same realized patch coordinates, matched isotropic random
directions, magnitude-only, shared base/support/spectrum-only, and
residual-spectrum-only probes.  Generator identity is
measured by leave-one-source-object-out four-way probing.  Candidate and
matched-null probes share the identical frozen signed input projection.

A cell passes only when direction balanced accuracy is `>=0.70`, AUROC is
`>=0.75`, and balanced-accuracy margin over every hard null is `>=0.05`.
Magnitude/base-support-spectrum/residual-spectrum probe accuracy must each be
`<=0.60`; object-level generator-ID balanced accuracy must be `<=0.35`
(chance `0.25`).

## Gate B — independent predictive exact-RN actionability

For each held shift, fit a source-only map from shifted frozen features to the
paired clean shift residual.  The input is a frozen rank-64 signed projection;
the output is a rank-16 source residual basis; centered relative ridge is
`0.01`.  The map sees no synthetic anomaly.  It is applied unchanged to the
held `T(x)` and `T(A(x))` cells.

Controls are no-op exact RN, residual-target shuffle, cyclic base-pair shuffle,
magnitude-only prediction, same-rank random output basis, and static shift
subspace projection.  Subtracting the paired clean shift residual is reported
only as an oracle upper bound.  All learned candidate/null maps share the
identical frozen signed input projection.

A cell passes only when:

- exact-RN shifted-normal q99 decreases by at least `5%`;
- anomaly residual mean cosine is `>=0.90` and median norm ratio is in
  `[0.80, 1.25]`;
- both corrected-normal-versus-clean and corrected-anomaly-versus-unshifted-
  anomaly RN maps have Spearman `>=0.90`, neighbor retention `>=0.80`, and
  distance-scale ratio in `[0.50, 2.00]`;
- the candidate's guarded normal-tail gain exceeds every non-oracle control by
  at least `0.01`.

Gate-A hard-null margin must positively predict Gate-B preservation-guarded
normal-tail gain across the 12 held cells with Spearman `>=0.30`.

## Sequential execution and stop rule

1. GPU0 runs `can` as a validity sentinel.  Only custody, finite/nonconstant
   maps, exact matching, deterministic duplicate RN, complete 12-cell output,
   and finite predictors authorize expansion; scientific success is not used
   to select expansion.
2. If valid, all eight held-object shards run support-only on dsba3 GPUs 0--3,
   two sequential objects per GPU, from one hashed repository snapshot.
3. Any invalid shard, any Gate-A or Gate-B held-cell failure, generator/
   confound leakage, topology violation, hard-null tie, or association below
   `0.30` yields `KILL_FOR_CLAIM / NO_CONTINUE`.
4. A complete pass yields only
   `CONTINUE_DIAGNOSTIC / NEW_LEARNER_PREREG_REQUIRED`; it does not authorize
   public/test evaluation or a deployment claim.

No hyperparameter sweep, nonlinear rescue, PCRE, CRCG, or automatic public
expansion is permitted after observing this experiment.
