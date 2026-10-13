# SIAI-S1 J0 oracle Stage -1 preregistration

Date: 2026-08-27  
Protocol: `siai-s1-j0-true-action-oracle-20260827`  
Status: `PREREGISTERED / RESULT-FREE`  
Claim tier: support-only physical-action feasibility diagnostic; not benchmark-comparable

## Decision and sole change

CRFR-S1 is closed because its nuisance code changed held reconstruction when
zeroed but did not carry sample-specific information: cyclically shuffling the
held code changed cross-recomposition error by at most approximately `7e-7`,
and no cell achieved the preregistered causal-factor gain.  SIAI-S1 removes the
free nuisance code, content head, decoder, LoRA, family router, retrieval
module, and score-side correction.  Its proposed mechanism is an explicit
simulator-coordinate action followed by a fixed analytic inverse in RGB space,
then the unchanged frozen DINOv2 readout and exact RN scorer.

This first assay makes only one new scientific observation: whether the known
signed J0 exposure parameter is sufficient to undo the held shift while
preserving a held synthetic anomaly.  It uses no learned estimator.  J1, J2,
other objects, estimator training, public/test evaluation, and hyperparameter
tuning are not authorized by this protocol.

## Data and custody

- Object: `can` only.
- One immutable 16-normal support manifest; support images come only from
  `train/good`.
- Evaluation: exactly four support-disjoint `can/train/good` bases.
- Shift: immutable `J0_exposure_ev01333` only.  The signed exposure is the
  existing deterministic value `a in {-0.1333,+0.1333}` selected by the
  existing held-base seed.
- Anomaly: existing evaluator-only held `G2_structural_scratch`, with the same
  anomaly instance shifted by the same signed exposure as its paired base.
- No anomaly image, mask, generator identity, or held base enters fitting,
  selection, threshold setting, or memory construction.
- TESTpublic, TESTprivate, every test split, ground truth, query adaptation,
  query memory updates, score calibration, and benchmark claims are forbidden.
- Only compact JSON/log/hash evidence may persist.  Images, dense features,
  score maps, anomaly maps, and checkpoints may not persist.
- Support manifest SHA-256:
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.

## Frozen action and arms

The existing forward action is

`T_a(x) = uint8(round(clip(x * 2**a, 0, 255)))`.

The sole candidate diagnostic is the fixed approximate inverse

`K_a(y) = uint8(round(clip(y * 2**(-a), 0, 255)))`.

Because the forward action quantizes and may saturate pixels, exact pixel
recovery is neither assumed nor required.  The run records paired pixel MAE,
maximum error, and forward saturation rate as diagnostics.

Arms are:

- `B0_SHIFTED_FROZEN_RN`: shifted image, frozen DINOv2, exact RN.
- `N4_STATIC_MEAN`: the existing family-specific feature mean correction from
  the 16 clean support images; privileged diagnostic hard null only.
- `N1_WRONG_SIGN`: apply the same fixed inverse with `-a` instead of the true
  `a`.  For this binary signed intervention it is the preregistered
  action-mismatch control; it cannot be promoted.
- `O_TRUE_ACTION`: apply `K_a` using the true held signed exposure before the
  frozen DINOv2 readout; sole feasibility candidate, never deployable evidence.
- `O_PAIRED_CLEAN`: replace the shifted representation with its paired clean
  representation; ceiling diagnostic only.

All arms use the identical clean support memory and unchanged exact 1-NN RN
score.  No score is fused, clipped, gated, calibrated, or bypassed.

## Required metrics

For each of the four held bases, the runner constructs
`x, T_a(x), K_a(T_a(x)), A(x), T_a(A(x)), K_a(T_a(A(x)))` and extracts the
existing channel-standardized, row-L2-normalized frozen DINOv2 ViT-L/14
feature at block 11 on the 16x16 grid.

The single J0 cell records held-base medians for:

- correction cosine and norm ratio against `F(x)-F(T_a(x))`;
- anomaly residual cosine and norm ratio, comparing
  `F(K_a(T_a(A(x))))-F(K_a(T_a(x)))` with `F(A(x))-F(x)`;
- exact-RN shifted-normal q99 and reduction versus B0;
- RN anomaly rank correlation, nearest-neighbor retention, and scale ratio
  against the paired-clean anomaly scores and neighbor indices;
- `O_TRUE_ACTION` correction-cosine margin over `N4_STATIC_MEAN`;
- `O_TRUE_ACTION` correction-cosine margin over the stronger of
  `N4_STATIC_MEAN` and `N1_WRONG_SIGN`, plus its pixel-MAE gain over wrong sign;
- pixel reconstruction and saturation diagnostics;
- support/held path-set hashes, source/snapshot hashes, finite/nonconstant
  checks, backbone frozen state, and forbidden-access ledger.

The primary run is repeated in full with the identical inputs and seed.
Every numeric value in the scientific signature must agree within `1e-7`, and
the canonical scientific-signature SHA-256 must match.  A mismatch is an
operational failure and has no scientific interpretation.

## Frozen gates and stop rule

`J0_TRUE_ACTION_FEASIBLE` requires all of:

- correction cosine `>=0.90` and correction norm ratio in `[0.80,1.25]`;
- anomaly cosine `>=0.90` and anomaly norm ratio in `[0.80,1.25]`;
- RN rank correlation and neighbor retention each `>=0.90`;
- RN scale ratio in `[0.80,1.25]`;
- normal q99 degradation versus B0 no worse than `0.005`;
- correction cosine exceeds both the static-mean and wrong-sign controls by
  at least `+0.01`, and paired-normal pixel MAE is at least 10% lower than the
  wrong-sign inverse;
- finite/nonconstant exact-RN scores, exact counts, custody, frozen-backbone,
  source/hash, output-schema, and forbidden-access checks all pass.

Forward saturation and maximum pixel error remain diagnostic because the
uint8 transform is not an exact group action.  The hard-null and action-use
margins above are fixed gates and cannot be weakened after observing results.

If the gate fails, close SIAI-S1 and do not implement J2 or a coefficient
estimator.  If it passes, it licenses only a separately preregistered
`can x J2` true-action sentinel.  It does not license a four-object panel,
estimator fitting, LoRA, or public/test evaluation.

## Frozen source identity

Result-bearing launch is forbidden until the result-free implementation,
tests, static review, and remote controller are complete.  Immediately before
launch, record SHA-256 values for the runner, analytic-action core, aggregator,
frozen readout bundle, source manifest, and remote controller here and in a
machine-readable `source_hashes.json`.  Remote preflight must recompute and
require exact equality from the immutable snapshot.

Frozen on 2026-08-27 before any result-bearing J0 launch:

- runner: `190c20ceb4e46b4cd9bc2aae09c2d3ab37cf8fb1b8653151262cf425b7579283`
- analytic-action core: `27e39ce7f5c6de1c27d34a958dadca23bb19ffff64e3feb36264760a71772dc2`
- aggregator: `d765bd55a7baeb6d46e5ecbd3dd7f51ee05bbe0f43fcf7e0f3ef7c488b9533f8`
- frozen readout bundle: `328aa50d8a61566630aa788559ff1ac96b16f61dc145e7ed619f44724f99385e`
- normalized remote controller: `69aaaeeef87ca0a698df3599db5a26ccbce9d49c190f46e384312969768ff84d`
- source manifest: `47810d45c3db6c03e6f73a66785de91e0a0d53f897f288fdb4a61d8a5c46d4c1`
- complete remote-controller file: `a6dfec74e912aea454a753e1f26fa7c8f5f3f05b20270167a93c4d604e63b793`
- frozen configuration: `a15f2d19445df43d709d6f7ef5bd0793d85155a60cff3287ada13306336be8ba`

The normalized controller digest replaces only the
`SOURCE_HASHES_SHA256=...` assignment with a fixed marker before hashing, which
avoids a manifest/controller self-reference while still binding all executable
controller content.  The immutable-snapshot preflight verifies both the exact
source-manifest digest and all five entries, including that normalized
controller digest.

## Remote execution boundary

All result-bearing tests, DINO extraction, and evaluation run only inside the
project-owned dsba3 container `hun_fsad_tta_012` on GPU 0.  Preflight may query
only selected GPU 0's index, UUID, memory, utilization, and compute UUID, and
must abort if it is occupied.  It must not enumerate or inspect any other
container, process, user, owner, file, volume, GPU, or job.  Cleanup may remove
only the exact immutable SIAI snapshot named by this run; compact results are
retained.
