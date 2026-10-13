# TARCD-LoRA-S1 Stage -1 preregistration

Date frozen: 2026-08-26, before any TARCD-LoRA result was observed.

This is a single-seed, support-adapted representation mechanism diagnostic. It
is not source-only, not a TESTpublic benchmark, and not evidence for a
segmentation or deployment claim.

## Identity and claim boundary

- Method: `TARCD-LoRA-S1` (Teacher-Anchored Residual Contrastive Distillation
  with LoRA).
- Protocol: `tarcd-lora-s1-stage-minus1-support-adapted-20260826`.
- Object schema: `tarcd-lora-s1-stage-minus1-object-v1`.
- Run schema: `tarcd-lora-s1-stage-minus1-run-v1`.
- Aggregate schema: `tarcd-lora-s1-stage-minus1-aggregate-v1`.
- Target: MVTec AD2 single-image at
  `/home/hunim/Volume/DATA/mvtec_ad_2`.
- Panel: `can`, `fabric`, `vial`, `walnuts`.
- One and only one experiment seed: `20260826`. There is no seed sweep and no
  post-result seed selection. A positive result cannot become a method claim
  until a separately preregistered multi-seed replication.
- SuperAD baseline source: the exact SuperAD-16 reference manifest
  `skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/superad16_dinov2_reference_paths.json`,
  SHA-256
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
  It supplies the only fitting bases and the frozen exact-RN memory. No
  TESTpublic metric is computed, so `benchmark_comparable=false`,
  `superad_seg_AUROC_0.05=N/A`, and `superad_seg_F1=N/A`.

The candidate is not a CRLP rank/ridge retune. CRLP fitted a representation-
external linear residual estimator on other objects and failed at estimator
identifiability. TARCD-LoRA changes the representation itself through a
target-support-fitted backbone PEFT path while a frozen teacher anchors clean
and anomaly geometry. The likely failure basins are near-identity adaptation,
synthetic generator/mask shortcuts, anomaly absorption, and equivalence to a
shuffled or random same-capacity LoRA.

## Data custody and split

- Fitting reads exactly the 16 manifest `train/good` images for the current
  object and deterministic descendants of those images. No other object is
  used for fitting.
- Evaluation reads exactly four deterministically hash-ranked current-object
  `train/good` images outside the support manifest. These four bases and every
  clean/shift/anomaly/composed descendant are sealed from all optimization,
  early stopping, target construction, and hyperparameter choice.
- The exact-RN memory is the 16 frozen teacher clean support maps.
- `target_support_used_for_fit=true` and
  `target_synthetic_used_for_fit=true` are required disclosures, not forbidden
  flags. `synthetic_generator_label_used=true`,
  `synthetic_mask_used_for_generation=true`, and
  `synthetic_mask_used_in_loss=true` are also required disclosures.
- `test_public`, `test_private`, `ground_truth`, query images, query-time
  updates, and public authorization are forbidden. The runner contains no
  public/test/GT discovery path and must emit an empty
  `access_ledger.forbidden_kinds_opened` list.

## Frozen representation and PEFT surface

- Frozen teacher/base: DINOv2 ViT-L/14.
- Canvas: 672-pixel shorter edge; readout after zero-based block 11, pooled to
  `16 x 16`, tokenwise population z-score with `1e-6` floor, then row-L2 with
  `1e-8` floor.
- Student: the same model instance and weights. The teacher path disables all
  adapters and runs without gradients.
- LoRA replaces the complete fused `attention.qkv` linear in zero-based blocks
  `[8, 9, 10, 11]`; rank `4`, alpha `8`, dropout `0`, base bias unchanged.
  All backbone, LayerNorm, projection, and MLP parameters remain frozen. LoRA
  `A` uses the single deterministic seed and `B` starts exactly at zero.
- Blocks `0--7` run once without gradients and their output is detached. Base
  and student suffixes share this detached prefix. The model stays in eval
  mode and xFormers is disabled for the deterministic Stage -1 path.
- Optimizer: AdamW over LoRA tensors only, learning rate `1e-4`, weight decay
  `0`, gradient norm clip `1.0`, one quadruplet per step, bf16 autocast with
  fp32 losses. If bf16 is unsupported the shard is invalid; there is no dtype
  fallback selected after results.

## Quadruplets, folds, and schedule

For support base `x`, shifted-assumed transform `T`, synthetic anomaly `A`, and
the identical realized shift applied after anomaly insertion, each training
item is `x, T(x), A(x), T(A(x))`.

- Shift families: `J0_exposure_ev01333`,
  `J1_gaussian_rms_exposure_control`, and
  `J2_spatial_illumination_cv015`.
- Generator fit families: `G0_poisson_patch`, `G1_material_blotch`, and
  `G4_frequency_local`.
- Held generator: `G2_structural_scratch`; it is absent from every fitting
  view and target. Evaluation forms three held cells, one per shift family,
  using only the four sealed held bases.
- Donors are cyclic within the same fit or held partition. Realized mask,
  shift sign/noise/field, donor, and composition order are deterministic.
- Training is one complete shuffled epoch of
  `16 bases x 3 shifts x 3 fit generators = 144` updates per trained arm.
  There is no validation-based stopping.

## Arms and losses

Let `f` be normalized frozen-teacher tokens and `g` adapted-student tokens.
Synthetic-mask means give equal total mass to inside and outside regions so a
small anomaly cannot disappear inside a global average.

- `B0_FROZEN_TEACHER`: no adapter correction.
- `B1_IDENTITY_ZERO_INIT`: zero-initialized LoRA before fitting; it must be
  numerically identical to `B0` within `1e-7`.
- `M0_CANONICAL_KD`: minimizes
  `L_can + 0.25 L_id`, where `L_id` anchors `g(x)->f(x)` and
  `g(A(x))->f(A(x))`, while `L_can` maps `g(T(x))->f(x)` and
  `g(T(A(x)))->f(A(x))`.
- `M1_TARCD_FULL` (primary): `M0 + 0.25 L_RN + 0.10 L_margin`.
  `L_RN` matches teacher-target and student soft nearest-support distributions
  at temperature `0.05` and penalizes the absolute log ratio of normalized
  soft-RN distance scales. `L_margin` is the explicitly anti-absorption hinge
  `mean_m relu(|cos(delta_shift, r_anom)| - 0.20)` on generated-anomaly mask
  patches, where `delta_shift` is the LoRA correction and `r_anom` is the
  frozen-teacher synthetic-anomaly residual. Penalizing both alignment and
  anti-alignment encourages the learned shift correction to remain locally
  orthogonal to anomaly evidence; this is not a positive attraction margin.
  It is an auxiliary optimization term, not independent mechanism evidence;
  its value is attributed only through `M1-M0` held performance.
- `N1_SHUFFLED_QUADRUPLET`: same capacity, optimizer, steps, and `M0` weights,
  but clean/anomaly teacher targets are cyclically reassigned to a different
  support base while preserving transform, generator, mask, and schedule.
- `N2_RANDOM_MATCHED_LORA`: random same-shape LoRA tensors independently
  rescaled to every fitted `M1` tensor Frobenius norm; no fitting.
- `N3_STATIC_MEAN_ORACLE`: a spatial mean teacher correction estimated only
  from fit-support shift pairs.
- `O_PAIRED_TEACHER`: unattainable paired clean/unshifted-anomaly teacher
  target, reported only as an upper bound.

No adversarial generator classifier is optimized in this first assay. A
post-hoc, leave-one-held-base-out nearest-centroid probe tests generator
identity from LoRA corrections across all four generators and three shifts.
Separate magnitude, residual-spectrum, mask-geometry, and base-feature probes
measure low-order shortcuts. Training accuracy is never used as evidence.

## Frozen gates

Every numeric tree must be finite; each RN map must be nonconstant; support
LOO exact-RN maps must be finite/nonconstant; duplicate scoring must agree
within `1e-7`; all required disclosures and forbidden-access flags must match.
These are validity conditions, not scientific success.

For each of the three held shift cells, `M1` passes only if all hold:

- shifted-normal exact-RN q99 reduction versus `B0` is at least `5%`;
- preservation-guarded q99 gain exceeds the best of `M0`, `N1`, `N2`, and
  `N3` by at least `0.01`;
- shifted anomaly residual versus teacher unshifted anomaly has mean cosine
  at least `0.90` and median norm ratio in `[0.80, 1.25]`;
- normal and anomaly RN topology versus their paired teacher targets each has
  rank correlation at least `0.90`, nearest-neighbor retention at least
  `0.80`, and distance-scale ratio in `[0.50, 2.00]`;
- the candidate correction approaches the paired teacher correction with mean
  cosine at least `0.70` and median norm ratio in `[0.50, 1.50]`.

At object level, all three cells must pass, held-base generator-ID balanced
accuracy from `M1` corrections must be at most `0.35` (chance `0.25`), and
each low-order confound probe must be at most `0.60`.

## Sequential execution and verdict

1. Run a GPU0 `can` validity sentinel. Expansion depends only on custody,
   deterministic configuration, finite/nonconstant outputs, complete arms,
   and schema validity, never on scientific pass/fail.
2. If valid, run the four-object panel on dsba3 host GPUs `0,1,2,3`, one object
   per GPU, in fixed container `hun_fsad_tta_012` from one immutable snapshot.
3. `PROMISING_DIAGNOSTIC` requires all four valid, at least three object passes,
   and positive median `M1` margin over `M0` and all hard nulls. This remains a
   single-seed support-only diagnostic and does not authorize public data.
4. `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC` requires at least two object passes
   and no more than one object with worse than `-5%` median q99 reduction. Its
   only permitted continuation is a separately frozen multi-seed replication
   of the same configuration.
5. Otherwise the verdict is `KILL_FOR_CLAIM / NO_CONTINUE`. Generator leakage,
   null equivalence, topology failure, anomaly absorption, nondeterminism, or
   invalid custody cannot be repaired by a seed or hyperparameter sweep.

No TESTpublic/private/GT branch is automatically authorized under any Stage -1
outcome. Dense `anomaly_maps/` trees are neither written nor retained.
