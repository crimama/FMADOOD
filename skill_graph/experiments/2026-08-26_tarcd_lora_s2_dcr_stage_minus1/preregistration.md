# TARCD-LoRA-S2-DCR Stage -1 Preregistration

Date: 2026-08-26  
Protocol: `tarcd-lora-s2-dcr-stage-minus1-support-adapted-20260826`  
Status: `PREREGISTERED / NO RESULTS INSPECTED`  
Seed: exactly one, `20260826`

## Negative Evidence Intake

TARCD-LoRA-S1 was valid on four objects and twelve held cells but retained the
paired teacher correction in `0/12`. Its RN-plus-orthogonality candidate was
worse than canonical endpoint KD on q99 reduction, correction direction, and
correction scale in every held cell. The S1 candidate itself repeated exactly,
while two control arms did not; S1 is frozen and no seed, weight, rank, or
threshold sweep is permitted.

S2-DCR is a separately named mechanism, not a retune. It replaces indirect
endpoint/orthogonality optimization with direct paired correction supervision,
cross-recomposition of anomaly evidence, and an RN constraint that activates
only outside a slack region. The failure basins under test are wrong-direction
correction, under-correction, anomaly absorption, RN-topology destruction,
synthetic-generator shortcuts, hard-null equivalence, and nondeterminism.

## Claim, Evidence, Boundary, and Positioning

- Claim: paired shifted-assumed augmentation plus synthetic anomalies can teach
  target-support LoRA a shift correction that approaches the frozen teacher's
  paired correction while preserving anomaly residuals and exact-RN topology.
- Evidence required: held-generator, support-disjoint correction direction and
  scale; anomaly recomposition direction and scale; RN topology; matched
  ablations and hard controls; repeatability and custody evidence.
- Boundary: single-seed, four-object, target-support-adapted mechanism
  diagnostic only. It is not a segmentation benchmark or SuperAD comparison.
- Positioning: a loss-mechanism test before any meta-learning or public branch.

## Data Custody

- Target dataset: MVTec AD2 single-image.
- Remote data root: `/home/hunim/Volume/DATA/mvtec_ad_2`.
- Objects: `can`, `fabric`, `vial`, and `walnuts`.
- Fit bases: the exact 16 `train/good` paths per object in
  `skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/superad16_dinov2_reference_paths.json`,
  SHA-256
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Held bases: four deterministic `train/good` paths per object disjoint from
  those 16. Every deterministic descendant of a held base is sealed from fit,
  model selection, stopping, calibration, and threshold choice.
- Exact-RN memory: frozen-teacher clean tokens from the same exact 16 support
  paths.
- Fit shifts: exposure `J0`, Gaussian RMS-matched control `J1`, and spatial
  illumination `J2`.
- Fit synthetic families: `G0_poisson_patch`, `G1_material_blotch`, and
  `G4_frequency_local`; held family: `G2_structural_scratch`.
- `TESTpublic`, `test_private`, ground truth, query images, query-memory updates,
  and public authorization are forbidden. The runner has no corresponding
  input path. Synthetic masks and labels may be used only for deterministic
  generation, balanced fit losses, and confound auditing.

## Representation and Schedule

- Frozen teacher/student backbone: DINOv2 ViT-L/14.
- Canvas: shorter edge 672; readout block 11; pooled `16 x 16` normalized tokens.
- LoRA: complete fused attention qkv in blocks 8--11, rank 4, alpha 8, zero B,
  no dropout; every non-LoRA parameter remains frozen.
- Optimizer: AdamW, learning rate `1e-4`, no weight decay, gradient clip `1.0`,
  bf16 forward and fp32 loss.
- Each trained arm uses the same complete deterministic schedule:
  `16 bases x 3 shifts x 3 fit generators = 144` updates. No early stopping.
- Activity scale `tau`: the 25th percentile of nonzero frozen-teacher clean
  paired-correction norms over fit-support shift pairs, clamped below by
  `1e-4`; computed once per object before arm fitting and shared by every arm.

## Direct Correction and Recomposition Loss

For frozen normalized teacher tokens define
`z0=f(x)`, `zs=f(T(x))`, `za=f(A(x))`, and `zsa=f(T(A(x)))`. For adapted
student tokens use `g0,gs,ga,gsa`.

Predicted and paired corrections are
`c_hat_n=gs-zs`, `c_star_n=z0-zs`,
`c_hat_a=gsa-zsa`, and `c_star_a=za-zsa`.
Corrected and target anomaly residuals are
`r_hat=gsa-gs` and `r_star=za-z0`.

For a target vector `v_star`, let
`w=stopgrad(||v_star||/(||v_star||+tau))`. Freeze

`D(v,v_star) = w * ((1-cos(v,v_star)) + 0.5 * smooth_l1(log((||v||+eps)/(||v_star||+eps)))) + (1-w) * smooth_l1((v-v_star)/tau)`.

Anomaly-vector losses use equal total mass inside and outside the synthetic
mask. Identity is cosine distance on unshifted clean/anomaly endpoints.

The RN guard uses the same soft support-neighbor distribution and scale as S1,
but applies squared hinge penalties only when `KL > 0.02` or
`|log(scale_ratio)| > log(1.25)`. It does not force exact distribution matching
inside the slack region.

## Arms

- `B0_FROZEN_TEACHER`: no adapted correction.
- `B1_IDENTITY_ZERO_INIT`: zero-init LoRA no-op.
- `M0_ENDPOINT_KD`: frozen S1 canonical endpoint KD control.
- `D0_DIRECT_CLEAN`: `L_corr_clean + 0.1 L_identity_clean`; no synthetic
  anomaly term contributes gradient.
- `D1_DIRECT_RECOMPOSITION`: `L_corr_clean + L_corr_anomaly +
  0.5 L_recomposition + 0.1 L_identity`.
- `D2_DCR_RN_GUARD` (candidate): `D1 + 0.1 L_RN_guard`.
- `N1_SHUFFLED_CORRECTION`: D2 capacity/schedule with clean and anomaly paired
  targets reassigned cyclically to a different support base.
- `N2_RANDOM_MATCHED_LORA`: random same-shape LoRA rescaled tensorwise to the
  fitted D2 state.
- `N3_STATIC_MEAN_ORACLE`: fit-support spatial mean paired clean correction.
- `O_PAIRED_TEACHER`: unattainable paired teacher endpoint, upper bound only.

The S1 absolute correction/anomaly orthogonality hinge is removed. Dense RN KL
matching is removed. Endpoint KD remains only as the explicit M0 control.

## Frozen Evaluation

Every held shift cell is evaluated with the held synthetic generator on all
four sealed bases. A D2 cell passes only if all hold:

- paired correction mean cosine `>=0.70` and median norm ratio in `[0.50,1.50]`;
- anomaly recomposition mean cosine `>=0.90` and median norm ratio in
  `[0.80,1.25]`;
- normal and anomaly exact-RN topology each has rank `>=0.90`, nearest-neighbor
  retention `>=0.80`, and distance-scale ratio in `[0.50,2.00]`;
- D2 correction cosine exceeds the best of M0, shuffled, random-matched, and
  static-mean controls by at least `0.05`;
- D1 recomposition error
  `(1-cosine)+|log(norm_ratio)|` is at most 90% of D0 error, while D1 correction
  cosine is no worse than D0 by more than `0.05`;
- per-topology RN violation is
  `relu(0.90-rank)+relu(0.80-NN-retention)+relu(|log(scale)|-log(2))`;
  clean and anomaly violations are summed, and D2 must be no larger than D1
  up to numeric tolerance `1e-12`; D2 correction cosine is no worse than D1 by
  more than `0.05`, and D2 recomposition error is no more than 10% worse;
- shifted-normal q99 reduction is at least `-5%` as a no-harm guard.

Absolute 5% q99 improvement is no longer a mechanism gate because the paired
teacher oracle met it in only `6/12` S1 cells. q99, paired-oracle q99, and
oracle-relative efficiency remain diagnostics.

At object level all three cells, all numeric/custody controls, and generator
confound guards must pass. Held-base generator-ID balanced accuracy from D2
corrections must be `<=0.35`; magnitude, spectrum, mask-geometry, and base
feature probes must each be `<=0.60`.

## Sequential Execution and Verdict

1. Local compile, focused tests, shell selftest, deterministic synthetic smoke,
   and actual CUDA model-path smoke must pass.
2. Run one GPU0 `can` validity sentinel. Expansion is authorized only by
   custody/schema/finite/no-op validity, never by scientific success.
3. If valid, run `can/fabric/vial/walnuts` concurrently on dsba3 host GPUs
   `0,1,2,3` in fixed container `hun_fsad_tta_012`, from one immutable snapshot.
4. `PROMISING_DIAGNOSTIC / MULTISEED_REQUIRED`: all four valid, at least three
   object mechanism passes, and positive median hard-null correction margin.
5. `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`: at least two object passes and no
   more than one object with median q99 reduction below `-5%`. Only one
   separately frozen replication/design decision may follow.
6. Otherwise `KILL_FOR_CLAIM / NO_CONTINUE`. No broad seed, rank, weight, or
   hyperparameter sweep is permitted.

No Stage -1 outcome automatically authorizes public/test/private/GT access.
`superad_seg_AUROC_0.05`, `method_seg_AUROC_0.05`, and `delta_vs_superad` are
`N/A`; `benchmark_comparable=false`. Dense `anomaly_maps/` are not written or
retained.
