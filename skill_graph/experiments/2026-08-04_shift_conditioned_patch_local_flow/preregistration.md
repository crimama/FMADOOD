# Shift-Conditioned Patch-Local Flow Preregistration

Date: 2026-08-04  
Status: `FROZEN_BEFORE_IMPLEMENTATION_AND_FEATURE_EXTRACTION`

## Negative Evidence Intake

This experiment follows the all-token Transformer Flow failure rather than
retuning it. The matched H+ experiment lost `0.008918` mean pixel pAUROC and
`0.021666` oracle F1, and reduced bad-image GT/background gap-z in all eight
objects. A raw DINO register-conditioned NF also failed the all-eight mean
gate, although it reduced good-map positive area and fragmentation in a
reduced morphology audit. Later acquisition-shift experiments showed that a
single hard image-global nuisance projection can suppress defect evidence,
while conservative local correction is safe but weak.

The new mechanism is therefore not a plain Transformer, raw-register, score
fusion, routing, or projection retry. It tests an augmentation-supervised,
support-relative shift descriptor that can read all frozen patches but cannot
update or directly mix them. The fitted MLP Flow is frozen, and the descriptor
can change a patch only through a bounded residual affine-coupling branch.

Likely failure basins are: a near-identity no-op, synthetic shift-proxy
overfitting, anomaly leakage into the query descriptor, and nuisance/defect
subspace overlap. The validity and held-support mechanism stage must reject
these before any public test image is opened.

## Motivation and Claim

Claim: a task-specific image descriptor can encode a recoverable shared part
of photometric acquisition shift, while a bounded patch-local conditional
adapter reduces shifted-normal false positives without erasing local anomaly
contrast.

Evidence required: relative shift prediction on unseen support transforms,
raw-Euclidean latent alignment, correct-register advantage over shuffled and
zero descriptors, unshifted baseline retention, synthetic local-defect
retention, and only then public pixel ranking/mask metrics.

Boundary: the experiment does not assume that every acquisition shift is one
global factor. It tests only the recoverable image-global component. The
descriptor is image-global, so "patch-local" means no direct patch-token
mixing; it does not mean that patch outputs are statistically independent.

Positioning: reduced-object public results are diagnostic. AD2 TESTpublic has
already influenced the design and remains kill-only. Fresh support seeds test
support robustness, not benchmark independence.

## Frozen Data and Detector Contract

- `target_dataset=MVTec AD2 single-image`
- `data_root=/home/hunim/Volume/DATA/mvtec_ad_2`
- fixed support manifest:
  `skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json`
- manifest SHA-256:
  `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`
- exactly 16 normal support identities per object; augmentations do not add
  support-memory entries
- backbone: `dinov3_vith16plus`, frozen, layers `[7,15,23,31]`
- feature fusion: `layer_norm_mean`
- resolution: existing per-object AD2 resolution contract
- DVT: support-fitted `position_mean`, alpha `1.0`; the original-support
  artifact is frozen and reused for augmented support and query images
- Stage-A Flow: two affine coupling layers, hidden multiplier `1`, three
  epochs, LR `2e-4`, clamp `1.9`, tail weight `0.3`, tail top-k `0.05`,
  logdet regularization `1e-3`, feature standardization on
- deployed score: exact raw-Euclidean latent 1-NN to one static support bank,
  LOO support standardization, `distance_weight=1`, `density_weight=0`, no
  context routing or score fusion
- public Phase-1 objects: `fruit_jelly,walnuts,vial,fabric`
- full-public continuation objects: all eight AD2 public objects

The historical matched H+ MLP/Transformer artifacts are context. E0 is
regenerated in the same process from the exact Stage-A checkpoint used by all
candidate arms. A strict paper-level SuperAD comparison is not claimed because
the H+ backbone contract is not the DINOv2-L/14 SuperAD paper contract.

## Corrected Conditional Architecture

- four learned register queries, register width 128
- one four-head cross-attention layer and one FFN of width 256, dropout zero
- frozen DVT patch features provide keys/values and are returned unchanged
- mean register pooling followed by a 64-dimensional descriptor projection
- support center is the detached channel-wise median of the 16 original
  support descriptors, recomputed during Stage B and frozen for inference
- support/query condition is `g(x)-g_S`
- each Stage-A coupling MLP is frozen
- each block owns a new zero-initialized residual output head
- the single bounded residual gain is initialized to `0.05` and constrained
  to `(0,0.1)`; it is not applied a second time inside FiLM
- exact baseline output at initialization follows from the zero residual head,
  while the nonzero gain keeps the residual-head gradient live
- reverse evaluation must receive the exact same descriptor used by forward;
  this is conditional invertibility, not invertibility from latent alone

The final inference scorer does not L2-normalize memory or query latents.
Normalization is allowed only inside the invariance and neighborhood losses.

## Stage-B Pair and Loss Contract

Photometric transforms use the frozen ranges from the proposal: brightness
`[0.7,1.3]`, contrast `[0.8,1.2]`, gamma `[0.8,1.2]`, RGB gains
`[0.9,1.1]`, Gaussian blur sigma `[0,1]`, and Gaussian noise std `[0,0.02]`.
The deterministic mixture is 50% one active transform, 30% two active
transforms, and 20% identity. Spatial transforms are excluded.

Because compound transforms are allowed, transform type uses six independent
multi-label targets and binary cross-entropy, not a mutually exclusive CE.
Continuous parameters are normalized, and regression is masked to active
transforms. The predictor input is relative:
`g(T_a(x))-g(x)`, not `g(T_a(x))` alone.

The full Stage-B loss is:

`L = L_nll + 1.0 L_inv + 0.1 L_shift + 0.5 L_nbr + 1.0 L_id + 1e-3 L_res`.

`L_nll` is patch NLL divided by feature dimension, making E2 and the full
loss contract explicit. `L_inv` is normalized-latent Huber consistency.
`L_nbr` uses a deterministic sample of at most 256 original-image patches,
teacher cosine top-10 neighbors, temperatures `0.1/0.1`, stop-gradient teacher
probabilities, and `KL(q_teacher || p_student)`. `L_id` compares unshifted
candidate latent to a stop-gradient Stage-A latent. `L_res` penalizes residual
scale and shift before the bounded gain.

Stage B trains only register queries/projections, relative shift predictor,
FiLM/residual heads, and bounded gains for 10 epochs at LR `1e-4`, weight decay
`1e-4`, gradient clipping `0.5`. Stage A and the backbone remain frozen.

## Arms

1. E0: exact Stage-A MLP Flow
2. E1: E0 checkpoint plus disabled/zero-output register branch
3. E2: register residual branch, normalized NLL only
4. E3: E2 plus paired relative shift prediction
5. E4: E3 plus latent consistency
6. E5: E4 plus neighborhood, identity, and residual regularization
7. E6: historical matched all-token Transformer Flow control

The initial implementation executes E0, E1, E5, zero/shuffled-register nulls,
and the Stage-B loss-component log. E2--E4 are retained as bounded ablations
only after the mechanism stage is valid; they are not required to access the
four-object smoke.

## Phase 0: Local Validity Gates

All must pass:

1. nontrivial residual forward/reverse reconstruction maximum error `<=1e-5`
2. forward/reverse logdet cancellation maximum error `<=1e-5`
3. zero-head E1/E0 latent and logdet maximum error `<1e-6`
4. residual-head and register paths receive finite nonzero gradients at the
   initial training state
5. deterministic repeated output for identical patch map and descriptor
6. after a nonzero residual witness, correct/shuffled descriptors change the
   output while patch extraction itself remains unchanged
7. memory scoring remains raw Euclidean and support/query memory shapes match

Any failure is `IMPLEMENTATION_INVALID`; no remote feature extraction follows.

## Phase -1: Support-Only Held-Shift Mechanism Gate

The runner may open only the 16 sealed `train/good` images and deterministic
derived transforms. It may not enumerate or open TESTpublic/TESTprivate paths.
Training and held-shift schedules use disjoint deterministic seeds. One
deterministic synthetic local defect family is used only after training for
retention diagnosis and cannot alter weights or gates.

Required aggregate continuation conditions across the four objects:

- exact E0/E1 parity and conditional-inverse validity remain satisfied
- correct-register held-shift raw 1-NN inflation is at least 5% lower than E0
- correct register beats both shuffled and zero-register correction on at
  least three of four objects
- unshifted support LOO score mean changes by at most `0.05` E0 LOO standard
  deviations and latent standard-deviation ratio stays in `[0.8,1.25]`
- synthetic defect/background gap retention is at least 95% of E0 macro and
  no object loses more than 10%
- relative shift prediction beats the zero predictor and remains finite

If validity fails, verdict is `EARLY_VALIDITY_KILL / NO_PUBLIC_ACCESS`. If
retention passes but shift correction is below 5%, verdict is
`KILL_FOR_CLAIM / NO_CONTINUE`. A partial correct-register signal on at least
two objects with no retention collapse permits exactly one
`CONTINUE_DIAGNOSTIC`, but still does not authorize public access unless the
full frozen support-only gate passes.

## Phase 1: Four-Object TESTpublic Smoke

This phase is authorized only by a machine-readable Phase -1 pass sentinel.
It evaluates full `test_public/good,bad` for
`fruit_jelly,walnuts,vial,fabric`, generates E0/E1/E5 maps from the same
Stage-A state, and evaluates each with the identical pixel evaluator.

Primary metrics are `seg_AUROC_0.05` and pooled best-threshold pixel
`seg_F1`, explicitly labeled oracle F1. Diagnostics include bad-image gap-z,
top-1% GT share, per-image IoU/precision/recall/component statistics,
shifted-good p95 inflation, bad-good p95 separation, register shuffle/zero
effects, and complete finite/nonconstant map coverage.

KEEP for all-eight continuation requires:

- E1 and E0 metrics/maps are equal within numerical tolerance
- E5 mean pAUROC is no worse than E0 by `0.003`
- mean bad-image gap-z delta is nonnegative and no object is below `-0.05`
- shifted-good p95 inflation or fixed-threshold shifted-good FPR falls by at
  least 5%
- `fruit_jelly` F1 does not fall by more than `0.05`
- all maps are finite, nonconstant, shape-matched, and complete

This is a reduced-object `PROMISING_DIAGNOSTIC` gate, never a method claim.

## Phase 2 and Phase 3 Continuation

Only a Phase-1 pass authorizes all-eight E0/E5 and bounded E2--E4/loss
ablations. All-eight continuation retains the proposal's pAUROC/gap-z/no-harm
gate and additionally reports the historical E6 control. Random P16 support
seeds 0/1/2 follow only after all-eight passes. Those seeds measure support
robustness; promotion still requires an untouched benchmark or TESTprivate.

## Remote Execution and Custody

- host: dsba5
- fixed container: `hun_fsad_tta_012`
- selected resource: host GPU 3, exposed as container `cuda:0`
- source is streamed to a new immutable
  `/workspace/isolated_runs/<run>/repo` snapshot with SHA-256 custody
- remote results: `/workspace/results_remote/<run>`
- local pullback: `results/remote_runs/dsba5/<run>`
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before CUDA initialization
- remote preflight verifies container visibility, data population, exact
  support hash, DINOv3 offline cache, imports, code hashes, disk, and idle GPU
- invalid runs are never overwritten; corrected executions receive new names

After a valid conclusion, all regenerable `anomaly_maps/` trees are removed
remotely and locally after compact metrics, manifests, logs, diagnostics, and
checksums have been pulled and verified.

