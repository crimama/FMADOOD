# CUG-RN-S1 Stage -1 Preregistration

Date frozen: 2026-08-25  
Protocol ID: `cug-rn-s1-stage-minus1-support-only-20260825`  
Claim status before execution: `DESIGN_STAGE / KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`

## Negative-evidence intake and failure-basin divergence

The branch starts after FACT-RN was almost identity and matched by its frame
shuffle, CIDER-CF separated generator identity rather than a transferable
mechanism, and MR-JW lost to its independent and shuffled controls. It is not
licensed as another support-only nuisance estimator or direct synthetic anomaly
classifier.

CUG-RN-S1 changes one mechanism: source synthetic anomalies supervise the
**counterfactual effect of a normal-derived update direction**. They never
define a deployed anomaly score and no target synthetic enters fitting. This is
an anti-absorption normal-only adaptation diagnostic, not a claim that the
assumed shifts represent MVTec AD2's real public acquisition shifts.

## Scope and custody

- Target: MVTec AD2 single-image, root
  `/home/hunim/Volume/DATA/mvtec_ad_2`.
- Target support: exact SuperAD-16 manifest
  `skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/superad16_dinov2_reference_paths.json`,
  SHA-256 `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Source fitting for target object `o`: exactly four deterministically ranked
  `train/good` images from each of the other seven objects.
- Target evaluation: at most eight deterministically ranked target
  `train/good` images disjoint from support-16.
- Forbidden before every Stage -1 gate passes: `test_public`, `test_private`,
  `ground_truth`, labels, condition tokens, query images, and GT-derived masks.
- Target support synthetics and held-target synthetics are sealed evaluation
  descendants only. They cannot fit an operator, choose a checkpoint, tune a
  transform, set a threshold, or select a branch.
- No query-time update, memory update, score fusion, anomaly classifier, Flow,
  object router, or B0/M0 fallback.
- `benchmark_comparable=false`; SuperAD segmentation metrics and unified
  `superad_*`, `method_*`, and `delta_vs_superad` fields are `N/A` in Stage -1.

## Frozen representation and one target update

Use frozen DINOv2 ViT-L/14 layer 11 features on the same 672-pixel canvas,
pooled to a `16 x 16` grid. Each token is channel-centered, divided by its
population channel standard deviation with floor `1e-6`, then L2-normalized
with floor `1e-8`; this is the frozen CIDER utility feature path. Let
source-only shift-residual PCA supply an orthonormal basis
`U in R^(1024 x 8)`. The identity-initialized adapter is

`z_theta(h) = normalize(h + ((h U) elementwise tanh(theta)) U^T)`.

PCA uses at most 16,384 residual rows selected at uniformly spaced indices
from the deterministic source/object/shift/token ordering, followed by a
fixed-seed (`0`) rank-16 range finder and rank-8 SVD. This cap and seed are
frozen before execution.

There are exactly four assumed-shift families:

- `J0_exposure`: multiplicative `1.18` or `0.82`, selected deterministically.
- `J1_contrast`: contrast `1.25` or `0.75` about `0.5`.
- `J2_gamma`: gamma `0.80` or `1.25`.
- `J3_blur`: Gaussian `5 x 5`, sigma `0.8`.

The inner objective is **only** paired shift consistency:

`L_pair(theta) = mean(1 - cosine(z_theta(x), z_theta(T_j(x))))`.

No LOO-RN term is included. At deployment all four families are used and

`g_N = grad_theta L_pair(theta=0)`,

`theta_S = -0.10 * P_phi g_N / max(||P_phi g_N||_2, 64 eps_fp32)`.

The source-learned operator is

`P_phi = cap_1(LL^T + diag(softplus(d)))`,

an `8 x 8` PSD matrix divided by `max(1, lambda_max)`; `L` is initialized from
`N(0, 0.01)` and `d` so the diagonal begins at one. Optimize with Adam,
`lr=0.03`, exactly 60 steps, no weight decay, frozen seeds
`1103, 2207, 3301`. Adapter displacement and parameter step norms are emitted.

The implementation evaluates the standard Adam equations directly with frozen
defaults `beta=(0.9, 0.999)` and `eps=1e-8`. This is algorithmically identical
to the declared optimizer but avoids importing the container's incompatible
`torch.optim -> torch._dynamo -> transformers` stack.

The deployed diagnostic score remains one exact hard nearest-normal distance
after applying the same adapter to support and evaluated descendants. `B0` is
raw exact RN; `M0_CUG` is post-adapter exact RN, not raw-RN invariance.

For source optimization, `soft-RN` means a temperature-`0.05` soft minimum
over the two inner images at the **same 16 x 16 token location**; this fixed
location match makes the 60-step meta-loop tractable and is never reported as
the deployed score. For target evaluation, `hard-RN` means the exact minimum
cosine distance over every token from every one of the 16 support images. An
image score is the 95th percentile of its 256 patch distances. Held-normal
tail reduction is the median, over the four assumed-shift families, of the
pre-minus-post 95th percentile image-score tail divided by `sigma`. Held-family
`Delta` is the median matched synthetic-minus-clean contrast change divided by
the same `sigma`.

Topology consists of three frozen statistics on held shifted-normal patch
maps: Spearman rank correlation between pre/post scores, exact nearest-support
token identity retention, and the post/pre `(Q95-Q50)` scale ratio. Every
candidate statistic must lie inside the random-operator 5th--95th percentile
envelope; degenerate zero-width scales fail closed.

## Source meta-episodes

For each source object, two ranked images form inner support and two form outer
evaluation. Each episode holds one shift family out of its inner gradient and
uses that family only in the outer normal loss. Thus all `7 x 4` source
object/held-shift episodes are explicit.

Four generator families are frozen, with every close variant kept in the same
family: `G0_poisson_patch`, `G1_material_blotch`,
`G2_structural_scratch`, and `G4_frequency_local`. For generator fold `k_h`,
`k_h` is absent from the objective, hyperparameters, stopping, and all fitting.
It is evaluated once as the held family.

For outer normal hard/soft RN score `s`, matched clean `x`, synthetic `A_k(x)`,
and pre-update support-LOO scale

`sigma = max(Q95(LOO_B0) - Q50(LOO_B0), 64 eps_fp32)`,

define

`Delta_k = [(s_post(A_k(x))-s_post(x)) - (s_pre(A_k(x))-s_pre(x))] / sigma`.

Training minimizes the mean held-shift soft-RN ratio plus

`max_{k in K_train} relu(-Delta_k)`.

Both terms are dimensionless; there is no fitted trade-off coefficient.
The maximum guard is taken across the three training generator families
inside each episode, then the combined normal-plus-guard loss is averaged over
the 28 episodes. A source episode's scale uses same-location cross-image LOO
scores from its two inner images; target evaluation uses exact global support
LOO scores from all 16 target supports.

## Frozen arms

Every arm shares encoder, basis, target normal gradient, step norm, support,
source episodes, and exact hard-RN scorer.

- `B0`: no update.
- `I_N`: identity preconditioner.
- `N_META`: the same PSD meta-operator trained on held normal shifts only.
- `M0_CUG`: held-generator-fold CUG operator.
- `S_PROJ`: source-only static projection. SVD of source anomaly-contrast
  gradients from `K_train` supplies at most two forbidden directions;
  `P=I-U_A U_A^T`. No target anomaly gradient is computed.
- `N_PAIR_SHUFFLE`: identical CUG training after rotating synthetic/clean base
  pairs within each two-image outer set.
- `R_RANDOM_00`--`R_RANDOM_15`: sixteen spectrum-matched random orthogonal
  rotations of the learned CUG eigenvalues.

FACT-RN, GB-CMF, OC-MAML/TSGA, and Metaformer/MT3 remain documented external
comparators; they are not forced into this first runner because they do not
share the adapter and optimization budget.

## Gate derivation (G-2)

No fixed 10% or `-0.02` efficacy threshold is used. For every object, seed, and
held-generator fold:

- Numeric MDE is
  `max(5 * duplicate_B0_max_abs_error, 64 eps_fp32)`.
- Directional null upper/lower bounds are the empirical 95th/5th percentiles
  across the sixteen spectrum-matched random operators plus the pair-shuffle
  arm.
- Topology bounds are the corresponding 5th/95th-percentile random-null
  envelope, not round-number retention thresholds.
- PSD and spectral-cap tolerances use `64 eps_fp64`; step-norm and duplicate
  controls use the recorded dtype-aware MDE.

`M0_CUG` passes a fold only when it is finite/nonconstant, PSD/cap/step valid,
inside the topology null envelope, has positive held-shift tail reduction,
has held-family `Delta` no worse than the lower null envelope, and exceeds
`N_META`, `S_PROJ`, `N_PAIR_SHUFFLE`, and the random-null upper envelope on
both normal-tail reduction and held-family `Delta` by more than the MDE.

This is deliberately a causal-activity gate. Strong synthetic separability
alone cannot pass it.

## Sequential dsba3 execution

1. Local CPU synthetic dry run and focused tests.
2. `can` validity sentinel on dsba3 host GPU 0 in fixed container
   `hun_fsad_tta_012`. It authorizes expansion only on custody, deterministic
   duplicate, finite/nonconstant, PSD/cap/step, and topology validity; it is not
   efficacy evidence.
3. If and only if the sentinel passes, run the four-object panel
   `can/fabric/vial/walnuts` on host GPUs `0/1/2/3`, three frozen seeds.
4. If and only if all custody/validity gates pass and at least three objects
   show a safe positive direction with at least two passing the full causal
   activity gate, run all eight support-only objects, two sequential objects
   per GPU.
5. No Stage -1 result authorizes TESTpublic automatically. A new preregistration
   and G-5 real-shift proxy calibration are required first.

Sentinel failure: `KILL_INVALID / NO_PANEL`. Panel failure with no surviving
positive direction: `KILL_FOR_CLAIM / NO_CONTINUE`. A safe positive but
sub-null effect is `NEAR_MISS` only because this preregistration explicitly
opens one bounded iteration lane; no hyperparameter sweep is allowed.

Each object contains exactly `3 seeds x 4 held-generator folds = 12` primary
rows. Object-level `safe_positive` and `full_causal_activity` require at least
8 of 12 corresponding row gates. Sentinel expansion checks validity only.
Panel expansion requires all four objects valid, at least three object-level
`safe_positive` gates, and at least two object-level
`full_causal_activity` gates.
The all-eight support-only verdict is `CONTINUE_DIAGNOSTIC` only when all eight
objects remain valid, at least six are `safe_positive`, and at least four pass
`full_causal_activity`; otherwise it is `KILL_FOR_CLAIM` (or `NEAR_MISS` when
some safe-positive evidence survives but the frozen counts are missed).

## Runway and retained value

Estimated cost is 1--2 GPU-hours for the sentinel and 4--8 aggregate GPU-hours
for the panel; all-eight is conditional. A negative result still leaves a
reusable PSD update/null harness and a direct answer to whether source anomaly
experience causally changes a target normal-only update. No dense anomaly map
is needed or retained.

## SuperAD and claim boundary

SuperAD baseline source for a later benchmark is the exact support-16 manifest
above plus a same-condition rerun; no historical metric is used here. Stage -1
primary endpoints are update causal activity, held-shift tail change,
held-generator contrast change, topology, and custody. Formal
`seg_AUROC_0.05`, `Seg-F1`, and `AU-PRO@0.05` remain unopened and `N/A`.
