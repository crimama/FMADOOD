# CUG-RN-S2-MTP Stage -1 Preregistration

Date frozen: 2026-08-25  
Protocol ID: `cug-rn-s2-mtp-stage-minus1-support-only-20260825`  
Claim status before execution: `DESIGN_STAGE / LAST_BOUNDED_CONTINUATION`

## Negative-evidence intake

CUG-RN-S1 was valid and safe-positive on all four support-only panel objects,
but it passed zero full-causal object gates. Its normal-tail effect failed the
strongest matched control in 41/48 rows, its held-generator contrast failed in
47/48 rows, `can` lost every RN-neighbor topology row, and `walnuts` lost every
distance-scale row. S1 optimized a baseline normal-tail ratio plus a one-sided
synthetic guard; matched-null superiority and exact-RN topology were evaluated
only after fitting.

S2-MTP is a separately named mechanism, protocol, schema, implementation, and
run. It changes the meta-objective rather than S1's seeds, step norm, learning
rate, thresholds, or panel. It is the single bounded continuation licensed by
the S1 report. Failure closes this exact branch; success licenses only a new
all-eight preregistration.

## Scope and custody

- Target: MVTec AD2 single-image at
  `/home/hunim/Volume/DATA/mvtec_ad_2`.
- Exact target support: SuperAD-16 manifest
  `skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/superad16_dinov2_reference_paths.json`,
  SHA-256 `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Target object source fit: four deterministically ranked `train/good` images
  from each of the other seven objects.
- Sealed target evaluation: at most eight deterministic target `train/good`
  images disjoint from support-16.
- Forbidden throughout this experiment: `test_public`, `test_private`,
  `ground_truth`, labels, query images, condition tokens, GT-derived masks,
  target-synthetic fitting, query-time updates, memory updates, score fusion,
  and runtime arm selection.
- Target synthetics are sealed evaluation descendants only. They cannot fit an
  operator, select a checkpoint, tune a transform, set a threshold, or choose
  a branch.
- `benchmark_comparable=false`; SuperAD segmentation metrics remain `N/A`.

## Frozen representation and deployment operation

S2-MTP preserves the complete S1 representation and deployment boundary:

- frozen DINOv2 ViT-L/14 layer 11 on a 672-pixel canvas;
- pooled `16 x 16` feature grid;
- channel centering and population-standard-deviation normalization with
  floor `1e-6`, followed by L2 normalization with floor `1e-8`;
- deterministic source shift-residual rank-8 basis with 16,384-row cap and
  range seed `0`;
- identity-initialized rank-8 residual adapter;
- spectrally capped `8 x 8` PSD update operator;
- one target normal paired-shift gradient and one update of exact norm `0.10`;
- exact global hard cosine nearest-normal scoring after the same adapter is
  applied to support and evaluated descendants.

The four shift families, four generator families, source episodes, held
generator folds, seeds `1103/2207/3301`, 60 standard Adam steps at `lr=0.03`,
and no-weight-decay equations are unchanged from S1.

## Global RN training probe

The S1 same-location soft-RN proxy is replaced only inside the S2 source
meta-objective. For each two-image source outer set, select exactly 16 patch
tokens per image at uniformly spaced indices over the frozen 256-token order.
This gives 32 query anchors. Compare every anchor against all 512 tokens from
the two inner support images. The cap is a compute contract frozen before
results, not a selected subset from scores.

For distance `d(q,s)=1-cos(q,s)` and `tau=0.05`, use the support-count-normalized
global soft RN

`softRN(q,S) = -tau * [logsumexp(-d(q,S)/tau) - log(|S|)]`.

Hard neighbor/scale constraints use the exact minimum over the same 512 source
support tokens, not the soft score.

## Matched-null margin objective

For source episode `e`, operator `P`, pre-update source LOO scale `sigma_e`,
held normal shift score tail `T_e`, clean tail `C_e`, and generator-family tail
`A_e,k`, define higher-is-better endpoints

`R_e(P) = [T_e(0) - T_e(P)] / sigma_e`

and

`D_e,k(P) = [(A_e,k(P)-C_e(P)) - (A_e,k(0)-C_e(0))] / sigma_e`.

The anomaly endpoint is the worst training-family value

`D_e(P) = min_{k in K_train} D_e,k(P)`.

Each held-generator fold excludes its held family from all fitting. Before the
candidate fit, construct source-only fixed controls with the same rank, target
gradient rule, update norm, and feature/scoring path:

- `N_META`: normal-only meta operator;
- `I_N`: identity preconditioner with the same normalized target-update budget;
- `S_PROJ`: static source anomaly-gradient projection;
- `N_PAIR_SHUFFLE`: synthetic/clean pair-shuffle meta operator.

`N_META`, `S_PROJ`, and `N_PAIR_SHUFFLE` are re-fit for S2 using the same
16-anchor global-RN probe and normalized soft-RN score above; they do not reuse
the S1 same-location training proxy. `I_N` is evaluated through that same
source endpoint path without fitting.

Four deterministic training-only orthogonal matrices `Q_00..Q_03` are derived
from the protocol ID, seed, held family, and rotation index. At every optimizer
step, `Q_m P Q_m^T` is an exact differentiable spectrum-matched orientation
null. The 16 target evaluation random operators use a disjoint `eval-random`
seed namespace and are never visible to fitting.

Let `C_e(P)` contain the four fixed controls plus four current
spectrum-matched rotations. Define

`mN_e(P) = R_e(P) - max_{C in C_e(P)} R_e(C)`

and

`mA_e(P) = D_e(P) - max_{C in C_e(P)} D_e(C)`.

The numeric margin floor is `delta_num = 64 * eps_fp32`; no efficacy margin is
tuned from S1 target results.

## RN topology constraints

For each source anchor, let `i0` be its exact raw nearest support token. Under
the candidate update, let `d_i0` be the distance to that same token and
`d_other` the minimum distance to every other source support token. The exact
neighbor-identity violation is

`v_neighbor = mean(relu(d_i0 - d_other)) / sigma_e`.

It is zero precisely when the raw nearest token remains a nearest token, up to
ties.

Let `w0=Q95(s0)-Q50(s0)` and `wP=Q95(sP)-Q50(sP)` for exact hard-RN anchor
scores. The multiplicative distance-scale violation is

`v_scale = relu(abs(log(max(wP,floor)/max(w0,floor))) - delta_num)`.

The per-episode objective is the maximum positive-part constraint

`L_e = max(relu(delta_num-mN_e), relu(delta_num-mA_e), v_neighbor, v_scale)`.

The optimizer minimizes `mean_e L_e`. There is no baseline-only tail term and
no fitted trade-off coefficient. The max form prevents a large efficacy gain
from compensating for an RN-neighbor or distance-scale violation.

## Frozen evaluation arms and target gates

All arms share support, basis, target gradient, step norm, scoring path, and
held generator:

- `B0`, `I_N`, `N_META`, `M0_CUG_MTP`, `S_PROJ`, `N_PAIR_SHUFFLE`;
- `R_RANDOM_00`--`R_RANDOM_15`, generated in the independent evaluation seed
  namespace with the learned candidate spectrum.

For each object, seed, and held-generator row:

- numeric MDE is `max(5 * duplicate_B0_max_abs_error, 64 eps_fp32)`;
- duplicate no-op determinism is an independent validity gate with frozen
  absolute tolerance `1e-7`; the MDE formula cannot relax this gate;
- candidate normal and held-generator endpoints must each exceed `I_N`,
  `N_META`, `S_PROJ`, `N_PAIR_SHUFFLE`, and the evaluation-random Q95 by more
  than MDE;
- rank correlation must be no lower than evaluation-random Q05 minus MDE;
- neighbor retention must be no lower than the maximum of `I_N`, `N_META`,
  `S_PROJ`, `N_PAIR_SHUFFLE`, and evaluation-random Q95 minus MDE;
- absolute log distance-scale error must be no greater than the minimum of the
  four fixed controls and evaluation-random error Q05 plus MDE;
- zero-width scale, nonfinite/constant maps, invalid PSD/cap, wrong step norm,
  non-deterministic duplicate, or forbidden access fails closed.

`safe_positive` retains the S1 diagnostic definition: valid, positive normal
tail reduction, and held-family delta no worse than the lower random/shuffle
envelope. `margin_topology_pass` requires both endpoint margins and all three
topology checks. An object passes topology or full activity only with at least
8/12 corresponding rows; validity requires 12/12.

## Sequential dsba3 gate

1. Run local compilation, focused tests, shell checks, and synthetic dry run.
2. Run a `can` validity sentinel on dsba3 host GPU 0 in fixed container
   `hun_fsad_tta_012`. Efficacy does not control sentinel expansion.
3. If and only if sentinel custody and validity pass, run the frozen panel
   `can/fabric/vial/walnuts` on host GPUs `0/1/2/3` under one immutable
   snapshot. Each distributed panel runner accepts exactly one shard from this
   tuple, and the aggregator requires the complete tuple exactly once.
4. Stop after the panel regardless of outcome. No all-eight, public, private,
   or GT lane is authorized by this protocol.

Panel success requires all four objects valid, all four object-level topology
gates passing, and at least three objects passing full matched-null activity.
It yields `PROMISING_DIAGNOSTIC / NEW_ALL8_PREREG_REQUIRED`, not a claim.
Any other valid panel result yields `KILL_FOR_CLAIM / NO_CONTINUE`. Sentinel
invalidity yields `KILL_INVALID / NO_PANEL`.

## Literature and positioning boundary

SNE/trustworthiness-continuity, metric-learning margins, learned optimizers,
one-class meta-learning, and synthetic anomaly training are established. The
bounded contribution being tested is narrower: a support-only update operator
that must beat budget-matched control operators under a direct source margin
while preserving exact RN neighbor identity and local distance scale. Synthetic
anomalies remain source outer-query instruments, never a deployed classifier.

## Retained value and cleanup

This is support-only mechanism evidence. A negative result closes the exact
CUG meta-update branch while retaining the reusable global-RN margin/topology
harness. No dense anomaly maps are required; local and remote result roots must
contain zero `anomaly_maps/` directories after conclusion.
