# CRFR-S1 Stage -1 preregistration

Date: 2026-08-27  
Protocol: `crfr-s1-cross-recomposed-factorized-representation-20260827`  
Status: `PREREGISTERED / RESULT-FREE`  
Claim tier: support-only mechanism diagnostic; not benchmark-comparable

## Decision and sole change

Protected PCGrad ended the synthetic QKV-LoRA branch after the correct paired
objective produced zero conflicts and zero held contribution.  CRFR-S1 makes
one representation-level change: remove all backbone LoRA and learn an
explicit content/anomaly representation, a low-dimensional nuisance code, and
a crossed factor recomposer on top of a completely frozen DINOv2 feature map.

This candidate was selected because it conforms to the successor boundaries
stated by CRLP and TARCD; those reports did not pre-approve its efficacy.  It
is not an optimizer, loss-weight, adapter-location, family router,
retrieval-key, score-bypass, threshold, or query-update continuation.  No
second representation candidate is authorized by this protocol.

CRFR is also not a simplified FACT-RN continuation: it uses no
support-conditioned transport, probabilistic part frame, invertible transport,
retrieval-time reranking, or support-set router.  Its trainable heads see only
the current frozen feature tensor; the support set enters only the unchanged
exact-RN memory after training.

## Data and custody

- Objects: `can`, `fabric`, `vial`, `walnuts`.
- One fixed 16-normal support manifest per object.  Fit rows and all derived
  donors come only from these support images.
- Evaluation uses exactly four support-disjoint `train/good` bases per object.
- Shift families are the immutable J0 exposure, J1 Gaussian-RMS exposure
  control, and J2 spatial-illumination transformations used by TARCD S2/S3.
- Fit generators are the immutable G0/G1/G4 set; candidate mechanism and gate
  evaluation use only held `G2_structural_scratch`.  A separate evaluator-only
  generator probe renders G0/G1/G4/G2 on the four held bases solely to measure
  generator-ID leakage; none of those probe rows enters fitting or selection.
- TESTpublic, TESTprivate, test images, ground truth, query images, filenames,
  condition metadata, query fitting, memory updates, and score calibration are
  forbidden.
- Only compact JSON/log/hash evidence may be retained.  Dense features, score
  maps, anomaly maps, checkpoints, and image derivatives are forbidden.
- The immutable support manifest SHA-256 is
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
  The frozen feature/readout, CRFR core, runner, and aggregator source hashes
  are recorded in the `Frozen source identity` section below before any
  result-bearing launch.  The remote archive SHA-256 is computed at preflight
  and every object summary must bind that archive hash to those exact source
  hashes.

## Frozen source identity

The following values were fixed after the result-free implementation and
static review, before any remote model smoke or result-bearing launch:

- runner `scripts/run_crfr_s1_stage_minus1.py`:
  `7112820affc861271cb5037600ac6a8ef259b885efb223026454d0d09dc02d9b`;
- CRFR core `src/flow_tte/crfr.py`:
  `0fa6aef7635b02406aa1d1b6e79cafd31570cd3595dcec9288d562940bfc2c3c`;
- aggregator `scripts/aggregate_crfr_s1_stage_minus1.py`:
  `2fedd7c2dfe5575ced0459c2a016b2699e3945760eb05fad5c9d747ff5490b99`;
- frozen TARCD/CUG readout bundle:
  `25dc7c78903294cf7787345f9e67a6aef612aae2c857a5cfb2061f9450a3b8cc`.

Those four machine-readable values are stored in `source_hashes.json`, whose
SHA-256 is
`bf2125135a258dd558ceb3ee7ed75110ecb3acd579e9af65ce7491ed266f3b81`.
The fixed remote controller SHA-256 is
`fd64ee826e05233d4836d9ba26e4935ea8b25bf7bc19ae8eeae7c55db9ed9a64`.
Remote preflight must recompute the four code/readout values from the immutable
snapshot and require exact equality with the manifest.  The aggregator must
load the same manifest as a required CLI input and reject every self-reported
object hash that differs.

## Frozen feature and representation equations

Let `F(v) in R^(P x D)` be the existing channel-standardized, row-L2-normalized
frozen DINOv2 ViT-L/14 feature at the immutable TARCD readout and 16x16 grid.
`D` is read from the runtime tensor and is not hard-coded.  The backbone and
normalization have no trainable parameters and receive no gradients.

The trainable content head is a residual MLP with bottleneck 64:

`C_phi(z) = row_norm(z + W_c2 GELU(W_c1 LN(z)))`.

`W_c1` uses Xavier-uniform initialization and zero bias; `W_c2` weight and bias
are exactly zero, so the initial deployable content representation is an exact
row-normalized identity.  The nuisance head's two linear weights use
Xavier-uniform initialization and zero biases.  The recomposer's content,
nuisance, and coordinate projections use Xavier-uniform initialization without
bias; its final weight uses Xavier-uniform initialization with gain `0.01` and
zero bias.  All initialization seeds are arm-specific deterministic derivations
of seed `20260826`.

The nuisance head uses only unlabeled frozen features.  It concatenates the
patch-wise mean and standard deviation, then maps them through a 64-unit MLP
to an 8-dimensional bounded code:

`N_theta(z) = tanh(W_n2 GELU(W_n1 LN([mean_p z, std_p z])))`.

The crossed recomposer is a token-wise residual MLP with hidden width 64.  It
receives `C`, a learned projection of the 8-D nuisance code, and fixed
normalized `(x,y)` token coordinates:

`D_psi(C,n,p) = row_norm(C_p + W_d2 GELU(W_dc LN(C_p) + W_dn n + W_dp xy_p))`.

All heads are object-specific.  Shift-family and generator labels are never
inputs.  At deployment the nuisance head and recomposer are discarded.  The
single representation `C_phi(F(v))` enters the unchanged exact 1-NN RN scorer
against `C_phi(F(Support))`; there is no second score stream, fusion, max/min
firewall, query-dependent memory, or threshold.

## Training relation and objective

Every fit row is the paired quadruplet `x, T(x), A(x), T(A(x))`.  The full CRFR
arm uses crossed targets:

- `D(C(x), N(T(x))) -> F(T(x))`;
- `D(C(A(x)), N(T(x))) -> F(T(A(x)))`.

The no-cross control has identical capacity and optimizer but uses
same-view auto-recomposition:

- `D(C(T(x)), N(T(x))) -> F(T(x))`;
- `D(C(T(A(x))), N(T(A(x)))) -> F(T(A(x)))`.

The fixed full-arm objective is:

`L = 1.00 L_cross + 0.50 L_content + 0.25 L_identity + 0.25 L_anomaly + 0.05 L_code`.

- `L_cross`: `0.5` times the clean reconstruction cosine mean plus `0.5`
  times the anomaly-mask-balanced anomaly reconstruction cosine mean.
- `L_content`: `0.5` times cosine distance `C(T(x)) -> F(x)` plus `0.5`
  times the mask-balanced, activity-aware residual distance from
  `C(T(A(x)))-C(T(x))` to `F(A(x))-F(x)` defined below.
- `L_identity`: cosine distance `C(x) -> F(x)`.
- `L_anomaly`: the clean candidate residual `C(A(x))-C(x)` and shifted candidate
  residual `C(T(A(x)))-C(T(x))` are each compared to the same canonical frozen
  target `r_A=F(A(x))-F(x)`, balanced inside/outside the anomaly mask.  The
  shifted frozen residual `F(T(A(x)))-F(T(x))` is recorded only as a diagnostic
  and is never the training target.
- `L_code`: mean squared code drift for `N(A(x))-N(x)` and
  `N(T(A(x)))-N(T(x))`.

Cosine terms use epsilon `1e-8`.  Before training, each object freezes
`tau_A=max(1e-4,q25(||r_A||_2))` over nonzero fit-schedule anomaly residual
rows.  For a candidate residual `p` and target `r_A`, target-active rows
`||r_A||>=tau_A` use

`1-cos(p,r_A) + 0.25 smooth_l1(log((||p||+1e-8)/(||r_A||+1e-8)),0)`.

Inactive rows use `mean_D smooth_l1(p/tau_A,0)`.  Mask balancing is performed
per batch row: compute the scalar token loss first, average inside and outside
tokens separately, assign `0.5/0.5` weight when both groups are nonempty, use
weight `1.0` for the sole nonempty group otherwise, then average the resulting
row losses uniformly across the batch.  Clean rows without a mask use the
uniform mean over tokens and batch rows.  `L_anomaly` gives its clean and
shifted candidate residual row losses weights `0.5/0.5`.  There is no
target-normalized sample weight, division by the target vector, or cosine on a
zero target.

The `N2_NO_CROSS` objective has exactly the full-arm coefficients above and
changes only the two recomposer inputs to the same-view definitions already
listed; all content, identity, anomaly, and code terms remain active.  The
`N3_CLEAN_ONLY` objective is exactly

`L_N3 = 1.00 cosine(D(C(x),N(T(x))),F(T(x)))`
`       + 0.50 cosine(C(T(x)),F(x))`
`       + 0.25 cosine(C(x),F(x))`.

For N3, `L_anomaly=L_code=0` and both coefficients are inactive; there is no
anomaly-mask reduction.  Its 144 steps are the shared deterministic order of
the 48 `(base,shift)` clean pairs, with each pair repeated once in each of the
three otherwise ignored generator slots.  Every step evaluates exactly one
clean pair with the scalar objective above.  Anomaly images, masks, generator
implementations, and anomaly-descendant tensors are neither loaded nor
constructed by N3.

Training is exactly one deterministic 144-row epoch (`16 bases x 3 shifts x 3
fit generators`), seed `20260826`, AdamW learning rate `1e-3`, weight decay
`1e-4`, global gradient clip `1.0`, no scheduler, no early stopping, and no
checkpoint selection.  Traces at steps `1/36/72/108/144` are observational.

## Arms and matched nulls

- `B0_FROZEN_RN`: frozen DINO representation and exact RN.
- `I0_IDENTITY_HEAD`: zero-init CRFR before training; must reproduce B0.
- `M1_CRFR_CROSS`: full crossed factorization; sole candidate.
- `N0_ZERO_CODE`: M1 recomposition with its nuisance code set to zero.
- `N1_SHUFFLED_CODE`: M1 recomposition with a deterministic cyclic held-base
  nuisance-code permutation inside each cell.
- `N2_NO_CROSS`: independently trained matched-capacity auto-recomposition.
- `N3_CLEAN_ONLY`: independently trained matched-capacity crossed model using
  only `x,T(x)` terms; anomaly images, masks, and generator identities do not
  enter its loss.
- `N4_STATIC_MEAN`: immutable shift-family mean correction hard null.  For
  object `o` and experimental transform `Jj`, freeze
  `mu_(o,j)=mean_(i=1..16)[F(x_i)-F(Jj(x_i))]` token by token over exactly the
  16 fit-support rows.  The corrected held representation is
  `row_norm(F(Jj(x))+mu_(o,j))`; its correction cosine compares `mu_(o,j)` with
  the frozen held target `F(x)-F(Jj(x))`.  J0/J1/J2 means are never pooled.
  The controlled held experiment supplies `Jj` directly; no label is inferred
  and no family router is trained or deployed.  N4 is privileged diagnostic
  only and cannot contribute a deployable score stream.
- `O_PAIRED_TEACHER`: paired-clean oracle, diagnostic only.

Only `M1`'s `C_phi(F(v))` is a deployable candidate.  N0/N1 measure whether
the recomposer actually uses nuisance information; they never alter the RN
score.  N4 and O remain privileged diagnostics and cannot be promoted.

M1, N2, and N3 use the identical 144-row order, optimizer, and step count.
Their initialization seeds are respectively `stable_seed(20260826,"M1")`,
`stable_seed(20260826,"N2")`, and `stable_seed(20260826,"N3")`; the schedule
seed is shared and independent of arm.  N3 repeats clean pairs across the three
generator schedule slots but never loads or computes an anomaly descendant.

## Required evidence

Each object/family cell records:

- full, zero-code, shuffled-code, and no-cross reconstruction error;
  all four are evaluated on the same held crossed intervention and frozen
  targets, including N2 (although N2 was trained by same-view
  auto-recomposition);
- correction cosine and correction norm ratio, where the target is
  `F(x)-F(T(x))` and an arm's correction is
  `C_arm(F(T(x)))-F(T(x))`;
- anomaly residual cosine and norm ratio on clean and shifted anomaly pairs;
  both conditions are retained separately.  For the gate, first take the
  four-held-base median within each condition; cell `anomaly_cosine` is the
  smaller of the clean and shifted medians, and cell `anomaly_norm_ratio` is
  the condition median with the larger absolute log-distance from `1.0`.
- nuisance leakage ratio
  `r_N = rms(N(T(A(x)))-N(T(x))) / max(rms(N(T(x))-N(x)), 1e-4)`;
- leave-one-base-out four-class generator-ID balanced accuracy.  Each probe row
  is the L2-normalized existing `correction_descriptor` of
  `C(A_g(x))-C(x)` for evaluator-only G0/G1/G4/G2 on the four held bases.  The
  classifier is the existing fold-train-z-scored nearest-centroid probe, with
  one held base per fold and macro class recall;
- exact-RN shifted-normal q99 reduction versus B0, computed per held cell in
  float64 as `1-q99(M1)/max(q99(B0),64*float32_eps)`;
- anomaly RN rank correlation, nearest-neighbor retention, and scale ratio,
  comparing M1 canonical G2 maps/indices with the paired-clean oracle maps and
  indices under the same exact support memory;
- M1 correction-cosine margin over the maximum of N2, N3, and N4 in the same
  held cell;
- no-op, finite/nonconstant, row-normalization, frozen-gradient, support
  custody, source/hash, output-schema, and forbidden-access checks.

The `can` primary run is repeated with the identical seed and schedule.
Scientific metrics must agree within `1e-7`; model-state SHA-256 must match.
Mismatch is operational nondeterminism and stops interpretation.  If either
the metric tolerance or model hash check fails, GPUs 1--3 and the full panel
are not opened; the run ends as a result-free operational stop.
Fabric, vial, and walnuts carry only an inherited `can` gate marker and are not
independently repeated in this Stage -1 protocol.

## Frozen gates and stop rule

Aggregation is held-base median -> family cell -> object median -> four-object
median.  Missing, nonfinite, constant, wrong-count, custody, no-op, normalization,
hash, or access failures are fatal.

The exact identity no-op tolerance is `1e-7`; the row-L2 normalization maximum
absolute error tolerance is `1e-6`.

For every arm and object/family cell, reconstruction error `E_arm` is first the
median of that arm's four held-base row errors.  Relative gain is then computed
from those already aggregated cell medians as
`(E_k-E_M1)/max(E_k,1e-8)`; no median of row-wise ratios is used.

`CAUSAL_FACTOR_SIGNAL` requires at least 6/12 cells where all hold:

- for each null `k` in N0/N1/N2, that cell-median relative gain is at least
  `0.10`;
- correction cosine is at least `0.70`;
- correction norm ratio is in `[0.50, 1.50]`.

`ANOMALY_ISOLATION_SIGNAL` requires at least 9/12 cells where anomaly cosine is
at least `0.90`, anomaly norm ratio is in `[0.80, 1.25]`, nuisance leakage is
at most `0.20`, RN rank correlation and neighbor retention are each at least
`0.90`, and RN scale ratio is in `[0.80, 1.25]`.  Generator-ID balanced
accuracy must be at most `0.35` on at least three objects.

`SYNTHETIC_CONTRIBUTION` passes a cell only if M1 exceeds N3 clean-only
correction cosine by at least `0.01`, M1 anomaly cosine is no more than `0.01`
below N3, and M1 q99 reduction is no more than `0.005` below N3.  Progressive
status requires at least 3/12 passing cells; strict status requires 6/12.

`PROGRESSIVE_CRFR_SIGNAL` requires the structural/custody and repeatability
gates, both signals above, progressive synthetic contribution, global
hard-null margin at least `-0.094`, plus at least 3/12 cells spanning at least two
objects where shifted-normal q99 reduction is at least `0.05` and the
candidate correction-cosine margin over the best hard null is at least
`+0.01`.

`STRICT_CRFR_SIGNAL` additionally requires global hard-null margin `>=0`, at
least three positive object medians, all four object anomaly medians at least
`0.90`, every object anomaly norm median in `[0.80,1.25]`, strict synthetic
contribution, and at least 6/12 positive utility/hard-null cells.

Immediate scientific stop applies if any of these is observed:

- causal-factor gate fails;
- generator-ID BA exceeds `0.35` on at least two objects;
- synthetic contribution cells are `0/12`;
- positive hard-null cells are `0/12`;
- cells jointly passing anomaly/RN guards and positive q99 are `0/12`.

On failure, do not change head width, nuisance dimension, decoder depth, loss
weights, learning rate, seed, shift/generator strength, threshold, or training
length.  Close the support-only synthetic representation branch.  A passing
progressive signal licenses only a separately preregistered repeat/held-transfer
cycle, never public/test evaluation.

## Remote execution boundary

All result-bearing computation must run in the fixed project-owned dsba3
container `hun_fsad_tta_012`, with `can/fabric/vial/walnuts -> GPU 0/1/2/3`.
Before launch, inspect only those selected GPU UUID/memory/utilization and their
compute UUID entries.  Abort if any selected GPU is occupied.  Never enumerate,
inspect, modify, stop, or clean another container, process, user, file, volume,
or job.  Cleanup may remove only the exact immutable CRFR snapshot named by the
run; compact results remain.
