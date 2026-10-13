# SRA-AD2-ALL8-SAME16-S3-v1: frozen implementation contract

Date: 2026-09-08. Status: READY_FOR_ONE_REVIEW. This is one production
matrix; complete every arm unless runtime, data custody, or comparability
fails. Cumulative deltas are order-dependent component contributions, not
universal causal effects. No HPO, early metric pilot, or metric-driven
redesign is permitted.

## Scope and custody

Target is MVTec AD2 single-image, data root
/home/hunim/Volume/DATA/mvtec_ad_2, all eight objects
[can,fabric,fruit_jelly,rice,vial,wallplugs,walnuts,sheet_metal], full
test_public/good,bad. Use frozen DINOv2 ViT-L/14, layers [5,11,17,23],
short-edge 672 except sheet_metal 448, canonical prepare_image and
extract_features. Every arm uses the exact same HAS16 16-reference manifest;
full manifest SHA-256 is runtime-pending and must be recorded.

Dataset enumeration is binding: call MVTecAD2Dataset.get_test_images(object),
assert the returned key set is exactly {good,bad}, and enumerate only those
lists. Do not recursively glob PNG files and do not enumerate or open any
ground_truth tree. The verified environment counts are
can=162, fabric=156, fruit_jelly=80, rice=132, sheet_metal=114, vial=140,
wallplugs=150, walnuts=150, total=1084 public queries.

Reference indices 0..11 are R_train; 12..15 are R_hold and remain closed
until all heads seal. Nuisance statistics and fitting use R_train only.
Public inference uses all 16 references. For training source i, exclude
i,(i+1)%12,(i+2)%12,(i-2)%12 from its support bank. The new protocol has no
C1 cache/checkpoint lineage.

Training A is exactly G0_poisson_patch, G1_material_blotch, and
G4_frequency_local. Training T is exactly global gains
e={0,-0.0667,-0.1333,-0.2}, applied as
PIL bicubic resize -> ToTensor -> float32 gain 2^e -> Normalize. No extra
fields, blur, clipping, rounding, or uint8 path is allowed.

## Shared feature budget

Cache support features for the 16 references once per object. For each
object/seed, use 240 logical view slots in the 144-quartet schedule:
12 clean x + 48 shifted-clean T_e x + 36 anomaly A_g x + 144 shifted-anomaly
T_e A_g x. The e=0 T0 slots reuse x and A_g, so only 192 distinct DINO
feature views are materialized. Each quartet is consumed by all four arms; no
DINO forward occurs per head update. Use a bounded FP16 patch-feature cache,
one resident quartet at a time, and no dense/native prediction-map files.

## One algorithm and cumulative arms

For each layer l and query patch q, retrieve K=8 nearest support patch vectors
under unsquared Euclidean distance and use the soft anchor
a_l=sum_k softmax(-d_k/tau_l) z_l(s_k). Set r_l=z_l(q)-a_l, with
tau_l=max(median_Rtrain(d_(8)),1e-3). Compute clean T displacement scales
sigma_l=max(1.4826*MAD(z_l(T_e x)-z_l(x)),1e-3), then freeze the deterministic
rank-4 left-singular basis U_l of the whitened clean displacements.

u_l=r_l/sigma_l; c_l=clip(U_l^T u_l,-3,3);
u_l_perp=u_l-U_l c_l. The bound 3 and rank 4 are fixed. One shared learned
bias-free projection P has shape 1024x16. Per layer use
q_raw_l=P u_l (16 channels), q_perp_l=P u_l_perp (16), and c_l (4);
concatenation over four layers is 64+64+16 channels.

Allocate the same modules for every arm:

    h_raw: Conv1x1(64,16)-GELU-DWConv3x3(16)-GELU-Conv1x1(16,1)
    h_res: Conv1x1(80,16)-GELU-DWConv3x3(16)-GELU-Conv1x1(16,1)
    gate:  Conv1x1(80,8)-GELU-Conv1x1(8,1)

For gate logit a=gate([q_perp,chi_tan*c]), score
s=h_raw(q_raw)+chi_perp*sigmoid(a)*h_res([q_perp,chi_tan*c]).
Switches are A0=(0,0,0), A1=(1,0,0), A2=(1,1,0), and A3=(1,1,1),
in order (chi_perp,chi_tan,anti_absorption). Disabled channels are zero,
not deleted. A0 is a learned full-vector control; RN-FMLK is independent.

## Exact training loss and steps

For mask m define B(v,m)=0.5*(mean_m softplus(-v)+mean_not_m softplus(v)).
The detached clean teacher y_x and 1024 non-tie rank pairs are recomputed
from R_train in this protocol. For each quartet x,T_e x,A_g x,T_e A_g x:

    L_clean = mean Huber_delta=.5(s_x-y_x)
              + .25*mean softplus[-sign(rp-rq)(s_p-s_q)]
    L_A     = B(s_A,m)
    L_TA    = B(s_TA,m)
    L_NT    = mean softplus(s_Tx)
    L_common = L_clean + L_A + L_TA + .25*L_NT
    L_AA = .5*B(a_A,m) + .5*B(a_TA,m)
           + .25*mean[softplus(a_x)+softplus(a_Tx)]
    A0,A1,A2: L=L_common; A3: L=L_common+L_AA

There is no LT, LATcons, output contrast-consistency, arm-specific
coefficient, or score fusion. L_AA supervises retention of the learned
full-feature residual branch itself, so it is not a rebranding of C1
contrast invariance. Empty masks/backgrounds, nonfinite values, and
constant outputs fail closed.

Use exactly 576 AdamW updates per object/seed/arm, batch 1, lr 3e-4,
weight_decay 1e-4, betas (0.9,0.999), eps 1e-8, gradient clip 1, no scheduler.
The 144 quartet order is deterministic and repeated four times. Seeds are
20260906, 20260907, 20260908; initialization and order are paired.

## Exact inference and evaluation

One DINO forward per public image is shared by A0..A3. Query K=8 support
patches from all 16 references, apply frozen sigma/U and learned P/head, and
restore each score map to native size with canonical bilinear interpolation.
No query adaptation, new basis, test GT, or post-hoc fusion. Stream FP16
scores and truthfully labeled GT into the canonical accumulators; persist no
dense maps.

Run independent same-condition SuperAD-16 and RN-FMLK/full-memory DINO-kNN.
The primary metrics are seg_AUROC_0.05, oracleF1, and AUPRO.05. Emit rows
first per object/seed/arm, then paired-seed per-object mean+SD, then the
eight-object macro mean; never pool seeds/pixels. Test GT is metric-only.
Each row contains protocol_id, method_id, object, seed, reference_manifest_sha256,
reference_count, method_<metric>, superad_<metric>, rn_fmlk_<metric>,
delta_vs_superad, delta_vs_rn_fmlk, comparable, map_coverage, finite,
nonconstant, and prediction_hash.

## Minimal runner API and ownership

Recommended new module API (existing C1 modules remain untouched):

    state = fit_shift_state(train_clean, train_shift, rank=4, bound=3)
    model = ShiftResidualHead(state, arm="A0|A1|A2|A3", seed=seed)
    loss = model.loss(quartet, teacher, mask)
    score = model.predict(query_features, references16)

Implement state/cache in src/flow_tte/shift_residual_cache.py, model/loss in
src/flow_tte/shift_residual_ablation.py, one-pass evaluation adapter and
StreamingSegmentationAccumulator integration in the evaluation lane, and
focused custody/shape/determinism tests in
tests/test_shift_residual_ablation.py. Remote route is dsba1 host GPU 6,
CUDA slot 0. Container, dataset, offline-DINO, and HAS16/all-eight readiness
are environment-confirmed; GPU 6 is currently occupied, so launch remains
pending. Perform one route verification when idle, then queue the full
8-object x 3-seed x 4-arm matrix.

## Final binding amendments

The following corrections are normative if any earlier shorthand conflicts:

- Each layer feature is [N,H,W,1024]. U_l is [1024,4], with orthonormal
  columns from the right singular vectors of the [samples,1024] whitened
  clean-displacement matrix; each column sign has its largest-absolute
  coordinate positive.
- The detached teacher is r_p=min_j ||zhat_23(x,p)-zhat_23(s_j)||_2/1024,
  where zhat is L2-normalized and j ranges over S_fit(i). Form equal-image
  q50/q95 over the 12 R_train maps and set
  y_p=clip((r_p-q95)/max(q95-q50,1e-6),-4,4). Enumerate flattened p<q
  lexicographically, retain r_p != r_q, and take the first 1024 pairs.
- P is shared across layers within one arm only. A0,A1,A2,A3 have independent
  parameters, initialized identically per paired seed and then optimized
  independently.
- The e=0 T0 slot reuses x and A_g; it is not a distinct feature view.
  Thus the shared cache is 192 distinct training views/object/seed, while
  the deterministic schedule remains 144 quartet slots repeated four times.
- Generator API is
  make_anomaly(base,donor,generator,seed)->(image_native,mask_native), with
  generator in {G0_poisson_patch,G1_material_blotch,G4_frequency_local};
  mask_to_grid(mask_native,grid_hw)->bool uses nearest interpolation and
  threshold 0.5.
- A0 is the normalized-residual control, A1 enables the residual branch, A2
  adds tangent coordinates, and A3 adds the synthetic gate auxiliary.
  A3 is not guaranteed to prevent anomaly absorption.
- RN-FMLK is an independent last-layer (layer 23), same-HAS16-reference
  comparator, not a full-memory baseline.
- The route uses dsba1 host GPUs 0,2,4, one independent object worker per
  GPU, with container slots and UUIDs assigned/verified by the runtime
  auditor. All eight objects and three paired seeds remain mandatory.
- Feature grids are ragged per reference/object after preprocessing. Store
  (H_l,W_l) for every image/layer, pool support patches with image/grid
  metadata, batch only same-shape views, and apply deterministic resizing only
  at final native-map restoration; never assume one global support grid.
- Resource override: use dsba1 host GPUs 0,2,4, one independent object worker
  per GPU; container slots and UUID mapping are runtime-auditor pending.

## Latest single-seed binding (authoritative)

This execution is protocol SRA-AD2-ALL8-SAME16-S1-v1: one scientific seed
20260906, four arms A0-A3, and 32 heads over all eight objects. Seeds
20260907/20260908 are excluded and remain optional future work. Existing
SuperAD-16 and RN-FMLK artifacts are preferred; baseline reruns are not
automatic. The audit records reference/split/preprocessing/metric evidence;
any mismatch sets comparable=false but never blocks candidate-arm execution.
Report per-object/per-seed rows followed by the eight-object macro; with one
seed, seed SD is null (not zero). Host GPU set remains dsba1 [0,2,4].

## Integration-check binding (authoritative, 2026-09-08)

Status: INTEGRATION_CHECK. The frozen initial matrix is the single seed
20260906: A0-A3, 32 heads, all eight objects, and 1084 TESTpublic queries.
baseline_reuse_manifest.json is the evidence pointer for existing SuperAD-16
and RN-FMLK results; no automatic baseline rerun is authorized. Historic
baseline values are FP32 whereas candidate streaming metrics are FP16, so
candidate-vs-historic rows must be recorded as comparable=false. The separate
historical_pair_valid field may be true only when the historical pair's own
manifest, split, preprocessing, metric, and precision metadata match; it
never upgrades candidate-vs-historic comparability. Record RUNNING only after
READY/launch evidence arrives. No training or evaluation has started at this
documentation checkpoint.

## Pre-launch deployment correction (authoritative, 2026-09-08)

Current status: DEPLOYING. Critic review and local preflight are complete;
remote launch approval has been issued, but no remote PID or launch receipt is
yet evidence, so RUNNING must not be recorded.

Evaluator contract: workers may enumerate the complete query inventory for
scheduling, but each query is handled as prediction -> GT-only metric update
-> immediate feature/map discard. This is querywise streaming evaluation, not a
global prediction barrier followed by a second metric pass, and no feature/map
may be retained for that purpose. Training and method selection remain
ground-truth-free.
