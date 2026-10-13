# Shift-Residual Full-Feature Ablation: Implementable Design

Date: 2026-09-08  
Protocol: SRA-AD2-ALL8-SAME16-S3-v1  
Status: DESIGN_READY_FOR_ONE_REVIEW

## 1. Method contract

The final method, A3, is a learned support-residual scorer over frozen
DINOv2 ViT-L/14 feature vectors. It has three structural ingredients:

1. a learned full-vector residual projection;
2. a bounded normal-shift orbit estimated from clean train-only references;
3. a synthetic-anomaly gate loss that prevents the shift-residual branch from
   being silently absorbed on masked anomalies.

It is not a passive feature projection and is not an RN-FMLK/kNN score:
the shared projection, local head, and branch gate are trained with map
supervision. The full-vector support residual is the input evidence; the
normal-shift basis is frozen after train-only construction.

## 2. Data and train-only construction

Let z_l(X,p) in R^1024 be the DINO feature at layer
l in {5,11,17,23} and patch p. The exact same 16 normal reference images
are used by every candidate arm. Only references 0..11 (R_train) may enter
fitting or nuisance construction; references 12..15 (R_hold) are released
after all heads seal. Public scoring uses all 16 references.

For a training source i, define:

~~~
S_fit(i) = R_train \ {i, (i+1)%12, (i+2)%12, (i-2)%12}
~~~

Use the same support exclusion for x, T_e x, A_g x, and T_e A_g x.
All support features and synthetic cells are regenerated under this protocol
ID; no C1 feature/cache/checkpoint hash is inherited.

### 2.1 Full-vector support residual

For each layer, retrieve the K=8 nearest support patch vectors from S_fit(i)
under unsquared Euclidean distance. Form a soft anchor:

~~~
w_k = softmax(-d_k / tau_l),  k=1..8
a_l(q,p) = sum_k w_k * z_l(s_k,p_k)
r_l(q,p) = z_l(q,p) - a_l(q,p)
~~~

tau_l is fixed before fitting to
max(median_train(d_(8)), 1e-3), where the median is over clean R_train
query patches and the eighth-neighbor distance. At public inference, the same
formula uses all 16 reference images. The top-8 residual directions are
retained in memory; only compact bounded feature tensors may be cached.

### 2.2 Bounded normal-shift model

For clean R_train images, generate only the four fixed global gains
e={0,-0.0667,-0.1333,-0.2}. At each layer, collect

~~~
d_l(x,e,p) = z_l(T_e x,p) - z_l(x,p)
~~~

Compute a coordinatewise robust scale
sigma_l=max(1.4826*MAD(d_l),1e-3) by streaming pass, whiten
u_l=sigma_l^(-1) r_l, and compute the leading rank=4 left singular
vectors U_l of the whitened clean displacement collection. Use deterministic
sign canonicalization (largest-magnitude coordinate positive) and freeze
U_l and sigma_l before any arm fit.

The nuisance coefficients and residual are:

~~~
c_l = clip(U_l^T u_l, -3, 3)
n_l = U_l c_l
u_l_perp = u_l - n_l
~~~

The coefficient bound 3 is fixed and is not tuned. This is a bounded
source-only normal orbit, not query-time adaptation. If a real defect lies in
the same subspace as the learned nuisance orbit, it is not identifiable from
these observations; the anti-absorption gate can expose that conflict but
cannot recover the lost distinction.

## 3. Fixed trainable architecture

All arms allocate the same modules and parameter shapes. DINO ViT-L/14
hidden size is asserted to be 1024 at all four selected layers.

Use one shared trainable bias-free matrix
P in R^(1024 x 16) for all layers. For each layer concatenate channels:

~~~
q_raw_l  = P u_l                 # 16 channels
q_perp_l = P u_l_perp            # 16 channels
c_l                                  # 4 channels
~~~

Across four layers this gives q_raw:64, q_perp:64, and c:16 channels.
All maps are on the common DINO patch grid.

Define three shared convolutional modules:

~~~
h_raw:
  Conv1x1(64,16) -> GELU -> depthwise Conv3x3(16)
  -> GELU -> Conv1x1(16,1)

h_res:
  Conv1x1(80,16) -> GELU -> depthwise Conv3x3(16)
  -> GELU -> Conv1x1(16,1)

g:
  Conv1x1(80,8) -> GELU -> Conv1x1(8,1)
~~~

Let a=g([q_perp,c]) be the gate logit. With fixed arm switches
chi_perp and chi_tan, the map logit is:

~~~
s = h_raw(q_raw)
    + chi_perp * sigmoid(a([q_perp, chi_tan*c]))
                  * h_res([q_perp, chi_tan*c])
~~~

The modules and P are initialized once per scientific seed and cloned
identically across the four paired arms. Disabled channels are zero, not
removed, so parameter allocation and optimizer state are matched.

The cumulative switches are exactly:

~~~
A0: chi_perp=0, chi_tan=0, anti_absorption=0
A1: chi_perp=1, chi_tan=0, anti_absorption=0
A2: chi_perp=1, chi_tan=1, anti_absorption=0
A3: chi_perp=1, chi_tan=1, anti_absorption=1
~~~

## 4. Fixed losses

For a map logit s and synthetic mask m, define:

~~~
B(s,m) = 0.5 * (mean_m softplus(-s) + mean_not_m softplus(s))
~~~

The detached clean teacher y_x is recomputed in this isolated protocol from
the normalized layer-23 full-feature support distance, with equal-image
R_train quantiles and no public/held GT. Use 1024 deterministic non-tie
teacher pairs. Define:

~~~
L_clean = mean Huber_delta=0.5(s_x - y_x)
          + 0.25 * mean softplus[-sign(rp-rq)(s_p-s_q)]

L_A     = B(s_A,m)
L_TA    = B(s_TA,m)
L_NT    = mean softplus(s_Tx)

L_common = L_clean + L_A + L_TA + 0.25*L_NT
~~~

The anti-absorption term is a gate loss, not C1's scalar
(l_TA-l_Tx)-(l_A-l_x) consistency loss:

~~~
L_AA = 0.5*B(a_A,m) + 0.5*B(a_TA,m)
       + 0.25*mean[softplus(a_x) + softplus(a_Tx)]
~~~

The arm objectives are:

~~~
A0, A1, A2: L = L_common
A3:          L = L_common + L_AA
~~~

There is no LT, LATcons, score-contrast consistency, hyperparameter sweep,
or arm-specific coefficient. L_AA supervises whether the full-feature
nuisance-residual branch itself remains active on the synthetic anomaly mask
and inactive on clean/shifted-clean maps. This is the anti-absorption
addition; it is not a rebranding of C1's existing contrast-invariance term.

Empty mask/background, nonfinite feature/algebra values, and nonconstant
outputs fail closed.

## 5. Optimization and deterministic cell schedule

Use exactly 576 updates per object/seed/arm:

~~~
optimizer=AdamW
lr=3e-4
weight_decay=1e-4
betas=(0.9,0.999)
eps=1e-8
batch_size=1
gradient_clip=1
scheduler=none
~~~

The deterministic 144-cell schedule is all
12 R_train bases x 3 generators x 4 gains, repeated four times in the
same order for every arm and paired seed. The four views x, T_e x, A_g x,
T_e A_g x are generated from the same source cell. Synthetic seeds are
derived from the new protocol ID, object, scientific seed, base, donor,
generator, gain, and role. No GT is opened.

## 6. Exact inference

For each public image:

1. Apply the canonical object resolution (672, or 448 for sheet_metal) and
   one frozen DINO forward at layers [5,11,17,23].
2. Query top-8 support patches from all 16 same-manifest references per layer
   and form r_l, u_l, u_l_perp, and c_l with the frozen sigma_l,U_l.
3. Build q_raw, q_perp, and c; run the selected arm's fixed route.
4. Restore the logit map to native image size with canonical bilinear
   interpolation. Do not query-time adapt, estimate a new nuisance basis, or
   use test GT.
5. Stream the map and native GT into the canonical segmentation accumulators
   only in the GT pass. Use the map logit directly; no post-hoc score fusion.

Candidate arms share one backbone forward and one feature query stream per
image. SuperAD and RN-FMLK remain independent same-run comparators. No dense
native map is written.

## 7. Evaluation and result schema

For each object x seed x arm, emit:

~~~
protocol_id
method_id
object
scientific_seed
reference_manifest_sha256
reference_count
train_reference_indices
test_population
backbone
feature_layers
method_<metric>
superad_<metric>
rn_fmlk_<metric>
delta_vs_superad
delta_vs_rn_fmlk
comparable
map_coverage
finite
nonconstant
prediction_hash
~~~

<metric> is each of seg_AUROC_0.05, oracleF1, and AUPRO.05. Report
per-seed rows first, then per-object macro mean +/- standard deviation over
paired seeds, then the all-eight-object macro mean. Never pool seeds or
pixels across seeds. oracleF1 is test-GT-only reporting and cannot select
an arm.

## 8. Storage and route

Use dsba1 host GPU 6 only, exposed as CUDA slot 0; verify UUID binding before
the production matrix. Remote availability may be audited in parallel. There
is one route verification followed by the full matrix; weak metrics do not
trigger a redesign or a second pilot.

Persist only:

- compact feature/nuisance cache with exact byte cap and hash manifest;
- configs, source hashes, reference manifest, logs, prediction hashes;
- streaming metric summaries and per-object/per-seed rows.

Never persist anomaly_maps/, NPY/TIFF maps, or full-resolution map lists.
Remove only new regenerable cache/output trees after the conclusion, retaining
compact evidence.

## 9. Code ownership recommendations

These are implementation assignments, not edits in this design step:

- src/flow_tte/shift_residual_ablation.py: P, support residual, nuisance
  basis, branch switches, exact losses, and head forward.
- src/flow_tte/shift_residual_cache.py: train-only feature streaming, top-8
  support queries, robust scale/SVD, bounded cache manifest.
- src/flow_tte/c1_evaluation.py or a new adapter: one-pass candidate query
  sharing and native map restoration; keep existing C1 paths untouched.
- src/post_eval integration: reuse StreamingSegmentationAccumulator;
  preserve truthful labels and per-object/per-seed aggregation.
- tests/test_shift_residual_ablation.py: shape/parameter, split custody,
  deterministic seed, disabled-switch, finite/nonconstant, and inference
  parity tests.
- Remote controller lane: dsba1 GPU6 binding, one route verification, full
  matrix launch, and compact pullback only.

The source implementation must not inherit C1 checkpoint/cache hashes.
Independent SuperAD-16 and RN-FMLK baseline artifacts remain separate.
