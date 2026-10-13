# ECO-G2 Native-Core Stage -1 preregistration

Date: 2026-08-28  
Protocol: `eco-g2-nativecore-20260828-v1`  
Status: `PREREGISTERED / RESULT-FREE / N0 BEFORE MODEL`  
Claim tier: support-only mechanism diagnostic; `benchmark_comparable=false`; `SuperAD=N/A`

## Question and stop condition

ECO-Audit v12 is complete and remains `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.
Its required pooled `ACTIVE_INTERIOR` had counts `0,0,1,0` across the four
held bases, while the independent J0/G2 operator audit was unbounded in all
32 cells. This new, separately named experiment asks only whether the missing
strict-core population exists and is feature-active on the frozen block-11
native lattice before ECO's explicit `48x104 -> 16x16` spatial pooling.

The run has two dependent gates:

1. `ECO-G2-N0A` is data/geometry-only. If any of the exact 16 support or four
   held masks has fewer than four `AREA_NATIVE_STRICT_CORE` cells, or any
   geometry/custody/recomputation check fails, execution stops before opening
   the model or a CUDA context.
2. `ECO-G2-N1A` runs only after a 20/20 N0 pass. If every one of the eight
   held base/readout records has at least four active native-core rows, global
   active target-energy coverage at least `0.95`, raw/standardized pass
   agreement, and an exact held-0 repeat, the result is
   `PROMISING_DIAGNOSTIC / OPERATOR_ONLY_SENTINEL_AUTHORIZED / NO_LORA`.
   Otherwise this native observation line closes.

No outcome in this experiment can revise the ECO v12 operator failure, open
LoRA, authorize AGIAS, or support detector/benchmark claims.

## Frozen inputs and lineage

- Dataset boundary: MVTec AD 2, `can/train/good` only, inside the fixed dsba3
  project container at `/home/hunim/Volume/DATA/mvtec_ad_2`.
- Support: the exact SuperAD-selected 16 paths from manifest SHA-256
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`;
  pathset SHA-256
  `4727e11fd630c49372835a64cf9efbeea3ca791427da7d0fce92167eef5c20da`.
- Support rows/donors are byte-bound to
  `../2026-08-27_eco_audit_v2_stage_minus1/support_calibration_commitment.json`
  (SHA-256 `66708c38671ceeb09837f5471669f490af163b0d978b6013063262f9102a7091`).
- Held rows/donors are byte-bound to
  `../2026-08-27_eco_audit_v2_stage_minus1/held_commitment.json`
  (SHA-256 `b51b6ca946b616fbc2578eaf1b20e54f2f905a52eb51deb9799abf715e5f7460`;
  pathset SHA-256
  `c1d648d70f0c8479731409cb6324f38241368436cf4533550a3f950b91259dfd`).
- Generator is exactly `G2_structural_scratch`.
- To reproduce the old masks rather than silently create a new intervention,
  support mask/synthesis seeds retain the ECO-v12 namespace
  `eco-audit-v2-ncfc-attribution-20260827-v1`; held mask/synthesis seeds retain
  the NCFC-v2 namespace
  `ncfc-s1-normal-coordinate-frozen-commutation-20260827-v2`.
- The exact v12 mask/synthesis seeds and source-mask/base/donor/anomaly hashes
  are additionally frozen in `v12_lineage_commitment.json` (SHA-256
  `903b26bc16cee7aab5d59c4bad034de90afc2d616ee73fe278163b0c19205b28`).
  N0 must reproduce all non-null lineage fields before its geometry verdict.
- N0 may decode only the committed 20 base train-good images needed for the
  original shape. Donor identities are checked but donor pixels need not be
  decoded. It cannot import/load the model or initialize CUDA.
- N1 may decode only the committed base/donor train-good images required to
  construct `x` and `A(x)`. Public/private test, ground truth, query images,
  filenames as condition labels, and any query update remain forbidden.

## N0A: exact area-native strict core

For an original boolean mask `G` with shape `H x W`, compute the realized
torchvision integer-resize dimensions `H_r x W_r` for shorter edge 672, then
crop bottom/right remainders for native grid
`g_h=floor(H_r/14)`, `g_w=floor(W_r/14)`.

For native row `r`, source row `i`, native column `c`, and source column `j`:

`L_y(r,i)=max(0,min(14(r+1)H,(i+1)H_r)-max(14rH,iH_r))`

`L_x(c,j)=max(0,min(14(c+1)W,(j+1)W_r)-max(14cW,jW_r))`

`N_rc=sum_ij G_ij L_y(r,i)L_x(c,j)`, with denominator `D=196HW`.
All arithmetic is signed int64. The projected native source bit is exactly
`2*N_rc >= D`; there is no tolerance, margin, interpolation choice, fallback,
image drop, resolution change, or mask enlargement. Apply center plus all
eight neighbours erosion with out-of-grid false. `core=erosion(source)`,
`boundary=dilation(source)-core`, and `outside=not dilation(source)` must be
disjoint and exhaustive.

For every mask, a patch-loop reference and separable integer-matrix
implementation must agree bit-for-bit. Bind original/source-mask hashes,
shapes, seeds, resize/crop/native-grid dimensions, int64 numerator hash,
row-major little-bit-order native source/core/boundary/outside bitsets, and
counts. N0 passes only at `20/20 core_count>=4` with exact commitment lineage.

### Frozen nondecision sensitivity

For audit only, resize a contiguous `{0,1}` float32 tensor with
`torchvision.transforms.functional.resize(size=672,
InterpolationMode.BICUBIC, antialias=True)`, clamp to `[0,1]`, crop the same
bottom/right remainders, average each `14x14` patch in float64, threshold
`>=0.5`, and apply the same erosion. Record only total projected support mass,
support expansion outside the area-primary source bitset, and core-count
difference. This tensor-backend sensitivity can neither determine N0 nor
replace/rescue the exact area primary.

## N1A: frozen native activity

- Backbone: frozen DINOv2 ViT-L/14, shorter edge 672, zero-based block 11,
  native patch tokens before spatial pooling; no trainable parameters.
- `RAW_NATIVE`: `model.norm` block-11 patch output cast to float32 before any
  spatial pooling.
- `STANDARDIZED_NATIVE_L2`: for each token, subtract its float32 channel mean,
  divide by population standard deviation floored at `1e-6`, then L2-normalize
  with `eps=1e-8`. It must match
  `pool_normalize_tokens(raw,native_grid,native_grid)` within
  `atol=1e-7, rtol=1e-6` on the frozen fixture.
- Target displacement is `u=R(A(x))-R(x)`. Norms and energy accumulation use
  float64. Activity is inclusive: `||u_k||_2 >= tau_R`.
- Calibrate a separate raw/std left-continuous empirical Q10 from the 16
  support images, restricted only to each image's N0 primary strict-core rows.
  Each image has exact mass `1/16`; each selected row has mass
  `Fraction(1,16*core_count_image)`. Sort ties by
  `(value, lexical_support_id, row_major_index)`. Pooled-v12 taus, held
  recalibration, alternate percentiles, interpolation, and per-image fallback
  are forbidden.
- Held population is exactly four committed bases times two readouts. Each
  record passes only if `ACTIVE_NATIVE_STRICT_CORE>=4` and global active target
  energy divided by total target energy is `>=0.95`. Both readouts must agree
  on pass/fail for each base and all eight must pass for N1 success.
- Repeat held index 0 after the four primary shards. Canonical scientific
  bytes must be identical. No tolerance-based repeat rescue is allowed.

Persist only canonical JSON scalars, hashes, provenance/version fields, and
row-major compact bitsets. Native feature tensors, pixel masks/images,
residual arrays, checkpoints, anomaly maps, optimizer states, and gradients
must never be persisted.

## Explicit exclusions

This run contains no J0 exposure, TA/AT order, correction candidate, N1--N5
comparison, loss, optimizer, gradient, adapter/LoRA, checkpoint, test/public/
private/GT item, query adaptation, query memory, benchmark metric, or favorable
target selection. `SuperAD=N/A` and `benchmark_comparable=false` because the
experiment is a support-only mechanism diagnostic rather than a detector
evaluation.

## Execution and non-interference

- Remote host is fixed by the ignored preset `configs/remote/dsba3.env` and
  must resolve to `hunim@147.47.39.144:2222`.
- Container is the already-running, identity-pinned project container
  `hun_fsad_tta_012`; it must not be created, restarted, stopped, reconfigured,
  or replaced.
- P0/P1/N0 use `CUDA_VISIBLE_DEVICES=''`. Only after N0 PASS may N1 support
  calibration use dsba3 GPU0 and held shards use dsba3 GPUs 0--3, one committed
  index per GPU, followed by held-0 repeat on GPU0.
- Status checks are limited to the exact pinned container, exact experiment
  paths, and aggregate telemetry/UUIDs for authorized GPUs 0--3. The controller
  never enumerates another user's PID, process, identity, container, files, or
  results and never kills or alters any process.
- A busy authorized GPU defers the phase; it does not trigger resource search.
- Snapshot and result roots are fresh and exact. Cleanup may remove only the
  exact experiment snapshot after a terminal compact result has been pulled
  and verified. The compact remote result is retained.

## Frozen verdicts

- N0 invalid or any mask `<4`: `N0_FAIL_CLOSE_EXISTING_NATIVE_ROUTE / NO_GPU`.
- N0 pass, N1 invalid/fail/disagreement: `N1_FAIL_CLOSE_NATIVE_OBSERVATION_LINE`.
- N1 pass: `PROMISING_DIAGNOSTIC / OPERATOR_ONLY_SENTINEL_AUTHORIZED / NO_LORA`.
- Any custody, forbidden-access, persistence, schema, hash, or repeat failure:
  `INVALID_FAIL_CLOSED / NO_CONTINUE`.

No threshold or method term may be changed after N0/N1 results. Any later
operator-only sentinel is a separately named and separately preregistered run.

## Result-free implementation amendment placeholder

Before P0, append the exact source-manifest hash, runner/core/aggregator/
verifier/remote-controller hashes, static-contract hash, test targets and
environment, production schema IDs, and immutable snapshot rule. Any source
change after that freeze requires a new run name and a result-free amendment.

The frozen model is additionally bound to ECO-v12 model-state SHA-256
`59a755e385188971c57f52a38773028b508722c804ce6c9279d1dc2ac14ccbe8`.
The prior compact support-seal artifact SHA-256
`99d01623dcc2f9c072290ffa95a0098c2cc30fd373f12912e2c2a9ded5e45f61`
is lineage evidence only; its pooled taus are explicitly forbidden in N1A.
