# ECO-O1 lifted-J0/G2 operator sentinel preregistration

Date: 2026-08-28  
Protocol: `eco-o1-lifted-j0-g2-operator-sentinel-20260828-v1`  
Generation: `eco-o1-g0`  
Parent: `eco_g2_nativecore_dsba3_20260827T212434Z_g2v5`  
Status: `PREREGISTERED / RESULT-FREE / SINGLE SENTINEL / REVISION_BUDGET=0`  
Claim tier: operator-only support diagnostic; `benchmark_comparable=false`;
`detector=false`; `SuperAD=N/A`; `LoRA=false`

## Question and hard stop

ECO-G2 v5 passed its unchanged N0/N1 contract: all 20 masks had a native
strict core, all eight held/readout targets were active with at least 0.95
energy coverage, and the exact held-0 repeat was byte-identical. This opens
exactly one operator-only sentinel. ECO-Audit v12 independently showed that
the legacy uint8 J0/G2 composition was noncommuting in every cell.

This sentinel changes one thing only: the J0/G2 pair is rendered from one
shared latent G2 blend, with J0 transported through both the base and scratch
carrier and a single final quantizer. It asks whether removing the known image
composition asymmetry also yields a valid native-feature residual that is not
absorbed by the exact legacy fixed-carrier `A(T(x))` hard null.

The stop is unconditional:

- pass opens a separately preregistered pixel-map detector bridge;
- valid scientific fail closes the ECO family;
- invalid/custody/repeat fail closes the ECO family;
- no second operator, geometry, activity, support-only, feature-only, or
  threshold rescue is permitted in this generation.

## Frozen parent and custody

- ECO-G2 aggregate:
  `results/remote_runs/dsba3/eco_g2_nativecore_dsba3_20260827T212434Z_g2v5/aggregate/aggregate.json`,
  SHA-256 `cfe02673d4a5001acafd4c54efb25705a0dad9e387ee666c927e5eb525795b46`.
- ECO-G2 N0 geometry SHA-256:
  `df8404a51aef2bb6148bb75d95b548b6debbf23c35d517a8da91016774b0932e`.
- ECO-G2 N1 support-seal SHA-256:
  `9eb369ddecc8b9587e93512fc44f68bb2e1bd5466e6790933afac5fd1d50f81b`.
- ECO-v12 held aggregate SHA-256:
  `e5c70cdcef737fe37bfe6f4dd14dd4971bb7f929d652c622ea864e032534e78e`.
- Held commitment SHA-256:
  `b51b6ca946b616fbc2578eaf1b20e54f2f905a52eb51deb9799abf715e5f7460`.
- G2 v12 lineage SHA-256:
  `903b26bc16cee7aab5d59c4bad034de90afc2d616ee73fe278163b0c19205b28`.
- Frozen DINOv2 state SHA-256:
  `59a755e385188971c57f52a38773028b508722c804ce6c9279d1dc2ac14ccbe8`.
- Post-G2 policy SHA-256:
  `95c0a83b4aeee1a7701661ab9c78c52f91ae4fd85ddf4e20ef9aaad6f6a32239`.

Population is exactly the committed four `can/train/good` held bases and their
cyclic donors, both J0 signs, and both frozen readouts. Support-16 is used only
to reproduce the already frozen ECO-G2 Q10 support seal. Support/held paths,
donors, seeds, masks, native core bitsets, model, preprocessing, block, feature
dimension, activity equality, and tau estimand cannot be changed.

The runner must reproduce the parent support-seal hash and all eight parent
target hashes before making a decision. This is exact replay, not a new
calibration. Pooled-v12 tau, a different percentile/interpolation, held
recalibration, row fallback, target drop, or sign/readout selection is invalid.

## The single operator change

Let `X=float32(x)/float32(255)` be the normalized committed uint8 base. The
scratch operands are reconstructed by the exact frozen G2 operation sequence:

1. instantiate `numpy.random.default_rng(synthesis_seed)`;
2. decode the donor, normalize it to float32, and if needed resize it to the
   base shape with `cv2.INTER_CUBIC` exactly as the frozen function does (G2
   does not otherwise consume donor pixels, but the custody path is retained);
3. compute `alpha=cv2.GaussianBlur(mask.astype(float32),(0,0),
   max(1.0,min(H,W)/120))[...,None]`;
4. set `C=X.copy()`; if the mask is nonempty, let
   `p0=(xs.min(),int(ys.mean()))`,
   `p1=(xs.max(),int(ys.mean()+rng.integers(-4,5)))`, draw color
   `tuple(float(v) for v in rng.uniform(.05,.95,3))`, and call
   `cv2.line(C,p0,p1,color,max(2,min(H,W)//100))` with the frozen OpenCV
   default line type.

The frozen CIDER source hash is bound in the source manifest, and this
reconstruction must reproduce its anomaly bytes before P0 can open the
result-bearing phases. Define the exact float32 blend

`B(X,C,alpha) = X*(1-alpha) + C*alpha`

and the sole quantizer

`Q32(z) = uint8(rint(clip(asarray(z,float32),0,1)*float32(255)))`.

At identity, `A0=Q(B)`. `A0`, the base, donor, source mask, mask seed,
synthesis seed, and source shape must reproduce the frozen v12 lineage hashes.

For sign `sigma in {-1,+1}`, let

`s_sigma = float32(2 ** (sigma*0.1333))`.

The canonical candidate prequant tensor and output are

`P_star = asarray(s_sigma * asarray(B(X,C,alpha),float32),float32)`

and `Y_star=Q32(P_star)`. Both logical composition labels
`Y_TA_star` and `Y_AT_star` must consume this same canonical `P_star` and must
have identical output hashes. Separately, the distributive identity

`s*B = (s*X)*(1-alpha) + (s*C)*alpha`

is audited in float64 before quantization. Its maximum absolute residual must
be at most
`8*finfo(float32).eps*max(1,max(abs(left)),max(abs(right)))`. Here the
float32 epsilon accounts for the canonical `P_star` rounding that is compared
against the float64-expanded right-hand side. This algebraic
audit and the byte-identical logical labels are implementation checks, not
independent scientific evidence. Neither operand may be clipped, rounded,
resized, or converted to uint8 before the shared final `Q32`. Exposure has no
spatial action, so `alpha` is unchanged. There is no new EV,
mask, line, thickness, color draw, generator, interpolation, image subset,
severity, learned parameter, or sweep.

## Anchor, identity, and hard null

- `I0_IDENTITY`: use `s=1` in both lifted paths. Both outputs must be
  bit-identical to the frozen `A0`; feature residual hash is the target hash,
  SRE is exactly zero, cosine is exactly one within the canonical float64
  serializer, and norm ratio is exactly one.
- `C0_FROZEN_TA_ANCHOR`: the legacy
  `T(A(x)) = Q(s_sigma * Q(B)/255)`. This is the correct-side reference/anchor,
  not a hard null. The lifted candidate may differ from it by at most one
  uint8 level per channel, the analytic maximum introduced by removing one
  intermediate round at the frozen J0 magnitude. Its feature metrics are
  reported but cannot define the hard-null envelope.
- `H0_LEGACY_FIXED_CARRIER_AT`: the unchanged legacy `A(T(x))`. First form
  `X_sigma_uint8 = frozen_apply_exposure(x,sigma*0.1333)`, then call exactly
  `frozen_cider.synthesize(X_sigma_uint8, donor_uint8, mask_bool,
  "G2_structural_scratch", synthesis_seed)`. This reconstructs `C` from the
  exposed base and redraws the same untransported absolute scratch color. It is
  not an interpolation of the original unexposed carrier with the exposed
  base. Frozen donor decode/resize, RNG consumption, alpha, endpoints,
  thickness, color, OpenCV behavior, clip, and quantizer are therefore reused
  byte-for-byte. This is the sole mechanism-matched hard null and its canonical
  image hash is persisted.

Including `C0_FROZEN_TA_ANCHOR` in a minimum-SRE hard-null envelope is
forbidden: it is precisely the output the lifted construction is intended to
preserve up to removal of one intermediate quantizer, and treating it as a
null would make the scientific contrast ill-posed. Identity is an
implementation/no-op check, not a competitor. No random, sign-selected,
linear-light, clipping, inverse-exposure, or one-factor rescue arm is allowed.

## Observations and decision cells

The eight image/operator records are `4 held x 2 signs`. The frozen DINOv2
readouts expand these to exactly `16` decision cells:

- `RAW_NATIVE`: block-11 native patch tokens after frozen `model.norm`;
- `STANDARDIZED_NATIVE_L2`: the unchanged per-token standardization and L2
  normalization from ECO-G2.

For each held/readout target, let

`u = R(A0) - R(X)`,

`v_star = R(Y_star) - R(X_sigma)`,

`v_anchor = R(C0_FROZEN_TA_ANCHOR) - R(X_sigma)`,

`v_null = R(H0_LEGACY_FIXED_CARRIER_AT) - R(X_sigma)`,

where `X_sigma` is the exact frozen `apply_exposure(x,sigma*0.1333)` image.
`Y_star` is the common byte-identical lifted TA/AT output. All vector norms,
inner products, and energy sums use float64.

Two selections are mandatory and neither may replace the other:

- `ACTIVE_GLOBAL`: rows satisfying the frozen target activity equality
  `||u_k||_2 >= tau_R`;
- `ACTIVE_NATIVE_STRICT_CORE`: `ACTIVE_GLOBAL` intersected with the exact
  parent `AREA_NATIVE_STRICT_CORE` bitset.

For each selection compute

`cos(u,v) = <u,v> / (||u||*||v||)`,

`SRE(u,v) = 2*||v-u|| / (||u||+||v||)`,

`norm_ratio(u,v) = ||v||/||u||`,

and the hard-null margin

`delta_vs_hard_null = SRE(u,v_null) - SRE(u,v_star)`.

Persist both anchor and null metrics, but only `H0` defines
`delta_vs_hard_null`.

Pixel quantities use HWC arrays normalized to float64 by division by `255.0`.
For a boolean region `M`, flatten all selected pixels and RGB channels before
the L2 norm. `ALL` is the complete image. `V12_PIXEL_BOUNDARY` is exactly
`eco_audit_v2.pixel_regions(source_mask,grid=(16,16))["boundary"]`; it is
retained only for direct operator comparability and does not replace the native
feature strict core. Define

`clean_action_norm = ||X_sigma-X||_2`,

`pixel_anomaly_norm_ratio = ||Y_star-X_sigma||_2 / ||A0-X||_2`,

and for outputs `a,b` with common transformed clean `t`,

`kappa_M(a,b;t) = 2*||(a-b)_M||_2 /
                   (||(a-t)_M||_2+||(b-t)_M||_2)`.

Any denominator `<=7.62939453125e-6` is undefined and invalid. Candidate kappa
uses `(Y_TA_star,Y_AT_star;X_sigma)`. Legacy kappa uses
`(C0_FROZEN_TA_ANCHOR,H0_LEGACY_FIXED_CARRIER_AT;X_sigma)`. A legacy
commutator is nonzero only if both its denominator and
`||C0-H0||_2` exceed that epsilon. A pixel output is nonconstant only if its
float64 peak-to-peak range is strictly positive; a candidate anomaly is
nondegenerate only if both numerator and denominator of the norm ratio exceed
the epsilon. Nonfinite input or result is invalid.

## Frozen gates

All eight image records and all 16 feature cells must be valid and pass. There
is no averaging, majority rule, favorable sign/readout, row/cell drop, or
post-result threshold change.

### Image/operator gates, all eight records

- frozen base/donor/mask/seed/G2 anomaly lineage: exact;
- clean J0 image: byte-identical to frozen `apply_exposure`;
- identity lifted output: byte-identical to frozen `A0`;
- lifted logical TA/AT hashes: byte-identical because they consume the same
  canonical `P_star`;
- float64 distributive residual: within the dtype-derived bound above;
- lifted pixel commutator energy and kappa: exactly zero as a construction
  sanity check, not independent scientific evidence;
- candidate and clean outputs: finite and nonconstant;
- clean action norm: `> 7.62939453125e-6`;
- candidate anomaly residual norm ratio
  `||Y_star-X_sigma||_2/||A0-X||_2`: `>=0.90`;
- candidate-versus-anchor maximum absolute difference: `<=1` uint8 level;
- legacy hard-null commutator status: valid and nonzero;
- `legacy_kappa_all - candidate_kappa_all >=0.01` and
  `legacy_kappa_boundary - candidate_kappa_boundary >=0.01`.

Preclip violation and boundary error share are persisted as diagnostics. They
are not independent vetoes for the lifted renderer because both lifted orders
share the same single final clip; exact TA/AT equality, anomaly norm retention,
and native-feature gates are the decision tests. This does not alter the v12
or ECO-G2 verdicts or thresholds.

### Native-feature gates, both selections in all 16 cells

- parent target hash: exact;
- active-global target-energy coverage: `>=0.95`;
- active-native-strict-core row count: `>=4`;
- selected target/candidate/null norms: finite and greater than
  `7.62939453125e-6`;
- candidate cosine: `>=0.90`;
- candidate SRE: `<=sqrt(0.2)=0.4472135954999579`;
- candidate norm ratio: `>=0.90`, with no upper bound;
- `delta_vs_hard_null >=0.01`;
- identity target hash exact, SRE exactly `0`, cosine exactly `1`, norm ratio
  exactly `1`.

The absolute cosine/SRE/norm/coverage/count thresholds and the `0.01` SRE
margin are inherited from the frozen ECO-v12/ECO-G2 contracts. A cosine margin
is not added because the prior null cosine was already near one; SRE is the
predeclared separating metric.

### Repeatability

After the four primary held shards, repeat held index 0 on the same authorized
GPU assignment. Canonical scientific bytes for both signs/readouts, including
operator hashes and all scalar metrics, must be byte-identical. Tolerance-based
repeat rescue is forbidden.

## Phases and remote safety

1. `P0`: pure CPU unit/adversarial/schema tests; no data/model/CUDA access.
2. `P1`: pure synthetic-array analytic/identity/hard-null dry run twice with
   byte-identical canonical JSON; no data/model/CUDA access.
3. `C0`: exact parent artifact/hash verification and exact frozen support-seal
   replay. No held access before the support-seal hash matches.
4. `HELD`: held indices 0--3 map one-to-one to dsba3 GPUs 0--3. Each container
   process sees only its assigned device as `cuda:0`.
5. `REPEAT`: held index 0 repeats on GPU0.
6. `AGGREGATE/VERIFY`: one fail-closed terminal decision after every shard.

Remote host is exactly `hunim@147.47.39.144:2222` from the ignored dsba3
preset. Container identity is the already running pinned project container
`hun_fsad_tta_012`; it cannot be created, restarted, stopped, reconfigured, or
replaced. Only authorized GPUs 0--3, the exact container, and exact experiment
paths may be checked. The controller must never enumerate or alter another
user's PID, process, container, file, result, or GPU allocation. A busy
authorized GPU defers the phase; it never triggers a resource search.

Actual data/model/GPU execution is remote only. Local work is limited to
static source edits, pure no-data tests, compact-result verification, and
reporting. Snapshot and result roots are fresh. Only an exact valid snapshot
may be deleted, and only after the compact result has been pulled and its
receipt independently verified. Invalid snapshots and all compact attempt
ledgers are retained.

## Persistence and forbidden access

Allowed persistence: canonical JSON scalars, hashes, compact bitsets,
provenance/version/source identities, timing/memory, and exact access ledger.

Forbidden: images, pixel masks, native feature/residual tensors, anomaly maps,
scores/maps for TESTpublic, checkpoints, optimizer state, gradients, adapters,
LoRA, query images, query updates, public/private test items, GT, or filenames
used as condition labels. No training or parameter update occurs.

## Attempt and revision policy

`scientific_revision_budget=0`. After any result-bearing held output, the
operator, arms, population, thresholds, metric, code path, and decision rule
cannot be changed or rerun as the same generation except for the exact required
held-0 repeat. A pre-result instrumentation/runtime failure may be repaired
only if it emitted no data/model-derived scientific metric, uses a fresh
immutable run name, preserves this preregistration byte-for-byte, and remains
in the complete attempt ledger. Result-bearing invalidity consumes the
sentinel and closes ECO.

## Terminal verdicts

- Every custody/operator/repeat check and all `16/16` cells pass:
  `PROMISING_DIAGNOSTIC / DETECTOR_BRIDGE_AUTHORIZED / NO_SECOND_SENTINEL / NO_LORA`.
- Structurally valid result but any image, absolute feature, or hard-null
  separation gate fails:
  `VALID_OPERATOR_SENTINEL_NO_SEPARATION / END_ECO_FAMILY`.
- Hash, schema, finite/nonconstant, identity, analytic equality, persistence,
  repeat, forbidden-access, or custody failure:
  `INVALID_FAIL_CLOSED / END_ECO_FAMILY`.

Passing is not a detector, benchmark, or LoRA result. It authorizes only the
already mandated separately preregistered detector bridge. That bridge must
emit a pixel map and then either enter the frozen three-object metric checkpoint
or close ECO; another support-only diagnostic is forbidden.

## Result-free implementation seal placeholder

Before remote P0, append a separate immutable implementation amendment with
the exact source-manifest hash, runner/core/aggregator/verifier/controller and
test hashes, static-contract hash, parent-artifact hashes, schema IDs, test
commands/results, snapshot rule, and environment identity. Any source change
after sealing requires a fresh pre-result run name and a new result-free
amendment; no scientific term above may change.
