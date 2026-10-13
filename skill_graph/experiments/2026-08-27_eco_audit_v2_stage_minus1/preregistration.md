# ECO-Audit v2 Stage -1 preregistration

Date: 2026-08-27  
Protocol: `eco-audit-v2-ncfc-attribution-20260827-v1`  
Status: `PREREGISTERED / NO HELD ACCESS / NO TRAINING`  
Claim tier: support-only frozen measurement audit; `benchmark_comparable=false`; `SuperAD=N/A`

## Question and immutable boundary

NCFC-S1 remains completed with verdict
`NORMAL_ONLY_NO_ANOMALY_SIGNAL / END_NCFC`. ECO-Audit v2 does not revise that
verdict and cannot promote its normal-only signal. It asks one narrower question:

> Is the frozen residual-commutation failure primarily an inactive-row metric
> component, image-operator noncommutation, boundary spillover, coordinate-only
> drift, or a genuine active local mismatch?

This is one separately named no-training attribution experiment. It cannot
claim detector improvement, real acquisition-shift generalization, deployable
canonicalization, or LoRA safety/necessity. No adapter, head, decoder, optimizer,
gradient, checkpoint, public/private test item, ground truth, query adaptation,
or query-memory update is permitted. A valid genuine mismatch may open only the
separately preregistered AGIAS Gate B identifiability assay; it does not authorize
backbone training.

## Frozen source experiment

- Object: `can` only.
- Dataset location: dsba3 project container path
  `/home/hunim/Volume/DATA/mvtec_ad_2`.
- Support: the immutable SuperAD-selected 16 `can/train/good` paths from manifest
  SHA-256 `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Support pathset SHA-256 remains
  `4727e11fd630c49372835a64cf9efbeea3ca791427da7d0fce92167eef5c20da`.
- The exact 16 lexical support IDs and cyclic donors are frozen in
  `support_calibration_commitment.json`; the P2 manifest must match that file
  byte-for-byte by source hash before encoding.
- Held commitment: exactly the four relative IDs and cyclic donors in
  `held_commitment.json`; pathset SHA-256
  `c1d648d70f0c8479731409cb6324f38241368436cf4533550a3f950b91259dfd`.
- Held files are not resolved, opened, decoded, or encoded before the P2
  pre-held manifest is sealed and byte-identically reloaded.
- Generator: `G2_structural_scratch` only.
- Actions: J0 exposure at exactly `-0.1333 EV` and `+0.1333 EV`.
- Orders: `T(A(x))` and `A(T(x))`, both evaluated for every sign/base/readout.
- Backbone/readout: frozen DINOv2 ViT-L/14 zero-based block 11, 672-pixel
  smaller-edge aspect-preserving canvas, deterministic pooling to `16x16`,
  `RAW_MODEL_NORM` and `STANDARDIZED_L2` from the same forward path.

The held construction reuses the completed NCFC protocol's base/donor/mask/
synthesis seeds and five input definitions so the old failure is reproduced,
not replaced. Support calibration uses the 16 support IDs in lexical relative-
path order, donor `j -> (j+1) mod 16`, and seeds
`stable_seed(protocol_id, base_relative, G2, "mask")` and
`stable_seed(protocol_id, base_relative, donor_basename, G2, "synthesize")`.
The mask, donor, source image, generator parameters, seeds, pre/post-action
arrays, preprocessing, and source/model identities are hashed into the P2
manifest.

## Single activity estimand

For support image `j`, readout `R`, and deterministic mask-interior token `k`,

`z_jk = ||R(A(x_j))_k - R(x_j)_k||_2`.

Each support image has exact total rational mass `1/16`; each of its valid
interior tokens has mass `Fraction(1, 16*|I_j|)`. Records are sorted by
`(z_jk, lexical_support_id, row_major_index)`. `tau_R` is the first observed
value whose exact cumulative mass is at least `1/10`. There is no interpolation.
Raw and standardized taus are separate and share images, masks, regions, and
ordering. `||u_i|| >= tau_R` is active. Pooled-token Q10, median-of-image Q10,
alternative percentiles, per-cell taus, and held recomputation are forbidden.

Calibration is invalid unless there are exactly 16 images, every image has a
nonempty interior, all values/hashes are finite/valid, every image mass is
exactly `1/16`, and total mass is exactly one.

## Geometry and pixel operator

Token `(r,c)` owns the half-open source-pixel footprint
`[floor(rH/16),floor((r+1)H/16)) x [floor(cW/16),floor((c+1)W/16))`.
The token mask is true iff the boolean footprint mean is `>=0.5`. With a
self-plus-8-neighbor grid and out-of-grid false:

- `INTERIOR = erosion(mask)`;
- `BOUNDARY = dilation(mask) - erosion(mask)`;
- `OUTSIDE = not dilation(mask)`.

These regions are disjoint and exhaustive. Pixel regions are exact unions of
the same footprints. For region `P`,

`kappa_P = 2||P(T(A)-A(T))|| / (||P(T(A)-T)|| + ||P(A(T)-T)||)`.

The denominator must exceed `eps_norm`; zero displacement is undefined. The
runner separately computes the pre-clip float exposure, pre-clip violation
rate, post-clip saturation rate, kappa for all/BOUNDARY, spatial commutator
energy, anomaly energy, and squared commutator error-energy share in BOUNDARY.
It bitwise checks that clipping/rounding reproduces the existing J0 uint8
operator. Fixed tolerances (`all .10`, `boundary .20`, `preclip .01`, boundary
energy `.25`) declare only `OPERATOR_BOUNDED_FOR_ATTRIBUTION`; they never prove
operator causality.

## Target records, cells, and diagnostics

There are exactly eight immutable target records `(4 held bases x 2 readouts)`.
For each, compute once
`u = R(A(x))-R(x)`, ACTIVE, regions, S90, N5, and hashes. Every target is
referenced by exactly four cells `(2 signs x 2 orders)`. The cell universe is
exactly 32.

Each cell retains the original NCFC all-token/mask-balanced cosine, residual
norm/top-10/RN fields, N1--N4, custody, and repeat checks. Added fields include
global and activity/region-stratified cosine, SRE
`2||v-u||/(||v||+||u||)` without denominator epsilon after validity, target
energy coverage, norm ratio, interaction-error energy, counts/energies, pixel
operator fields, and N5 margins. Active coordinate success requires cosine
`>=.90`, SRE `<=sqrt(.2)`, target-energy coverage `>=.95`, and existing norm
retention. CKA and token-geometry Spearman use `.90` only for explanatory
labels and cannot rescue coordinate failure.

S90 serializes scalars only: validity, smallest `k` reaching 90% target energy,
achieved coverage, total energy, and cutoff energy. No index or mask is retained.
S90 cannot influence ACTIVE, tau, status, label, reducer, branch, or training.

N5 partitions `u` by active/inactive x interior/boundary/outside. A stratum of
size at least two receives a deterministic one-row cyclic roll; a singleton
receives a protocol-hashed nonzero channel roll. Membership, row-norm multiset,
and energy must be preserved while the residual changes above `eps_norm`.
Required comparison strata need at least four rows. True active cosine must
exceed N5 by `.01`, and N5 SRE must exceed true SRE by `.01`; otherwise the
dependent cell is invalid.

Every stratum records count, target/candidate energy, and exactly one status:
`INVALID_NONFINITE`, `EMPTY_STRATUM`, `INSUFFICIENT_ACTIVE_SUPPORT`,
`INVALID_SMALL_STRATUM`, `UNDEFINED_ZERO_TARGET`,
`UNDEFINED_ZERO_CANDIDATE`, `UNDEFINED_CONSTANT_GEOMETRY`, or `VALID`.
Required invalid/undefined fields fail closed; no imputation or omission exists.

## Frozen reducer and handoff

Cell precedence is invalid, operator-confounded, original all-token pass,
low-activity component, boundary spillover, coordinate drift, genuine local
mismatch, then unresolved mixture. Program reduction is stricter:

1. any custody/metric/RN/N1--N5/repeat invalidity ->
   `UNRESOLVED_OR_INVALID`;
2. raw/standardized disagreement -> `UNRESOLVED_READOUT`;
3. TA/AT disagreement -> `UNRESOLVED_ORDER`;
4. any operator-confounded cell -> `OPERATOR_CONFOUNDED`;
5. remove only `NO_LOCAL_FAILURE`;
6. no remaining labels -> `NO_REPRODUCED_FAILURE`;
7. identical remaining label -> that label; otherwise `UNRESOLVED_MIXTURE`.

No mean, median, majority, favorable order/readout, S90, alternate tau, CKA,
correction cosine, or RN retention may override the reducer. Only a valid,
repeatable `GENUINE_LOCAL_MISMATCH_CANDIDATE` under bounded operator and full
raw/std + order agreement opens AGIAS Gate B. All other outcomes close or
redirect the branch exactly once as specified in `static_contract.json`.

## P0/P1/P2/held execution gate

- P0: inside fixed dsba3 container with `CUDA_VISIBLE_DEVICES=""`; pure contract,
  adversarial status, geometry, Q10, manifest, S90, N5, and serializer tests.
  No dataset, model, held, CUDA, or training access.
- P1: same remote CPU boundary; production synthetic dry run must emit exactly
  eight target records and 32 cells twice with byte-identical hashes, exercise
  every reducer label, and retain zero access counters.
- P2: only container-visible GPU0; encode the 16 support calibration pairs,
  create `tau_raw/tau_std`, seal/reload the pre-held manifest, verify frozen
  state and exact repeat, and prove held access remains zero.
- Held: only after independent prelaunch approval and P0/P1/P2 hash-chain pass.
  Use dsba3 GPU0--3, one held index per GPU, then repeat held index 0 on GPU0.
  Every shard emits eight cells and two target records. Aggregate only compact
  JSON/log/hash/status evidence.

The remote target is fixed to `hunim@147.47.39.144:2222`, container
`hun_fsad_tta_012`. Snapshot and result roots are derived from one safe run name.
PYTHONPATH contains only the immutable snapshot, `src`, and `scripts`. The
controller may read aggregate memory/utilization/UUID mapping for the authorized
GPUs but must never enumerate another user's PID, identity, process, container,
file, or resource; it must never kill, stop, reconfigure, or switch to another
user's resource. A busy authorized GPU defers the phase without resource search.

Exact repeat requires scientific-signature identity and maximum scalar
difference `<=1e-7`. Any failure stops before the downstream phase. Dense
arrays, images, masks, residual tensors, anomaly maps, and checkpoints are
forbidden. Compact outputs remain under the experiment's own result root;
cleanup may remove only the exact immutable snapshot after terminal status.

## Pre-execution amendment rule

Implementation may not change the scientific contract. Before P0, append the
frozen source-file SHA-256 manifest, static-contract SHA-256, held-commitment
SHA-256, support-manifest SHA-256, remote-controller SHA-256, and production
schema identities. After these hashes are appended, code changes require a new
run name, new source manifest, and a documented result-free amendment. Held
results may never motivate a contract change within this experiment.

## Result-free pre-P0 implementation amendment

This amendment was frozen before P0, before any ECO held-path resolution, and
without result-bearing model or dataset execution. It resolves implementation
terms that were underspecified above; it does not change the experiment family,
thresholds, inputs, cell universe, or reducer.

- Existing active norm retention means the Frobenius residual norm ratio
  `||v_ACTIVE|| / ||u_ACTIVE|| >= 0.90`, equivalently active residual-norm
  reduction `<=0.10`. No upper-bound or post-hoc norm threshold is added.
- `inactive_retains_deficit` means the inactive block has status `VALID` and
  fails the same coordinate predicate. An empty, small, zero, or nonfinite
  inactive block cannot establish the deficit.
- `boundary_feature_energy` is target energy in `ACTIVE intersect BOUNDARY`
  divided by target energy in `ACTIVE`.
- `boundary_error_share` is interaction-error energy in
  `ACTIVE intersect BOUNDARY` divided by interaction-error energy in `ACTIVE`.
- `boundary_token_share` is the token count in `ACTIVE intersect BOUNDARY`
  divided by the token count in `ACTIVE`.
- `ACTIVE_INTERIOR` and `ACTIVE_BOUNDARY` coordinate strata must both have
  status `VALID` before boundary-versus-local attribution.  An empty, small,
  zero, or nonfinite required stratum makes the dependent cell
  `UNRESOLVED_OR_INVALID`; it is never converted into a failed coordinate gate.
- Only the active-global N5 true/null comparison is decision-required. The six
  activity-by-region N5 strata remain descriptive status and invariant evidence;
  their membership, row-norm multiset, and energy invariants must still hold.
- Exact repeat is one global held-index-0 scientific-signature repeat after the
  four primary shards, not an asserted or separately selected per-cell repeat.
- Support calibration uses the ECO protocol seed namespace. Held construction
  retains the frozen NCFC seed namespace exactly so the completed NCFC failure
  is reproduced rather than regenerated under a new seed.
- Pixel arrays are normalized to `[0,1]`. Pre-clip violation rate concatenates
  `T(x)` and `T(A(x))`, and the bitwise reconstruction check covers both uint8
  outputs. Undefined/nonfinite pixel quantities invalidate the dependent cell;
  a valid quantity beyond a frozen threshold is `OPERATOR_CONFOUNDED`.
- Every forbidden-access and persistence flag is a required typed boolean
  `false` in P0/P1/P2 artifacts, outer phase records, held shards, and the held
  repeat. A missing, null, numeric, or string-valued flag fails closed.
- The exact five P0 test files are source-manifest inputs and are restated with
  their hashes and ordered pytest targets in P0 evidence. Editing a test after
  this freeze invalidates the source manifest before pytest can authorize P0.
- Every primary held shard has an exact scientific-signature schema, canonical
  SHA-256, and top-level equality check. The repeat must equal held index 0's
  scientific signature; the aggregate records hashes for all four shard inputs
  and the repeat input.

## Frozen source identity

This identity was frozen result-free after all pre-P0 amendments and before
remote P0, support encoding, held-path resolution, or held execution.

- source manifest: `source_hashes.json`
- source-manifest SHA-256:
  `0ae79bae35596804c5e6d32b7860f7ab8b1c395600727755d31cfd35df520f7a`
- static-contract SHA-256:
  `6fee9bb3dd1e6c6dfcec786c5d44f0b23b1dd750630c04282b6f4f14cefc4100`
- support-calibration commitment SHA-256:
  `66708c38671ceeb09837f5471669f490af163b0d978b6013063262f9102a7091`
- held commitment SHA-256:
  `b51b6ca946b616fbc2578eaf1b20e54f2f905a52eb51deb9799abf715e5f7460`
- frozen support-manifest SHA-256:
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`
- remote-controller SHA-256:
  `5301bb107881e156a4bf80fd9029635e887dc788668d00149264b56a9a5dafa0`
- production schemas: `eco-audit-v2-static-contract-v1`,
  `eco-audit-v2-stage-minus1-run-v1`,
  `eco-audit-v2-stage-minus1-object-v1`,
  `eco-audit-v2-preflight-chain-v1`, and
  `eco-audit-v2-stage-minus1-aggregate-v1`.

## Post-P0 result-free recovery amendment / Frozen source identity v2

The first immutable attempt,
`eco_audit_v2_dsba3_20260828Tfrozen01Z_v1`, stopped inside P0 before pytest
collection because the fixed container's default `/usr/bin/python3` did not
provide pytest. That attempt observed no dataset, model, support encoding, held
construction, CUDA/GPU result, or scientific metric, and its snapshot and
result roots will not be reused.

The replacement controller creates a new run-scoped, snapshot-local P0-only
test environment from pinned binary wheels: pytest `9.1.1`, pluggy `1.6.0`,
packaging `26.3`, iniconfig `2.3.0`, and pygments `2.20.0`. Its path is added
only to the P0 CPU process. P1, P2, held, repeat, and aggregation do not inherit
this test environment.

The recovery also closes result-free provenance and safety gaps found during
prelaunch review:

- P0 copies the frozen source manifest into its artifact chain, records its
  relative path and hash, and the verifier recomputes source-file hashes and
  compares the exact ordered five-test mapping before accepting pytest output.
- GPU-map validation requires exactly four host and container rows indexed
  `0,1,2,3`, exact per-index UUID equality, and exactly the selected row for a
  single-GPU phase; an empty or malformed query fails closed.
- The static contract, support-calibration commitment, held commitment, and
  frozen support manifest are checked against compiled SHA-256 constants before
  remote execution; the support file is checked again before encoding.
- Host/container snapshot, result, and dataset roots are readonly controller
  constants. Presets must equal the frozen allowlisted paths and cannot redirect
  execution to another tree.
- Required forbidden-access flags remain typed boolean `false`; the active
  interior/boundary validity gates and exact primary scientific-signature chain
  remain unchanged.

This amendment changes execution hardening only. It does not change the data,
operators, thresholds, seeds, target records, cell universe, metrics, reducer,
or decision rule. It was frozen before any result-bearing model/dataset/GPU
execution and before any held-path resolution.

- replacement run identity:
  `eco_audit_v2_dsba3_20260828Tfrozen02Z_v2`
- source manifest: `source_hashes.json`
- source-manifest SHA-256:
  `fa58db935764bd6b37dfb06aa6768f64d609dd7d97cabccc32b52fb4e9679d23`
- remote-controller SHA-256:
  `4c47dc540ffe8f40969a6784dde4ecc1c506961eacf9f16ca589b785d3f35631`
- static-contract SHA-256 (unchanged):
  `6fee9bb3dd1e6c6dfcec786c5d44f0b23b1dd750630c04282b6f4f14cefc4100`
- support-calibration commitment SHA-256 (unchanged):
  `66708c38671ceeb09837f5471669f490af163b0d978b6013063262f9102a7091`
- held commitment SHA-256 (unchanged):
  `b51b6ca946b616fbc2578eaf1b20e54f2f905a52eb51deb9799abf715e5f7460`
- frozen support-manifest SHA-256 (unchanged):
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`

## Result-free external source-anchor amendment / Frozen source identity v5

The v4 identity was never launched remotely. A final result-free adversarial
review showed that a modified `source_hashes.json` could be accompanied by
resealed P0 evidence, the pre-P2 receipt, P2 seal, phase chain, artifact
manifest, outer manifest seal, and chain-derived expected bindings. The v4
standalone verifier therefore proved internal consistency but did not have an
independent source-identity anchor. No dataset, model, support encoding, CUDA
kernel, GPU metric, or held-path result was accessed under v4.

The v5 verifier independently hashes the exact 15 canonical source paths in the
frozen snapshot and requires exact equality with the source manifest. It also
requires canonical, byte-exact JSON serialization. In addition, every remote
P2/preflight/held-reverification/aggregation invocation receives the local
frozen source-manifest SHA-256 as an external launch argument. This trusted
digest is computed from the local frozen manifest at controller invocation and
is never derived from the remote chain or `preflight_expected.json`. The remote
snapshot copy must match it before P0 proceeds, the pre-P2 verifier must match
it before any selected-GPU query or support/model/data access, and the
aggregator must match it before emitting a scientific result.

The exact full-reseal attack is now a regression test: it modifies a source
entry and consistently reseals P0 evidence, the pre-P2 receipt, P2 support
seal, all phase/source/hash bindings, the artifact manifest, outer manifest
seal, and attacker-controlled expected bindings. The verifier must return
`PREFLIGHT_FAIL/NO_LAUNCH` against the original external trusted digest. A
separate regression test rejects semantically valid but noncanonical raw
manifest bytes. The exact five P0 files contain 53 passing pure/static tests.

These changes affect evidence custody only. They do not change the ECO data,
operators, thresholds, seeds, target records, cell universe, features, metrics,
reducer, decision rule, remote GPU set, or scientific protocol ID. The run
remains fixed to `dsba3` GPU 0--3; the separately authorized `dsba2` GPU 6 is
not mixed into this frozen identity.

This v5 identity was frozen before any remote execution, result-bearing
model/dataset/GPU execution, or held-path resolution.

- replacement run identity:
  `eco_audit_v2_dsba3_20260828Tfrozen05Z_v5`
- source manifest: `source_hashes.json`
- source-manifest SHA-256:
  `f67732fbf39866da6da4c8789d091a56990ccf388d877fd3334be1e32578a4c9`
- runner SHA-256:
  `9ebde136be08289530238c02d2b8d01a3db2d801cc7b3a698c8afed940ba44ec`
- aggregator SHA-256:
  `dc94c14e731b9057b411c58c55416b6bfd762dc27dbe29d2e9be894c30cb05e0`
- verifier SHA-256:
  `602ffe823af98451f9c531f45bce828fbc6918d9b8285fd0f7a03f4e0c2bfda6`
- remote-controller SHA-256:
  `9fd2b2df1c7fa526dea97c92640dc92c81dbcb4b420ac8cc2d0b75c3d9ffa301`
- container identity SHA-256 (unchanged):
  `b442f1cb5c6278affb2d5160efb9299d6a57e19fafc8c0972aa707c3f8033c68`
- static-contract SHA-256 (unchanged):
  `6fee9bb3dd1e6c6dfcec786c5d44f0b23b1dd750630c04282b6f4f14cefc4100`
- support-calibration commitment SHA-256 (unchanged):
  `66708c38671ceeb09837f5471669f490af163b0d978b6013063262f9102a7091`
- held commitment SHA-256 (unchanged):
  `b51b6ca946b616fbc2578eaf1b20e54f2f905a52eb51deb9799abf715e5f7460`
- frozen support-manifest SHA-256 (unchanged):
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`

## Held-shard path-expansion correction / Frozen controller identity v12

The v11 P0 gate passed all 58 exact tests, P1 produced byte-identical fixtures,
and P2 produced a valid 16/16 source-footprint support seal and strict preflight
chain. The first held invocation then exposed a controller-only path-expansion
bug. Its remote loop protected `$1` from local heredoc expansion but placed the
result inside single quotes, so all four GPU tasks used the literal shared
paths `held_$1` and `held_$1.log` rather than shard-specific paths. The tasks
raced on one immutable output root. Exactly one incomplete-run held3 artifact
was observed to exist; no scientific field or metric from it was opened or
read. No exact repeat, aggregate, verdict, or complete marker was produced.
The immutable v11 root is abandoned and is never reused.

The v12 controller removes positional `$1/$2` routing from the remote loop.
It iterates an explicit `shard in 0 1 2 3`, maps each shard to the same host GPU
index, and expands that remote variable inside double-quoted distinct output
and log paths `held_${shard}` and `held_${shard}.log`. A controller regression
test requires the explicit loop and double-quoted paths and rejects both prior
single-quoted literal forms. This is a controller-only correction. The v11
source-footprint activity estimand, static contract, 13 scientific sources,
support seal schema, held construction, metrics, reducer, seeds, and dsba3 GPU
0--3 mapping are unchanged.

This v12 identity was frozen before any v12 remote execution and before any
v11 held scientific value was read:

- replacement run identity:
  `eco_audit_v2_dsba3_20260828Tfrozen12Z_v12`
- pinned 13-entry source-manifest SHA-256 (unchanged):
  `d4a3e5e94bf4b00f2b07e9c52feae10ea64b42ca1a38b1366606a042f90d8a7e`
- static-contract SHA-256 (unchanged):
  `299557a46aab3ad5ef688d6f0726054fbb3dd9ea260991d89819edacaf6e6de2`
- launcher root-of-trust controller SHA-256:
  `66ca7f8e09b833eda20a72ee80962bb9ede50f40f288413088da058f8ac8d8a8`
- separately audited controller-test SHA-256:
  `a8bdca69c3f0adfa93d4281a848ba2cc609b58db26ac374eeb5495bd92d619b9`
- container identity SHA-256 (unchanged):
  `b442f1cb5c6278affb2d5160efb9299d6a57e19fafc8c0972aa707c3f8033c68`

## Support-footprint activity estimand amendment / Frozen source identity v11

The v10 P0 and P1 phases completed successfully. Its P2 semantic, source,
container, and GPU0 gates also passed, and the runner entered actual frozen
DINO support encoding. It then stopped before producing a support seal or any
held authorization because the first support image with an empty eroded
interior triggered `support image has no interior calibration tokens`. No held
path, training, gradient, optimizer, query update, dense artifact, tau, or
scientific verdict was produced. The incomplete immutable v10 P2 root is
abandoned and is never reused.

A support-only geometry diagnostic on the frozen v10 snapshot used no model,
GPU, held path, or result-bearing metric. It found that 11 of the 16 committed
G2 ellipses have zero 8-neighbour-eroded interior tokens at the pooled 16x16
grid. All 16 have a nonempty raw tokenized source footprint, with 8--17 source
tokens per image; nonempty interiors contain only 1--2 tokens. This establishes
an estimand/geometry incompatibility rather than a resource or DINO failure.

The v11 amendment changes exactly one scientific definition. For every one of
the same 16 support images, activity calibration now uses every token in the
raw tokenized source footprint `M_j`, where a token is selected when its exact
half-open pixel footprint has boolean source-mask occupancy at least 0.5:

`z_jk = ||R(A(x_j))_k - R(x_j)_k||_2, k in M_j`.

Each image still has exact mass `1/16`, and each selected source token has mass
`Fraction(1,16*|M_j|)`. The exact Q10, lexical/row-major ordering, no
interpolation, raw/std separation, active equality, support identities, donor
mapping, G2 masks, seeds, model, preprocessing, and held construction are
unchanged. There is no image-specific interior-to-source fallback. Empty-
interior images are not dropped and masks are not enlarged.

This is a new preregistered activity estimand, not a v10 implementation repair.
The global protocol/seed namespace is retained so the committed G2 masks do
not change; the new method definition is identified by
`eco-audit-v2-source-footprint-equal-image-q10-v11`. The support seal records
this contract and, for each image, source/interior/boundary/outside token
counts. The independent verifier requires 16 unique support IDs, the same
source-footprint stratum for all images, raw/std image-count identity, exact
`1/16` image masses, total mass one, Q10 witness validity, and exhaustive
geometry counts.

The compact verifier re-derives count consistency, equal masses, and quantile
witnesses from the sealed runner output, but deliberately does not reopen
support images to regenerate masks. Original-mask independence therefore rests
on the exact runner/generator source hashes, deterministic mask seeds, and
recorded source-mask hashes rather than a second data-reading verifier pass.

Held attribution remains strictly unchanged: eroded interior, 8-neighbour
boundary, and outside are still used for the six descriptive strata and the
required `ACTIVE_INTERIOR`/`ACTIVE_BOUNDARY` validity gates. Empty or invalid
held strata still fail closed to `UNRESOLVED_OR_INVALID`; v11 does not relax
that rule. The raw source footprint may contain partial-occupancy boundary
tokens, so its tau can reflect boundary/blur spillover and can enlarge the held
ACTIVE set. That is an explicit interpretation boundary, and v11 must be
judged by its own preregistered active-count/coverage evidence rather than as a
numerically comparable v10 tau.

P2 success is limited to a valid 16/16 support calibration seal and strict
chain verification. It is not a positive method result. Only after that gate
passes may the unchanged four held shards run on fixed dsba3 GPUs 0--3. dsba2
GPU6 is not part of this frozen experiment. SuperAD remains N/A and
`benchmark_comparable=false` because this is an operator/metric attribution
audit, not a detector claim or LoRA authorization.

This v11 identity was frozen before any v11 remote execution:

- replacement run identity:
  `eco_audit_v2_dsba3_20260828Tfrozen11Z_v11`
- activity-calibration contract ID:
  `eco-audit-v2-source-footprint-equal-image-q10-v11`
- static-contract SHA-256:
  `299557a46aab3ad5ef688d6f0726054fbb3dd9ea260991d89819edacaf6e6de2`
- pinned 13-entry source-manifest SHA-256:
  `d4a3e5e94bf4b00f2b07e9c52feae10ea64b42ca1a38b1366606a042f90d8a7e`
- runner SHA-256:
  `3fc26e931dc0a28a4187dc1b5b1f17b7c1aaf0b9362d82f3c6c145214f508634`
- verifier SHA-256:
  `c37c382fc957ce7737e11ca791396521fcbca94810d087d397fbecbe6e90fcdb`
- runner-test SHA-256:
  `a0c064fc6ede914c49b443d727e91fedb2b8e100227a2659e1ab2a8fd5900dca`
- verifier-test SHA-256:
  `53263e5e604ca0ac9cff36464446107b14d688b0bb94328e79e782190d25646a`
- launcher root-of-trust controller SHA-256:
  `b1f5101da5f4835ea991ea55b06d10fb6ff9d22f323f16f6f70d3f9aed3002b4`
- separately audited controller-test SHA-256:
  `9992159f021d1486367ac77626b739bbad80f3a982d878572207f2d451a50d09`
- container identity SHA-256 (unchanged):
  `b442f1cb5c6278affb2d5160efb9299d6a57e19fafc8c0972aa707c3f8033c68`

## P2 lazy-backbone initialization correction / Frozen source identity v10

The v9 P0/P1 phases and P2 semantic/resource gates completed successfully.
The support-seal runner then created the CUDA runtime and DINO backbone object,
but stopped before image decoding or DINO forward because the backbone wrapper
is lazily constructed by `prepare_image`; accessing `backbone._wrapper.model`
first returned a null wrapper. The incomplete v9 P2 root is abandoned and is
never reused. No held path, training, gradient, optimizer, dense artifact, or
scientific result was produced.

The v10 runner requires exact support16 at model initialization, reads the
first committed support image, calls `backbone.prepare_image`, and only then
retrieves and freezes the model. A pure fake-backbone regression test verifies
the exact `read -> prepare -> model/eval/freeze` order without local model or
GPU execution. All other custody and scientific contracts are unchanged. The
exact five P0 files now contain 54 passing tests.

This v10 identity was frozen before any v10 remote execution.

- replacement run identity:
  `eco_audit_v2_dsba3_20260828Tfrozen10Z_v10`
- pinned 13-entry source-manifest SHA-256:
  `63cdfaaffee2b80dc85e084781b8fa7728c2b547fb57d8e50714db0b2831416c`
- runner SHA-256:
  `5dd91db0ba3e511507798d2b4509f43b927490cf8cbaa7fab2518b0d1fbb9146`
- runner-test SHA-256:
  `876f188eecb3cb7ebb3528a3c6c817e4f6e63cd254387450ac63c6bffd675470`
- launcher root-of-trust controller SHA-256:
  `e1dd58ab100a1b851f5203481f58d8094c6fe8c14d268a9291b33894b75030e6`
- separately audited controller-test SHA-256:
  `3e9b1293c1ab368f54c492f85371efadcb333dbc81f4be8ad31a09967d61b0a4`
- container identity SHA-256 (unchanged):
  `b442f1cb5c6278affb2d5160efb9299d6a57e19fafc8c0972aa707c3f8033c68`

## P2 nested-runtime helper correction / Frozen source identity v9

The v8 P0 and P1 phases completed successfully. Its P2 semantic verifier,
pinned source-root checks, fixed-container check, and selected GPU0
index/UUID/memory/utilization gate also passed. The controller persisted the
pre-P2 authorization receipt and entered the support-seal runner. The frozen
support manifest/path set was resolved, but before model construction, image
decoding, DINO forward, or any held access, the runner called
`base._torch_runtime`; the frozen NCFC module exposes that helper through its
nested TARCD base as `base.BASE._torch_runtime`. The immutable empty/incomplete
v8 P2 root is abandoned and is never reused. No training, gradient, optimizer,
query update, held path, dense feature, or scientific result was produced.

The v9 runner changes this single helper boundary and extends the existing
nested-frozen-base regression test to require `base.BASE._torch_runtime` and
reject `base._torch_runtime`. The controller working-directory correction and
non-circular launcher root-of-trust design remain unchanged. No scientific
operator, threshold, seed, target, metric, reducer, or GPU mapping changes.

This v9 identity was frozen before any v9 remote execution.

- replacement run identity:
  `eco_audit_v2_dsba3_20260828Tfrozen09Z_v9`
- pinned 13-entry source-manifest SHA-256:
  `d867c25419a68c7112e954073f98f8c390df92a7dcc0c9f6446c52958600e019`
- runner SHA-256:
  `68d59a43079326040b99a60841b83fc6f097cf0cf4d7628cf95c474e34c07168`
- runner-test SHA-256:
  `f4ea95f5392cbac78182a485a9a8c8f31e0d4c3a41597d79c222f4c4b490a18a`
- launcher root-of-trust controller SHA-256:
  `0fe7ec026fd803bb767145f34a40957a508e157b788ff0a37c49dc546a7a603a`
- separately audited controller-test SHA-256:
  `41cbd347f2a76e8159fc9cb506f89631716670f3d81bb5c0f7872fd62a276af2`
- container identity SHA-256 (unchanged):
  `b442f1cb5c6278affb2d5160efb9299d6a57e19fafc8c0972aa707c3f8033c68`

## Pre-P2 import-custody correction / Frozen controller identity v8

The v7 P0 and P1 phases completed successfully: the fixed remote P0 environment
passed all 53 tests, and the two P1 fixture runs completed at the distinct
allowlisted `a/b` paths with byte identity. The first P2 invocation then
stopped inside the CPU-only semantic prefix, before the selected-GPU map query,
because the remote shell had not changed directory to the frozen snapshot.
Python consequently resolved the container's ambient `/workspace/scripts`
package rather than the frozen snapshot namespace. No P2 receipt or directory
was written, and no GPU, model, support image, dataset image, or held path was
accessed. The v7 root is abandoned and is never reused.

The v8 controller changes directory to the exact frozen snapshot at the start
of every P1, pre-P2, P2 support-seal, chain-construction, and held shell. The
controller test requires exactly six snapshot-directory anchors including the
existing P0 anchor. Because the controller and its test are the separately
audited launcher root-of-trust surface, this correction does not alter the
pinned 13-entry scientific-source manifest or its trusted literal. It changes
no scientific data, model, operator, metric, reducer, or GPU allocation.

This v8 identity was frozen before any v8 remote execution or result-bearing
model/dataset/GPU access.

- replacement run identity:
  `eco_audit_v2_dsba3_20260828Tfrozen08Z_v8`
- pinned 13-entry source-manifest SHA-256 (unchanged):
  `c3ce2f60a126d56f161e9bbb70c845e57e6f0b16c032c5555bd9a0b5931f321d`
- launcher root-of-trust controller SHA-256:
  `738d9ecbaa80c15bd913e9875b7960c4a087c5e4e1588e29559c95ced264ea13`
- separately audited controller-test SHA-256:
  `db61aee054022e74cb5f98599c98805800332365ab33806a3f0d9a21b4b1d23c`
- container identity SHA-256 (unchanged):
  `b442f1cb5c6278affb2d5160efb9299d6a57e19fafc8c0972aa707c3f8033c68`

## Non-circular launcher root-of-trust amendment / Frozen source identity v7

The v6 identity was not launched. An independent result-free review correctly
observed that hashing the local manifest at controller invocation was stronger
than a chain-derived binding but was not an immutable external anchor: a local
source change followed by manifest resealing would cause the controller to
accept the new digest.

The v7 custody design removes the unavoidable self-reference by defining the
local remote controller as the explicit launcher root of trust. The controller
and its static controller test are excluded from the exact scientific-source
manifest; the remaining 13 frozen runner, verifier, aggregator, core, inherited
primitive, shift, and scientific-test files remain exact-key and actual-byte
validated. The controller pins the canonical 13-entry manifest SHA-256 as a
readonly literal, rejects the local manifest unless it matches that literal,
requires the remote snapshot manifest to match the same literal, and passes
that literal to pre-P2, strict-chain verification, held reverification, and
aggregation. Consequently the trusted digest is neither chain-derived nor
recomputed as an authorization value after a local reseal.

The exact five P0 tests are still executed. Their P0 evidence records the four
tests inside the 13-entry scientific-source manifest; the controller safety
test runs in the same pinned P0 invocation but the controller and its test are
audited separately as the launcher root-of-trust surface. The controller and
controller-test hashes are recorded below and in this preregistration, but are
intentionally outside the self-referential manifest. The P1 double-quoted
`a/b` path-expansion correction from v6 is retained.

These are custody-only changes. The scientific protocol, data, operators,
metrics, reducer, decision gates, and fixed `dsba3` GPU 0--3 allocation remain
unchanged. This v7 identity was frozen before any v7 remote execution or any
result-bearing model/dataset/GPU access.

- replacement run identity:
  `eco_audit_v2_dsba3_20260828Tfrozen07Z_v7`
- pinned 13-entry source-manifest SHA-256 / launcher trusted literal:
  `c3ce2f60a126d56f161e9bbb70c845e57e6f0b16c032c5555bd9a0b5931f321d`
- launcher root-of-trust controller SHA-256:
  `121f349c38909b2ca359a0b746121740d3451d30ebcd255ae8e36666c0661b38`
- separately audited controller-test SHA-256:
  `827dfd28d9bdd6a83c50fa18e418571623b89f4112eacea11b96d53209b97c4a`
- runner SHA-256:
  `8c419c598c65e2dc6eedc86dbad1dcf7ef4918ff9a1d9341433378f396426965`
- verifier SHA-256:
  `5e633acae2a7bbbe34199a81535ace89dd9ca4206744b22795b1fb1596141294`
- verifier-test SHA-256:
  `68cfacb04ad735c9a8f69d9925816aa3d66596558e90111bb627ff0fbb4d49f6`
- aggregator SHA-256 (unchanged from v5/v6):
  `dc94c14e731b9057b411c58c55416b6bfd762dc27dbe29d2e9be894c30cb05e0`
- container identity SHA-256 (unchanged):
  `b442f1cb5c6278affb2d5160efb9299d6a57e19fafc8c0972aa707c3f8033c68`

## Post-P1 result-free controller correction / Frozen source identity v6

The v5 P0 phase completed with 53 passing tests in the pinned, snapshot-scoped
remote CPU environment. Its first P1 attempt then stopped before writing P1
evidence because the controller enclosed the remote loop variable `${out}` in
single quotes. Both deterministic iterations therefore targeted the same
literal path, and the second iteration was rejected by the immutable-output
guard. No dataset, model, support encoding, CUDA kernel, GPU metric, or held
path was accessed. The incomplete v5 result root is abandoned and is never
reused.

The v6 controller double-quotes the P1 output path so the remote loop expands
to the exact distinct allowlisted paths `p1/a` and `p1/b`. A static regression
contract requires the double-quoted expansion form and rejects the prior
single-quoted literal form. This correction changes only controller path
expansion; all scientific and external-source-anchor contracts remain
unchanged. The exact five P0 files still contain 53 passing pure/static tests.

This v6 identity was frozen before its remote execution and before any
result-bearing model/dataset/GPU execution or held-path resolution.

- replacement run identity:
  `eco_audit_v2_dsba3_20260828Tfrozen06Z_v6`
- source manifest: `source_hashes.json`
- source-manifest SHA-256:
  `2f125136135cc0ee482c45e3cd3ff3bb1c4701e285dce018cb7f1fdc1dcd6ac0`
- remote-controller SHA-256:
  `92151cdadf9e31bae67c7f7c09bc6486448b7c0c3df06e134f6ad1a1257c3f9f`
- remote-controller test SHA-256:
  `984034901f7e91922657e7a64a67b9492dd6c7d7cc24541d5f96257c3d110d42`
- runner SHA-256 (unchanged from v5):
  `9ebde136be08289530238c02d2b8d01a3db2d801cc7b3a698c8afed940ba44ec`
- aggregator SHA-256 (unchanged from v5):
  `dc94c14e731b9057b411c58c55416b6bfd762dc27dbe29d2e9be894c30cb05e0`
- verifier SHA-256 (unchanged from v5):
  `602ffe823af98451f9c531f45bce828fbc6918d9b8285fd0f7a03f4e0c2bfda6`
- container identity SHA-256 (unchanged):
  `b442f1cb5c6278affb2d5160efb9299d6a57e19fafc8c0972aa707c3f8033c68`

## Result-free adversarial custody amendment / Frozen source identity v4

The v3 identity was never launched remotely and produced no snapshot, P0/P1
artifact, data/model access, CUDA/GPU execution, or held result. Result-free
adversarial verification found that an in-root intermediate-directory symlink
could alias the two declared P1 paths, a hard link could make both paths the
same inode, the full-chain artifact root could be a symlink alias, and a
resealed strict chain could substitute a noncanonical in-root P2 path.

The v4 verifier closes only those custody gaps:

- every relative artifact path component is checked and any symlink fails
  closed;
- the strict chain rejects a symlink artifact-root leaf;
- the two P1 fixtures must use the exact ordered paths and resolve to distinct
  files/inodes;
- strict P0/P1/P2 primary paths and ordered phase audit allowlists are exact;
- the persisted pre-P2 receipt is semantically revalidated during the full P2
  chain, including schema, verdict, P2-only scope, `held_authorized=false`,
  fixed container/source bindings, typed zero-access ledger, and exact hashes
  for all five P0/P1 inputs;
- resealed alternate P2 paths, receipt authorization changes, missing receipt
  inputs, root/intermediate symlinks, and hard-linked fixture aliases are
  covered by adversarial regression tests.

These checks do not change the scientific contract, data, operators,
thresholds, seeds, target records, cell universe, metrics, reducer, or remote
resource policy. The exact five P0 files now contain 51 passing pure/static
tests. Independent read-only adversarial review found no remaining HIGH or
BLOCKER issue for the fixed remote paths.

This v4 identity was frozen before any result-bearing model/dataset/GPU
execution and before any held-path resolution.

- replacement run identity:
  `eco_audit_v2_dsba3_20260828Tfrozen04Z_v4`
- source manifest: `source_hashes.json`
- source-manifest SHA-256:
  `c141b647f78474fcf383c88c72225f60a0bbf4bfc00cc5c515cfead83b6bdded`
- remote-controller SHA-256 (unchanged from v3):
  `5555d23d2a19c28ba0a9d3f9f6ac94a14bf4b484cffdaceb01935d315f1747b0`
- verifier SHA-256:
  `1a732dbc1f89d0619eea75ffb921480db253cdc375d65f9a6ac1232fb0796470`
- container identity SHA-256 (unchanged):
  `b442f1cb5c6278affb2d5160efb9299d6a57e19fafc8c0972aa707c3f8033c68`
- static-contract SHA-256 (unchanged):
  `6fee9bb3dd1e6c6dfcec786c5d44f0b23b1dd750630c04282b6f4f14cefc4100`
- support-calibration commitment SHA-256 (unchanged):
  `66708c38671ceeb09837f5471669f490af163b0d978b6013063262f9102a7091`
- held commitment SHA-256 (unchanged):
  `b51b6ca946b616fbc2578eaf1b20e54f2f905a52eb51deb9799abf715e5f7460`
- frozen support-manifest SHA-256 (unchanged):
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`

## Post-P0 result-free recovery amendment / Frozen source identity v3

The second immutable attempt,
`eco_audit_v2_dsba3_20260828Tfrozen02Z_v2`, stopped before pytest collection
because pytest `9.1.1` on the fixed Python `3.10.12` runtime additionally
requires `exceptiongroup` and `tomli`; `exceptiongroup` also requires
`typing-extensions`. No dataset, model, support encoding, CUDA kernel, GPU
metric, or held-path result was accessed. The v2 snapshot/result root is
abandoned and is not reused by v3.

The v3 P0-only environment pins the complete observed dependency closure:
pytest `9.1.1`, pluggy `1.6.0`, packaging `26.3`, iniconfig `2.3.0`, pygments
`2.20.0`, exceptiongroup `1.3.1`, tomli `2.4.1`, and typing-extensions
`4.15.0`. It additionally requires Python `3.10.12`, NumPy `1.26.4`,
`PYTHONNOUSERSITE=1`, and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. A result-free
probe in the abandoned v2 snapshot confirmed this exact environment imports;
it did not open model, data, held, or GPU resources.

Before P2 may query the selected GPU map or open support/model/data, the v3
verifier now re-derives the P0/P1 prefix from persisted bytes. It requires the
exact P0 source path, exact ordered P1 `a/b` paths, copied-versus-snapshot source
byte identity, typed zero-access claims, frozen test environment, two
byte-identical P1 runs, eight unique target records, 32 exact Cartesian cells,
four references per target, exact reducer labels/cases, and the internal
fixture hash. A deterministic receipt records the exact five input artifact
hashes, fixed-container hash, `authorization_scope=P2_SUPPORT_SEAL_ONLY`, and
`held_authorized=false`. The receipt is persisted only after the selected GPU
is available, so a resource-busy deferral remains safely retryable without
overwriting an immutable phase artifact.

The authorized fixed container was resolved once by an exact-name,
single-object `docker inspect`; no container, process, PID, user, or resource
enumeration was performed. Every subsequent `docker exec` uses the full
pre-registered container ID rather than the name. Each controller invocation
fails closed unless ID, image ID, name, and running state all match exactly:

- container name: `hun_fsad_tta_012`
- container ID:
  `16c4c25074fff40040aed82b552a8cd77fbdfcf52bb553b0a318f7faeef5def6`
- image ID:
  `sha256:8dbd129dc64fbe1bbce27212902b27e2ecd3fea01794565c80f5ef415369b2f1`
- container identity SHA-256: SHA-256 of the UTF-8 bytes
  `<container-id><space><image-id>` with no trailing newline:
  `b442f1cb5c6278affb2d5160efb9299d6a57e19fafc8c0972aa707c3f8033c68`

The aggregator also converts extreme numeric overflow into a structural-invalid
fail-closed verdict rather than propagating an exception. These changes harden
execution and evidence custody only. They do not change the ECO data,
operators, thresholds, seeds, target records, cell universe, features, metrics,
reducer, decision rule, or scientific protocol ID. The exact five P0 files now
contain 46 passing pure/static tests.

This v3 identity was frozen before any result-bearing model/dataset/GPU
execution and before any held-path resolution.

- replacement run identity:
  `eco_audit_v2_dsba3_20260828Tfrozen03Z_v3`
- source manifest: `source_hashes.json`
- source-manifest SHA-256:
  `cb6a0495d436ae533e29aa301a1c90748d66a51673e2aed91daad458edd726ba`
- remote-controller SHA-256:
  `5555d23d2a19c28ba0a9d3f9f6ac94a14bf4b484cffdaceb01935d315f1747b0`
- verifier SHA-256:
  `70e3c64f85b72977f0870353c4daaf938da67b53660b00b6d3c2eea15098c13e`
- static-contract SHA-256 (unchanged):
  `6fee9bb3dd1e6c6dfcec786c5d44f0b23b1dd750630c04282b6f4f14cefc4100`
- support-calibration commitment SHA-256 (unchanged):
  `66708c38671ceeb09837f5471669f490af163b0d978b6013063262f9102a7091`
- held commitment SHA-256 (unchanged):
  `b51b6ca946b616fbc2578eaf1b20e54f2f905a52eb51deb9799abf715e5f7460`
- frozen support-manifest SHA-256 (unchanged):
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`
