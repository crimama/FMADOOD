# Flow Latent Geometry Stage -1 Report

Date: 2026-08-07
Protocol: `flow-latent-geometry-stage-minus1-20260806`
Runs: `flow_latent_geometry_20260806_stagea_v3` (Stage A, `can`/seed0),
`flow_latent_geometry_20260807_stageb_v2` (Stage B, all nine cases)

## Outcome

`INCONCLUSIVE_WITNESS_SENTINEL / H1_BLOCKED / EXPLORATORY_OBSERVATIONS_ONLY`

The confirmatory question — whether the separation margin
`M = median(s | defect_proxy) / median(s | witness)` shows that the A6 Flow
fails to separate defect directions from proven normal-variation directions —
**cannot be answered**. Witness discovery accepted fewer than the preregistered
minimum of three directions in eight of nine cases, and the preregistration
makes a witness-count sentinel block every witness-relative claim. The
aggregate verdict is `BLOCKED_BY_VALIDITY_FAILURE`.

What the run did establish is descriptive and, under Amendment 3, explicitly
exploratory: **witness scarcity is structural across objects and seeds**, and
the Flow's directional amplification is **not selective for defects**.

## Sentinel and Validity Gate

Stage A passed every infrastructure and custody condition:

| Check | Result |
|---|---|
| Support manifest SHA-256 vs recorded WTCF value | matched byte-for-byte (`dd9f44f8...`) |
| Access custody | coverage `1.0`, zero test/ground-truth attempts |
| `identity_exact` (identity arm sensitivities `== 1.0`) | passed, all nine cases |
| Flow inverse error | `7.2e-07` – `9.5e-07`, limit `1e-5` |
| `defect_contrast_valid` (defect ⟂ witness) | passed; cosines `0.027` – `0.052` |
| `witness_sufficient` (≥ 3) | **failed in 8/9 cases** |

The defect/witness cosines confirm the contrast is non-degenerate: defect-proxy
directions are close to orthogonal to the witness directions wherever both
exist, so the near-unity margins are not an artifact of the two families
collapsing onto each other.

## Witness Scarcity

Every case mined exactly 648 mutual-nearest-residual candidates under the
frozen WTCF constants (cosine `0.85`, at least three source pairs and three
fold pairs, at most eight witnesses).

| Case | Accepted witnesses |
|---|---:|
| `can` seed0 / seed1 / seed2 | 2 / 1 / **0** |
| `fabric` seed0 / seed1 / seed2 | **0** / **0** / **0** |
| `fruit_jelly` seed0 / seed1 / seed2 | **0** / **0** / 3 |

**5,832 candidates across nine cases yielded six accepted witnesses.** This is
not specific to `can`, the object Stage A happened to test, nor to any seed.
The one case clearing the bar (`fruit_jelly`/seed2) did so at exactly the
minimum diversity on every axis (three source pairs, three fold pairs, three
witnesses), i.e. marginally.

## Descriptive Metrics Only

Per-family median JVP sensitivity `s(a, v) = ||J_f(a) v|| / ||v||` for the `a6`
arm. The `identity` arm is analytically `1.0` everywhere and is omitted.
`unsupported` is generated to match the witness count, so it is empty wherever
witnesses are.

| Case | wit | `M` | witness | principal | defect | unsupported |
|---|---:|---:|---:|---:|---:|---:|
| `can` seed0 | 2 | 1.0046 | 1.0948 | 0.8478 | 1.0998 | **1.1388** |
| `can` seed1 | 1 | 1.0470 | 1.0756 | 0.8485 | 1.1262 | **1.1613** |
| `can` seed2 | 0 | — | — | 0.8512 | 1.1216 | — |
| `fabric` seed0 | 0 | — | — | 0.7741 | 1.0755 | — |
| `fabric` seed1 | 0 | — | — | 0.7731 | 1.0961 | — |
| `fabric` seed2 | 0 | — | — | 0.7774 | 1.0985 | — |
| `fruit_jelly` seed0 | 0 | — | — | 0.9123 | 1.1755 | — |
| `fruit_jelly` seed1 | 0 | — | — | 0.8947 | 1.1522 | — |
| `fruit_jelly` seed2 | 3 | 1.0748 | 1.1159 | 0.9107 | 1.1993 | **1.2334** |

Three observations recur without exception.

**1. Random directions are amplified more than defect directions.** In all
three cases where `unsupported` exists, it exceeds `defect_proxy` by a nearly
constant margin (`+0.039`, `+0.035`, `+0.034`). Deterministic probes
orthogonalized against witness and principal directions — carrying no defect
information by construction — are stretched *more* than actual synthetic
defect residuals.

**2. The map contracts principal directions and expands everything else.**
`principal` is below `1.0` in all nine cases; every other family is above.
Within an object the spread across seeds is at most `0.018`:

| Object | principal | defect | effective rank (identity → a6) |
|---|---|---|---|
| `fabric` | 0.773 – 0.777 | 1.076 – 1.099 | 159 – 166 → 316 – 318 |
| `can` | 0.848 – 0.851 | 1.100 – 1.126 | 138 – 142 → 318 – 329 |
| `fruit_jelly` | 0.891 – 0.912 | 1.152 – 1.199 | 102 – 108 → 255 – 258 |

Effective rank roughly doubles and latent RMS drops to `0.81` – `0.89`. This is
the signature of a whitening-like transform: variance is pulled out of the
dominant support directions and spread over the remainder.

**3. Curvature is concentrated on principal directions.** Median curvature is
`0.043` – `0.065` for `principal` versus `0.0057` – `0.0082` for
`defect_proxy` — an order of magnitude. Sensitivity along principal directions
varies strongly with anchor position, so a single global linear correction
cannot describe the map there.

Where `M` is computable it is `1.0046`, `1.0470`, `1.0748`, against `1.0` for
identity. These are reported as observations; the H1 comparison they were
designed to feed is blocked.

## Interpretation (Exploratory)

The two headline findings are plausibly one fact. The support is sixteen images
captured before the AD2 test-time domain shift, so directions along which
normals move under shift are simply absent from it. That predicts both results
at once: there is no repeated cross-image normal-variation direction to mine
(witness scarcity), and the whitening-like map treats those unseen shift
directions as low-variance and therefore **amplifies** them (indiscriminate
expansion). Under this reading the Flow does not merely fail to suppress
nuisance variation — it enlarges it, which is a mechanism for Normal Inflation
rather than a defense against it.

The same reading retro-explains several closed branches: WTCF sought to
contract proven normal-variation directions that mostly do not exist here;
`crossfit_only` had no directional information to be selective with and
absorbed anomalies; `pca_residual` and `shrinkage_mahalanobis` estimated
covariance structure that omits the shift axis; and WAFF found no point on the
distance/NLL simplex separating shifted normals from defects because both
scores derive from the same non-selective geometry.

**This interpretation is not established by this experiment.** It is a
hypothesis consistent with nine cases of exploratory measurement.

## Principal Threat to the Interpretation

Witness scarcity may be a property of the *criterion* rather than of the data.
Acceptance requires absolute cosine `0.85` between residual directions in a
feature space of order 10^3 dimensions, where near-parallel agreement is
demanding. This experiment cannot distinguish:

- (a) there is no shared normal-variation axis; from
- (b) an axis exists but `0.85` is effectively unreachable at this dimension.

`fruit_jelly`/seed2 shows the threshold is not literally unsatisfiable, but a
single marginal case does not settle it. The discriminating measurement is
cheap and needs no GPU: the pairwise-cosine distribution over the 648 candidate
residual directions per case. If its upper tail falls well short of `0.85`, the
threshold is the binding constraint — which would also mean the WTCF family
inherited a constant it could rarely satisfy, and would require this report's
interpretation to be revised.

## Implementation and Verification

**The implementation was written in the throwaway `worktree-flow-work` branch
and deliberately discarded after this report was filed. It is not in the
repository.** Only the record below, the raw run artifacts on dsba5, and one
upstreamed defect fix survive. A follow-up must reimplement from this
description; that is an accepted cost of not carrying a blocked diagnostic's
scaffolding in the tree.

What existed (~2,478 lines):

- `src/flow_tte/latent_geometry.py` — measurement core, reusing
  `jvp_sensitivity`, `discover_normal_witnesses`, `_canonical_direction`,
  `_effective_rank`, `_unsupported_directions`, `stable_seed`, and `_choice`
  from the retained `src/flow_tte/wtcf.py`.
- `scripts/flow_latent_geometry_case.py`, `run_flow_latent_geometry_stage_minus1.py`,
  `aggregate_flow_latent_geometry_stage_minus1.py`,
  `smoke_flow_latent_geometry.py`, `seal_flow_latent_geometry_hashes.py`,
  `run_flow_latent_geometry_stage_minus1_remote.sh`.
- `tests/test_latent_geometry.py` — 18 unit tests, all passing at discard time;
  no regression in the existing suite.

Upstreamed to `main`: the `patch_torch_pytree_compat()` shim in
`scripts/smoke_wtcf_core.py` (defect 1 below), verified on the dsba5 container
as failing before and passing after.

Execution: dsba5, container `hun_fsad_tta_012`, host GPU 3 → container CUDA
slot 0, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, cooperative flock on
`.flow_latent_geometry_gpu3.lock`. Stage B completed nine cases in 90 minutes.

Four defects were found and fixed during execution, each recorded here because
three of them silently affected or halted runs:

1. **Container environment regression.** `torch 2.1.2` with `transformers 4.56.2`
   raises on `torch.utils._pytree.register_pytree_node` as soon as
   `torch.optim` reaches `torch.onnx`. Pre-existing and not caused by this work
   — `scripts/smoke_wtcf_core.py` fails identically today — so the WTCF preflight
   would no longer pass on this container either. Fixed by applying the repo's
   existing `patch_torch_pytree_compat()` shim in the smoke script; the runners
   already had it.
2. **Unbounded `defect_proxy` family** (Amendment 2). Every stimulus patch row
   became a direction (~27,600 per case, ~10^7 JVP calls), and run
   `..._stagea_v2` was invalidated after 5h54m with its first case unfinished
   and no measurement produced. Capped at 256 directions, evenly and
   deterministically subsampled across the twelve sources.
3. **NaN serialization crash.** A zero-witness case makes `separation_margin`
   NaN, which the preregistration calls a valid reported outcome, but
   `_atomic_json` writes with `allow_nan=False`. Run `..._stageb_v1` died at
   `can`/seed2 after three cases. Non-finite scalars are now written as JSON
   `null` with every substituted path recorded, and the aggregator maps them
   back to NaN.
4. **Sentinel/schema mismatch.** The remote controller's sentinel assumed
   per-family validity flags at the case top level; they live under each arm.
   Caught before execution because the sentinel uses direct indexing rather
   than `.get()` defaults, so the mismatch would have raised rather than
   silently passed.

Manifest custody: `can`/seed0 was re-derived and verified against the recorded
2026-08-04 WTCF hash; the other eight cases carry hashes sealed by
`seal_flow_latent_geometry_hashes.py` before any Stage B feature extraction and
are labelled in the registry as lacking independent cross-run corroboration.

## Decision

The confirmatory claim is closed as `INCONCLUSIVE_WITNESS_SENTINEL`. It cannot
be revived by lowering the witness minimum after the fact, and no result in
this report may be used to tune any constant in its preregistration.

Two follow-ups are indicated, in order:

1. **Threshold-vs-data diagnostic.** Measure the pairwise-cosine distribution
   over candidate residual directions. Until this is known, the interpretation
   above remains conditional. Cheap, CPU-only.
2. **A new preregistration for the witness-free contrast.** The
   `unsupported` > `defect_proxy` inversion held in 3/3 measurable cases with a
   near-constant gap and requires no witness family, so it can serve as a
   confirmatory primary endpoint. Its design must decouple `unsupported` from
   the witness count — the present coupling is why six cases produced no
   comparison at all.

Compact evidence:
`results/remote_runs/dsba5/flow_latent_geometry_20260807_stageb_v2/`.
