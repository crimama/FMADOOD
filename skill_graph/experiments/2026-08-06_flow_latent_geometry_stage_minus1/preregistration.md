# Flow Latent Geometry Diagnostic — Stage -1 Preregistration

Date: 2026-08-06
Status: `FROZEN_BEFORE_GEOMETRY_FEATURE_EXTRACTION`
Protocol: `flow-latent-geometry-stage-minus1-20260806`

## Execution environment

- Host `dsba5`, fixed container `hun_fsad_tta_012`, matching
  `configs/remote/dsba5.env`.
- Host GPU id `3` (the fourth of four), exposed to the container as CUDA slot
  `0`; the container must report exactly one visible CUDA device.
- Cooperative flock on `/workspace/isolated_runs/.flow_latent_geometry_gpu3.lock`
  so this run cannot contend with a concurrent WTCF run on the same GPU.

## Question

Does the trained normal-only Flow induce a latent geometry in which proven
normal-variation directions are stretched at least as much as defect-like
directions, so that Euclidean 1-NN cannot separate them? That is: is Normal
Inflation a *geometric identifiability failure of the Flow map itself*, rather
than a scoring or memory-coverage artifact?

This is a **measurement-only diagnostic**. No new scorer, loss, or method is
proposed or tuned here. No arm is selected on the basis of detector metrics.

## Frozen detector contract

Identical to `wtcf-stage-minus1-support-only-20260804`:

- MVTec AD 2 single-image setting;
- DINOv2 ViT-L/14 layers `[5, 11, 17, 23]`, layer-normalized mean fusion;
- 16 normal support images, DVT off, density score off;
- two affine coupling layers, hidden multiplier 1, three initial NLL epochs;
- no query adaptation, memory expansion, routing, public/private test access,
  or ground-truth access.

The Flow states under measurement are frozen inputs. Nothing in this experiment
updates any Flow parameter.

## Frozen Stage -1 scope

- Objects: `can`, `fruit_jelly` (prior weak), `fabric` (prior strong).
- Support seeds: `0, 1, 2`, reusing the sealed 16-path manifests from
  `2026-08-04_wtcf_stage_minus1` byte-for-byte. Manifest and fold SHA-256
  hashes must match that experiment's recorded hashes; a mismatch is a hard
  abort.

### Amendment 1 (2026-08-06, before any feature extraction)

Factual correction, not a tuning change. On inspecting
`results/remote_runs/dsba5/wtcf_stage_minus1_20260804_v6` it emerged that the
WTCF run **early-killed after its single `can`/seed0 case** (sentinel
`jvp_directional: false`). Consequently:

- a recorded `support_manifest_sha256` exists for `can`/seed0 only
  (`dd9f44f8...`), sealed here in `reference_manifest_hashes.json`;
- **no fold-assignment hash was ever recorded** by that run, for any case;
- the remaining eight cases have no reference hash and therefore cannot
  satisfy the strict-match condition as originally written.

No constant, arm, direction family, or gate threshold is changed by this
amendment. The strict-match requirement stands unchanged for `can`/seed0.

**Execution is therefore staged.** Stage A runs `can`/seed0 alone under strict
matching, to demonstrate that this experiment's manifest pipeline reproduces
the WTCF manifest byte-for-byte. Stage B (the remaining eight cases) is not
authorized by this document: whether to seal self-computed hashes as the new
reference is a separate decision to be recorded as Amendment 2, taken only
after Stage A's hash comparison is known and before Stage B features are
extracted.

### Amendment 2 (2026-08-06, no result yet produced)

Compute-feasibility constant, not a tuning change. The original text capped
`witness` (at most eight), `principal` (top-8), and `unsupported` (matched to
witness) but **left `defect_proxy` uncapped**. Every stimulus patch row is a
candidate direction, so at resolution 672 with DINOv2 patch 14 the family grew
to roughly 27,600 directions — about a thousand times every other family, and
roughly ten million JVP calls per case once anchors and curvature steps are
included. Run `flow_latent_geometry_20260806_stagea_v2` was invalidated after
5h54m with its first case still unfinished and **no summary, margin, or
geometric quantity of any kind produced**.

- New constant `GeometryConfig.max_defect_directions = 256`, split evenly
  across the twelve stimulus sources (21 per source) by deterministic
  subsampling seeded from the protocol/object/seed triple.
- Reported provenance now records, per source, both the sampled count and the
  full available count, so the truncation is visible rather than silent.

No gate threshold, arm, anchor rule, direction definition, or validity
condition is changed. The constant affects only the sample size behind
`median(s | defect_proxy)`; 256 samples is ample for a median. Because the
invalidated run produced no measurement, this cannot be post-hoc tuning: there
was no result to tune toward.

### Amendment 3 (2026-08-07, AFTER Stage A results were seen)

**This amendment is post-result and is therefore declared exploratory. It
cannot revive, soften, or re-open any confirmatory claim.**

Stage A (`flow_latent_geometry_20260806_stagea_v3`, `can`/seed0) completed with
the manifest hash matching the recorded WTCF reference byte-for-byte, full
access custody, and every arm-level geometric check passing except one: witness
discovery accepted **2** directions from 648 candidates, below the frozen
minimum of three. Per this document that is a valid reported outcome which
**blocks every witness-relative claim, including `M` and therefore H1**.

Stage B asks a different, descriptive question:

> Is witness scarcity specific to `can`/seed0, or a structural property of this
> support regime across objects and seeds?

Two protocol changes are required and are recorded here in full:

1. **Early-abort suppressed.** The controller's sentinel is still evaluated and
   written to `sentinel_gate.json`, but no longer terminates the run. Without
   this, `can`/seed0's witness sentinel would stop Stage B before it could
   observe the very distribution it exists to observe.
2. **Remaining hashes sealed.** `scripts/seal_flow_latent_geometry_hashes.py`
   derives manifest and fold hashes for all nine cases *before* any Stage B
   feature extraction. `can`/seed0 is re-derived and **compared** against the
   WTCF reference (mismatch aborts); the other eight carry no independent
   cross-run corroboration and are labelled as such in the registry.

**Standing of Stage B results.** Descriptive only. The H1 verdict stays
permanently blocked at `INCONCLUSIVE_WITNESS_SENTINEL`. Witness counts, family
sensitivity spectra, curvature, and effective rank may be reported as
observations. Any hypothesis suggested by them — including the witness-free
`unsupported` vs `defect_proxy` contrast that Stage A surfaced — requires a
**new preregistration frozen before its own data** to become confirmatory.
No constant, threshold, or direction definition is otherwise changed.

### Context correction

The companion analysis note
`2026-08-06_normal_inflation_representation_space_stage_minus1_analysis.md`
tabulates `wtcf` detector metrics for `can`, `fabric`, and `fruit_jelly` and
marks them `PASSED`. Those rows do not correspond to any completed run: only
`can`/seed0 was ever executed, and it failed its sentinel. The frozen WTCF
witness-JVP bounds reused as constants here have therefore never been
satisfied in practice. This does not invalidate the present diagnostic — it
sharpens it, since explaining *why* those directional bounds are unreachable
is exactly what the separation margin `M` measures.
- Four deterministic image-disjoint folds per object/seed, reused unchanged.
- Support (`train/good`) features only. Zero test/label access.

## Frozen measurement arms

Two frozen latent maps are measured, never trained:

1. `identity` — no Flow (raw standardized fused features), i.e. the
   `a7_identity` latent space.
2. `a6` — the fitted A6 Flow state, restored from the same construction
   recipe as `2026-08-04_wtcf_stage_minus1`.

## Frozen direction families

At each probe anchor, sensitivity is measured for four direction families.
All directions are unit-norm and canonicalized by `_canonical_direction`.

- `witness` — repeated cross-image normal-residual directions from
  `discover_normal_witnesses` with the frozen WTCF constants (cosine `0.85`,
  at least three source pairs and three fold pairs, at most eight witnesses).
  These are the *proven normal-variation* directions.
- `principal` — top-8 PCA directions of the support latents of the arm under
  measurement, rank-32 fit from at most 4096 deterministic training patches.
- `defect_proxy` — residual directions from the frozen synthetic C/S/D/SD
  stimuli, computed as (stimulus patch latent - its source clean patch
  latent), normalized. Exactly one SHA-first diagnostic source per fold is
  used, and that source is excluded from witness mining and PCA fitting.
- `unsupported` — deterministic random probes Gram-Schmidt orthogonalized
  against every `witness` and `principal` direction, matched in count to
  `witness`.

Anchors are the witness anchors plus, per fold, the SHA-first clean patch rows
of the excluded diagnostic source, capped deterministically.

## Frozen measured quantities

All are computed with `jvp_sensitivity(..., create_graph=False)` under
`torch.no_grad()` for the Flow arm and analytically (ratio `1.0`) for
`identity`.

1. **Anisotropy spectrum** `s(a, v) = ||J_f(a) v|| / ||v||` for every
   anchor/direction pair, reported per family as median and `q05/q25/q75/q95`.
2. **Separation margin** `M = median(s | defect_proxy) / median(s | witness)`,
   per object/seed/arm. `M > 1` means the Flow stretches defect directions
   more than normal-variation directions; `M <= 1` means it does not.
3. **Curvature** — for each anchor/direction, recompute `s` at anchors
   displaced by `t in {-2, -1, +1, +2} * eps` along the same direction, with
   `eps = 0.05 * median ||anchor||`. Report
   `C = max_t |s(a + t v, v) - s(a, v)| / s(a, v)` and the tangent rotation
   `1 - cos(J_f(a+tv) v, J_f(a) v)`.
4. **Effective rank** of the support latents per arm via `_effective_rank`,
   and the latent RMS, to place the above on a common scale.

Every quantity is reported for `identity` and `a6` so that the Flow's own
contribution is the paired difference.

## Validity before performance

Interpretation of any geometric quantity is blocked unless all hold:

- support manifest and fold SHA-256 hashes match the recorded
  `2026-08-04_wtcf_stage_minus1` values exactly;
- 100% selected-support access coverage and zero test/label access;
- every reported sensitivity, curvature, and rank value is finite and the
  per-family sample is nonconstant;
- witness count is at least three, with the preregistered source-pair and
  fold-pair diversity; a zero-witness or fewer-than-three-witness sentinel is
  a valid reported outcome but blocks every `witness`-relative claim;
- `identity` arm sensitivities equal `1.0` within `1e-6` (self-consistency
  check on the JVP path);
- Flow inverse error at each anchor is at most `1e-5`;
- `defect_proxy` directions are confirmed non-degenerate: median absolute
  cosine to the nearest `witness` direction is reported, and a value above
  `0.85` invalidates the defect/normal contrast for that object/seed.

## Keep/kill gate

The diagnostic claim under test is:

> **H1 (Normal Inflation is geometric):** the A6 Flow fails to separate normal
> from defect directions, i.e. `M <= 1.05` on at least two of three objects,
> with the same direction of effect on at least two of three seeds per object.

- **Claim confirmed** if H1 holds *and* `M` for `a6` is not materially larger
  than `M` for `identity` (within `0.05`), i.e. Flow training did not buy
  directional separation over raw features.
- **Claim killed** if `a6` achieves `M >= 1.20` on at least two of three
  objects with seed agreement. That result would mean the Flow *does* separate
  directions and Normal Inflation must be attributed elsewhere (memory
  coverage or the scoring rule), redirecting the research line away from the
  Flow map.
- **Inconclusive** (`1.05 < M < 1.20`, or no seed agreement) is a reported
  outcome and blocks both the geometric and the non-geometric attribution.

Curvature is reported as supporting evidence only and cannot by itself
confirm or kill H1.

This is a diagnostic with no performance metric, so there is no detector-metric
gate. Correspondingly, **no result here may be used to tune any constant in
this file, and no arm may be added after unblinding.** A completed failure is
terminal for the geometric attribution of Normal Inflation.
