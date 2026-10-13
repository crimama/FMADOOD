# Flow Latent Geometry Stage -1 — Execution Log

## 2026-08-06 — Preregistration frozen

- Froze a measurement-only Stage -1 diagnostic asking whether the A6 Flow map
  separates proven normal-variation directions from defect-like directions, or
  whether Normal Inflation is a geometric identifiability failure of the map.
- Reused the `wtcf-stage-minus1-support-only-20260804` detector contract,
  sealed 16-path manifests, and four image-disjoint folds unchanged.
- Two arms (`identity`, `a6`), four direction families (`witness`,
  `principal`, `defect_proxy`, `unsupported`), primary quantity
  `M = median(s | defect_proxy) / median(s | witness)` via JVP sensitivity.
- Amendment 1: the 2026-08-04 WTCF run early-killed after `can`/seed0, so a
  recorded manifest hash exists for that case only and no fold hash exists at
  all. Execution staged accordingly; the analysis note's `wtcf` PASSED rows for
  three objects correspond to no completed run.

## 2026-08-06 — `..._stagea_v2` invalidated (Amendment 2)

- Preflight initially failed on a pre-existing container regression:
  `torch 2.1.2` + `transformers 4.56.2` break on
  `torch.utils._pytree.register_pytree_node`. Fixed by applying the repo's
  `patch_torch_pytree_compat()` shim in the smoke script.
- Run invalidated after 5h54m with its first case unfinished. Cause: the
  `defect_proxy` family was uncapped, taking every stimulus patch row
  (~27,600 directions, ~10^7 JVP calls; GPU 7% / CPU 98%).
- Amendment 2 added `GeometryConfig.max_defect_directions = 256` with even
  deterministic subsampling. No gate, arm, or definition changed; the
  invalidated run produced no measurement.

## 2026-08-06 — Stage A complete (`..._stagea_v3`)

- `can`/seed0 in 10m41s. Manifest hash matched the recorded WTCF value
  byte-for-byte; access custody, `identity_exact`, inverse error, and defect
  contrast all passed.
- Sentinel failed on `witness_sufficient`: 2 accepted from 648 candidates.
- Observed `M = 1.0046` vs identity `1.0`; `unsupported` (1.1388) exceeded
  `defect_proxy` (1.0998); `principal` contracted to 0.8478.

## 2026-08-07 — Stage B (Amendment 3, exploratory)

- Amendment 3 recorded as post-result and exploratory: the early-abort sentinel
  was suppressed so the witness distribution could be observed at all, and the
  remaining eight manifest/fold hashes were sealed before any feature
  extraction (`can`/seed0 re-verified against WTCF, `verified=1 sealed=8`).
  The H1 verdict stays permanently blocked.
- `..._stageb_v1` failed at `can`/seed2 after three cases: a zero-witness case
  makes `separation_margin` NaN, which `_atomic_json(allow_nan=False)` rejects.
  Non-finite scalars now serialize as JSON `null` with substituted paths
  recorded; the aggregator maps them back.
- `..._stageb_v2` completed all nine cases in ~90 minutes.

## 2026-08-07 — Outcome

- Verdict: `BLOCKED_BY_VALIDITY_FAILURE`; claim closed as
  `INCONCLUSIVE_WITNESS_SENTINEL / H1_BLOCKED`.
- Witness scarcity is structural: 5,832 candidates over nine cases yielded six
  accepted witnesses; eight of nine cases fell below the minimum of three.
- `unsupported` exceeded `defect_proxy` in all 3/3 measurable cases by a
  near-constant `+0.034` – `+0.039`; `principal` contracted in 9/9 cases with
  at most `0.018` spread across seeds within an object; effective rank roughly
  doubled and latent RMS fell to `0.81` – `0.89`.
- Principal-direction curvature exceeded defect-direction curvature by an order
  of magnitude (`0.043` – `0.065` vs `0.0057` – `0.0082`).
- Report:
  `skill_graph/experiments/2026-08-06_flow_latent_geometry_stage_minus1/report.md`.
- Open threat: witness scarcity may reflect the `0.85` cosine criterion at
  ~10^3 dimensions rather than the data. Resolve with the candidate
  pairwise-cosine distribution before relying on the interpretation.

## 2026-08-07 — Implementation discarded

- Filed the report, then discarded the `worktree-flow-work` branch and its
  ~2,478 lines of experiment code. The blocked diagnostic's scaffolding is not
  carried in the tree; a follow-up reimplements from the report.
- Upstreamed one defect to `main`: `patch_torch_pytree_compat()` in
  `scripts/smoke_wtcf_core.py`, verified on the dsba5 container as failing
  before and passing after. The other three defects existed only in the
  discarded code.
- Raw run artifacts remain on dsba5 at
  `/workspace/results_remote/flow_latent_geometry_20260807_stageb_v2` and
  `..._20260806_stagea_v3`; the local pulls were discarded with the worktree.
