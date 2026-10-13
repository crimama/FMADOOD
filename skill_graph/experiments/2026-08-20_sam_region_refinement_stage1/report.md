# SAM Region-Refinement Stage-1 Report

Date: 2026-08-23
Protocol: `sam-region-refinement-stage1-20260820` (preregistration FROZEN 2026-08-22)
Status: `COMPLETE / KILL_FOR_CLAIM / NO_CONTINUE_SAM_MAX_LIFT`
Compute: local RTX 4090 only, container `hun_fsad_tta_local_stage1a0`
(image committed as `hun_fsad_tta_image:stage1a0-local-hf` with read-only host
HF cache mount; offline DINOv3-H+ load verified).

## Verdict

All four preregistered continuation gates failed. The frozen verdict is
`KILL_FOR_CLAIM`; no iteration lane was declared before unblinding, so the
max-lift refinement operator is closed without parameter retries
(`q_prompt`/`P`/`agg` were not tuned post-hoc).

## Results (Phase-1 objects, pooled threshold-free)

AU-PRO@0.05:

| object | B0 | M1 | N0 | N1 | N2 |
|---|---|---|---|---|---|
| sheet_metal | 0.4860 | 0.4130 | 0.4121 | 0.4860 | 0.3935 |
| rice | 0.6909 | 0.6909 | 0.6909 | 0.6909 | 0.6853 |
| can | 0.1167 | 0.0918 | 0.0895 | 0.1166 | 0.0722 |
| fabric | 0.4826 | 0.2481 | 0.2386 | 0.4826 | 0.4500 |
| **macro** | **0.4440** | **0.3609** | **0.3578** | **0.4440** | **0.4002** |

Secondary canonical metrics (macro): pAUROC@0.05 B0/M1 = `0.8437/0.8132`;
Seg-F1(raw oracle) B0/M1/N2 = `0.4796/0.4323/0.5184`.

## Gate outcomes

1. M1 >= B0 by >= +0.01 macro AU-PRO on >= 3/4 AND macro-positive:
   **FAIL** (`0/4` wins; macro delta `-0.0831`).
2. M1 component recall >= B0 macro-positive: **FAIL** (delta exactly `0.0000`
   on all four objects).
3. M1 beats N0/N1/N2 on AU-PRO macro: **FAIL**
   (`N1=B0 0.4440 > N2 0.4002 > M1 0.3609 > N0 0.3578`).
4. Safety: anomaly-preservation **PASS** (mean defect-score ratios
   `1.0044/1.0016/1.0520/1.0286`); per-object no-harm **FAIL**
   (`fabric -0.2346`, `sheet_metal -0.0730`, `can -0.0248`, `rice -0.0000`).

## Mechanism diagnosis

1. **The lift dilutes rather than absorbs.** The defect-preservation invariant
   held on every image (no pixel exceeded `base.max()`; mean defect scores
   unchanged-to-slightly-raised), but lifting whole SAM regions by their
   90th-percentile raises large stretches of normal tissue around prompted
   components. On diffuse-texture `fabric` and fine-structure `sheet_metal`
   this floods the ranking with false positives and collapses AU-PRO.
   Evidence was not destroyed; it lost spatial contrast.
2. **Anomaly-conditioned prompting added nothing.** `N2_map_agnostic_SAM` beat
   `M1` at macro (`0.4002 vs 0.3609`); once generic region lifting is
   controlled, prompting from actual map peaks contributes negatively.
   `N1_generic_morph` (fixed 3x3 grayscale closing) ties `B0` to four decimals,
   so plain smoothing is a no-op at this map scale.
3. **Component-recall endpoint is structurally insensitive to max-lift
   operators.** The global-oracle threshold pins to `max(scores over positive
   pixels)`; because `M' <= base.max()` pointwise and the argmax is preserved,
   no component-recovery decision can flip. The exact `0.0000` deltas are a
   property of the endpoint contract, not evidence of neutrality at other
   operating points. Any Phase-2 redesign must first fix this measurement
   contract before re-testing region operators.
4. `rice` is a near no-op (`-0.0005`): its frozen `q_prompt=8.156` sits high
   relative to the map distribution, producing almost no prompt components.

## Guideline compliance notes

- G-2 two-lane: direction negative AND magnitude far below gate -> plain kill;
  `NEAR_MISS` does not apply.
- G-6: fabric-driven harm is reported per object; the failure is not
  fabric-only (`can/sheet_metal` also lose), so no `OBJECT_DRIVEN_FAILURE`
  downgrade applies.
- G-1 status: this is strike 1 for the "frozen foundation-model region prior
  for region coherence" question family. The family is not closed; any next
  candidate requires the measurement-contract repair from diagnosis 3 plus a
  preregistered `failure-basin divergence` section.

## Custody and access

- Support manifest SHA-256 identical across objects (single fixed support JSON):
  see `summary.json` `custody_repair_20260823.support_manifest_sha256_per_object`.
- Base maps: 564 TIFFs, tree SHA-256 `2e90e46043e026a6de11...` (repaired hash;
  the in-evaluator hash returned the empty-input value because pathlib does not
  traverse symlinked directories).
- SAM checkpoint SHA-256 `03168895e7477b2e...`; code tree SHA recorded at
  evaluation time (includes two pre-results plumbing fixes documented in the
  preregistration amendment: evaluator path-type fix and PYTHONPATH launch fix;
  GT pixels were not read before either fix).
- GT opened only after all arm maps existed and passed manual coverage +
  invariant verification; `query_or_test_private_access=false`;
  `benchmark_comparable=false`.
- Dense artifacts (`refined_maps/`, basemap TIFFs) removed after compact
  summary + hashes were secured; regeneration is deterministic from the
  frozen configs and manifests.
