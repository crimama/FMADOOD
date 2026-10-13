# Cross-Image-Calibrated Region Completion Stage -1

Date: 2026-08-10

Protocol: `cross-image-calibrated-region-completion-stage-minus1-20260810`

Status: `PREREGISTERED / NOT YET EVALUATED`

## Negative Evidence Intake

This is not a retry of per-image rank fusion. The earlier
`region-completion-map-level-all8-20260809` protocol was `INVALID` before its
endpoint because independently ranking every query image destroyed the pooled
cross-image score order. Per-image rank, min/max, z-score, and query-quantile
normalization are forbidden here.

This is also not a continuation of feature-space two-band DVT. That branch
preserved the medium signal but caused anomaly absorption and AU-PRO collapse.
Current and low-pass DVT are kept as separately fitted frozen score-map branches
and are combined only after scoring.

Normal-tail identification and support-bank illumination augmentation are not
reopened. The candidate targets the measured defect-side deficit: current DVT
strengthens pixels in already detectable regions without increasing component
recall, while current and low-pass DVT have different medium/small scale
profiles. Spatial complementarity between the two maps remains a hypothesis and
is falsified by the rolled-map and current-only reconstruction controls.

Likely failure basins are: calibrated max absorbing the effect, generic
morphology absorbing the effect, spatial hard-null absorption, query-normal
false-region growth, and absence of a union-component recoverability ceiling.

## Motivation

- Baseline1-core private AU-PRO@0.05 is `0.5440`, trailing the RoBiS context row
  by `0.1285`; public macro `gt_component_recall_at_oracle` is about `0.484`.
- Current DVT improves medium AP/top-5 coverage but weakens small top-1
  concentration. Fixed low-pass DVT repairs the mean small top-1 loss and has a
  larger diagnostic AU-PRO gain, while retaining less medium evidence.
- The latest spatial-illumination measurement moved shift-normal FPR by only
  `-3.5%` against a `-20%` gate and pAUROC by about `+.0007`.

Claim: a single support-fitted score scale shared across query images can make
the current/low-pass maps commensurable, after which current-DVT seeds can
complete only low-pass-supported connected regions while low-pass-exclusive
small peaks remain unexpanded.

Evidence sought: AU-PRO and matched component recovery beyond calibrated max,
current-only reconstruction, and a marginal-preserving spatial roll.

Boundary: Stage -1 is a four-object diagnostic. Its maximum positive verdict is
`PROMISING_DIAGNOSTIC`; it cannot support SuperAD superiority or deployment.

Positioning: the candidate contribution, if any, is the combination of
P16-only image-disjoint cross-image calibration, validated DVT branch role
separation, and a spatial-null-specific continuous component reconstruction.
Neither calibration nor morphology alone is claimed as novel.

## Implementable Design

### Frozen task and inputs

- `target_dataset=MVTec AD2 single-image`
- Remote data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- Local read-only mount source: `/home/hun/Volume/DATA/mvtec_ad_2`
- Objects: `can,fabric,vial,rice`
- Population: full `test_public/good,bad` for every object
- Support: exact SuperAD-selected P16 manifest, SHA-256
  `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`
- Backbone/scorer: Baseline1-core DINOv3-H+/16 `[7,15,23,31]`, layer-norm
  mean fusion, three-epoch patch-local Flow, raw latent 1-NN, density weight
  zero, no query update, no metadata, no private access
- Score branches: `position_current` and fixed `position_lowpass`
- `S128`, `S128\\S16`, TESTpublic normal statistics, labels, condition suffixes,
  and private feedback are forbidden for fitting or selecting the operator.

### P16-only image-disjoint calibration

For each object and branch, construct 16 leave-one-support-image-out folds.
For fold `i`, fit the DVT artifact, Flow, score calibration, and latent memory on
the other 15 support images, then score support image `i`. No patch or feature
from the held image may enter that fold's DVT, Flow fit, or 1-NN memory.

Pool the held-image patch scores across the 16 folds. Fit one deterministic
strictly increasing empirical-tail map per object and branch:

```text
g_b(s) = -log10 P_support-LOO(S_b >= s),  b in {current, lowpass}
```

The stored mapping uses fixed probability knots `0.001..0.999`, collapses
duplicate raw-score knots deterministically, and uses positive-slope linear
extrapolation outside the fitted range. It must not clip or reorder distinct
query scores. The same frozen `g_b` is applied to every query image of the
object. Calibration parameters, held/fit path hashes, fold counts, score counts,
and monotonicity audits are emitted in the object JSON.

### Fixed operator

Let `C=g_current(M_current)` and `L=g_lowpass(M_lowpass)`.

```text
tau_seed    = 3.0   # support-LOO upper-tail probability 1e-3
tau_support = 2.0   # support-LOO upper-tail probability 1e-2
B            = max(C, L)
S            = C >= tau_seed
M            = (L >= tau_support) OR S
```

Use 8-connectivity. For every connected component `R` of `M` that intersects
`S`, define `q_R=min(max_R C, max_R L)` and lift that component continuously:

```text
P(x) = max(B(x), q_R), x in R
P(x) = B(x), otherwise
```

Low-pass-only extreme peaks remain present through `B` but are not expanded
without a current-DVT seed. No radius, threshold, weight, object policy, or
post-hoc parameter search is allowed.

### Arms

- `A0_current`: unmodified current-DVT map.
- `A1_cal_current`: `C`; calibration-only validity control.
- `A2_cal_lowpass`: `L`; complementary branch control.
- `A3_cal_max`: `max(C,L)`; simple score-fusion/recoverability null.
- `A4_current_reconstruction`: the identical seed/support/lift rule using `C`
  for both roles; generic morphology/component null.
- `A5_ci_rc`: proposed fixed operator above.
- `N1_roll_lowpass`: proposed operator after rolling `L` by
  `(H//2,W//2)`; preserves every per-image low-pass marginal exactly.

### GPU allocation

Use every GPU that is idle at submission time without pre-empting another
user's process. Initial shards are:

- local RTX 4090 GPU 0: `can,fabric,vial`
- dsba5 host GPU 3 / container CUDA 0: `can,rice`

`can` is deliberately duplicated for cross-machine parity. Busy dsba3 GPUs are
not available resources; if a dsba3 slot becomes idle before completion, only a
preregistered object shard may be moved there without changing an arm or gate.

## Evaluation Alignment

This protocol evaluates the motivating claim because calibration-only,
calibrated-max, current-only reconstruction, and spatial-roll controls separate
score-scale, generic component lifting, and aligned cross-branch information.

The run is not same-backbone SuperAD comparable. Existing SuperAD-16 and
RN-FMLK numbers are context only. Unified fields are fixed to:

```text
superad_seg_AUROC_0.05 = N/A
superad_seg_F1 = N/A
delta_vs_superad = N/A
comparable = false
```

Primary diagnostic metric: `seg_AUPRO_0.05`. Co-primary mechanism metric:
`gt_component_recall_at_oracle`. Guardrails: `seg_AUROC_0.05`, `seg_F1`, pixel
AP, normal-image pooled FPR, threshold-free top-1/top-5 coverage/component hit,
and small/medium scale diagnostics.

## Preregistered Gates

Gate evaluation stops at the first failed validity gate. Endpoint values are not
read when validity fails.

### Validity

1. Every requested object/arm has complete prediction/GT coverage, finite and
   nonconstant maps, identical support/config/evaluator custody, and no missing
   split.
2. Each calibrator has exactly 16 folds, 15 fit images per fold, zero held/fit
   overlap, the exact P16 union, finite nonconstant score support, strictly
   increasing stored knots, and no query/private/metadata access.
3. `A1_cal_current` versus `A0_current`, and the calibrated low-pass map versus
   its raw anchor, preserve pAUROC, F1, AU-PRO, and pixel AP within `1e-3` for
   every object. This tests global monotonicity under the float16 evaluator.
4. `N1` has nonzero roll offsets and identical sorted low-pass marginal hashes
   before/after rolling for every image.
5. Within every arm, Flow parameters are unchanged before versus after memory
   population. Same-machine repeated arms require identical checksums;
   cross-machine comparisons use metric parity, not byte equality.
6. Duplicated `can` shared continuous metrics agree within `1e-3` across local
   and dsba5.
7. Dense maps are retained only until combination/evaluation, then removed from
   both completed result roots after compact evidence is pulled.

### Strict diagnostic gate

All conditions must hold on the four-object macro:

- `A5 - A0`: AU-PRO `>= +.015`, component recall `>= +.050`.
- `A5 - A3`: AU-PRO `>= +.005`, component recall `>= +.020`.
- `A5 - A4`: AU-PRO `>= +.005`, component recall `>= +.020`.
- `A5 - N1`: AU-PRO `>= +.010`, component recall `>= +.030`.
- `A5 - A0`: pAUROC and F1 each `>= -.005`.
- No object has F1 or AU-PRO delta below `-.030` versus `A0`.
- For every object, normal pooled FPR is at most
  `A0 + max(0.10*A0, 1e-4)`.
- Small top-1 coverage is not lower than `A0` by more than `.005`, and medium
  AP/top-5 coverage are not lower by more than `.010`.

Passing gives `PROMISING_DIAGNOSTIC` and licenses one frozen all-eight
replication. It is not a method claim.

### Single-step continuation gate

If strict fails, exactly one all-eight continuation is allowed only if validity
passes, `A5-A3` has AU-PRO `>=+.002` and component recall `>=+.010`, pAUROC/F1
and object no-harm gates pass, and neither spatial nor morphology null fully
absorbs both mechanism metrics. Verdict:
`KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.

### Hard stop

Use `KILL_FOR_CLAIM / NO_CONTINUE` if the calibrated-max recoverability ceiling
is absent, `A3` matches/beats `A5`, either null absorbs both mechanism metrics,
normal FPR exceeds its guard, small/medium evidence collapses, or any validity
failure exposes leakage or an unsound operator.

## Code Modification / Creation

Pending. Only the opt-in support-LOO calibration emitter, fixed map operator,
sharded runner, aggregator, and focused tests are licensed.

## Added Code Evaluation

Pending. Required before GPU execution: `py_compile`, focused pytest, Ruff,
`git diff --check`, synthetic strict-monotonicity/no-op tests, roll marginal
tests, component reconstruction tests, config/custody dry run, and local
container dependency sanity.

## Remote Execution

Pending. The dsba5 fixed project container is required. Submission must use an
immutable source snapshot, record host GPU separately from container CUDA slot,
verify the GPU is idle immediately before launch, and preserve the result root.

## Results and Analysis

Pending. No endpoint has been observed.

## Conclusion

Pending. Allowed verdict vocabulary is limited to `PROMISING_DIAGNOSTIC`,
`KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`, `KILL_FOR_CLAIM / NO_CONTINUE`, or
`INVALID`.

## Post-Conclusion Storage Cleanup

Pending. Dense `anomaly_maps/` are regenerable and must be removed only after
the fixed operator has produced compact object metrics and the final report is
recorded. Compact calibration, custody, maps hashes, logs, object JSON, and
summary JSON are retained.
