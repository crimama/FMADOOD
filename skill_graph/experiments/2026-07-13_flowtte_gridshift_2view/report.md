# FlowTTE q4096/guided stack and two-view grid-shift smoke

## 1. Motivation

The retained Phase 1 q4096 condition alignment and Phase 2 RGB guided-r8
filter are the two no-harm winners and need a fixed-order composition check.
Separately, the two-view diagnostic tests whether patch-16 phase sampling
suppresses small defects and boundaries. This is an active-observation
mechanism diagnostic, not a retune of a closed threshold, radius, descriptor,
or support-admission branch. The likely failure basin is near-identity/no-op
or phase drift that requires calibration.

## 2. Implementable design

- Dataset: MVTec AD2 single-image public shadow, all eight objects.
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`.
- Immediate baseline: retained FlowTTE gap-decomposition anchor at
  `/workspace/results_remote/flowtte_gapdecomp_anchor_20260712_v1`.
- Task A arms: identity, q4096 only, guided-r8 only, q4096 then guided-r8.
- Task B views: `(0,0)` and resized-tensor `(8,8)`, with native back-alignment.
- Task B banks: original support bank for arm A; shifted-support DVT + flow
  refit for arm C. Shifted query extraction is shared.
- Metrics: pooled float16 oracle F1/threshold, `seg_AUROC_0.05`, GT component
  recall, boundary tolerant F1 at 0/4/8 px, and normal-image mean FPR.
- This smoke is diagnostic-only; it does not establish a same-condition
  SuperAD or RN-FMLK method claim.

## 3. Evaluation alignment

The view0 path uses the frozen Phase 3 anchor configuration and exact
per-object F1 parity as a hard stop. All arms share object population,
supports, preprocessing, metric granularity, and evaluator. Arm C changes
only the preregistered support/query sampling phase and refits DVT/flow from
the same pre-fit RNG state. The user owns the KEEP decision.

## 4. Code modification / creation

- `scripts/analyze_flowtte_stack_qm_guided.py`
- `scripts/run_flowtte_stack_qm_guided_remote.sh`
- `scripts/run_flow_tte_gridshift_2view.py`
- `scripts/run_flow_tte_gridshift_2view_remote.sh`
- `src/flow_tte_gridshift_2view.py`
- `tests/test_flowtte_stack_qm_guided.py`
- `tests/test_flow_tte_gridshift_2view.py`

All runtime additions use separate entrypoints; the existing default runner
path is unchanged.

## 5. Added code evaluation

- Full pytest: `418 passed, 5 warnings`.
- Focused new/existing module suite: `24 passed`.
- Python compilation: passed.
- Remote shell syntax: passed.
- Dense variant TIFF writer: absent; Task B asserts zero TIFFs before marking
  completion.

## 6. Remote experiment execution

Target container: `hun_fsad_tta_012` on dsba3, host GPUs 0/1/2/3 for Task B.
The local preset is sourced but excluded from tar sync. Two Task A launch
attempts were rejected by the managed execution sandbox before SSH with
`socket: Operation not permitted`. Therefore no remote process was launched,
no result was pulled, and the retained anchor was not modified.

## 7. Evaluation results and analysis

Pending remote execution. Historical compact artifacts establish only the
three requested single-module expectations: identity `0.530635`, q4096
`0.539296`, guided-r8 `0.550676`; the new analyzer must reproduce them within
`1e-6` before Task B may launch. No stack or grid-shift result is claimed.

## 8. Conclusion

`BLOCKED_DATA` for execution evidence in this managed session: implementation
and local verification are complete, but remote artifacts are unavailable
because outbound SSH is prohibited. The exact continuation is to run Task A
through its prepared controller; hard-stop on any reference mismatch, else
launch the prepared four-GPU Task B controller and enforce exact view0 parity.

## 9. Post-conclusion storage cleanup

No dense output was created locally or remotely. The retained anchor is
explicitly preserved. Task B is metrics-only and its controller refuses a
completion marker if any TIFF exists.
