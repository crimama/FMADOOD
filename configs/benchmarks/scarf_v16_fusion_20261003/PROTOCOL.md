# SCARF v16: strong simple fusion comparators

Recorded before these new runs; directory date is Korea local time, 2026-10-03.
User authorized the v16 revision, additional experiments on the four assigned
dsba3 GPUs, and a LaTeX draft. This is a bounded development comparison.

## Question and method boundary

Does support-calibrated excess fusion provide a useful advantage over deep-only,
equal layer averaging, globally weighted shallow addition, and a hard gate?
The main uncertainty is incremental value over strong simple comparators, not
recovery relative to unrestricted shallow addition. No method is changed.
Use existing `fusion_factorial` modes and `layer_fusion_gamma`.

For q95-calibrated scores, let d=(z11+z23)/2 and t=Q99(D5)/max(Q95(D5),1e-4).
Candidates are d+lambda*z5 for lambda in {0, .125, .25, .5, 1, 2};
the hard gate is d+z5*1[z5>t], and SCARF's fusion rule is d+max(z5-t,0).
Existing equal mean is (z5+z11+z23)/3. It is proportional to lambda=.5
before score quantization. Since the evaluator casts raw scores to float16,
exact lambda=.5 is also evaluated to avoid assuming identical finite-precision
ties. This does not repeat the existing mean3 study.

## Fixed conditions

- MVTec AD 2 `test_public`, all eight classes; already seen development data.
- Seed 0; 2 and 16 normal supports from the ordered existing
  `configs/methods/VOID_support_{2,16}.json` manifests.
- Frozen DINOv2-L/14, layers [5,11,23], short edges [448,672,896,1120],
  4 random overlapping banks, cap256/support, k1, q95 normalization, q99 gate.
- Original FP32 feature path and chunk128; no optimized runner or autocast change.
- Existing `no_mask_no_rotation` preprocessing and native query/evaluator shapes.
- Bag mean, scale mean, existing CQ postprocessing fixed across arms.
- Noise routing is disabled, matching the original VOID/figure-pilot fusion
  controls. This isolates fusion; it is not a new complete-system benchmark.
- Existing compact fusion diagnostics enabled, including raw/CQ metrics and
  acquisition strata. No additional backbone calls, saved maps, or weights.
- Primary: macro AUPRO at FPR<=.05, higher better. Secondary: pixel AUROC at
  FPR<=.05. Oracle F1 is diagnostic only. Report class deltas and regressions.
- Labels are used after prediction for evaluation and for the declared global
  weight selection, never fitting or prediction. Neither this run nor later
  support repeats constitute an untouched holdout.

## Reuse and selection

Reuse deep-only, full addition (lambda=1), and residual for 2/16 from
`results/imported/void_instrumentation_20260922`, and q95 mean3 from
`results/fig1_pilot_20260929/{stage2_full8,remote_full8}`.
Before interpreting paired deltas verify ordered support content, query/GT
content, backbone weights, bank hashes and evaluator contract per class.
Absolute prefixes or manifest file hashes alone do not establish a mismatch:
VOID16 and SD16 use different root strings, requiring relative identities and
content hashes. Preserve run/config/environment differences explicitly.

Select exactly one global lambda using **2-shot seed0 full8 macro AUPRO** across
the six candidates, including lambda=0. Break a numerical tie (absolute
difference <=1e-10 in unscaled metric) toward the smaller lambda. Lock the choice
before comparing 16-shot results. The 16-shot sweep is sensitivity/transfer
evidence, not an additional selection objective. Do not tune by class or report
an oracle per-shot best as the deployed baseline.

## Execution budget and validity stop

10 new configurations (four nontrivial positive weights plus hard gate, each
at 2/16 shots), eight classes each: **80 new class-runs**. Reused cells are not
counted as new execution. Four shards use two classes each, one worker per
already exposed dsba3 GPU UUID:

| Shard | GPU UUID | Classes |
|---|---|---|
| gpu0 | GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f | can, fruit_jelly |
| gpu1 | GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae | fabric, walnuts |
| gpu2 | GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639 | rice, wallplugs |
| gpu3 | GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62 | sheet_metal, vial |

Order all 2-shot configs first (w0125,w025,w05,w2,hard), then all 16-shot configs
in that order. Existing container `hun_fsad_tta_012` is reused. Per-class timeout
7200 seconds. Use `scripts/experiments.py run` and its recorded status/logs.
Stop a failing shard for nonfinite metrics, missing coverage, or execution error;
retain failed artifacts and retry only a diagnosed safe execution issue.
No performance-based early stopping, discarded classes, or extra parameter search.

The server is configured for `REMOTE_CODE_MODE=local`; use the existing
`run_remote.sh update-code` immutable snapshot path, with its exclusive lock,
to record the exact validated local source/config release. No git push or
container replacement is needed. Do not activate code during running experiments.

## Collection and subsequent decision

Pull only compact metrics/config/manifests/status with the existing remote helper.
Combine disjoint shards using `scripts/experiments.py collect --input-dirs` and
update the canonical paired Markdown/CSV ledger through that collector. Preserve
partial status until all expected cells complete. Store reuse/provenance checks,
global-weight selection and the final interpretation in the suite comparison.md.

If SCARF has a defensible advantage over deep-only and the locked weight, the
next bounded study adds missing paired deep/weighted support repeats and frozen
transfer comparisons on MVTec/VisA. If results are similar, quantify that rather
than claiming superiority. If SCARF loses, revise the paper's central claim and
review one bounded method revision before allocating broader experiments.
Do not automatically launch the full transfer/repetition grid before this
decision; this is a sequencing decision within the authorized research task.

## Audit-driven amendment — 2026-10-03 03:31 KST

The initial 80-cell candidate plan above is preserved. The completed
[reuse audit](../../../results/remote_runs/dsba3/scarf_v16_fusion_20261003/REUSE_AUDIT.md)
found historical VOID gated/mean3 bank and runtime mismatches against off/all,
and a package-code mismatch between every historical arm and the new suite.
The inspected new fabric/rice banks and runtime match historical off/all, but
a subset bridge would not establish an identical-code primary comparison.
This amendment is driven by provenance, before any new full8 candidate macro
exists; it does not add weights or alter the selection objective.

**Decision: rerun all three controls at both shots, full8: 48 additional
class-runs; total 128.** Use the existing immutable-release configurations
in configs/benchmarks/void_instrumentation: void_s{02,16}_{off,all,gated}.yaml.
All common fields match the new weight configs; only shallow mode and gamma
differ as intended. Controls run under the same source snapshot, runtime,
GPU/class allocation and output settings as the candidates. No code activation
or container/GPU exposure change occurs.

The same four shards are scheduled after their respective candidate queues
exit successfully, using the existing scripts/experiments.py run. A one-shot
process dependency checks the predecessor's complete status before invoking
the runner. A failed predecessor blocks that shard's controls. Control order:
off/all/gated at 2 shots, then off/all/gated at 16 shots. Compact results go to
the suite's controls/gpu{0,1,2,3} directories. Launch commands, PIDs, config
hashes and snapshot are in launch_controls.json in the results root.

**The primary global-weight selection uses the new off/all endpoints.**
Do not select until all six two-shot candidates have complete, valid full8
results and matched support/query/GT/backbone/bank/config/runtime provenance.
Keep the 1e-10 tie rule and lock before interpreting sixteen-shot candidates.
Old mean3/VOID scores remain descriptive historical context; exact w05 is
the current-suite corresponding weight. The separately recorded 16-shot
FF_A--FF_E factorial retains its own within-class matching scope and must not
be conflated with the cross-environment VOID shot sweep.

Collect candidate shards and control shards separately through the generic
collector, then compare their preserved artifacts in the suite report. Do not
combine unequal-label shard plans as if they were one disjoint-class plan.
No full8 result or selected weight is recorded by this amendment.

## Partial collection amendment — 2026-10-03 04:08 KST

Completed candidate artifacts are registered while the remaining runs continue,
using the existing collector's `--import-runs` mode with all ten configs and an
explicit eight-class scope. The stable output is
`results/remote_runs/dsba3/scarf_v16_fusion_20261003/candidates_full8`; its name
declares the intended scope, while `status.json` remains `partial` until all
80 candidate cells are present. Only a label with eight completed classes
receives a macro. Active config-only directories are skipped, not counted as
scientific failures. Subsequent refreshes use the identical import sources,
config/label order, object scope, split, and output identity. Do not switch this
owned import to `--input-dirs`. Controls use a separate `controls_full8` import
with their six-config plan after completed controls become available.

The collector's first label, w0125_s02, is an accounting reference for displayed
deltas, **not a selected weight or the primary scientific comparator**. The
selection gate above is unchanged. Class and macro records were added through
the collector to the canonical Markdown/CSV pair; 5,348 unrelated CSV rows
were verified unchanged in their original order. Detailed ongoing status and
the interpretation boundary are recorded in the suite's comparison.md.
