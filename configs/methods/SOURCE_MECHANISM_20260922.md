# Source-fixed threshold and paired shift mechanism audit

User authorized local + dsba3 execution on 2026-09-22. This immediate stage
addresses the mechanism and fixed-threshold gaps identified after the factorial.
It is a post-hoc-motivated, prospectively fixed development audit, not final test.

## Hypotheses and scope

Does source-tail scaling and residual gating improve operating behavior under
actual acquisition shift, beyond increasing condition-specific AUPRO? Do the
normal/defect responses explain that behavior? Reducing a score mechanically
does not establish nuisance removal or defect preservation.

SM_B/C/D/E exactly reproduce FF_B/C/D/E: q95+ungated, MAD+gated, q95+gated
(RSG3), q95+deep-only. Normal16 seed0 fixed SD_support_16.json; full8,
32 class evaluations. No parameters, backbone features, bank sampling, resolution,
CQ or normal inference are changed. Multi-resolution/bagging remain fixed;
no trainable parameters. D then B/C/E on each lane. Stop after these32 cells;
shot1/4/16 and seed replication are deferred, not automatically launched.

## Source calibration and operating metrics

Per class, select the first16 sorted train/good regular images excluding actual
support paths; reject support duplicate content. Same calibration images across
all four arms. These images were previously available development data (and may
overlap earlier diagnostic calibration), not an untouched holdout. They never
enter fitting or prediction-state updates. Source calibration supplies16 extra
normal images beyond the16 method supports: explicitly a16-support+16-calibration
operating evaluation, not a claim that total normal-data cost remains16.

Before querying test_public, run calibration inference and pool ALL native pixel
scores into float16 ordered histograms. Separate raw and CQ thresholds; fixed
source nominal FPR1% is PRIMARY,0.1% and5% are sensitivity points. Choose the
smallest order code whose cumulative count reaches ceil((1-FPR)*N), then use
strict score>threshold in the same float16 contract. Record actual calibration
FPR, threshold/code, content hash and all paths. Discrete ties may make actual
source FPR conservative. No target label, score or acquisition condition tunes
these thresholds. Record16 leave-one-calibration-image-out thresholds and evaluate
the primary threshold's min/max sensitivity; this is NOT a confidence interval.

At each frozen operating point report normal pixel FPR, defect pixel TPR and
component-balanced PRO (8-connectivity), for raw/CQ and every actual condition.
Provide good-image-only normal FPR separately from normal background in all
queries. Region-weighted aggregate PRO and per-scene values are both retained.
Nominal source FPR is not assumed to hold on regular test or shifted test.

## Mechanism evidence

Capture actual pre-normalization layer NN distances, normalized layer scores,
deep mean, full shallow term, hypothetical gated residual and actually used
shallow term during the SAME query inference. No extra NN/backbone passes for
component tracing. Average bags, bilinear-project each resolution to native
space and average resolutions; deep+used_shallow must reproduce pre-CQ output
within float32 interpolation tolerance (rtol3e-5,atol3e-6). Preserve regional
normal/defect mean/std and standardized gap, not dense arrays. All labels are
consumed only by the evaluator after prediction. Gate state never depends on GT.

Pair regular/shift_1 by class/kind/scene identity. Compare region summaries, NOT
pixelwise subtraction (geometric alignment has not been established). Save paired
normal/defect component changes and fixed-threshold FPR/TPR/PRO changes. Report
other conditions descriptively, without choosing the most favorable shift.

## Interpretation gates (fixed before reading these results)

- Execution validity:32 complete, paired inputs/banks/calibration across arms,
  finite full coverage, same scores/AUPRO as prior matched FF arms; any mismatch
  is diagnosed before scientific interpretation. Safe runtime retry stays inside
  the32-cell budget and preserves failed logs.
- Primary robustness contrast D−B at CQ source FPR1%, regular→shift_1: smaller
  class-macro FPR increase AND no worse class-macro shift_1 PRO (both checked),
  with at least6/8 classes not worsening the FPR increase. Also report direct
  shift_1 FPR/TPR, regular operating quality, exceptions, and LOO sensitivity.
  A lower FPR purchased with worse PRO is a tradeoff, not defect preservation.
- Report D−C (scaling) and D−E (shallow contribution) without cherry-picking.
  Mechanism support requires concordance with operating results, not just lower
  normal mean scores or gate activation. Raw/normalized quantities across
  differently scaled arms cannot be compared as if they shared physical units.
- No significance/generalization/shot-dependence claim from seed0. Failure of
  this framing does not negate prior AUPRO evidence; retain both results.

## Execution and artifacts

Existing local4090: can/fabric. Existing dsba3 hun_fsad_tta_012 GPUs0/1/2/3:
sheet_metal, rice/vial, fruit_jelly/wallplugs, walnuts. Explicit UUIDs; same
class stays on one device/runtime across arms. Remote isolated Python3.11.16 /
Torch2.6.0+cu124. Reuse container and content-addressed local snapshot workflow;
no Git push or provisioning. Generic experiments.py run/collect only.
Per-class timeout14400s. Budget32 evaluations +16 calibration predictions each
(512 extra normal-image inferences); four resolutions each. Calibration wall
time is recorded separately from fitting and ordinary query prediction.
Keep configs, manifests, quantitative summaries and training/run logs. Maps and
weights OFF. Retain remote logs; pull only compact evidence. Do not stop container.
