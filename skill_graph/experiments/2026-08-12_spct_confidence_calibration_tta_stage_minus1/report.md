# SPCT Confidence, Calibration, Synthetic Gate, and Key-TTA Stage -1 Report

Date: 2026-08-12
Status: `COMPLETE`
Protocol verdict: `VALID_NO_SIGNAL`
Research verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`

See [preregistration.md](preregistration.md). The strict aggregate is
`results/remote_runs/merged/spct_confidence_stage_minus1_all14_v2/summary.json`
(SHA-256 `5415696802d2839f53270230aa634b579d4bd2faf094a022838176ee1dd0c340`).

## 1. Motivation

The prior SPCT diagnostic left a useful but incomplete result: normal support
transport could help, while a shifted query did not reliably reveal when to
use that transport. This experiment tested four bounded ways to resolve that
decision without test-time labels or query-memory insertion: support-LOO
confidence, source-block conformal calibration, synthetic-anomaly activation,
and key-only TTA. It also paired the fixed 16-shot set with all available
`train/good` support to test the user's support-scaling hypothesis.

This is not a retune of a passive score-fusion branch. The new evidence source
was support-source reliability and exchangeability, with activation, global,
patch-IID, label-swap, mask-roll, and wrong-part TTA nulls.

## 2. Implemented design

- Dataset boundary: `MVTec AD2 single-image`, four objects (`can`, `fabric`,
  `vial`, `walnuts`), normal `train/good` only.
- Paired held probes: the same sealed 16 sources under both budgets.
- Budgets: `support16` and `full_train_good` (`412/387/291/432` sources).
- Four folds, eight signed photometric codes, local/broad held synthetic
  defect families, and the frozen 14-arm panel.
- Query/test access, GT access, memory updates, persistent cross-query state,
  and dense-map persistence were all false. `O0` remained a forbidden
  clean-key oracle and was excluded from every verdict.

## 3. Evaluation alignment

The run directly tests whether support-derived confidence/calibration can
choose or calibrate normal transport, whether synthetic anomalies can train
that choice, whether key-only TTA helps beyond a wrong-part null, and whether
more normal support improves the underlying representation match.

It does **not** use full `test_public`, `seg_F1`, or a same-condition SuperAD
artifact. Therefore `benchmark_comparable=false`; these pseudo-defect pAUROC
values cannot support a SuperAD or paper-level method claim.

## 4. Execution and code changes

Run `spct_confidence_stage_minus1_all14_v2` completed all eight shards:

- dsba3 host GPU3: `vial -> walnuts`, each `support16 -> full_train_good`.
- dsba5 host GPU3: `fabric -> can`, each `support16 -> full_train_good`.

The zero-result `all14_v1` snapshot was excluded after profiling found
per-query quadratic conformal preparation and TN0 scalar synchronization.
The v2 repair preserved formulas while preparing calibration references once
per fold and vectorizing query scoring.

At aggregation, 657 individual maps—all `CN0_global_conformal`—were constant
because finite source-block conformal p-values are discrete. Every CN0 shard
was globally nonconstant (13–65 distinct scores, finite counts complete,
positive range/std). The validator was corrected to retain row-level finite
and custody checks while enforcing non-collapse at arm/shard scope. This
changed no experimental score.

## 5. Verification

- All eight shards: `validity=true`, `3,584` records each, exact manifests and
  held-source exclusion.
- Final aggregate: `validity_ok=true`, no invalid reasons, 10,000 fixed-seed
  hierarchical bootstrap replicates.
- Oracle rows excluded from all verdicts.
- Focused post-result verification: 35 tests passed; Ruff, `py_compile`, shell
  syntax, compact-artifact checks, and checksum verification passed.
- No regenerable dense result tree exists. Both immutable remote source
  snapshots were removed after the verified pull; compact remote/local result
  evidence was retained.

## 6. Results

Macro values below are decimal diagnostic segmentation pAUROC@0.05. FPR is
held-normal FPR at the support-fit q95 threshold.

| Arm | support16 pAUROC / FPR | full pAUROC / FPR | full - 16 pAUROC |
| --- | ---: | ---: | ---: |
| B0 retrieval raw | 0.655671 / 0.052857 | 0.720301 / 0.056108 | +0.064629 |
| B1 unconditional tangent | 0.689831 / 0.056352 | 0.769953 / 0.070388 | +0.080122 |
| G0 LOO confidence gate | 0.655671 / 0.052887 | 0.720301 / 0.056130 | +0.064629 |
| C0 hierarchical conformal | 0.644601 / 0.000000 | 0.746056 / 0.000439 | +0.101455 |
| S0 synthetic gate | 0.655652 / 0.052916 | 0.720284 / 0.056121 | +0.064632 |
| T0 key-only TTA | 0.655089 / 0.164944 | 0.718885 / 0.194826 | +0.063796 |
| TN0 wrong-part TTA | 0.655148 / 0.165383 | 0.719881 / 0.195852 | +0.064733 |
| O0 clean-key oracle | 0.702626 / 0.054705 | 0.816969 / 0.050700 | +0.114343 |

Support scaling is the only robust positive result. Full support improved B0
on all four objects by `+0.048277/+0.086877/+0.008824/+0.114540`; its 10,000
draw macro CI was `[+0.022957,+0.107459]`. B1 likewise improved all four by
`+0.080122` macro, CI `[+0.039786,+0.117664]`.

The four proposed mechanisms nevertheless failed their independent gates:

- **G0:** activated only `0.000666` (16-shot) and `0.000010` (full), so it
  collapsed to B0. Against B1 its pAUROC deltas were `-0.034160` and
  `-0.049653`; the gate nulls were identical in pAUROC.
- **C0:** full support produced a real tradeoff—`+0.025755` pAUROC versus G0
  while reducing FPR by `-0.055692`—but coverage was `0.999975`, outside the
  preregistered `[0.92,0.98]`, and the support16 result lost `-0.011070`.
  It did not meet the joint calibration gate.
- **S0:** was effectively neutral versus G0 (`-0.000019` support16 and
  `-0.000017` full), failed the label-swap null, and gave no anomaly-aligned
  activation evidence.
- **T0:** changed pAUROC by only `-0.000583`/`-0.001416` versus G0, was matched
  or beaten by wrong-part TN0, and raised FPR by `+0.112057/+0.138696`.

## 7. Analysis

The central inference is narrower than “support cannot model query nuance.”
More normal support clearly improves retrieval and tangent matching, including
the clean-key oracle. The failure lies in **query-conditional selection**:
support-only LOO reliability could not tell when tangent projection was safe,
synthetic anomalies did not teach that distinction, and key refinement moved
normal scores without anomaly-specific benefit.

Full support also did not close the clean-key oracle gap: B1 reached
`0.769953` while O0 reached `0.816969`. Thus both coverage and shifted-query
retrieval identity remain limiting. C0 shows that source-aware calibration can
almost eliminate normal false positives, but its near-100% coverage is
overconservative rather than well-calibrated discrimination.

## 8. Conclusion

Strict claim: **no**. All four new mechanisms are `NO_SIGNAL`, the run is not
SuperAD-comparable, and TTA specifically fails its matched-null and FPR gates.

Continuation: **yes, for the best absolute deployable arm—not for proving a
monotone support-scaling law or retuning these gates.** The exact LOO threshold,
synthetic activation head, conformal mixture, and three-step key-TTA branches
are closed in their current form.

Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC` (aggregate protocol verdict
`VALID_NO_SIGNAL`).

## 9. Single next diagnostic

Promote `full_train_good/B1_unconditional_tangent`, the best usable absolute
arm (`pAUROC=0.769953`, `AUPRC=0.500850`), directly to a full TESTpublic
segmentation evaluation. The numerically higher O0 is an unavailable clean-key
oracle, and SN0's `+0.000038` over B1 is an invalid label-swap control rather
than a deployable method.

The next run should cover all eight public objects and report
`seg_AUROC_0.05`, best-threshold `seg_F1`, AU-PRO/map coverage, a matched-budget
full-memory DINO kNN/RN-FMLK baseline, and SuperAD-16 as explicitly
budget-incomparable paper context. No support64 interpolation is required: the
decision criterion is absolute TESTpublic performance. Hard-stop B1 only if it
is beaten by the matched full-memory baseline or loses map/F1 coverage despite
its pseudo-defect pAUROC.
