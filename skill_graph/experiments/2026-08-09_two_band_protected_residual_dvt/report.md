# Two-Band Protected-Residual DVT Diagnostic

Date: 2026-08-09

Verdict: `ANOMALY_ABSORPTION / NO_CONTINUE`

Scope: all-eight MVTec AD2 mechanism diagnostic; SuperAD/RN-FMLK `N/A`; `comparable=false`

## Negative Evidence Intake

The preceding fixed low-pass DVT diagnostic repaired the mean small-defect top-1 loss but retained only
`67.48%/71.71%` of current DVT's medium AP/top-5 gain. Correct low-frequency position also beat its
equal-energy shuffled null. This authorized exactly one structural test: correct only the low-frequency
branch and pass the query high-frequency residual through untouched. The experiment did not reopen DVT
alpha, Gaussian scale, scorer, threshold, score fusion, support admission, or query adaptation.

## Motivation

The test asked whether the current DVT tradeoff could be separated by frequency: preserve the validated
position-specific medium-defect benefit in a corrected low band while protecting small local defects in an
uncorrected high band. A generic doubled-dimensional band split and an equal-energy shuffled-position arm
were mandatory controls.

## Implementable Design

For the fixed Gaussian operator `G(sigma=2.0, truncate=3.0)`, every arm used
`concat(G(x)-b, x-G(x))/sqrt(2)` before the unchanged three-epoch Flow and raw 1-NN scorer.

- `B0_band_identity`: `b=0`, the dimension/band-split baseline.
- `B1_band_position`: `b=G(mu_p-g)`, the candidate.
- `B2_band_position_shuffled`: the same correction rolled by `(H//2,W//2)`, the hard null.

The target was single-image MVTec AD2 at `/home/hunim/Volume/DATA/mvtec_ad_2`, all eight objects and full
`test_public/good,bad`, with exact fixed SuperAD-selected support-16, DINOv3-H+ layers `[7,15,23,31]`,
`layer_norm_mean`, resolution 672 except `sheet_metal=448`, density weight 0, and no query update.

## Evaluation Alignment

All arms shared support basenames, TESTpublic rows and ground-truth counts, preprocessing, Flow, scorer,
and scale-bin definitions. This is a diagnostic against B0, current DVT anchors, and B2; it is not a
same-condition SuperAD method comparison. Accordingly, `superad_seg_AUROC_0.05`, `superad_seg_F1`, and
`delta_vs_superad` remain `N/A`, and `comparable=false`.

## Code Modification / Creation

- `src/flow_tte/denoising.py`: fixed two-band transforms and frozen Gaussian/scale contract.
- `scripts/flow_tte_mvtec_ad2_core.py`: transform dimension, correction-energy, roll, and branch audit fields.
- `scripts/run_flow_tte_mvtec_ad2.py`: two-band CLI modes.
- `scripts/run_flow_tte_phase3_scorer_suite.py`: raw-1NN-only execution and compact scale evidence.
- `scripts/run_two_band_protected_residual_dvt_diagnostic.sh`: arm-selectable all-eight runner.
- `scripts/aggregate_two_band_protected_residual_dvt_diagnostic.py`: custody, hard-null, no-harm, and
  local/dsba5 B0 parity gates.
- `tests/test_two_band_dvt_transform.py` and
  `tests/test_aggregate_two_band_protected_residual_dvt_diagnostic.py`: transform and verdict regressions.

## Added Code Evaluation

- Related regression suite: `37 passed`.
- Final two-band transform/aggregator suite after parity and high-branch strengthening: `9 passed`.
- New two-band files: Ruff pass, Python compilation pass, shell syntax pass, and `git diff --check` pass.
- Repository-wide Ruff was not clean because the pre-existing shared runners still expose 18 unrelated
  style/complexity findings; no new two-band-only Ruff finding remains.
- The B0/B1/B2 high halves are compared directly on the same input in the transform regression test.
- Local and remote runtime hashes matched byte-for-byte for `denoising.py`, the core, and the Phase-3
  scorer runner. The orchestration shell differed only because the final parity requirement was added
  locally after the remote snapshot was launched; this did not alter either runtime arm.

## Remote Execution

- Local fixed container `hun_fsad_tta_local_stage1a0`: B1 then B0, exit 0.
- dsba5 fixed container `hun_fsad_tta_012`, exposed host GPU 3 / container CUDA slot 0: B2 then duplicated
  B0, controller exit 0.
- Remote compact pull: `results/remote_runs/dsba5/two_band_dvt_20260809_v1`.
- Local primary root: `results/two_band_protected_residual_dvt_20260809/local`.
- B0 parity compared 4,740 standard and per-image scale metrics. Maximum absolute delta was
  `0.0002775465`, below the frozen `0.001` tolerance, with zero custody failures and zero mismatches.
- Support-manifest SHA-256:
  `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`.

## SuperAD Baseline and Unified Metrics

This stage cannot produce a method-level SuperAD claim. Absolute B1 diagnostic metrics were:

| Field | Value |
| --- | ---: |
| `method_seg_AUROC_0.05` | 0.836181 |
| `method_seg_F1` | 0.518099 |
| `method_seg_AUPRO_0.05` | 0.438063 |
| `superad_seg_AUROC_0.05` | `N/A` |
| `superad_seg_F1` | `N/A` |
| `delta_vs_superad` | `N/A` |
| `comparable` | `false` |

## Results and Analysis

All validity gates passed: all-eight coverage, support/config/GT custody, exact `2C` output dimension,
fixed `1/sqrt(2)` scale, B1/B2 equal correction energy, nonzero B2 roll, unchanged high branch, tie-area
audits, and local/dsba5 parity.

Macro contrasts:

| Contrast | Small AP | Small top-1 | Medium AP | Medium top-5 | pAUROC | F1 | AU-PRO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 - B0 | -.019267 | -.057738 | +.073014 | +.070049 | +.029908 | +.057124 | -.060379 |
| B1 - B2 | -.019644 | -.079536 | +.096389 | +.106679 | — | — | — |

B1 preserved `96.44%/95.69%` of the frozen current-DVT medium AP/top-5 gains and decisively beat the
shuffled null on both medium endpoints. The candidate therefore retained a real position-specific medium
signal. It nevertheless failed the purpose of the design: the protected high branch did not protect small
defects after concatenation and Flow. Mean small AP/top-1 both fell, and B1 was worse than the shuffled
null on small top-1.

The mean pAUROC/F1 gains hide severe object and scale collapse. `rice` medium AP fell `-.174271` and F1
fell `-.078695`; `sheet_metal` small top-1 fell `-.213864`; and `wallplugs` F1 fell `-.073774`. These cross
the preregistered catastrophic no-harm thresholds. AU-PRO also fell `-.060379`, showing that improved
best-threshold F1 did not preserve region-level localization quality.

## Continuation Assessment

- Strict diagnostic gate: fail.
- Continuation gate: fail.
- Hard-null absorption: false; the medium positional effect is real.
- Catastrophic no-harm failure: true.
- The result supports no strict method claim and no further diagnostic. The failure is not insufficient
  medium retention but anomaly absorption/reweighting introduced by the two-band representation and Flow.
- Do not tune band weights, sigma, dimensional scaling, Flow capacity, score fusion, or object-specific
  policies. Those would be post-hoc retuning of a branch that failed its direct preservation gate.

## Conclusion

`ANOMALY_ABSORPTION / NO_CONTINUE`. The two-band decomposition preserved nearly all of the medium-defect
benefit and passed the positional hard null, but it amplified rather than repaired the small-defect loss and
caused categorical no-harm failures. This closes the scale-selective DVT continuation branch. The retained
asset is a clean mechanistic boundary: low-frequency position correction carries genuine medium-scale
signal, but simply concatenating an untouched high residual before the existing Flow does not preserve
local anomalies.

Primary artifact:
`results/two_band_protected_residual_dvt_20260809/local/summary.json`
(`sha256=749d7b7142f0f40dfc336ccd317db4e58b7cd146796c5531020981911f5a6d4b`).

## Post-Conclusion Storage Cleanup

No `anomaly_maps/` directory was created under the local or pulled remote result roots. Compact JSON,
TSV, manifests, and logs were retained. The dsba5 immutable source snapshot was removed after runtime-hash,
exit-code, pullback, and parity verification; compact remote results were retained.
