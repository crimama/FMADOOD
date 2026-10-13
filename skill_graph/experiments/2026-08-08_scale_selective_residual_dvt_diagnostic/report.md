# Scale-Selective Residual DVT Diagnostic

Date: 2026-08-08  
Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`  
Scope: bounded mechanism diagnostic; SuperAD/RN-FMLK `N/A`; `comparable=false`

## Negative Evidence Intake

This diagnostic follows `LOSS_AT_DVT_SUBTRACTION` and
`POSITION_SPECIFIC_MEDIUM_GAIN_WITH_SMALL_DEFECT_COST`. It does not reopen DVT alpha/k tuning, thresholding,
score fusion, support admission, query adaptation, or metadata policies. The only candidate is the
preregistered fixed low-pass support positional artifact, with an equal-energy shuffled-position hard null.

## Implementable Design

Arms:

- `raw`: reuse `results/dvt_alignment_defect_scale_attribution_20260808/local/fused_flow`
- `position_current`: reuse `results/dvt_alignment_defect_scale_attribution_20260808/local/dvt_flow`
- `position_lowpass`: newly run fixed `sigma=2.0`, `truncate=3.0` low-pass DVT
- `position_lowpass_shuffled`: newly run same low-pass artifact after deterministic `(H//2, W//2)` roll

The GPU run is intentionally limited to the two new arms.

## Evaluation Alignment

- Dataset: `target_dataset=MVTec AD2 single-image`
- Objects: all eight public objects
- Population: full `test_public/good,bad`
- Support: fixed SuperAD-selected support-16
- Scorer: Phase-3 `raw_1nn` only
- Primary endpoints: small `coverage_top1`; medium `pixel_ap` and `coverage_top5`
- Unified fields: `superad_seg_AUROC_0.05=N/A`, `superad_seg_F1=N/A`, `delta_vs_superad=N/A`,
  `comparable=false`

## Code Modification / Creation

- `src/flow_tte/denoising.py`: fixed `position_lowpass` and `position_lowpass_shuffled` DVT modes.
- `scripts/flow_tte_mvtec_ad2_core.py`: low-pass artifact diagnostics in existing DVT audit payloads.
- `scripts/run_flow_tte_mvtec_ad2.py`: CLI accepts the two frozen DVT modes.
- `scripts/run_scale_selective_residual_dvt_diagnostic.sh`: runs only candidate/null and aggregates against
  reused raw/current roots.
- `scripts/aggregate_scale_selective_residual_dvt_diagnostic.py`: custody, energy/null, strict, continuation,
  and frozen verdict gates.
- `tests/test_flow_tte_dvt_causal_controls.py`: low-pass denoiser regression tests.
- `tests/test_aggregate_scale_selective_residual_dvt_diagnostic.py`: strict, continuation, null-absorbed,
  and invalid-energy fixtures.

## Added Code Evaluation

- `python3 -m py_compile src/flow_tte/denoising.py scripts/flow_tte_mvtec_ad2_core.py scripts/run_flow_tte_mvtec_ad2.py scripts/run_flow_tte_phase3_scorer_suite.py scripts/aggregate_scale_selective_residual_dvt_diagnostic.py tests/test_flow_tte_dvt_causal_controls.py tests/test_aggregate_scale_selective_residual_dvt_diagnostic.py` -> pass
- `.venv/bin/python -m pytest -q tests/test_flow_tte_dvt_causal_controls.py tests/test_aggregate_scale_selective_residual_dvt_diagnostic.py` -> `10 passed`
- `.venv/bin/python -m ruff check src/flow_tte/denoising.py scripts/aggregate_scale_selective_residual_dvt_diagnostic.py tests/test_flow_tte_dvt_causal_controls.py tests/test_aggregate_scale_selective_residual_dvt_diagnostic.py` -> pass
- `git diff --check -- <touched files>` -> pass
- raw/current reuse audit -> all eight raw/current object JSONs present and expose `dvt_artifact_l2_mean`

## Remote Execution

- Local GPU: `position_lowpass` and a same-environment `position_lowpass_shuffled` all-8 run.
- dsba5 GPU: an independent `position_lowpass_shuffled` all-8 replication in the fixed
  `hun_fsad_tta_012` container.
- Remote exit code: `0`.
- Relevant local/remote null metrics: 4,740 values compared; maximum absolute difference
  `0.0002437241`, with no difference above `0.001`.
- The immutable remote source snapshot was removed after pull; compact JSON/log artifacts were retained.

Local/new-arm command:

```bash
scripts/run_scale_selective_residual_dvt_diagnostic.sh \
  results/scale_selective_residual_dvt_diagnostic_20260808/local
```

Parallel arm commands:

```bash
ARMS=position_lowpass AGGREGATE=0 scripts/run_scale_selective_residual_dvt_diagnostic.sh \
  results/scale_selective_residual_dvt_diagnostic_20260808/local

ARMS=position_lowpass_shuffled AGGREGATE=0 scripts/run_scale_selective_residual_dvt_diagnostic.sh \
  results/scale_selective_residual_dvt_diagnostic_20260808/local

AGGREGATE=1 ARMS= scripts/run_scale_selective_residual_dvt_diagnostic.sh \
  results/scale_selective_residual_dvt_diagnostic_20260808/local
```

Remote/new-arm command shape:

```bash
DATA_ROOT=/home/hunim/Volume/DATA/mvtec_ad_2 \
PROJECT_ROOT=/workspace/fsad_tta \
scripts/run_scale_selective_residual_dvt_diagnostic.sh \
  /workspace/results_remote/scale_selective_residual_dvt_diagnostic_20260808
```

## Results and Analysis

All validity gates passed. The fixed Gaussian low-pass retained `85.32--92.41%` of the current DVT
patch-wise artifact energy across all eight objects, while the shuffled null retained exactly the same
low-pass energy and changed only spatial correspondence.

Macro defect-scale contrasts:

| Contrast | Small AP | Small top-1 coverage | Medium AP | Medium top-5 coverage |
| --- | ---: | ---: | ---: | ---: |
| current DVT - raw | +.000696 | -.011532 | +.075706 | +.073207 |
| low-pass DVT - raw | +.009410 | +.001132 | +.051089 | +.052500 |
| low-pass DVT - current DVT | +.008714 | +.012665 | -.024617 | -.020707 |
| low-pass DVT - shuffled low-pass | +.015578 | +.000696 | +.056823 | +.063081 |

The low-pass operator therefore removed the average small-defect top-1 penalty and improved small AP,
but it also discarded part of the medium-defect benefit. Relative to current DVT, it preserved `67.48%`
of medium AP gain and `71.71%` of medium top-5 coverage gain. These values pass the continuation floor
of `60%` but fail the strict `75%` preservation gate.

The equal-energy shuffled null did not absorb the result. Correct low-frequency position correspondence
beat the null strongly on medium AP and top-5 coverage (`+.056823/+.063081`). It did not meet the strict
small top-1 null margin: `+.000696 < +.005`. Thus, the experiment supports a real low-frequency positional
component for medium defects, but does not show that the small-defect rescue itself is position-specific.

Standard all-scale metrics for low-pass DVT versus raw were pAUROC `+.034767`, F1 `+.051028`, and AUPRO
`+.010513`. These remain diagnostic metrics, not a new SuperAD-comparable benchmark claim.

## Continuation Assessment

- Strict rescue gate: fail.
- Continuation gate: pass.
- Hard-null absorbed: false.
- Allowed next step: one preregistered two-band decomposition that preserves the validated low-frequency
  positional component while explicitly protecting the high-frequency/local residual. This is not an
  invitation to sweep sigma, alpha, gates, or fusion weights.
- Still closed: unconditional DVT tuning, arbitrary multi-scale search, scorer tuning, and promotion to a
  method contribution before an independent defect-preservation gate passes.

## Conclusion

`KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`. DVT's tradeoff is genuinely scale-selective: low-frequency
positional subtraction carries a substantial, position-dependent medium-defect benefit, while removing
high-frequency content repairs the mean small-defect top-1 loss. A pure low-pass replacement is still too
aggressive because it loses roughly one third of the medium AP gain. The next justified design is therefore
not “make DVT stronger,” but “separate low-frequency positional nuisance correction from a protected
high-frequency defect residual” under a single frozen diagnostic.

Primary artifact:
`results/scale_selective_residual_dvt_diagnostic_20260808/local/summary.json`
(`sha256=6a3efe9f6e414dbb5b446044b7cc196e2f6af24e024549ea4686594e5b266419`).
