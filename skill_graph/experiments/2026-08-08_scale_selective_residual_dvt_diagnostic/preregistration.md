# Scale-Selective Residual DVT Diagnostic 사전등록

Date: 2026-08-08  
Protocol: `scale-selective-residual-dvt-diagnostic-all8-20260808`  
Stage: bounded mechanism diagnostic; no method promotion

## 1. Negative evidence intake and motivation

직전 DVT signal-path attribution은 small-defect top-1% coverage 손실이 Flow나 deployed scoring이
아니라 `post_dvt_feature` 경계, 즉 DVT subtraction 직후에 시작됨을 보였다. 이어진 positional
causal controls는 current DVT의 medium-scale AP/top-5% coverage 이득이 정확한 position
correspondence에 의존하지만, small-defect no-harm gate는 실패한다고 판정했다.

따라서 이번 질문은 DVT alpha, k, threshold, score fusion, support admission, metadata/query
adaptation을 다시 여는 것이 아니다. 단 하나만 묻는다.

> support-repeatable positional artifact에서 coarse/low-frequency component만 제거하면 current
> DVT의 medium-scale 이득은 유지하면서 small-defect peak coverage 손실을 줄일 수 있는가?

Failure basin은 passive feature surgery가 hard null에 흡수되는 것, low-pass operator가 current
DVT의 약한 alpha retune으로만 작동하는 것, medium 이득 보존 없이 small signal만 raw 쪽으로
되돌리는 것이다.

## 2. Target and fixed common contract

- `target_dataset=MVTec AD2 single-image`
- local data root: `/home/hun/Volume/DATA/mvtec_ad_2`
- remote data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- objects: `can`, `fabric`, `fruit_jelly`, `rice`, `vial`, `wallplugs`, `walnuts`, `sheet_metal`
- population: full `test_public/good,bad`
- support: fixed SuperAD-selected 16 normal images/object from the existing Baseline1 manifest
- backbone/scorer: DINOv3-H+ layers `[7,15,23,31]`, 672 except `sheet_metal=448`, fused representation,
  Flow 3 epochs, density weight `0`, raw latent 1-NN, `raw_1nn` Phase-3 scorer only
- no metadata/filename/condition/query adaptation/test-private/oracle access before scoring
- GT use: only after all score maps are computed, for existing threshold-free defect-scale attribution
- SuperAD/RN-FMLK: `N/A`, `comparable=false`; this is an internal DVT mechanism diagnostic, not a
  deployable method or Table-1 comparison

Existing raw/current artifacts may be reused if custody checks pass:

- raw baseline: `results/dvt_alignment_defect_scale_attribution_20260808/local/fused_flow`
- current DVT: `results/dvt_alignment_defect_scale_attribution_20260808/local/dvt_flow`

The candidate and hard null must be newly run unless complete same-schema artifacts already exist under this
experiment root.

## 3. Frozen arms

Let support-fitted DVT artifact be \(a_p=\mu_p-g\), where \(\mu_p\) is the support position mean and \(g\) is
the support global mean in fused DINOv3-H+ feature space.

1. `raw`: no DVT.
2. `position_current`: deployed current DVT, subtract \(a_p\), alpha fixed to `1.0`.
3. `position_lowpass`: candidate. Subtract \(L(a)_p\), where \(L\) is a fixed separable Gaussian low-pass on
   the artifact grid with `sigma=2.0` patch cells and `truncate=3.0`. This is the preregistered
   raw-fine+DVT-coarse dual path: high-frequency/query-local residuals are not subtracted, while the same
   support positional template is retained at coarse scale. Alpha remains `1.0`.
4. `position_lowpass_shuffled`: equal-capacity/energy hard null. Apply the same low-pass operator, then
   deterministic roll by `(H//2, W//2)` before subtraction. This preserves the candidate artifact values and
   L2 norm while destroying spatial correspondence.

No other low-pass radius, alpha, layer subset, scorer, k, object policy, threshold, fusion weight, learned
calibration, metadata rule, or query-conditioned switch may be evaluated in this diagnostic.

## 4. Metrics and aggregation

Use the existing Phase-3 object JSON schema:

- standard custody metrics: `seg_AUROC_0.05`, `seg_F1`, `seg_AUPRO_0.05`
- primary scale rows: per bad image and defect scale `pixel_ap`, `full_auroc`, tie-inclusive
  `coverage_top1`, `coverage_top5`, `component_hit_top1`, `component_hit_top5`
- scale bins: small `<=1%`, medium `1--5%`, large `>5%`
- aggregation: paired image delta -> object mean -> equal-weight object macro

Primary diagnostic endpoints:

- small: `coverage_top1` and `component_hit_top1`
- medium: `pixel_ap` and `coverage_top5`

Unified output fields must include:

- `superad_seg_AUROC_0.05=N/A`, `superad_seg_F1=N/A`
- `method_<metric>` for candidate standard metrics
- `delta_vs_superad=N/A`
- `comparable=false`
- `candidate_minus_raw`, `current_minus_raw`, `candidate_minus_current`, and
  `candidate_minus_lowpass_shuffled` scale contrasts

## 5. Validity gates

All must pass before interpreting direction or magnitude:

- all 8 objects present for all four arms;
- support-16 basenames, backbone, feature layers, support transform, Flow config, scorer config, and
  test-public population match across arms except `dvt_denoise_mode`;
- raw/current reused artifacts match prior all-8 DVT effect custody and contain no dense `anomaly_maps/`;
- candidate and null write no dense anomaly maps;
- GT/component/native-shape custody matches across arms for every paired image-scale row;
- top-area selected pixels never exceed `2x` nominal area due to ties;
- low-pass candidate records `dvt_artifact_lowpass_sigma=2.0`, `truncate=3.0`;
- candidate and lowpass-shuffled artifact L2 totals match within relative tolerance `1e-6`;
- lowpass-shuffled roll shift is nonzero for every valid feature grid;
- candidate artifact is neither identity/no-op nor current clone:
  `0.15 <= dvt_artifact_l2_total / current_dvt_artifact_l2_total <= 0.95`;
- no metadata/query/test-private flags are set.

Any failure yields `INVALID_SCALE_SELECTIVE_DVT_DIAGNOSTIC`.

## 6. Strict and continuation gates

Strict gate `SCALE_SELECTIVE_RESCUE` requires:

- current DVT reproduces the prior medium gain: current-minus-raw medium `pixel_ap > 0.05` and
  `coverage_top5 > 0.05`;
- candidate preserves at least 75% of current medium gain for both `pixel_ap` and `coverage_top5`;
- candidate-minus-raw small `coverage_top1 >= 0.0`;
- candidate-minus-current small `coverage_top1 >= 0.0075`;
- candidate small `component_hit_top1` is not worse than raw by more than `0.005`;
- candidate beats `position_lowpass_shuffled` on medium `pixel_ap` and `coverage_top5` by at least `0.01`;
- candidate beats `position_lowpass_shuffled` on small `coverage_top1` by at least `0.005`;
- no eligible object has candidate-minus-raw medium `pixel_ap < -0.03`.

Continuation gate `BOUNDED_CONTINUE_SIGNAL` requires validity plus:

- candidate preserves at least 60% of current medium `pixel_ap` and `coverage_top5` gains;
- candidate improves small `coverage_top1` over current by at least `0.005`;
- candidate is not matched by the hard null on both primary scale families.

Hard stop conditions:

- hard null matches or beats candidate on both medium preservation and small recovery;
- candidate medium preservation is below 60% of current DVT gain;
- candidate does not improve small `coverage_top1` over current by at least `0.005`;
- validity fails.

## 7. Frozen verdicts

- `SCALE_SELECTIVE_RESCUE_DIAGNOSTIC`: validity and strict gate pass.
- `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`: validity passes, strict gate fails, continuation gate passes.
- `KILL_FOR_CLAIM / NO_CONTINUE`: validity passes but strict and continuation gates fail.
- `HARD_NULL_ABSORBED`: validity passes and the lowpass-shuffled hard null matches or beats the candidate
  under the hard-stop rule.
- `INVALID_SCALE_SELECTIVE_DVT_DIAGNOSTIC`: any validity gate fails.

No verdict is `KEEP` or `CONDITIONAL KEEP`. A strict positive remains diagnostic only because SuperAD and
RN-FMLK are not compared.

## 8. Execution and reporting

- Implement only fixed denoiser modes, compact runner shell, compact aggregator, focused tests, and report
  skeleton.
- Do not launch the all-8 GPU run until code/tests/commands are reported to the leader.
- Expected local result root:
  `results/scale_selective_residual_dvt_diagnostic_20260808/local/`
- Candidate output:
  `.../position_lowpass/`
- Hard-null output:
  `.../position_lowpass_shuffled/`
- Preserve JSON/TSV/log/summary only. Delete and verify absence of any `anomaly_maps/` directories after
  conclusion.
