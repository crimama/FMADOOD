# DVT Positional Causal Controls 결과 보고서

Date: 2026-08-08  
Protocol: `dvt-positional-causal-controls-all8-20260808`  
Verdict: `POSITION_SPECIFIC_MEDIUM_GAIN_WITH_SMALL_DEFECT_COST`  
Claim boundary: mechanism diagnostic only; no KEEP or benchmark claim

## 1. Negative Evidence Intake

직전 실험은 DVT 이득이 support positional alignment가 높고 결함 면적이 1--5%인 경우에
집중됨을 보였지만, 올바른 위치 대응이 원인인지 단순 centering/feature surgery가 원인인지는
분리하지 못했다. 따라서 alpha, gating, fusion, architecture를 튜닝하지 않고 current DVT를
두 hard null과 직접 비교했다.

## 2. Motivation

질문은 하나다. support에서 얻은 positional artifact를 **같은 위치**에서 빼는 것이 중요한가?
그렇다면 artifact 값과 L2 norm은 같지만 좌표만 틀린 `position_shuffled`, 그리고 위치 정보가
전혀 없는 `global_center`보다 current DVT가 명확히 좋아야 한다.

## 3. Implementable Design

- `raw`: 변환 없음.
- `position_current`: 현재 DVT, `position_mean - global_mean`을 같은 좌표에서 차감.
- `position_shuffled`: current artifact를 `(H//2, W//2)`만큼 deterministic roll한 뒤 차감.
- `global_center`: support global feature mean을 모든 위치에서 동일하게 차감.

모든 subtractive arm은 alpha `1.0`이며 같은 support-16, DINOv3-H+ `[7,15,23,31]`,
Flow 3 epochs, raw latent 1-NN을 사용했다. 기존 raw/current 결과를 재사용하고 두 control만
local RTX 4090과 dsba5 TITAN RTX에서 새로 실행했다.

## 4. Evaluation Alignment

전체 8개 MVTec AD2 TESTpublic 객체를 사용했다. primary endpoint는 bad image의 GT component를
small `<=1%`, medium `1--5%`, large `>5%`로 나눈 paired image -> object -> macro AP, full AUROC,
top-1/5% coverage, component hit다. metadata/filename/condition은 추론에 사용하지 않았고 GT는
score map 생성 후 평가에만 접근했다. SuperAD/RN-FMLK 직접 비교는 `N/A`, `comparable=false`다.

## 5. Code Modification / Creation

- `src/flow_tte/denoising.py`: `position_shuffled`, `global_center` 지원과 artifact audit.
- `scripts/run_flow_tte_mvtec_ad2.py`, `scripts/flow_tte_mvtec_ad2_core.py`: 새 mode와 diagnostics 연결.
- `scripts/run_flow_tte_phase3_scorer_suite.py`: 실제 phase3 JSON에 L2/roll/공간상수 audit 직렬화.
- `scripts/aggregate_dvt_positional_causal_controls.py`: 네 arm custody, scale contrast, gate, verdict 집계.
- causal-control 및 aggregator regression test 추가.

첫 local 집계는 phase3 scorer가 새 공간상수 audit 필드를 저장하지 않아
`INVALID_CAUSAL_ABLATION`이었다. 사전등록 gate를 완화하지 않고 serialization만 수정한 뒤 두
control을 재실행했다. 점수 함수나 학습 설정은 바꾸지 않았다.

## 6. Added Code Evaluation

- Fresh focused regression:
  `.venv/bin/python -m pytest -q tests/test_denoising.py tests/test_flow_tte_dvt_causal_controls.py tests/test_aggregate_dvt_positional_causal_controls.py tests/test_aggregate_dvt_alignment_defect_scale_attribution.py tests/test_flow_tte_phase3_scorer_suite.py`
  -> `50 passed`.
- `python3 -m py_compile` on the denoiser, runner entrypoint, runner core, aggregator, and causal-control tests:
  pass.
- `git diff --check` on modified causal-control code/docs: pass.
- Ruff on the directly touched runner/core target set still reports 16 pre-existing debt items
  (`PLR0913`, `C901`, `PLR0912`, `EM102`, `S101`, `UP007`); this was not broadened into a cleanup pass.
- current/shuffled artifact L2 relative audit: 전 객체 pass (`<=1e-6`).
- global-center spatial-constant audit: 전 객체 pass.
- raw/current 이전 summary 복제, support/config/GT custody, tie-area gate: 모두 pass.

## 7. Remote Execution

local과 dsba5에서 `position_shuffled`와 `global_center`를 서버별 병렬 실행했다. 7개 객체는
672px, `sheet_metal`은 448px로 실행했으며 네 chunk 모두 exit `0`이었다. DINOv3 gated 모델은
각 컨테이너의 offline cache를 사용했다. dsba5에서도 validity와 최종 verdict가 동일했다.
Aggregate parity는 `results/dvt_positional_causal_controls_20260808/parity.json`에 저장했다.
사전등록 기준으로 per-image 배열과 score-dependent integer/tie/count field를 제외한 aggregate
continuous value `1404`개를 비교했으며 worst absolute delta는 `6.6936e-6`였다. Verdict와
preregistered gate boolean mismatch는 `0`개였다.

## 8. SuperAD Baseline and Unified Metrics

이 실험은 새 방법의 benchmark가 아니라 Baseline1 내부 원인 규명이다. 따라서 공통 custody용
`seg_AUROC_0.05`, `seg_F1`, `seg_AUPRO_0.05`는 보존했지만 primary 판정은 threshold-free
scale metric으로만 수행했다. raw/current는 직전 attribution과 `1e-8` 이내로 복제됐다.

## 9. Results and Analysis

### 9.1 Medium defect positional specificity

| current 대비 control | AP | full AUROC | top-5 coverage | hit@5 |
| --- | ---: | ---: | ---: | ---: |
| `position_shuffled` | +0.090245 | +0.042567 | +0.087314 | +0.022917 |
| `global_center` | +0.075706 | +0.043168 | +0.073207 | +0.012500 |

Medium bin은 6개 객체에서 eligible이었다. current는 두 control 각각보다 AP가 `>=75%` 객체에서
높았고, 모든 preregistered specificity gate를 통과했다. 반대로 shuffled는 raw 대비 medium AP와
coverage가 함께 양수가 아니었고 global-center도 AP가 양수가 아니었다. 같은 artifact 값/norm을
틀린 좌표에서 빼거나 전역 평균만 빼는 것으로는 current 이득을 설명할 수 없다.

### 9.2 Small-defect safety

current-minus-raw small macro는 AP `+0.000696`, full AUROC `+0.002348`이지만 top-1 coverage
`-0.011532`, hit@1 `-0.006069`였다. macro coverage nonnegative gate와 객체별 AP/coverage collapse
guard가 실패했다. 즉 올바른 positional correction은 medium defect에는 인과적으로 유용하지만,
아주 작은 결함의 peak evidence 일부도 함께 지울 수 있다.

### 9.3 직관적 결론

현재 DVT의 효과는 “feature를 조금 움직여 Flow가 좋아졌다”가 아니다. **정상 support에서 반복되는
좌표별 구조를 올바른 위치에서 제거했기 때문에 medium defect가 더 잘 드러난 것**이다. 그러나 이
operator는 결함 크기를 알지 못하므로 small defect에는 안전하지 않다. 따라서 DVT는
distribution-shift corrector나 범용 denoiser가 아니라, scale-dependent object-position prior로
기술해야 한다.

## 10. Continuation Assessment

Verdict에 KEEP는 없다. specificity는 지지됐지만 small safety가 실패했으므로 unconditional DVT,
alpha/gating/fusion tuning은 종료한다. 이 결과는 DiT, Flow Matching, Diffusion 도입 근거도 아니다.
다음 연구가 필요하다면 DVT를 더 세게/약하게 튜닝하는 대신, small-defect evidence를 보존하면서
position-template만 제거할 수 있는 새로운 독립 requirement와 사전등록 gate가 먼저 필요하다.

## 11. Conclusion

`POSITION_SPECIFIC_MEDIUM_GAIN_WITH_SMALL_DEFECT_COST`. Baseline1의 DVT medium-scale 이득은
정확한 positional correspondence에 인과적으로 의존한다. generic centering과 arbitrary spatial
subtraction 가설은 기각된다. 동시에 small-defect no-harm은 기각되므로 배포 가능한 범용 개선으로
승격하지 않는다.

## 12. Post-Conclusion Storage Cleanup

- compact object JSON, leaderboard, logs, local/remote summary와 parity만 보존한다.
- `anomaly_maps/`는 생성하지 않았으며 최종 검사에서도 없어야 한다.
- dsba5 실행 결과는 `results/remote_runs/dsba5/dvt_positional_causal_controls_20260808_dsba5/`에 회수했다.
