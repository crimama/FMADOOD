# Baseline1 DVT All-8 Threshold-Free Ranking/Coverage 결과

Date: 2026-08-08
Protocol: `baseline1-dvt-all8-rank-coverage-20260808`
Verdict: `GENERAL_LOCALIZATION_PRIOR_SUPPORTED` (diagnostic only)

## 1. 질문과 결론

앞선 4-object threshold-conditioned 결과가 `fabric` 또는 oracle threshold에 의존한
현상인지 확인했다. 동일한 `fused+Flow`에서 DVT off/on만 바꾸고, 전체 8개 객체의 bad
image에 대해 threshold-free pixel ranking과 고정 top-area coverage를 측정했다.

사전등록 gate는 모두 통과했다. DVT는 bad-image pixel AP를 macro `+0.01064`, full AUROC를
`+0.00817`, top-5% defect-pixel coverage를 `+0.00635`, top-5% component hit를
`+0.02067` 높였다. AP와 AUROC는 각각 `7/8`, top-5% coverage는 `6/8` 객체에서 양수였고,
`fabric`을 제외한 macro도 AP `+0.00875`, top-5% coverage `+0.00554`였다.

따라서 DVT의 이득은 특정 threshold나 `fabric` 하나만의 현상으로 설명되지 않는다. 현재
증거상 DVT는 **object-aligned spatial localization prior**로서 결함 픽셀의 전반적 순위와
중간 폭의 coverage를 개선한다.

## 2. 사전등록 및 정보 경계

- objects: `can, fabric, fruit_jelly, rice, vial, wallplugs, walnuts, sheet_metal`
- full `TESTpublic`; fixed SuperAD-selected support-16
- DINOv3-H+ layers `[7,15,23,31]`, resolution 672; `sheet_metal`만 448
- common Flow, raw latent 1-NN, density weight `0`
- arms: `fused_flow` (DVT off), `dvt_flow` (`position_mean`, alpha 1.0)
- condition token, filename metadata, split metadata는 추론과 지표 계산에 사용하지 않았다.
- top 1%/5%는 score cutoff의 모든 동점을 포함하며 실제 선택 면적을 별도 기록했다.

## 3. 실행 및 유효성

- local: `hun_fsad_tta_local_stage1a0`, RTX 4090
- remote: dsba5 `hun_fsad_tta_012`, host GPU3/container CUDA0
- support, prediction/GT coverage, finite score, shape, arm-invariant pixel/component count를 검사했다.
- anomaly-map TIFF와 `anomaly_maps/`는 생성·보존하지 않았다.
- float16 ordered-code histogram으로 AP/AUROC를 계산해 동점 의미론을 보존했다.
- evaluator regression + aggregator tests: `20 passed`; Ruff: pass
- local/dsba5 핵심 metric 108개 worst absolute delta `5.1280e-6 < 1e-5`; verdict/gate 일치

## 4. Macro 결과

| Metric | DVT off | DVT on | Δ on-off |
| --- | ---: | ---: | ---: |
| bad-image pixel AP | 0.381641 | 0.392278 | +0.010637 |
| bad-image full AUROC | 0.918130 | 0.926297 | +0.008167 |
| defect coverage @ top 1% | 0.480063 | 0.474650 | -0.005413 |
| defect coverage @ top 5% | 0.680457 | 0.686806 | +0.006349 |
| component hit @ top 1% | 0.665869 | 0.660139 | -0.005730 |
| component hit @ top 5% | 0.819357 | 0.840030 | +0.020672 |
| seg pAUROC@0.05 | 0.796952 | 0.839483 | +0.042530 |
| seg F1 | 0.466814 | 0.529027 | +0.062212 |
| seg AU-PRO@0.05 | 0.506896 | 0.512333 | +0.005437 |

## 5. 객체별 핵심 변화

모든 값은 `DVT on - DVT off`다.

| Object | Δ AP | Δ full AUROC | Δ coverage@5% | Δ component@5% |
| --- | ---: | ---: | ---: | ---: |
| can | +0.00503 | +0.00380 | +0.01419 | -0.01111 |
| fabric | +0.02382 | +0.00806 | +0.01201 | +0.01222 |
| fruit_jelly | -0.01463 | -0.00073 | +0.01073 | +0.02306 |
| rice | +0.04645 | +0.02352 | +0.02702 | +0.07222 |
| sheet_metal | +0.00230 | +0.02407 | +0.02590 | -0.00694 |
| vial | +0.00958 | +0.00129 | +0.00234 | 0.00000 |
| wallplugs | +0.00911 | +0.00356 | -0.00463 | +0.04444 |
| walnuts | +0.00345 | +0.00175 | -0.03678 | +0.03148 |

AP의 유일한 음수 객체는 `fruit_jelly`, top-5% coverage의 큰 음수 객체는 `walnuts`다.
즉 효과는 일반적 경향이지만 객체별 모든 localization 지표를 동시에 개선하는 단조 효과는 아니다.

## 6. 직관적 해석

DVT는 같은 위치의 support-normal 평균을 빼므로, 객체 정렬이 유효할 때 반복되는 정상 구조를
약화시키고 위치상 이례적인 잔차를 강조한다. 이번 threshold-free 결과에서 AP/AUROC와 top-5%
coverage가 넓게 개선된 것은 이 잔차 강조가 oracle threshold 선택 이전의 **순위 자체**에 존재함을
뜻한다. 따라서 앞선 threshold-conditioned defect TPR 증가는 단순 threshold 산물이 아니다.

반면 top-1% coverage와 component hit는 macro 음수다. DVT가 anomaly score의 가장 높은 극소수
픽셀을 항상 defect core에 더 집중시키는 것은 아니다. 더 안전한 설명은 “초정밀 peak sharpening”이
아니라 “결함 영역 전반의 ranking 및 moderate-area coverage 개선”이다.

## 7. Gate 판정과 주장 한계

- AP macro positive, positive object count `7/8`: pass
- full AUROC macro positive, positive object count `7/8`: pass
- top-5% coverage macro positive: pass
- top-5% component macro `>= -0.01`: pass
- AP `< -0.03`, top-5% coverage `< -0.05` 객체 붕괴 없음: pass
- non-fabric AP/coverage macro positive: pass
- verdict: `GENERAL_LOCALIZATION_PRIOR_SUPPORTED`

이는 Baseline1 메커니즘에 대한 diagnostic verdict이지 새 방법의 benchmark KEEP가 아니다. DVT를
distribution-shift estimator/corrector라고 부를 근거도 아니며, Flow Matching·Diffusion·DiT의
선택 근거도 아니다. 특히 `fruit_jelly` AP와 `walnuts` coverage의 음수, top-1% macro 음수는
객체 및 spatial-scale 의존성이 남아 있음을 보여준다.

## 8. Baseline 논리의 갱신

현재 가장 매끄러운 설명은 다음과 같다.

`강한 DINOv3 representation → DVT object-position prior가 defect ranking/coverage 개선 → Flow의 작은 상호작용`

DVT의 기여는 이제 (1) factorial main effect, (2) condition에 거의 무관한 defect TPR 증가,
(3) all-8 threshold-free AP/AUROC 및 top-5% coverage 개선으로 세 단계에서 지지된다. 따라서
Baseline1을 이해하는 다음 질문은 “DVT가 shift를 어떻게 추정하는가”가 아니라 “어떤 공간 정렬과
defect scale에서 DVT prior가 유효하거나 실패하는가”다.

## 9. Artifacts

- preregistration: `skill_graph/experiments/2026-08-08_baseline1_dvt_all8_rank_coverage/preregistration.md`
- local summary: `results/baseline1_dvt_all8_rank_coverage_20260808/local/summary.json`
- remote summary: `results/remote_runs/dsba5/baseline1_dvt_all8_rank_coverage_20260808_dsba5/summary.json`
- parity: `results/baseline1_dvt_all8_rank_coverage_20260808/parity.json`
- object JSON: each summary root의 `{fused_flow,dvt_flow}/objects/*.json`

