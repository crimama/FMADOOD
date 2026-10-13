# Baseline1 DVT × Representation × Scorer Factorial 결과

Date: 2026-08-08  
Verdict: `SCORER_DEPENDENT_DVT_ATTRIBUTION / EFFECT_SIZE_MOSTLY_RETAINED`

## 1. Negative Evidence Intake

이 실험은 k를 튜닝하거나 새 scorer를 contribution으로 제안하지 않는다. 전체 8객체 A/B에서
K5는 이미 pAUROC와 collapse gate를 실패했으므로 기본 scorer는 K1로 유지한다. 여기서는
K1의 outlier 민감도가 앞서 관측한 DVT 기여도를 과장했는지만 확인했다. 고차원·소표본에서
ill-conditioned이고 in-sample calibration optimism이 있는 shrinkage Mahalanobis는 실행하지 않았다.

## 2. Motivation

앞선 2×2에서 DVT가 큰 주효과, Flow가 작은 양의 interaction을 보였다. 동일한
`DVT off/on × identity/Flow` 네 arm에서 K1과 K5를 한 번의 feature/Flow 평가로 함께 계산해,
DVT 귀인이 nearest outlier 하나에 의존하는지 분리한다.

## 3. Implementable Design

`fruit_jelly, walnuts, vial, fabric`, fixed SuperAD-16 support, DINOv3-H+ `[7,15,23,31]`,
resolution 672와 기존 3-epoch Flow를 고정했다. 각 arm은 `raw_1nn`과 `knn_mean_k5`만 산출했다.
Density, context, query update, metadata, morphology는 사용하지 않았다. 상세 gate는
[사전등록](preregistration.md)에 있다.

## 4. Evaluation Alignment

Aggregator가 네 arm의 DVT/Flow 상태, shots=16, density=0, scorer subset, support basename,
map/GT count, shape, finite score, TIFF 미보존을 검증했다. Reduced 4-object diagnostic이므로
benchmark 또는 새 method claim으로 확장하지 않는다.

## 5. Code Modification / Creation

- `scripts/run_baseline1_dvt_representation_scorer_factorial_arm.sh`: 단일 arm K1/K5 실행.
- `scripts/run_baseline1_dvt_representation_scorer_factorial.sh`: 네 arm 순차 실행.
- `scripts/aggregate_baseline1_dvt_representation_scorer_factorial.py`: factorial contrast와 gate.
- `tests/test_aggregate_baseline1_dvt_representation_scorer_factorial.py`: manifest, support,
  pass/fail regression.

## 6. Added Code Evaluation

- Focused test `7 passed`.
- Ruff, `py_compile`, 두 shell runner의 `bash -n`, `git diff --check` 통과.
- 로컬과 dsba5에서 중복 실행한 `fused_flow`의 K1/K5 세 지표 최대 절대 차이는
  `9.36e-7`로 parity를 통과했다.

## 7. Remote Execution

로컬 RTX 4090에서 DVT-on 두 arm과 `fused_flow` parity arm을 실행했고, dsba5 고정 컨테이너
`hun_fsad_tta_012`의 host GPU 3/container CUDA 0에서 DVT-off 두 arm을 실행했다. 최종 집계는
dsba5 `fused_raw`와 로컬 나머지 세 arm을 사용했다. 네 arm 모두 compact JSON만 보존했고
`anomaly_maps/`는 없다.

## 8. Results and Analysis

### 8.1 Macro 성능

| representation arm | K1 pAUROC / F1 / AU-PRO | K5 pAUROC / F1 / AU-PRO |
| --- | ---: | ---: |
| fused + raw | .808299 / .505523 / .562370 | .809052 / .505727 / .567529 |
| fused + Flow | .810399 / .507866 / .558749 | .812301 / .509481 / .564030 |
| DVT + raw | .839836 / .570710 / .569636 | .836831 / .568302 / .565468 |
| DVT + Flow | .848213 / .581588 / .570856 | .846504 / .585607 / .566313 |

### 8.2 핵심 contrasts

| contrast | K1 | K5 | K5 - K1 |
| --- | ---: | ---: | ---: |
| DVT main pAUROC | +.034675 | +.030991 | -.003684 |
| DVT main F1 | +.069455 | +.069351 | -.000104 |
| DVT main AU-PRO | +.009686 | +.000111 | -.009576 |
| DVT×Flow pAUROC | +.006278 | +.006424 | +.000146 |

K5에서도 DVT pAUROC 주효과는 `+.030991`로 K1 효과의 약 89%를 유지했고, F1 주효과는
사실상 동일하다. Flow interaction 역시 `+.006424`로 유지됐다. 따라서 “DVT 효과가 한 개의
1-NN outlier 때문에 생겼다”는 설명은 지지되지 않는다.

다만 엄격한 gate에서는 K1의 DVT-positive 객체가 `3/4`였던 반면 K5는 `2/4`였다.
`fruit_jelly`가 `+.006656`에서 `-.000676`으로 경계선을 넘었고, `walnuts`는
`-.005408 → -.007069`로 계속 음수였다. 이 한 항목 때문에 최종 verdict는
`SCORER_DEPENDENT_DVT_ATTRIBUTION`이다. 이는 효과 크기의 소실이 아니라 객체 간 일반성에 대한
엄격한 판정이 scorer에 민감하다는 뜻이다.

또한 K5는 DVT-off arm의 AU-PRO를 약 `+.0052` 높였지만 DVT-on arm에서는 약 `-.0042~-.0045`
낮춰, DVT의 AU-PRO 주효과를 거의 상쇄했다. DVT가 공간적으로 sharpen한 score를 여러 이웃 평균이
완만하게 만드는 현상과 정합적이지만, 이는 현재 결과에 대한 해석이지 별도 인과 검증은 아니다.

## 9. Conclusion and Cleanup

결론은 두 층으로 구분한다.

1. **강한 결론:** DVT의 큰 pAUROC/F1 기여와 작은 양의 Flow interaction은 K5에서도 대부분
   유지된다. Baseline1의 주된 설명은 계속 `DINOv3 representation → DVT position prior → small
   Flow interaction`이다.
2. **제한:** 객체별 부호와 AU-PRO 기여는 scorer에 민감하다. 따라서 DVT를 scorer-independent한
   보편 모듈로 주장할 수 없고, K5를 default로 재채택하지도 않는다.

후속 우선순위는 k 추가 sweep이나 Mahalanobis가 아니다. DVT가 small defect를 손상시키는 기존
증거와 이번 AU-PRO 민감성을 함께 고려해, **DVT의 positional prior는 유지하되 scale-selective하게
적용하는 구조**를 탐색하는 편이 직접적이다.

결과: `results/baseline1_dvt_representation_scorer_factorial_20260808_v1/summary.json`  
원격 parity: `results/remote_runs/dsba5/baseline1_dvt_repr_scorer_20260808_v1_pull/`
