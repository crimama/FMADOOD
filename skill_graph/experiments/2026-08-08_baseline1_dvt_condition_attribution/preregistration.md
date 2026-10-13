# Baseline1 DVT 조건별 기여 분해 사전등록

Date: 2026-08-08
Protocol: `baseline1-dvt-condition-attribution-20260808`
Scope: reduced-object mechanism diagnostic; post-hoc evaluation only

## 질문

앞선 DVT × Flow factorial에서 DVT는 Flow-on 기준 pAUROC `+.040953`, F1
`+.076065`, AU-PRO `+.012107`의 큰 향상을 보였다. 그러나 이 향상이 (A) 분포 변화가
있는 정상 이미지의 오탐을 억제한 결과인지, (B) 실제 결함의 국소 증거를 강화한 결과인지,
혹은 둘 모두인지는 아직 구분되지 않았다.

## 정보 사용 경계

- 파일명 suffix의 condition 정보는 feature 추출, DVT, Flow 학습, scoring, threshold
  선택에 사용하지 않는다.
- 모든 anomaly map 생성이 끝난 뒤 평가 record를 `regular`, `photometric`
  (`overexposed`, `underexposed`), `geometric` (`shift_*`)으로 나눌 때만 사용한다.
- condition별 threshold를 다시 최적화하지 않는다. 각 object/arm 전체 test-public에서
  계산한 하나의 pooled oracle threshold를 모든 condition에 그대로 적용한다.
- 따라서 본 실험은 metadata 기반 방법을 제안하거나 성능을 높이는 실험이 아니라,
  이미 생성된 예측의 오차 기여도를 사후 분해하는 진단이다.

## 고정 조건과 비교 arm

- MVTec AD2 single-image; `fruit_jelly, walnuts, vial, fabric` 전체 TESTpublic.
- fixed SuperAD-selected support-16, DINOv3-H+/16 `[7,15,23,31]`, resolution 672.
- normalized layer-mean fusion, 3-epoch Flow, raw latent 1-NN, density weight 0.
- DVT off: `fused_flow`; DVT on: `dvt_flow` (position-mean, alpha `1.0`).
- no query adaptation/memory update/metadata-conditioned inference/morphology.

## 조건별 고정-threshold 측정량

각 condition과 arm에 대해 다음을 기록한다.

- image 수, good/bad image 수와 good/defect pixel 수.
- good-only score `p99.9`와 global oracle threshold 이상인 good-pixel 비율.
- defect-pixel TPR at the same global oracle threshold.
- GT connected-component recall at the same global oracle threshold.

조건별 지표는 condition마다 표본 수가 다르므로 pixel-weighted 결과와 object-macro
방향성을 함께 보고한다. threshold가 서로 다른 DVT off/on arm의 절대 FPR을 공정하게
비교하기 위해, 추가적으로 각 arm의 score를 그 arm의 global oracle threshold로 나눈
threshold-relative tail(`good p99.9 / threshold`)도 보고한다. 이 값은 사후 재튜닝이
아니며 기존 decision boundary에 대한 상대적 정상 꼬리를 나타낸다.

## 판별 규칙

- `NORMAL_SHIFT_FP_SUPPRESSION`: DVT on이 shifted condition
  (`photometric`, `geometric`)의 good-pixel exceedance와 threshold-relative tail을
  낮추며, 그 감소가 regular보다 크고, shifted defect TPR/component recall을 손상시키지
  않는다.
- `DEFECT_EVIDENCE_GAIN`: DVT on이 condition 전반에서 defect-pixel TPR 또는 component
  recall을 일관되게 높이지만 shifted-normal tail/FPR 감소는 주효과가 아니다.
- `MIXED`: 위 두 패턴이 모두 관찰된다.
- `INCONCLUSIVE`: 객체별 방향이 불안정하거나 두 메커니즘 어느 쪽도 반복되지 않는다.

구체적인 효과 크기는 탐색적으로 보고하되, 한 condition 또는 한 객체만의 결과로
메커니즘을 확정하지 않는다. 최소 `3/4` 객체의 동일 방향을 재현성 기준으로 사용한다.

## 주장 한계

본 결과는 DVT의 관찰된 성능 향상을 설명하기 위한 reduced-object 귀인 진단이다.
인과적 shift correction, 전체 MVTec AD2 일반화, 방법론 KEEP/KILL 또는 metadata 활용의
정당성을 단독으로 주장하지 않는다.
