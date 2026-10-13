# Baseline1 k=5 scorer default A/B 사전등록

날짜: 2026-08-07  
상태: `DESIGN_FROZEN`

## 질문과 범위

이 실험은 k를 방법론 기여로 최적화하지 않는다. 단일 최근접 이웃의 우연한
outlier 민감도를 줄이기 위한 운용 기본값으로 `k=5` 평균 거리가 적절한지만
검증한다. 비교 후 추가 k 탐색이나 객체별 k 선택은 금지한다.

## 고정 조건

- 데이터: MVTec AD2 `test_public`, 8개 객체 전체.
- support: 객체별 고정 SuperAD-16, SHA-256
  `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`.
- Backbone/DVT/Flow: Baseline1 명세 그대로. DINOv3-H+/16, `[7,15,23,31]`,
  `layer_norm_mean`, position-mean DVT alpha 1.0, 2-layer patch-local Flow.
- 학습: 3 epochs, LR `2e-4`, tail 0.3, logdet `0.02`, seed 0.
- 공통 score: 각 arm별 support-LOO 표준화 후 동일한 NLL density term 0.25를
  결합한다. 후처리, TTA, query update는 없다.
- 두 arm은 같은 학습된 Flow와 latent memory를 공유하며 거리 집계만 다르다.

## Arms

- `K1`: query와 가장 가까운 support latent 하나의 Euclidean 거리.
- `K5`: 가장 가까운 support latent 5개의 Euclidean 거리 평균.

## 정량 평가

주 지표는 객체별 및 8-object macro `seg_AUROC_0.05`, `seg_F1_raw`,
`seg_AUPRO_0.05`이다. 보조 지표는 pixel AP, `TPR@FPR=1e-4`, 정상 이미지
pooled FPR, good-pixel 99.9 percentile 및 GT component recall이다. 모든 map의
개수, shape, 비상수 여부, non-finite 개수를 기록한다.

## 판정 규칙

`K5`를 운용 기본값으로 채택하려면 모두 만족해야 한다.

1. macro `seg_AUROC_0.05` delta가 `>= -0.002`.
2. macro `seg_F1_raw`와 `seg_AUPRO_0.05` delta가 각각 `>= -0.005`.
3. 어떤 객체도 pAUROC가 `-0.02` 미만 또는 F1/AUPRO가 `-0.03` 미만으로
   하락하지 않는다.
4. 정상 tail 개선이 실제로 존재한다: macro 정상-image pooled FPR 또는
   good-pixel p99.9 중 하나가 감소한다.
5. 8개 객체와 모든 coverage 검사가 완결된다.

하나라도 실패하면 `k=1`을 유지한다. 성공하더라도 결론은 scorer의 안정적
운용 선택에 한정하며, 새로운 방법론 contribution으로 주장하지 않는다.

## 선행 증거의 취급

2026-07-12 shadow scorer-suite는 동일 경로에서 K5가 정상 tail을 줄였지만
pAUROC/F1/AP를 소폭 낮춘 것으로 관찰됐다. 당시 AU-PRO가 없었으므로 본
실험의 판정을 대체하지 않으며, 본 사전등록의 임계값은 재실행 전에 고정한다.
