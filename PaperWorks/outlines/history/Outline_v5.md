# Support-Calibrated Residual Fusion: A Simple Score-Level Method for Few-Shot Industrial Anomaly Detection

**Hun Im\*, Author Two, Author Three, Pilsung Kang\***

---

## Index

- Abstract
- 1. Introduction

  - 1.1. Background: Few-Shot Normal-Reference Matching
  - 1.2. Main Problem: Layer-Wise Score Reliability Mismatch
  - 1.3. Direct Response: A Simple Support-Calibrated Fusion Rule
  - 1.4. Contributions
- 2. Pilot Evidence: Why Support-Calibrated Fusion Is Needed

  - 2.1. Layer Complementarity as Background Motivation
  - 2.2. Support-Limited Scale and Normal-Tail Mismatch
  - 2.3. Controlled Evidence for Calibration and Gating
- 3. Method: Support-Calibrated Residual Fusion

  - 3.1. Problem Formulation
  - 3.2. Framework Overview: From Input Image to Score Map
  - 3.3. Asymmetric Residual Spike Gating (RSG3)
  - 3.4. Tail-Calibrated Quantile Normalization (Tailscale)
  - 3.5. Score Projection and Component Quantile Postprocessing
  - 3.6. Anomaly Scoring and Computational Efficiency
- 4. Evidence and Analysis

  - 4.1. Experimental Setup and Reading Rules
  - 4.2. Main Few-Shot Comparison on MVTec AD 2
  - 4.3. Controlled Factorial: What the Fusion Rule Changes
  - 4.4. Few-Shot Scaling on MVTec AD and VisA
  - 4.5. Clean-Support to Corrupted-Test Distribution Shift
  - 4.6. Source-Fixed Operating-Point and Mechanistic Audit
  - 4.7. Evidence Boundary
- 5. Discussion & Conclusion

  - 5.1. Capacity vs. Support-Calibrated Fusion
  - 5.2. Boundary of Claims and Future Work

---

## Abstract

Few-shot 산업 이상 탐지에서는 소수의 정상 reference만으로 검사 기준을 만들어야 한다. 이 reference가 정상 외형의 다양성을 충분히 포함하지 못하면, 정상 query와 reference의 차이가 결함 점수처럼 나타날 수 있다. Foundation backbone의 layer·resolution·bank별 distance는 정상 support에 대해서도 서로 다른 scale과 upper-tail behavior를 보이므로, 보정되지 않은 multi-layer fusion은 일부 component의 큰 정상 반응을 anomaly evidence로 누적할 수 있다. 따라서 핵심 문제는 더 많은 feature를 사용하는 것이 아니라, 제한된 정상 support만으로 각 component score의 상대적 신뢰도를 정하는 것이다. 본 연구는 component별 leave-source-out q95로 score scale을 맞추고, deep score를 base로 유지하면서 support q99를 넘은 shallow residual만 조건부로 추가하는 Support-Calibrated Residual Fusion을 제안한다. MVTec AD 2의 16-shot seed0, full8 development 조건에서 동일한 backbone, support, resolution, bank, CQ 및 evaluator를 유지한 controlled factorial 비교를 수행했다. RSG3는 AUPRO@0.05 59.14576으로 다섯 fusion 구성 중 가장 높았으며, q95 조건의 ungated shallow 대비 residual gating이 +5.13007pp, gated 조건에서 tail scaling이 +1.66385pp의 차이를 보였다. 반면 deep-only 대비 shallow residual의 추가 이득은 +0.47482pp였고 pAUROC는 소폭 감소했다. 이 결과는 inductive-bias correction이나 representation restoration보다, 정상 support를 기준으로 불안정한 shallow contribution을 제한하는 score-level 설계를 지지한다. Acquisition shift에서의 fixed operating behavior는 별도의 paired audit에서 평가한다.

---

## 1. Introduction

### 1.1. Background: Few-Shot Normal-Reference Matching

- **Few-shot deployment**: 신규 품목과 공정 전환이 잦은 산업 현장에서는 대규모 정상 데이터를 다시 수집하고 모델을 학습하기 어렵다. 따라서 소수의 정상 reference로 검사 기준을 구성하는 설정이 필요하다.
- **Normal-reference matching**: Frozen foundation backbone을 사용하는 최근접 이웃 기반 방법은 정상 support의 patch feature를 bank에 저장하고, query patch와의 거리를 anomaly score로 사용한다. 이 방식은 추가 학습 부담이 작지만, 큰 거리가 결함 때문인지 support가 포괄하지 못한 정상 변화 때문인지는 거리 자체만으로 구분하기 어렵다.
- **Limited coverage**: 정상 외관은 조명, 반사, 센서, 자세와 생산 편차에 따라 변한다. 소수 support는 이 변동 범위의 일부만 관찰하므로, 정상 query도 support bank와 큰 거리를 가질 수 있다. Acquisition shift는 이러한 query-reference 차이를 더 크게 만들 수 있지만, few-shot과 shift의 상호작용은 별도의 실험 문제로 다룬다.
- **Multi-layer evidence**: Foundation backbone의 서로 다른 layer와 resolution은 동일한 query에 대해 서로 다른 distance map을 만든다. 이 반응 차이는 여러 정보를 제공할 가능성이 있지만, 각 score를 같은 크기와 신뢰도로 취급할 수 있다는 뜻은 아니다.

Inductive bias mismatch는 이 연구의 직접적인 해결 대상이 아니라 계층별 반응 차이를 설명하는 배경 동기다. 사전학습된 backbone은 객체 수준의 안정성과 국소 변화 민감도를 layer마다 다르게 나타낼 수 있다. 본 연구는 backbone의 representation, attention 또는 inductive bias를 변경하지 않고, 그 결과로 관찰되는 layer-wise score의 신뢰도 차이를 다룬다.

---

### 1.2. Main Problem: Layer-Wise Score Reliability Mismatch

Few-shot normal-reference matching에서 핵심 문제는 여러 layer를 사용할지 여부가 아니라, 각 layer가 만든 거리를 어떤 기준으로 비교하고 결합할 것인가이다.

- **Scale mismatch**: Layer, resolution, bank가 달라지면 정상 support에서 관찰되는 distance scale도 달라진다. 따라서 동일한 수치의 score가 component마다 동일한 비정상성을 뜻하지 않는다.
- **Normal-tail mismatch**: Shallow response는 국소 변화에 민감한 동시에 support에 포함되지 않은 정상 texture, 반사, 조명 변화에도 크게 반응할 수 있다. 이 upper tail을 그대로 합산하면 소수의 큰 정상 반응이 low-FPR 구간을 지배할 수 있다.
- **Reliability asymmetry**: Deep score는 상대적으로 안정적인 base가 될 수 있고 shallow score는 보완적인 국소 evidence를 제공할 수 있지만, shallow 전체를 항상 신뢰할 근거는 없다. 각 contribution의 허용 범위는 정상 support에서 관찰된 변동을 기준으로 정해야 한다.
- **Naive fusion failure mode**: 보정되지 않은 score를 평균하거나 더하면 component 수가 늘어날수록 evidence가 풍부해지는 것이 아니라, 불안정한 component의 정상 tail이 최종 anomaly map에 누적될 수 있다.

따라서 본 연구의 질문은 다음과 같다.

> 제한된 정상 support만 사용하여 component별 score scale을 맞추고, shallow response 중 정상 범위를 벗어난 부분만 anomaly evidence로 허용할 수 있는가?

이 문제는 backbone의 inductive bias를 수정하거나 결함 표현을 복원하는 문제가 아니다. Frozen representation 위에서 생성되는 score의 상대적 신뢰도를 support statistics로 정하는 score-construction 문제다.

---

### 1.3. Direct Response: A Simple Support-Calibrated Fusion Rule

문제의 각 요소는 간단한 score-level rule에 직접 대응한다.

| Observed problem | Direct response |
| :-- | :-- |
| Component별 distance scale 차이 | 정상 support의 leave-source-out q95로 component score scaling |
| Shallow normal-tail의 과도한 기여 | 정상 support q99를 넘은 shallow residual만 통과 |
| Deep/shallow reliability asymmetry | Deep score는 base로 유지하고 shallow residual만 조건부 추가 |
| Few-shot에서 별도 학습 부담 | Frozen backbone과 parameter-free score construction 사용 |

Tailscale은 layer·resolution·bank별 distance를 해당 component의 support-derived q95로 나누어 비교 가능한 상대 scale로 변환한다. RSG3는 normalized deep score를 base로 유지하고, normalized shallow score가 support q99를 초과한 양만 더한다. 이후 bank와 resolution score를 고정 규칙으로 집계한다.

이 방법은 backbone feature, attention 또는 representation geometry를 변경하지 않는다. 따라서 “inductive bias를 해결한다”, “결함 신호를 복원한다”, “normal nuisance를 제거한다”는 주장을 하지 않는다. 직접 검증할 주장은 support calibration과 residual gating이 동일한 frozen pipeline에서 naive shallow fusion보다 나은 anomaly map을 만드는지 여부다.

---

### 1.4. Contributions

- **C1. Few-shot layer-wise score reliability 문제를 정식화함**:
  - 제한된 정상 support에서 component별 distance scale과 normal tail이 달라 naive multi-layer fusion이 불안정해지는 문제를 다룸.
- **C2. 단순한 score-level 방법을 제안함**:
  - layer·resolution·bank별 score를 support 통계로 보정하고, deep base에 shallow residual을 조건부로 추가하는 Support-Calibrated Residual Fusion을 제안함.
- **C3. 동일 pipeline의 controlled factorial 근거를 제시함**:
  - RSG3, ungated fusion, MAD scaling, q95 scaling, deep-only를 동일 조건에서 비교해 gate와 scaling의 효과를 분리함.
- **C4. operating behavior와 구성 요소를 분석함**:
  - source-fixed threshold, regular→shift_1 paired response, raw/CQ score와 layer contribution을 이용해 성능 차이가 언제 나타나고 어디서 제한되는지 분석함.

---

## 2. Pilot Evidence: Why Support-Calibrated Fusion Is Needed

이 장은 Introduction에서 제기한 문제를 다시 설명하는 장이 아니라, 방법 설계에 직접 사용된 pilot evidence를 간결하게 제시한다. 각 결과는 일반 이론이나 인과적 증명으로 해석하지 않고, layer choice와 fusion rule을 정하는 개발 근거로 사용한다.

---

### 2.1. Layer Complementarity as Background Motivation

Shallow와 deep layer가 서로 다른 반응을 보인다는 사실은 multi-layer score를 고려하는 출발점이지만, 그 자체가 본 연구의 해결 문제는 아니다.

- Shallow layer는 texture, edge와 작은 appearance change에 상대적으로 민감할 수 있다. 이 민감도는 결함 단서와 정상적인 조명·반사 변화에 모두 반응할 수 있다.
- Deep layer는 더 안정적인 base score를 제공할 수 있지만 모든 국소 결함에서 우월하다는 보장은 없다.
- 이러한 차이를 inductive bias mismatch의 직접 증명으로 해석하지 않는다. 현재 방법은 layer representation이나 attention을 바꾸지 않기 때문이다.
- 따라서 pilot evidence가 답해야 할 질문은 “어떤 layer가 본질적으로 옳은가”가 아니라 “정상 support에서 각 layer score가 어느 범위까지 안정적인가”이다.

---

### 2.2. Support-Limited Scale and Normal-Tail Mismatch

각 component $(r,b,l)$에 대해 정상 support bank $B_{r,b,l}$와 query patch $p$ 사이의 거리를 다음과 같이 둔다.

$$
d_{r,b,l}(p)=\min_{u\in B_{r,b,l}}D\bigl(f_{r,l}^{q}(p),u\bigr).
$$

이 raw distance의 크기와 upper tail은 layer, resolution, bank에 따라 달라질 수 있다. 따라서 $d_{r,b,l_1}(p)>d_{r,b,l_2}(p)$라는 사실만으로 첫 번째 component가 더 강한 anomaly evidence라고 판단할 수 없다.

Few-shot setting에서는 이 문제가 더 중요하다. Support 수가 적으면 정상 변동의 tail을 조밀하게 관찰하기 어렵고, support에 포함되지 않은 정상 pattern이 일부 component에서 큰 distance로 나타날 수 있다. 여러 raw score를 그대로 합치면 이러한 component가 최종 score를 과도하게 지배한다.

이에 따라 본 연구는 feature representation에 대한 원인 가설보다 support에서 직접 관찰할 수 있는 두 통계량에 집중한다.

- component별 leave-source-out normal distance의 q95: score scale의 기준
- normalized shallow response의 q99: shallow contribution을 허용하는 기준

이 통계량은 target distribution이나 defect label로 조정하지 않는다. 또한 target FPR이나 defect preservation을 보장하지 않으므로, source-fixed operating analysis에서 별도로 검증한다.

---

### 2.3. Controlled Evidence for Calibration and Gating

설계 근거는 표현 수준의 인과 주장보다 동일 pipeline의 controlled contrasts에서 얻는다.

- q95 scaling을 고정했을 때, gated residual인 FF_D는 ungated shallow인 FF_B보다 AUPRO가 `+5.13007pp` 높았다.
- residual gating을 고정했을 때, q95 scaling인 FF_D는 MAD scaling인 FF_C보다 `+1.66385pp` 높았고 8/8 classes에서 개선되었다.
- FF_D는 deep-only FF_E보다 AUPRO가 `+0.47482pp` 높았지만 pAUROC는 `−0.24459pp`였다.

따라서 현재 evidence는 세 가지를 구분한다. Ungated shallow contribution을 제한하는 효과는 크고, tail-aware scaling은 gated fusion을 추가로 안정화한다. 반면 shallow evidence 자체의 순증가는 작고 metric-dependent하다. 이 결과는 inductive-bias correction이나 defect-signal restoration보다 support-calibrated score reliability라는 해석에 직접 부합한다.

---

### 2.4. Scope and Limits of the Pilot Evidence

- 본 pilot은 layer별 score distribution과 fusion behavior를 관찰한 개발 근거이며, ViT attention의 인과 구조나 representation-level defect preservation을 증명하지 않는다.
- Frozen backbone을 사용하는 것은 현재 방법의 범위를 명확히 하기 위한 설계 선택이다. 일부 평가된 trainable alternatives가 특정 16-shot 조건에서 열세였다는 사실을 모든 adapter 또는 parameter learning의 일반적 실패로 확대하지 않는다.
- q95와 q99은 정상 support에서 얻은 상대적 기준이다. 이 값은 target shift의 FPR, 결함 TPR, calibration transfer를 자동으로 보장하지 않는다.
- FF_D의 deep-only 대비 AUPRO 이득은 작고 pAUROC는 낮다. 따라서 shallow residual이 항상 유용하거나 모든 low-FPR 영역에서 우월하다고 주장하지 않는다.
- 현재 근거가 직접 지지하는 범위는 동일 frozen pipeline에서 `support-tail scaling + residual gating`이 tested development condition의 ungated shallow fusion보다 안정적이었다는 것이다.

---

## 3. Proposed Method: Support-Calibrated Residual Fusion

---

### 3.1. Problem Formulation

- **입력**: $N$-shot 골든 정상 서포트 집합 $\mathcal{S}_N = \{x_1^s, \dots, x_N^s\} \subset \mathcal{M}_{normal}$.
- **현실 제약 (Coverage Deficiency)**: 서포트는 물리적 정상 다양체 $\mathcal{M}_{normal}$의 극히 국소적인 영역만 포괄하며, 테스트 쿼리 $x^q$는 서포트가 관측하지 못한 커버리지 공백(Coverage Void)에 위치할 수 있음.
- **학습 제약**: 백본 동결(`requires_grad=False`), 추가 학습 파라미터 0개($\text{Params}=0$).
- **출력**: 쿼리 $x^q$에 대한 연속 이상 맵 $A(x^q) \in \mathbb{R}^{H \times W}$.
- **Support fitting**: 정상 support images만 사용해 feature bank, leave-source-out q95 scales, shallow q99 gate references를 계산한다. Query images, defect labels, and evaluation masks are not used in fitting or score construction.
- **Prediction contract**: each query is passed through the frozen backbone, compared with the fixed support banks, normalized component-wise, gated, projected to native space, and then evaluated by the common streaming evaluator.

---

### 3.2. Framework Overview

```text
[입력 Query 이미지 (동일 평가 전처리)]
                         │
        [Frozen Foundation Backbone (본 연구: DINOv2-L)]
                         │
       ┌─────────────────┴─────────────────┐
       ▼                                   ▼
Layer별 patch feature               Layer별 patch feature
(L5 shallow, L11/L23 deep)          (네 개의 고정 해상도)
       │                                   │
       ▼                                   ▼
[Tailscale 정규화]                  [Tailscale 정규화]
(component별 LOO q95 나눗셈)         (component별 LOO q95 나눗셈)
       │                                   │
       └─────────────────┬─────────────────┘
                         ▼
        [Deep 기본 점수와 Shallow 초과 반응의 선택적 결합 (RSG3)]
        (q95-scaled deep base + q99 초과 shallow residual)
                         │
                         ▼
        [Native-space projection 및 고정 CQ 후처리]
                         │
                         ▼
               최종 native-resolution anomaly map
```

#### 입력 이미지에서 최종 anomaly map과 image-level anomaly score까지의 처리 순서

여기서 RSG3는 **비대칭 residual spike gating**을 뜻한다. 이 용어는 하나의 연산이 수행하는 네 가지 특성을 설명한다.

- **비대칭**: deep score와 shallow score를 같은 방식으로 평균하지 않는다. Deep score는 항상 기본 신호로 유지하고, shallow score는 조건을 만족할 때만 추가한다.
- **Residual**: shallow score 전체를 사용하지 않고 정상 기준을 초과한 부분만 사용한다.
- **Spike**: 정상 support에서 관찰되는 일반적인 shallow 반응보다 큰 상위-tail 반응을 뜻한다.
- **Gating**: shallow 반응이 정상 기준 안에 있으면 0으로 만들고, 기준을 초과한 부분만 통과시키는 선택 연산이다.

한 위치 $p$에서의 개념적 계산은 다음과 같다.

$$
r(p)=\max\bigl(0,z_5(p)-\tau_{5,q99}\bigr),
$$

$$
S_{RSG3}(p)=\operatorname{mean}\bigl(z_{11}(p),z_{23}(p)\bigr)+r(p).
$$

예를 들어 shallow score가 정상 q99보다 낮으면 $r(p)=0$이므로 deep base만 남는다. Shallow score가 정상 q99를 초과하면 초과한 크기만 deep base에 더해진다. 따라서 RSG3는 “shallow feature를 추가하는 방법”이라기보다 “shallow feature 중 정상 범위를 벗어난 반응만 보완 신호로 허용하는 방법”으로 이해할 수 있다.

이 단계에서 생성되는 것은 먼저 native-resolution **anomaly map**이다. Evaluator는 이 map과 GT를 이용해 segmentation metrics를 계산하고, image-level diagnostic score는 최종 map의 maximum으로 기록한다. 따라서 anomaly map과 image-level anomaly score는 같은 query prediction에서 파생되지만 서로 다른 출력이다.

1. **입력과 전처리**: query 이미지를 support 이미지와 동일한 방식으로 전처리하고 원본 이미지 크기를 기록한다. method에 전달되는 query에는 label이나 mask를 함께 제공하지 않는다.
2. **다중 해상도 feature 추출**: frozen DINOv2-L backbone이 이미지를 448, 672, 896, 1120 해상도에서 처리한다. 각 해상도에서 `[5, 11, 23]` layer feature를 추출하고, 각 feature를 해당 spatial grid를 가진 L2-normalized token matrix로 변환한다.
3. **정상 reference와 비교**: 각 query token을 해당 resolution과 bank의 고정 support bank와 비교해 1-nearest-neighbor distance를 계산한다.
4. **component 보정**: 각 layer·resolution·bank의 distance map을 support에서 계산한 scale로 나눈다. 이 과정을 통해 layer별 raw distance가 아니라 support 기준의 상대적 크기를 layer fusion에 전달한다.
5. **layer 결합**: layer 11과 23의 score로 deep base를 만든다. layer 5의 response를 support q99와 비교하고, q99를 초과한 양수 부분만 deep base에 더한다. 그 결과 각 bank와 resolution마다 하나의 score map이 생성된다.
6. **bank와 resolution 결합**: 각 score map을 token grid에서 원본 이미지 크기의 native space로 bilinear projection한다. 한 resolution 안에서는 네 bank의 출력을 평균하고, 다시 네 resolution의 map을 평균해 후처리 전 anomaly map을 만든다.
7. **고정 후처리와 출력**: label 없이 공통 component-quantile 후처리를 적용한다. 최종 native-resolution map은 streaming evaluator로 전달하고, map의 maximum은 image-level diagnostic score로 기록한다.

이 순서에서 각 구성 요소의 역할은 분리되어 있다. backbone은 서로 다른 layer의 feature evidence를 제공하고, support bank는 정상 reference를 제공한다. Tailscale은 component별 score scale을 보정하고, RSG3는 shallow contribution을 제한한다. 마지막으로 aggregation과 projection이 token score를 dense image-space output으로 변환한다. Query 평가 중에는 backbone, support bank, scale, gate threshold를 갱신하지 않는다.

---

### 3.3. Asymmetric Residual Spike Gating (RSG3)

- **목적**: shallow layer는 작은 결함의 경계와 texture 변화를 포착할 수 있지만 정상적인 조명·반사·센서 변화에도 민감하다. 따라서 shallow score 전체를 deep score에 더하면 정상 query의 높은 반응까지 anomaly evidence로 누적될 수 있다. RSG3는 shallow를 항상 켜 두지 않고, 정상 support에서 보기 드문 shallow 반응만 보완 신호로 사용한다.
- **입력**: 각 native-space 위치 $p$에서 q95로 scaling된 shallow score $z_5(p)$와 deep base score $z_d(p)$를 사용한다. 현재 구성에서 deep base는 layer 11과 23의 score를 평균한 값으로 정의한다.
- **support 기준**: support normal feature bank에서 계산한 shallow normalized score의 q99를 $\tau_{5,q99}$로 둔다. 이 threshold는 query label이나 defect mask로 조정하지 않는다.
- **개념식**:

  $$
  z_l(p)=\frac{d_l(p)}{q_{95,l}^{\mathrm{normal}}}, \qquad
  z_d(p)=\operatorname{mean}\bigl(z_{11}(p),z_{23}(p)\bigr),
  $$

  $$
  S_{RSG3}(p)=z_d(p)+\operatorname{ReLU}\bigl(z_5(p)-\tau_{5,q99}\bigr).
  $$
- **직관**: shallow 반응이 support 정상 범위 안에 있으면 residual은 0이므로 deep base만 남는다. shallow 반응이 정상 상위 tail을 넘으면 초과한 양만 추가된다. 따라서 shallow의 전체 크기가 아니라 정상 기준을 넘어선 부분이 최종 score에 영향을 준다.
- **실제 aggregation**: 이 계산은 하나의 layer·resolution·bank component를 설명하는 축약식이다. 실제 pipeline에서는 각 component를 계산하고 native space로 투영한 뒤 bank와 resolution에 대해 고정된 평균 규칙을 적용한다.
- **해석의 한계**: RSG3는 defect-only gate나 attention-level 정보 복원을 보장하지 않는다. gate가 normal response를 줄일 수도 있고 defect response를 줄일 수도 있으므로, response decomposition과 fixed-threshold metrics를 함께 검증한다.

---

### 3.4. Tail-Calibrated Quantile Normalization (Tailscale)

- **문제**: 서로 다른 layer와 resolution의 nearest-neighbor distance는 같은 단위를 갖더라도 정상 분포의 spread와 tail 크기가 다를 수 있다. 이 상태에서 단순 평균하면 score가 큰 component가 결함 관련성보다 scale 차이 때문에 fusion을 지배할 수 있다.
- **support-only 계산**: component $m$에 대해 support image 자체를 자기 이웃으로 사용하지 않도록 leave-source-out 방식으로 정상 distance를 모은다. 그 분포의 q95를 $q_{95}^{(m)}$로 저장한다.
- **정규화**:

  $$
  \widetilde{d}_m(p)=\frac{d_m(p)}{q_{95}^{(m)}+\epsilon}.
  $$

  $\epsilon$은 degenerate scale을 방지하기 위한 수치 안정화 항이며, 실제 설정값은 configuration에 기록한다.
- **직관**: 각 component에 대해 “이 layer에서 정상 support의 상위 반응과 비교해 query가 얼마나 큰가”를 계산한다. 따라서 raw distance의 절대 크기 대신 support-relative magnitude를 fusion에 전달한다.
- **q95와 operating threshold의 구분**: q95는 component score의 scale을 맞추는 reference이다. source FPR 1% threshold나 shifted target FPR을 직접 보장하지 않는다. source-fixed operating threshold는 별도의 calibration pool과 evaluator 단계에서 계산한다.

### 3.4.1. Fixed Feature, Resolution, and Bank Aggregation

- **Feature 추출**: 현재 RSG3는 frozen DINOv2-L의 `[5, 11, 23]` layer를 `[448, 672, 896, 1120]` 네 가지 해상도에서 사용한다.
- **Support bank**: support feature를 네 개의 고정 bank로 나누어 저장한다. 각 bank는 query 예측 전에 구성되며 비교 대상 arm 전체가 동일한 bank를 공유한다.
- **Distance 계산**: 각 query feature 위치에서 해당 support bank까지의 1-nearest-neighbor distance를 계산한다. query image나 query label은 bank에 삽입하지 않는다.
- **Projection**: 각 resolution의 token score map을 bilinear 방식으로 native image space에 투영한다. 한 resolution 안에서는 네 bag의 score를 평균하고, 그 결과를 네 resolution에 대해 다시 평균한다. 최종 결과는 native-resolution anomaly map이다.
- **연산 순서**: feature 추출 → nearest-neighbor distance 계산 → component q95 scaling → deep/shallow score 구성 → q99 residual gate → 고정된 component aggregation → CQ/evaluator 후처리 순서로 처리한다.
- **비교 통제**: FF_A–FF_E는 동일한 support manifest, bank 구성, backbone 호출, resolution 집합, query 순서, evaluator를 사용한다. 따라서 비교 차이는 data pipeline이 아니라 normalization과 shallow gating에서 발생하도록 설계했다.

---

### 3.5. Score Projection and Component Quantile Postprocessing (CQ)

- **역할**: score map을 evaluator가 요구하는 native spatial contract에 맞춘 뒤, 현재 설정의 `component_quantile(percentile=99, q=1.0, dilate_size=0)` 후처리를 적용한다. 먼저 query score map의 99번째 백분위로 high-score mask를 만들고, 8-connectivity component를 찾는다. 이후 각 component의 pixel 값을 해당 component의 q-quantile로 치환한다. 현재 `q=1.0`이므로 각 component의 maximum 값으로 치환된다. 이 단계에서 GT나 query label은 사용하지 않는다.
- **왜 별도 기여로 해석하지 않는가**: 현재 RSG3 factorial에서는 CQ가 모든 arm에 공통으로 고정되어 있다. 따라서 FF_D와 다른 arm의 차이를 CQ의 독립 효과로 해석할 수 없다.
- **region recovery와의 구분**: 별도로 평가한 recovery 후보는 RSG3 대비 AUPRO를 개선하지 않았다. 따라서 연결 성분 diffusion이나 고립 오탐 제거를 현재 방법의 확인된 독립 기여로 서술하지 않는다.
- **재현 조건**: CQ의 독립 효과를 주장하려면 동일한 support, backbone, resolution, bank, seed, query 범위에서 CQ-off/on을 별도로 비교해야 한다.

### 3.5.1. Score Map에서 보고 anomaly score까지

- 최종 출력은 native-resolution pixel anomaly map이다. evaluator는 이 map과 GT를 이용해 segmentation metrics를 계산하지만, method는 GT를 사용하지 않는다.
- image-level diagnostic score는 최종 map의 maximum으로 기록된다. 따라서 pixel map의 spatial structure와 image-level ranking은 같은 query prediction에서 파생된다.
- `component_quantile`은 component 내부의 상대적인 score shape를 보정할 수 있지만, 새로운 query feature나 support 정보를 추가하지 않는다. 이 때문에 성능을 해석할 때에는 fusion rule의 효과와 postprocessing의 효과를 분리해야 한다.

---

### 3.6. Anomaly Scoring and Computational Efficiency

- **Parameter and computation accounting**:
  - 제안 프레임워크는 추가 학습 파라미터와 역전파를 사용하지 않는다.
  - Runtime과 throughput은 frozen backbone inference, distance computation, projection 및 후처리를 포함한 별도 benchmark로 보고하며, 측정되지 않은 FPS 주장은 하지 않는다.
- **한 번 수행하는 support 단계**:
  1. 정상 support image를 frozen backbone으로 통과시켜 layer·resolution별 feature를 추출한다.
  2. 각 bank의 support feature를 고정하고 leave-source-out normal distance에서 q95 scale을 계산한다.
  3. normalized shallow support response에서 q99 gate reference를 계산한다.
  4. 이 단계가 끝나면 bank, q95, q99는 query 평가 동안 변경하지 않는다.
- **각 query에 대해 수행하는 단계**:
  1. query를 동일한 frozen backbone과 resolution 집합으로 처리한다.
  2. 각 layer·resolution·bank component의 nearest-neighbor distance를 계산한다.
  3. support-derived q95로 component를 scaling한다.
  4. deep base를 만들고 shallow q99 residual을 추가한다.
  5. bank와 resolution score를 native space에서 고정된 규칙으로 결합한다.
  6. CQ/evaluator contract를 적용해 최종 anomaly map과 metrics를 계산한다.
- 이 순서는 support fitting과 query evaluation을 분리해, query label이나 GT가 method fitting에 유입되지 않도록 한다. 또한 source calibration audit에서 사용하는 별도 normal calibration images는 operating threshold 계산에만 사용하며 feature fitting이나 prediction state를 갱신하지 않는다.

---

## 4. Evidence and Analysis

---

### 4.1. Experimental Setup and Reading Rules

- **Evidence layers**: 이 절은 (i) MVTec AD 2의 strict paired few-shot 비교, (ii) 동일 pipeline 안의 RSG3 factorial, (iii) MVTec AD와 VisA의 저장된 R1 실행, (iv) MVTec AD clean-support-to-corrupted-test 분석을 구분한다. 표 사이의 절대 수치를 직접 순위 비교하지 않는다.
- **MVTec AD 2 controlled setting**: `test_public_seen_development`, fixed support manifest, 16-shot, seed0, frozen DINOv2-L, full8이다. Query label과 mask는 evaluator에서만 사용하며 sealed final holdout이 아니다.
- **MVTec AD / VisA setting**: 정상 support만으로 bank, q95 scale, q99 gate를 적합한 R1 실행이다. MVTec AD는 15개 클래스, VisA는 12개 클래스의 macro average이며 1/2/4/8/16-shot 값은 저장된 seed0 run에서 계산했다.
- **Metrics**: primary는 `seg_AUPRO@0.05`, guardrail은 `seg_AUROC@0.05` (pAUROC), diagnostic은 `seg_F1_oracle`이다. Oracle F1은 test label로 threshold를 선택하므로 source-fixed alarm transfer와 동등하게 해석하지 않는다.

---

### 4.2. Main Few-Shot Comparison on MVTec AD 2

먼저 기존 main table의 strict paired P16 비교를 제시한다. 두 행은 동일한 SuperAD-selected support-16 manifest, public query population, original-resolution evaluator를 사용한 DINOv2-L/14 비교다. `Ours Basic`은 4.3의 FF_D factorial과 실행 identity가 다른 이전 R1 baseline이므로, 이 표는 few-shot baseline context를 제공한다.

| Method                        | Backbone        | Normal-data budget   |   pAUROC@0.05 (%) |   AU-PRO@0.05 (%) | Seg-F1 oracle (%) |
| :---------------------------- | :-------------- | :------------------- | ----------------: | ----------------: | ----------------: |
| SuperAD-16                    | DINOv2 ViT-L/14 | P16                  |           76.1857 | **53.4924** | **36.8423** |
| Ours Basic (strict paired R1) | DINOv2 ViT-L/14 | P16, same references | **77.2635** |           53.3408 |           35.5676 |

- Ours Basic is `+1.0778pp` pAUROC, `-0.1516pp` AU-PRO, and `-1.2748pp` oracle F1 relative to SuperAD-16. 따라서 external baseline에 대한 우월성 주장이 아니라 동일 few-shot 조건의 tradeoff를 기록하는 표다.
- DINOv3, private-server, RoBiS, SuperADD 행은 support budget, backbone, postprocessing 또는 metric provenance가 이 strict paired block과 다르므로 주 비교의 승패 근거로 사용하지 않는다.

---

### 4.3. Controlled Factorial: What the Fusion Rule Changes

현재 방법의 직접 근거는 이전 main table이 아니라 아래 5-arm factorial이다. Backbone, support manifest, four resolutions, four banks, CQ, evaluator와 query population을 모두 고정하고 tail scale 및 shallow contribution만 바꾸었다.

| Method                                      | Backbone | Trainable Params |     AUPRO@0.05 (%) |    pAUROC@0.05 (%) |      F1-oracle (%) |
| :------------------------------------------ | :------- | :--------------: | -----------------: | -----------------: | -----------------: |
| FF_A: MAD + ungated shallow                 | DINOv2-L |        0        |           51.15843 |           73.01798 |           29.97541 |
| FF_B: q95 + ungated shallow                 | DINOv2-L |        0        |           54.01569 |           74.03991 |           32.90033 |
| FF_C: MAD + gated residual                  | DINOv2-L |        0        |           57.48191 |           76.15659 |           33.89213 |
| **FF_D: q95 + gated residual (RSG3)** | DINOv2-L |   **0**   | **59.14576** |           77.10613 |           35.86782 |
| FF_E: q95 + deep-only                       | DINOv2-L |        0        |           58.67094 | **77.35072** | **37.11050** |

- 이 표는 `test_public_seen_development`, 16-shot seed0, full8의 controlled screening이며 final holdout, seed-level statistical generalization, 또는 모든 FPR에서의 우월성을 나타내지 않는다.
- `FF_D − FF_B`: q95 조건에서 residual gating은 `+5.13007pp` AUPRO (6/8 class wins)다. `FF_D − FF_C`: gated condition에서 q95 scaling은 `+1.66385pp` (8/8 wins)다.
- `FF_D − FF_E`: deep-only 대비 shallow residual의 추가 효과는 `+0.47482pp`이고 pAUROC는 `−0.24459pp`다. 즉, 이 결과가 지지하는 핵심은 shallow response의 무조건적 추가가 아니라 ungated shallow response를 support-tail residual로 제한하는 rule이다.

---

### 4.4. Few-Shot Scaling on MVTec AD and VisA

이 결과는 동일한 R1 score-level pipeline의 저장된 cross-dataset 실행이다. MVTec AD와 VisA에는 4.3 factorial의 strict paired controls가 없으므로, dataset/backbone generalization의 확정 근거나 다른 방법과의 직접 비교가 아니라 support 수에 따른 동작 기록으로 해석한다.

| Dataset / backbone    |  1-shot |  2-shot |  4-shot |  8-shot | 16-shot |
| :-------------------- | ------: | ------: | ------: | ------: | ------: |
| MVTec AD, DINOv2-L/14 | 78.3878 | 81.7502 | 83.6327 | 84.7105 | 85.3614 |
| MVTec AD, DINOv3-L/16 | 75.3031 | 83.0547 | 85.2780 | 86.3316 | 87.0533 |
| VisA, DINOv2-L/14     | 72.5875 | 79.0020 | 81.7122 | 83.5876 | 84.8191 |
| VisA, DINOv3-L/16     | 63.8828 | 74.2170 | 77.9157 | 79.2498 | 81.5638 |

표의 값은 `seg_AUPRO@0.05 (%)` macro average다. MVTec AD 16-shot의 DINOv2/DINOv3 pAUROC는 88.9163/91.1067이고, VisA 16-shot은 91.1344/92.9551이다. 두 dataset 모두 shot 증가에 따라 각 저장된 seed0 run의 AUPRO가 상승하지만, repeated seed, matched ablation, 또는 cross-backbone statistical claim은 아니다.

---

### 4.5. Clean-Support to Corrupted-Test Distribution Shift on MVTec AD

MVTec AD clean train/good support를 고정하고 test의 정상·결함 이미지만 ICCV 2023 corruption protocol의 severity-3 corruption으로 변경했다. GT mask는 원본과 동일하며, target corruption으로 support bank나 threshold를 다시 적합하지 않았다.

| Target test condition | pAUROC@0.05 (%) | AUPRO@0.05 (%) | Δ AUPRO from clean (pp) |
| :-------------------- | --------------: | -------------: | -----------------------: |
| Clean                 |         88.9163 |        85.3614 |                  +0.0000 |
| Brightness            |         88.2585 |        84.4245 |                  -0.9369 |
| Contrast              |         87.4368 |        83.1513 |                  -2.2101 |
| Defocus blur          |         86.7114 |        81.9237 |                  -3.4377 |
| Gaussian noise        |         78.4576 |        64.6741 |                 -20.6873 |

- 네 corruption 모두 15-class macro에서 clean보다 AUPRO가 낮다. Gaussian noise의 큰 하락은 support-calibrated fusion이 모든 appearance shift를 흡수한다는 주장을 반박하는 중요한 negative result다.
- 이 표의 F1은 각 target test split에서 얻은 oracle F1이므로 fixed-threshold alarm transfer를 측정하지 않는다. 따라서 ranking/segmentation degradation을 보이지만 operating-threshold robustness를 증명하지 않는다.

---

### 4.6. Source-Fixed Operating-Point and Mechanistic Audit

이 audit은 support와 분리된 normal calibration 16장으로 source nominal FPR 1% threshold를 고정한 뒤 regular와 `shift_1`의 normal FPR, defect TPR, PRO를 비교하도록 설계했다. Threshold는 target query나 target label로 조정하지 않으며, layer-wise raw/normalized response, deep/shallow/residual contribution, gate activation, paired scene summary를 함께 기록한다.

- 현재 32 cell 중 28 cell만 완료되어 full8 macro를 등록하지 않는다. 완료된 7-class partial 결과에서 CQ source1% D−B의 PRO는 `+7.10004pp`였지만 FPR increase는 `+0.78731pp` 더 컸고, FPR no-worse criterion은 4/7 class만 만족했다. 이 partial signal은 robustness claim의 근거가 아니라 sheet_metal recovery 뒤 full8로 확인할 tradeoff다.
- Support-derived q95/q99은 component scale과 gate를 정의하지만 target FPR을 보장하지 않는다. Gate activation만으로 nuisance removal 또는 defect preservation을 주장하지 않으며, fixed-source FPR·TPR·PRO와 paired response가 같은 방향을 보일 때만 해석한다.
- CQ/component-quantile postprocessing은 factorial의 모든 arm에 고정된 pipeline이다. 별도 CQ-off/on positive ablation이 없으므로 독립 기여로 서술하지 않는다.

---

### 4.7. Evidence Boundary

- 4.3은 현재 논문의 가장 직접적인 evidence다: 동일 condition에서 support calibration과 residual gate만 바꾼 비교다.
- 4.2, 4.4, 4.5는 각각 strict baseline context, cross-dataset/scaling record, synthetic shift stress test를 제공하지만 protocol이 다르므로 FF_D의 component effect를 대체하지 않는다.
- 전체 결과는 single seed 또는 제한된 public development population에 기반한다. Seed replication, sealed evaluation, source-fixed audit의 full8 completion, CQ 및 multi-resolution/bank의 독립 ablation이 논문-ready claim을 확장하기 위한 후속 근거다.

---

## 5. Discussion & Conclusion

---

### 5.1. Capacity vs. Support-Calibrated Fusion

- **모델 용량(Capacity) 대 표현 보정(Calibration)의 패러다임 전환**:
  - 본 실험은 제한된 정상 support에서 추가 layer response를 무조건 누적하기보다, support-derived scale과 residual gate를 사용해 fusion을 조절하는 것이 유리할 수 있음을 보여준다.
  - 이 결과는 특정 backbone의 표현 용량이나 모든 adapter의 실패를 증명하는 것이 아니며, 동일한 development 조건에서 관찰된 controlled comparison으로 해석한다.

---

### 5.2. Boundary of Claims and Future Work

- **주장 범위(Boundary of Claims) 및 이론적 한계의 명확화**:
  - 본 연구의 유효 주장 범위는 16-shot seed0, full8의 development 조건에서 frozen DINOv2 representation을 support-calibrated fusion으로 결합하는 방법과 그 controlled comparison으로 한정됨.
  - 결과는 single seed와 seen development population에 기반하며, final holdout, statistical generalization, universal shift robustness를 확정하지 않는다.
  - Multi-resolution, multi-bank, CQ의 독립 기여와 DINOv3 transferability는 동일 조건의 별도 검증 없이는 주장하지 않는다.
  - 부품의 임의적 3D 회전이나 시점 불일치를 수반하는 전면적인 카테고리 전이(Category Shift) 해결은 본 연구의 직접적 주장 범위를 넘어서는 과제로 규정함.
- **향후 연구 과제(Future Work)**:
  - source-fixed audit 완료 후 seed/shot replication과 sealed evaluation을 수행하고, 동일 protocol의 backbone transfer 및 multi-resolution/multi-bank/CQ ablation을 검토함.
