# Support-Calibrated Residual Fusion: Removing the Support Dependence of Multi-Layer Fusion in Few-Shot Industrial Anomaly Detection

**Hun Im\*, Author Two, Author Three, Pilsung Kang\***

> **v6 변경 요약** (v5 대비)
> 1. 내부 시도 식별자(`FF_*`, `SM_*`, `TC*`, `MS*`, `CB*`, `BC*`, `LR_*`, `CM_*`) **전면 제거**, 표준 ablation 표기로 교체.
> 2. 측정 noise floor 0.865pp를 **사전 판정 규칙**으로 승격, 5-seed 재현을 반영해 v5의 과도한 single-seed 면책 정정.
> 3. few-shot scaling·acquisition shift·공식 server shift 증거를 본문에 편입.
> 4. **§4.3을 성분 수준 shot-축 ablation으로 교체.** 1~16-shot에서 arm별 비교를 수행해, v5·v6 초안이 "직접 근거 없음"으로 유보했던 저샷 성분 주장을 근거로 확정.
> 5. **§4.6을 사전 등록 가설의 기각 기록으로 재작성.** coverage-void를 두 조건의 공유 매개로 두는 가설을 계측으로 검정해 **기각**했고, 그 결과를 숨기지 않고 negative result로 보고.
> 6. 위 기각에 따라 **§1.2의 중심 명제를 잠재 상태(coverage void)에서 측정 가능한 score 성분(보정되지 않은 shallow 기여)으로 교체.**

---

## Index

- Abstract
- 1. Introduction
  - 1.1. Two Independent Deployment Conditions
  - 1.2. A Shared Fragile Component: Uncalibrated Shallow Contribution
  - 1.3. The Asymmetry That Forces a Support-Only Remedy
  - 1.4. Main Problem: Layer-Wise Score Reliability under Limited Support
  - 1.5. Direct Response and Testable Predictions
  - 1.6. Contributions
- 2. Pilot Evidence and Measurement Discipline
  - 2.1. Layer Complementarity as Background Motivation
  - 2.2. Support-Limited Scale and Normal-Tail Mismatch
  - 2.3. The Measurement Noise Floor as a Pre-Registered Decision Rule
  - 2.4. Why Support-Only Adaptive Selection Cannot Work
  - 2.5. Scope and Limits of the Pilot Evidence
- 3. Method: Support-Calibrated Residual Fusion
  - 3.1. Problem Formulation
  - 3.2. Framework Overview
  - 3.3. Asymmetric Residual Spike Gating
  - 3.4. Tail-Calibrated Quantile Normalization
  - 3.5. Score Projection and Component Quantile Postprocessing
  - 3.6. Anomaly Scoring and Computational Efficiency
- 4. Evidence and Analysis
  - 4.1. Experimental Setup and Reading Rules
  - 4.2. Component Ablation: What the Fusion Rule Changes
  - 4.3. Condition A — Behavior as the Support Budget Shrinks
  - 4.4. Condition B — Behavior as the Acquisition Condition Changes
  - 4.5. A Failure Mode Outside the Claimed Range
  - 4.6. A Rejected Hypothesis: Instrumenting a Shared Latent State
  - 4.7. Source-Fixed Operating-Point Audit
  - 4.8. Evidence Boundary
- 5. Discussion & Conclusion
  - 5.1. Capacity and Adaptation vs. Support Calibration
  - 5.2. Boundary of Claims and Future Work

---

## Abstract

Few-shot 산업 이상 탐지에서는 소수의 정상 reference만으로 검사 기준을 만들어야 한다. 실제 배치에서는 두 가지 조건이 동시에 작용한다. 하나는 **정상 데이터 예산이 작다는 제약(few-shot)**이고, 다른 하나는 **촬영 환경이 변한다는 변화(distribution shift)**다. 이 둘은 서로 독립적인 외생 조건이지만, 두 조건 모두에서 성능 저하를 담지하는 것은 동일한 score 성분이다. 즉 **정상 support 기준으로 보정되지 않은 shallow 기여**다. Shallow response는 결함 단서와 관측되지 않은 정상 변동에 함께 반응하므로, support가 좁아지거나 query 획득 조건이 이동할 때 그 정상 tail이 그대로 anomaly evidence로 누적된다. 이 관점의 검증 가능한 함의는 그 성분을 제한하는 **단일 규칙이 두 조건 모두에서 큰 이득을 내야 한다**는 것이다.

두 조건의 교집합에서 사용 가능한 정보는 정상 support 통계뿐이다. Target label도, target 분포도, 추가 학습 신호도 없다. 따라서 본 연구는 component별 leave-source-out q95로 score scale을 맞추고, deep score를 base로 유지하면서 support q99를 넘은 shallow residual만 조건부로 추가하는 **Support-Calibrated Residual Fusion**을 제안한다. 추가 학습 파라미터는 0개이며 backbone은 동결한다.

MVTec AD 2에서 backbone, support manifest, resolution, bank, postprocessing, evaluator를 고정한 ablation 결과, 제안 규칙은 16-shot AUPRO@0.05 59.15로 모든 대안 구성보다 높았고, residual gating이 +5.13pp, tail calibration이 +1.66pp를 기여했다. 두 값 모두 bank-subsampling seed로 측정한 0.865pp 측정 noise floor를 상회하며, 주 구성은 5개 support-split seed에서 59.48 ± 0.60으로 재현되었다.

핵심 증거는 support 예산을 1장까지 줄이면서 동일 ablation을 반복한 120-cell 실험이다. **Residual gating의 이득은 저샷에서 사라지지 않고 오히려 커진다**(16/8/4/2-shot에서 6.51 / 6.66 / 7.50 / 11.49pp, 획득 조건이 이동한 `shift_1`에서 4.61 / 4.60 / 5.09 / 10.02pp). 원인은 두 구성의 support 의존성 차이다. 2→16-shot 구간에서 **제안 규칙의 성능 변화는 0.44pp로 측정 floor 미만인 반면, 보정 없이 shallow를 합산하는 구성은 5.41pp 변한다.** 그 결과 **제안 규칙은 정상 이미지 2장만으로 보정 없는 구성의 16장 성능을 6.07pp 앞선다.** 동일 backbone과 bank에서 fusion 규칙만 다른 내부 통제 비교다. 반면 shallow 가지 자체의 순이득은 다섯 shot 수준 전부에서 floor 미만(−0.15 ~ +0.81pp)이므로, 기여는 shallow evidence의 추가가 아니라 그 정상 tail의 제한에 있다.

Support를 고정한 채 query 획득 조건만 바꾼 비교에서 제안 규칙은 `shift_1` 절대 AUPRO 59.40으로 1위였고, 공식 MVTec AD 2 private→private-mixed 서버 평가에서 상대 Seg-F1 저하가 -4.11%로 비교 방법 중 가장 작았다. 두 구성 요소의 역할은 구분된다. **Gating은 두 조건 모두에서 큰 절대 이득을 제공하지만 상대 저하폭을 줄이지는 않으며, 저하폭 감소는 tail calibration에서 관찰된다.** 또한 severity-3 Gaussian noise에서 -20.69pp의 하락이 나타나므로, 주장 범위를 photometric·acquisition shift로 한정하고 high-frequency sensor corruption을 명시적 실패 모드로 보고한다. 본 논문은 두 배치 조건을 하나의 잠재 상태로 묶는 설명을 사전 등록 후 계측으로 검정해 **기각**했으며(§4.6), 그 대신 직접 측정 가능한 score 성분 수준의 설명을 채택한다.

---

## 1. Introduction

### 1.1. Two Independent Deployment Conditions

신규 품목과 공정 전환이 잦은 산업 현장에서 이상 탐지 시스템은 두 가지 조건을 동시에 만난다.

- **조건 A — 배치 제약 (few-shot)**: 대규모 정상 데이터를 다시 수집하고 모델을 학습하기 어렵다. 따라서 소수의 정상 reference로 검사 기준을 구성해야 한다. 이는 **설계 시점에 이미 알려진 정적 제약**이다. Support 개수는 우리가 아는 값이다.
- **조건 B — 환경 변화 (distribution shift)**: 조명, 노출, 카메라 위치, 센서 특성은 배치 이후에 변한다. 이는 **배치 이후에 발생하는 동적 변화**이며, 언제 어떤 방향으로 일어날지 사전에 알 수 없다.

두 조건은 발생 원인이 다르다. 조건 A는 데이터 수집 예산의 문제이고, 조건 B는 물리적 촬영 환경의 문제다. 한쪽을 해결한다고 다른 쪽이 해결되지 않으며, 두 조건은 독립적으로 가해질 수 있다. 본 논문은 이 둘을 **두 개의 서로 독립적인 외생 조건**으로 취급한다.

Frozen foundation backbone을 사용하는 최근접 이웃 기반 방법은 정상 support의 patch feature를 bank에 저장하고, query patch와의 거리를 anomaly score로 사용한다. 이 방식은 추가 학습 부담이 작지만, 큰 거리가 결함 때문인지 support가 포괄하지 못한 정상 변화 때문인지는 거리 자체만으로 구분하기 어렵다.

---

### 1.2. A Shared Fragile Component: Uncalibrated Shallow Contribution

두 조건은 서로 다른 원인에서 출발한다. 그러나 frozen backbone 위의 nearest-neighbour score에서 **두 조건의 피해를 실제로 담지하는 성분은 같다.**

```text
[조건 A] Few-shot  ─ support가 정상 변동의 ─┐
  배치 제약, 정적, 기지   일부만 관측         │
                                            ├─→  보정되지 않은 shallow 기여가
                                            │    정상 반응을 anomaly evidence로
[조건 B] Distribution shift ─ query가 관측 ─┘    누적  ─→  low-FPR 구간 오염
  환경 변화, 동적, 미지      범위 밖으로 이동
```

- **Shallow response는 양의성(兩義性)을 갖는다.** 작은 결함의 경계와 texture 변화를 포착하는 동시에, 조명·반사·센서·자세의 정상적 변화에도 크게 반응한다. Score만으로는 두 원인을 구분할 수 없다.
- **조건 A는 support 쪽에서** 이 양의성을 악화시킨다. Support가 좁으면 정상 변동의 상당 부분이 관측되지 않은 채 남고, 그 부분이 shallow에서 큰 거리로 나타난다.
- **조건 B는 query 쪽에서** 같은 일을 한다. Support는 고정된 채 획득 조건이 이동하면, 정상 query가 관측 범위 밖으로 밀려나 같은 방식으로 shallow를 자극한다.
- Deep score는 같은 자극에 상대적으로 덜 반응한다. 따라서 문제는 "multi-layer를 쓸 것인가"가 아니라 **"shallow 기여를 어떤 기준으로 허용할 것인가"**이다.

여기서 중심 명제는 잠재 상태가 아니라 **관측 가능한 score 성분**에 대한 것이다. 이 구분이 중요하다. 우리는 초고에서 두 조건이 공유하는 잠재 상태(정상 query가 support 관측 범위 밖에 놓이는 "coverage void")를 가정하고, 그것을 support 보정의 사전 null 대비 편차로 계측하는 실험을 사전 등록해 수행했다. **그 가설은 기각되었다**(§4.6). 계측된 양은 두 축 어디에서도 성능 손실을 설명하지 못했고, support 예산 축에서는 예측과 반대 방향으로 움직였다. 반면 score 성분 수준의 명제는 같은 실험에서 강하게 지지되었다. 따라서 본 논문은 다음을 주장한다.

> **두 배치 조건은 독립적이지만, 두 조건 모두에서 성능 저하를 담지하는 것은 보정되지 않은 shallow 기여다. 정상 support 기준을 넘은 부분만 허용하도록 그 성분을 제한하면, 두 조건 모두에서 측정 floor를 크게 상회하는 이득을 얻는다.**

이 명제는 개입 지점을 특정한다. 본 논문의 방법은 support 수를 늘리지도(조건 A 제거), 도메인 적응을 수행하지도(조건 B 제거) 않는다. **두 조건이 공유하는 취약 성분에서만 개입한다.**

---

### 1.3. The Asymmetry That Forces a Support-Only Remedy

두 조건은 같은 성분 위에서 작동하지만 성격은 비대칭이다. 이 비대칭이 해법의 형태를 강제한다.

| | 조건 A: Few-shot (배치 제약) | 조건 B: Shift (환경 변화) |
| :-- | :-- | :-- |
| 성격 | 정적, 설계 시점에 **알려짐** | 동적, 배치 후에 **알려지지 않음** |
| void 발생 축 | support 쪽이 좁아짐 | query 쪽이 이동함 |
| void 크기 | support 수로 결정, 예측 가능 | shift 종류에 의존, 방향이 사전에 미지 |
| 개입 시 가용 정보 | 정상 support 통계 | 정상 support 통계 **뿐** |

조건 B에서는 target 분포를 미리 알 수 없으므로, target 데이터에 의존하는 개입은 배치 시점에 사용할 수 없다. 조건 A에서는 결함 sample이 존재하지 않으므로, 판별 신호를 학습하는 개입도 사용할 수 없다. **두 조건의 교집합에서 남는 정보는 정상 support 통계뿐이다.** 따라서 개입은 support-derived quantile 위에서 이루어져야 한다. 이는 편의를 위한 설계 선택이 아니라 문제 구조가 강제하는 제약이며, §2.4와 §5.1은 이 제약을 위반한 방법군이 어떻게 실패했는지를 기록한다.

---

### 1.4. Main Problem: Layer-Wise Score Reliability under Limited Support

제한된 정상 support 하에서 핵심 문제는 여러 layer를 사용할지 여부가 아니라, 각 layer가 만든 거리를 어떤 기준으로 비교하고 결합할 것인가이다.

- **Scale mismatch**: Layer, resolution, bank가 달라지면 정상 support에서 관찰되는 distance scale도 달라진다. 동일한 수치의 score가 component마다 동일한 비정상성을 뜻하지 않는다.
- **Normal-tail mismatch**: Shallow response는 국소 변화에 민감한 동시에, support에 포함되지 않은 정상 texture·반사·조명 변화에도 크게 반응한다. 즉 **§1.2의 취약 성분이 가장 직접적으로 드러나는 곳이 shallow의 upper tail이다.** 이를 그대로 합산하면 소수의 큰 정상 반응이 low-FPR 구간을 지배한다.
- **Reliability asymmetry**: Deep score는 상대적으로 안정적인 base가 될 수 있고 shallow score는 보완적인 국소 evidence를 제공할 수 있지만, shallow 전체를 항상 신뢰할 근거는 없다.
- **Naive fusion failure mode**: 보정되지 않은 score를 평균하거나 더하면 component 수가 늘수록 evidence가 풍부해지는 것이 아니라, 불안정한 component의 정상 tail이 최종 anomaly map에 누적된다.

따라서 본 연구의 질문은 다음과 같다.

> 제한된 정상 support만 사용하여 component별 score scale을 맞추고, shallow response 중 정상 범위를 벗어난 부분만 anomaly evidence로 허용할 수 있는가? 그리고 그렇게 만든 하나의 규칙이 few-shot과 shift 두 조건 모두에서 작동하는가?

이 문제는 backbone의 inductive bias를 수정하거나 결함 표현을 복원하는 문제가 아니다. Frozen representation 위에서 생성되는 score의 상대적 신뢰도를 support statistics로 정하는 score-construction 문제다.

---

### 1.5. Direct Response and Testable Predictions

문제의 각 요소는 간단한 score-level rule에 직접 대응한다.

| Observed problem | Direct response |
| :-- | :-- |
| Component별 distance scale 차이 | 정상 support의 leave-source-out q95로 component score scaling |
| Shallow normal-tail의 과도한 기여 | 정상 support q99를 넘은 shallow residual만 통과 |
| Deep/shallow reliability asymmetry | Deep score는 base로 유지하고 shallow residual만 조건부 추가 |
| Target 분포·결함 label 부재 | Frozen backbone과 parameter-free, support-only score construction |

**검증 가능한 예측.** §1.2의 명제는 다음 예측을 낳는다. 각 예측은 4장에서 검증하며, 실패 시 어떤 서술을 철회할지 미리 명시한다.

| | 예측 | 검증 | 결과 |
| :-- | :-- | :-- | :-- |
| **P1** | Support 예산이 줄어도 gating의 이득이 **사라지지 않아야** 한다. 취약 성분이 조건 A에서 악화된다면 이득은 오히려 **커져야** 한다 | §4.3 | **지지**. 16→2-shot에서 6.51 → 11.49pp |
| **P2** | Support 고정·query만 이동시켜도 gated 구성이 ungated 구성보다 **절대 성능 우위**를 유지해야 한다 | §4.4 | **지지**. `shift_1`에서 +4.61 ~ +10.02pp |
| **P3** | 보정 없는 구성은 support 예산에 크게 의존하고, gated 구성은 **덜 의존해야** 한다 | §4.3 | **지지**. 2→16-shot 변화 5.41pp 대 0.44pp |
| **P4** | 두 조건이 공유하는 **잠재 상태**가 존재하여, 양축에서 만든 그 상태값이 같은 곡선 위에서 성능 저하와 연결되어야 한다 | §4.6 | **기각**. 아래 참조 |

P1~P3은 관측 가능한 score 성분과 그 제거 효과에 대한 예측이고, P4는 그보다 강한 잠재 상태 주장이다. 우리는 P4를 사전 등록된 판정 규칙과 함께 계측으로 검정했고 **기각했다**(§4.6). 본 논문의 주장은 따라서 P1~P3이 지지하는 범위로 한정되며, 두 조건을 단일 잠재 변수로 환원하지 않는다.

---

### 1.6. Contributions

- **C1. 두 배치 조건이 공유하는 취약 성분을 특정함**:
  - Few-shot(배치 제약)과 distribution shift(환경 변화)를 독립적인 두 외생 조건으로 두고, 두 조건의 피해를 담지하는 것이 보정되지 않은 shallow 기여임을 개입 지점으로 정식화함.
  - 두 조건의 비대칭으로부터 해법이 support-only여야 함을 연역함.
- **C2. 단순한 score-level 방법을 제안함**:
  - layer·resolution·bank별 score를 support 통계로 보정하고, deep base에 shallow residual을 조건부로 추가하는 Support-Calibrated Residual Fusion. 추가 학습 파라미터 0개.
- **C3. 동일 pipeline의 통제된 ablation 근거를 제시함**:
  - scaling 방식과 shallow 기여 방식을 교차한 ablation으로 두 구성 요소의 기여를 분리하고, 두 구성 요소가 서로 다른 역할을 함을 보임.
- **C4. 두 축 모두에서 성분 수준으로 검증함**:
  - 1~16-shot × 3 arm × 8 object의 120-cell 통제 실험으로, gating 이득이 저샷에서 유지될 뿐 아니라 커짐을 보이고, 제안 규칙이 보정 없는 구성의 support 의존성을 제거함을 정량화함.
  - Support-고정 acquisition shift와 공식 server real-shift 평가에서 동일 규칙의 이득이 유지됨을 보임.
- **C5. 판정 기준, 실패 경계, 그리고 기각된 가설을 명시함**:
  - bank-subsampling seed로 0.865pp 측정 noise floor를 확립해 유효·무효 개선의 판정 규칙을 사전 고정함.
  - 두 조건을 단일 잠재 상태로 환원하는 더 강한 설명을 사전 등록 후 검정해 **기각**하고 그 근거를 보고함(§4.6). 이는 본 방법의 유효성과 무관하지만, 어떤 설명이 데이터의 지지를 받는지에 대한 경계를 정한다.
  - Gaussian noise corruption을 명시적 실패 모드로 보고함.

---

## 2. Pilot Evidence and Measurement Discipline

이 장은 Introduction의 문제를 반복하지 않고, 방법 설계에 직접 사용된 pilot evidence와 측정 규율을 제시한다. 각 결과는 일반 이론이나 인과적 증명이 아니라 설계 근거로 사용한다.

---

### 2.1. Layer Complementarity as Background Motivation

Shallow와 deep layer가 서로 다른 반응을 보인다는 사실은 multi-layer score를 고려하는 출발점이지만, 그 자체가 본 연구의 해결 문제는 아니다.

- Shallow layer는 texture, edge와 작은 appearance change에 상대적으로 민감하다. 이 민감도는 결함 단서와 정상적인 조명·반사 변화에 모두 반응한다.
- Deep layer는 더 안정적인 base score를 제공할 수 있지만 모든 국소 결함에서 우월하다는 보장은 없다. 단일 layer 진단에서 구조적 객체는 deep이 유리했고, 세 layer의 무보정 결합은 오히려 deep의 이점을 희석했다.
- 이러한 차이를 inductive bias mismatch의 직접 증명으로 해석하지 않는다. 현재 방법은 layer representation이나 attention을 바꾸지 않기 때문이다.
- 따라서 pilot evidence가 답해야 할 질문은 "어떤 layer가 본질적으로 옳은가"가 아니라 **"정상 support에서 각 layer score가 어느 범위까지 안정적인가"**이다.

---

### 2.2. Support-Limited Scale and Normal-Tail Mismatch

각 component $(r,b,l)$에 대해 정상 support bank $B_{r,b,l}$와 query patch $p$ 사이의 거리를 다음과 같이 둔다.

$$
d_{r,b,l}(p)=\min_{u\in B_{r,b,l}}D\bigl(f_{r,l}^{q}(p),u\bigr).
$$

이 raw distance의 크기와 upper tail은 layer, resolution, bank에 따라 달라진다. 따라서 $d_{r,b,l_1}(p)>d_{r,b,l_2}(p)$라는 사실만으로 첫 번째 component가 더 강한 anomaly evidence라고 판단할 수 없다.

제한된 support 하에서는 이 문제가 더 중요하다. Support가 좁거나 query가 이동하면 정상 변동의 tail을 조밀하게 관찰할 수 없고, 관측되지 않은 정상 pattern이 일부 component에서 큰 distance로 나타난다. 여러 raw score를 그대로 합치면 이러한 component가 최종 score를 지배한다.

이에 따라 본 연구는 feature representation에 대한 원인 가설보다 support에서 직접 관찰할 수 있는 두 통계량에 집중한다.

- component별 leave-source-out normal distance의 q95: score scale의 기준
- normalized shallow response의 q99: shallow contribution을 허용하는 기준

이 통계량은 target distribution이나 defect label로 조정하지 않는다. 또한 target FPR이나 defect preservation을 보장하지 않으므로 §4.7에서 별도로 검증한다.

---

### 2.3. The Measurement Noise Floor as a Pre-Registered Decision Rule

개선 주장을 하기 전에 측정 분해능을 확정했다. 동일한 코드와 파라미터에서 **memory bank token subsampling seed만** 바꾼 5회 실행의 결과 범위는 다음과 같다.

| 지표 | 5회 실행 범위 |
| :-- | --: |
| AUPRO@0.05 | **0.865pp** |
| pAUROC@0.05 | 0.908pp |
| F1 oracle | 0.501pp |

> **판정 규칙 (본 논문 전체에 적용)**: 단일 실행 간 **0.865pp 미만의 AUPRO 차이는 개선 근거로 사용하지 않는다.** 그러한 차이는 방향성으로만 기술한다. 이 기준은 결과 확인 이전에 고정했으며, 본 논문의 모든 ablation 판정에 동일하게 적용한다.

이 규칙의 직접적 귀결로, 본 논문은 자신에게 유리한 여러 결과를 근거에서 제외한다. 예를 들어 국소 관계 항 추가는 5-seed 평균 +0.110pp, 공간 일관성 제약과 형태학적 region recovery는 각각 0pp 및 -0.051pp로, 전부 floor 미만이므로 기여로 주장하지 않는다.

추가로 주 구성은 서로 다른 5개 support-split seed에서 반복 측정되었다.

| | AUPRO@0.05 |
| :-- | --: |
| 5-seed 평균 | **59.478** |
| 표준편차 | 0.600 |

따라서 본 논문의 주 결과는 single-seed screening이 아니다. 다만 sealed final holdout 평가는 수행하지 않았다(§4.8).

---

### 2.4. Why Support-Only Adaptive Selection Cannot Work

§1.3은 개입이 support-only여야 함을 논증했다. 그러나 support-only라고 해서 무엇이든 가능한 것은 아니다. 이 경계를 명확히 하는 것이 고정 규칙 설계의 직접적 근거다.

- 16장의 정상 support만으로 layer 가중치를 학습하거나 선택하는 selector 5종을 구현해 평가했다. **다섯 모두 8개 클래스 중 5개만 맞혀 우연 수준에 머물렀다.**
- 원인은 구조적이다. 최적 selector가 필요로 하는 양은 결함 반응과 정상 반응의 비율인데,

  $$\text{selector goal} = \frac{\mathbb{E}_{defect}[D(x)]}{\mathbb{E}_{normal}[D(x)]},$$

  정상 support 집합 $\mathcal{S}_N$에는 결함 sample이 하나도 없으므로 **분모만 추정할 수 있다.**
- 합성 결함으로 분자를 대체하려는 시도도 갭을 메우지 못했다. 합성 결함은 layer 간 분리도가 사실상 동일했던 반면(0.7457 대 0.7453), 실제 결함은 뚜렷이 갈렸다(0.677 대 0.800). 합성 결함은 "일관되게 틀린 texture"이고 실제 산업 결함은 sub-patch 미세 흠결이기 때문이다.

> **설계 귀결**: support 통계는 component의 **scale을 보정할 수는 있어도 어떤 component가 판별적인지를 고를 수는 없다.** 따라서 본 방법은 학습되거나 적응되는 가중치를 두지 않고, support quantile로 보정한 뒤 **고정된 규칙**으로 결합한다.

---

### 2.5. Scope and Limits of the Pilot Evidence

- 본 pilot은 layer별 score distribution과 fusion behavior를 관찰한 개발 근거이며, ViT attention의 인과 구조나 representation-level defect preservation을 증명하지 않는다.
- Frozen backbone 사용은 §1.3의 논증에서 따라오는 설계 선택이다. 일부 trainable alternatives가 특정 16-shot 조건에서 열세였다는 사실을 모든 adapter 또는 parameter learning의 일반적 실패로 확대하지 않는다.
- q95와 q99은 정상 support에서 얻은 상대적 기준이다. 이 값은 target shift의 FPR, 결함 TPR, calibration transfer를 자동으로 보장하지 않는다.
- 본 논문은 두 배치 조건을 단일 잠재 상태로 환원하는 설명을 사전 등록 후 계측으로 검정해 **기각했다**(§4.6). 따라서 §4의 증거는 "동일 개입이 두 조건에서 큰 이득을 낸다"는 관측 수준의 주장을 지지하며, 그 이상의 잠재 변수 주장은 하지 않는다.

---

## 3. Method: Support-Calibrated Residual Fusion

### 3.1. Problem Formulation

- **입력**: $N$-shot 골든 정상 서포트 집합 $\mathcal{S}_N = \{x_1^s, \dots, x_N^s\} \subset \mathcal{M}_{normal}$.
- **현실 제약 (Coverage Void)**: 서포트는 물리적 정상 다양체 $\mathcal{M}_{normal}$의 국소 영역만 포괄하며, 테스트 쿼리 $x^q$는 서포트가 관측하지 못한 영역에 위치할 수 있다. §1.2에 따라 이 상태는 support 예산이 작아서(조건 A) 또는 query 획득 조건이 이동해서(조건 B) 발생한다.
- **학습 제약**: 백본 동결(`requires_grad=False`), 추가 학습 파라미터 0개($\text{Params}=0$).
- **출력**: 쿼리 $x^q$에 대한 연속 이상 맵 $A(x^q) \in \mathbb{R}^{H \times W}$.
- **Support fitting**: 정상 support images만 사용해 feature bank, leave-source-out q95 scales, shallow q99 gate references를 계산한다. Query images, defect labels, evaluation masks는 적합이나 score 구성에 사용하지 않는다.
- **Prediction contract**: 각 query는 frozen backbone을 통과해 고정 support bank와 비교되고, component별로 정규화·게이팅된 뒤 native space로 투영되어 공통 streaming evaluator로 평가된다.

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
[Tail-calibrated 정규화]            [Tail-calibrated 정규화]
(component별 LOO q95 나눗셈)         (component별 LOO q95 나눗셈)
       │                                   │
       └─────────────────┬─────────────────┘
                         ▼
        [Deep 기본 점수와 Shallow 초과 반응의 선택적 결합]
        (q95-scaled deep base + q99 초과 shallow residual)
                         │
                         ▼
        [Native-space projection 및 고정 CQ 후처리]
                         │
                         ▼
               최종 native-resolution anomaly map
```

여기서 **비대칭 residual spike gating**은 하나의 연산이 수행하는 네 가지 특성을 뜻한다.

- **비대칭**: deep score와 shallow score를 같은 방식으로 평균하지 않는다. Deep score는 항상 기본 신호로 유지하고, shallow score는 조건을 만족할 때만 추가한다.
- **Residual**: shallow score 전체가 아니라 정상 기준을 초과한 부분만 사용한다.
- **Spike**: 정상 support에서 관찰되는 일반적인 shallow 반응보다 큰 상위-tail 반응을 뜻한다.
- **Gating**: shallow 반응이 정상 기준 안에 있으면 0으로 만들고, 기준을 초과한 부분만 통과시키는 선택 연산이다.

한 위치 $p$에서의 개념적 계산은 다음과 같다.

$$
r(p)=\max\bigl(0,z_5(p)-\tau_{5,q99}\bigr),
\qquad
S(p)=\operatorname{mean}\bigl(z_{11}(p),z_{23}(p)\bigr)+r(p).
$$

Shallow score가 정상 q99보다 낮으면 $r(p)=0$이므로 deep base만 남고, q99를 초과하면 초과한 크기만 더해진다. 따라서 이 연산은 "shallow feature를 추가하는 방법"이라기보다 **"shallow feature 중 정상 범위를 벗어난 반응만 보완 신호로 허용하는 방법"**이다. §1.2의 용어로는, **shallow가 관측되지 않은 정상 변동에 반응해 생성한 tail을 차단하고 그 너머만 통과시키는 연산**이다.

#### 입력 이미지에서 최종 anomaly map까지

1. **입력과 전처리**: query 이미지를 support 이미지와 동일하게 전처리하고 원본 크기를 기록한다. method에 전달되는 query에는 label이나 mask를 제공하지 않는다.
2. **다중 해상도 feature 추출**: frozen DINOv2-L이 448, 672, 896, 1120 해상도에서 이미지를 처리한다. 각 해상도에서 `[5, 11, 23]` layer feature를 추출해 L2-normalized token matrix로 변환한다.
3. **정상 reference와 비교**: 각 query token을 해당 resolution과 bank의 고정 support bank와 비교해 1-nearest-neighbor distance를 계산한다.
4. **component 보정**: 각 layer·resolution·bank의 distance map을 support에서 계산한 q95 scale로 나눈다.
5. **layer 결합**: layer 11과 23의 score로 deep base를 만들고, layer 5의 response가 support q99를 초과한 양수 부분만 더한다.
6. **bank와 resolution 결합**: 각 score map을 token grid에서 native space로 bilinear projection한 뒤, 한 resolution 안에서 네 bank를 평균하고 다시 네 resolution을 평균한다.
7. **고정 후처리와 출력**: label 없이 공통 component-quantile 후처리를 적용한다. 최종 native-resolution map을 streaming evaluator로 전달하고, map의 maximum을 image-level diagnostic score로 기록한다.

Query 평가 중에는 backbone, support bank, scale, gate threshold를 갱신하지 않는다.

---

### 3.3. Asymmetric Residual Spike Gating

- **목적**: shallow layer는 작은 결함의 경계와 texture 변화를 포착하지만 정상적인 조명·반사·센서 변화에도 민감하다. Support가 좁아지거나 query 획득 조건이 이동할수록 이 민감도는 결함이 아닌 정상 변동에 반응한다. 따라서 shallow score 전체를 더하면 그 정상 반응까지 anomaly evidence로 누적된다. 본 연산은 shallow를 항상 켜 두지 않고, 정상 support에서 보기 드문 shallow 반응만 보완 신호로 사용한다.
- **입력**: 각 native-space 위치 $p$에서 q95로 scaling된 shallow score $z_5(p)$와 deep base score $z_d(p)$. Deep base는 layer 11과 23 score의 평균이다.
- **support 기준**: support normal feature bank에서 계산한 shallow normalized score의 q99를 $\tau_{5,q99}$로 둔다. Query label이나 defect mask로 조정하지 않는다.
- **개념식**:

  $$
  z_l(p)=\frac{d_l(p)}{q_{95,l}^{\mathrm{normal}}}, \qquad
  z_d(p)=\operatorname{mean}\bigl(z_{11}(p),z_{23}(p)\bigr),
  $$

  $$
  S(p)=z_d(p)+\operatorname{ReLU}\bigl(z_5(p)-\tau_{5,q99}\bigr).
  $$
- **실제 aggregation**: 위 식은 하나의 layer·resolution·bank component를 설명하는 축약식이다. 실제로는 각 component를 계산하고 native space로 투영한 뒤 bank와 resolution에 대해 고정된 평균 규칙을 적용한다.
- **해석의 한계**: 이 연산은 defect-only gate나 attention-level 정보 복원을 보장하지 않는다. Gate가 normal response를 줄일 수도 defect response를 줄일 수도 있으므로, response decomposition과 fixed-threshold metrics를 함께 검증한다(§4.6, §4.7).

---

### 3.4. Tail-Calibrated Quantile Normalization

- **문제**: 서로 다른 layer와 resolution의 nearest-neighbor distance는 같은 단위를 갖더라도 정상 분포의 spread와 tail 크기가 다르다. 단순 평균하면 score가 큰 component가 결함 관련성이 아니라 scale 차이 때문에 fusion을 지배한다.
- **support-only 계산**: component $m$에 대해 support image 자체를 자기 이웃으로 사용하지 않도록 leave-source-out 방식으로 정상 distance를 모으고, 그 분포의 q95를 $q_{95}^{(m)}$로 저장한다.
- **정규화**:

  $$
  \widetilde{d}_m(p)=\frac{d_m(p)}{q_{95}^{(m)}+\epsilon}.
  $$

  $\epsilon$은 degenerate scale을 방지하기 위한 수치 안정화 항이며 실제 설정값은 configuration에 기록한다.
- **대안 대비 선택 근거**: 동일 조건에서 raw distance, MAD scaling, empirical rank transform과 비교했을 때 tail q95 division이 가장 높은 AUPRO와 pAUROC를 보였다. Rank transform은 절대 거리 기하를 파괴해 pAUROC를 떨어뜨렸고, MAD는 극단 정상 상위 tail에 덜 반응했다.
- **q95와 operating threshold의 구분**: q95는 component score의 scale을 맞추는 reference이며 source FPR 1% threshold나 shifted target FPR을 보장하지 않는다. Source-fixed operating threshold는 별도 calibration pool과 evaluator 단계에서 계산한다(§4.7).
- **Support 크기에 따른 기준 드리프트 (측정된 한계)**: leave-source-out 기준은 support 개수에 따라 체계적으로 이동한다. §4.6의 계측에서 정상 query의 q99 게이트 발화율은 16-shot에서 설계값 1%와 일치했으나(0.994배), support가 줄면 8/4/2-shot에서 0.95 / 0.74 / 0.64배로 낮아졌다. 원인은 추정 구조에 있다. Support가 적을수록 각 support image의 leave-source-out 이웃이 희소해 LOO 거리가 커지고, 그만큼 q95·q99 기준이 크게 잡혀 query score가 상대적으로 작아 보인다. 즉 **기준은 고샷에서 낙관적이고 저샷에서 보수적이다.** 이 드리프트는 본 방법의 성능을 해치지 않았으며(§4.3.1에서 2→16-shot 변화 0.44pp), 오히려 저샷에서 게이트를 보수적으로 만들어 자기 보정처럼 작동했을 수 있다. 다만 게이트의 동작점이 support 크기에 대해 불변이 아니라는 사실은 명시적 한계다.
- **1-shot에서의 추정량 교체**: support가 1장이면 leave-source-out이 정의되지 않으므로, 구현은 같은 이미지 안의 leave-token-out으로 대체한다. 한 이미지 내 인접 토큰은 이미지 간보다 훨씬 유사하므로 기준이 작게 잡히고 게이트가 과도하게 발화한다(§4.6 기준 3.25배). 1-shot 결과를 다른 shot 수준과 같은 척도로 해석하지 않는다.

#### 3.4.1. Fixed Feature, Resolution, and Bank Aggregation

- **Feature 추출**: frozen DINOv2-L의 `[5, 11, 23]` layer를 `[448, 672, 896, 1120]` 네 해상도에서 사용한다. 이 피라미드는 단일 고해상도로 대체할 수 없다. 기저 해상도를 제거하면 texture 클래스가 크게 무너지고, 고해상도만 사용하면 클래스별 반응이 비단조적으로 갈린다.
- **Support bank**: support feature를 네 개의 고정 bank로 나누어 저장한다. 각 bank는 query 예측 전에 구성되며 비교 대상 arm 전체가 동일한 bank를 공유한다. Bagged sparse random bank가 exhaustive bank나 coreset 기반 선택보다 안정적이었다.
- **Distance 계산**: 각 query feature 위치에서 해당 support bank까지의 1-nearest-neighbor distance를 계산한다. query image나 query label은 bank에 삽입하지 않는다.
- **Projection**: 각 resolution의 token score map을 bilinear로 native image space에 투영하고, resolution 내부에서 네 bank를 평균한 뒤 네 resolution에 대해 다시 평균한다.
- **비교 통제**: §4.2의 모든 arm은 동일한 support manifest, bank 구성, backbone 호출, resolution 집합, query 순서, evaluator를 사용한다. 따라서 비교 차이는 data pipeline이 아니라 normalization과 shallow gating에서만 발생한다.

---

### 3.5. Score Projection and Component Quantile Postprocessing (CQ)

- **역할**: score map을 evaluator의 native spatial contract에 맞춘 뒤 `component_quantile(percentile=99, q=1.0, dilate_size=0)` 후처리를 적용한다. query score map의 99번째 백분위로 high-score mask를 만들고 8-connectivity component를 찾은 뒤, 각 component의 pixel 값을 해당 component의 q-quantile(현재 maximum)로 치환한다. 이 단계에서 GT나 query label은 사용하지 않는다.
- **동기**: AUPRO는 각 GT 연결 성분을 자기 면적으로 정규화하므로 부분적으로만 검출된 성분에 큰 불이익을 준다. CQ는 이 불일치를 완화한다.
- **왜 별도 기여로 주장하지 않는가**: §4.2 ablation에서 CQ는 모든 arm에 공통으로 고정되어 있다. 따라서 arm 간 차이를 CQ의 독립 효과로 해석할 수 없다. 독립 효과를 주장하려면 동일 조건에서 CQ-off/on을 별도로 비교해야 한다.
- **region recovery와의 구분**: 별도로 평가한 형태학적 recovery 후보는 AUPRO를 개선하지 않았다(-0.051pp, floor 미만). 따라서 연결 성분 diffusion이나 고립 오탐 제거를 확인된 독립 기여로 서술하지 않는다.

#### 3.5.1. Score Map에서 보고 anomaly score까지

- 최종 출력은 native-resolution pixel anomaly map이다. evaluator는 이 map과 GT로 segmentation metrics를 계산하지만, method는 GT를 사용하지 않는다.
- image-level diagnostic score는 최종 map의 maximum으로 기록된다.
- `component_quantile`은 component 내부의 상대적 score shape를 보정할 뿐 새로운 query feature나 support 정보를 추가하지 않는다.

---

### 3.6. Anomaly Scoring and Computational Efficiency

- **Parameter and computation accounting**: 추가 학습 파라미터와 역전파를 사용하지 않는다. Runtime과 throughput은 frozen backbone inference, distance computation, projection 및 후처리를 포함한 별도 benchmark로 보고하며, 측정되지 않은 FPS 주장은 하지 않는다.
- **한 번 수행하는 support 단계**:
  1. 정상 support image를 frozen backbone으로 통과시켜 layer·resolution별 feature를 추출한다.
  2. 각 bank의 support feature를 고정하고 leave-source-out normal distance에서 q95 scale을 계산한다.
  3. normalized shallow support response에서 q99 gate reference를 계산한다.
  4. 이후 bank, q95, q99는 query 평가 동안 변경하지 않는다.
- **각 query에 대해 수행하는 단계**: 동일 backbone·resolution 처리 → component별 nearest-neighbor distance → q95 scaling → deep base + shallow q99 residual → bank·resolution 결합 → CQ/evaluator contract 적용.
- 이 순서는 support fitting과 query evaluation을 분리해 query label이나 GT가 method fitting에 유입되지 않도록 한다. §4.7의 source calibration audit에서 사용하는 별도 normal calibration images는 operating threshold 계산에만 사용하며 feature fitting이나 prediction state를 갱신하지 않는다.

---

## 4. Evidence and Analysis

### 4.1. Experimental Setup and Reading Rules

- **Evidence layers**: 이 장은 (i) 16-shot component ablation, (ii) support 예산 축을 따른 component ablation, (iii) support 고정 acquisition shift 실험, (iv) 합성 corruption stress test, (v) 기각된 잠재 상태 가설의 계측 기록, (vi) source-fixed operating audit를 구분한다. **서로 다른 query population을 사용하는 표 사이의 절대 수치를 직접 순위 비교하지 않는다.**
- **MVTec AD 2 controlled setting**: `test_public_seen_development`, fixed support manifest, frozen DINOv2-L, 8 objects. Query label과 mask는 evaluator에서만 사용하며 sealed final holdout이 아니다.
- **Metrics**: primary는 `seg_AUPRO@0.05`, guardrail은 `seg_AUROC@0.05` (pAUROC), diagnostic은 `seg_F1_oracle`이다. Oracle F1은 test label로 threshold를 선택하므로 source-fixed alarm transfer와 동등하게 해석하지 않는다.
- **판정 규칙 (§2.3에서 사전 고정)**: **0.865pp 미만 AUPRO 차이는 개선 근거로 사용하지 않고 방향성으로만 기술한다.** 주 구성은 5개 support-split seed에서 59.478 ± 0.600으로 반복 측정되었다.
- **표기 규칙**: 개발 과정의 내부 실행 식별자를 사용하지 않고, 모든 비교를 제안 방법 기준의 ablation 표기로 제시한다.

---

### 4.2. Component Ablation: What the Fusion Rule Changes

Backbone, support manifest, 네 해상도, 네 bank, CQ, evaluator, query population을 모두 고정하고 **scaling 방식과 shallow 기여 방식만** 바꾸었다. 구조는 (tail q95 vs MAD) × (gated residual vs ungated)에 deep-only를 더한 형태다.

| Configuration | Trainable Params | AUPRO@0.05 (%) | pAUROC@0.05 (%) | F1-oracle (%) | Δ AUPRO |
| :-- | :--: | --: | --: | --: | --: |
| **Ours** (tail-calibrated scaling + gated shallow residual) | **0** | **59.15** | 77.11 | 35.87 | — |
| w/o residual gating (shallow 전체 합산) | 0 | 54.02 | 74.04 | 32.90 | **−5.13** |
| w/o tail calibration (MAD scaling으로 대체) | 0 | 57.48 | 76.16 | 33.89 | **−1.66** |
| w/o both | 0 | 51.16 | 73.02 | 29.98 | −7.99 |
| w/o shallow branch (deep-only) | 0 | 58.67 | **77.35** | **37.11** | −0.47 |

- **Residual gating**: tail scaling을 고정했을 때 shallow 응답을 게이팅하면 **+5.13pp**(8클래스 중 6승). 보정되지 않은 shallow의 무조건 합산은 anomaly map을 크게 오염시킨다. floor의 5.9배.
- **Tail calibration**: gating을 고정했을 때 tail q95 scaling이 MAD scaling보다 **+1.66pp**(8/8 전승). floor의 1.9배.
- **가산성**: 두 요소를 모두 제거하면 −7.99pp로 개별 효과의 합(−6.79pp)과 같은 자릿수다. 두 구성 요소는 대체로 독립적으로 기여한다.
- **Shallow 가지의 순이득**: deep-only 대비 **+0.47pp이며 pAUROC는 −0.24pp**다. **floor 미만이므로 clean 조건에서 shallow 가지의 순이득은 검증되지 않았다.** 즉 이 표가 지지하는 핵심은 shallow 응답의 추가가 아니라 **보정되지 않은 shallow 응답을 support-tail residual로 제한하는 규칙**이다.

클래스별로 보면 gating 이득은 texture·입자형 클래스(fabric +15.35, rice +11.44, sheet_metal +9.70)에 집중되고, 구조적 소형 객체(can, wallplugs)에서는 오히려 감소한다. 이는 shallow가 만든 정상 tail의 크기가 클래스 표면 특성에 의존함을 시사한다.

---

### 4.3. Condition A — Behavior as the Support Budget Shrinks

조건 A를 단독으로 가동한다. Query 분포를 `regular`로 고정하고 support 예산만 1장까지 줄인다. Support manifest는 **엄격히 중첩**되어 있어(N-shot 집합이 2N-shot 집합의 부분집합) shot 효과와 manifest 추첨 운이 교란되지 않는다.

#### 4.3.1. Component Ablation across Support Budgets

Backbone, bank 구성, 네 해상도, CQ, evaluator, query population을 고정하고 shot 수와 shallow 기여 방식만 바꾼 120-cell 실험이다. 8 objects macro, AUPRO@0.05 (%).

| Configuration | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot | 2→16 변화 |
| :-- | --: | --: | --: | --: | --: | --: |
| **Ours** (tail-calibrated + gated residual) | 55.94 | **60.82** | 60.42 | 60.22 | **61.26** | **0.44** |
| w/o residual gating (shallow 전체 합산) | 48.01 | 49.34 | 52.92 | 53.56 | 54.75 | **5.41** |
| w/o shallow branch (deep-only) | 56.09 | 60.65 | 60.22 | 59.50 | 60.83 | 0.61 |
| **Gating 이득** (Ours − ungated) | **7.93** | **11.49** | **7.50** | **6.66** | **6.51** | |
| Shallow 가지 순이득 (Ours − deep-only) | −0.15 | 0.17 | 0.19 | 0.72 | 0.43 | |

- **P1 지지 — gating 이득은 저샷에서 사라지지 않고 커진다.** 다섯 shot 수준 전부에서 이득이 측정 floor(0.865pp)의 **7.5~13배**다. 16-shot의 6.51pp가 2-shot에서 11.49pp로 커진다. 따라서 게이팅은 support가 충분할 때의 미세 조정이 아니라 **support가 부족할수록 더 필요한 연산**이다.
- **P3 지지 — 규칙이 support 의존성을 제거한다.** 2→16-shot 구간에서 제안 규칙의 변화는 **0.44pp로 측정 floor 미만**, 즉 통계적으로 평탄하다. 같은 구간에서 보정 없이 shallow를 합산하는 구성은 **5.41pp** 변한다. 보정되지 않은 shallow 기여가 방법을 support-hungry하게 만드는 주범이며, gating이 그 의존성을 걷어낸다.
- **내부 통제된 support efficiency**: **제안 규칙은 정상 이미지 2장으로 60.82를 얻어, 보정 없는 구성이 16장으로 얻는 54.75를 +6.07pp 앞선다.** 동일 backbone, 동일 bank 구성, 동일 evaluator에서 fusion 규칙만 다르므로, 이 8배 격차는 데이터 예산이 아니라 score 구성 방식에 귀속된다.
- **Shallow 가지 자체는 여전히 기여하지 않는다.** deep-only 대비 순이득이 −0.15 ~ +0.72pp로 **다섯 shot 수준 전부에서 floor 미만**이다. §4.2의 16-shot 판독이 shot 축 전체에서 재확인된다. 본 방법의 기여는 shallow evidence의 추가가 아니라 **그 정상 tail의 제한**이다.

> **1-shot 셀에 대한 주석 (필수)**: support가 1장이면 leave-source-out 보정이 정의되지 않아 구현은 **같은 이미지 내 leave-token-out**으로 대체한다. 즉 1-shot 열만 보정 기준의 추정량이 다르다. 1-shot에서 관찰되는 성능 하락(2-shot 대비 −4.88pp)은 support 부족과 이 추정량 교체가 교란되어 있어 분리되지 않는다. 본 논문은 1-shot을 **동작 기록으로만** 인용하고 추세 주장의 근거로 쓰지 않는다.

#### 4.3.2. External Baseline Context across Support Budgets

위 통제 실험과 별개로, 동일 split과 protocol에서 독립 환경으로 실행한 외부 baseline 비교다. MVTec AD 2, AUPRO@0.05 (%).

| Shots | RegAD | RD4AD | PatchCore | FastRecon | RN-FMLK | SuperAD (v2) | SuperAD (v3) | **Ours** | Δ vs best baseline |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 1 | 15.56 | 36.89 | 38.07 | 35.11 | 39.25 | 47.47 | 48.18 | **52.49** | **+4.31** |
| 2 | 14.50 | 38.02 | 36.77 | 34.16 | 38.56 | 48.25 | 48.87 | **53.31** | **+4.44** |
| 4 | 12.68 | 40.07 | 42.10 | 38.04 | 41.39 | 51.23 | 50.69 | **56.14** | **+4.91** |
| 8 | 17.26 | 38.33 | 48.17 | 47.37 | 43.11 | 55.83 | 54.99 | **60.04** | **+4.21** |
| 16 | 15.93 | 32.23 | 48.83 | 47.05 | 42.51 | 53.49 | 54.87 | **59.48** | **+4.61** |

- 최고 baseline 대비 격차는 +4.21 ~ +4.91pp이며 범위가 0.70pp로 **측정 noise floor 이내**다. 즉 shot 수에 따른 체계적 추세가 없다.
- 제안 방법의 2-shot 성능(53.31)은 비교 방법의 16-shot 성능(53.49)과 **측정 오차 내에서 동등**하다(−0.18pp, floor 미만). **우위는 아니다.** 우위를 주장할 수 있는 것은 §4.3.1의 내부 통제 비교뿐이다.
- 저장된 R1 실행에서 support 수에 따른 동작은 다른 데이터셋에서도 단조 증가했다(MVTec AD DINOv2-L 78.39→85.36, VisA DINOv2-L 72.59→84.82, 1→16-shot AUPRO). 이들에는 통제된 ablation이 없으므로 dataset generalization의 확정 근거가 아니라 동작 기록으로 읽는다.
- **판독 제한**: 4.3.1과 4.3.2는 query population과 실행 identity가 다르다(전자는 조건별 층, 후자는 전체 query). 두 표의 절대 수치를 교차 비교하지 않는다.

---

### 4.4. Condition B — Behavior as the Acquisition Condition Changes

조건 B를 단독으로 가동한다. Support manifest, backbone, 네 해상도, seed, evaluator를 고정하고 **query의 획득 조건만** 바꾼다. 각 조건은 독립적으로 평가하며 target adaptation은 수행하지 않는다.

| Configuration | Regular | Shift | Δ (pp) | Shift 절대 순위 |
| :-- | --: | --: | --: | :--: |
| **Ours** (tail-calibrated + gated) | **61.26** | **59.40** | −1.86 | **1** |
| w/o shallow branch (deep-only) | 60.83 | 58.77 | −2.06 | 2 |
| w/o tail calibration (MAD + gated) | 60.10 | 57.27 | −2.83 | 3 |
| w/o residual gating (tail + ungated) | 54.75 | 54.79 | **+0.04** | 4 |
| w/o both | 52.40 | 51.50 | −0.90 | 5 |

- **P2 충족**: shift 하 절대 AUPRO에서 제안 규칙이 1위이며, ungated 대비 **+4.61pp**, MAD scaling 대비 +2.13pp, deep-only 대비 +0.63pp다.
- **개입 효과의 조건 간 안정성**: 동일 개입이 서로 다른 조건에서 같은 부호·자릿수의 이득을 낸다.

  | 개입 | development | regular | shift | 범위 |
  | :-- | --: | --: | --: | :-- |
  | Residual gating | +5.13 | +6.51 | +4.61 | **4.6 ~ 6.5** |
  | Tail calibration | +1.66 | +1.16 | +2.13 | **1.2 ~ 2.1** |
  | Shallow 가지 순이득 | +0.47 | +0.43 | +0.63 | 0.4 ~ 0.6 (전부 floor 미만) |

  두 주효과 모두 모든 조건에서 floor를 상회하며 크기가 안정적이다.

#### 4.4.1. The Two Conditions Interact on the Same Component

조건 A와 조건 B를 교차시키면, gating 이득이 두 축 모두에서 유지되며 저샷에서 커지는 동일한 패턴이 나타난다. `shift_1`, 8 objects macro, AUPRO@0.05 (%).

| Configuration | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
| :-- | --: | --: | --: | --: | --: |
| **Ours** | 54.35 | **58.62** | 57.73 | 57.71 | **59.40** |
| w/o residual gating | 47.62 | 48.61 | 52.63 | 53.11 | 54.79 |
| w/o shallow branch | 54.66 | 58.60 | 57.54 | 56.89 | 58.77 |
| **Gating 이득** | **6.73** | **10.02** | **5.09** | **4.60** | **4.61** |
| Shallow 가지 순이득 | −0.32 | 0.02 | 0.19 | 0.81 | 0.63 |

- Shift 조건에서도 gating 이득은 저샷에서 커진다(16-shot 4.61 → 2-shot 10.02pp). 조건 B가 가해진 상태에서도 조건 A에 대한 방어가 유지된다.
- **두 조건은 서로 다른 외생 원인이지만 같은 성분 위에서 작동한다.** 이것이 §1.2가 주장하는 범위이며, 두 조건을 단일 잠재 변수로 환원하는 더 강한 주장은 §4.6에서 기각된다.
- Shallow 가지 순이득은 shift에서도 전 구간 floor 미만이다.
- **두 구성 요소의 역할 분리 (정직한 판독)**: Tail calibration은 저하폭을 줄인다(−1.86 대 −2.83). 그러나 **gating은 저하폭을 줄이지 않는다.** Ungated 구성이 오히려 가장 평탄하다(+0.04). 따라서 본 논문의 주장은 **"shift 하 최고 절대 성능"**이며 **"최소 저하"가 아니다.** 저하폭 차분 0.97pp는 단일 측정의 차분이므로 방향성으로만 기술한다.
- **Shallow 가지의 조건 의존성 (경향)**: shallow 순이득이 clean +0.43에서 shift +0.63으로 증가한다. 그러나 **두 값 모두 floor 미만이므로 경향으로만 기술하며**, 이 방향성을 기여 근거로 사용하지 않는다.

#### 4.4.2. External Real Acquisition Shift

공식 MVTec AD 2 evaluation server의 private → private-mixed 비교다. 이 shift는 미공지 실제 획득 변화이며 참가자가 사전에 알 수 없다.

| Method | Backbone | Normal Budget | priv Seg-F1 | mixed Seg-F1 | 절대 저하 (pp) | 상대 저하 |
| :-- | :-- | :--: | --: | --: | --: | --: |
| SuperAD-16 | DINOv2-L/14 | P16 | 47.18 | 42.51 | −4.67 | −9.90% |
| RoBiS | DINOv2-B/14-reg | Pfull | 51.00 | 46.52 | −4.48 | −8.78% |
| SuperADD | DINOv3-H+/16 | Pfull | 57.42 | 54.35 | −3.07 | −5.35% |
| **Ours** | DINOv3-H+/16 | **P16** | 45.94 | 44.05 | **−1.89** | **−4.11%** |

- 가장 적은 정상 데이터 예산(P16)으로 **가장 작은 상대 저하**를 보였다. AU-PRO는 54.40→53.14로 97.68% 유지되었고 Class-F1은 저하하지 않았다(+0.0069).
- **판독 제한**: 이 표는 backbone과 normal budget이 서로 다르므로 **benchmark evidence이며 특정 구성 요소에 대한 attribution이 아니다.** 구성 요소 귀속은 §4.4의 통제된 비교가 담당한다.

---

### 4.5. A Failure Mode Outside the Claimed Range

MVTec AD의 clean `train/good` support를 고정하고 test의 정상·결함 이미지만 ICCV 2023 corruption protocol의 severity-3으로 변경했다. GT mask는 원본과 동일하며 target corruption으로 support bank나 threshold를 다시 적합하지 않았다.

| Target test condition | pAUROC@0.05 (%) | AUPRO@0.05 (%) | Δ AUPRO from clean (pp) |
| :-- | --: | --: | --: |
| Clean | 88.92 | 85.36 | +0.00 |
| Brightness | 88.26 | 84.42 | −0.94 |
| Contrast | 87.44 | 83.15 | −2.21 |
| Defocus blur | 86.71 | 81.92 | −3.44 |
| Gaussian noise | 78.46 | 64.67 | **−20.69** |

- Photometric 변화(밝기·대비)와 blur는 견디지만 **Gaussian noise에서 붕괴한다.**
- **해석 가설**: support q95/q99는 *정상 feature 분포의 tail*을 기준으로 삼는다. Photometric shift는 foundation feature가 상당 부분 흡수하므로 기준의 상대적 의미가 보존되는 반면, high-frequency sensor noise는 query token 자체를 support 분포 밖으로 밀어내 기준의 의미를 훼손할 수 있다.
- **이 가설은 검증되지 않았다 (명시)**: (i) 기준 훼손을 보이는 직접 계측이 없고, (ii) 이 표에는 제안 방법만 있어 **동일 corruption에서 deep-only나 ungated 구성도 같은 크기로 붕괴하는지 확인되지 않았다.** 후자가 사실이라면 이 붕괴는 본 방법 특유의 현상이 아니라 nearest-neighbor 계열의 일반적 실패다. 두 확인이 끝나기 전까지 본 논문은 이 결과를 해석 없이 **명시적 실패 모드**로만 보고하고, 주장 범위를 photometric·acquisition shift로 한정한다. §4.6의 계측 인프라로 arm 간 비교를 수행하는 것이 직접적인 후속 작업이다.
- 이 표의 F1은 각 target split의 oracle F1이므로 fixed-threshold alarm transfer를 측정하지 않는다.

---

### 4.6. A Rejected Hypothesis: Instrumenting a Shared Latent State

§1.2는 두 배치 조건이 같은 score 성분 위에서 작동한다고만 주장한다. 우리는 그보다 강한 설명을 시도했다. 즉 두 조건이 **하나의 잠재 상태**로 수렴하며 그 상태가 성능 저하를 매개한다는 가설이다. 이 절은 그 가설을 계측으로 검정하고 **기각한 기록**이다. 방법의 유효성과는 독립이며, 어떤 설명이 데이터의 지지를 받는지에 대한 경계를 정한다.

#### 4.6.1. Hypothesis, Measurement, and Pre-Registration

가설의 잠재 상태는 "정상 query가 support가 관측한 정상 변동 범위 밖에 놓인 정도"(coverage void)다. 이 양은 직접 관측되지 않지만, 방법이 support quantile로 보정되어 있으므로 **사전 null이 정해진다.** Support와 동일 분포의 정상 query라면 q99 게이트는 정상 pixel의 1%에서 발화해야 한다. 따라서 정상 pixel 게이트 발화율 $V_1$과 그 null 대비 비율 $V_5=V_1/0.01$을 상태의 대리 측정량으로 둔다. 보조로 초과 질량 $V_2=\mathbb{E}[\mathrm{ReLU}(z_5-\tau_{q99})]$, 게이트 선택도 $V_4$(결함/정상 발화율 비)를 기록한다. Mask는 정상·결함 pixel 층화에만, 예측 완료 이후에 사용한다.

설계는 MVTec AD 2 8 objects, seed 0, frozen DINOv2-L, 중첩 manifest 1/2/4/8/16-shot, 획득 조건 6종, arm 3종의 **120 cells**이며, 판정 규칙(H1~H5)은 수집 전에 등록했다(`configs/benchmarks/void_instrumentation/PROTOCOL_20260922.md`). 결과는 120/120 완료, 실패 0건이다.

#### 4.6.2. What the Instrumentation Confirmed

- **보정의 사전 null이 실증되었다.** 16-shot `regular`에서 $V_5 = 0.994$로 예측값 1.000과 일치한다. q99 게이트는 in-distribution 정상 query에서 설계된 비율로 정확히 발화한다. 이는 §3.3–3.4의 보정이 의도대로 동작함을 뒷받침하는 독립 증거다.
- **획득 조건은 예측대로 이 양을 움직인다(H2 PASS).** 16-shot에서 $V_5$는 `underexposed` 0.81 < `regular` 1.00 < `overexposed` 1.25 < `shift_2` 1.23, `shift_3` 1.29, `shift_1` 1.34으로, 노출 변화보다 acquisition shift에서 더 크다.
- **게이트 선택도는 $V_5$와 함께 떨어진다(H3 PASS).** $\log V_5$에 대한 $V_4$의 기울기는 −3.71 (n=30)이다.

#### 4.6.3. Why the Hypothesis Is Rejected

| | 등록된 가설 | 판정 |
| :-- | :-- | :-- |
| H1 | support 예산이 잠재 상태를 움직인다 (shot↓ → $V_5$↑, 비 ≥ 2.0) | **FAIL** |
| H2 | 획득 조건이 잠재 상태를 움직인다 | PASS |
| H3 | $V_5$↑ → 선택도↓ | PASS |
| H4 | 두 축이 만든 상태가 같은 곡선 위에 놓인다 | PASS (아래 단서 참조) |
| H5 | 개입 효과가 잠재 상태를 따라간다 | **FAIL** |

- **H1 FAIL — 방향이 반대다.** `regular`에서 $V_5$는 2/4/8/16-shot에 대해 0.64 / 0.74 / 0.95 / 1.00으로, support가 **늘수록** 커진다. 기전은 해석 가능하다. Leave-source-out 기준은 저샷에서 희소한 bank로 추정되어 크게 잡히고, 그 결과 query score가 상대적으로 작아 보인다. 즉 $V_5$는 coverage void가 아니라 **support-derived 기준의 shot-의존 낙관 편향**을 측정하고 있다(§3.4 참조). 1-shot 셀은 추정량 자체가 다르므로(§4.3.1 주석) 이 추세에서 제외했다.
- **H5 FAIL.** gating 이득은 $\log V_5$에 대해 기울기 −0.92로 **감소**한다. 등록된 규칙에 따라 gating은 잠재 상태의 해법이 아니라 경험적 이득으로 기술한다.
- **H4의 PASS는 결론을 지지하지 못한다.** 두 적합선의 최대 간격은 0.476pp로 floor 미만이지만, 각 축 내부 적합이 애초에 무의미하다. 획득 조건 축에서 $\log V_5$에 대한 AUPRO 손실의 결정계수는 $R^2=0.097$(기울기 +2.34), support 예산 축에서는 $R^2=0.016$(기울기 −0.27)이며 **부호가 서로 반대다.** 즉 $V_5$는 두 축 어디에서도 성능 손실을 설명하지 못한다. 거의 평평하고 잡음뿐인 두 직선은 좁은 중첩 구간에서 당연히 일치한다.
- **등록 규칙의 결함을 함께 보고한다.** H4는 "두 적합선이 일치할 것"만 요구하고 **"각 적합선이 유의미할 것"을 전제하지 않았다.** 이 누락 때문에 H4는 형식상 통과하면서도 판정 능력을 갖지 못했다. 유사한 매개 검정을 설계할 때에는 각 축의 설명력 하한을 선행 조건으로 등록해야 한다.

**결론**: 두 배치 조건을 단일 잠재 상태로 환원하는 설명은 기각한다. 본 논문은 §1.2의 성분 수준 명제, 즉 두 조건이 **보정되지 않은 shallow 기여**라는 관측 가능한 성분 위에서 작동한다는 주장만 유지하며, 이는 §4.3–4.4에서 지지된다.

---

### 4.7. Source-Fixed Operating-Point Audit

이 audit은 support와 분리된 normal calibration 16장으로 source nominal FPR 1% threshold를 고정한 뒤, regular와 shift 조건의 normal FPR, defect TPR, PRO를 비교한다. Threshold는 target query나 target label로 조정하지 않으며, layer-wise raw/normalized response, deep/shallow/residual contribution, gate activation, paired scene summary를 함께 기록한다.

- 현재 32 cell 중 28 cell만 완료되어 8-class macro를 등록하지 않는다. 완료된 7-class 부분 결과에서 gated 대 ungated의 PRO 차이는 +7.10pp였으나 FPR 증가가 +0.79pp 더 컸고, FPR no-worse 기준은 7개 중 4개 클래스만 만족했다. 이 부분 신호는 robustness 주장의 근거가 아니라 잔여 클래스 복구 후 확인할 tradeoff다.
- Support-derived q95/q99은 component scale과 gate를 정의하지만 target FPR을 보장하지 않는다. Gate activation만으로 nuisance removal이나 defect preservation을 주장하지 않으며, fixed-source FPR·TPR·PRO와 paired response가 같은 방향을 보일 때만 해석한다.
- **따라서 §4.4의 shift 결과는 ranking·segmentation 근거이며 fixed-threshold alarm transfer 근거가 아니다.** 두 주장을 구분해 기술한다.

---

### 4.8. Evidence Boundary

- **§4.3.1이 본 논문의 가장 강한 evidence다.** 120 cells, 실패 0건의 통제 실험에서 support 예산과 shallow 기여 방식만 바꾸었고, gating 이득이 다섯 shot 수준 전부에서 측정 floor의 7.5~13배로 나타났으며, 규칙이 support 의존성을 제거함을 2→16-shot 변화 0.44pp 대 5.41pp로 정량화했다.
- §4.2는 16-shot에서 두 구성 요소를 분리하며 5-seed에서 재현되었다. §4.4는 획득 조건 축에서 같은 패턴을 확인하고, §4.4.1은 두 조건이 교차해도 유지됨을 보인다.
- §4.3.2, §4.4.2, §4.5는 각각 외부 baseline context, 외부 real shift, 합성 corruption stress를 제공한다. Protocol이 서로 달라 §4.3.1의 component attribution을 대체하지 않으며, 표 사이 절대 수치를 직접 비교하지 않는다.
- §4.6은 **기각된 가설의 기록**이다. 본 논문의 주장을 지지하지 않으며, 지지하지 않는다는 사실 자체를 보고한다.
- **미완 항목**: (i) Gaussian noise 실패의 arm 간 특이성, (ii) §4.7 audit의 잔여 cell, (iii) CQ 및 multi-resolution/bank의 독립 ablation, (iv) 1-shot 보정 fallback과 성능 하락의 분리, (v) void 계측의 seed 일반화, (vi) sealed final holdout 평가.
- **정합성 주석**: 서로 다른 query population의 수치는 교차 비교하지 않는다. §4.2의 개발 조건 수치, §4.3.1·§4.4의 조건별 층 수치, §4.3.2의 벤치마크 실행 수치는 실행 identity가 다르므로 각 표 내부에서만 비교한다.

---

## 5. Discussion & Conclusion

### 5.1. Capacity and Adaptation vs. Support Calibration

- **§1.3 논증의 경험적 대조군**: 두 조건의 교집합에서 support 통계만 사용 가능하다는 논증은, 그 제약을 위반한 방법군의 결과로 뒷받침된다. 동일한 16-shot 조건에서 reverse distillation(32.2%), masked contextual prediction(14.2%), contextual scoring(5.7~15.0%), latent normalizing flow, synthetic boundary learning(47.6%), test-time shift restoration(31.9%)은 모두 단순 nearest-neighbor baseline(52.5%) 아래로 떨어졌다. 실패 양상은 일관된다. 학습되거나 적응되는 구성 요소가 **관측되지 않은 정상 변동을 결함으로 취급**한다.
- **용량이 아니라 보정**: 이 결과들은 제한된 정상 support 하에서 추가 layer 반응을 무조건 누적하거나 표현을 적응시키기보다, support-derived scale과 residual gate로 fusion을 조절하는 것이 유리할 수 있음을 시사한다.
- **Support 의존성이 알고리즘 선택의 문제임**: §4.3.1은 이 점을 직접 보인다. 동일한 backbone과 bank에서 fusion 규칙만 바꾸면, 성능의 support 예산 의존성이 5.41pp에서 0.44pp로 줄고 정상 이미지 2장이 보정 없는 구성의 16장을 앞선다. 즉 few-shot 성능은 대체로 표현 용량이나 데이터 예산의 함수로 논의되지만, 이 조건에서는 **score를 어떻게 구성하느냐가 지배적 요인**이었다.
- 다만 이 결과는 특정 backbone의 표현 용량이나 모든 adapter의 실패를 증명하지 않으며, 동일한 development 조건에서 관찰된 통제 비교로 해석한다.

---

### 5.2. Boundary of Claims and Future Work

- **주장 범위**:
  - 유효 주장 범위는 frozen DINOv2 representation을 support-calibrated fusion으로 결합하는 방법과 그 통제된 비교로 한정된다.
  - 본 논문은 두 배치 조건을 **관측 가능한 score 성분** 수준에서 연결하며, 단일 잠재 상태로 환원하지 않는다. 후자는 사전 등록 후 계측으로 검정해 기각했다(§4.6).
  - §4.3.1의 support-efficiency 주장은 **내부 통제 비교**에 한정된다. 외부 방법 대비로는 2-shot이 비교 방법 16-shot과 측정 오차 내 동등(−0.18pp)이며 우위가 아니다(§4.3.2).
  - 주 결과는 5개 support-split seed에서 반복되었고 shot-축 실험의 효과 크기가 floor의 7배 이상이나, **sealed final holdout 평가와 통계적 유의성 검정은 수행하지 않았다.** shot-축 실험 자체는 seed 0 단일이다.
  - Shift에 대한 주장은 **photometric·acquisition shift 하의 최고 절대 성능**이며, 최소 저하나 보편적 shift robustness가 아니다. High-frequency sensor corruption은 명시적 실패 모드이고 그 원인은 미검증이다.
  - 1-shot 결과는 보정 추정량이 다른 조건에서 얻어졌으므로 다른 shot 수준과 같은 척도로 비교하지 않는다.
  - Multi-resolution, multi-bank, CQ의 독립 기여와 DINOv3 transferability는 동일 조건의 별도 검증 없이 주장하지 않는다.
  - 부품의 임의적 3D 회전이나 시점 불일치를 수반하는 전면적 category shift 해결은 본 연구의 직접적 주장 범위를 넘는다.
- **향후 연구**:
  - Corruption 조건에서 arm 간 비교를 수행해 Gaussian noise 실패가 본 방법 특유인지 nearest-neighbor 계열의 일반 현상인지 판별한다(§4.5).
  - 1-shot 보정 fallback을 대체 추정량으로 교체해, 1-shot 성능 하락 중 얼마가 support 부족이고 얼마가 추정량 교체인지 분리한다(§3.4, §4.3.1).
  - Shot-축 실험을 복수 seed로 반복하고, §4.7 audit 완료 후 operating-point transfer를 별도로 판정한다.
  - sealed evaluation과 backbone transfer를 동일 protocol로 검토한다.
