# Support-Calibrated Fusion with Query-Side Checks: Few-Shot Industrial Anomaly Detection under Acquisition Shift

**Hun Im\*, Author Two, Author Three, Pilsung Kang\***

> **v7 변경 요약** (v6 대비, 2026-09-24)
>
> 1. **중심 원칙 교체**: "support 통계만 사용"에서 **"support는 기준을 정하고, 각 query는 그 기준에 비춰 점검한다"**로. few-shot은 정상 기준을 추정하는 데이터 제약이고 acquisition shift는 query 분포의 변화다. 각 구성 요소의 설계 동기는 이 구분에서 나오지만, 실험은 원인의 배타적 1:1 대응을 증명하지 않는다.
> 2. **noise guard를 방법 본체에 편입**(§3.7). Gaussian s3는 개발 조건, 설계에 쓰지 않은 5개 노이즈 조건은 평가 조건으로 분리해 보고한다.
> 3. **수치 정정**(모두 원본 run과 대조): noise floor 설명과 값(§2.3), Table 1 전면 재검증(4-shot 57.33 등), "support 의존성 제거·평탄성" 주장 폐기, DINOv3는 이득이 아님(§4.6), source-fixed audit 결론 반영(§4.4), 공식 서버 행은 RSG3가 아님을 명시.
> 4. 근거 문서: `PaperWorks/experiment_results.md`와 `results/v7_evidence/`의 최신 비교 보고서(2026-09-25 확인). 논문 원고(`.tex`)는 아직 이 정정을 반영하지 않았다.

---

## Index

- Abstract
- 1. Introduction

  - 1.1. Two Deployment Conditions
  - 1.2. What Each Condition Breaks
  - 1.3. Design Principle: Support Sets the Reference, Each Query Is Checked Against It
  - 1.4. Problem Statement
  - 1.5. Testable Predictions
  - 1.6. Contributions
- 2. Pilot Evidence and Measurement Discipline

  - 2.1. Layer Complementarity as Background Motivation
  - 2.2. Support-Limited Scale and Normal-Tail Mismatch
  - 2.3. The Noise Floor as a Decision Rule
  - 2.4. Limits Observed for Support-Only Adaptive Selection
  - 2.5. Scope and Limits of the Pilot Evidence
- 3. Method

  - 3.1. Problem Formulation and Constraints
  - 3.2. Framework Overview
  - 3.3. Asymmetric Residual Spike Gating
  - 3.4. Tail-Calibrated Quantile Normalization
  - 3.5. Fixed Feature, Resolution, and Bank Aggregation
  - 3.6. Component Quantile Postprocessing
  - 3.7. Query-Side Acquisition Check (Noise Guard)
  - 3.8. Scoring Protocol and Computational Cost
- 4. Evidence and Analysis

  - 4.1. Setup and Reading Rules
  - 4.2. Component Ablation at 16 Shots
  - 4.3. Condition A — Shrinking the Support Budget
  - 4.4. Condition B-1 — Controlled Acquisition Shift
  - 4.5. Condition B-2 — Synthetic Corruptions and the Query-Side Check
  - 4.6. Backbone Swap (Negative Result)
  - 4.7. A Rejected Hypothesis: A Shared Latent State
  - 4.8. Evidence Boundary
- 5. Discussion & Conclusion

  - 5.1. Calibration and Checks vs. Capacity and Adaptation
  - 5.2. Boundary of Claims and Future Work

---

## Abstract

Few-shot 산업 이상 탐지 시스템은 배치 시 두 조건을 동시에 만난다. **정상 이미지가 적다는 제약**과, **촬영 조건이 배치 이후에 바뀐다는 변화**다. 두 조건은 서로 다른 곳에서 기준을 무너뜨린다. 정상 support가 좁으면 support로 정한 기준 자체가 불완전하고, query에만 생기는 변화는 support에 흔적을 남기지 않아 support 통계로는 보이지 않는다. 본 논문은 이 구분에서 하나의 설계 원칙을 끌어낸다. **support는 기준을 정하고, 각 query는 그 기준에 비춰 점검한다.** 이 원칙 아래 두 구성 요소를 둔다. (i) component별 leave-source-out q95로 score scale을 맞추고, 정상 support의 q99를 넘은 shallow residual만 deep score에 더하는 **support-calibrated residual fusion**, (ii) 각 query의 센서 노이즈를 support 기준과 비교해, 기준을 크게 벗어난 query만 저해상도 경로로 처리하는 **query-side acquisition check**다. 추가 학습 파라미터는 0개이며 target 레이블과 target 데이터 적합(fitting)을 쓰지 않는다.

MVTec AD 2에서 residual gating은 16-shot AUPRO@0.05를 +5.13pp, tail calibration은 +1.66pp 높였고, 주 구성의 5개 support-split seed 평균은 59.48 ± 0.60이었다. 한 support 선택(seed 0)의 shot ablation에서는 gating 이득이 16-shot +6.51pp, 2-shot +11.49pp였으며, 별도 MVTec AD와 VisA 실험도 저샷에서 더 큰 평균 이득을 보였다. 이 경향의 support-seed 재현(E1)은 진행 중이다. MVTec AD 2에서 외부 방법 대비 1~16-shot 모든 budget의 AUPRO가 가장 높았고, 동일 16-shot support로 비교한 세 획득 shift stratum에서도 절대 AUPRO가 가장 높았다(+3.29/+1.92/+3.20pp). 이는 하락폭 최소의 증거는 아니다. MVTec AD Gaussian noise s3에서는 −20.69pp 하락했고, 고해상도 가지를 제외하면 하락이 5.09pp 완화되었다. query-side check는 개발 조건에서 +4.34pp, 별도 MVTec AD 노이즈 5조건에서 평균 +7.94pp를 얻었다. 기록된 비노이즈 점검 18,431회에서는 발화하지 않았지만 brightness s5의 3/1725건은 발화했다. VisA에서는 clean 점검 2162회에 발화가 없고 Gaussian s3에서 +4.76pp였으며, 나머지 노이즈 조건은 평가 중이다. 이득은 주로 AUPRO에 집중되며 고정 임계값 FPR 안정성은 사전 등록 기준을 통과하지 못했다.

---

## 1. Introduction

### 1.1. Two Deployment Conditions

- **조건 A — 정상 데이터 제약 (few-shot)**: 신규 품목과 공정 전환이 잦으면 대규모 정상 데이터를 다시 모으기 어렵다. 소수의 정상 reference로 검사 기준을 만들어야 한다. 이 제약은 **설계 시점에 알려져 있고 정적이다.**
- **조건 B — 획득 조건 변화 (distribution shift)**: 조명, 노출, 카메라 위치, 센서 상태는 배치 이후 바뀐다. 이 변화는 **배치 이후에 생기고, 언제 어떤 방향일지 미리 알 수 없다.**
- 두 조건은 문제 설정에서 구별된다. A는 정상 reference 수의 제약이고 B는 query 획득 조건의 변화다. 실제 오류 원인은 서로 얽힐 수 있으며 이 구분만으로 구성 요소별 배타적 원인이 정해지지는 않는다.

Frozen foundation backbone 위의 nearest-neighbour 방법은 정상 support의 patch feature를 bank에 저장하고 query patch와의 거리를 anomaly score로 쓴다. 학습 부담은 작지만, 큰 거리가 결함 때문인지 support가 담지 못한 정상 변화 때문인지는 거리만으로 구분되지 않는다.

---

### 1.2. What Each Condition Breaks

두 조건은 기준의 서로 다른 측면을 시험한다.

- **정상 reference가 적으면** 정상 변동의 일부가 관측되지 않는다. 보정되지 않은 shallow 기여가 정상 반응을 anomaly evidence로 합산할 수 있다. gating 이득이 저샷에서 커진 관측은 이 설계와 일치하지만, seed 반복 전에는 일반적 인과 설명으로 확정하지 않는다(§4.3.2).
  - 코멘트
    - Outline 에는 "seed 반복 전에는 일반적 인과 설명으로 확정하지 않는다"이와 같은 실험 테크닉적인 서술은 모두 제외.
    - 여기서 바로 "shallow 기여가 정상 반응을 anomaly evidence로 합산할 수 있다" 라는 것은 다소 논리적인 비약이 존재하며, 또한 우선 정상 reference가 적은 경우 정상 변동의 일부가 관측되지 않아 anomaly score에 대한 신뢰가 감소한다 라는 대전제 부터 논리를 끌고 가야 함
- **query 획득 조건이 바뀌면** 고정된 support 기준과 query가 어긋날 수 있다. 합성 corruption에서는 support는 깨끗했고 노이즈는 query에만 있었다. 고해상도 경로를 끈 단일 ablation이 Gaussian 하락을 줄였지만, 다른 support 구성 요소가 원인에 기여하지 않는다는 결론은 나오지 않는다(§4.5).
  - 이 역시 마찬가지며 query 단계에서 shift에 따른 anomaly score에 대한 신뢰가 감소한다 라는 대전제로 시작되어야 함

---

### 1.3. Design Principle: Support Sets the Reference, Each Query Is Checked Against It

§1.2의 구분에서 설계 원칙이 나온다.

> **support는 기준을 정하고, 각 query는 그 기준에 비춰 점검한다.**

| 조건                | 무너지는 곳                               | 대응 구성 요소                                     | 쓰는 정보                                 |
| :------------------ | :---------------------------------------- | :------------------------------------------------- | :---------------------------------------- |
| A: 정상 데이터 제약 | 기준 추정에 쓰는 정상 예시가 적음         | support-calibrated residual fusion (§3.3–3.4)    | support에서 계산한 q95, q99               |
| B: query 쪽 변화    | 고정 support와 query의 획득 조건이 어긋남 | 측정 가능한 노이즈에 대한 query-side check (§3.7) | support 기준 + query 자신의 노이즈 추정치 |

이 원칙이 허용하는 것과 허용하지 않는 것을 분명히 한다.

- **허용**: 모든 기준값을 정상 support에서 미리 정하고, 각 query를 그 기준에 비춰 **독립적으로** 처리한다. query를 보고 처리 경로를 바꾸는 것은 허용한다.
- **허용하지 않음**: target 레이블 사용, target 데이터 분포 적합(fitting), 여러 query를 모아 통계를 내는 방식, query에 따른 파라미터 갱신.

이 점에서 query-side check는 test-time adaptation과 다르다. 학습하지 않고, target 분포를 추정하지 않으며, 각 query를 따로 처리한다. 이미지가 한 장씩 들어오는 생산 라인에서 그대로 쓸 수 있고, "노이즈 비율 > 9라서 저해상도로 전환" 같은 판단 근거를 기록할 수 있다.

한편 support 통계로 할 수 있는 일에는 한계가 있다(§2.4). support에는 결함이 없으므로, support만으로는 어떤 component가 결함을 잘 구분하는지 고를 수 없다. 그래서 두 구성 요소 모두 학습되는 가중치 없이 **고정된 규칙**을 쓴다.

---

### 1.4. Problem Statement

제한된 정상 support 하에서, 그리고 query의 획득 조건이 바뀔 수 있는 상황에서,

> 정상 support만으로 component별 score scale을 맞추고 shallow response 중 정상 범위를 넘은 부분만 anomaly evidence로 허용할 수 있는가? 그리고 support로는 보이지 않는 query 쪽 변화를, target 데이터에 적합하지 않고 query마다 점검해 막을 수 있는가?

이 문제는 backbone의 inductive bias를 고치거나 결함 표현을 복원하는 문제가 아니다. frozen representation 위에서 score를 어떻게 구성하느냐의 문제다.

---

### 1.5. Testable Predictions

|              | 예측                                                                           | 검증    | 결과                                                                                                             |
| :----------- | :----------------------------------------------------------------------------- | :------ | :--------------------------------------------------------------------------------------------------------------- |
| **P1** | gating 이득이 저샷에서 커지는가                                                | §4.3.2 | seed 0에서 16-shot +6.51 → 2-shot +11.49pp; E2에서도 저샷 평균 이득이 큼. E1 seed 반복 대기                     |
| **P2** | 그 결과 외부 방법 대비 우위가**모든 support budget에서** 유지되어야 한다 | §4.3.1 | **지지**: AUPRO +4.21 ~ +6.10pp, 1~16-shot 전부 1위                                                        |
| **P3** | 같은 support에서 acquisition shift 하 절대 AUPRO 우위가 유지되는가             | §4.4   | E10의 shift_1/2/3에서 best comparator 대비 +3.29/+1.92/+3.20pp; 하락폭 최소는 아님                               |
| **P4** | Gaussian noise 하락에 고해상도 경로가 기여하는가                               | §4.5.2 | 저해상도만 쓰면 하락 +5.09pp 완화. E5의 MAD 교체는 하락을 3.19pp 악화; 원인 배제는 불가                          |
| **P5** | support 기준 query 점검이 측정된 노이즈에서 유익하고 clean에서 무해한가        | §4.5.3 | MVTec AD dev +4.34, 별도 5조건 평균 +7.94; VisA clean 0/2162 발화, Gaussian s3 +4.76; 나머지 VisA 노이즈 평가 중 |
| P6           | 두 조건이 하나의**잠재 상태**를 공유한다                                 | §4.7   | **기각**                                                                                                   |

---

### 1.6. Contributions

- **C1. 설계 원칙**: 적은 정상 reference와 변하는 query 획득 조건을 구분하고, "support는 기준, query는 점검"이라는 원칙으로 두 조건을 함께 다룬다. 배타적 원인 분리는 주장하지 않는다.
- **C2. Support-calibrated residual fusion**: support 통계로 component scale을 맞추고 shallow residual만 조건부로 더한다. 추가 파라미터 0개.
- **C3. Query-side acquisition check**: support 기준 대비 query 노이즈가 크면 저해상도 경로로 처리한다. 레이블 없이 query별 독립 처리하고, 미발화 query에는 기존 점수 경로를 적용한다.
- **C4. 두 축의 통제된 검증**: 1~16-shot × 3 arm 성분 ablation(120 cells), MVTec AD/VisA의 paired gating 비교, 외부 방법 대비 budget 비교, support 고정 acquisition shift(E10), corruption 비교, 단일 구성 요소 ablation(120 cells), 개발 조건과 별도 노이즈 조건의 대응 검증. 반복 support seed와 VisA 노이즈 조건 평가는 미완이다.
- **C5. 측정 규율과 기각된 설명**: noise floor를 판정 규칙으로 사용하고, 두 조건을 하나의 잠재 상태로 환원하는 가설을 사전 등록 후 검정해 기각했다. 실패한 audit(고정 임계값 FPR 안정성)과 음성 결과(backbone 교체)도 보고한다.

---

## 2. Pilot Evidence and Measurement Discipline

### 2.1. Layer Complementarity as Background Motivation

- Shallow layer는 texture, edge, 작은 appearance 변화에 민감하다. 이 민감도는 결함과 정상적인 조명·반사 변화에 모두 반응한다.
- Deep layer는 더 안정적인 base score를 주지만 모든 국소 결함에서 우월하지는 않다. 단일 layer 진단에서 구조적 객체는 deep이 유리했고, 세 layer를 보정 없이 합치면 deep의 이점이 희석되었다.
- 이 차이를 inductive bias mismatch의 증명으로 해석하지 않는다. 방법은 representation이나 attention을 바꾸지 않는다.
- 따라서 질문은 "어떤 layer가 옳은가"가 아니라 **"정상 support에서 각 layer score가 어느 범위까지 안정적인가"**이다.

### 2.2. Support-Limited Scale and Normal-Tail Mismatch

각 component $(r,b,l)$에서 query patch $p$의 거리를

$$
d_{r,b,l}(p)=\min_{u\in B_{r,b,l}}D\bigl(f_{r,l}^{q}(p),u\bigr)
$$

로 둔다. 이 거리의 크기와 upper tail은 layer, resolution, bank마다 다르다. 따라서 $d_{r,b,l_1}(p)>d_{r,b,l_2}(p)$라는 사실만으로 첫 component가 더 강한 anomaly evidence라고 판단할 수 없다. support가 좁으면 정상 변동의 tail을 조밀하게 관측할 수 없고, 관측되지 않은 정상 pattern이 일부 component에서 큰 거리로 나타난다. 본 연구는 support에서 직접 관측할 수 있는 두 통계량만 쓴다.

- component별 leave-source-out 정상 거리의 **q95**: score scale의 기준
- 정규화된 shallow response의 **q99**: shallow 기여를 허용하는 기준

이 통계량은 target 분포나 결함 레이블로 조정하지 않으며, target FPR이나 결함 보존을 보장하지 않는다(§4.4).

### 2.3. The Noise Floor as a Decision Rule

- **Bank seed 범위**: 코드와 설정을 고정하고 bank subsampling seed만 바꾼 세 번의 실행(BC0–BC2)에서 AUPRO 범위는 **0.86pp**였다. 표본이 세 개뿐이라 실제 잡음을 과소평가했을 수 있다.
- **Support-split 범위**: 최종 구성을 서로 다른 5개 support-split seed로 돌렸을 때 **59.48 ± 0.60, 범위 1.40pp**였다.
- **개발 판정 관례**: 이 세 bank seed에서 얻은 0.86pp를 작은 차이를 선별하는 경험적 기준으로 쓴다. 0.86~1.40pp 차이는 **경계선 수준**으로 표시한다. 이는 보편적 통계적 noise floor나 다른 실험의 신뢰구간이 아니다.
- *정정 기록*: v6는 "bank seed 5회"로 floor를 확립했다고 썼다. 실제로는 BC3·BC4가 bank 용량 실험이었다(tokens/source 1024, 4096). 5개를 모두 넣으면 범위는 1.06pp다.

이 규칙 때문에 여러 결과를 근거에서 뺀다. local relation(+0.11), coherent matching(±0, 실제로는 예측이 바뀌지 않은 no-op), region recovery(−0.05), shallow 가지 자체의 순이득(+0.47)이 그렇다.

### 2.4. Limits Observed for Support-Only Adaptive Selection

- 정상 support 16장만으로 layer 가중치를 학습하거나 고르는 selector 5종은 **모두 8개 클래스 중 5개만 맞혀 우연 수준**이었다.
- 가능한 설명은 정상 support만으로 결함 쪽 분리도를 알기 어렵다는 점이다. 여기서 검토한 selector의 목적량은

  $$
  \text{selector goal}=\frac{\mathbb{E}_{defect}[D(x)]}{\mathbb{E}_{normal}[D(x)]}
  $$

  인데, 정상 support에는 결함이 없으므로 **분모만 직접 추정할 수 있다.** 다른 support-only 선택 규칙까지 불가능하다는 증명은 아니다.
- 합성 결함으로 분자를 대신하려는 시도도 실패했다. 합성 결함은 layer 간 분리도가 사실상 같았고(0.7457 대 0.7453), 실제 결함은 뚜렷이 갈렸다(0.677 대 0.800).

> **설계 귀결**: 이 실험의 support-only selector는 component 판별력 선택에 실패했고, 정상 support의 quantile은 score scale 기준으로 쓸 수 있었다. 그래서 학습되는 가중치 없이 보정한 뒤 **고정된 규칙**으로 결합한다. §3.7의 query-side check도 학습 없이 고정 임계값을 쓴다.

### 2.5. Scope and Limits of the Pilot Evidence

- pilot은 layer별 score 분포와 fusion 동작을 관찰한 개발 근거이며, attention의 인과 구조나 representation 수준의 결함 보존을 증명하지 않는다.
- 일부 학습·적응 방법이 16-shot에서 열세였다는 사실을 모든 adapter의 일반적 실패로 넓히지 않는다.
- q95와 q99는 정상 support에서 얻은 상대적 기준이며, shift 하의 FPR이나 결함 TPR을 보장하지 않는다.

---

## 3. Method

### 3.1. Problem Formulation and Constraints

- **입력**: $N$-shot 정상 support 집합 $\mathcal{S}_N=\{x_1^s,\dots,x_N^s\}$.
- **현실 제약**: support는 정상 다양체의 일부만 관측한다(조건 A). query의 획득 조건은 support와 다를 수 있다(조건 B).
- **학습 제약**: backbone 동결, 추가 학습 파라미터 0개.
- **Support fitting**: 정상 support만으로 feature bank, leave-source-out q95 scale, shallow q99 gate, support 노이즈 기준값을 계산한다.
- **Query 처리**: 각 query는 독립적으로 처리한다. 위 기준값은 query 평가 중 갱신하지 않는다. query 레이블, 결함 mask, 다른 query의 정보는 쓰지 않는다.
- **출력**: query별 연속 anomaly map $A(x^q)\in\mathbb{R}^{H\times W}$.

### 3.2. Framework Overview

```text
[Query 이미지] ──► [Query-side check: 노이즈 비율 = σ(query) / median σ(support)]
                        │
          ratio ≤ 9 ────┴──── ratio > 9
             │                    │
   4개 해상도 [448,672,896,1120]   저해상도 [448,672]만
             └─────────┬──────────┘
                       ▼
        [Frozen DINOv2-L, layer 5 / 11 / 23 patch feature]
                       ▼
        [support bank와 1-NN 거리, component별]
                       ▼
        [Tail-calibrated 정규화: component별 LOO q95로 나눔]
                       ▼
        [deep base + (shallow − support q99)⁺ residual]
                       ▼
        [bank 평균 → native space 투영 → 해상도 평균 → CQ 후처리]
                       ▼
               native-resolution anomaly map
```

비율이 9 이하인 query는 기존 점수 경로를 그대로 탄다. 동일 하드웨어의 clean 실행에서 저장된 지표가 같았으며, 모든 환경과 출력의 바이트 동일성까지 검증한 것은 아니다.

### 3.3. Asymmetric Residual Spike Gating

- **목적**: shallow layer는 작은 결함의 경계를 잡지만 정상적인 조명·반사·센서 변화에도 민감하다. support가 좁을수록 이 민감도는 결함이 아닌 정상 변동에 반응한다. shallow를 항상 켜 두지 않고, 정상 support에서 보기 드문 반응만 보완 신호로 쓴다.
- **개념식**:

  $$
  z_l(p)=\frac{d_l(p)}{q_{95,l}^{\mathrm{normal}}},\qquad
  z_d(p)=\operatorname{mean}\bigl(z_{11}(p),z_{23}(p)\bigr),\qquad
  S(p)=z_d(p)+\operatorname{ReLU}\bigl(z_5(p)-\tau_{5,q99}\bigr).
  $$
- shallow가 정상 q99 아래면 deep base만 남고, 넘으면 넘은 만큼만 더해진다. 즉 "shallow를 더하는 방법"이 아니라 **"shallow 중 정상 범위를 넘은 반응만 허용하는 방법"**이다.
- **한계**: 이 gate는 결함만 통과시킨다고 보장하지 않는다. 정상 반응을 줄일 수도, 결함 반응을 줄일 수도 있다.

### 3.4. Tail-Calibrated Quantile Normalization

- **문제**: layer와 resolution마다 정상 거리의 spread와 tail이 다르다. 단순 평균하면 scale이 큰 component가 fusion을 지배한다.
- **support-only 계산**: component $m$마다 support image가 자기 자신을 이웃으로 쓰지 않도록 leave-source-out 정상 거리를 모으고, 그 q95를 $q_{95}^{(m)}$로 저장한다. 정규화는 $\widetilde{d}_m(p)=d_m(p)/(q_{95}^{(m)}+\epsilon)$이다.
- **대안 대비 근거**: 같은 조건에서 raw, MAD, rank transform보다 AUPRO와 pAUROC가 높았다. rank는 거리 기하를 없애 pAUROC가 떨어졌고, MAD는 정상 upper tail에 덜 반응했다.
- **측정된 한계 — 기준 드리프트**: leave-source-out 기준은 support 수에 따라 움직인다. 정상 query의 q99 gate 발화율은 16-shot에서 설계값 1%와 일치했지만(0.994배), 8/4/2-shot에서 0.95 / 0.74 / 0.64배로 낮아졌다. support가 적으면 LOO 이웃이 희소해 기준이 크게 잡히기 때문이다. gate의 동작점은 support 수에 대해 불변이 아니다.
- **1-shot 추정량 교체**: support가 1장이면 leave-source-out이 정의되지 않아, 구현은 같은 이미지 안의 leave-token-out을 쓴다. 1-shot 결과는 다른 shot과 같은 척도로 해석하지 않는다.

### 3.5. Fixed Feature, Resolution, and Bank Aggregation

- frozen DINOv2-L의 `[5, 11, 23]` layer를 `[448, 672, 896, 1120]` 네 해상도에서 쓴다. 기저 해상도를 빼면 texture 클래스가 무너지고, 고해상도만 쓰면 클래스별 반응이 비단조적이다.
- support feature를 네 개의 random bag bank로 나눈다. exhaustive bank나 coreset 선택보다 안정적이었다.
- 각 query 위치에서 bank까지의 1-NN 거리를 계산한다. 최종 구성은 $k=1$이다.
- resolution마다 token score를 bilinear로 native space에 투영하고, bank 평균 후 resolution 평균을 낸다.

### 3.6. Component Quantile Postprocessing

- query map의 99번째 백분위로 high-score mask를 만들고, 8-연결 성분마다 값을 그 성분의 최댓값으로 채운다. GT는 쓰지 않는다.
- 동기: AUPRO는 GT 성분을 면적으로 정규화하므로 부분 검출에 불이익이 크다.
- CQ의 이득(+0.96pp)은 경계선 수준이며(§2.3), 모든 arm에 공통으로 적용되므로 독립 기여로 주장하지 않는다.

### 3.7. Query-Side Acquisition Check (Noise Guard)

- **왜 필요한가**: query에만 생기는 변화는 support에 흔적이 없어, §3.3–3.4의 support 기준으로는 감지되지 않는다(§1.2). 합성 corruption에서 Gaussian noise 실패는 고해상도 가지에서 생겼다(§4.5.2).
- **측정량**: Immerkær(1996)의 고속 노이즈 추정치 $\sigma$. native 해상도 grayscale에 3×3 Laplacian 차분 한 번으로 계산한다. 레이블과 모델 출력을 쓰지 않는다.
- **기준**: support 이미지들의 $\sigma$ 중앙값 $\sigma_S$를 fit 단계에서 저장한다.
- **판단**: query마다 $\rho=\sigma(x^q)/\sigma_S$를 계산해, $\rho>k$이면 저해상도 `[448, 672]` 가지만으로 점수를 낸다. 그 외에는 기존 경로 그대로다.
- **임계값**: $k=9$. 모델 출력이나 레이블 없이 이미지 통계만으로 정했다. 레이블 없는 표본에서 노이즈가 아닌 이미지의 최대 비율은 6.08, 노이즈 이미지의 최소 비율은 13.46이었고, 9는 둘의 기하평균이다.
- **성질**: 각 query를 독립적으로 처리하고, 파라미터를 갱신하지 않으며, target 데이터를 모으지 않는다. 미발화 query는 기존 점수 경로를 쓰며 같은 GPU의 clean 실험에서 저장된 지표가 같았다(§4.5.3). 비용은 이미지당 합성곱 한 번이다.
- **알려진 작동 범위 한계**: 밝기가 강해질수록 비율이 올라가 임계값과의 여유가 줄어든다(brightness s3/s4/s5 최대 비율 6.20 / 8.10 / 9.81). brightness s5에서는 1725개 중 3개 query가 작동한다(`pill` 2/167, `toothbrush` 1/42).
- **대응 동작의 선택 근거**: 어떤 대응을 쓸지는 ablation 결과를 보기 전에 규칙으로 고정했다. 시험한 제거 arm 중 고해상도 가지 제외가 등록된 완화 기준을 넘어서 저해상도 경로가 선택되었다.

### 3.8. Scoring Protocol and Computational Cost

- 추가 학습 파라미터와 역전파가 없다. 측정하지 않은 FPS는 주장하지 않는다.
- **support 단계(한 번)**: feature 추출 → bank 구성 → LOO q95 → shallow q99 → support 노이즈 기준값.
- **query 단계(매번)**: 노이즈 점검 → 선택된 해상도에서 feature 추출과 1-NN 거리 → q95 정규화 → deep base + q99 residual → bank·해상도 결합 → CQ.

---

## 4. Evidence and Analysis

### 4.1. Setup and Reading Rules

- **데이터**: MVTec AD 2(8개 클래스), MVTec AD(15개), VisA(12개). frozen DINOv2-L/14. primary metric은 `seg_AUPRO@0.05`, guardrail은 `seg_AUROC@0.05`(pAUROC), diagnostic은 oracle F1이다(oracle F1은 test 레이블로 임계값을 고르므로 고정 임계값 성능과 같지 않다).
- **개발 판정 관례**: §2.3의 0.86pp는 관측된 세 bank seed 범위이며 보편적 유의성 검정이 아니다.
- **개발 조건과 평가 조건의 분리**: query-side check는 Gaussian noise s3를 보고 설계했다. 이 조건은 **개발 조건**으로 표시하고, 일반화의 근거는 설계에 쓰지 않은 5개 노이즈 조건(shot/impulse/speckle s3, Gaussian s1/s5)으로 둔다. 이 조건들은 이미지별 seed로 생성했고 두 기기에서 픽셀 단위로 같다.
- **미발화 결과 확인 범위**: 같은 GPU의 MVTec AD clean 15/15 클래스와 VisA clean 12/12 클래스에서 저장된 지표가 같았다. 기록된 MVTec AD 2 split 등에서 미발화가 관측되었지만 모든 출력의 바이트 동일성을 전 범위에서 검증한 것은 아니다. brightness s5는 3/1725건 발화했다.
- **표기**: 내부 실험 식별자 대신 ablation 표기를 쓴다. 서로 다른 query population의 표끼리 절대 수치를 비교하지 않는다. 대부분의 결과는 seed 0이다.

### 4.2. Component Ablation at 16 Shots

MVTec AD 2, 16-shot seed 0, 8개 클래스. backbone, support, 해상도, bank, CQ, evaluator를 고정하고 scaling과 shallow 기여만 바꿨다.

| Configuration                                  |      AUPRO@0.05 |     pAUROC@0.05 |       F1-oracle |         Δ AUPRO |
| :--------------------------------------------- | --------------: | --------------: | --------------: | ---------------: |
| **Ours** (tail scaling + gated residual) | **59.15** |           77.11 |           35.87 |               — |
| w/o residual gating                            |           54.02 |           74.04 |           32.90 | **−5.13** |
| w/o tail calibration (MAD)                     |           57.48 |           76.16 |           33.89 | **−1.66** |
| w/o both                                       |           51.16 |           73.02 |           29.98 |           −7.99 |
| w/o shallow branch (deep-only)                 |           58.67 | **77.35** | **37.11** |           −0.47 |

- gating과 tail calibration은 각각 floor를 크게 넘고, 대체로 독립적으로 기여한다(둘 다 제거 −7.99 ≈ 합 −6.79).
- **shallow 가지 자체의 순이득(+0.47)은 floor 미만**이고 pAUROC와 F1은 오히려 deep-only가 높다. 이 표가 지지하는 것은 shallow를 더하는 것이 아니라 **shallow의 정상 tail을 제한하는 것**이다.
- 5-seed 재현: 59.48 ± 0.60.

### 4.3. Condition A — Shrinking the Support Budget

#### 4.3.1. External Comparison across Budgets (MVTec AD 2, PRO@0.05, seed 0)

모든 값은 저장된 run에서 다시 계산했다(`experiment_results.md` Table 1).

| Shots | RegAD | RD4AD | PatchCore | FastRecon | RN-FMLK | SuperAD (v2) | SuperAD (v3) |  **Ours** |      Δ vs best |
| :---- | ----: | ----: | --------: | --------: | ------: | -----------: | -----------: | --------------: | --------------: |
| 1     | 15.56 | 36.89 |     38.07 |     35.11 |   39.25 |        47.47 |        48.18 | **52.49** | **+4.31** |
| 2     | 14.50 | 38.02 |     36.77 |     34.16 |   38.56 |        48.25 |        48.87 | **53.31** | **+4.44** |
| 4     | 12.68 | 40.07 |     42.10 |     38.04 |   41.39 |        51.23 |        50.69 | **57.33** | **+6.10** |
| 8     | 17.26 | 38.33 |     48.17 |     47.37 |   43.11 |        55.83 |        54.99 | **60.04** | **+4.21** |
| 16    | 15.93 |    — |     48.83 |     47.05 |   42.51 |        53.49 |        54.87 | **59.15** | **+4.28** |

- **P2 지지**: 1~16-shot 모든 budget에서 AUPRO 1위, 격차 +4.21 ~ +6.10pp.
- 우리 2-shot(53.31)은 SuperAD 16-shot(53.49)과 **동등**하다(−0.18pp, floor 미만). 앞선다고 쓰지 않는다. 1-shot(52.49)은 1.00pp 낮으므로 "같다"고 쓰지 않는다.
- **다른 지표에서는 최고가 아니다.** pAUROC는 1~2-shot에서 SuperAD-v3가 높고(75.64 대 73.53), oracle F1은 모든 shot에서 SuperAD-v3가 높다(16-shot 43.35 대 35.87). 이득은 **AUPRO에 집중**된다.
- RD4AD 16-shot은 해당 baseline run이 없어 비웠다.

#### 4.3.2. Component Ablation across Budgets (120 cells)

MVTec AD 2 `test_public`, seed 0. support manifest는 **엄격히 중첩**되어 있다(N-shot 집합은 16-shot 집합의 앞 N장). `regular` stratum, AUPRO@0.05.

| Configuration          |              1 |               2 |              4 |              8 |             16 |
| :--------------------- | -------------: | --------------: | -------------: | -------------: | -------------: |
| **Ours (gated)** |          55.94 |           60.82 |          60.42 |          60.22 |          61.26 |
| w/o residual gating    |          48.01 |           49.34 |          52.92 |          53.56 |          54.75 |
| w/o shallow branch     |          56.09 |           60.65 |          60.22 |          59.50 |          60.83 |
| **Gating gain**  | **7.93** | **11.49** | **7.50** | **6.66** | **6.51** |
| Shallow net            |         −0.15 |            0.17 |           0.19 |           0.72 |           0.43 |

- **관측된 P1 경향**: 이 seed에서 gating 이득은 모든 budget에서 0.86pp보다 크고, 2-shot 이득이 16-shot보다 큰 클래스가 8개 중 7개다(`can` 예외). paired arm은 같은 support를 썼다. 그러나 1-shot은 다른 추정량이고 값은 단조롭지 않으며, support 선택을 반복하는 E1이 끝나야 저샷 경향의 재현성을 판단할 수 있다.
- **다른 데이터셋 E2**: support hash로 짝지은 MVTec AD full15의 1/2/4/8/16-shot gating 이득은 +5.70/+6.00/+4.31/+3.60/+2.91pp, VisA full12는 +10.80/+12.29/+8.63/+6.09/+5.15pp였다. 두 데이터셋에서 저샷(1/2/4) 평균 이득이 16-shot 이득보다 컸다. seed 0 결과다.
- **E7 추정량 점검**: 사전 등록된 2-shot `regular` stratum에서 token 61.35573608, source 60.82266095로 Δ_est=+0.53307513pp였다. 추정량 교체가 등록된 손실 기준을 충족하지 않았지만, 이 결과만으로 1-shot 손실을 support 부족에 배타적으로 귀속할 수 없다. whole-split 값 59.267/58.704는 이 판정에 쓰지 않는다.
- `shift_1` stratum에서도 같다: gating 이득 6.73 / 10.02 / 5.09 / 4.60 / 4.61.
- shallow 가지 자체는 모든 budget, 두 stratum 모두에서 floor 미만이다.
- **폐기한 주장**: "support 의존성을 제거한다." gated의 2→16-shot 변화(0.44pp)는 평균만 평탄하고, 클래스별로는 −6.52(`can`) ~ +7.34(`walnuts`)로 크게 움직인다.
- **보류한 주장**: "gated 2장이 ungated 16장을 이긴다"(60.82 대 54.75)는 8개 중 7개 클래스에서 성립하지만, 서로 다른 support 집합의 비교이고 이득이 `fabric`(+21.8)과 `wallplugs`(+14.8)에 몰려 있으며 seed가 하나다. seed 반복 전에는 주장하지 않는다.
- 1-shot 열은 추정량이 다르다(§3.4).

### 4.4. Condition B-1 — Controlled Acquisition Shift

MVTec AD 2, support·backbone·해상도·seed·evaluator 고정, query의 획득 조건만 다르다. target adaptation 없음. 이 데이터에서 query-side check는 작동하지 않으므로 아래 수치는 check 유무와 무관하다.

| Configuration        |         Regular |           Shift |              Δ | Shift 순위 |
| :------------------- | --------------: | --------------: | --------------: | :---------: |
| **Ours**       | **61.26** | **59.40** |          −1.86 | **1** |
| w/o shallow branch   |           60.83 |           58.77 |          −2.06 |      2      |
| w/o tail calibration |           60.10 |           57.27 |          −2.83 |      3      |
| w/o residual gating  |           54.75 |           54.79 | **+0.04** |      4      |
| w/o both             |           52.40 |           51.50 |          −0.90 |      5      |

- **P3 지지 범위**: 이 표의 shift_1에서 gating 제거 대비 +4.61pp. E10은 같은 16-shot support hash를 확인한 8개 방법 비교이며 RSG3가 shift_1/shift_2/shift_3에서 best comparator 대비 각각 +3.293/+1.925/+3.200pp 앞섰다. `regular`의 matched 7-class lead +3.8865pp가 shift_2에서는 +1.9245pp로 줄었다.
- **하락폭이 최소는 아니다.** gating을 뺀 구성이 가장 평탄하다(+0.04). 주장은 "shift 하 최고 절대 성능"이다. tail calibration은 하락을 줄였지만(−1.86 대 −2.83, 차이 0.97pp) 경계선 수준이다.
- **고정 임계값 audit(사전 등록)은 실패했다.** 별도 정상 16장으로 source FPR 1% 임계값을 고정했을 때, Ours는 gating 제거 대비 shift PRO가 +8.93pp 높았지만 FPR 증가는 +0.73pp 더 컸고, FPR이 나빠지지 않은 클래스는 4/8이었다(기준 6/8). 따라서 **고정 임계값에서의 FPR 안정성과 nuisance 제거는 주장하지 않는다.**
- **공식 서버(private → private-mixed)**: 표의 "Ours"는 **RSG3가 아니라** DINOv3-H+ 기반의 이전 파이프라인(Baseline1-core)이다. 상대 Seg-F1 하락은 −4.11%로 가장 작았지만 절대 Seg-F1(45.94)도 가장 낮고, backbone과 정상 데이터 예산이 행마다 다르다. RSG3로 다시 제출하기 전까지는 참고 자료로만 쓴다.

### 4.5. Condition B-2 — Synthetic Corruptions and the Query-Side Check

#### 4.5.1. Cross-Method Corruption Comparison (MVTec AD, 16-shot seed 0, AUPRO@0.05)

모든 방법이 같은 깨끗한 support 16장, DINOv2-L, evaluator를 쓴다. query만 corruption되고, target adaptation은 없다.

| Method                       |           Clean |      Brightness |        Contrast |    Defocus blur |  Gaussian noise |    Mean shifted |   Shift − clean |
| :--------------------------- | --------------: | --------------: | --------------: | --------------: | --------------: | --------------: | ---------------: |
| RegAD                        |           38.11 |           36.14 |           32.01 |           37.33 |           28.12 |           33.40 |           −4.71 |
| RD4AD                        |           76.42 |           70.95 |           71.95 |           72.01 |           40.41 |           63.83 |          −12.59 |
| PatchCore                    |           83.97 |           83.43 |           77.15 |           81.58 | **74.03** |           79.05 |           −4.92 |
| FastRecon                    | **85.74** |           77.53 |           69.72 | **82.35** |           57.70 |           71.82 |          −13.91 |
| RN-FMLK                      |           74.53 |           68.65 |           69.50 |           70.99 |           46.56 |           63.93 |          −10.60 |
| SuperAD                      |           84.49 |           83.68 |           82.27 |           82.17 |           71.10 | **79.81** | **−4.69** |
| RSG3                         |           85.36 | **84.42** | **83.15** |           81.92 |           64.67 |           78.54 |           −6.82 |
| **RSG3 + query check** |           85.36 | **84.42** | **83.15** |           81.92 |  69.01*(dev)* |           79.63 |           −5.73 |

- 광도 변화(brightness, contrast)에서 1위다. check는 이 조건들에서 한 번도 작동하지 않으므로 해당 열은 RSG3와 같다.
- Gaussian noise에서 RSG3는 20.69pp 떨어졌고, 같은 backbone의 kNN 방법(PatchCore −9.94, SuperAD −13.39)보다 훨씬 컸다. **실패는 kNN 일반 현상이 아니라 우리 파이프라인 특유의 문제다.**
- check를 켜면 4종 평균이 78.54(3위)에서 **79.63(2위)**이 되어 SuperAD(79.81)와의 차이가 0.18pp로 줄어든다. Gaussian 열은 개발 조건이다.

#### 4.5.2. Attribution of the Noise Failure (120 cells, 사전 등록)

각 arm은 구성 요소 하나만 바꾼다. control은 저장된 결과를 모든 클래스에서 정확히 재현했다. signed 하락 = noise − clean, 완화폭 = 하락(arm) − 하락(control)이며, 3.0pp 이상이면 이 ablation의 등록된 기여 기준을 충족한다.

| Arm                             | Clean | Gaussian s3 |    Drop |          완화폭 | Clean 비용 | 판정                |
| :------------------------------ | ----: | ----------: | ------: | --------------: | ---------: | :------------------ |
| Control                         | 85.36 |       64.67 | −20.69 |              — |         — | —                  |
| layer 5 제거                    | 85.12 |       66.08 | −19.04 |           +1.65 |     −0.24 | 기준 미달           |
| **저해상도 [448, 672]만** | 84.61 |       69.01 | −15.60 | **+5.09** |     −0.75 | **기준 충족** |
| CQ 제거                         | 85.83 |       64.45 | −21.38 |          −0.69 |      +0.47 | 기준 미달           |

- **P4 지지 범위**: 고해상도 가지를 제외하면 이 Gaussian s3 조건의 하락이 5.09pp 완화된다. 이 단일 ablation은 고해상도 경로의 기여를 보이지만, support 쪽 원인을 모두 배제하지는 않는다. E5에서 q95를 MAD로 바꾸면 clean 85.30943386, noise 61.43696840, signed drop −23.87246546pp로, control의 85.3614/64.6741(약 −20.687pp)보다 하락이 약 3.19pp 커졌다. 대체가 완화 기준에 미달했다는 결과로만 해석한다.
- `bottle`과 `zipper`는 모든 arm에서 무너진다(노이즈 AUPRO 약 30, 약 9). SuperAD도 같이 무너지고(31.1, 19.5), PatchCore만 버틴다(61.3, 30.3). PatchCore의 3×3 이웃 feature 평균이 원인일 가능성이 있으나 검증하지 않았다.
- 저해상도를 **모든 query에 적용**하면 clean에서 −0.75pp를 잃는다. 그래서 노이즈가 감지된 query에만 적용하는 check가 필요하다.

#### 4.5.3. The Query-Side Check: Development vs. Held-Out

대응 동작은 원인 분리 결과를 보기 전에 정한 규칙으로 선택되었다.

| Condition         | 역할           |  RSG3 | + query check |             Gain |      작동 |
| :---------------- | :------------- | ----: | ------------: | ---------------: | --------: |
| Gaussian noise s3 | **개발** | 64.67 |         69.01 |            +4.34 | 1725/1725 |
| Shot noise s3     | held-out       | 56.00 |         65.10 |  **+9.10** | 1725/1725 |
| Impulse noise s3  | held-out       | 54.02 |         58.67 |  **+4.65** | 1725/1725 |
| Speckle noise s3  | held-out       | 47.55 |         60.56 | **+13.01** | 1725/1725 |
| Gaussian noise s1 | held-out       | 75.06 |         75.86 |            +0.80 | 1725/1725 |
| Gaussian noise s5 | held-out       | 36.34 |         48.49 | **+12.16** | 1725/1725 |

- **P5 지지 범위**: 설계에 쓰지 않은 MVTec AD 노이즈 5조건 중 4개에서 경험적 0.86pp 기준을 넘고 평균 **+7.94pp**다. 평균이 개발 조건 이득보다 크다는 사실만으로 과적합이 없다고 판정할 수는 없다.
- 이득은 노이즈가 셀수록 커진다(s1 +0.80 → s3 +4.34 → s5 +12.16). 약한 노이즈에서는 고해상도 가지가 아직 신호를 담고 있어 버리는 손해와 얻는 이득이 비슷하다.
- **미발화 범위**: 열거된 점검 **18,431회에서 0회**(MVTec AD 2 `test_public`을 서로 다른 support로 반복한 횟수 포함). 이 수는 brightness s5를 포함한 모든 비노이즈 조건이 아니다. brightness s5에서는 3/1725건 발화했다. 같은 GPU의 MVTec AD clean 15/15 클래스와 별도 VisA clean 12/12 클래스에서 저장된 지표가 같았고, VisA clean은 0/2162회 발화했다.
- **VisA E4 진행 상황**: Gaussian s3 full12에서 guard의 AUPRO 이득은 +4.75956pp이다. 사전 등록된 shot/impulse/speckle 조건이 완료되기 전에는 VisA의 전체 P 기준을 판정하지 않는다.
- **한계**: Gaussian s3에서 여전히 SuperAD(−2.10)와 PatchCore(−5.02)보다 낮다. `bottle` −2.55, `zipper` −0.69로 소폭 나빠진다(두 클래스는 check 유무와 관계없이 원래 무너져 있다). brightness s5에서 3/1725 오작동한다(`pill` 2, `toothbrush` 1). 모두 seed 0이다.

### 4.6. Backbone Swap (Negative Result)

같은 RSG3를 DINOv3-L/16으로 바꿨다(MVTec AD 2, 16-shot, 8개 클래스).

| Backbone    |      AUPRO@0.05 |     pAUROC@0.05 |       F1-oracle |
| :---------- | --------------: | --------------: | --------------: |
| DINOv2-L/14 | **59.15** |           77.11 |           35.87 |
| DINOv3-L/16 |           55.87 | **82.36** | **45.54** |

- DINOv3는 AUPRO에서 **3.28pp 낮고**, pAUROC와 F1에서는 높다. 지표 사이에서 방향이 갈린다.
- 저샷에서는 AUPRO 차이가 더 크다(1/2/4/8-shot: 40.96 / 44.21 / 51.04 / 56.60, DINOv2는 52.49 / 53.31 / 57.33 / 60.04).
- 이전 문서의 "DINOv3 66.62%"는 8개 중 쉬운 4개 클래스의 평균이었다. 나머지 4개는 45.11%다. **"backbone을 키우면 좋아진다"는 주장은 하지 않는다.**

### 4.7. A Rejected Hypothesis: A Shared Latent State

v6까지는 두 조건이 하나의 잠재 상태("coverage void": 정상 query가 support 관측 범위 밖에 놓인 정도)로 수렴한다고 가정했다. 이를 정상 pixel의 q99 gate 발화율과 그 사전 기댓값(1%)의 비율 $V_5$로 계측했다(사전 등록, 120 cells).

- **확인된 것**: 보정은 설계대로 동작한다. 16-shot `regular`에서 $V_5=0.994$(예측 1.000). 획득 조건 shift는 $V_5$를 키우고(H2 PASS), gate 선택도는 $V_5$와 함께 떨어진다(H3 PASS).
- **기각 근거**: $V_5$는 support가 **늘수록** 커졌다(2/4/8/16-shot: 0.64 / 0.74 / 0.95 / 1.00, §3.4의 기준 드리프트). 두 축 어디에서도 성능 손실을 설명하지 못했다($R^2$ 0.097, 0.016, 기울기 부호 반대). gating 이득은 $V_5$와 함께 줄었다(H5 FAIL).
- H4는 등록된 규칙상 통과했지만 판정 능력이 없었다. 규칙에 "각 축의 적합이 유의미할 것"이라는 전제가 빠져 있었다.
- **결론**: 측정한 $V_5$를 두 조건의 공통 잠재 상태로 채택하지 않는다. 다른 잠재 설명을 모두 기각한 것은 아니다. v7은 정상 기준 추정과 query 획득 변화의 구분을 설계 틀로 사용한다.

### 4.8. Evidence Boundary

- **가장 강한 성능 근거**: §4.3.2와 E2의 같은-support gating 비교, E10의 support-hash 확인 shift 비교, §4.5.2–4.5.3의 Gaussian ablation과 별도 MVTec AD 노이즈 조건. 이들은 구성 요소의 배타적 인과 귀속을 증명하지 않는다.
- **보조 근거**: §4.2(5-seed 재현), §4.3.1(외부 비교), §4.4(통제된 shift), §4.5.1(방법 간 corruption 비교).
- **실패와 음성 결과**: §4.4의 고정 임계값 audit, §4.6의 backbone 교체, §4.7의 기각된 가설.
- **미완 항목**: E1의 support-split seed 반복, E4의 VisA shot/impulse/speckle 노이즈 조건, RSG3의 공식 서버 제출, `bottle`/`zipper` 노이즈 실패, 1-shot 추정량과 support 수의 영향 분리, sealed final holdout.

---

## 5. Discussion & Conclusion

### 5.1. Calibration and Checks vs. Capacity and Adaptation

- **초기 학습·적응 시도는 이 설정에서 낮았다.** 같은 16-shot 조건에서 reverse distillation(32.2%), masked contextual prediction(14.2%), contextual scoring(5.7~15.0%), synthetic boundary learning(47.6%), test-time shift restoration(31.9%)은 단순 nearest-neighbour baseline(52.5%)보다 낮았다. 이 결과로 방법군 전체의 공통 실패 원인을 단정하지 않는다.
- **측정된 score 구성의 효과**: DINOv3 교체는 이 설정의 AUPRO를 높이지 않았고(§4.6), 같은 backbone의 fusion ablation에서는 저샷에서 7~11pp 차이가 났다(§4.3.2). 이를 모든 backbone·용량 선택에 일반화하지 않는다.
- **query를 점검하는 것이 현실적이다.** 실제 배치에서 support는 고정되지만 query는 계속 달라진다. support에 흔적을 남기지 않는 변화는 support 통계로 볼 수 없으므로, support 기준에 비춘 query 점검이 필요하다. 이 점검은 학습·적응 없이, query를 한 장씩 독립적으로, 판단 근거를 남기며 수행할 수 있다.

### 5.2. Boundary of Claims and Future Work

- **주장 범위**:
  - 이득은 AUPRO에 집중된다. pAUROC와 oracle F1에서는 최고가 아니다.
  - shift에 대한 주장은 "절대 성능이 높다"이다. 하락폭 최소, 고정 임계값 FPR 안정성, 보편적 shift 강건성은 주장하지 않는다.
  - query-side check는 **측정 가능한 query 쪽 변화(센서 노이즈) 하나**를 다룬다. 자세 변화나 새로운 정상 패턴 같은 다른 shift는 이 통계로 감지되지 않는다. "query 점검"이라는 원칙의 한 사례로 제시한다.
  - check는 Gaussian noise 실패를 보고 설계했다. 설계에 쓰지 않은 MVTec AD 노이즈 조건에서는 이득을 관측했고, VisA clean과 Gaussian s3에서는 각각 0회 발화와 +4.76pp를 확인했다. VisA의 다른 노이즈 조건은 진행 중이다.
  - brightness s5의 3/1725 query에서는 노이즈가 없어도 check가 발화했다. 따라서 비노이즈 전체에서 무발화 또는 바이트 동일성을 주장하지 않는다.
  - 대부분 seed 0이고, 주 구성만 5-seed로 반복했다. sealed final holdout 평가는 하지 않았다.
  - multi-resolution, multi-bank, CQ의 독립 기여는 주장하지 않는다. 임의의 3D 회전이나 시점 변화를 동반하는 category shift는 범위 밖이다.
- **향후 과제**:
  - shot 축 실험의 seed 반복(seed 1~4)
  - RSG3(+check)의 공식 서버 제출
  - 다른 데이터셋(VisA)에서의 계획된 노이즈 평가
  - `bottle`/`zipper`의 노이즈 실패(이웃 feature pooling 검토)
  - 밝기 변화와 노이즈를 구분하는 더 정교한 query 통계
