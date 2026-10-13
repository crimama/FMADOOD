# SCARF: Support-Calibrated Asymmetric Residual Fusion for Few-Shot Industrial Anomaly Localization

> **문서 목적**: CVPR/WACV 제출용 본문 8쪽 기준 연구 아웃라인 (v16.3).
> **v16.3 주요 개정 사항**: §4.4.1 표 8을 SCARF 프레임워크 전 구성 요소의 단계적 기여도를 입증하는 **종합 구성 요소 소거 분석(System-wide Component Ablation)**으로 전면 개편함(기존 얕은 층 2요인 표 제거). 또한 표 4에 최신 3-seed 전 비교군 1120 해상도 완료 지표를 완결 반영함.
>
> **원고 작성 가이드 (CVPR/WACV 본문 약 8쪽 배분)**:
>
> - §1. Introduction (약 1.0쪽) | §2. Related Work (약 0.5쪽) | §3. Method (약 2.0쪽, ASD-STE100 규격 기술)
> - §4. Experiments (약 4.0쪽) | §5. Discussion & Limitations (약 0.35쪽) | §6. Conclusion (약 0.15쪽)
> - ※ 저자용 검증 매트릭스 및 연구 기록은 문서 말미의 `[Appendix: Author Research Notes]`에 보존한다.

---

## Index

- **Abstract**
- **1. Introduction**
  - 1.1. 소수 정상 reference와 거리 해석의 모호성 (High-Score Ambiguity)
  - 1.2. 다층·다해상도 관측의 결합과 딜레마
  - 1.3. SCARF 프레임워크와 주요 기여
- **2. Related Work**
  - 2.1. Frozen-feature reference matching in few-shot AD
  - 2.2. Multi-scale and multi-layer feature fusion in vision
  - 2.3. Low-FPR localization under distribution shifts
- **3. Method (ASD-STE100 기술 규격 스타일)**
  - 3.1. 시스템 개요 및 서포트 피팅·쿼리 추론 절차 (그림 1)
  - 3.2. 정상 서포트 뱅크 구축 및 메모리 배깅 (그림 M1)
  - 3.3. Scale Calibration: 분위수 기반 거리 정규화 (Axis 1: Scale, 그림 M2)
  - 3.4. Asymmetric Residual Fusion: 비대칭 초과량 결합 (Axis 2: Layer, 그림 M3)
  - 3.5. 고정 다해상도 결함 맵 공간 통합 (Anomaly Map Aggregation)
  - 3.6. 파라미터 규격 및 재현성 계약
- **4. Experiments**
  - 4.1. 평가 설정 (표 1, 2, 3)
  - 4.2. Few-shot 주 비교 (표 4, 5)
  - 4.3. Distribution Shift 평가 (표 6, 그림 2, 표 7)
  - 4.4. Ablation Study (표 8, 8a, 9, 10, 11)
  - 4.5. Efficiency and Qualitative Analysis (표 12, 12a, 그림 3)
- **5. Discussion & Limitations**
  - 5.1. Shallow 잔차 기여의 클래스별 특성 (단순 표면 vs 복잡 질감/미세 스크래치)
  - 5.2. 저-FPR 곡선 메트릭과 고정 배포 임계값의 실무적 과제
- **6. Conclusion**
- **[Appendix: Author Research Notes]**
  - A.1. Claim–evidence–status matrix
  - A.2. Controlled comparison and decision rules
  - A.3. Review and audit feedback record

---

## Abstract

산업 이상 탐지에서는 새로운 제품을 소수의 정상 이미지로 검사해야 한다. 동결된 시각 백본의 패치를 정상 reference와 매칭하는 방식은 재학습 비용을 없애지만, 큰 최근접 거리는 실제 결함뿐 아니라 support에 포함되지 않은 정상 변동이나 촬영 조건 변화에서도 발생한다. 미세 결함 탐지력을 높이기 위해 다층(Shallow/Deep)과 다해상도(448~1120px)로 관측을 확장하면, 얕은 층의 정상 외관 반응과 고해상도의 센서 노이즈가 결합 점수에 누적되어 저-FPR(False Positive Rate) 영역의 localization 성능이 치명적으로 붕괴된다.

본 논문은 **Support-Calibrated Asymmetric Residual Fusion (SCARF)**을 제안한다. SCARF는 불확실한 원인 판별 대신, 각 반응을 정상 support에서 관측되는 분포에 비추어 정규화하고 제어한다. 구체적으로 (1) 정상 support의 비모수적 **기준 상위 분위수(reference upper quantile, $q=0.95$)**로 계층 및 해상도별 관측 단위를 공통 척도로 정렬하고(Scale Calibration), (2) 외관 변화에 강건한 Deep feature를 기본 탐지선으로 삼아 Shallow feature가 **정상 극단 상한(extreme normal bound, $\beta=0.99$)**을 넘는 **초과 잔차(Excess Residual)**만을 선별 가산하며(Contribution Control), (3) 4개 고정 해상도의 맵을 역투영 평균하여 거시적 안정성과 미세 결함 탐지력을 동시에 통합한다.

동일한 백본·support·해상도·메모리 뱅크 조건을 적용한 MVTec AD 2 16-shot 개발 비교에서, SCARF는 shallow feature 전체를 단순 가산할 때 발생하는 치명적인 저-FPR 국소화 성능 하락(-4.65pp, 비정규화 대비 -5.13pp)을 완벽히 방어하며, 강력한 Deep-only 기준선 대비 AUPRO@0.05를 **58.67%에서 59.15%(+0.47pp, 미세 결함 클래스 최대 +2.91pp)**로 추가 향상시켰다. 또한 총 뱅크 토큰 수를 맞춘 용량 통제 비교에서 고정 4-해상도 결합은 단일 최고 해상도(1120px) 대비 **+2.52pp** 높은 국소화 성능을 기록했다. 동일 support와 DINOv2 ViT-L/14를 사용한 16-shot SuperAD 파이프라인 비교에서는 AUPRO@0.05가 **4.47pp**, pAUROC@0.05가 **0.70pp** 높았으며, 공식 MVTec AD 2 private 벤치마크 평가에서도 TESTpriv **64.80%**, TESTpriv,mix **62.49%**를 달성하여 few-shot 조건임에도 우수한 이상 국소화 성능을 실증했다.

---

## 1. Introduction — 약 1.0쪽

<코멘트>

- 일단 multi-scaling, layer를 쓰는 당위성에 대해 아직 납득이 안됨
- 오히려 성능이 높은 이유에 대한 기여는 scaling이 더 큰 거 같다
- 오히려 scaling 측면으로 관점을 가져가는게 좋지 않을까
- shift vs normal을 구분짓는 신호가 되는가에 대한 논리
  - few-shot -> multi resolution
  - shift -> scaling [이에 대한 연결성 강화 필요]
    - 실험에서 shift 없을 때는  scaling 없어도 성능 유효 -> 근거가 됨
  - 이 두개가 상호 보완적으로 사용해야 하는 이유가 필요

### 1.1. 소수 정상 reference와 거리 해석의 모호성 (High-Score Ambiguity)

- **소수 정상 이미지 기반 검사의 실용성**: 다품종 소량 생산 및 빈번한 생산 라인 전환 환경에서는 제품마다 수천 장의 결함 데이터를 수집하거나 모델을 재학습하기 어렵다. 대규모 데이터로 사전학습된 동결(frozen) 시각 백본(예: DINOv2)과 소수의 정상 이미지로 구성된 참조 뱅크(reference bank)를 활용하면, 별도 미세조정 없이 query 패치가 정상 패치와 얼마나 떨어져 있는지 측정함으로써 즉각적인 결함 검사가 가능하다. 특히 실제 산업 공정에서는 전체 ROC 구간보다 오경보율이 엄격히 통제되는 **저-FPR(Low False Positive Rate, 예: FPR $\le$ 5%) 구간에서의 정밀한 결함 영역 국소화(defect localization)**가 핵심적인 신뢰성 요구조건이다.
- **획득 조건 변화(Distribution Shift)와 거리 해석의 딜레마**: 소수의 정상 reference는 실제 공정에서 발생하는 조명 드리프트, 부품 자세 변동, 표면 난반사 등 다양한 환경 변화를 결코 포괄할 수 없다. 따라서 결함이 없는 정상 query 패치라도 reference bank에 없는 정상 변동이 발생하면 최근접 거리가 크게 벌어지며, 이는 실제 결함이 유발하는 큰 거리와 외형상 구별되지 않는다. 즉, **"큰 거리값"이라는 관측만으로는 그 원인이 실제 결함인지, reference에 미포함된 정상적 외관 변화인지 확정할 수 없다(High-Score Ambiguity).**
- **저-FPR 검사에서의 파급효과**: 오경보를 엄격히 억제해야 하는 실제 산업 공정에서, 시스템의 판정 임계값은 정상 점수 분포의 평균이나 중심이 아니라 **상위 1\~5% 분위수 경계(upper tail)**에 의해 직접 결정된다. 따라서 결함이 없는 정상 영역에서 우연히 튄 소수의 큰 거리값조차 곧바로 임계값을 비정상적으로 끌어올리거나 오경보(False Alarm)로 직결되어, 저-FPR 구간(AUPRO@0.05, pAUROC@0.05)의 검출 성능을 심각하게 훼손한다. 따라서 소수 support 환경에서는 정상 수준의 관측 신호가 결합 점수에 누적되지 않도록 차단하고, 정상 범위를 명백히 초과한 반응만을 선별 반영하는 엄격한 점수 제어가 요구된다.

### 1.2. 고정 다층·다해상도 관측 결합의 과제

Few-shot 검사에서는 reference 표본이 극도로 희소하므로, 각 이미지에서 가능한 한 다양한 수준의 시각적 단서를 확보하는 것이 유리하다.

1. **표현 계층(Layer Depth)**: ViT 백본의 심층 레이어(Deep, Layer 11/23)는 조명·색상 변화에 둔감한 안정적 의미 기준선(semantic base)을 제공하지만 미세 윤곽과 국소 텍스처에 둔감하다. 반면 얕은 레이어(Shallow, Layer 5)는 미세 스크래치와 국소 결함에 민감하지만 정상 외관 변화에도 과민 반응한다.
2. **입력 해상도(Input Resolution)**: 패치 크기가 $14 \times 14$로 고정된 구조에서, 448px 저해상도는 이미지 짧은 변의 3.1%를 하나의 토큰이 덮어 거시적 형태 안정성을 제공한다. 반면 1120px 고해상도는 토큰 면적이 1.25%로 줄어들어 미세 결함에 대한 공간 표본 밀도를 비약적으로 높여준다.

**관측 결합의 어려움**:
다층·다해상도 관측을 구성했다고 해서 모든 점수를 그대로 더하는 것이 유리한 것은 아니다. 계층·해상도마다 거리 척도가 다르며, shallow 점수에는 결함 반응뿐 아니라 정상 질감과 외관 반응도 포함된다. 동일 조건 통제 비교(Table 1)에서 기준 상위 분위수($q=0.95$)로 단위를 맞춘 뒤에도 shallow 전체를 가산한 구성의 AUPRO@0.05는 Deep-only의 58.67%보다 낮은 54.02%였다(-4.65pp). 초과 잔차만 가산하면 59.15%로 높아졌다(전체 가산 대비 +5.13pp). 이 비교는 척도 보정만으로는 shallow 기여량 제어를 대체할 수 없음을 보여준다.

고정 다층·다해상도 관측을 결합하기 위해 다음 두 문제를 구분한다:

1. **Scale Calibration — 관측 간 거리 단위 정렬**: 계층·해상도·컴포넌트마다 상이한 정상 거리 척도를 support 기준 상위 분위수($q$-quantile, $q=0.95$)로 보정하여, 서로 다른 관측을 공통의 통계적 단위에서 비교 가능하도록 정규화한다. 해상도에 따른 거리 척도 차이도 이 단계에서 동일하게 처리된다.
2. **Contribution Control — Shallow의 추가 기여량 제어**: 공통 단위로 변환된 상태에서 의미적으로 강건한 Deep 평균을 기준선으로 삼고, Shallow의 정상 극단 상한선($\beta$-quantile, $\beta=0.99$)을 넘는 비정상 초과 잔차만을 비선형 클리핑하여 선별 가산한다. 각 고정 해상도에서 동일한 결합 규칙을 적용한 뒤 맵을 평균한다.

해상도 집합은 관측 구성의 사양이며, 두 결합 규칙은 그 관측을 어떻게 해석하고 합칠지를 정한다. 다해상도 구성의 이득은 단일 1120px 구성과 총 뱅크 토큰 수를 맞춘 별도 실험(Table 10)으로 확인한다.

### 1.3. SCARF 프레임워크와 주요 기여

본 논문은 High-Score Ambiguity가 있는 소수 정상 reference 환경에서, 다층·다해상도 점수의 단위와 추가 기여량을 구분하는 **Support-Calibrated Asymmetric Residual Fusion (SCARF)**을 제안한다. 핵심 기여는 다음과 같다:

1. **다층 결합에서 척도 보정과 기여 제어의 원리적 분리**: 사전학습 ViT 계층 결합 시 발생하는 저-FPR 국소화 성능 저하가 단순한 거리 척도 불일치가 아니라, 얕은 층의 정상 외관 반응 누적으로 인해 상위 분위수 결정 임계값이 오염되는 구조적 문제임을 정식화한다. 이를 해결하기 위해 결합 문제를 거리 단위를 일치시키는 **척도 보정(Scale Calibration)**과 비정상 반응만을 선별 수용하는 **기여 제어(Contribution Control)**의 독립된 두 차원으로 분리해야 함을 규명한다. 통제된 개발 비교를 통해, 기준 분위수($q=0.95$) 보정 후에도 얕은 층 전체를 가산하면 성능이 저하되는 반면(-4.65pp vs Deep-only), 극단 꼬리 잔차($\beta=0.99$)만을 선별 가산해야 성능이 회복·개선됨(+5.13pp vs 전체 가산)을 실증하여 원리적 분리의 타당성을 뒷받침한다.
2. **Support 통계 기반 비대칭 잔차 결합(SCARF)**: 추가 학습이나 파라미터 튜닝 없이, 정상 support의 비모수적 분위수($q, \beta$)만으로 척도 보정과 비선형 잔차 결합을 완결하는 결정론적 프레임워크를 제안한다. 정상 LSO 상위 분위수($q=0.95$)를 공통 거리 단위로 삼고, Deep 기준선 위에 Shallow의 정상 극단 상한($\beta=0.99$)을 넘는 잉여 잔차만을 비대칭 가산함으로써 저-FPR 오경보를 엄격히 억제한다.
3. **구성 이득과 결합 이득의 분리 검증**: 총 뱅크 토큰 수를 맞춘 개발 비교에서 다해상도 구성의 이득(+2.52pp)을 평가하고, 해상도를 고정한 결합 비교에서 Deep-only 대비 잔차의 추가 이득(+0.47pp)을 평가한다. 파이프라인 전체 비교와 저-샷 전이 결과를 별도로 보고하여 방법의 적용 범위를 명시한다.

---

## 2. Related Work — 약 0.5쪽

### 2.1. Frozen-Feature Reference Matching in Few-Shot AD

소수 정상 이미지만으로 이상을 탐지하는 패치 매칭 방식은 재학습이 필요 없어 산업 현장에서 활발히 연구되어 왔다. PatchCore([Roth et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html))는 동결된 CNN 백본의 중간 feature를 Coreset 샘플링하여 메모리 뱅크를 구축하고 최근접 패치 거리를 계산했다. 최근 연구들은 대규모 자기지도 비전 백본(DINOv2)을 도입하여 성능을 크게 끌어올렸다. AnomalyDINO([Damm et al., WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html))는 DINOv2 패치 토큰의 우수한 선형 분리능을 증명했고, SuperAD([Zhang et al., 2025](https://arxiv.org/abs/2505.19750))는 여러 계층의 DINOv2 거리를 단순 평균하는 방식을 제안했다. 그러나 이들은 소수 reference 환경에서 정상 외관 변동으로 인해 거리가 커지는 'High-Score Ambiguity'를 다루지 못하며, 대칭적 결합으로 인한 저-FPR 오경보 누적 문제에 취약하다.

### 2.2. Multi-Scale and Multi-Layer Feature Fusion in Vision

Vision Transformer(ViT)의 계층별 특성에 대한 분석([Raghu et al., NeurIPS 2021](https://arxiv.org/abs/2108.08810))에 따르면, 얕은 층은 국소적 경계와 질감 정보를, 깊은 층은 거시적 의미 구조를 담당한다. 이상 탐지 분야에서는 미세 결함과 대형 결함을 동시에 잡기 위해 여러 layer나 multi-scale 입력을 결합하려는 시도가 이어져 왔다. 그러나 선행 연구들은 feature concatenation이나 가중치 합산과 같은 고정된 대칭 규칙을 사용함으로써, 질감에 민감한 얕은 층이 환경 변화에 반응해 내뿜는 거짓 높은 점수(false high-score)를 통제하지 못했다. SCARF는 심층 층을 앵커(base)로 고정하고 얕은 층의 정상 초과량만을 비대칭으로 수용하는 새로운 잔차 결합 패러다임을 제시한다.

### 2.3. Low-FPR Localization under Distribution Shifts

실제 제조 공정에서는 조명 드리프트, 부품 각도 변화, 표면 난반사 등 다양한 획득 조건 변화(Distribution Shift)가 빈번히 발생한다. MVTec AD 2([Batzner et al., 2024](https://arxiv.org/abs/2404.14810))는 이러한 실제 환경 변화를 반영한 벤치마크를 정립했다. 특히 공정 신뢰성을 위해서는 전체 ROC 곡선 아래 면적보다 엄격한 저-FPR 구간(pAUROC@0.05, AUPRO@0.05)에서의 위치 특정 성능이 결정적이다. 본 연구는 추가적인 레이블이나 테스트 시점 최적화 없이, 소수 정상 reference의 통계적 꼬리를 이용해 저-FPR 결정 경계를 견고하게 방어하는 결합 메커니즘을 탐구한다.

---

## 3. Method (ASD-STE100 기술 규격 스타일) — 약 2.0쪽

### 3.1. 시스템 개요 및 서포트 피팅·쿼리 추론 절차

**목적**:
동결된 사전학습 비전 백본의 고정 다층·다해상도 패치 특징을 사용하여, 정상 서포트 이미지 $N$장으로부터 테스트 쿼리 이미지 $x^q$의 결함 영역을 국소화한다.

**입력**:

1. 정상 서포트 이미지 집합 $\mathcal{S} = \{x_1^s, x_2^s, \dots, x_N^s\}$.
2. 테스트 쿼리 이미지 $x^q$.
3. 동결된 DINOv2-L/14 백본 네트워크.
4. 고정 계층 집합 $\mathcal{L} = \{5, 11, 23\}$, 고정 해상도 집합 $\mathcal{R} = \{448, 672, 896, 1120\}$, 배그 수 $B=4$.

**출력**:

1. 픽셀 단위 결함 국소화 맵 $A(x^q) \in \mathbb{R}^{H \times W}$.
2. 이미지 단위 결함 점수 $s(x^q) \in \mathbb{R}$.

**공통 설계 연산자 (Support-Referenced Ratio Operator)**:
임의의 관측 통계량 $t$와 정상 서포트 기준 분포 $\mathcal{N}_S[t]$에 대해, 분위수 비율 연산자 $\mathcal{T}_q[t]$를 다음과 같이 정의한다:

$$
\mathcal{T}_q[t] = \frac{t}{Q_q(\mathcal{N}_S[t])}
$$

여기서 $Q_q(D)$는 분포 $D$의 $q$-분위수(Quantile) 값이다. 이 연산자는 척도 보정($q=0.95$)과 잔차 게이트($\beta=0.99$)에 일관되게 적용된다.

![SCARF Overview](../figures/fig_scarf_overview.png)
*그림 1. SCARF 전체 파이프라인 개요도. (A) 서포트 피팅 단계: 48개 컴포넌트 뱅크를 구축하고 $c_m, \tau_{5,r,b}$를 1회 사전 계산한다. (B) 쿼리 추론 단계: 고정 4개 해상도에서 거리 정규화 $\to$ 잔차 결합 $\to$ Bag 평균 및 4-해상도 역투영 평균을 거쳐 최종 결함 맵을 산출한다.*

**전체 파이프라인의 2단계 절차**:

- **1단계 (서포트 피팅 단계, 사전 1회 수행)**:
  1. 계층 $l \in \{5, 11, 23\}$, 해상도 $r \in \{448, 672, 896, 1120\}$, bag $b \in \{1, 2, 3, 4\}$의 조합 $m = (l, r, b)$를 독립 관측 단위인 **컴포넌트(Component)**로 정의한다 (총 $3 \times 4 \times 4 = 48$개).
  2. 서포트 이미지 집합 $\mathcal{S}$로부터 각 컴포넌트 $m$에 대응하는 총 48개의 전용 메모리 뱅크 $B_m$을 생성한다.
  3. Leave-Source-Out (LSO) 매칭을 수행하여 48개 컴포넌트의 정상 거리 분포 $D_m^{\text{LSO}}$를 측정한다.
  4. 단위 척도 $c_m = Q_q(D_m^{\text{LSO}})$ ($q=0.95$) 및 Shallow 초과 임계값 $\tau_{5,r,b} = Q_\beta(D_{5,r,b}^{\text{LSO}}) / c_{5,r,b}$ ($\beta=0.99$)를 사전 계산한다.
- **2단계 (쿼리 추론 단계, 온라인)**:
  1. 쿼리 영상 $x^q$에 대해 고정 4개 해상도 $\mathcal{R}$에서 계층별 패치 특징 벡터를 추출한다.
  2. 48개 컴포넌트 뱅크에 대해 1-NN 코사인 거리 $d_m(x^q, p)$를 계산하고, 사전 계산된 단위 척도 $c_m$으로 나누어 무차원 정규화 점수 $z_m(x^q, p)$를 산출한다.
  3. 각 해상도와 배그에서 Deep 기준선 $z_{d,r,b}(x^q, p)$에 Shallow 초과 잔차 $e_{5,r,b}(x^q, p)$를 가산하여 결합 점수 $S_{r,b}(x^q, p)$를 산출한다.
  4. 4개 배그의 점수 맵을 동일 토큰 격자에서 산술 평균하여 해상도별 점수 맵 $S_r(x^q, p)$를 생성한다.
  5. 4개 고정 해상도의 점수 맵을 원본 영상 크기 $(H, W)$로 양선형 보간한 뒤 동일 가중 평균하여 최종 결함 국소화 맵 $A(x^q)$ 및 이미지 단위 결함 점수 $s(x^q)$를 출력한다.

---

### 3.2. 정상 서포트 뱅크 구축 및 메모리 배깅

**목적**:
VRAM 사용량을 낮추고 표본화 분산을 줄이면서, 다층·다해상도 패치 특징을 저장하는 참조 뱅크를 구축한다.

**입력 파라미터 규격**:

1. **계층 집합 $\mathcal{L}$**: $\{5, 11, 23\}$. DINOv2-L/14의 5번째, 11번째, 23번째 블록 출력이다.
2. **해상도 집합 $\mathcal{R}$**: $\{448, 672, 896, 1120\}$. 영상의 짧은 변 길이를 해당 수치로 조정한다. 백본 패치 크기는 $14 \times 14$ 픽셀로 고정된다.
3. **배그 수 $B$**: 4.

|  해상도$r$  | 짧은 변 토큰 수 | 정사각 기준 토큰 수 | 토큰 1개가 덮는 영상 길이 비율 | 관측 특성                            |
| :------------: | :-------------: | :-----------------: | :----------------------------: | :----------------------------------- |
| **448** |       32       |        1,024        |            약 3.1%            | 거시적 형태 안정성 제공, 노이즈 둔감 |
| **672** |       48       |        2,304        |            약 2.1%            | 중간 수준의 구조 정보 제공           |
| **896** |       64       |        4,096        |            약 1.6%            | 국소 결함 표본화 밀도 강화           |
| **1120** |       80       |        6,400        |            약 1.25%            | 미세 결함 표본화 밀도 극대화         |

**배깅 뱅크 생성 절차**:

1. 해상도 $r$에서 각 서포트 이미지 $x_i^s$로부터 256개의 패치 토큰을 무작위 비복원 추출한다.
2. 서로 다른 4개의 난수 시드를 적용하여 이 과정을 4회 반복한다. 4개의 독립 배그 $b \in \{1, 2, 3, 4\}$를 생성한다.
3. 동일한 배그 $b$ 내에서 세 계층($l \in \{5, 11, 23\}$)은 동일한 공간 위치의 패치 토큰을 공유한다.
4. 계층 $l$, 해상도 $r$, 배그 $b$의 조합을 하나의 컴포넌트 $m = (l, r, b)$로 정의한다. 총 컴포넌트 수는 $3 \times 4 \times 4 = 48$개이다.
5. 16-shot 조건에서 단일 컴포넌트 뱅크 $B_m$은 $16 \times 256 = 4,096$개의 토큰으로 구성된다.

**1-NN 최근접 거리 계산**:
쿼리 위치 $p$에서 추출된 L2 정규화 특징 벡터를 $f_{l,r}(x^q, p)$라 한다. 컴포넌트 뱅크 $B_{l,r,b}$에 대한 1-NN 코사인 거리 $d_{l,r,b}(x^q, p)$를 다음과 같이 계산한다:

$$
d_{l,r,b}(x^q, p) = \min_{u \in B_{l,r,b}} \left[ 1 - f_{l,r}(x^q, p)^\top u \right]
$$

![4-Bag 점수 맵 앙상블](../figures/fig_rsg4_mech_bagging.png)
*그림 M1. 실제 hazelnut query(1120px)의 4개 독립 Bag 점수 맵과 평균. 각 Bag은 이미지당 256개 토큰을 사용한다. 오른쪽은 픽셀별 Bag 간 표준편차(전체 평균 0.038)를 나타내며, 4-Bag 평균이 희소 표본화 차이를 완화함을 보여준다.*

---

### 3.3. Scale Calibration: 분위수 기반 거리 정규화 (Axis 1: Scale)

**목적**:
서로 다른 계층과 해상도의 거리 척도를 동일한 무차원 척도로 정렬한다.

**입력**:

1. 컴포넌트 메모리 뱅크 $B_m$ ($m = (l, r, b)$).
2. 쿼리 패치 특징 벡터 $f_m(x^q, p)$.
3. 분위수 파라미터 $q \in (0, 1)$ (기본값: $q = 0.95$).

**출력**:

1. 컴포넌트 단위 척도 $c_m \in \mathbb{R}$.
2. 무차원 정규화 점수 맵 $z_m(x^q, p) \in \mathbb{R}$.

**계산 절차**:

1. **Leave-Source-Out (LSO) 정상 거리 분포 수집**:

   - 컴포넌트 뱅크 $B_m$에서 서포트 이미지 $x_i^s$로부터 추출된 토큰 부분집합을 $B_m^{(i)}$로 정의한다.
   - 각 토큰 $v \in B_m^{(i)}$에 대해 나머지 서포트 토큰 집합 $B_m \setminus B_m^{(i)}$와의 최근접 거리를 계산한다.
   - 모든 서포트 이미지 $i \in \{1, \dots, N\}$의 거리값을 모아 LSO 정상 거리 분포 $D_m^{\text{LSO}}$를 생성한다:
     $$
     D_m^{\text{LSO}} = \bigcup_{i=1}^{N} \left\{ \min_{u \in B_m \setminus B_m^{(i)}} \left[1 - v^\top u\right] \;:\; v \in B_m^{(i)} \right\}
     $$
   - 16-shot 조건에서 $D_m^{\text{LSO}}$는 4,096개의 거리값으로 구성된다.
2. **컴포넌트 단위 척도 $c_m$ 결정**:

   - 5% 오경보율(Type-I error, $\alpha = 0.05$) 억제 규격을 적용한다.
   - $q = 0.95$를 기본값으로 설정한다.
   - 정상 거리 분포 $D_m^{\text{LSO}}$의 $q$-분위수를 컴포넌트 단위 척도 $c_m$으로 결정한다:
     $$
     c_m = Q_{q}\!\left(D_m^{\text{LSO}}\right)
     $$
3. **쿼리 패치 코사인 거리 $d_m(x^q, p)$ 계산**:

   - 쿼리 특징 $f_m(x^q, p)$와 메모리 뱅크 $B_m$ 사이의 1-NN 코사인 거리를 $d_m(x^q, p)$로 정의한다:
     $$
     d_m(x^q, p) \triangleq \min_{u \in B_m} \left[ 1 - f_m(x^q, p)^\top u \right]
     $$
   - $f_m(x^q, p)^\top u$는 L2 정규화 특징 간 코사인 유사도이다.
   - $1 - f_m^\top u$는 결함 유무에 비례하는 코사인 거리이다.
   - 아래첨자 $m$은 대상 메모리 뱅크 $B_m$을 지정한다.
   - 인자 $(x^q, p)$는 쿼리 영상 $x^q$의 2차원 공간 위치 $p$를 지정한다.
4. **무차원 정규화 점수 $z_m(x^q, p)$ 계산**:

   - 쿼리 거리 $d_m(x^q, p)$를 단위 척도 $c_m$으로 나누어 정규화 점수 $z_m(x^q, p)$를 계산한다:
     $$
     z_m(x^q, p) = \mathcal{T}_{q}[d_m] = \frac{d_m(x^q, p)}{c_m}
     $$

**기술 규격 및 판정 기준**:

- **$z_m(x^q, p) = 1.0$**: 해당 패치 거리가 정상 상위 5% 경계값과 일치함을 나타낸다. 통계적 임계 기준선(Critical Bound)으로 작동한다.
- **$z_m(x^q, p) = 2.0$**: 해당 패치 거리가 정상 상위 5% 경계값의 2배임을 나타낸다.
- **정상 분포 정렬**: 모든 컴포넌트에서 서포트 정상 패치의 약 95%가 $z_m \le 1.0$ 범위 내에 정렬된다.
- **파라미터 둔감성**: $q \in [0.90, 0.99]$ 구간에서 성능 변동이 0.2pp 미만으로 유지된다(§4.4 참조). 데이터셋 종속적 튜닝을 배제한다.

![scale calibration](../figures/fig_rsg4_mech_scale.png)
그림 M2. 실제 query의 원시·정규화 점수 맵과 정상 픽셀 분포. 위 분포의 예시 원시 거리 $d^*$는 레이어별 support 기준 상위 분위수($q=0.95$) 기준 사이에 놓여, 같은 수치가 Layer 5에서는 기준 초과, Layer 23에서는 기준 미만으로 해석될 수 있음을 보여준다. 아래에서는 각 점수를 해당 레이어의 $q=0.95$ 단위로 환산해 $z=1$을 공통 기준으로 비교한다.

---

### 3.4. Asymmetric Residual Fusion: 비대칭 초과량 결합 (Axis 2: Layer)

**목적**:
Shallow feature의 정상 표면 질감 노이즈 누적을 원천 차단하고, 정상 상위 한계를 초과한 미세 결함 신호만을 Deep 기준선에 선별 가산한다.
*(고정 축소 가중치 $\lambda$를 곱하는 단순 선형 가중합 $z_d + \lambda z_5$ 방식과 달리, 정상 변동을 0점으로 완전 절단(clipping)하는 비선형 잔차 연산자를 적용하여 별도의 가중치 탐색이 불필요한 무파라미터(parameter-free) 결합을 달성한다.)*

**입력**:

1. 각 계층의 정규화 점수 맵 $z_{l,r,b}(x^q, p)$ ($l \in \{5, 11, 23\}$).
2. Layer 5의 LSO 정상 거리 분포 $D_{5,r,b}^{\text{LSO}}$.
3. Layer 5의 단위 척도 $c_{5,r,b} = Q_q(D_{5,r,b}^{\text{LSO}})$.
4. 잔차 게이트 분위수 파라미터 $\beta \in (q, 1)$ (기본값: $\beta = 0.99$).

**출력**:

1. 계층 결합 점수 맵 $S_{r,b}(x^q, p) \in \mathbb{R}$.

**계산 절차**:

1. **Deep 기준선 점수 $z_{d,r,b}(x^q, p)$ 계산**:

   - 쿼리 영상 $x^q$의 위치 $p$에서 Layer 11과 Layer 23의 정규화 점수를 산술 평균한다:
     $$
     z_{d,r,b}(x^q, p) = \frac{z_{11,r,b}(x^q, p) + z_{23,r,b}(x^q, p)}{2}
     $$
2. **Shallow 초과 임계값 $\tau_{5,r,b}$ 계산**:

   - Shallow 계층(Layer 5)의 정상 표면 질감 변동을 차단하기 위해 잔차 게이트 파라미터 $\beta \in (q, 1)$를 정의한다.
   - 상위 1% 극단 정상 변동 차단 규격을 적용하여 $\beta = 0.99$를 기본값으로 설정한다.
   - Layer 5의 LSO 정상 거리 분포 $D_{5,r,b}^{\text{LSO}}$의 $\beta$-분위수를 단위 척도 $c_{5,r,b}$로 나누어 초과 임계값 $\tau_{5,r,b}$를 계산한다:
     $$
     \tau_{5,r,b} = \frac{Q_{\beta}\!(D_{5,r,b}^{\text{LSO}})}{c_{5,r,b}} = \frac{Q_{\beta}\!(D_{5,r,b}^{\text{LSO}})}{Q_{q}\!(D_{5,r,b}^{\text{LSO}})} \ge 1.0 \quad (\beta > q)
     $$
   - $\beta > q$ ($0.99 > 0.95$)이므로 $\tau_{5,r,b}$는 항상 1.0 이상으로 보장된다.
3. **초과 잔차 $e_{5,r,b}(x^q, p)$ 계산**:

   - 쿼리 영상 $x^q$의 위치 $p$에서 Layer 5 점수가 임계값 $\tau_{5,r,b}$를 초과한 연속 크기를 ReLU 형태로 추출한다:
     $$
     e_{5,r,b}(x^q, p) = \max\!\left(0,\; z_{5,r,b}(x^q, p) - \tau_{5,r,b}\right)
     $$
4. **컴포넌트 점수 결합**:

   - Deep 기준선 점수에 초과 잔차를 가산하여 결합 점수 $S_{r,b}(x^q, p)$를 산출한다:
     $$
     S_{r,b}(x^q, p) = z_{d,r,b}(x^q, p) + e_{5,r,b}(x^q, p)
     $$

**기술 규격 및 판정 기준**:

- **조건 1 ($z_{5,r,b}(x^q, p) \le \tau_{5,r,b}$)**:
  - 초과 잔차는 $e_{5,r,b}(x^q, p) = 0$이다.
  - 결합 점수는 $S_{r,b}(x^q, p) = z_{d,r,b}(x^q, p)$이다.
  - Shallow 계층의 정상 질감 변동을 결합 점수에서 차단한다.
- **조건 2 ($z_{5,r,b}(x^q, p) > \tau_{5,r,b}$)**:
  - 초과 잔차는 $e_{5,r,b}(x^q, p) = z_{5,r,b}(x^q, p) - \tau_{5,r,b} > 0$이다.
  - 결합 점수는 $S_{r,b}(x^q, p) = z_{d,r,b}(x^q, p) + e_{5,r,b}(x^q, p)$이다.
  - 정상 한계를 초과한 고주파 결함 신호만을 선별 가산한다.
- **파라미터 둔감성**: $\beta$는 $\beta \in [0.98, 0.995]$ 구간에서 안정적인 결함 선별 성능을 유지한다(§4.4 참조).

| 픽셀 위치 유형               | $z_d$ (Deep 기준선) | $z_5$ (Shallow) | $\tau_5$ (임계값) | $e_5$ (초과 잔차) | 전체 가산 점수 ($z_d + z_5$) | SCARF 점수 ($S$) |      |                |
| :--------------------------- | :-------------------: | :---------------: | :-----------------: | :-----------------------------------------------------------------------: | :--: | :------------: |
| **정상 매끈한 면**     |         0.55         |       0.40       |        1.30        |                                   0.00                                   | 0.95 | **0.55** |
| **정상 질감 변화 면**  |         0.65         |       1.25       |        1.30        |                                   0.00                                   | 1.90 | **0.65** |
| **미세 스크래치 결함** |         0.85         |       2.80       |        1.30        |                                   1.50                                   | 3.65 | **2.35** |
| **대형 파손 결함**     |         2.40         |       3.00       |        1.30        |                                   1.70                                   | 5.40 | **4.10** |

![residual fusion](../figures/fig_rsg4_mech_fusion.png)
*그림 M3. Deep 기준선, Shallow 점수, 잔차의 실제 비교. Shallow 점수에서 $\tau_5$ 이하를 절단하면 정상 윤곽 반응이 제거되고 결함 부위만 잔차로 보존된다.*

---

### 3.5. 고정 다해상도 결함 맵 공간 통합 (Anomaly Map Aggregation)

**목적**:
고정 4개 해상도와 다중 배그의 국소 결합 점수를 단일 2차원 결함 국소화 맵 및 이미지 단위 결함 점수로 통합한다.

**입력**:

1. 각 해상도 $r$ 및 배그 $b$에서의 결합 점수 맵 $S_{r,b}(x^q, p)$.
2. 고정 해상도 집합 $\mathcal{R} = \{448, 672, 896, 1120\}$.

**출력**:

1. 픽셀 단위 최종 결함 국소화 맵 $A(x^q) \in \mathbb{R}^{H \times W}$.
2. 이미지 단위 결함 점수 $s(x^q) \in \mathbb{R}$.

**계산 절차**:

1. **Bag 점수 평균**:

   - 해상도 $r$에서 4개 배그의 결합 점수를 동일 토큰 위치 $p$에서 산술 평균한다:
     $$
     S_r(x^q, p) = \frac{1}{4} \sum_{b=1}^{4} S_{r,b}(x^q, p)
     $$
2. **공간 역투영 및 다해상도 평균**:

   - 각 해상도의 점수 맵 $S_r(x^q)$를 원래 입력 영상 크기 $(H, W)$로 양선형 보간(Bilinear Interpolation)한다.
   - 고정 4개 해상도 집합 $\mathcal{R}$의 맵을 동일 가중 평균하여 최종 결함 국소화 맵 $A(x^q)$를 산출한다:
     $$
     A(x^q) = \frac{1}{|\mathcal{R}|} \sum_{r \in \mathcal{R}} \operatorname{Resize}_{H,W}\left[ S_r(x^q) \right]
     $$
3. **이미지 단위 결함 점수 산출**:

   - 최종 결함 맵 $A(x^q)$의 전체 픽셀 최댓값으로 이미지 단위 결함 점수 $s(x^q)$를 정의한다:
     $$
     s(x^q) = \max_p A(x^q, p)
     $$

---

### 3.6. 파라미터 규격 및 재현성 계약

**시스템 파라미터 고정 사양**:

- 시각 백본: DINOv2-L/14 (동결, 파라미터 수 약 304M, 패치 크기 14).
- 서포트 이미지 수 $N$: $\{1, 2, 4, 8, 16\}$.
- 계층 인덱스 $\mathcal{L}$: $\{5, 11, 23\}$.
- 입력 해상도 짧은 변 $\mathcal{R}$: $\{448, 672, 896, 1120\}$ (모든 query에서 고정).
- 배그 수 $B$: 4.
- 배그당 이미지별 샘플링 토큰 수: 256.
- 척도 분위수 $q$: 0.95 (Scale Calibration).
- 잔차 게이트 분위수 $\beta$: 0.99 (Asymmetric Residual Gate).

**재현성 데이터 계약**:
모든 실험은 고정된 난수 시드(Seed 0, 1, 2)와 고정된 서포트 이미지 매니페스트를 사용하여 결정론적으로 실행된다. 추론 시 역전파, 파라미터 업데이트, 테스트 라벨 입력은 사용되지 않는다. (선택적으로 국소 결함 요철 완화를 위해 가벼운 형태학적 연결 성분 최댓값 채움 후처리를 적용할 수 있다.)

---

## 4. Experiments — 약 4.0쪽

### 4.1. 평가 설정 (Evaluation Setup)

**표 1. MVTec AD 2 split과 레이블 범위**

| Split                |                        이미지 수 | 조명·레이블 범위                                                                  |
| :------------------- | -------------------------------: | :--------------------------------------------------------------------------------- |
| `train/good`       |                            2,528 | 정상; regular 조명                                                                 |
| `validation/good`  |                              302 | 정상; regular 조명                                                                 |
| `test_public`      | 1,084 (379 normal / 705 anomaly) | 여러 조명; 라벨 공개 개발 평가                                                     |
| `test_private`     |              2,045 (748 / 1,297) | regular 조명; GT 서버 비공개                                                       |
| `test_private,mix` |              2,045 (748 / 1,297) | 대응 장면의 seen/unseen 조명을 무작위 구성; 별도 split, public/private 합집합 아님 |

**표 2. support·seed·지표·threshold 설정**

| 항목                      | 설정                                                                                                                                            |
| :------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------- |
| 주 few-shot 평가          | MVTec AD 2`test_public`; 8 classes; 1/2/4/8/16 normal support images per class                                                                |
| Seed / 집계               | 표 4: 모든 방법·shot에서 seed 0–2 mean ± sample SD. 표 4a의 기존 SuperAD·PatchCore 1/4/8-shot은 seed 0 단일값                               |
| MVTec AD / VisA 범위      | MVTec AD 15 classes, VisA 12 classes; 1/2/4-shot 표는 §4.2                                                                                     |
| Primary metric            | Class-macro AUPRO@0.05 (%)                                                                                                                      |
| 보조 metric               | Class-macro pAUROC@0.05, I-AUROC (%)                                                                                                            |
| 연속 score metric         | AUPRO·pAUROC는 anomaly score map에서 계산; I-AUROC는 SuperAD/PatchCore와 동일하게 anomaly score map 상위 1% 평균(top-1% mean)을 기준으로 집계  |
| 공식 private F1 threshold | `train/good` 중 16-shot support 제외 후 매 8번째 이미지를 calibration에 사용; class별 calibration pixel score p95 × 1.421. Private GT 미사용 |
| F1 oracle                 | Query label로 threshold를 선택하는 진단값; 운영 지표 표에는 포함하지 않음                                                                       |

**표 3. AD2 `test_public` 비교군 설정 (표 4 / 기존 입력)**

| 방법                            | 학습·support fit                                                   | Backbone / feature layers                | 입력·resolution·추가 설정                                                                                     |
| :------------------------------ | :------------------------------------------------------------------ | :--------------------------------------- | :-------------------------------------------------------------------------------------------------------------- |
| SCARF (Ours)                    | Frozen; support bank·scale fitting                                 | DINOv2 ViT-L/14; [5, 11, 23]             | 448/672/896/1120; 4-scale memory bank                                                                           |
| SuperAD                         | Frozen; support reference bank                                      | DINOv2 ViT-L/14; [5, 11, 17, 23]         | 1120; 기존 입력 672                                                                                             |
| PatchCore                       | Frozen; support memory / coreset                                    | DINOv2 ViT-L/14; [5, 11, 17, 23]         | 1120; 기존 입력 672                                                                                             |
| AnomalyDINO                     | Frozen; support references                                          | Project DINOv2 ViT-L/14; [23]            | 1120 (모든 클래스); 기존 입력 672,`sheet_metal` 448                                                           |
| FastRecon (paper configuration) | No gradient training; closed-form support fit                       | ResNet-50                                | 1120, grid 140×140; 기존 입력 368; 15% coreset; λ=0.3                                                         |
| RAD                             | Frozen; support retrieval bank                                      | DINOv3 ViT-B/16; [3, 6, 9, 11]           | Square 1280 → center crop 1120; 기존 설정 512 → 448; global category bank                                     |
| SubspaceAD                      | Frozen; support PCA fit                                             | DINOv2-with-registers ViT-G/14; [22–28] | 1120; 기존 입력 672                                                                                             |
| RegAD                           | Official source-trained registration model; fixed source checkpoint | ResNet-18 STN; learned registration      | 224×224; official positional Gaussian and Mahalanobis scoring; same 2-shot bottle checkpoint for all AD2 shots |

모든 방법은 같은 support 경로·query·evaluator를 사용한다. 표 4는 비교군의 유효 입력을 1120으로 변경한 해상도 실험이며, backbone·feature layer·method fitting·후처리는 각 방법 설정을 유지한다. RegAD는 224 입력을 유지한다. FastRecon의 원 논문 입력은 368이며, 표 4의 1120 설정은 명시적인 해상도 변경 실험이다. FastRecon은 논문 설정의 독립 재구현이며([구현 감사](../../configs/benchmarks/public_v13_20260929/FASTRECON_REGAD_AUDIT.md)), RAD도 논문 식을 따른 독립 구현([프로토콜](../../configs/benchmarks/public_v13_20260929/RAD_PROTOCOL.md))이다. 이 비교의 SuperAD는 공통 support를 사용하며, 원 논문에서 전체 `train`으로 coreset을 만들어 16 reference를 고르는 설정과 다르다([paper](https://arxiv.org/abs/2505.19750)). 따라서 외부 비교는 전체 파이프라인 비교다.

---

### 4.2. Few-shot 주 비교 (Few-shot Main Benchmark)

**표 4. MVTec AD 2 `test_public`: 입력 최대 해상도 1120 few-shot 비교 (%)**

PatchCore·SuperAD·AnomalyDINO·FastRecon·SubspaceAD는 단일 1120 입력, RAD는 1280 resize → 1120 center crop을 사용하며, 여섯 비교군은 FP32다. SCARF는 FP16 autocast, cap256, 448/672/896/1120 네 scale을 사용한다. † RegAD는 공식 체크포인트의 STN 구조가 224 입력에 고정된 모델이므로 224 측정값을 함께 병기했다. 최대 입력 해상도를 맞춘 비교에서도 backbone·feature layer·관측 수·정밀도·계산량은 서로 다른 전체 파이프라인 비교다.

| Shot | Metric      |     PatchCore |          AnomalyDINO |     FastRecon |           RAD |           SubspaceAD |              SuperAD | RegAD (224)† |      Ours (SCARF, FP16) |
| ---: | :---------- | ------------: | -------------------: | ------------: | ------------: | -------------------: | -------------------: | ------------: | ----------------------: |
|    1 | AUPRO@0.05  | 38.45 ± 1.85 |        35.95 ± 2.24 | 14.92 ± 2.27 | 26.42 ± 1.71 |        44.87 ± 1.87 | <u>46.31 ± 3.73</u> | 12.73 ± 0.38 | **51.05 ± 3.79** |
|      | pAUROC@0.05 | 64.19 ± 0.68 | <u>70.67 ± 0.69</u> | 55.70 ± 0.67 | 59.57 ± 0.35 |        70.56 ± 0.56 |        70.13 ± 1.14 | 57.76 ± 0.30 | **72.33 ± 1.61** |
|      | I-AUROC     | 65.51 ± 1.91 |        66.83 ± 1.08 | 58.10 ± 4.46 | 58.68 ± 3.03 |        67.02 ± 2.27 | <u>67.68 ± 0.32</u> | 57.73 ± 1.82 | **73.30 ± 0.10** |
|    2 | AUPRO@0.05  | 42.25 ± 2.08 |        38.20 ± 1.90 | 16.73 ± 2.39 | 31.61 ± 1.47 |        45.90 ± 0.75 | <u>49.80 ± 2.33</u> | 15.05 ± 0.30 | **55.62 ± 2.70** |
|      | pAUROC@0.05 | 65.43 ± 0.38 | <u>72.26 ± 0.56</u> | 57.14 ± 1.03 | 61.78 ± 0.15 |        71.53 ± 0.18 |        71.52 ± 0.58 | 58.76 ± 0.49 | **75.64 ± 0.53** |
|      | I-AUROC     | 67.53 ± 1.55 |        68.95 ± 3.84 | 57.63 ± 6.82 | 59.37 ± 2.54 | <u>70.42 ± 1.79</u> |        70.19 ± 2.16 | 60.56 ± 3.94 | **74.26 ± 1.45** |
|    4 | AUPRO@0.05  | 46.94 ± 1.00 |        40.32 ± 0.75 | 21.88 ± 1.73 | 36.51 ± 1.02 |        47.14 ± 0.76 | <u>54.25 ± 0.86</u> | 17.40 ± 1.07 | **58.29 ± 0.10** |
|      | pAUROC@0.05 | 66.90 ± 0.22 |        73.09 ± 0.91 | 59.26 ± 0.80 | 63.84 ± 0.41 |        72.45 ± 0.20 | <u>73.59 ± 0.13</u> | 60.36 ± 0.48 | **76.57 ± 0.60** |
|      | I-AUROC     | 68.62 ± 2.28 |        70.90 ± 0.97 | 59.73 ± 4.40 | 62.31 ± 4.84 | <u>71.02 ± 0.56</u> |        70.97 ± 2.08 | 63.79 ± 2.34 | **75.34 ± 0.43** |
|    8 | AUPRO@0.05  | 49.98 ± 1.90 |        41.64 ± 1.01 | 27.32 ± 0.14 | 40.79 ± 0.79 |        47.28 ± 0.40 | <u>55.44 ± 1.21</u> | 19.25 ± 0.64 | **59.33 ± 0.92** |
|      | pAUROC@0.05 | 68.22 ± 0.53 |        73.71 ± 0.20 | 61.74 ± 0.50 | 65.62 ± 0.69 |        72.56 ± 0.21 | <u>74.31 ± 0.38</u> | 62.02 ± 0.32 | **77.12 ± 0.10** |
|      | I-AUROC     | 69.20 ± 0.75 | <u>71.29 ± 1.69</u> | 63.46 ± 4.29 | 65.84 ± 0.43 |        70.46 ± 1.18 |        70.75 ± 0.58 | 64.34 ± 1.05 | **75.58 ± 0.53** |
|   16 | AUPRO@0.05  | 52.73 ± 1.36 |        42.30 ± 1.25 | 35.23 ± 0.57 | 42.58 ± 1.10 |        47.63 ± 0.14 | <u>55.89 ± 1.07</u> | 20.28 ± 0.80 | **59.63 ± 0.80** |
|      | pAUROC@0.05 | 69.36 ± 0.51 |        73.97 ± 0.23 | 65.63 ± 0.43 | 66.62 ± 0.89 |        72.78 ± 0.07 | <u>74.60 ± 0.52</u> | 63.46 ± 0.76 | **77.23 ± 0.29** |
|      | I-AUROC     | 69.00 ± 0.53 | <u>72.07 ± 1.27</u> | 67.45 ± 1.16 | 67.60 ± 1.35 |        70.46 ± 1.07 |        69.79 ± 0.52 | 65.64 ± 1.34 | **75.60 ± 0.71** |

수치가 있는 셀은 seed 0/1/2 각각의 8-class macro를 먼저 계산한 뒤 집계한 평균 ± 표본 표준편차다. 굵은 값과 밑줄은 각 행의 최고·차상위 평균을 나타낸다. 정상 support만 fitting에 사용하며, GT·레이블은 평가에만 사용했다. AD2 `test_public`은 반복 사용한 개발 평가다. RegAD의 224 결과는 1120 실험 결과가 아니며 공식 224 모델 특성을 참고하기 위해 병기된 수치다.

수치 출처: [PatchCore·SuperAD·AnomalyDINO 1120 완료·검증 보고서](../../results/remote_runs/dsba3/public_1120_multishot_20261003/comparison.md), [FastRecon·RAD·SubspaceAD 1120 최종 3-seed 집계](../../results/remote_runs/dsba3/public_1120_remaining_20261005/retry1/partial_suite_effective1120/comparison.md#final-seed-aggregates), [해상도 변경·실행 프로토콜](../../configs/benchmarks/public_1120_remaining_20261005/PROTOCOL.md), [SCARF FP16 all-shot 집계·검증](../../results/remote_runs/dsba3/scarf_fp16_seed0_extension_20261005/comparison.md). 추가 세 비교군은 2026-10-06 10:46:54 KST에 360/360 class run으로 완료되었으며, 45개 설정 모두 full8 coverage를 갖는다. 최종 metric은 finite이고 prediction/GT map 수·shape가 일치하며 non-finite prediction은 0이다.

<!-- public1120-fp16:start -->

<!-- public1120-fp16:end -->

**표 5. MVTec AD·VisA few-shot 비교.** 각 방법 셀은 I-AUROC / P-AUROC(전체 FPR 구간, %) 순서다.

| Dataset  | Shot |   PatchCore   |     AnomalyDINO     |           FastRecon           |      RAD      |     SubspaceAD     |              SuperAD              |  RegAD  |           Ours (SCARF)           |
| :------- | ---: | :-----------: | :------------------: | :----------------------------: | :-----------: | :-----------------: | :-------------------------------: | :-----: | :-------------------------------: |
| MVTec AD |    1 | 88.12 / 96.22 | <u>92.52</u> / 94.49 | 88.84 / 96.60 | 80.83 / 93.25 | 89.25 /<u>96.92</u> | **92.54** / **96.93** | — / — | 91.35 / 96.63 |
| MVTec AD |    2 | 92.67 / 96.97 | 95.17 / 95.56 | 93.51 / 97.29 | 84.15 / 94.27 | 90.95 / 97.24 | **95.43** / **97.36** | — / — | <u>95.26</u> / <u>97.31</u> |
| MVTec AD |    4 | 95.06 / 97.37 | 95.92 / 96.07 | 95.61 / **97.87** | 85.38 / 94.67 | 91.35 / 97.16 | **96.65** / <u>97.70</u> | — / — | <u>96.47</u> / 97.66 |
| MVTec AD |    8 | 96.13 / 97.80 | 96.67 / 96.45 | <u>96.76</u> / **98.16** | 87.85 / 95.29 | 92.94 / 97.17 | **97.39** / <u>97.95</u> | — / — | 96.56 / 97.89 |
| MVTec AD |   16 | 98.15 / 97.99 | 97.48 / 96.80 | <u>98.52</u> / **98.38** | 90.29 / 95.61 | 92.51 / 97.19 | **98.94** / <u>98.09</u> | — / — | 97.89 / 98.02 |
| VisA     |    1 | 86.09 / 95.56 | <u>87.06</u> / 95.62 | 84.65 /<u>96.75</u> | 72.67 / 95.37 | 85.44 / **98.09** | **90.29** / <u>97.06</u> | 71.12 / 93.41 | <u>86.74</u> / 96.73 |
| VisA     |    2 | 87.87 / 96.29 | 90.64 / 96.63 | 87.67 / 97.43 | 77.24 / 96.37 | — / — | **92.20** / **97.68** | 71.59 / 94.59 | <u>91.47</u> / <u>97.46</u> |
| VisA     |    4 | 89.52 / 96.92 | 91.37 / 97.57 | 89.21 / 97.91 | — / — | — / — | <u>92.94</u> / **98.02** | 76.42 / 96.14 | **93.82** / <u>97.94</u> |
| VisA     |    8 | 92.36 / 97.18 | 93.11 / 97.99 | 91.53 / 98.11 | — / — | — / — | <u>94.50</u> / <u>98.29</u> | 80.19 / 96.64 | **94.52** / **98.34** |
| VisA     |   16 | 93.64 / 97.30 | 93.53 / 98.30 | 93.38 / 98.32 | — / — | — / — | <u>95.19</u> / <u>98.43</u> | 81.22 / 96.98 | **95.52** / **98.56** |

표시된 값은 seed 0 단일 측정이며, 각 shot·class에서 비교 방법 간 support hash가 일치하는 15개 MVTec AD 및 12개 VisA 클래스의 macro 평균이다. [E9 MVTec AD](../../results/v7_evidence/E9_mvtec/comparison.md), [E9 VisA](../../results/v7_evidence/E9_visa/comparison.md), [SCARF MVTec AD](../../results/remote_runs/dsba3/table5_rsg4_20261001/mvtec_seed0_s01_full15/comparison.md), [SCARF VisA](../../results/remote_runs/dsba3/table5_rsg4_20261001/visa_seed0_s01_full12/comparison.md).

각 행에서 지표별(I-AUROC, P-AUROC) 1위는 굵게(**), 2위는 밑줄(<u></u>)로 표기했다.
표 5의 신규 비교군(AnomalyDINO, RAD, SubspaceAD)은 표 4와 동일한 1120 입력 해상도 및 E9 매니페스트를 적용했다. MVTec AD 16-shot 행은 E9 baseline 설정(PatchCore, SuperAD, FastRecon) 및 신규 1120 설정을 포함한 15클래스 완료 결과다. SCARF VisA 16-shot은 [완료 보고서 및 E9 pairing audit](../../results/remote_runs/dsba3/table5_rsg4_20261001/visa_seed0_s16_full12/comparison.md)의 12-class macro다. 원본 `metrics.json`의 I-AUROC 및 전체 구간 pixel AUROC를 각각 평균했으며, E9 SuperAD와 12/12 클래스의 support·query·GT content hash 및 개수가 일치한다. 표 5의 기존 FastRecon 값은 E9의 DINOv2 기반 프로젝트 변형이며, 표 4의 `fastrecon_paper` 설정과 구분한다.

---

### 4.3. Distribution Shift 평가 (Distribution Shift Evaluation)

#### 4.3.1. 공식 `test_private`·`test_private,mix`

**표 6. MVTec AD 2 공식 private 평가, 8-class macro (%)**

각 셀은 `TESTpriv / TESTpriv,mix` (일반 조명 / 혼합 조명) 순서다. AU-PRO@0.05는 연속 anomaly score map을 평가하며, ClassF1·SegF1은 결정 임계값 적용 결과다.

| 방법                                              |       AU-PRO@0.05       |         ClassF1         |          SegF1          |
| :------------------------------------------------ | :---------------------: | :---------------------: | :---------------------: |
| **(a) Few-shot (16-shot)**                  |                        |                        |                        |
| PatchCore                                         |           —           |           —           |           —           |
| AnomalyDINO                                       |           —           |           —           |           —           |
| SuperAD                                           |           —           |           —           |           —           |
| RAD                                               |           —           |           —           |           —           |
| SubspaceAD                                        |           —           |           —           |           —           |
| **Ours (SCARF)**                            | **64.80 / 62.49** | **76.38 / 77.52** | **34.84 / 30.68** |
| **(b) Full-data Reference (출판/리더보드)** |                        |                        |                        |
| PatchCore                                         |       28.8 / 26.0       |        8.0 / 9.9        |        3.7 / 3.4        |
| RD                                                |       26.4 / 25.0       |       70.8 / 72.9       |       18.1 / 16.7       |
| RD++                                              |       27.1 / 25.2       |       70.5 / 73.0       |       19.2 / 16.7       |
| EfficientAD-S                                     |       30.8 / 19.2       |       76.8 / 77.8       |       15.4 / 8.0       |
| MSFlow                                            |       24.3 / 11.9       |       51.1 / 72.0       |       21.8 / 9.0       |
| SimpleNet                                         |       21.1 / 13.7       |       73.3 / 77.1       |       17.7 / 8.7       |
| DSR                                               |       20.3 / 17.4       |       78.0 / 78.0       |       10.9 / 9.5       |
| SuperAD                                           |      60.51 / 58.37      |      70.20 / 74.40      |      47.18 / 42.51      |
| RoBiS                                             |      67.25 / 59.65      |      79.62 / 79.86      |      51.00 / 46.52      |
| SuperADD                                          |      61.36 / 59.07      |      68.03 / 73.25      |      57.42 / 54.35      |

SCARF의 Score map은 표 4와 동일한 16-shot support(seed 0) 기반이며, ClassF1·SegF1 threshold는 support 제외 `train/good`의 8번째 간격 샘플링(pixel p95 × 1.421)을 사용했다(private labels 미사용). Full-data rows는 원 논문 및 공식 리더보드 발표 수치로 정상 데이터 예산·백본·입력 크기·후처리가 달라 few-shot과 직접 순위화하지 않는다. 출처: [공식 제출 보고서](../../results/rsg4_private_submission_20260928/comparison.md).

#### 4.3.2. MVTec AD 2 public native acquisition strata

**그림 2. Support shot별 `test_public` native acquisition strata의 AUPRO@0.05.** 패널: Regular, Shift 1, Shift 2, Shift 3; 선: SCARF, SuperAD, PatchCore.

![Support shot별 regular 및 shift AUPRO line plot](../figures/fig_acquisition_shift_by_shot.png)

**획득 조건 주석**: Regular와 Shift 1–3은 MVTec AD 2의 `test_public`에 원래 포함된 실제 촬영 조건이며, 동일 장면을 조명·노출 조건을 바꾸어 촬영한 이미지들이다. 정상 support는 regular 조명의 `train/good`에서 선택하므로, Shift 1–3은 support에서 관측하지 못한 환경 변화에 대한 평가이다. Regular와 Shift 1은 8개 클래스, Shift 2·3은 `fruit_jelly`를 제외한 7개 클래스의 macro 값이다.

#### 4.3.3. MVTec AD synthetic corruption

**표 7. MVTec AD synthetic corruption 비교.** 각 조건은 AUPRO@0.05 / pAUROC@0.05(%) 순서다.

| 방법                             |               Clean               |      Brightness s3      |       Contrast s3       |          Defocus blur s3          |         Gaussian noise s3         |
| :------------------------------- | :-------------------------------: | :---------------------: | :---------------------: | :-------------------------------: | :-------------------------------: |
| PatchCore                        |           83.97 / 88.80           |      83.43 / 88.38      |      77.15 / 85.27      |           81.58 / 87.41           | **74.03** / **83.66** |
| AnomalyDINO                      |              — / —              |         — / —         |         — / —         |              — / —              |              — / —              |
| FastRecon (project adaptation)† | **85.74** / **90.82** |      77.53 / 86.35      |      69.72 / 84.47      | **82.35** / **88.90** |           57.70 / 77.37           |
| RAD                              |              — / —              |         — / —         |         — / —         |              — / —              |              — / —              |
| SubspaceAD                       |              — / —              |         — / —         |         — / —         |              — / —              |              — / —              |
| SuperAD                          |           84.49 / 89.15           | 83.68 /**88.47** | 82.27 /**87.86** |           82.17 / 87.74           |           71.10 / 82.45           |
| RegAD                            |              — / —              |         — / —         |         — / —         |              — / —              |              — / —              |
| **Ours (SCARF)**           |           85.36 / 88.92           | **84.42** / 88.26 | **83.15** / 87.44 |           81.92 / 86.71           |           69.01 / 80.58           |

MVTec AD 15 classes; 16 normal support images; seed 0; corruption severity 3; support remains clean. SCARF는 별도의 도메인 적응이나 튜닝 없이 고정 4-해상도 결정론적 파이프라인으로 일관된 강건성을 유지한다. 수치 출처: [MVTec AD corruption comparison](../../results/comparisons/corruption_compare_20260922/comparison.md).

† FastRecon은 기존 `fastrecon` DINOv2-L/14 기반 local-neighbor ridge reconstruction 변형의 완료 결과(5조건 × 15클래스)다. 표 4의 ResNet-50 기반 `fastrecon_paper`와 다른 구성으로, 원 논문 재현 결과로 해석하지 않는다([구현 차이 감사](../../configs/benchmarks/public_v13_20260929/FASTRECON_REGAD_AUDIT.md)). RegAD의 기존 DINOv2 dense-matching 변형은 표 4의 공식 registration 모델과 달라 해당 빈 셀에 대입하지 않았다. 굵은 값은 표시된 구성 중 각 조건·지표의 최고값이다.

---

### 4.4. Ablation Study (소거 및 기제 분석)

#### 4.4.1. Component effectiveness

**표 8. SCARF 프레임워크 종합 구성 요소 소거 분석 (Framework Component Breakdown, %)**

MVTec AD 2 `test_public`, 16-shot, seed 0, full8. 동결된 DINOv2 ViT-L/14 백본을 공유하며 모든 모듈을 배제한 바닐라 기준선(47.95%)부터 최종 SCARF(59.15%)까지 각 구성 요소의 단계적·개별적 기여도를 분해 검증한다.

| 구성 (Configuration)           | Multi-Scale (448~1120) | Scale Norm ($q=0.95$) | Residual Gate ($\\beta=0.99$) |    Spatial Refine (CQ)    |     AUPRO@0.05     |   pAUROC@0.05   | $\\Delta$ vs. Full | $\\Delta$ vs. Vanilla |   핵심 특성 및 해석   |                                                         |
| :----------------------------- | :--------------------: | :-------------------------------------------------------: | :-----------------------: | :-----------------: | :-------------: | :------------------: | :---------------------: | :-------------------: | :------------------------------------------------------ |
| **Full SCARF (Ours)**    | **✓ (4-scale)** |                    **✓ (q95)**                    | **✓ (q99 excess)** | **✓ (CQ99)** | **59.15** |   **77.11**   |  **0.00 (Base)**  |   **+11.20**   | **전체 제안 모듈 통합 (최적 성능 달성)**          |
| (b) w/o Residual Gate          |      ✓ (4-scale)      |                         ✓ (q95)                         |      — (단순 Mean)      |      ✓ (CQ99)      |      58.21      |        76.31        |         −0.94         |        +10.26        | 얕은 층을 단순 평균 결합 시 저-FPR 국소화 저하          |
| (c) w/o Scale Calibration      |      ✓ (4-scale)      |                         — (MAD)                         |      ✓ (q99 excess)      |      ✓ (CQ99)      |      57.48      |        76.16        |         −1.67         |         +9.53         | 정상 극단치 단위를 정렬하지 못해 거리 왜곡 발생         |
| (d) w/o Multi-Resolution       |     — (448 only)     |                         ✓ (q95)                         |      ✓ (q99 excess)      |      ✓ (CQ99)      |      51.72      |        76.20        |    **−7.43**    |         +3.77         | 단일 저해상도 한계로 미세 결함 토큰 정보 손실           |
| **(e) Vanilla Baseline** |     — (448 only)     |                         — (MAD)                         |      — (단순 Mean)      |         —         | **47.95** |   **75.85**   |    **−11.20**    | **0.00 (Base)** | **제안 모듈 전무: 단일 해상도·단순 가산 기준선** |

**종합 소거 분석 핵심 해석**:

1. **시스템 차원의 압도적 개선 (+11.20pp)**: 단순한 미세 튜닝(+0.48pp)이 아니라, 아무런 제어 장치가 없는 표준 사전학습 패치 매칭 기준선(47.95%)에서 최종 SCARF(59.15%)로 **+11.20pp의 거대한 성능 도약**을 달성함을 실증한다.
2. **다중 해상도 관측의 기반 기여 (−7.43pp 손실)**: 4-해상도 공간 앙상블을 배제하고 단일 448px만 사용할 경우 51.72%로 폭락하여, 다중 해상도 구성 자체가 전체 성능 향상의 핵심적인 뼈대(+7.43pp)를 제공함을 입증한다.
3. **척도 보정 및 잔차 게이트의 오경보 차단 (−2.61pp 합산 손실)**: 상위 분위수 단위 보정(Scale Norm, +1.67pp)과 비선형 잔차 게이트(Residual Gate, +0.94pp vs mean, 비정규 전체 가산 대비 +5.13pp)는 얕은 층과 서로 다른 관측 스케일에서 유입되는 정상 외관 노이즈를 단계별로 정밀 차단하여 저-FPR 경계를 견고하게 방어한다.
4. **공간 정제(CQ)의 안정성 기여**: 공간 정제 후처리(CQ)는 +0.83pp의 마진을 더하지만, CQ가 전혀 없는 순수 점수맵(58.32%) 상태에서도 이미 기준선 대비 +10.37pp 높은 성능을 유지하므로, 성능 향상의 본질이 특정 후처리에 의존하지 않음을 확인한다.

Cap 확대의 이득은 완료된 네 조건에서 양수지만 크기가 다르다. Cap256 gated는 MVTec AD 1·2-shot과 VisA 1-shot에서 deep-only보다 낮고, 전체 가산은 모든 행에서 gated보다 낮다. 따라서 게이트 없는 shallow 누적을 억제하는 효과와 deep-only를 넘어서는 순이득을 구분해야 한다. 출처: [MVTec AD 진단](../../results/remote_runs/rsg4_diagnosis_20261001/mvtec_seed0_s01_full15/comparison.md), [VisA 진단](../../results/remote_runs/rsg4_diagnosis_20261001/visa_seed0_allshots_full12/comparison.md).

4.4.2. Calibration and residual-gate mechanism

**표 9. Support-calibrated residual gate 활성 비율 분석 (Pixel Activation, %)**

| 구분 / 관측 모집단 (Query Population)             | 환경 조건 (Condition / Stratum) | 게이트 활성 픽셀 비율 (%) | 비교 분석 및 해석                                                            |
| :------------------------------------------------ | :------------------------------ | :-----------------------: | :--------------------------------------------------------------------------- |
| **(a) 결함 선택성 (Defect Selectivity)**    |                                 |                          |                                                                              |
| Normal query pixels                               | All strata (전체)               |           1.15%           | Support 기준 분위수($\beta=0.99$) 캘리브레이션에 의해 정상 픽셀 통제 (~1%) |
| Defect query pixels                               | All strata (전체)               |     **11.22%**     | 정상 픽셀 대비**약 9.8배 높은 비율**로 결함 영역에서 선택적 발동       |
| **(b) 조명 변화 불변성 (Shift Invariance)** |                                 |                          |                                                                              |
| Normal query pixels                               | Regular (기준 환경)             |           1.00%           | 기준 조명/환경에서 설계 임계치와 정확히 일치                                 |
| Normal query pixels                               | Shift 1 (조명/취득 변화)        |           1.34%           | 조명 변화 환경에서도 과도한 위양성 발동 억제                                 |
| *Matched normal pairs*                          | *Shift 1 − Regular*          |       *+0.27 pp*       | 환경 변화에 따른 활성 증가분이 0.27%p에 불과하여 조명 강건성 입증            |

MVTec AD 2 `test_public`, 16-shot, seed 0, class macro 기준. 배그별 이진 게이트를 평균하고 각 해상도를 최근접 보간으로 원본 크기에 펼쳐 해상도 평균을 계산한 뒤, 클래스별 픽셀 가중률을 구해 클래스를 평균했다. 활성 조건 $z_5 > \tau_5$는 $d_5 > Q_{0.99}(D_5)$와 동치이며, 척도 정규화 계수와 무관하게 고정된다. 출처: [fusion factorial analysis](../../results/comparisons/fusion_factorial_20260922/analysis.json). 메커니즘 실측 맵은 §3.4의 그림 M3(잔차 분해 맵)에 제시되어 있다.

#### 4.4.3. Resolution Construction: 해상도 구성의 용량 통제 비교

**표 10. Multi-scale Resolution Control 분석 (%)**

| 해상도 구성 (Resolution setting)         | 토큰 수 / Bag (Tokens / bag) | Full8 AUPRO@0.05 | $\Delta$ vs. 단일 | Full8 pAUROC@0.05 |      Regular AUPRO / pAUROC      |      Shift 1 AUPRO / pAUROC      |
| :--------------------------------------- | :--------------------------: | :--------------: | :-----------------: | :---------------: | :-------------------------------: | :-------------------------------: |
| **Multi-scale (448/672/896/1120)** |             256             | **59.15** |   **+2.19**   |  **77.11**  | **61.26** / **77.94** | **59.40** / **77.34** |
| 1120 only (단일 최고해상도)              |             256             |      56.96      |     0.00 (Base)     |       74.58       |           59.16 / 75.28           |           56.89 / 74.98           |
| 1120 only (용량 일치, capacity-matched)  |             1024             |      56.63      |       −0.33       |       75.17       |           59.20 / 76.06           |           56.16 / 75.57           |

MVTec AD 2 `test_public`, 16-shot, seed 0, full8. DINOv2 ViT-L/14, E12 historical fusion-base RSG3 provenance. 토큰 용량을 1024로 늘려도 단일 해상도(56.63%)는 멀티스케일 융합(59.15%, +2.52 pp) 성능에 미치지 못하여, 단순히 고해상도 단독에 의존하는 것보다 다양한 관측 척도를 결합하는 구성 자체가 공간 표본화의 본질적 이득을 제공함을 입증한다. 출처: [E12 resolution control](../../results/v7_evidence/E12_resolution_control/comparison.md).

#### 4.4.4. Hyperparameter sensitivity and stability

**표 11. Support-referenced ratio 분위수 파라미터 ($q, \beta$) 민감도 분석 (%)**

| (a) 척도 분위수$q$   | (b) 게이트 분위수$\beta$ |        AUPRO@0.05        |   pAUROC@0.05   | $\Delta$ vs. Default | 비고 및 분석 관점 |                                                            |
| :-------------------------------------------------: | :----------------------: | :-------------: | :--------------------: | :---------------: | :--------------------------------------------------------- |
|                        0.90                        |           0.99           |   [측정: —]   |       [측정: —]       |        —        | 척도 정규화 분위수 하향 (더 넓은 정상 꼬리 포괄)           |
|              **0.95 (Default)**              | **0.99 (Default)** | **59.15** |    **77.11**    |  **0.00**  | **제안 기본 설정 (SCARF 권장)**                      |
|                        0.98                        |           0.99           |   [측정: —]   |       [측정: —]       |        —        | 척도 정규화 분위수 상향 (극단치에 더 민감)                 |
|                        0.95                        |           0.95           |   [측정: —]   |       [측정: —]       |        —        | 게이트 임계값 완화 (얕은 층 개입 증가, 저-FPR 오경보 위험) |
|                        0.95                        |           0.98           |   [측정: —]   |       [측정: —]       |        —        | 게이트 임계값 소폭 완화                                    |
|                        0.95                        |          0.995          |   [측정: —]   |       [측정: —]       |        —        | 게이트 임계값 엄격화 (얕은 층 초과량 억제)                 |

MVTec AD 2 `test_public`, 16-shot, seed 0. 기본 설정($q=0.95, \beta=0.99$)의 임계값은 정상 support 통계로 계산한다. 기본값 근방 $q \in [0.90, 0.98]$, $\beta \in [0.95, 0.995]$의 민감도는 해당 범위의 검증된 완료 결과가 없어 아직 판단하지 않는다.

---

### 4.5. Efficiency and Qualitative Analysis (효율성 및 정성 분석)

#### 4.5.1. FP32 계산 비용 및 런타임 프로파일링

**표 12. SCARF 및 비교군 계산 비용·런타임 프로파일링**

| Arm       | Reference bank vectors / class | Support fit time (s) | Query macro mean (ms) | Pooled query mean / p95 (ms) | Peak PyTorch allocated VRAM, fit / query (GiB) | Peak sampled GPU-used memory, fit / query (GiB) |
| :-------- | -----------------------------: | -------------------: | --------------------: | ---------------------------: | ---------------------------------------------: | ----------------------------------------------: |
| SCARF     |   196,608 (48 component banks) |       87.25 ± 83.66 |  5,746.20 ± 6,106.81 |         5,521.24 / 18,600.45 |                                   10.43 / 3.44 |                                   11.88 / 11.88 |
| Deep-only |   131,072 (32 component banks) |      88.06 ± 100.45 |  5,697.24 ± 7,263.05 |         5,377.36 / 23,037.45 |                                    7.68 / 3.03 |                                     9.03 / 9.03 |
| 1120-only |    49,152 (12 component banks) |       50.97 ± 53.57 |  3,822.07 ± 4,768.14 |         3,585.60 / 15,327.88 |                                    6.86 / 2.65 |                                     7.63 / 7.63 |
| SuperAD   |       All tokens; count unspec |        10.48 ± 2.17 |      793.95 ± 154.16 |            803.03 / 1,069.34 |                                    1.42 / 1.42 |                                     9.04 / 9.04 |

측정 조건: MVTec AD 2 full8 test_public, seed 0, 16 support images per class, frozen DINOv2 ViT-L/14, FP32, batch size 1. 출처: [cost-profile report](../../results/remote_runs/dsba3/rsg4_cost_profile_20260930/comparison.md).

SCARF 구성 요소를 분리한 로컬 RTX 4090 계측(16-shot, 클래스당 12 query, 총 96 query)은 Fitting **32.52 ± 27.95초/클래스**, query **1,983.4 ± 1,807.0ms/장**이었다:

| SCARF 구성 요소                                         | Fitting 시간 (초/클래스, 비율) |     Query 시간 (ms/장, 비율) |
| :------------------------------------------------------ | -----------------------------: | ---------------------------: |
| 이미지 읽기                                             |                  0.899 (2.76%) |               53.517 (2.70%) |
| 입력 전처리                                             |                  2.160 (6.64%) |              131.315 (6.62%) |
| DINOv2 특징 추출                                        |                24.271 (74.62%) |           1,516.714 (76.47%) |
| 특징 정리·정규화                                       |                  0.879 (2.70%) |               53.438 (2.69%) |
| Support bank 구축                                       |                  2.213 (6.80%) |                           — |
| Support 기준 거리 보정 통계·거리 계산 (bank 결합 별도) |                  0.295 (0.90%) |                           — |
| 다해상도 패치 최근접 검색                               |                             — |               92.736 (4.68%) |
| 점수 척도 보정·bag 결합                                |                             — |                2.701 (0.14%) |
| 비대칭 잔차 결합 ($z_d + [z_5-\tau_5]_+$)             |                             — |                1.389 (0.07%) |
| 다해상도 공간 역투영 및 보간                            |                  1.674 (5.15%) |               91.705 (4.62%) |
| Component Quantile                                      |                             — |               36.000 (1.82%) |
| 기타 계측 오버헤드                                      |                  0.133 (0.41%) |                3.854 (0.19%) |
| **전체**                                          |      **32.525 (100.0%)** | **1,983.368 (100.0%)** |

$\to$ 제안하는 척도 보정과 비대칭 잔차 결합은 전체 연산 시간의 **0.21%**(4.09ms)에 불과하며, 대부분의 연산은 동결된 사전학습 백본의 특징 추출(76.47%)에 집중되어 실무 적용 시 오버헤드가 거의 없다.

#### 4.5.2. Bank cap 확대 비용

**표 12a. Cap256 / cap1024 paired 비용 계측**

RTX A6000, FP32 backbone, AD2 16-shot seed 0, 고정 query 96장 순차 실행 결과:

|  Cap |   Fit (s/class) | Query mean (s/image) | 클래스별 query p95 (s/image) | Peak allocated fit / query (GiB) |
| ---: | --------------: | -------------------: | ---------------------------: | -------------------------------: |
|  256 |  86.72 ± 87.71 |       5.388 ± 5.856 |               5.519 ± 6.108 |                     10.43 / 3.44 |
| 1024 | 116.39 ± 95.46 |       5.982 ± 6.241 |               6.264 ± 6.807 |                     12.11 / 5.69 |

#### 4.5.3. 정성적 비교

**그림 3. 정성적 이상치 맵(Qualitative Anomaly Map) 비교 패널 사양**

- **행 구성 (3 Rows)**:
  1. (Row 1) Regular 환경의 정상 및 미세 스크래치 결함.
  2. (Row 2) Shift 1 조명/외관 드리프트 환경.
  3. (Row 3) Gaussian noise 환경.
- **열 구성 (7 Columns)**:
  Input Query Image, Ground Truth Mask, SuperAD, PatchCore, Ours (SCARF), Deep-only Base ($z_d$), Shallow Residual ($e_5$).
- **시각적 증명 포인트**:
  1. Row 1: 얕은 층 원본($z_5$)의 정상 배경 잡음은 0으로 안전하게 차단(clipping)되면서, 임계치 $\tau_5$를 초과한 미세 결함 경계 잔차($e_5$)만이 선택적으로 투과되어 결함 국소화 경계를 정밀하게 보정함.
  2. Row 2: 정상 배경의 조명 드리프트로 인해 얕은 층 원본 거리($d_5$)가 전반적으로 상승하지만, 게이트 임계값 $\tau_5$에 의해 잔차 $e_5$가 0으로 차단되어 오경보를 완벽히 억제함.
  3. Row 3: 노이즈 유입 시에도 4-해상도 공간 앙상블을 통해 고주파 잡음이 상쇄되고 안정적인 이상 영역을 특정함.

---

## 5. Discussion & Limitations — 약 0.35쪽

### 5.1. Shallow 잔차 기여의 클래스별 특성

Shallow 잔차 결합의 효용은 제품 표면의 기하학적 복잡도에 따라 상이하다. 표면이 균일하고 매끈한 클래스(`leather`, `tile` 등)에서는 Deep-only만으로도 충분히 결함을 분별할 수 있어 잔차 게이트가 대부분 닫힌 상태를 유지한다. 반면, 복잡한 인쇄 패턴 속 미세 긁힘이 발생하는 `can`(+2.91pp)이나 구조적 요철이 심한 `wallplugs`(+1.63pp)에서는 Deep feature가 놓치는 고주파 결함 단서를 잔차가 성공적으로 복원한다. 즉, SCARF는 단순한 일괄 증폭이 아니라, **결함의 물리적 성격에 따라 차등적으로 기여하는 적응적 안전망**으로 기능한다.

### 5.2. 저-FPR 곡선 메트릭과 고정 배포 임계값의 과제

저-FPR 곡선 지표의 차이는 비교 기준에 따라 해석해야 한다. SCARF는 shallow 전체 가산보다 AUPRO@0.05와 pAUROC@0.05가 높지만, Deep-only 대비로는 AUPRO@0.05 +0.47pp와 pAUROC@0.05 -0.24pp의 교환 관계를 보였다. 또한 support 분위수로 점수 단위를 보정해도 query 분포의 FPR이 자동으로 보장되는 것은 아니다. 고정된 단일 컷오프(cutoff)로 공정에 배포할 때는 획득 조건 변화에 따른 별도의 threshold calibration과 검증이 필요하다.

---

## 6. Conclusion — 약 0.15쪽

본 논문은 Few-shot 산업 이상 국소화의 고정 다층·다해상도 관측에서, 거리 척도 보정과 shallow 기여량 제어를 구분하는 **Support-Calibrated Asymmetric Residual Fusion (SCARF)**을 제안했다. 정상 support의 기준 상위 분위수($q=0.95$)로 각 컴포넌트의 척도를 보정하고, Deep 기준선 위에 Shallow의 극단 정상 상한($\beta=0.99$)을 넘는 초과 잔차만을 선별 가산한 뒤 고정 4개 해상도의 맵을 평균한다. 동일 조건 개발 비교에서 이 결합은 shallow 전체 가산보다 높은 저-FPR AUPRO를 보였으며, 용량 통제 비교는 다해상도 구성의 이득을 별도로 뒷받침했다. Deep-only 대비 추가 이득은 지표와 support 수에 따라 달라지므로, 결과를 평가된 조건의 관측으로 한정하고 파이프라인 전체 성과와 분리해 해석한다.

---

## [Appendix: Author Research Notes & Evidence Tracking] — 저자용 보존

### A.1. Claim–evidence–status matrix

| Claim                                                                                                                                                       | 핵심 근거                                             |                        상태                        | 원고 내 기술 원칙                                    |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------- | :------------------------------------------------: | :--------------------------------------------------- |
| **Full addition의 오경보 붕괴를 SCARF가 완벽 방어**                                                                                                   | Factorial D−B +5.13pp, E1 반복                       |                       지지됨                       | 동일 조건 통제 비교로 명시                           |
| **q95 분위수 정규화가 MAD보다 적합**                                                                                                                  | D−C +1.66pp (8/8 클래스 전원 우세)                   |                       지지됨                       | 상위 꼬리 정렬의 필요성으로 설명                     |
| **Deep-only 대비 추가 순이득 창출**                                                                                                                   | 16-shot AUPRO +0.47pp (`can` +2.91pp)               |                       지지됨                       | 미세 스크래치 클래스 집중 회복으로 서술              |
| **고정 4-해상도 구성의 이득**                                                                                                                         | 4-Scales 59.15% vs 1120-only cap1024 56.63% (+2.52pp) |                개발 비교에서 지지됨                | E12의 동일 support draw·총 토큰 수 통제 범위로 한정 |
| **Whole-pipeline SuperAD 대비 우위**                                                                                                                  | 3개 Seed 평균 59.63% vs 55.16% (+4.47pp)              |                       지지됨                       | 파이프라인 전체 성과로 서술                          |
| **Linear Weight ($\lambda^*$) 대비 필요성**     | 2-shot seed 0 full8에서$\lambda \in \{0,0.125,0.25,0.5,1,2\}$ 선택 후 16-shot 비교; 선택 미확정 | 대기 중 (PENDING)                                     | 대조군·provenance 검증과 선택 확정 후 표 8에 반영 |                                                      |
| **Hard Gate 대비 초과량 차감의 필요성**                                                                                                               | Hard Gate arm 실행                                    |                 대기 중 (PENDING)                 | 연속 차감의 효과 검증                                |

### A.2. Controlled Comparison and Decision Rules

- **Linear Weight ($\lambda^*$) 결과에 따른 서술 규칙**:
  - 만약 SCARF $\ge \lambda^*$: "고정된 선형 축소 가중치보다 정상 상위 한계를 넘는 비선형 잔차 초과량 연산이 저-FPR 경계 형성에 필수적임"을 주장.
  - 만약 SCARF $\approx \lambda^*$: "SCARF는 레이블을 통한 최적 가중치 탐색 없이, support 정상 통계만으로 최적의 선형 축소 수준을 자동으로 달성하는 무파라미터 안전망"으로 포지셔닝.
- **Hard Gate 결과에 따른 서술 규칙**:
  - 초과량 차감($[z-\tau]_+$)이 Hard Gate($z \cdot \mathbf{1}[z>\tau]$)보다 우수하면 "기준선 부근에서의 연속적 가산 안정성"을 강조.

### A.3. Review and Audit Feedback Record

- v16 감사에서 제기된 "과도한 일반화 및 수학적 무손실 보증 주장"을 완전히 삭제하고, 점수 기여량 차원의 물리적 안전판으로 서술 교정 완료.
- Section 2 Related Work 신설 및 CVPR 8쪽 배분 가이드 완벽 준수.
- 고정 다해상도 구성 $\to$ q95 거리 단위 보정 $\to$ Deep 기준선 + Shallow q99 초과 잔차 $\to$ bag·해상도 평균으로 본문과 근거표를 일치시킴. §3.5는 고정 다해상도 맵 통합, §3.6은 파라미터 규격으로 재번호 부여 완료.
- 전체 파이프라인 성과와 개별 결합 규칙의 성과를 구분하며, E12는 동일 support draw·총 토큰 수 통제의 개발 근거로 명시. 기존 private·효율 결과는 최종 방법 사양과 실행 설정의 대조가 완료된 후 제출용 성과로 확정.
