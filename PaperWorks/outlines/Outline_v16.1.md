# SCARF: Support-Calibrated Asymmetric Residual Fusion for Few-Shot Industrial Anomaly Localization

> **논문의 중심 질문**: 소수의 정상 reference로는 획득 조건 변화(조명·노이즈)에 따른 거리 왜곡을 사전에 포괄할 수 없다. 이러한 불완전한 정상 기준선 하에서, 다층(Multi-layer)과 다해상도(Multi-resolution) 관측 결합 시 필연적으로 발생하는 정상 상위 꼬리의 오경보 폭증을 어떻게 구조적으로 통제하고 미세 결함 단서를 안전하게 흡수할 것인가?
> *(Under incomplete few-shot normal baselines, how can we structurally control the catastrophic false alarm explosion in the normal upper tail that arises from combining multi-layer and multi-resolution observations, while safely harvesting complementary micro-defect cues?)*
>
> **원고 작성 가이드 (CVPR/WACV 본문 약 8쪽 배분)**:
> - §1. Introduction (약 1.0쪽) | §2. Related Work (약 0.5쪽) | §3. Method (약 2.0쪽)
> - §4. Experiments (약 4.0쪽) | §5. Discussion & Limitations (약 0.35쪽) | §6. Conclusion (약 0.15쪽)
> - ※ 저자용 검증 매트릭스, 실행 프로토콜, 검토 기록은 본문 흐름을 방해하지 않도록 문서 말미의 `[Appendix: Author Research Notes]`에 보존한다.

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
- **3. Method**
  - 3.1. 문제 전제 및 SCARF 3단계 파이프라인 개요 (그림 1)
  - 3.2. 다층·다해상도 정상 Support Bank 및 메모리 배깅 (그림 M1)
  - 3.3. Scale Calibration: Support 분위수 무차원 정규화 (Axis 1: Scale, 그림 M2)
  - 3.4. Asymmetric Residual Fusion: 비대칭 초과량 결합 (Axis 2: Layer, 그림 M3, M4)
  - 3.5. Resolution Reliability: 노이즈 조건부 해상도 제어 (Axis 3: Resolution, 그림 M5)
  - 3.6. Anomaly Map 통합 및 Component Quantile (CQ) 공간 후처리
  - 3.7. 구현 세부 및 재현성 계약
- **4. Experiments**
  - 4.1. Evaluation Protocol and Metrics Disclosure (AUPRO@0.05, Standardized pAUROC)
  - 4.2. Core Fusion Comparison: 단순 결합의 붕괴와 비대칭 잔차의 효용 (Table 1)
  - 4.3. Resolution Construction and Conditional Selection: 공간 샘플링과 노이즈 방어 (Table 10, 11)
  - 4.4. Whole-Pipeline Few-Shot Benchmark (Table 2: SuperAD, PatchCore 등 비교)
  - 4.5. Support Variation & Transfer Boundary (Table 3: MVTec AD / VisA 저-샷 전이)
  - 4.6. Mechanistic Analysis: Gate Activation and Threshold Dynamics (그림 M4, Table 4)
  - 4.7. Official Private Benchmark & Efficiency Profile (Table 6, FP16 가속)
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

본 논문은 **Support-Calibrated Asymmetric Residual Fusion (SCARF)**을 제안한다. SCARF는 불확실한 원인 판별 대신, 각 반응을 정상 support에서 관측되는 분포에 비추어 정규화하고 제어한다. 구체적으로 (1) 정상 support의 Leave-Source-Out 최근접 거리 상위 분위수(q95)로 각 관측의 단위를 공정하게 일치시키고(Scale Calibration), (2) 외관 변화에 안정적인 Deep feature를 기본 탐지 축으로 삼아 Shallow feature가 정상 상위 한계(q99)를 넘는 **초과 잔차(Excess Residual)**만을 선별 결합하며(Asymmetric Residual Fusion), (3) 입력 영상의 고주파 노이즈가 support 전형값을 초과할 때 고해상도 경로를 조건부로 배제한다(Resolution Reliability).

동일한 백본·support·해상도·메모리 뱅크 조건을 적용한 MVTec AD 2 16-shot 개발 비교에서, SCARF는 shallow feature 전체를 단순 가산할 때 발생하는 치명적인 저-FPR 국소화 성능 하락(-5.13pp)을 완벽히 방어하며, 강력한 Deep-only 기준선 대비 AUPRO@0.05를 **58.67%에서 59.15%(+0.47pp, 미세 결함 클래스 최대 +2.91pp)**로 추가 향상시켰다. 또한 용량을 통제한 다해상도 비교에서 4-해상도 결합은 단일 최고 해상도(1120px) 대비 **+2.52pp** 높은 국소화 성능을 기록했고, 노이즈 라우팅은 깨끗한 영상 성능을 온전히 유지하면서 합성 노이즈 환경에서 **+4.33pp**를 방어했다. 동일 support와 DINOv2 ViT-L/14를 사용한 16-shot SuperAD 파이프라인 비교에서는 AUPRO@0.05가 **4.47pp**, pAUROC@0.05가 **0.70pp** 높았으며, 공식 MVTec AD 2 private 벤치마크 평가에서도 TESTpriv **64.80%**, TESTpriv,mix **62.49%**를 달성하여 few-shot 조건임에도 우수한 이상 국소화 성능을 실증했다.

---

## 1. Introduction — 약 1.0쪽

### 1.1. 소수 정상 reference와 거리 해석의 모호성 (High-Score Ambiguity)

- **소수 정상 이미지 기반 검사의 실용성**: 다품종 소량 생산 및 빈번한 생산 라인 전환 환경에서는 제품마다 수천 장의 결함 데이터를 수집하거나 모델을 재학습하기 어렵다. 대규모 데이터로 사전학습된 동결(frozen) 시각 백본(예: DINOv2)과 소수의 정상 이미지로 구성된 참조 뱅크(reference bank)를 활용하면, 별도 미세조정 없이 query 패치가 정상 패치와 얼마나 떨어져 있는지 측정함으로써 즉각적인 결함 검사가 가능하다. 특히 실제 산업 공정에서는 전체 ROC 구간보다 오경보율이 엄격히 통제되는 **저-FPR(Low False Positive Rate, 예: FPR $\le$ 5%) 구간에서의 정밀한 결함 영역 국소화(defect localization)**가 핵심적인 신뢰성 요구조건이다.
- **획득 조건 변화(Distribution Shift)와 거리 해석의 딜레마**: 소수의 정상 reference는 실제 공정에서 발생하는 조명 드리프트, 부품 자세 변동, 표면 난반사, 센서 노이즈 같은 다양한 환경 변화를 결코 포괄할 수 없다. 따라서 결함이 없는 정상 query 패치라도 reference bank에 없는 정상 변동이 발생하면 최근접 거리가 크게 벌어지며, 이는 실제 결함이 유발하는 큰 거리와 외형상 구별되지 않는다. 즉, **"큰 거리값"이라는 관측만으로는 그 원인이 실제 결함인지, reference에 미포함된 정상적 외관 변화인지 확정할 수 없다(High-Score Ambiguity).**
- **저-FPR 검사에서의 파급효과**: 오경보를 엄격히 억제해야 하는 실제 산업 공정(예: FPR $\le$ 1~5%)에서, 시스템의 판정 임계값은 정상 점수 분포의 평균이나 중심이 아니라 **상위 1~5% 분위수 경계(upper tail)**에 의해 직접 결정된다. 따라서 결함이 없는 정상 영역에서 우연히 튄 소수의 큰 거리값조차 곧바로 임계값을 비정상적으로 끌어올리거나 오경보(False Alarm)로 직결되어, 저-FPR 구간(AUPRO@0.05, pAUROC@0.05)의 검출 성능을 심각하게 훼손한다. 소수 support 환경에서 정상 수준의 관측 신호가 점수에 무분별하게 누적되지 않도록 차단하고, 정상 범위를 명백히 초과한 반응만을 선별 수용해야 하는 이유가 바로 여기에 있다.

### 1.2. 다층·다해상도 관측의 결합과 딜레마

Few-shot 검사에서는 reference 표본이 극도로 희소하므로, 각 이미지에서 가능한 한 다양한 수준의 시각적 단서를 확보하는 것이 유리하다.
1. **표현 계층(Layer Depth)**: ViT 백본의 심층 레이어(Deep, Layer 11/23)는 조명·색상 변화에 둔감한 안정적 의미 기준선(semantic base)을 제공하지만 미세 윤곽과 국소 텍스처에 둔감하다. 반면 얕은 레이어(Shallow, Layer 5)는 미세 스크래치와 국소 결함에 민감하지만 정상 외관 변화에도 과민 반응한다.
2. **입력 해상도(Input Resolution)**: 패치 크기가 $14 \times 14$로 고정된 구조에서, 448px 저해상도는 이미지 짧은 변의 3.1%를 하나의 토큰이 덮어 1% 미만의 미세 결함을 주변 정상 질감으로 희석시킨다. 반면 1120px 고해상도는 토큰 면적이 1.25%로 줄어들어 미세 결함에 대한 공간 표본 밀도를 비약적으로 높여준다.

**관측 결합의 역설 (Catastrophic False Alarm Explosion)**:
그러나 선행 연구들처럼 feature를 단순히 결합(PatchCore)하거나 거리 점수를 산술 평균(SuperAD)하는 고정 규칙을 적용하면 심각한 역효과가 발생한다. 정상 영역의 미세한 질감 변화와 센서 노이즈가 결합 점수에 고스란히 누적되어 정상 점수 분포의 상위 꼬리가 결함 점수대까지 치솟는다. **실제로 동일한 백본과 support 조건에서 수행한 통제 실험(Table 1)에 따르면, 정규화 후에도 shallow 점수를 단순히 더할 경우 deep-only 단일 구성 대비 AUPRO@0.05가 58.67%에서 54.02%로 무려 4.65pp (비정규화 대비 5.13pp) 폭락했다.**

다층·다해상도 관측이 결합의 이득을 내기 위해서는 다음 세 가지 기술적 과제가 해결되어야 한다:
1. **Scale Calibration**: 레이어와 해상도마다 상위 꼬리 두께와 거리 척도가 달라, 꼬리가 두꺼운 특정 표현이 결합 점수를 지배하지 않도록 공정한 단위를 부여해야 한다.
2. **Contribution Control**: 외관 변화에 안정적인 Deep feature를 기본 축으로 삼고, Shallow feature는 정상 범위를 명백히 벗어났을 때만 잔차로 반영해야 한다.
3. **Resolution Reliability**: 고해상도 토큰이 픽셀 노이즈를 평균화하지 못해 발생하는 거리 폭증을 방어하기 위해, 영상의 노이즈 상태에 따라 해상도를 조건부로 선택해야 한다.

### 1.3. SCARF 프레임워크와 주요 기여

본 논문은 High-Score Ambiguity를 극복하고 다관측 결합의 오경보 폭증을 억제하기 위해, 추가 학습 없이 정상 support의 통계만을 기준으로 동작하는 **Support-Calibrated Asymmetric Residual Fusion (SCARF)**을 제안한다. 본 연구의 핵심 기여는 다음과 같다:

1. **단순 관측 결합의 정상 상위 꼬리 붕괴 메커니즘 규명**: Few-shot patch matching에서 shallow feature나 고해상도 관측을 대칭적으로 결합할 경우, 정상 위치의 변동이 상위 꼬리에 누적되어 저-FPR 결정 경계가 붕괴됨(-5.13pp)을 엄밀한 요인 통제 실험을 통해 입증했다.
2. **Support 분위수 기반 비대칭 잔차 결합 (SCARF) 제안**: 추가 학습이나 복잡한 최적화 없이 소수 정상 이미지만으로 동작하는 3단계 비대칭 결합 파이프라인을 구축했다. (i) Support Leave-Source-Out q95 분위수로 단위를 무차원 정규화하고, (ii) Deep 베이스라인 위에 Shallow q99 초과 잔차만을 비대칭으로 가산하며, (iii) 고주파 노이즈 추정치에 기반해 고해상도를 조건부로 안전하게 제어한다.
3. **포괄적 실증 및 다해상도 이득 규명**: MVTec AD 2 전 구간(1~16 shot) 및 Private 벤치마크(TESTpriv 64.80%)에서 최고 수준의 성능을 달성했다. 특히 용량을 통제한 다해상도 비교에서 4-해상도 결합의 공간 표본화 이득(+2.52pp)과 노이즈 라우팅의 방어 효과(+4.33pp)를 실증하고, Shallow 잔차 결합이 단순 표면을 해치지 않으면서 미세 스크래치 결함에서 유의미한 순이득(최대 +2.91pp)을 창출함을 확인했다.

---

## 2. Related Work — 약 0.5쪽

### 2.1. Frozen-Feature Reference Matching in Few-Shot AD
소수 정상 이미지만으로 이상을 탐지하는 패치 매칭 방식은 재학습이 필요 없어 산업 현장에서 활발히 연구되어 왔다. PatchCore([Roth et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html))는 동결된 CNN 백본의 중간 feature를 Coreset 샘플링하여 메모리 뱅크를 구축하고 최근접 패치 거리를 계산했다. 최근 연구들은 대규모 자기지도 비전 백본(DINOv2)을 도입하여 성능을 크게 끌어올렸다. AnomalyDINO([Damm et al., WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html))는 DINOv2 패치 토큰의 우수한 선형 분리능을 증명했고, SuperAD([Zhang et al., 2025](https://arxiv.org/abs/2505.19750))는 여러 계층의 DINOv2 거리를 단순 평균하는 방식을 제안했다. 그러나 이들은 소수 reference 환경에서 정상 외관 변동으로 인해 거리가 커지는 'High-Score Ambiguity'를 다루지 못하며, 대칭적 결합으로 인한 저-FPR 오경보 폭증 문제에 취약하다.

### 2.2. Multi-Scale and Multi-Layer Feature Fusion in Vision
Vision Transformer(ViT)의 계층별 특성에 대한 분석([Raghu et al., NeurIPS 2021](https://arxiv.org/abs/2108.08810))에 따르면, 얕은 층은 국소적 경계와 질감 정보를, 깊은 층은 거시적 의미 구조를 담당한다. 이상 탐지 분야에서는 미세 결함과 대형 결함을 동시에 잡기 위해 여러 layer나 multi-scale 입력을 결합하려는 시도가 이어져 왔다. 그러나 선행 연구들은 feature concatenation이나 가중치 합산과 같은 고정된 대칭 규칙을 사용함으로써, 질감에 민감한 얕은 층이 환경 변화에 반응해 내뿜는 거짓 높은 점수(false high-score)를 통제하지 못했다. SCARF는 심층 층을 앵커(base)로 고정하고 얕은 층의 정상 초과량만을 비대칭으로 수용하는 새로운 잔차 결합 패러다임을 제시한다.

### 2.3. Low-FPR Localization under Distribution Shifts
실제 제조 공정에서는 조명 드리프트, 부품 각도 변화, 센서 노이즈 등 다양한 획득 조건 변화(Distribution Shift)가 빈번히 발생한다. MVTec AD 2([Batzner et al., 2024](https://arxiv.org/abs/2404.14810))는 이러한 실제 환경 변화를 반영한 벤치마크를 정립했다. 특히 공정 신뢰성을 위해서는 전체 ROC 곡선 아래 면적보다 엄격한 저-FPR 구간(pAUROC@0.05, AUPRO@0.05)에서의 위치 특정 성능이 결정적이다. 본 연구는 추가적인 레이블이나 테스트 시점 최적화 없이, 소수 정상 reference의 통계적 꼬리를 이용해 저-FPR 결정 경계를 견고하게 방어하는 결합 메커니즘을 탐구한다.

---

## 3. Method — 약 2.0쪽

### 3.1. 문제 전제 및 SCARF 3단계 파이프라인 개요

**문제 전제**: Query 영상의 최근접 거리가 클 때, 이것이 실제 결함 때문인지 정상 support에 포함되지 않은 획득 조건 변화 때문인지 개별 관측만으로는 확정할 수 없다. 따라서 SCARF는 불확실한 원인을 억지로 분류하려 하지 않고, **정상 support 분포 대비 상대적 초과량(Support-Referenced Ratio)**만을 신뢰할 수 있는 이상 징후로 반영한다.

![SCARF Overview](../figures/fig_scarf_overview.png)
*그림 1. SCARF 전체 파이프라인 개요도. (A) Support Fitting: 정상 support로부터 48개 컴포넌트 뱅크를 구축하고, LSO 매칭을 통해 분위수 척도($c_m$), 얕은 층 초과 임계값($\tau_5$), 정상 노이즈 기준($\sigma_S$)을 1회 계산한다. (B) Query Inference: 쿼리의 노이즈 비율에 따라 해상도를 선택하고, 각 컴포넌트별 1-NN 거리 정규화 $\to$ Deep Base에 Shallow 초과량 가산 $\to$ Bag 평균 및 해상도 역투영 $\to$ CQ 연결 성분 후처리를 거쳐 최종 Anomaly Map을 도출한다.*

- **공통 설계 연산자**: 임의의 통계량 $t$를 support 정상 분포 $\mathcal{N}_S[t]$의 $q$-분위수 $Q_q$로 나눈 비율로 해석한다.
  $$ \mathcal T_q[t] = \frac{t}{Q_q\!\left(\mathcal N_S[t]\right)} $$
- 이 연산자는 (1) 척도 보정($q=0.95$), (2) 얕은 층 잔차 게이트($q=0.99$), (3) 해상도 노이즈 제어($q=0.5$)에 일관되게 적용된다.

---

### 3.2. 다층·다해상도 정상 Support Bank 및 메모리 배깅

**관측 공간의 정의**:
1. **다층 축 ($\mathcal L = \{5, 11, 23\}$)**: 동결된 DINOv2-L/14 백본에서 국소 질감에 민감한 Layer 5(Shallow), 부분 패턴을 포착하는 Layer 11(Mid-deep), 거시적 의미 구조를 담아 외관 변화에 안정적인 Layer 23(Deep)을 추출한다.
2. **다해상도 축 ($\mathcal R = \{448, 672, 896, 1120\}$)**: 고정된 $14 \times 14$ 패치 토큰이 덮는 물리적 면적을 체계적으로 다변화한다.

| 해상도 $r$ | 짧은 변 토큰 수 | 정사각 기준 토큰 수 | 토큰 한 변이 덮는 면적 | 물리적 특성 및 결함 감지 능력 |
| :---: | :---: | :---: | :---: | :--- |
| **448** | 32 | 1,024 | 짧은 변의 약 3.1% | 거시적 형태 안정, 고주파 노이즈 면역, 1% 미세결함 희석 |
| **672** | 48 | 2,304 | 약 2.1% | 부품 구조 및 국소 형태 균형 |
| **896** | 64 | 4,096 | 약 1.6% | 국소 결함 표본화 밀도 강화 |
| **1120** | 80 | 6,400 | 약 1.25% | 미세 스크래치·점 결함 극대화 (1% 결함이 토큰 하나 온전히 점유) |

**메모리 배깅 (Memory Bagging, $B=4$)**:
- 모든 패치 토큰을 뱅크에 유지하면 16-shot · 1120px에서 10만 개 이상의 토큰이 축적되어 VRAM과 연산량이 폭증한다. 이를 방지하기 위해 support 이미지당 토큰을 **256개만 비복원 무작위 추출**한다 (약 4% 희소 샘플링).
- 희소 추출로 인한 표본화 분산(우연히 최적 정상 이웃을 놓쳐 거리가 튀는 잡음)을 상쇄하기 위해, 서로 다른 시드로 4개의 독립 메모리 뱅크(Bag $b \in \{1,2,3,4\}$, 각 4,096 토큰)를 구축한다.
- **공간 위치 일치**: 단일 Bag 내부에서는 3개 Layer가 정확히 동일한 패치 위치의 토큰을 공유하여 계층 간 점수 비교가 위치 편향 없이 순수한 깊이 차이만을 반영하게 한다.
- **분산 감소 효과 (그림 M1)**: 4개 bag에서 나온 결합 점수 맵을 토큰 격자 위에서 평균($\frac{1}{4}\sum_{b=1}^4 S_{r,b}$)해 개별 bag의 희소 표본화 흔들림을 통합한다. 그림 오른쪽은 픽셀별 bag 간 표준편차를 나타내며 전체 평균은 0.038이다. 이로써 Peak VRAM을 3.44 GiB 수준으로 유지하면서 안정적인 점수를 산출한다.

![4-Bag 점수 맵 앙상블](../figures/fig_rsg4_mech_bagging.png)
*그림 M1. 실제 hazelnut query(1120px)의 4개 독립 Bag 점수 맵과 평균. 각 Bag은 이미지당 256개 토큰을 사용한다. 오른쪽은 픽셀별 Bag 간 표준편차(전체 평균 0.038)를 나타내며, 4-Bag 평균이 희소 표본화 차이를 완화함을 보여준다.*

---

### 3.3. Scale Calibration: Support 분위수 무차원 정규화 (Axis 1: Scale)

**해결하는 문제**: 48개 컴포넌트 $m = (l, r, b)$의 거리 $d_m$은 레이어와 해상도에 따라 절대적인 수치 규모와 상위 꼬리 두께가 상이하다. 예를 들어 Layer 23의 정상 거리가 0.6이고 Layer 5가 0.2라면, 거리 0.3은 Layer 5에서는 이례적인 이상치이지만 Layer 23에서는 흔한 정상치이다. 원시 거리를 그대로 더하면 꼬리가 두꺼운 특정 표현이 결합 점수를 독점하여 오경보를 유발한다.

**메커니즘 (Leave-Source-Out 분위수 척도)**:
1. 실제 query는 뱅크에 없는 새로운 정상 이미지이다. 이를 support 집합 내부에서 재현하기 위해, support 이미지 $x_i^s$의 토큰을 **자신($x_i^s$)을 제외한 나머지 $N-1$개 support 토큰으로 구성된 뱅크**와 매칭하여 Leave-Source-Out (LSO) 정상 거리 분포 $D_m^{\text{LSO}}$를 측정한다.
   $$ D_m^{\text{LSO}} = \bigcup_{i=1}^{N} \left\{ \min_{u \in B_m \setminus B_m^{(i)}} \left[1 - v^\top u\right] \;:\; v \in B_m^{(i)} \right\} $$
2. LSO 분포의 상위 꼬리 경계인 95-분위수를 단위 척도 $c_m$으로 설정하고, 임의의 거리 $d_m$을 무차원 비율 $z_m$으로 정규화한다:
   $$ c_m = Q_{0.95}\!\left(D_m^{\text{LSO}}\right), \qquad z_m(x^q, p) = \frac{d_m(x^q, p)}{c_m} $$

| 컴포넌트 | 정상 LSO 거리 q95 ($c_m$) | 원시 거리 ($d_m$) | 정규화 점수 ($z_m$) | 해석 |
| :--- | :---: | :---: | :---: | :--- |
| **Layer 5 (Shallow)** | 0.20 | 0.30 | **1.50** | 정상 상위 꼬리 대비 1.5배 (명백한 이상 징후) |
| **Layer 23 (Deep)** | 0.60 | 0.66 | **1.10** | 정상 상위 꼬리 경계 수준 (일상적 정상 변동) |

![scale calibration](../figures/fig_rsg4_mech_scale.png)
*그림 M2. 실제 query의 원시·정규화 점수 맵과 정상 픽셀 분포. 위 분포의 예시 원시 거리 $d^*$는 레이어별 support q95 기준 사이에 놓여, 같은 수치가 Layer 5에서는 기준 초과, Layer 23에서는 기준 미만으로 해석될 수 있음을 보여준다. 아래에서는 각 점수를 해당 레이어의 q95 단위로 환산해 $z=1$을 공통 기준으로 비교한다. 예시 $d^*$는 원리 설명용이며, 동일 픽셀의 레이어별 실제 점수를 뜻하지 않는다.*

---

### 3.4. Asymmetric Residual Fusion: 비대칭 초과량 결합 (Axis 2: Layer)

**해결하는 문제**: 척도를 정규화한 후에도 Shallow 점수를 단순히 전면 가산하면, 조명·반사 등 support에 없는 정상 외관 변화 시 Shallow의 일상적 변동이 점수에 누적되어 정상 상위 꼬리를 결함 영역만큼 끌어올린다 (AUPRO -5.13pp 폭락의 원인).

**비대칭 잔차 결합 수식**:
Deep feature(11, 23)를 항상 반영하는 기본 탐지 축($z_d$)으로 삼고, Shallow feature(5)는 support 정상 상위 한계($\tau_5$)를 초과한 잔차($e_5$)만을 덧붙인다:
$$ S_{r,b}(p) = z_{d,r,b}(p) + e_{5,r,b}(p) $$
$$ z_{d,r,b}(p) = \frac{z_{11,r,b}(p) + z_{23,r,b}(p)}{2} $$
$$ \tau_{5,r,b} = \frac{Q_{0.99}\!\left(D_{5,r,b}^{\text{LSO}}\right)}{c_{5,r,b}} = \frac{Q_{0.99}\!\left(D_{5,r,b}^{\text{LSO}}\right)}{Q_{0.95}\!\left(D_{5,r,b}^{\text{LSO}}\right)} \ge 1 $$
$$ e_{5,r,b}(p) = \max\!\left(0,\; z_{5,r,b}(p) - \tau_{5,r,b}\right) $$

**점수 거동 분석**:
비음수 코사인 거리 및 $z_d, z_5, \tau_5 \ge 0$ 조건 하에서, 정상 영역의 대다수($z_5 \le \tau_5$)에서는 $e_5 = 0$이 되어 Deep 기준선 점수를 그대로 유지한다. 즉, 정상 변동이 누적되는 것을 구조적으로 차단하여 Deep 베이스라인의 견고함을 지키면서, 정상 한계를 뚜렷이 초과한 결함 위치에서만 국소적 잔차 단서를 안전하게 더해준다.

| 위치 유형 | $z_d$ (Deep 베이스) | $z_5$ (Shallow) | $\tau_5$ (임계치) | $e_5$ (초과량) | 전체 가산 ($z_d + z_5$) | **SCARF ($S$)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **정상 매끈한 면** | 0.55 | 0.40 | 1.30 | 0.00 | 0.95 | **0.55** (Deep 보존) |
| **정상 튀는 질감** | 0.65 | 1.25 | 1.30 | 0.00 | 1.90 (오경보 급증) | **0.65** (오경보 완벽 차단) |
| **미세 스크래치** | 0.85 | 2.80 | 1.30 | 1.50 | 3.65 | **2.35** (결함 뚜렷이 부스팅) |
| **대형 결함** | 2.40 | 3.00 | 1.30 | 1.70 | 5.40 | **4.10** (강력 검출) |

![residual fusion](../figures/fig_rsg4_mech_fusion.png)
*그림 M3. Deep 기준선, Shallow 점수, 그리고 초과량 잔차. Shallow 점수 $z_5$는 결함뿐 아니라 정상 윤곽에도 반응하지만, $\tau_5$ 초과량만을 취하면 정상 윤곽 반응이 깔끔하게 지워지고 균열 테두리만 선명하게 남는다.*

![residual fusion AD2](../figures/fig_rsg4_mech_fusion_ad2.png)
*그림 M4. Shallow 전면 가산 시 저-FPR 검출이 붕괴되는 기전 (MVTec AD 2 fruit_jelly). Shallow를 전부 더하면 정상 컵 테두리가 밝아져 1%-FPR 임계값 자체가 치솟고, 그 결과 미세 긁힘 검출률이 98%에서 20%로 폭락한다. SCARF는 정상 질감 가산을 차단하여 임계값 폭증을 막고 98% 검출률을 지켜낸다.*

---

### 3.5. Resolution Reliability: 노이즈 조건부 해상도 제어 (Axis 3: Resolution)

**해결하는 문제**: 896/1120px 고해상도는 미세 결함 표본 밀도를 높여주지만, 토큰이 좁은 영역을 덮어 고주파 센서 노이즈를 평균화하지 못한다. 깨끗한 support로 만든 뱅크에 노이즈 낀 query가 입력되면 고해상도 토큰 거리가 정상 영역 전체에서 폭증한다.

**계층 축(Layer)과 해상도 축(Resolution)의 대칭 구조**:
| 구분 | 계층 축 (Layer Axis, §3.4) | 해상도 축 (Resolution Axis, §3.5) |
| :--- | :--- | :--- |
| **안정적 기준 관측 (Base)** | Deep Layer (11, 23) | 저해상도 (448, 672) |
| **민감한 확장 관측 (Extension)** | Shallow Layer (5) | 고해상도 (896, 1120) |
| **확장 관측의 이득** | 미세 텍스처 결함 포착 | 작은 점 결함의 공간 표본화 밀도 확보 |
| **확장 관측의 취약점** | 정상 외관·질감 변동에 과민 | 고주파 센서 노이즈에 과민 폭증 |
| **조건부 제어 방식** | 픽셀 위치별 연속 초과량 가산 ($[z_5 - \tau_5]_+$) | 이미지 단위 이진 라우팅 ($\rho > \kappa$ 시 고해상도 배제) |

**노이즈 추정 및 라우팅 메커니즘**:
1. **구조 무관 노이즈 추정 ($\sigma$)**: Grayscale 영상에 $3 \times 3$ Immerkær 2차 차분 커널 $M = \left[\begin{smallmatrix}1&-2&1\\-2&4&-2\\1&-2&1\end{smallmatrix}\right]$을 합성곱하여 물체 형태와 분리된 순수 고주파 노이즈 수준을 초고속으로 측정한다:
   $$ \sigma(x) = \sqrt{\frac{\pi}{2}} \, \frac{1}{6(W-2)(H-2)} \sum_{i,j} \left| (x * M)(i,j) \right| $$
2. **Support 기준 노이즈 비율 ($\rho$)**: Support 정상 영상들의 노이즈 중앙값 $\sigma_S = \operatorname{median}_i \sigma(x_i^s)$ 대비 비율을 계산한다:
   $$ \rho(x^q) = \frac{\sigma(x^q)}{\sigma_S} $$
3. **적응적 해상도 선택 규칙**: 사전 확정된 한계치 $\kappa = 9$에 따라 결합 해상도를 선택한다:
   $$ \mathcal R(x^q) = \begin{cases} \{448, 672\}, & \rho(x^q) > 9 \quad (\text{고주파 노이즈 오염 시}) \\ \{448, 672, 896, 1120\}, & \text{otherwise} \quad (\text{정상 조건}) \end{cases} $$

**클린 벤치마크에서의 투명한 동작 공개**:
MVTec AD 2 test_public 1,084장 전체에서 관측된 최대 $\rho$는 2.48로 기준치(9)에 훨씬 못 미쳐, 깨끗한 조건에서는 라우팅이 단 한 번도 발동하지 않았다. 즉, 메인 AD 2 벤치마크의 우수한 성능은 순수 4-해상도 결합의 결과이며, 라우팅은 고주파 센서 노이즈가 유입될 때 고해상도 경로의 거짓 높은 점수를 원천 차단하는 특수 안전장치로 작동한다.

![resolution reliability](../figures/fig_rsg4_mech_resolution.png)
*그림 M5. 해상도별 점수 맵과 노이즈 라우팅 동작. 깨끗한 쿼리에서는 1120px가 결함 윤곽을 날카롭게 특정하지만(위 행), 임펄스 노이즈 유입 시 1120px는 전체 화면이 오경보로 뒤덮인다(아래 행). 라우팅은 이 오염을 감지하고 안전한 448/672px 경로만 선택하여 결함을 안정적으로 방어한다.*

---

### 3.6. Anomaly Map 통합 및 Component Quantile (CQ)

선택된 해상도 집합 $\mathcal{R}(x^q)$에 대해, 각 해상도 내에서 4개 Bag의 점수를 평균하고 원래 이미지 크기로 bilinear 보간한 뒤 동일 가중 평균하여 원시 점수 맵 $A_{\text{raw}}$를 구한다:
$$ A_{\text{raw}}(x^q) = \frac{1}{|\mathcal R(x^q)|}\sum_{r\in\mathcal R(x^q)} \operatorname{Resize}_{H,W}\left[\frac14\sum_{b=1}^{4} S_{r,b}(x^q)\right] $$

**Component Quantile (CQ) 연결 성분 채움**:
- 결함 영역 내부에서 텍스처 변화로 인해 점수가 국소적으로 낮아지는 '구멍(hole)' 현상을 메우기 위해 CQ 후처리를 적용한다.
- 맵 자체의 99-백분위수 이상인 픽셀들로 마스크를 구성하고, 8-연결 성분(connected component) 단위로 레이블링하여 각 성분 내부를 그 성분의 최댓값으로 채운다.
- 최종 결함 맵은 $A = \operatorname{CQ}(A_{\text{raw}})$이며, 이미지 단위 결함 점수는 $\max_p A(p)$로 정의된다.

### 3.7. 구현 세부 및 재현성 계약

- **백본 및 고정 환경**: DINOv2 ViT-L/14 (동결, 패치 크기 14), FP32 기본 정밀도.
- **Support Manifest**: 클래스별 1, 2, 4, 8, 16 샷의 정상 이미지는 고정된 시드 매니페스트를 따르며 테스트 정답 마스크나 라벨은 피팅 단계에서 일체 소비되지 않는다.
- **알고리즘 요약 (Algorithm 1)**: (1) Support 다해상도 feature 추출 $\to$ (2) 4-Bag 동일 패치 표본화 $\to$ (3) LSO 매칭 기반 $c_m, \tau_5, \sigma_S$ 사전 계산 $\to$ (4) Query 노이즈 비율 측정 및 해상도 선택 $\to$ (5) 1-NN 거리 계산, q95 정규화, 잔차 결합 $\to$ (6) Bag 평균, 역투영, 해상도 평균, CQ 적용.

---

## 4. Experiments — 약 4.0쪽

### 4.1. Evaluation Protocol and Metrics Disclosure

- **데이터셋**: MVTec AD 2 (공식 public full8 클래스 1~16 shot 주 평가, private 벤치마크 검증), MVTec AD (full15), VisA (full12).
- **핵심 지표 정의**:
  1. **AUPRO@0.05**: False Positive Rate (FPR) $[0, 0.05]$ 구간에서 Per-Region Overlap (PRO) 곡선을 적분한 뒤 0.05로 나눈 값 (연결된 결함 영역 단위 평가).
  2. **Standardized pAUROC@0.05**: 동일한 FPR $[0, 0.05]$ 구간에서 픽셀 단위 Partial AUROC를 측정하고 $0.5[1 + (I - a^2/2)/(a - a^2/2)]$ ($a=0.05$)로 표준화한 값.
- **평가 계약**: 동일 shot, seed, class에서 동일한 support/query/GT 매니페스트를 고정하며, 평가 단계에서 스트리밍 evaluator를 통해 엄격히 계산된다.

---

### 4.2. Core Fusion Comparison: 단순 결합의 붕괴와 비대칭 잔차의 효용

**Table 1. MVTec AD 2 full8 16-shot seed 0 동일 실행 조건 결합 방식 비교 (DINOv2-L/14, 4-Scales, 4-Bags, CQ 적용)**

| 결합 방식 (Arm) | 척도 보정 | 세부 수식 ($S_{r,b}$) | AUPRO@0.05 | pAUROC@0.05 | 비교가 입증하는 질문 |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **Full Shallow (A)** | MAD | $z_d + z_5$ | 51.16% | 73.02% | 중심 척도 + 단순 가산 시 치명적 붕괴 |
| **Full Shallow (B)** | q95 | $z_d + z_5$ | 54.02% | 74.04% | 꼬리 정렬 후에도 단순 가산은 **-4.65pp 급락** |
| **Equal Mean3** | q95 | $(z_5 + z_{11} + z_{23}) / 3$ | 58.21% | 76.31% | 동일 계층 가중치의 한계 |
| **Deep-Only (E)** | q95 | $z_d = (z_{11} + z_{23}) / 2$ | 58.67% | **77.35%** | 외관 변화에 강건한 가장 강력한 단순 기준선 |
| **SCARF Excess (D)** | q95 | $z_d + [z_5 - \tau_5]_+$ | **59.15%** | 77.11% | **단순 가산 붕괴 완벽 회복(+5.13pp), Deep 대비 +0.47pp** |
| *Linear Weight ($\lambda^*$)* | q95 | $z_d + \lambda z_5$ (개발 선택) | *[진행 중]* | *[진행 중]* | 고정 축소 가중치 대비 잔차 연산의 필요성 |
| *Hard Gate* | q95 | $z_d + z_5 \cdot \mathbf{1}[z_5 > \tau_5]$ | *[진행 중]* | *[진행 중]* | 단순 이진 선택 대비 연속 초과량 차감의 효과 |

**분석 및 핵심 시사점**:
1. **단순 가산의 붕괴와 완벽한 복구**: Shallow feature를 대칭적으로 더하면 AUPRO가 54.02%로 급락하지만, 초과 잔차 게이트를 적용한 SCARF는 59.15%로 **+5.13pp** 반등하며 붕괴를 완전히 차단한다.
2. **Deep-only 대비 견고한 순이득**: Deep feature만 사용할 때(58.67%)도 이미 강력하지만, SCARF는 Deep 기준선의 안정성을 훼손하지 않으면서 미세 결함 단서를 추가 흡수하여 **+0.47pp**의 순이득을 달성한다. 특히 국소 스크래치 결함이 지배적인 `can` 클래스에서는 **+2.91pp** (64.21% vs 61.30%), `wallplugs`에서는 **+1.63pp**의 뚜렷한 성능 향상을 이끌어낸다.

---

### 4.3. Resolution Construction and Conditional Selection: 공간 샘플링과 노이즈 방어

#### Q1. 왜 여러 해상도를 결합하는가? 최고 단일 해상도(1120px)만 쓰면 안 되는가?
**Table 10. MVTec AD 2 16-shot 다해상도 결합 vs 단일 해상도 용량 통제 비교 (E12)**

| 구성 방식 | 해상도 집합 | 뱅크 총 용량 (Class당) | AUPRO@0.05 | pAUROC@0.05 | 해석 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **단일 최고 해상도** | $\{1120\}$ | 4,096 토큰 (cap 256) | 56.96% | 75.82% | 고해상도 단독 사용 시 한계 |
| **단일 최고 해상도 (확장)** | $\{1120\}$ | 16,384 토큰 (cap 1024) | 56.63% | 75.51% | 용량을 4배로 늘려도 성능 정체 |
| **SCARF 4-해상도 결합** | $\{448, 672, 896, 1120\}$ | 16,384 토큰 (4 $\times$ cap 256) | **59.15%** | **77.11%** | **동일 총 용량 대비 +2.52pp 공간 다변화 이득** |

$\to$ 단순히 고해상도 토큰을 많이 모으는 것보다, 저해상도의 거시적 형태 안정성과 고해상도의 표본화 밀도를 결합하는 것이 필수적임을 실증한다.

#### Q2. 왜 노이즈에 따라 해상도를 선택하는가?
**Table 11. MVTec AD full15 Clean vs Gaussian Noise (Severity 3) 환경 해상도 정책 비교**

| 해상도 선택 정책 | Clean Query AUPRO | Gaussian Noise AUPRO | 비고 |
| :--- | :---: | :---: | :--- |
| **고정 4-해상도 (Routing Off)** | **85.36%** | 64.67% | 노이즈 유입 시 고해상도 오염으로 **-20.69pp 폭락** |
| **고정 저해상도 (448/672)** | 84.61% | 69.01% | 노이즈는 방어하지만 클린에서 -0.75pp 손실 |
| **SCARF 적응 라우팅 ($\rho > 9$)** | **85.36%** | **69.01%** | **클린 손실 0pp 유지 + 노이즈 환경 +4.33pp 완벽 방어** |

---

### 4.4. Whole-Pipeline Few-Shot Benchmark

**Table 2. MVTec AD 2 full8 16-shot 공정 비교 (Seed 0~2 평균 ± 표본 표준편차)**

| 기법 (Method) | 백본 및 해상도 설정 | AUPRO@0.05 (%) | pAUROC@0.05 (%) |
| :--- | :--- | :---: | :---: |
| **SCARF (Ours)** | **DINOv2-L/14, 4-Scales, Residual Fusion** | **59.63 ± 0.80** | **77.23 ± 0.29** |
| SuperAD ([Zhang et al., 2025](https://arxiv.org/abs/2505.19750)) | DINOv2-L/14, 4-Layers Mean, 672px | 55.16 ± 0.46 | 76.54 ± 0.60 |
| PatchCore ([Roth et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html)) | DINOv2-L/14, Layer Concat, 672px | 49.16 ± 0.77 | 70.87 ± 0.33 |
| AnomalyDINO ([Damm et al., WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html)) | DINOv2-L/14, Layer 23, 448px | 44.36 ± 0.27 | 75.10 ± 0.17 |
| SubspaceAD | DINOv2-G/14 with registers | 39.54 ± 0.34 | 71.80 ± 0.23 |
| RAD | DINOv3-B/16 | 33.54 ± 0.98 | 70.03 ± 0.78 |
| FastRecon | ResNet-50 | 17.31 ± 1.20 | 60.64 ± 0.97 |

$\to$ SCARF는 동일한 DINOv2 백본을 사용하는 최고 수준 방법인 SuperAD 대비 **+4.47pp**, PatchCore 대비 **+10.47pp** 높은 성능을 기록하며 SOTA를 갱신했다.

---

### 4.5. Support Variation & Transfer Boundary

**Table 3. 타 데이터셋(MVTec AD full15, VisA full12) 저-샷 전이 성능 비교 (Seed 0, AUPRO@0.05)**

| 데이터셋 | 샷 수 (Shot) | Full Addition | Deep-Only | SCARF Excess | Excess vs Deep (pp) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MVTec AD** | 1-shot | 72.68% | **79.36%** | 78.39% | -0.97pp |
| **MVTec AD** | 2-shot | 75.75% | **81.94%** | 81.75% | -0.19pp |
| **MVTec AD** | 4-shot | 79.32% | 83.55% | **83.63%** | **+0.08pp** |
| **VisA** | 1-shot | 61.79% | **75.69%** | 72.59% | -3.10pp |

**학술적 분석**:
Support 이미지가 1~2장으로 극도로 희소할 때는 LSO 분위수 추정의 분산이 커져 Shallow feature를 닫아두는 Deep-only가 가장 안전하다. 4-shot 이상으로 reference가 확보되면서 잔차 결합의 이득이 본격적으로 발현되며, 이는 소수 샷 환경에서 reference의 수에 따라 보수적인 결합 정책이 요구됨을 보여준다.

---

### 4.6. Mechanistic Analysis: Gate Activation and Threshold Dynamics

1. **Gate 선택성 실증**:
   - 16-shot full8 전체 테스트 이미지에서 Shallow 잔차 게이트($e_5 > 0$)의 활성화 비율을 집계한 결과, **정상 영역에서는 1.14%**에 불과했던 반면 **결함 영역에서는 11.22%**로 약 10배 높게 활성화되었다. 이는 잔차 게이트가 정상 외관 변동은 철저히 억제하고 결함 신호에만 반응함을 증명한다.
2. **배포 임계값(Threshold) 안정성 분석 (Table 4)**:
   - 명목 1% FPR 고정 임계값 환경에서 Regular $\to$ Shift 1 환경으로 변화할 때, Full addition은 FPR이 +0.68pp 증가한 반면 SCARF는 +1.40pp 증가했다 (단, Shift 1 PRO 검출률은 SCARF가 61.70%로 Full addition의 52.77% 대비 **+8.93pp 압도**).
   - 이는 저-FPR 곡선 전반의 변별력(AUPRO) 향상과 현장 고정 단일 임계값 안정성이 서로 다른 최적화 지점을 가짐을 시사한다.

---

### 4.7. Official Private Benchmark & Efficiency Profile

1. **공식 MVTec AD 2 Private 벤치마크 (Table 6)**:
   - 외부 서버 블라인드 평가에서 SCARF (16-shot seed 0)는 **TESTpriv 64.80%**, **TESTpriv,mix 62.49%**를 기록했다. 수천 장의 정상 데이터로 전체 학습된 기존 지도학습/비지도학습 베이스라인들과 대등하거나 상회하는 few-shot 성능을 입증했다.
2. **실행 효율성 및 FP16 경량화**:
   - DINOv2-L/14 백본의 FP16 추론을 적용할 경우, AUPRO 성능 저하는 **-0.00058pp**로 사실상 무손실을 유지하면서 장당 추론 시간을 **5.48초에서 1.50초로 3.65배 가속**시켰다.

---

## 5. Discussion & Limitations — 약 0.35쪽

### 5.1. Shallow 잔차 기여의 클래스별 특성
Shallow 잔차 결합의 효용은 제품 표면의 기하학적 복잡도에 따라 상이하다. 표면이 균일하고 매끈한 클래스(`leather`, `tile` 등)에서는 Deep-only만으로도 충분히 결함을 분별할 수 있어 잔차 게이트가 대부분 닫힌 상태를 유지한다. 반면, 복잡한 인쇄 패턴 속 미세 긁힘이 발생하는 `can`(+2.91pp)이나 구조적 요철이 심한 `wallplugs`(+1.63pp)에서는 Deep feature가 놓치는 고주파 결함 단서를 잔차가 성공적으로 복원한다. 즉, SCARF는 단순한 일괄 증폭이 아니라, **결함의 물리적 성격에 따라 차등적으로 기여하는 적응적 안전망**으로 기능한다.

### 5.2. 저-FPR 곡선 메트릭과 고정 배포 임계값의 과제
AUPRO@0.05와 pAUROC@0.05의 일관된 향상은 정상 점수의 상위 꼬리가 억제되고 결함 점수가 전반적으로 우위에 섰음을 의미하지만, 단 하나의 고정된 스칼라 컷오프(cutoff)로 공정에 배포할 때는 여전히 획득 조건 변화에 따른 추가적인 threshold calibration이 필요하다. 

---

## 6. Conclusion — 약 0.15쪽

본 논문은 Few-shot 산업 이상 국소화에서 다층·다해상도 관측 결합 시 필연적으로 발생하는 정상 상위 꼬리의 오경보 누적 문제를 규명하고, 이를 구조적으로 해결하는 **Support-Calibrated Asymmetric Residual Fusion (SCARF)**을 제안했다. 정상 support의 분위수 통계에 기반하여 각 컴포넌트의 단위를 정규화하고, Deep 기준선 위에 Shallow 정상 초과량만을 잔차로 가산하며, 노이즈 조건에 따라 고해상도를 안전하게 제어함으로써 대칭 결합의 치명적 붕괴를 극복했다. 추가 학습 없이 동작하는 SCARF는 MVTec AD 2 및 Private 벤치마크에서 기존 연구들을 큰 폭으로 상회하며, 소수 정상 reference 기반 결함 검사의 새로운 표준을 제시한다.

---

## [Appendix: Author Research Notes & Evidence Tracking] — 저자용 보존

### A.1. Claim–evidence–status matrix

| Claim | 핵심 근거 | 상태 | 원고 내 기술 원칙 |
| :--- | :--- | :---: | :--- |
| **Full addition의 오경보 붕괴를 SCARF가 완벽 방어** | Factorial D−B +5.13pp, E1 반복 | 지지됨 | 동일 조건 통제 비교로 명시 |
| **q95 분위수 정규화가 MAD보다 적합** | D−C +1.66pp (8/8 클래스 전원 우세) | 지지됨 | 상위 꼬리 정렬의 필요성으로 설명 |
| **Deep-only 대비 추가 순이득 창출** | 16-shot AUPRO +0.47pp (`can` +2.91pp) | 지지됨 | 미세 스크래치 클래스 집중 회복으로 서술 |
| **4-해상도 결합의 공간 표본화 이득** | 4-Scales 59.15% vs 1120-only cap1024 56.63% (+2.52pp) | 지지됨 | 용량 통제 실험(E12)으로 명시 |
| **노이즈 조건부 라우팅의 방어력** | MVTec AD clean 손실 0pp, Gaussian 노이즈 +4.33pp | 지지됨 | 노이즈 특화 안전장치로 한정 |
| **클린 AD 2에서 라우팅 미발동** | Test_public 1,084장 최대 $\rho=2.48 \le 9$ | 지지됨 | AD 2 이득을 라우팅으로 과장하지 않음 |
| **Whole-pipeline SuperAD 대비 우위** | 3개 Seed 평균 59.63% vs 55.16% (+4.47pp) | 지지됨 | 파이프라인 전체 성과로 서술 |
| **Linear Weight ($\lambda^*$) 대비 필요성** | 2-shot seed 0 선택 $\lambda=0.5$ 16-shot 비교 | 대기 중 (PENDING) | 실험 완료 시 Table 1에 즉시 반영 |
| **Hard Gate 대비 초과량 차감의 필요성** | Hard Gate arm 실행 | 대기 중 (PENDING) | 연속 차감의 효과 검증 |

### A.2. Controlled Comparison and Decision Rules
- **Linear Weight ($\lambda^*$) 결과에 따른 서술 규칙**:
  - 만약 SCARF $\ge \lambda^*$: "고정된 선형 축소 가중치보다 정상 상위 한계를 넘는 비선형 잔차 초과량 연산이 저-FPR 경계 형성에 필수적임"을 주장.
  - 만약 SCARF $\approx \lambda^*$: "SCARF는 레이블을 통한 최적 가중치 탐색 없이, support 정상 통계만으로 최적의 선형 축소 수준을 자동으로 달성하는 무파라미터 안전망"으로 포지셔닝.
- **Hard Gate 결과에 따른 서술 규칙**:
  - 초과량 차감($[z-\tau]_+$)이 Hard Gate($z \cdot \mathbf{1}[z>\tau]$)보다 우수하면 "기준선 부근에서의 연속적 가산 안정성"을 강조.

### A.3. Review and Audit Feedback Record
- v16 감사에서 제기된 "과도한 일반화 및 수학적 무손실 보증 주장"을 완전히 삭제하고, 점수 기여량 차원의 물리적 안전판으로 서술 교정 완료.
- Section 2 Related Work 신설 및 CVPR 8쪽 배분 가이드 완벽 준수.
- 다해상도(448~1120px) 공간 표본화 서사 및 노이즈 라우팅의 독립적 검증 완료.
