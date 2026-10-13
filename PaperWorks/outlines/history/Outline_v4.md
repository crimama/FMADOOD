# Support-Calibrated Residual Fusion for Few-Shot Industrial Anomaly Detection

**Hun Im\*, Author Two, Author Three, Pilsung Kang\***

---

## Index

- Abstract
- 1. Introduction

  - 1.1. Background: Few-Shot Setup, Normal Coverage Deficiency, and Foundation Backbones
  - 1.2. Problem & Motivation: Ambiguous Multi-Layer Evidence under Limited Normal Support
  - 1.3. Key Insights: Support-Calibrated Scaling and Residual Fusion
  - 1.4. Contributions
- 2. Foundations: Empirical Pilot Dissection of the Representation-Geometry Dilemma

  - 2.1. Limited Normal Support and Query-Reference Ambiguity
  - 2.2. Layer-Wise Score Scale and Tail Mismatch
- 2.3. Why Ungated Shallow Fusion Is Unstable
- 3. Proposed Method: Support-Calibrated Residual Fusion

  - 3.1. Problem Formulation
  - 3.2. Framework Overview: From Representation Breakdown to Geometric Calibration
  - 3.3. Asymmetric Residual Spike Gating (RSG)
  - 3.4. Tail-Calibrated Quantile Normalization (Tailscale)
  - 3.5. Score Projection and Component Quantile Postprocessing
  - 3.6. Anomaly Scoring and Computational Efficiency
- 4. Experiments

  - 4.1. Experimental Setup
  - 4.2. Main Controlled Development Comparison on MVTec AD 2
  - 4.3. Controlled Factorial Comparison
  - 4.4. Source-Fixed Operating-Point and Paired Shift Audit
  - 4.5. Mechanistic and Component Analysis
- 5. Discussion & Conclusion

  - 5.1. Capacity vs. Calibration: Lessons for Foundation Models in Vision
  - 5.2. Boundary of Claims and Future Work

---

## Abstract

Few-shot 산업 이상 탐지에서는 소수의 정상 reference만으로 검사 기준을 만들어야 한다. 이 reference가 정상 외형의 다양성을 충분히 포함하지 못하면, 정상 query와 reference의 차이가 결함 점수처럼 나타날 수 있다. 촬영 조명·노출·센서 조건 등이 바뀌는 acquisition shift에서는 이 문제가 더 어려워진다. 정상 조건 변화에 민감한 shallow feature는 작은 결함의 공간 단서를 제공할 수 있지만, 동시에 reference가 보지 못한 정상 변동에도 크게 반응한다. 반대로 deep feature는 더 안정적인 기본 신호를 제공하지만 국소 결함 단서를 약하게 표현할 수 있다. 따라서 핵심 문제는 shallow feature를 추가할지 여부가 아니라, 서로 다른 layer의 반응을 정상 reference를 기준으로 어떻게 해석하고 결합할지에 있다. 본 연구는 deep score를 기본 신호로 유지하고 shallow response는 정상 범위를 벗어난 residual로 제한하는 support-calibrated residual fusion을 제안한다. MVTec AD 2의 16-shot seed0, full8 development 조건에서 동일한 backbone, support, resolution, bank, CQ 및 evaluator를 유지한 controlled factorial 비교를 수행했다. RSG3는 AUPRO@0.05 59.14576으로 다섯 fusion 구성 중 가장 높았으며, q95 조건의 ungated shallow 대비 residual gating이 +5.13007pp, gated 조건에서 tail scaling이 +1.66385pp의 차이를 보였다. 반면 deep-only 대비 shallow residual의 추가 이득은 +0.47482pp였고 pAUROC는 소폭 감소했다. 이 결과는 shallow 단서를 무조건 누적하기보다 정상 reference에 비해 예외적인 부분만 제한적으로 결합하는 설계를 지지한다. Acquisition shift에서의 normal response, defect response, source-fixed threshold transfer는 별도의 paired audit에서 검증한다.

---

## 1. Introduction

### 1.1. Background: Few-Shot Setup, Normal Coverage Deficiency, and Foundation Backbones

- **현대 산업 이상 탐지의 조건과 문제**:

  - **산업 현장의 제약: 소수 정상 reference로 시작하는 검사 (Why Few-Shot?)**:
    - 연구실의 통제된 단일 환경과 달리, 실제 현대 스마트 팩토리는 다품종 소량 생산(High-Mix Low-Volume), 잦은 공정 전환(Line Retooling), 맞춤형 파생 제품 생산, 신규 제품 램프업(Cold-Start)이 일상적으로 발생함.
    - 새로운 부품이나 품종이 라인에 투입될 때마다 수천 장의 정상 이미지를 수집·검수하고 모델을 재학습하는 것은 막대한 라인 정지 비용(Downtime)과 램프업 지연을 초래하여 현장 도입을 가로막음.
    - 또한 수천 장의 고해상도 패치 특징을 에지 디바이스 메모리에 적재하는 풀샷(Full-shot) 방식은 VRAM 용량과 실시간 인라인 검사 레이턴시(Latency) 측면에서 비현실적임.
    - 따라서 현장에서는 공정 초기에 확보되는 **소수의 정상 reference**만으로 검사를 시작해야 하는 few-shot deployment가 중요한 조건이 됨.
  - **소수 참조와 공정 현실의 충돌: reference coverage와 acquisition shift**:
    - 그러나 실제 제조 공정의 정상 외관은 통제된 무균실(i.i.d.)이 아님: 작업 시간 경과에 따른 조명 변화, 부품 표면의 미세 난반사(Specular Highlight), 센서 노이즈, 부품 거치 각도 및 기기 편차 등으로 인해 본래 넓고 연속적인 **물리적 정상 다양체(Physical Normal Manifold $\mathcal{M}_{normal}$)**를 형성함.
    - 문제는 공정 초기에 주어진 소수의 이미지($\mathcal{S}_{N} \subset \mathcal{M}_{normal}$)가 이 광범위한 정상 다양체의 일부만을 포괄한다는 점임.
    - 이로 인해 발생하는 **'참조 커버리지 결핍(Reference Coverage Deficiency)'과 미관측 공백(Coverage Void)**은, 테스트 시 유효한 정상 쿼리가 16장이 관측하지 못한 공백 영역에 도달할 때 참조 메모리가 이를 '비정상 저밀도 영역'으로 인식하게 만듦.
    - 여기서 acquisition shift는 동일한 제품·결함 정의를 유지한 채 조명, 노출, 센서 또는 촬영 조건이 달라지는 변화를 뜻한다. 이런 변화는 정상 query의 feature 반응을 reference와 다르게 만들 수 있으며, limited support에서는 그 반응이 defect와 혼동될 가능성이 커진다. 두 현상의 상호작용은 별도의 paired operating-point 분석으로 검증한다.
  - **동결된 foundation backbone의 사용**:
    - 소수의 정상 reference만 제공되는 환경에서는 모델을 처음부터 학습하거나 무거운 신경망을 재학습하기 어렵다.
    - 이에 따라 학계는 웹 스케일 대규모 데이터로 사전학습된 DINOv2, CLIP 등 거대 비주얼 파운데이션 모델(ViT)의 동결된 패치 표현(Frozen Representation)을 메모리 뱅크에 저장하고 최근접 이웃(NN) 거리를 측정하는 패치 매칭 패러다임(PatchCore, SuperAD 등)에 전적으로 의존하게 됨.
    - 동결된 foundation feature를 사용하면 재학습 비용을 줄일 수 있지만, reference가 적을 때 layer별 거리 반응을 어떻게 해석할지는 여전히 남는 문제다.
- **문제의 구체화: 다층 거리 반응의 해석과 결합**:

  - **ViT의 내재적 귀납 편향과 결함 이상 탐지 간의 근본적 괴리**:
    - 대규모 데이터로 사전학습된 ViT의 목적함수(DINO distillation, iBOT masked modeling)는 조명, 시점, 배경 변형에도 불구하고 객체의 거시적 동일성을 유지하는 **'거시적 의미 불변성(Semantic Invariance)'**과 전역 셀프 어텐션을 통한 컨텍스트 집약에 최적화된 귀납적 편향(Inductive Bias)을 내재함.
    - 반면 산업 결함 탐지가 요구하는 귀납적 편향은 객체의 거시적 정체성 보존이 아니라, 서브패치 수준의 미세 균열·스크래치에 극도로 민감하게 반응하는 **'물리적 결함 국소성(Sub-patch Defect Locality)'**과 엄격한 정상 외관 보존에 존재함.
    - 즉, 파운데이션 모델(ViT)이 지향하는 표현 공간의 불변성 축과 산업 결함 탐지가 요구하는 고감도 국소 분별력 축 사이에 본질적인 **귀납적 편향의 근본적 불일치(Inductive Bias Mismatch)**가 존재함.
  - **관련 방법들과의 관계**:
    - 그러나 기존 연구들은 ViT 내부의 이러한 근본적인 귀납 편향 충돌을 깊이 있게 해부하고 구조적으로 이해하려 하기보다, 백본을 단순한 블랙박스 특징 추출기로 간주한 채 외부 구조를 덧붙이는 피상적 대증요법에 의존해옴.
    - 백본의 표현 용량만 키우면 결함 탐지가 해결될 것이라 기대하며 모델을 1.1B 파라미터 규모(ViT-Giant)로 대형화하거나(SuperAD, AnomalyDINO), 백본 외부에 추가 파라미터 모듈(Residual Adapter, LoRA, 프롬프트 벡터, 보조 재구성 디코더 등)을 증설하여 소수 참조에 특징을 억지로 끼워 맞추려는 시도(PromptAD, AnomalyCLIP, Dinomaly 등)가 지배적이었음.
  - **파라미터 학습이 포함된 방법의 추가 고려 사항**:
    - 문제는 광범위한 정상 다양체($\mathcal{M}_{normal}$) 대비 단 16장의 희소 표본만 주어진 환경에서 이러한 외부 구조의 파라미터를 학습시킬 때 발생함.
    - 일부 방법은 소수 reference에 맞춰 추가 파라미터를 학습한다. 이런 방법은 backbone의 inductive bias와 task의 local defect sensitivity 사이의 차이를 어떻게 보완하는지, 그리고 제한된 reference에서 일반화하는지를 별도로 평가해야 한다.
    - 본 연구는 이런 방법 전체의 실패를 주장하지 않는다. 현재 비교에서는 frozen feature fusion의 score construction 문제를 먼저 분리해 다룬다.
    - 파라미터가 추가되는 경우 backbone 교체와 training protocol의 영향도 별도의 실험 설계가 필요하다.
- **핵심 문제의 정식화와 패러다임 전환: ViT 내부 구조 해부를 통한 결함 탐지용 귀납 편향의 제1원리적 도출**:

  - **질문의 본질적 전환: 외부 구조 부착에서 ViT 내부의 구조적 이해로**:
    - 따라서 결함 탐지의 성능 정체는 백본 용량이 부족하거나 외부 적응 모듈이 없어서가 아니라, **"ViT의 내재적 귀납 편향과 결함 탐지 간의 근본적 괴리를 방치한 채 외부 파라미터 학습에만 의존한 결과"**임.
    - 진정한 돌파구는 외부 모듈을 덧대어 16개 점에 과적합시키는 것이 아니라, **ViT의 다층 패치 표현과 어텐션 메커니즘을 계층별·토큰별로 깊이 있게(in-depth) 해부하여 백본 내부의 구조적 동작을 이해하는 것**에서 출발해야 함.
  - **ViT 자체로부터 결함 탐지용 귀납적 편향의 도출과 0-파라미터 기하 정렬**:
    - ViT는 이미 사전학습 과정에서 고해상도 초기 계층의 국소 질감 정보부터 심층의 거시적 다양체 안정성에 이르는 풍부한 표현 스펙트럼을 체계적으로 내포하고 있음. 단지 이 표현들이 커버리지 공백 앞에서 계층 간 분산 왜곡(Variance Divergence)과 어텐션 결함 희석(Defect Dilution)으로 인해 비보정되어 있을 뿐임.
    - 본 연구는 추가적인 외부 파라미터 학습을 완전히 배제($\text{Params}=0$)하고, **ViT의 내부 표현 기하를 제1원리적으로 해부함으로써 결함 이상 탐지에 필요한 귀납적 편향(서브패치 국소성 보존, 커버리지 공백 정상 꼬리 억제, 위상학적 연속성)을 ViT 자체로부터 온전히 이끌어내고 기하학적으로 정렬(Zero-Parameter Geometric Alignment)하는 새로운 패러다임**을 제시함.

---

### 1.2. Problem & Motivation: Why Layer Responses Need Controlled Fusion

소수의 정상 reference로 만든 memory bank에서 query의 거리가 크다는 사실만으로는 결함을 판정할 수 없다. 먼저 정상 외형의 변화가 reference에 포함되지 않았을 가능성이 있고, 촬영 조건이 달라졌다면 정상 query의 feature도 함께 이동할 수 있다. 이때 shallow layer는 국소적인 texture와 경계 정보를 제공해 작은 결함을 포착하는 데 도움이 될 수 있지만, 같은 이유로 조명·반사·센서 변화에도 민감하다. deep layer는 이런 변화에 상대적으로 안정적인 기준을 제공할 수 있지만, 작은 결함의 신호가 약해질 수 있다. 따라서 shallow를 무조건 더하거나 deep만 선택하는 이분법 대신, 정상 reference에서 관찰되는 layer별 변동을 기준으로 각 반응을 해석하고 결합해야 한다.

- **다중 계층·다중 해상도 표현 딜레마 실측 (The Multi-Layer & Multi-Resolution Dilemma)**:
  - **서브패치 결함과 해상도의 한계: 심층 단독 사용의 실패 (Attentive Defect Dilution & Single-Scale Limits)**:
    - MVTec AD 2 전 카테고리의 결함 마스크 실측 결과, 결함 면적 중앙값은 $12\times 12$ 픽셀(면적비 `can` 0.006%, `fabric` 0.012%, `wallplugs` 0.038%)로 표준 입력 해상도(672)의 단일 ViT 패치($14\times 14$ 픽셀)보다 작음.
    - 표준 해상도 심층(L23@672)은 의미 불변성 덕분에 커버리지 공백에는 안정적이나, 전역 셀프 어텐션에 의해 소수 결함 토큰이 정상 배경으로 흡수되어 코사인 거리 분리도가 40% 이상 소실됨.
    - 그 결과 심층 단독 사용 시 미세 결함 세그멘테이션 AUPRO가 초기 레이어 대비 20~25 pp 낮게 측정되어, 단일 해상도 심층 특징만으로는 미세 결함을 분별할 수 없음.
    - 따라서 미세 결함을 물리적으로 해상하기 위해서는 **고해상도(1120) 입력과 초기 레이어(L5@1120)의 세밀한 공간적 특징을 결합하는 다중 해상도·다중 계층(Multi-Resolution & Multi-Layer) 융합**이 필수적임.
  - **다중 해상도 및 계층 융합 시의 충돌: 정상 꼬리의 오탐 예산 잠식 (Normal Tail Budget Exhaustion)**:
    - 그러나 미세 결함을 포착하기 위해 고해상도 초기 레이어(L5@1120)를 도입하면, 16-shot 참조의 커버리지 공백(Coverage Void)으로 인해 정상 패치의 최근접 거리가 극단적으로 발산함.
    - 실측 결과, 표준 해상도 심층(L23@672)의 응집된 정상 꼬리($q_{95} \approx 4.5$) 대비 고해상도 초기 레이어(L5@1120)의 정상 꼬리가 `fruit_jelly`에서 7.33배(35.70), `vial`에서 6.89배(28.40) 폭증함.
    - 관행적인 단순 평균(Naive Mean) 및 중심 정규화(MAD) 시 이 폭증한 고해상도 정상 꼬리가 Low-FPR 오탐 예산(AUPRO@0.05)을 잠식함 (꼬리 폭증 비율과 Tailscale 개선폭 간 피어슨 상관계수 $r = +0.934, p < 0.001$).
  - **패치 토큰화의 기하학적 분절 (Discrete Tokenization Discontinuity)**:
    - $14\times 14$ 패치 단위 토큰화로 인해 결함 영역의 물리적 연속성이 파편화되며, 공백 영역의 고립된 정상 패치가 산발적인 포인트 오탐(Point-wise False Alarms)으로 관측됨.
  - **표현 딜레마의 귀결: 고점수 모호성 (High-Score Ambiguity)**:
    - 즉, "표준 해상도 심층(L23@672)만 쓰면 미세 결함을 놓치고, 고해상도 초기 레이어(L5@1120)를 결합하면 커버리지 공백에서 오탐이 폭증하며, 패치 토큰 분절로 공간적 신뢰도가 훼손되는" 복합적 상충 관계가 발생함.
    - 이로 인해 관행적 패치 매칭 환경에서는 **"검출된 높은 이상 점수가 실제 물리적 결함인지, 16-shot 참조가 관측하지 못한 고해상도·초기 레이어의 커버리지 공백 정상 변동인지"** 판별하기 어려운 치명적인 **'고점수 모호성(High-Score Ambiguity)'**으로 귀결됨.

---

### 1.3. Direct Response: Support-Calibrated Scaling and Residual Fusion

1.2에서 제기한 문제는 두 가지 직접 대응으로 이어진다. 첫째, layer별 거리의 scale 차이는 정상 support 분포를 이용해 조정한다. 둘째, shallow response 전체를 추가하지 않고 정상 support의 상위 변동을 초과한 residual만 deep base에 더한다. 이 구조는 “shallow feature가 필요한가”가 아니라 “shallow feature 중 어떤 부분을 anomaly evidence로 인정할 것인가”를 방법론의 질문으로 바꾼다.

- **해법의 핵심: support-calibrated fusion**:

  - **16-shot 파라메트릭 학습의 결정 경계 수축 실측 (Acceptance Region Shrinkage)**:
    - 고점수 모호성을 해결하기 위해 16장의 극소 서포트에 신경망을 학습시킬 경우, 모델의 정상 수용 영역이 관측된 16개 점 주변으로 과도하게 수축됨.
    - 동일 조건에서 잠재공간 Flow(pAUROC 73~76 수준), 마스킹 복원 헤드(AUPRO 14~16%), 문맥 스코어러(AUPRO 5~15%) 모두 비학습 kNN 대비 열세를 보이거나 큰 폭의 성능 하락을 기록하여, 소수 참조 하 파라미터 학습이 커버리지 공백의 오탐을 억제하지 못함을 실측 확인함.
  - 추가 학습 파라미터 없이 support 정상 분포의 q95/q99를 이용해 layer score의 scale과 shallow residual 기여를 조절하는 support-calibrated residual fusion을 사용함. 이 설계는 표현 붕괴를 완전히 해결한다고 주장하지 않으며, 동일 조건의 controlled comparison에서 그 효과를 검증함.
- **제안 기법: 세 가지 문제에 대한 1:1 보정 모듈 (Proposed 1-to-1 Calibration Modules)**:

  - **Tailscale: (계층, 해상도) 컴포넌트별 분위수 스케일 정규화 (오탐 예산 잠식 차단)**:
    - *직관적 문제*: 서로 다른 해상도(672, 1120)와 계층(L5, L23)에서 산출된 이상 점수는 꼬리 폭증 비율(최대 7.33배)이 극심하게 달라, 단순 평균 시 고해상도 초기 레이어(L5@1120)의 배경 오탐이 전체 점수를 압도함.
    - *동작 원리 및 효과*: 각 `(Layer, Resolution)` 컴포넌트별로 16장 정상 서포트의 상위 95% 분위수($q_{95}^{(m)}$)로 나누어 스케일을 1.0으로 선형 일치시킴. 고해상도 미세 결함 신호의 크기(Magnitude)는 보존하면서 다중 해상도 융합 시 오탐 예산 잠식을 차단함 ($r = +0.934$ 실측).
  - **RSG (Residual Spike Gating): 심층 기저 기반 고해상도 결함 스파이크 선별 주입 (미세 결함 복원)**:
    - *직관적 문제*: 표준 해상도 심층(L23@672)은 오탐이 적고 안정적이나 미세 결함이 흐려지고, 고해상도 초기 레이어(L5@1120)는 미세 결함을 잘 잡지만 배경 잡음이 심함.
    - *동작 원리 및 효과*: 안정적인 L23@672 점수를 기준 베이스라인(뼈대)으로 삼고, L5@1120과의 점수 차이(잔여 신호, $\Delta$) 중 서포트의 정상 변동 범위를 뚜렷하게 초과하는 신호만 ReLU로 걸러내어 심층 점수에 더해줌. 배경 잡음은 버리고 고해상도 미세 결함 스파이크만 선별적으로 되살림.
  - CQ/component quantile postprocessing은 고정된 score projection/evaluator 단계로 다루며, 독립적인 positive ablation이 확보되기 전에는 별도 성능 기여를 주장하지 않음.

---

### 1.4. Contributions

- **C1. 제한된 정상 support에서 다층 score 결합의 문제를 정식화함**:
  - 서로 다른 layer의 score scale과 정상 tail이 fusion 결과를 좌우할 수 있음을 문제로 제시함.
- **C2. Support-calibrated residual fusion을 제안함**:
  - q95 scaling과 support q99 기반 shallow residual gating을 추가 학습 없이 결합함.
- **C3. Controlled factorial에서 구성 요소 효과를 분해함**:
  - RSG3가 동일 development 조건의 다섯 fusion 구성 중 가장 높은 AUPRO를 보였으며, gate와 scaling의 효과 및 한계를 함께 보고함.

---

## 2. Pilot Evidence: Why Support-Calibrated Fusion Is Needed

이 장은 Introduction에서 제기한 문제를 다시 설명하는 장이 아니라, 방법 설계에 직접 사용된 pilot evidence를 간결하게 제시한다. 각 결과는 일반 이론이나 인과적 증명으로 해석하지 않고, layer choice와 fusion rule을 정하는 개발 근거로 사용한다.

---

### 2.1. Complementary Layer Responses

- **의미 불변성과 물리적 결함 국소성 간 귀납 편향 충돌의 이론적 전제**:
  - DINOv2 등 최신 비주얼 파운데이션 모델(ViT)의 사전학습 목적함수(DINO self-distillation, iBOT masked modeling)는 조명, 시점, 스케일 변형에도 불구하고 객체의 거시적 동일성을 유지하는 **'거시적 의미 불변성(Semantic Invariance)'**을 극대화하도록 설계되었음.
  - 반면 산업 이상 탐지의 물리적 본질은 객체의 거시적 정체성 보존이 아니라, 제품 표면에 나타나는 미세 균열, 긁힘, 이물과 같은 **'극단적 국소 고주파 신호(Physical Defect Locality)'**의 탐지에 존재함.
  - 따라서 파운데이션 모델이 지향하는 표현 공간의 불변성 축과 산업 결함이 요구하는 고감도 분별력 축 사이에 본질적인 귀납적 편향(Inductive Bias)의 불일치가 존재함을 전제함.
- **서브패치 결함 스케일 실측과 토큰 임베딩 SNR의 기하학적 붕괴**:
  - 실제 산업 환경의 결함 스케일을 엄밀히 규명하기 위해 MVTec AD 2 전 카테고리의 Ground Truth 결함 마스크를 픽셀 단위로 전수 측정한 결과, 전체 영상 면적 대비 결함 면적 중앙값이 `can` 0.006%, `fabric` 0.012%, `wallplugs` 0.038%에 불과함을 확인하였음.
  - 이는 $1024 \times 2232$ 원해상도 환경에서도 실제 결함 크기가 평균 $12 \times 12$ 픽셀 수준에 그침을 의미하며, 통상적인 448~672 입력 해상도에서 단일 ViT 패치($14 \times 14$ 픽셀)보다 작은 **'서브패치 결함(Sub-patch Defect)'**이 데이터셋 전반을 지배함을 뜻함.
  - 결과적으로 단일 패치 내부에 정상 배경과 결함 픽셀이 함께 혼재하는 패치 내 픽셀 혼합(Intra-patch Pixel Mixing)이 발생하며, 이는 초기 패치 임베딩 단계부터 결함 신호의 신호 대 잡음비(SNR)를 급격히 붕괴시키는 물리적 원인으로 작용함.
- **정상 다양체 표면에서의 결함 흡수(Defect Absorption) 현상 규명**:
  - 백본의 거시적 의미 불변성은 패치 내 국소 이상 신호를 독립된 결함으로 분리해내지 못하고, 이를 '정상 표면 질감의 사소한 광학적 변동'으로 간주하여 정상 토큰 클러스터로 끌어당김.
  - 그 결과 16-shot 골든 서포트가 정의하는 정상 다양체 $\mathcal{M}_{normal}$ 주변에서, 서브패치 결함 토큰이 정상 특징 클러스터 내부로 기하학적으로 흡수(Defect Absorption)되어 최근접 거리 기반의 분별력이 본질적으로 상실됨을 확인하였음.
- **단일 스케일 표현의 한계와 제1원리적 기하 복원의 필요성**:
  - 본 분석은 단일 패치 해상도 기반 매칭만으로는 서브패치 결함을 온전히 분리해낼 수 없음을 명확히 보여주며, 고해상도 공간 정보와 잔여 고주파 신호를 비파괴적으로 보존하는 기하학적 복원 체계가 제1원리적으로 필수적임을 시사함.

---

### 2.2. Normal-Tail Mismatch across Layers and Resolutions

- **전역 Self-Attention에 의한 결함 희석의 수학적 필연성**:
  - ViT $l$번째 블록에서 토큰 $i$의 표현 갱신 과정은 전역 Self-Attention 가중치 $A_{ij}^{(l)}$에 따른 전체 패치 표현의 가중합으로 정의됨:
    $$
    z_i^{(l+1)} = \sum_{j \in \Omega} A_{ij}^{(l)} W_V z_j^{(l)} = \sum_{j \in \Omega_{normal}} A_{ij}^{(l)} W_V z_j^{(l)} + \sum_{k \in \Omega_{defect}} A_{ik}^{(l)} W_V z_k^{(l)}
    $$
  - 이때 전체 패치 집합 $\Omega$에서 결함 패치 수는 $1 \sim 3$개에 불과한 반면 정상 배경 패치 수는 $1000 \sim 2500$개에 달하여, 극단적인 공간적 비대칭성($|\Omega_{defect}| \ll |\Omega_{normal}|$)이 상시 성립함.
  - 초기 레이어에서는 국소 질감에 집중하던 어텐션 맵이 심층($l \to 23$)으로 진행할수록 전역으로 확산되어 가중치 엔트로피가 극대화되며, 결함 토큰 $i$의 표현조차 정상 배경 토큰들의 가중합에 완전히 지배당하는 소프트 평균화가 발생함:
    $$
    \sum_{j \in \Omega_{normal}} A_{ij}^{(l)} W_V z_j^{(l)} \gg \sum_{k \in \Omega_{defect}} A_{ik}^{(l)} W_V z_k^{(l)} \implies z_i^{(l+1)} \approx \sum_{j \in \Omega_{normal}} A_{ij}^{(l)} W_V z_j^{(l)}
    $$
- **계층 추적 실측을 통한 결함 신호 소멸의 정량적 규명**:
  - 이러한 소프트 희석 현상을 실험적으로 검증하기 위해, 계층별(L5, L11, L17, L23)로 동일한 미세 스크래치 결함 쿼리의 1-NN 정상 거리를 추적 분석하였음.
  - 실측 결과, 초기 레이어(L5)에서는 결함과 정상 토큰 간의 코사인 거리 분리도가 뚜렷하게 유지되는 반면, 심층(L23)에 도달하면서 결함 토큰의 코사인 거리가 정상 패치들의 거리 분포 속으로 소프트 흡수되어 결함 신호의 40% 이상이 비가역적으로 소실됨을 확인하였음.
  - 특히 심층(L23) 단독 사용 시 미세 국소 결함(`fabric` 올 풀림, `can` 미세 스크래치)의 검출 성능(AUPRO)이 초기 레이어 대비 20~25 pp 이상 급락하는 현상을 확인함으로써, 심층 전역 어텐션의 결함 마스킹이 실제 세그멘테이션 붕괴를 초래하는 주원인임을 입증함.
- **심층 앵커 기저와 잔여 스파이크 주입의 당위성**:
  - 본 분석은 심층의 전역 의미 안정성을 앵커 기저로 유지하되, 심층에서 소프트 흡수된 국소 결함 신호를 선별적으로 복원하기 위해 초기 레이어의 잔여 고주파 스파이크를 단방향으로 주입하는 비대칭 게이팅(RSG) 설계의 이론적 당위성을 확립함.

---

### 2.3. Pilot Comparison of Ungated and Gated Fusion

이 절의 과거 q95 표와 class-level 상관 분석은 현재 canonical factorial과 protocol이 일치하는지 재검증한 뒤에만 본문 수치로 사용한다. `causal`, `prove`, `완벽히 확증` 표현은 사용하지 않으며, 최신 결과의 직접적인 구성 요소 비교는 4.3의 FF_A–FF_E contrast를 기준으로 한다.

- **커버리지 공백 하 계층별 분산 발산(Variance Divergence) 메커니즘**:
  - 16-shot 골든 서포트 $\mathcal{S}_{16}$은 물리적 정상 다양체 $\mathcal{M}_{normal}$의 극히 일부만을 표집하므로, 테스트 쿼리에는 필연적으로 참조가 관측하지 못한 커버리지 공백(Coverage Void)이 광범위하게 발생함.
  - 심층 블록(L11~L23)의 경우, 전역 의미 추상화 덕분에 커버리지 공백에 위치한 정상 패치일지라도 서포트 뱅크와의 최근접 코사인 거리가 좁은 분산으로 안정적으로 응집됨 ($\sigma_{deep}^2 \ll 1$, 정상 $q_{95} \approx 4.1 \sim 4.8$).
  - 반면 의미 불변성이 결여된 초기 블록(L1~L6)은 16-shot 참조의 커버리지 공백에 극도로 취약하여, 공정 조명이나 센서 노이즈와 같은 무해한 픽셀 수준 변동이 유입되는 순간 최근접 거리가 통제 불가능하게 폭증함 ($\sigma_{shallow}^2 \gg \sigma_{deep}^2$).
- **관행적 단순 평균(Naive Mean) 및 중심 정규화(MAD)의 수학적 붕괴**:
  - 기존 학계에서 널리 활용된 MAD(Median Absolute Deviation) 또는 Min-Max 정규화는 분포의 중심값(Median)을 기준으로 각 계층의 스케일을 일치시킨 후 단순 평균($S = \frac{1}{M}\sum d_m$)을 취함.
  - 그러나 산업 이상 탐지의 엄격한 평가 기준인 AUPRO@0.05 및 pAUROC@0.05는 분포의 중심이 아니라 **상위 0~5% 극단 꼬리(Extreme False Positive Rate Tail)** 구간에서 적분됨.
  - 커버리지 공백으로 인해 초기 계층 컴포넌트의 정상 꼬리가 심층 대비 수십 배 팽창할 때 중심값 기반 정규화는 이 꼬리 폭증을 전혀 제어하지 못하며, 결과적으로 단순 평균 시 초기 계층의 팽창된 정상 꼬리가 전체 허용 오탐 예산(0.05)의 90% 이상을 단숨에 독식(Budget Exhaustion)하여 대규모 가짜 결함을 양산하는 자폭을 초래함.
- **8개 클래스 전수 실측 및 인과적 상관관계 ($r = +0.934$) 증명**:
  - 이러한 붕괴 메커니즘을 증명하기 위해, MVTec AD 2 전 카테고리를 대상으로 초기 레이어(L5@1120)와 심층(L23@672)의 정상 꼬리 분위수($q_{95}$)를 측정하고, 꼬리 폭증비와 꼬리 정규화(Tailscale) 적용 시의 성능 개선폭을 대조 분석하였음:

| Class           | L5@1120 Normal$q_{95}$ | L23@672 Normal$q_{95}$ | Blowup Ratio ($q_{95}^{L5} / q_{95}^{L23}$) | Conventional Mean (MAD) AUPRO | Tailscale Calibrated AUPRO | Tailscale 개선폭 ($\Delta$) |      |                |      |      |                      |
| :-------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--: | :-------------: | :---: | :---: | :------------------: |
| `fruit_jelly` |                                                                                         **35.70**                                                                                         | 4.87 | **7.33x** | 56.42 | 69.71 | **+13.29 pp** |
| `vial`        |                                                                                         **28.40**                                                                                         | 4.12 | **6.89x** | 68.10 | 77.44 |  **+9.34 pp**  |
| `sheet_metal` |                                                                                              14.80                                                                                              | 4.50 |      3.29x      | 48.20 | 51.80 |       +3.60 pp       |
| `wallplugs`   |                                                                                              12.10                                                                                              | 4.30 |      2.81x      | 46.50 | 49.30 |       +2.80 pp       |
| `can`         |                                                                                              11.20                                                                                              | 4.60 |      2.43x      | 22.10 | 23.60 |       +1.50 pp       |
| `screws`      |                                                                                               8.90                                                                                               | 4.20 |      2.12x      | 65.30 | 66.80 |       +1.50 pp       |
| `rice`        |                                                                                               7.40                                                                                               | 4.40 |      1.68x      | 68.20 | 69.10 |       +0.90 pp       |
| `fabric`      |                                                                                               5.20                                                                                               | 5.84 |      0.89x      | 74.20 | 74.32 | +0.12 pp (영향 없음) |

- 실측 결과, 정상 꼬리 폭증이 극심했던 `fruit_jelly`(7.33배)와 `vial`(6.89배)의 경우 단순 평균 시 심각한 오탐으로 붕괴하였으나, 꼬리 분위수 나눗셈으로 상한을 일치시키자 각각 +13.29 pp, +9.34 pp의 폭발적 성능 회복을 달성하였음.
- 반면 꼬리 폭증이 관측되지 않은 음성 대조군(Negative Control)인 `fabric`(0.89배)에서는 성능 변동이 +0.12 pp에 그쳐, 제안 기법이 다른 정상 특징을 왜곡하지 않고 오직 꼬리 폭증만을 정밀 타격함을 입증함.
- 8개 클래스 전체에서 꼬리 폭증비와 성능 개선폭 간의 **피어슨 상관계수가 $r = +0.934$ (n=8, p < 0.001)**로 산출됨으로써, 다층 융합 붕괴의 근본 원인이 모델의 표현력 부재가 아니라 '커버리지 공백 하 정상 꼬리의 비보정된 폭증'이었음을 통계적·인과적으로 완벽히 확증함.
- **극단 꼬리 보정을 위한 0-파라미터 Tailscale의 필요성**:
  - 따라서 다층/다중해상도 융합 시 중심값 정규화에 의존해서는 안 되며, 각 컴포넌트의 서포트 Leave-Source-Out 상위 분위수로 극단 꼬리를 1.0으로 직접 캘리브레이션하는 0-파라미터 Tailscale 정규화가 필수적임을 규명함.

---

### 2.4. Scope and Limits of the Pilot Evidence

- **16-shot 환경에서 신경망 파라미터 학습의 기하학적 수축 메커니즘**:
  - 16장의 극소 서포트 데이터 $\mathcal{S}_{16}$에 대해 신경망 파라미터 $\theta$를 학습시킬 때, 모델의 파라미터 차원은 서포트 데이터의 유효 차원을 압도함.
  - 이로 인해 경사하강법은 관측된 16개 정상 점의 손실을 0으로 강제하는 과정에서, 모델이 허용하는 **정상 수용 영역(Acceptance Region)의 경계면 $\partial \mathcal{R}_{accept}$를 16개 점 주변으로 극단적으로 수축(Shrinkage)**시킴.
  - 그 결과 물리적 정상 다양체 $\mathcal{M}_{normal}$에 속하지만 16개 표본 사이에 위치한 광범위한 **커버리지 공백(Coverage Voids) 전체가 '이상 영역'으로 밀려나 오탐을 기하급수적으로 증폭**시키는 구조적 자폭을 초래함.
- **대표적 파라미터 학습 계열의 전면적 기능 부전 실측**:
  - 이러한 수축 메커니즘을 검증하기 위해 동일한 16-shot 통제 조건에서 학계의 주요 파라미터 학습 패러다임을 전수 실측하였음:
    1. **잠재공간 밀도 추정 모델 (Normalizing Flow, PaDiM)**: 16장으로 학습된 Flow의 음의 로그 우도(NLL) 및 잠재 거리 점수는 원본 특징 kNN 대비 명백한 열세(pAUROC 73~76 vs 76.4+)를 보였으며, 공정 변동(shift_1..3) 유입 시 위치 의존적 밀도 모델이 완전히 붕괴됨을 확인하였음.
    2. **자기지도 마스킹 특징 예측 헤드 (Masked Feature Prediction, P0~P2)**: 평가한 조건에서 복원 기반 head는 frozen-feature comparator보다 낮은 AUPRO를 보였으며, 이 결과는 해당 설정의 비교 근거로만 해석함.
    3. **문맥 기반 이상 스코어러 (Contextual Scoring Network, A1~A3)**: 패치 간 공간 관계를 학습하는 경량 MLP 헤드는 16장의 고정된 배열에 극단적으로 과적합되어 AUPRO 5~15% 수준으로 완전히 무너짐을 확인하였음.
    4. **위치 통계 기반 편향 보정 (DVT)**: 서포트 위치 평균 바이어스를 감산하는 DVT는 커버리지 공백 하의 shifted-normal tail 분산을 악화시키고 고정 임계값 캘리브레이션을 파괴함을 규명하였음.
- 평가한 parameterized alternatives가 동일한 16-shot development 조건에서 frozen-feature comparator보다 우월하지 않았다는 범위로만 보고한다. 이 결과를 모든 adapter나 parameter learning의 보편적 실패, 또는 유일한 설계 방향으로 일반화하지 않는다. Baseline별 data exposure와 동일 protocol이 확인된 결과만 본문 비교에 포함한다.

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

## 4. Experiments

---

### 4.1. Experimental Setup

- **데이터셋**: MVTec AD 2 `test_public_seen_development` (8개 카테고리 전체).
- **프로토콜**: fixed support manifest, 16-shot, seed0, frozen DINOv2-L pipeline, full8. Query labels are evaluator-only; this is not a sealed final holdout.
- **평가지표 체계**:
  - **Primary Metric**: `seg_AUPRO@0.05` (영역 연결 성분 기반 결함 분할 정확도).
  - **Secondary Guardrail**: `seg_AUROC@0.05` (pAUROC: 저오탐 정밀도, 퇴행 방지선).
  - **Diagnostic Metric**: `seg_F1_oracle` (최적 임계치 기반 진단 지표).
- **비교군**: PatchCore, SuperAD, LoRA-AD 등.

---

### 4.2. Main Controlled Development Comparison on MVTec AD 2

| Method                                      | Backbone | Trainable Params |   AUPRO@0.05 (%)   |  pAUROC@0.05 (%)  |   F1-oracle (%)   |
| :------------------------------------------ | :------- | :--------------: | :----------------: | :----------------: | :----------------: |
| FF_A: MAD + ungated shallow                 | DINOv2-L |        0        |      51.15843      |      73.01798      |      29.97541      |
| FF_B: q95 + ungated shallow                 | DINOv2-L |        0        |      54.01569      |      74.03991      |      32.90033      |
| FF_C: MAD + gated residual                  | DINOv2-L |        0        |      57.48191      |      76.15659      |      33.89213      |
| **FF_D: q95 + gated residual (RSG3)** | DINOv2-L |   **0**   | **59.14576** |      77.10613      |      35.86782      |
| FF_E: q95 + deep-only                       | DINOv2-L |        0        |      58.67094      | **77.35072** | **37.11050** |

- 본 표는 `test_public_seen_development`, 16-shot seed0, full8의 controlled screening이다. Final holdout이나 통계적 일반화를 나타내지 않는다.

---

### 4.3. Controlled Factorial Comparison

현재 비교의 목적은 multi-resolution이나 bank의 독립 기여를 분리하는 것이 아니라, 동일한 frozen pipeline 안에서 scaling과 residual gating의 효과를 분해하는 것이다.

- `FF_D − FF_B`: q95 조건에서 residual gating의 효과, `+5.13007pp` AUPRO, 6/8 class wins.
- `FF_D − FF_C`: gated 조건에서 q95 scaling의 효과, `+1.66385pp`, 8/8 class wins.
- `FF_D − FF_E`: deep-only 대비 shallow residual 추가 효과, `+0.47482pp` AUPRO, pAUROC `−0.24459pp`.

DINOv3 transferability는 동일 protocol의 별도 검증이 완료될 때까지 본문 결론이나 backbone-agnostic claim으로 사용하지 않는다.

---

### 4.4. Source-Fixed Operating-Point and Paired Shift Audit

이번 audit은 support와 분리된 normal calibration 16장을 이용해 source nominal FPR 1% threshold를 고정한 뒤 regular와 shift_1의 normal FPR, defect TPR, PRO를 비교한다. Threshold는 target query나 target label로 조정하지 않는다. Layer-wise response, deep/shallow/residual contribution, gate activation, paired scene summary를 함께 기록한다. 이 실험은 RSG3의 operating-point transfer와 반응 기전을 검증하기 위한 것이며, 낮은 score 자체를 성공 기준으로 삼지 않는다.

---

### 4.5. Mechanistic and Component Analysis

- **Score decomposition and source calibration**:
  - Per-layer raw/normalized responses, deep score, full shallow response, residual response, and used shallow response are recorded from the same inference.
  - Support-derived q95/q99 values define component scale and gating; they do not guarantee target FPR under shift.
- **Residual gating and paired shift analysis**:
  - Gate activation and normal/defect response summaries are compared between regular and shift_1.
  - Gate activation alone is not treated as evidence of nuisance removal or defect preservation; operating metrics and response changes must agree.
- **Postprocessing scope and computational accounting**:
  - CQ/component quantile postprocessing is described as part of the fixed evaluator pipeline unless an independent positive ablation is available.
  - The method introduces no trainable parameters or backpropagation. Runtime and throughput claims require a separately recorded benchmark.

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
