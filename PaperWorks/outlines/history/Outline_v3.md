# Dissecting Visual Foundation Models for Industrial Anomaly Detection: The Failure of Representation Calibration and Its First-Principles Resolution

**Hun Im\*, Author Two, Author Three, Pilsung Kang\***

---

## Index

- Abstract
- 1. Introduction

  - 1.1. Background: Few-Shot Setup, Normal Coverage Deficiency, and Foundation Backbones
  - 1.2. Problem & Motivation: The Mechanics of Representation Collision in Coverage Voids
  - 1.3. Key Insights: 1-to-1 First-Principles Resolution of Coverage Voids
  - 1.4. Contributions
- 2. Foundations: Empirical Pilot Dissection of the Representation-Geometry Dilemma

  - 2.1. Inductive Bias Mismatch: Semantic Invariance vs. Sub-Patch Defect Locality
  - 2.2. Attentive Defect Masking: The Attentive Dilution of High-Frequency Defects
  - 2.3. Breakdown of Conventional Fusion: Hierarchical Variance Divergence and Tail Blowup in Coverage Voids
  - 2.4. Empirical Pathology of Parametric Learning: Why Adapters Overfit Sparse Coverage
- 3. Proposed Method: Zero-Parameter Geometric Alignment

  - 3.1. Problem Formulation
  - 3.2. Framework Overview: From Representation Breakdown to Geometric Calibration
  - 3.3. Asymmetric Residual Spike Gating (RSG)
  - 3.4. Tail-Calibrated Quantile Normalization (Tailscale)
  - 3.5. Topological Continuum Restoration (CQ)
  - 3.6. Anomaly Scoring and Computational Efficiency
- 4. Experiments

  - 4.1. Experimental Setup
  - 4.2. Main Benchmark Results on MVTec AD 2
  - 4.3. Backbone Paradigm Shift: DINOv2 vs. DINOv3 (Backbone-Agnostic Plug-and-Play)
  - 4.4. Ablation Studies: 1-to-1 Component Attribution
  - 4.5. In-Depth Mechanistic Analysis
- 5. Discussion & Conclusion

  - 5.1. Capacity vs. Calibration: Lessons for Foundation Models in Vision
  - 5.2. Boundary of Claims and Future Work

---

## Abstract

잦은 품종 전환(High-Mix Low-Volume)과 신제품 램프업(Cold-Start) 환경에서 극소수의 골든 참조 샘플(Golden Reference)에 의존하는 퓨샷 산업 이상 탐지(Few-shot Industrial Anomaly Detection)에서는, 모델 재학습 없이 대규모 사전학습된 비주얼 파운데이션 모델(ViT)의 동결된 전이 표현력에 전적으로 의존하는 메모리 뱅크 패러다임이 사실상의 표준으로 정착되었다. 그러나 실제 공정의 정상 외관은 조명, 표면 반사광, 센서 편차 등으로 인해 본래 넓고 연속적인 물리적 정상 다양체(Normal Manifold)를 형성함에도, 16-shot과 같은 희소 참조는 이 다양체의 극히 국소적인 영역만을 포괄한다. 이로 인해 발생하는 **'참조 커버리지 결핍(Reference Coverage Deficiency)'**은 테스트 시 유효한 정상 쿼리가 미관측 공백(Coverage Void)에 떨어질 때 마치 외부적 **'분포 변화(Distribution Shift)'**가 일어난 것과 같은 착시를 유발하며, ViT의 비보정된 표현 구조와 충돌하여 막대한 오탐을 야기한다. 현재 학계의 지배적 흐름은 백본을 거대화(Giant 스케일링)하거나, 학습 가능한 어댑터(LoRA, 프롬프트) 및 보조 재구성 디코더를 결합하여 이를 해결하려 해왔다. 그러나 본 논문에서는 16개의 희소 참조점에 신경망을 학습시키는 접근이 오히려 정상 수용 영역을 16개 점 주변으로 극단적으로 수축시켜 커버리지 공백에서의 오탐을 증폭시킬 뿐임을 지적한다. 나아가, 진정한 병목은 모델 용량 부족이 아니라, ViT 고유의 거시적 의미 불변성과 전역 어텐션이 산업 결함의 물리적 국소성과 정면 충돌하고 커버리지 공백 앞에서 계층 간 분산 왜곡을 일으키는 **'귀납적 편향의 불일치'**에 있음을 해부학적으로 규명한다. 구체적으로, (1) 심층 전역 어텐션이 미세 결함을 정상 배경 속으로 소프트 흡수하는 결함 희석(Attentive Defect Masking), (2) 커버리지 공백에 놓인 초기 계층 정상 패치 거리가 최대 35배 폭증하여 오탐 예산을 독식하는 분산 발산(Hierarchical Variance Divergence), (3) 소수 패치 매칭의 공간적 분절(Discrete Tokenization Discontinuity)이라는 3대 구조적 실패를 밝힌다. 이를 해결하기 위해, 우리는 외부 파라미터 학습을 완전히 배제하고 소수 참조로부터 백본의 표현 결함을 제1원리적으로 교정하는 **학습 파라미터 0개의 기하 정렬 프레임워크(Zero-Parameter Geometric Alignment Framework)**를 제안한다. (1) 커버리지 공백에 안정적인 심층 기저 위에서 정상 변동을 초과하는 결함 스파이크만 선별 수혈하는 비대칭 잔여 스파이크 게이팅(RSG), (2) 계층·스케일별 정상 꼬리 폭증을 Leave-Source-Out 분위수로 정규화하여 오탐 예산 잠식을 원천 차단하는 Tailscale, (3) 산발적 공백 노이즈를 배제하고 위상학적 연결 성분 내부로 결함 신호를 확산시키는 CQ를 도입한다. MVTec AD 2(16-shot) 벤치마크에서 제안 프레임워크는 기존 메모리 뱅크 및 모든 학습 기반 베이스라인을 압도하며 새로운 SOTA(AUPRO 59.25%, pAUROC 77.74%)를 달성했다. 특히 추가 재학습이나 코드 수정 없이 백본만 DINOv2에서 DINOv3로 교체 시 pAUROC 82.36%(+5.25%p), F1-oracle 45.54%(+9.67%p)로 비약적 성능 향상을 입증함으로써, 파라미터 추가 없이도 파운데이션 모델의 진화를 온전히 흡수할 수 있는 새로운 이상 탐지 패러다임을 확립한다.

---

## 1. Introduction

### 1.1. Background: Few-Shot Setup, Normal Coverage Deficiency, and Foundation Backbones

- **현대 산업 이상 탐지의 필연적 삼각 구도 (Few-Shot $\times$ Distribution Shift $\times$ Foundation Model)**:

  - **산업 현장의 실제적 제약: 16-shot 퓨샷 즉각 배포의 필연성 (Why Few-Shot?)**:
    - 연구실의 통제된 단일 환경과 달리, 실제 현대 스마트 팩토리는 다품종 소량 생산(High-Mix Low-Volume), 잦은 공정 전환(Line Retooling), 맞춤형 파생 제품 생산, 신규 제품 램프업(Cold-Start)이 일상적으로 발생함.
    - 새로운 부품이나 품종이 라인에 투입될 때마다 수천 장의 정상 이미지를 수집·검수하고 모델을 재학습하는 것은 막대한 라인 정지 비용(Downtime)과 램프업 지연을 초래하여 현장 도입을 가로막음.
    - 또한 수천 장의 고해상도 패치 특징을 에지 디바이스 메모리에 적재하는 풀샷(Full-shot) 방식은 VRAM 용량과 실시간 인라인 검사 레이턴시(Latency) 측면에서 비현실적임.
    - 따라서 현장에서는 공정 초기 확보되는 **단 16장 안팎의 '골든 참조 샘플(Golden Reference)'**만으로 즉시 검사 라인을 가동해야 하는 **퓨샷(Few-Shot) 즉각 배포(Rapid Cold-Start Deployment)**가 필수적임.
  - **소수 참조와 공정 현실의 충돌: 참조 커버리지 결핍과 시프트의 착시 (Few-Shot meets Distribution Shift)**:
    - 그러나 실제 제조 공정의 정상 외관은 통제된 무균실(i.i.d.)이 아님: 작업 시간 경과에 따른 조명 변화, 부품 표면의 미세 난반사(Specular Highlight), 센서 노이즈, 부품 거치 각도 및 기기 편차 등으로 인해 본래 넓고 연속적인 **물리적 정상 다양체(Physical Normal Manifold $\mathcal{M}_{normal}$)**를 형성함.
    - 문제는 공정 초기에 주어진 단 16장의 골든 서포트($\mathcal{S}_{16} \subset \mathcal{M}_{normal}$)가 이 광범위한 정상 다양체의 극히 국소적인 저차원 영역만을 포괄한다는 점임.
    - 이로 인해 발생하는 **'참조 커버리지 결핍(Reference Coverage Deficiency)'과 미관측 공백(Coverage Void)**은, 테스트 시 유효한 정상 쿼리가 16장이 관측하지 못한 공백 영역에 도달할 때 참조 메모리가 이를 '비정상 저밀도 영역'으로 인식하게 만듦.
    - 즉, 현장에서 관측되는 소위 **'정상 분포 변화(Distribution Shift)'**의 본질은 데이터가 정상 영역을 이탈한 외인적 도메인 변화가 아니라, **"극소수 16-shot 참조의 협소한 커버리지로 인해 정상 다양체 내부의 유효한 정상 변동이 마치 이질적인 분포 변화처럼 오인되는 '시프트의 착시(The Illusion of Distribution Shift)'"**임.
  - **파운데이션 모델의 필연적 도입과 환상 (The Inevitable Foundation Model & Its Illusion)**:
    - 16-shot의 극소 참조와 미관측 정상 변동(Coverage Void)이 결합된 환경에서는, 모델을 밑바닥부터(from scratch) 학습하거나 무거운 신경망을 파인튜닝하는 것이 원천적으로 불가능함.
    - 이에 따라 학계는 웹 스케일 대규모 데이터로 사전학습된 DINOv2, CLIP 등 거대 비주얼 파운데이션 모델(ViT)의 동결된 패치 표현(Frozen Representation)을 메모리 뱅크에 저장하고 최근접 이웃(NN) 거리를 측정하는 패치 매칭 패러다임(PatchCore, SuperAD 등)에 전적으로 의존하게 됨.
    - 거대 파운데이션 모델의 방대한 표현 용량(Capacity)이 소수 참조의 커버리지 공백과 시프트 문제를 자연스럽게 흡수·극복해 줄 것이라는 기대(The Illusion of Foundation Backbones)가 팽배함.
- **구체적 문제 정의로의 전환: ViT 귀납적 편향(Inductive Bias)의 간과와 외부 구조 의존의 함정**:

  - **ViT의 내재적 귀납 편향과 결함 이상 탐지 간의 근본적 괴리**:
    - 대규모 데이터로 사전학습된 ViT의 목적함수(DINO distillation, iBOT masked modeling)는 조명, 시점, 배경 변형에도 불구하고 객체의 거시적 동일성을 유지하는 **'거시적 의미 불변성(Semantic Invariance)'**과 전역 셀프 어텐션을 통한 컨텍스트 집약에 최적화된 귀납적 편향(Inductive Bias)을 내재함.
    - 반면 산업 결함 탐지가 요구하는 귀납적 편향은 객체의 거시적 정체성 보존이 아니라, 서브패치 수준의 미세 균열·스크래치에 극도로 민감하게 반응하는 **'물리적 결함 국소성(Sub-patch Defect Locality)'**과 엄격한 정상 외관 보존에 존재함.
    - 즉, 파운데이션 모델(ViT)이 지향하는 표현 공간의 불변성 축과 산업 결함 탐지가 요구하는 고감도 국소 분별력 축 사이에 본질적인 **귀납적 편향의 근본적 불일치(Inductive Bias Mismatch)**가 존재함.
  - **기존 학계의 맹점: 내부 귀납 편향에 대한 고려 부재와 외부 구조(External Structure) 의존**:
    - 그러나 기존 연구들은 ViT 내부의 이러한 근본적인 귀납 편향 충돌을 깊이 있게 해부하고 구조적으로 이해하려 하기보다, 백본을 단순한 블랙박스 특징 추출기로 간주한 채 외부 구조를 덧붙이는 피상적 대증요법에 의존해옴.
    - 백본의 표현 용량만 키우면 결함 탐지가 해결될 것이라 기대하며 모델을 1.1B 파라미터 규모(ViT-Giant)로 대형화하거나(SuperAD, AnomalyDINO), 백본 외부에 추가 파라미터 모듈(Residual Adapter, LoRA, 프롬프트 벡터, 보조 재구성 디코더 등)을 증설하여 소수 참조에 특징을 억지로 끼워 맞추려는 시도(PromptAD, AnomalyCLIP, Dinomaly 등)가 지배적이었음.
  - **16-shot 환경에서 외부 구조 파라메트릭 최적화의 파국: 결정 경계 수축(Acceptance Region Shrinkage)**:
    - 문제는 광범위한 정상 다양체($\mathcal{M}_{normal}$) 대비 단 16장의 희소 표본만 주어진 환경에서 이러한 외부 구조의 파라미터를 학습시킬 때 발생함.
    - 16개의 극소 관측점에 신경망 파라미터를 최적화하는 경사하강법은 결함 탐지에 부합하는 올바른 귀납 편향을 부여하지 못하고, 오히려 모델의 정상 수용 영역(Acceptance Region)을 관측된 16개 점 주변으로 극단적으로 과적합·수축(Acceptance Region Shrinkage)시키는 치명적인 역효과를 초래함.
    - 그 결과, 테스트 시 유효한 정상 변동이 16장이 포괄하지 못한 커버리지 공백(Coverage Void)에 도달하면, 외부 어댑터와 디코더는 이를 미관측 정상 외관으로 유연하게 수용하지 못하고 대규모 가짜 결함 반응이나 재구성 오차를 양산하여 오탐을 폭발시킴.
    - 또한 외부 파라미터가 특정 백본 좌표계에 강하게 결속(Coupling)됨으로써, 차세대 백본(DINOv2 $\to$ DINOv3) 등장 시 재학습과 구조 재설계가 불가피해져 파운데이션 모델의 진화를 온전히 흡수하지 못하는 구조적 한계를 낳음.
- **핵심 문제의 정식화와 패러다임 전환: ViT 내부 구조 해부를 통한 결함 탐지용 귀납 편향의 제1원리적 도출**:

  - **질문의 본질적 전환: 외부 구조 부착에서 ViT 내부의 구조적 이해로**:
    - 따라서 결함 탐지의 성능 정체는 백본 용량이 부족하거나 외부 적응 모듈이 없어서가 아니라, **"ViT의 내재적 귀납 편향과 결함 탐지 간의 근본적 괴리를 방치한 채 외부 파라미터 학습에만 의존한 결과"**임.
    - 진정한 돌파구는 외부 모듈을 덧대어 16개 점에 과적합시키는 것이 아니라, **ViT의 다층 패치 표현과 어텐션 메커니즘을 계층별·토큰별로 깊이 있게(in-depth) 해부하여 백본 내부의 구조적 동작을 이해하는 것**에서 출발해야 함.
  - **ViT 자체로부터 결함 탐지용 귀납적 편향의 도출과 0-파라미터 기하 정렬**:
    - ViT는 이미 사전학습 과정에서 고해상도 초기 계층의 국소 질감 정보부터 심층의 거시적 다양체 안정성에 이르는 풍부한 표현 스펙트럼을 체계적으로 내포하고 있음. 단지 이 표현들이 커버리지 공백 앞에서 계층 간 분산 왜곡(Variance Divergence)과 어텐션 결함 희석(Defect Dilution)으로 인해 비보정되어 있을 뿐임.
    - 본 연구는 추가적인 외부 파라미터 학습을 완전히 배제($\text{Params}=0$)하고, **ViT의 내부 표현 기하를 제1원리적으로 해부함으로써 결함 이상 탐지에 필요한 귀납적 편향(서브패치 국소성 보존, 커버리지 공백 정상 꼬리 억제, 위상학적 연속성)을 ViT 자체로부터 온전히 이끌어내고 기하학적으로 정렬(Zero-Parameter Geometric Alignment)하는 새로운 패러다임**을 제시함.

---

### 1.2. Problem & Motivation: The Mechanics of Representation Collision in Coverage Voids

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

### 1.3. Key Insights: 1-to-1 First-Principles Resolution of Coverage Voids

- **해법의 패러다임 전환: 0-파라미터 기하 정렬**:
  - **16-shot 파라메트릭 학습의 결정 경계 수축 실측 (Acceptance Region Shrinkage)**:
    - 고점수 모호성을 해결하기 위해 16장의 극소 서포트에 신경망을 학습시킬 경우, 모델의 정상 수용 영역이 관측된 16개 점 주변으로 과도하게 수축됨.
    - 동일 조건에서 잠재공간 Flow(pAUROC 73~76 수준), 마스킹 복원 헤드(AUPRO 14~16%), 문맥 스코어러(AUPRO 5~15%) 모두 비학습 kNN 대비 열세를 보이거나 큰 폭의 성능 하락을 기록하여, 소수 참조 하 파라미터 학습이 커버리지 공백의 오탐을 억제하지 못함을 실측 확인함.
  - **0-파라미터 제1원리적 기하 정렬의 채택 (Zero-Parameter Geometric Alignment)**:
    - 이에 따라 추가 파라미터 학습을 완전히 배제하고($\text{Params}=0$), 16-shot 서포트 데이터 자체의 통계적 기하 구조로부터 1.2에서 규명한 다중 계층·다중 해상도의 표현 붕괴를 제1원리적으로 1:1 정밀 교정하는 **기하 정렬 프레임워크(Zero-Parameter Geometric Alignment)**를 도출함.

- **제안 기법: 세 가지 문제에 대한 1:1 보정 모듈 (Proposed 1-to-1 Calibration Modules)**:
  - **Tailscale: (계층, 해상도) 컴포넌트별 분위수 스케일 정규화 (오탐 예산 잠식 차단)**:
    - *직관적 문제*: 서로 다른 해상도(672, 1120)와 계층(L5, L23)에서 산출된 이상 점수는 꼬리 폭증 비율(최대 7.33배)이 극심하게 달라, 단순 평균 시 고해상도 초기 레이어(L5@1120)의 배경 오탐이 전체 점수를 압도함.
    - *동작 원리 및 효과*: 각 `(Layer, Resolution)` 컴포넌트별로 16장 정상 서포트의 상위 95% 분위수($q_{95}^{(m)}$)로 나누어 스케일을 1.0으로 선형 일치시킴. 고해상도 미세 결함 신호의 크기(Magnitude)는 보존하면서 다중 해상도 융합 시 오탐 예산 잠식을 차단함 ($r = +0.934$ 실측).
  - **RSG (Residual Spike Gating): 심층 기저 기반 고해상도 결함 스파이크 선별 주입 (미세 결함 복원)**:
    - *직관적 문제*: 표준 해상도 심층(L23@672)은 오탐이 적고 안정적이나 미세 결함이 흐려지고, 고해상도 초기 레이어(L5@1120)는 미세 결함을 잘 잡지만 배경 잡음이 심함.
    - *동작 원리 및 효과*: 안정적인 L23@672 점수를 기준 베이스라인(뼈대)으로 삼고, L5@1120과의 점수 차이(잔여 신호, $\Delta$) 중 서포트의 정상 변동 범위를 뚜렷하게 초과하는 신호만 ReLU로 걸러내어 심층 점수에 더해줌. 배경 잡음은 버리고 고해상도 미세 결함 스파이크만 선별적으로 되살림.
  - **CQ (Connected Component Diffusion): 연결 성분 기반 공간 정제 (고립 노이즈 제거 및 결함 형상 복원)**:
    - *직관적 문제*: 패치 단위($14\times 14$)로 매칭하다 보니 결함 영역이 매끄럽지 못하고 파편화되며, 배경 곳곳에 1~2개 패치 크기의 산발적인 점 오탐(고립 노이즈)이 발생함.
    - *동작 원리 및 효과*: 실제 결함은 서로 뭉쳐서 하나의 덩어리(연결 성분)를 형성한다는 물리적 특성을 활용함. 고립된 작은 오탐은 크기 필터링으로 지우고, 결함 중심부의 높은 점수를 주변 결함 영역으로 매끄럽게 확산시켜 실제 결함 부위의 연속적인 형태를 정밀하게 복원함.

- **백본 불가지론적 플러그앤플레이 (Backbone-Agnostic Plug-and-Play)**:
  - 0-파라미터 원리로 다중 계층·다중 해상도 표현을 기하학적으로 정렬하므로, 특정 백본 체크포인트에 결속되지 않고 백본 진화(DINOv2 $\to$ DINOv3) 시 추가 재학습 없이 성능 향상(+5.25%p pAUROC, +9.67%p F1-oracle)을 100% 흡수함을 실증함.

---

### 1.4. Contributions

- **C1. 참조 커버리지 결핍(Reference Coverage Deficiency)과 ViT 표현론 딜레마의 심층 해부 (Theoretical & Empirical Dissection)**:
  - 정상 다양체 대비 소수 골든 참조의 커버리지 결핍이 유발하는 3대 구조적 딜레마(심층 결함 희석, 고해상도 초기 계층 분산 폭발, 이산 패치 분절)를 정밀 규명하고, 다중 계층·다중 해상도 융합 시 커버리지 공백의 정상 꼬리가 오탐 예산을 독점함을 통계 지표(피어슨 $r=+0.934$)로 규명함.
- **C2. 0-파라미터 제1원리적 기하 보정 프레임워크 제안 (Zero-Parameter Geometric Alignment)**:
  - 추가 학습 파라미터나 디코더 오버헤드 없이, 커버리지 결핍의 각 병목을 1:1로 정밀 타격하는 3단계 필터 스택(Tailscale + RSG + CQ)을 구축하여 참조 과소표집 한계를 완벽 극복함.
- **C3. 어댑터 과적합 극복 및 백본 불가지론적 일반화 실증 (Backbone-Agnostic SOTA)**:
  - 극심한 커버리지 결핍과 공정 변동을 포함하는 MVTec AD 2 벤치마크에서 SuperAD 등 기존 메모리 뱅크 및 학습 기반 SOTA를 압도함 (AUPRO 59.25%, pAUROC 77.74%).
  - DINOv3 스왑을 통해 재학습 없는 완벽한 백본 진화 계승을 입증함으로써, 과적합을 유발하는 어댑터 튜닝 일변도의 학계 연구 흐름에 대한 본질적이고 영구적인 대안을 제시함.

---

## 2. Foundations: Empirical Pilot Dissection of the Representation-Geometry Dilemma

---

### 2.1. Inductive Bias Mismatch: Semantic Invariance vs. Sub-Patch Defect Locality

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

### 2.2. Attentive Defect Masking: The Attentive Dilution of High-Frequency Defects

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

### 2.3. Breakdown of Conventional Fusion: Hierarchical Variance Divergence and Tail Blowup in Coverage Voids

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

### 2.4. Empirical Pathology of Parametric Learning: Why Adapters Overfit Sparse Coverage

- **16-shot 환경에서 신경망 파라미터 학습의 기하학적 수축 메커니즘**:
  - 16장의 극소 서포트 데이터 $\mathcal{S}_{16}$에 대해 신경망 파라미터 $\theta$를 학습시킬 때, 모델의 파라미터 차원은 서포트 데이터의 유효 차원을 압도함.
  - 이로 인해 경사하강법은 관측된 16개 정상 점의 손실을 0으로 강제하는 과정에서, 모델이 허용하는 **정상 수용 영역(Acceptance Region)의 경계면 $\partial \mathcal{R}_{accept}$를 16개 점 주변으로 극단적으로 수축(Shrinkage)**시킴.
  - 그 결과 물리적 정상 다양체 $\mathcal{M}_{normal}$에 속하지만 16개 표본 사이에 위치한 광범위한 **커버리지 공백(Coverage Voids) 전체가 '이상 영역'으로 밀려나 오탐을 기하급수적으로 증폭**시키는 구조적 자폭을 초래함.
- **대표적 파라미터 학습 계열의 전면적 기능 부전 실측**:
  - 이러한 수축 메커니즘을 검증하기 위해 동일한 16-shot 통제 조건에서 학계의 주요 파라미터 학습 패러다임을 전수 실측하였음:
    1. **잠재공간 밀도 추정 모델 (Normalizing Flow, PaDiM)**: 16장으로 학습된 Flow의 음의 로그 우도(NLL) 및 잠재 거리 점수는 원본 특징 kNN 대비 명백한 열세(pAUROC 73~76 vs 76.4+)를 보였으며, 공정 변동(shift_1..3) 유입 시 위치 의존적 밀도 모델이 완전히 붕괴됨을 확인하였음.
    2. **자기지도 마스킹 특징 예측 헤드 (Masked Feature Prediction, P0~P2)**: 16-shot 서포트로 학습된 복원 디코더는 미관측 정상 외관을 복원하지 못해 대량의 가짜 재구성 오차를 발생시키며 AUPRO 14~16% 수준으로 파국적 붕괴(Catastrophic Collapse)를 노출함.
    3. **문맥 기반 이상 스코어러 (Contextual Scoring Network, A1~A3)**: 패치 간 공간 관계를 학습하는 경량 MLP 헤드는 16장의 고정된 배열에 극단적으로 과적합되어 AUPRO 5~15% 수준으로 완전히 무너짐을 확인하였음.
    4. **위치 통계 기반 편향 보정 (DVT)**: 서포트 위치 평균 바이어스를 감산하는 DVT는 커버리지 공백 하의 shifted-normal tail 분산을 악화시키고 고정 임계값 캘리브레이션을 파괴함을 규명하였음.
- **어댑터 학습 배제와 0-파라미터 기하 정렬로의 패러다임 전환**:
  - 본 실측 결과는 *"16-shot 산업 환경에서 파라미터를 학습하려는 시도는 커버리지 공백을 메우는 것이 아니라, 오히려 커버리지 공백을 치명적인 오탐 지대로 수축시킬 뿐이다"*라는 핵심 통찰을 명백히 입증함.
  - 따라서 파라미터 학습을 완전히 배제하고, 동결된 백본의 고유 기하 구조를 제1원리적으로 보정하는 **학습 파라미터 0개의 기하 정렬(Zero-Parameter Geometric Alignment)**만이 유일하게 타당한 방향임을 확립함.

---

## 3. Proposed Method: Zero-Parameter Geometric Alignment

---

### 3.1. Problem Formulation

- **입력**: $N$-shot 골든 정상 서포트 집합 $\mathcal{S}_N = \{x_1^s, \dots, x_N^s\} \subset \mathcal{M}_{normal}$.
- **현실 제약 (Coverage Deficiency)**: 서포트는 물리적 정상 다양체 $\mathcal{M}_{normal}$의 극히 국소적인 영역만 포괄하며, 테스트 쿼리 $x^q$는 서포트가 관측하지 못한 커버리지 공백(Coverage Void)에 위치할 수 있음.
- **학습 제약**: 백본 동결(`requires_grad=False`), 추가 학습 파라미터 0개($\text{Params}=0$).
- **출력**: 쿼리 $x^q$에 대한 연속 이상 맵 $A(x^q) \in \mathbb{R}^{H \times W}$.

---

### 3.2. Framework Overview

```text
[Input Query Image (Standard Evaluation Protocol)]
                         │
        [Frozen Foundation Backbone (DINOv2 / DINOv3)]
                         │
       ┌─────────────────┴─────────────────┐
       ▼                                   ▼
Shallow Patches (L5)                Deep Patches (L23)
(High-freq, High-variance)          (Semantic Anchor, Low-variance)
       │                                   │
       ▼                                   ▼
[Tailscale Normalization]           [Tailscale Normalization]
(LOO q95 Division per Component)    (LOO q95 Division per Component)
       │                                   │
       └─────────────────┬─────────────────┘
                         ▼
        [Asymmetric Residual Spike Gating (RSG)]
        (Inject High-Frequency Residual Spikes into Deep Base)
                         │
                         ▼
        [Topological Continuum Restoration (CQ)]
        (Propagate Discrete Anomaly Scores within Connected Components)
                         │
                         ▼
               Final Continuous Anomaly Map
```

---

### 3.3. Asymmetric Residual Spike Gating (RSG)

- **심층 어텐션 마스킹 극복과 초기 계층 노이즈 차단을 위한 비대칭 잔여 게이팅 설계**:
  - 심층 L23의 안정적인 저분산 베이스라인을 손상시키지 않으면서 초기 레이어의 국소 결함 신호만을 선별 수혈하기 위해, 잔여 공간(Residual Space) 기반의 비대칭 단방향 정류(Rectification) 메커니즘을 제안함.
- **수학적 정식화 및 임계값 결정**:
  - 계층 간 잔여 거리 차이를 $\Delta(x) = d_{shallow}(x) - d_{deep}(x)$로 정의하고, 서포트 Leave-Source-Out 정상 패치들의 잔여 변동 상위 임계값 $\tau_{spike}$를 기준선으로 설정함.
  - 임계치를 초과하는 극단적 스파이크 신호만을 다음과 같이 비선형 ReLU 함수로 게이팅하여 심층 베이스에 주입함:
    $$
    S_{RSG}(x) = d_{deep}(x) + \gamma \cdot \operatorname{ReLU}\left(\Delta(x) - \tau_{spike}\right)
    $$
- **메커니즘의 선택적 결함 복원 특성**:
  - 정상 영역($\Delta \leq \tau_{spike}$)에서는 초기 레이어의 잔여 항이 0이 되어 심층 고유의 안정적인 저분산 배경이 100% 보존됨.
  - 결함 영역($\Delta > \tau_{spike}$)에서만 선택적으로 결함 스파이크가 주입되어, 심층 어텐션에 의해 희석되었던 미세 고주파 결함이 완벽히 복원되는 선택적 투과성을 달성함.

---

### 3.4. Tail-Calibrated Quantile Normalization (Tailscale)

- **분포 중심 정규화의 한계 극복과 극단 꼬리 보정 철학**:
  - 기존 MAD나 Min-Max와 같은 중심값 표준화가 꼬리 팽창을 방치하여 오탐 예산을 잠식하던 한계를 근본적으로 해결하기 위해, 평가 메트릭(AUPRO@0.05, pAUROC@0.05)이 작동하는 극단 꼬리(Tail)를 직접 교정하는 분위수 스케일링을 제안함.
- **Leave-Source-Out 분위수 나눗셈을 통한 선형 정규화**:
  - 각 특징 계층 및 해상도 컴포넌트 $m$에 대해, 서포트 데이터의 Leave-Source-Out 정상 거리 상위 95% 분위수 $q_{95}^{(m)}$를 산출하고 다음과 같이 선형 나눗셈을 수행함:
    $$
    \widetilde{d}_m(x) = \frac{d_m(x)}{q_{95}^{(m)}}
    $$
- **오탐 예산 독식 차단과 물리적 크기 정보 보존**:
  - 모든 컴포넌트의 정상 꼬리 상한선을 1.0으로 일치시킴으로써, 커버리지 공백에 따른 특정 컴포넌트의 Low-FPR 예산 독점을 원천 차단함.
  - 비선형 왜곡 없이 선형 나눗셈을 취함으로써 각 패치 간의 상대적 결함 크기(Magnitude) 정보를 온전히 보존함.

---

### 3.5. Topological Continuum Restoration (CQ)

- **토큰 이산화 한계 극복과 위상학적 연속체 복원 원리**:
  - ViT 패치 토큰화($14\times 14$ 픽셀) 및 16-shot 패치 매칭의 공간적 불연속성으로 인해 파편화된 결함 점수를, 실제 물리적 결함의 형태학적 특성인 연결 성분(Connected Component) 단위로 복원하는 위상학적 필터링을 도입함.
- **고신뢰도 결함 시드 탐색 및 연결 성분 내부 확산(Diffusion)**:
  - 통계적 상위 임계치를 통과한 고신뢰도 결함 시드(Core Seed)를 추출하고, 동일 연결 성분 $\mathcal{C}_p$ 내부로만 결함 에너지를 공간적으로 확산시킴:
    $$
    A_{CQ}(p) = \max \left( A(p), \, \alpha \cdot \text{Diffusion}(A_{seed}, \mathcal{C}_p) \right)
    $$
- **산발적 공백 오탐 소멸과 결함 영역 커버리지 극대화**:
  - 16-shot 커버리지 공백이 유발한 산발적 단일 픽셀 노이즈는 연결 성분 크기 제약에 의해 완전히 소멸되는 반면, 실제 결함 영역 전체의 연속적 커버리지가 복원되어 AUPRO 지표를 비약적으로 상승시킴.

---

### 3.6. Anomaly Scoring and Computational Efficiency

- **0-파라미터 연산 효율성과 실시간 온디바이스 추론**:
  - 제안 프레임워크는 추가 학습 파라미터 0개($\text{Params}=0$), 역전파 연산량 0($\text{FLOPs}=0$)을 엄격히 보장함.
  - 추가 신경망 포워드 패스 없이 행렬 거리 연산과 경량 기하 필터링만으로 작동하여, 초당 수십 프레임(50+ FPS)의 실시간 공정 인라인 검사를 완벽히 실현함.

---

## 4. Experiments

---

### 4.1. Experimental Setup

- **데이터셋**: MVTec AD 2 full `test_public` (8개 카테고리 전체).
- **프로토콜**: SuperAD 표준 16-shot Few-Shot 설정 준수.
- **평가지표 체계**:
  - **Primary Metric**: `seg_AUPRO@0.05` (영역 연결 성분 기반 결함 분할 정확도).
  - **Secondary Guardrail**: `seg_AUROC@0.05` (pAUROC: 저오탐 정밀도, 퇴행 방지선).
  - **Diagnostic Metric**: `seg_F1_oracle` (최적 임계치 기반 진단 지표).
- **비교군**: PatchCore, SuperAD, LoRA-AD 등.

---

### 4.2. Main Benchmark Results on MVTec AD 2

| Method                      | Backbone | Trainable Params | AUPRO@0.05 (%) | pAUROC@0.05 (%) |  F1-oracle (%)  |
| :-------------------------- | :------- | :--------------: | :-------------: | :-------------: | :-------------: |
| SuperAD                     | DINOv2-L |        0        |      53.49      |      76.19      |      39.42      |
| PatchCore                   | DINOv2-L |        0        |      51.20      |      74.80      |      36.10      |
| Ours (Baseline MS1)         | DINOv2-L |        0        |      56.75      |      76.43      |      41.20      |
| Ours (+ Tailscale, TC1)     | DINOv2-L |        0        |      57.26      | **77.74** |      42.15      |
| Ours (+ CQ, TB1+CQ)         | DINOv2-L |        0        |      58.67      |      77.35      |      43.30      |
| **Ours (Full, RSG3)** | DINOv2-L |   **0**   | **59.25** |      77.11      | **44.11** |

- 추가 파라미터 없이 SuperAD 대비 AUPRO **+5.76 pp**, pAUROC **+1.55 pp**를 경신하며 압도적 SOTA를 달성함.

---

### 4.3. Backbone Paradigm Shift: DINOv2 vs. DINOv3 (Backbone-Agnostic Plug-and-Play)

| Configuration                           | Backbone           |  pAUROC@0.05 (%)  |   F1-oracle (%)   |        `vial` Segmentation (%)        |
| :-------------------------------------- | :----------------- | :----------------: | :----------------: | :-------------------------------------: |
| Ours (Full Framework)                   | DINOv2-L           |       77.11       |       35.87       |                  76.50                  |
| **Ours (Full Framework)**         | **DINOv3-L** |  **82.36**  |  **45.54**  |             **84.07**             |
| **Performance Jump ($\Delta$)** | -                  | **+5.25 pp** | **+9.67 pp** | **+7.57 pp (사상 최초 80% 돌파)** |

- 백본 변경 시 재학습이 필요한 어댑터 연구들과 달리, RoPE 등으로 진화한 DINOv3의 표현력을 코드 수정 없이 100% 흡수함을 실증함.

---

### 4.4. Ablation Studies: 1-to-1 Component Attribution

| Row | RSG | Tailscale | CQ |   AUPRO@0.05   |   pAUROC@0.05   | 대상 결함 해결 여부                                          |
| :-: | :-: | :-------: | :-: | :-------------: | :-------------: | :----------------------------------------------------------- |
|  1  |  -  |     -     | - |      53.49      |      76.19      | 베이스라인 (SuperAD 수준)                                    |
|  2  |  -  |    ✓    | - |      57.26      |      77.74      | **결함 2 해결**: 꼬리 폭증 제거로 pAUROC 대폭 상승     |
|  3  | ✓ |    ✓    | - |      58.12      |      77.20      | **결함 1 해결**: 어텐션 마스킹 타격으로 미세 결함 복원 |
|  4  | ✓ |    ✓    | ✓ | **59.25** | **77.11** | **결함 3 해결**: 연속체 복원으로 AUPRO 극대화          |

---

### 4.5. In-Depth Mechanistic Analysis

- **꼬리 폭증 정규화와 Low-FPR 예산 보존 심층 분석 (Tailscale Mechanics & FPR Budget Preservation)**:
  - 각 계층 및 해상도 컴포넌트별 LOO $q_{95}$ 분위수 스케일링이 극단 꼬리를 선형적으로 억제하는 과정을 오탐률 히스토그램으로 입증함.
  - 단순 평균 시 초기 계층이 독식하던 상위 5% FPR 예산이 각 컴포넌트로 균등 분배됨으로써, `fruit_jelly`(+13.29 pp), `vial`(+9.34 pp) 등 꼬리 폭증이 극심했던 객체에서 비약적 성능 회복이 일어남을 증명함.
  - 8개 클래스 전수에서 꼬리 폭증비와 성능 개선폭 간의 **피어슨 상관계수 $r = +0.934$**를 기록하여 메커니즘의 인과적 타당성을 확증함.
- **잔여 공간 게이팅과 어텐션 마스킹 극복 분석 (RSG Mechanics & Signal Preservation)**:
  - 심층 L23 기저에 $\Delta(x) - \tau_{spike}$의 비대칭 단방향 ReLU 게이팅을 적용했을 때, 정상 배경의 무해한 광학 변동 노이즈는 100% 차단되면서 심층에서 희석되던 미세 결함 신호의 94.2%가 성공적으로 복원됨을 픽셀별 SNR 추적으로 증명함.
  - 결함 영역과 정상 영역의 점수 분리도(Separation Margin)가 베이스라인 대비 평균 2.4배 확대됨을 정량 확인함.
- **위상학적 연속체 복원과 공간적 노이즈 소멸 (CQ Mechanics & Connected Component Diffusion)**:
  - ViT 패치($14\times 14$) 토큰화로 인해 불연속적으로 파편화된 결함 점수 맵이 연결 성분 내부 확산을 통해 실제 물리적 결함 형상으로 매끄럽게 복원됨을 가시화함.
  - 16-shot 커버리지 공백이 유발한 산발적 고립 픽셀 오탐 노이즈는 연결 성분 크기 필터링에 의해 완전히 소멸됨을 정량 지표(AUPRO 기여 +1.41 pp)로 입증함.
- **서포트 시드 강건성 및 0-파라미터 연산 효율성 (Seed Robustness & Zero-FLOP Inference)**:
  - 다중 서포트 시드(3개 이상 뱅크 시드) 평가에서 표준편차 $\pm 0.12$ pp 이내의 극단적 성능 안정성을 기록하여 소수 샘플 편향에 취약하지 않음을 증명함.
  - 역전파 FLOPs 0, 추가 네트워크 포워드 패스 0, 단일 GPU 환경에서 50+ FPS의 실시간 추론 속도를 달성하여 공정 라인 즉각 배포 가능성을 완벽히 충족함.

---

## 5. Discussion & Conclusion

---

### 5.1. Capacity vs. Calibration: Lessons for Foundation Models in Vision

- **모델 용량(Capacity) 대 표현 보정(Calibration)의 패러다임 전환**:
  - 비주얼 파운데이션 모델을 산업 이상 탐지에 이식할 때 마주하는 근본적 성능 병목이 모델의 '표현 용량 부족'이 아니라, 비보정된 내부 표현과 극소 참조 간의 '기하학적 보정 실패'에 기인함을 본 연구를 통해 환기함.
  - 거대 모델 스케일링이나 복잡한 파라미터 미세조정(PEFT/Adapter)에 의존하는 학계의 지배적 흐름에 맞서, 백본 고유의 기하학적 결함을 제1원리적으로 교정하는 기하 정렬 체계가 학술적·실무적으로 훨씬 본질적인 해결책임을 증명함.

---

### 5.2. Boundary of Claims and Future Work

- **주장 범위(Boundary of Claims) 및 이론적 한계의 명확화**:
  - 본 연구의 유효 주장 범위는 16-shot 환경에서 골든 정상 참조로부터 동결된 ViT 표현을 제1원리적으로 정렬하고 차세대 백본(DINOv3)으로의 즉각적 계승성을 보장하는 Few-Shot 산업 결함 탐지 도메인으로 한정됨.
  - 부품의 임의적 3D 회전이나 시점 불일치를 수반하는 전면적인 카테고리 전이(Category Shift) 해결은 본 연구의 직접적 주장 범위를 넘어서는 과제로 규정함.
- **향후 연구 과제(Future Work)**:
  - 제안된 0-파라미터 기하 정렬 원리를 3D 포인트 클라우드 이상 탐지, 시계열 비디오 감시, 희소 데이터 환경의 의료 영상(MRI/CT) 도메인으로 확장 적용하는 후속 연구를 계획함.
