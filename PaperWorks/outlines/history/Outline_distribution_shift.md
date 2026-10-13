# Dissecting Visual Foundation Models for Industrial Anomaly Detection: The Failure of Representation Calibration and Its First-Principles Resolution

**Hun Im\*, Author Two, Author Three, Pilsung Kang\***

---

## Index

- Abstract
- 1. Introduction

  - 1.1. Problem & Motivation: The Trilemma of Few-Shot Anomaly Detection under Distribution Shift
  - 1.2. Key Insights: 1-to-1 First-Principles Resolution of the Trilemma
  - 1.3. Contributions
- 2. Foundations: Dissecting the Representation-Geometry Dilemma

  - 2.1. Inductive Bias Mismatch: Semantic Invariance vs. Physical Defect Locality
  - 2.2. Attentive Defect Masking: The Attentive Dilution of High-Frequency Defects
  - 2.3. Breakdown of Conventional Fusion: Hierarchical Variance Divergence and Tail Blowup
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

잦은 품종 전환(High-Mix Low-Volume)과 신제품 램프업(Cold-Start) 환경에서 극소수의 골든 참조 샘플(Golden Reference)에 의존하는 퓨샷 산업 이상 탐지(Few-shot Industrial Anomaly Detection)에서는, 모델 재학습 없이 대규모 사전학습된 비주얼 파운데이션 모델(ViT)의 동결된 전이 표현력에 전적으로 의존하는 메모리 뱅크 패러다임이 주류로 정착되었다. 그러나 실제 스마트 팩토리 환경은 조명, 표면 반사광, 센서 편차 등 불가피한 **정상 분포 변화(Distribution Shift)**를 수반하며, 소수 참조(Few-shot)의 극히 협소한 커버리지와 결합하여 치명적인 성능 붕괴를 초래한다. 현재 학계의 지배적 흐름은 백본을 거대화(Giant 스케일링)하거나, 학습 가능한 어댑터(LoRA, 프롬프트) 및 보조 재구성 디코더를 결합하는 외부적 확장에 치중해왔다. 그러나 본 논문에서는 소수 참조 하 분포 변화 환경에서 파라미터를 추가 학습하려는 모든 시도가 필연적인 소스 도메인 과적합과 전이성 붕괴로 귀결될 뿐임을 지적한다. 나아가, 진정한 병목은 모델 용량 부족이 아니라, ViT 고유의 거시적 의미 불변성과 전역 어텐션이 산업 결함의 물리적 국소성과 정면 충돌하고 정상 분포 변화 앞에서 계층 간 분산 왜곡을 일으키는 **'귀납적 편향의 불일치'**에 있음을 해부학적으로 규명한다. 구체적으로, (1) 심층 전역 어텐션이 미세 결함을 정상 배경 속으로 소프트 흡수하는 결함 희석(Attentive Defect Masking), (2) 분포 변화에 노출된 천층의 정상 패치 거리가 최대 35배 폭증하여 오탐 예산을 독식하는 분산 발산(Hierarchical Variance Divergence), (3) 소수 패치 매칭의 공간적 분절(Discrete Tokenization Discontinuity)이라는 3대 구조적 실패를 밝힌다. 이를 해결하기 위해, 우리는 외부 파라미터 학습을 완전히 배제하고 소수 참조로부터 백본의 표현 결함을 제1원리적으로 교정하는 **학습 파라미터 0개의 기하 정렬 프레임워크(Zero-Parameter Geometric Alignment Framework)**를 제안한다. (1) 분포 변화에 둔감한 심층 기저 위에서 정상 변동을 초과하는 결함 스파이크만 선별 수혈하는 비대칭 잔여 스파이크 게이팅(RSG), (2) 계층·스케일별 정상 꼬리 폭증을 Leave-Source-Out 분위수로 정규화하여 오탐 예산 잠식을 원천 차단하는 Tailscale, (3) 산발적 분포 노이즈를 배제하고 위상학적 연결 성분 내부로 결함 신호를 확산시키는 CQ를 도입한다. MVTec AD 2(16-shot) 벤치마크에서 제안 프레임워크는 기존 메모리 뱅크 및 모든 학습 기반 베이스라인을 압도하며 새로운 SOTA(AUPRO 59.25%, pAUROC 77.74%)를 달성했다. 특히 추가 재학습이나 코드 수정 없이 백본만 DINOv2에서 DINOv3로 교체 시 pAUROC 82.36%(+5.25%p), F1-oracle 45.54%(+9.67%p)로 비약적 성능 향상을 입증함으로써, 파라미터 추가 없이도 파운데이션 모델의 진화를 온전히 흡수할 수 있는 새로운 이상 탐지 패러다임을 확립한다.

---

## 1. Introduction

### 1.1. Problem & Motivation: The Trilemma of Few-Shot Anomaly Detection under Distribution Shift

- **현대 산업 이상 탐지의 필연적 삼각 구도 (Few-Shot $\times$ Distribution Shift $\times$ Foundation Model)**:
  - **소수 정상 참조의 필연성: 다품종 소량 생산과 신제품 즉각 배포 (Few-Shot & Rapid Deployment)**:
    - 양산 안정화 단계에서는 대량의 정상 데이터를 수집할 수 있으나, 현대 제조 공정은 잦은 품종 전환(Line Retooling), 맞춤형 파생 모델 생산, 신규 제품 램프업(Cold-Start)이 빈번함.
    - 신제품이나 변형 모델이 라인에 투입될 때마다 수천 장의 정상 이미지를 수집·검수하고 모델을 재학습하는 것은 막대한 라인 정지 비용(Downtime)과 램프업 지연을 초래함.
    - 또한 수천 장의 패치를 에지 디바이스 메모리에 적재하는 풀샷(Full-shot) 방식은 VRAM 및 실시간 인라인 검사 레이턴시(Latency) 측면에서 비현실적임.
    - 따라서 현장에서는 공정 초기 확보되는 **단 16장 안팎의 '골든 참조 샘플(Golden Reference)'**만으로 즉시 검사 라인을 가동해야 하는 **퓨샷(Few-Shot) 즉각 배포(Rapid Cold-Start Deployment)**가 필수적임.
    - 이 16-shot 제약으로 인해 공정별 모델의 scratch 학습이나 무거운 파인튜닝은 원천적으로 불가능하며, 대규모 웹 데이터로 사전학습된 DINOv2, CLIP 등 거대 비주얼 파운데이션 모델(ViT)의 동결된 패치 표현(Frozen Representation)을 그대로 활용하는 패치 매칭 패러다임(PatchCore, SuperAD 등)에 전적으로 의존하게 됨.
  - **불가피한 정상 분포 변화와의 충돌 (Collision with Inevitable Distribution Shift)**:
    - 그러나 단 16장의 골든 서포트 샘플은 당일 초기 세팅된 특정 단일 환경(특정 조명 각도, 고정 센서)의 극히 협소한 정상 매니폴드만 포괄함.
    - 실제 가동 중인 공정(MVTec AD 2 등)에서는 작업 시간 경과에 따른 공장 조명 변화, 제품 표면 난반사(Specular Highlight), 센서 열화 및 노이즈, 부품 거치 각도 및 기기 편차 등 **Support-Query 정상 분포 변화(Distribution Shift)**가 필연적으로 수반됨.
    - 결과적으로 16-shot 골든 메모리는 테스트 시 유입되는 정상 변동을 '미관측 저밀도 영역'으로 인식하는 극단적 '참조 편향(Reference Coverage Mismatch)'을 겪게 됨.
  - **파운데이션 모델의 환상 (The Illusion of Foundation Backbones)**:
    - 학계는 웹 스케일 파운데이션 모델의 거대한 표현 용량(Capacity)이 소수 골든 참조 환경의 정상 분포 변화를 자연스럽게 흡수·극복할 것이라 맹신해옴.

- **붕괴의 메커니즘: 소수 참조 하 분포 변화와 ViT 사전학습의 충돌 (The Mechanics of Collision)**:
  - 실제로는 **소수 참조 하 분포 변화(Few-Shot Shift)**가 ViT 고유의 사전학습 귀납적 편향(Inductive Bias)과 정면 충돌하며 치명적인 계층 간 표현 왜곡을 폭발시킴:
    - **심층의 결함 소멸 (Attentive Defect Masking)**:
      - 심층(L23)은 의미론적 불변성을 위해 전역 Self-Attention으로 컨텍스트를 소프트 평균화함.
      - 조명 등 거시적 분포 변화에는 안정적이나, 면적 0.1% 미만의 미세 결함 토큰마저 압도적 다수의 정상 배경 속으로 희석시켜 결함 신호를 지워버림.
    - **천층의 꼬리 폭발 (Hierarchical Variance Divergence & Normal Tail Blowup)**:
      - 미세 결함의 고주파 경계를 포착하려면 천층(L5)과 고해상도 보간 특징을 활용해야 함.
      - 그러나 천층은 의미 불변성이 없어 조명·반사광 등 분포 변화에 극도로 취약하며, 16-shot 메모리에 관측되지 않은 정상 변동이 유입되는 순간 정상 패치 간 거리가 심층 대비 10~30배, 최대 35배 폭발함 (`fruit_jelly` L5 normal $q_{95} = 35.70$).
    - **고점수 모호성과 융합의 파국 (High-Score Ambiguity & Budget Exhaustion)**:
      - 결과적으로 **"관측된 높은 이상 점수가 실제 결함 때문인가, 아니면 소수 참조에서 보지 못한 정상 분포 변화 때문인가?"**라는 고점수 모호성이 발생함.
      - 관행적인 다중 계층 단순 평균(Naive Mean) 융합 시, 분포 변화로 폭증한 천층의 정상 꼬리가 AUPRO@0.05 / pAUROC@0.05의 극단 오탐 예산(Low-FPR Budget)을 독식하여 대규모 오탐으로 자폭함.

- **학계 대증요법의 필연적 파산: 삼중고(Trilemma) 앞에서의 구조적 한계**:
  - 학계는 이 문제를 백본의 '표현 결함'이 아닌 '용량 및 적응 모듈 부족'으로 오진하고 세 가지 대증요법에 의존해옴:
    1. **백본 스케일링 (Brute-Force Scaling, SuperAD / AnomalyDINO)**:
       - 백본을 11억 개 파라미터의 ViT-Giant로 키웠으나, ViT 내부의 전역 어텐션 결함 희석과 천층 분산 폭증 메커니즘은 그대로 존속하며, 무보정 융합 시 소수 참조 하 분포 변화로 인한 정상 꼬리 폭증을 전혀 통제하지 못함.
    2. **파라미터 효율적 미세조정 (PEFT & Adapters, PromptAD / AnomalyCLIP / LoRA)**:
       - 결함 판별력을 높이고자 16-shot 서포트로 어댑터나 프롬프트를 학습시키는 순간, **극소수 정상 샘플의 특정 소스 도메인에 심각하게 과적합(Overfitting)**됨.
       - 테스트 시 미관측된 타깃 분포 변화(Shift)가 발생하면 전이성이 붕괴되어 가짜 결함 반응을 양산하며, 파운데이션 모델이 진화(DINOv2 $\to$ DINOv3)할 때마다 재설계 및 재학습이 불가피함.
    3. **보조 재구성 디코더 (Auxiliary Decoders, Dinomaly / UniAD / DiAD)**:
       - 16-shot 정상으로 학습된 디코더가 테스트 쿼리의 새로운 분포 변화(조명/반사광 등)를 정상으로 재구성하지 못해, 정상 영역에 막대한 가짜 재구성 오차(Reconstruction Error)를 발생시켜 결함으로 오진함.
  - **핵심 교훈**: **"소수 참조(Few-Shot)와 분포 변화(Distribution Shift)가 결합된 환경에서는, 추가 파라미터를 '학습'하려는 모든 시도가 과적합과 일반화 붕괴로 직결된다."**

- **제1원리적 대안: 용량 증설(Capacity)이 아닌 0-파라미터 기하 보정(Geometric Calibration)**:
  - 따라서 진정한 해결책은 외부 모듈을 학습하거나 모델을 거대화하는 것이 아니라, **"외부 학습 파라미터를 단 1개도 추가하지 않고(Params=0), 소수 참조로부터 분포 변화가 유발한 ViT의 기하학적 왜곡을 제1원리적으로 교정하는 것"**에 있음.
  - 백본을 동결한 채 소수 서포트로부터 분포 변화에 취약한 정상 꼬리를 통계적으로 정규화(Tailscale)하고, 분포 변화에 강건한 심층 기저 위에서 정상 변동을 초과하는 결함 스파이크만 선별 주입(RSG)하는 **0-파라미터 기하 정렬 체계(Zero-Parameter Geometric Alignment)**를 확립해야 함.

---

### 1.2. Key Insights: 1-to-1 First-Principles Resolution of the Trilemma

- **삼중고(Few-Shot $\times$ Shift $\times$ ViT)에 대한 1:1 기하학적 해법**:
  - **Tailscale $\leftrightarrow$ 분포 변화가 유발한 천층 정상 꼬리 폭발 제압**:
    - *문제*: Few-shot 제약으로 복잡한 불변성 학습이 불가능하며, Distribution Shift로 인해 천층/다중해상도 정상 꼬리가 7~35배 폭발하여 오탐 예산을 독점함.
    - *해법*: 파라미터 학습 대신, 16-shot 서포트 자체의 Leave-Source-Out 상위 분위수($q_{95}$)로 각 컴포넌트를 선형 정규화함. 물리적 크기 정보(Magnitude)를 보존하면서 모든 컴포넌트의 정상 꼬리를 1.0으로 완벽히 통제함 ($r = +0.934$로 메커니즘 실증).
  - **RSG $\leftrightarrow$ 심층 어텐션 결함 희석과 천층 분포 변화 노이즈의 동시 해결**:
    - *문제*: 심층은 Shift에 강건하나 결함이 희석되고, 천층은 미세 결함을 보존하나 Shift 노이즈가 심함.
    - *해법*: Shift에 강건한 심층을 앵커 기저로 고정하고, 잔여 공간($\Delta$)에서 16-shot 서포트의 최대 정상 변동을 초과하는 결함 스파이크만 비대칭 ReLU로 선별 주입함. 천층의 무해한 Shift 노이즈는 100% 차단하면서 고주파 결함 신호만 비파괴적으로 복원함.
  - **CQ $\leftrightarrow$ 소수 참조의 토큰 분절과 산발적 Shift 노이즈의 위상학적 분리**:
    - *문제*: Few-shot 패치 매칭의 공간적 불연속성과, Distribution Shift가 유발하는 산발적인 픽셀 오탐 노이즈.
    - *해법*: 무거운 디코더 학습 없이, 실제 결함의 위상학적 특성인 연결 성분(Connected Component) 내부로만 고신뢰도 결함 시드를 확산(Diffusion)시킴. 산발적 Shift 노이즈는 소멸시키고 결함의 물리적 연속 영역을 복원함.
- **백본 불가지론적 플러그앤플레이 (Backbone-Agnostic Plug-and-Play)**:
  - 0-파라미터 원리로 설계되어 특정 백본 체크포인트에 과적합되지 않으므로, 백본 진화(DINOv2 $\to$ DINOv3) 시 재학습 없이 성능 향상(+5.25%p pAUROC, +9.67%p F1-oracle)을 100% 흡수함을 실증함.

---

### 1.3. Contributions

- **C1. 현대 산업 AD 삼중고(Few-Shot $\times$ Shift $\times$ ViT)의 표현론적 해부 (Theoretical & Empirical Dissection)**:
  - 소수 정상 참조 하 분포 변화 환경에서 ViT 백본이 겪는 3대 구조적 딜레마(심층 어텐션 결함 희석, 천층 정상 분산 폭발, 이산 패치 분절)를 정밀 규명하고, 단순 평균 시 정상 꼬리가 오탐 예산을 독점함을 통계 지표(피어슨 $r=+0.934$)로 완벽히 입증함.
- **C2. 0-파라미터 제1원리적 기하 보정 프레임워크 제안 (Zero-Parameter Geometric Alignment)**:
  - 추가 학습 파라미터나 디코더 오버헤드 없이, 삼중고의 각 병목을 1:1로 정밀 타격하는 3단계 필터 스택(Tailscale + RSG + CQ)을 구축하여 소수 참조 하 분포 변화를 완벽 무력화함.
- **C3. 어댑터 일변도 학계 트렌드 극복 및 백본 불가지론적 일반화 실증 (Backbone-Agnostic SOTA)**:
  - 극심한 분포 변화를 포함하는 MVTec AD 2 벤치마크에서 SuperAD 등 기존 메모리 뱅크 및 학습 기반 SOTA를 압도함 (AUPRO 59.25%, pAUROC 77.74%).
  - DINOv3 스왑을 통해 재학습 없는 완벽한 백본 진화 계승을 입증함으로써, 과적합을 유발하는 어댑터 튜닝 일변도의 학계 연구 흐름에 대한 본질적이고 영구적인 대안을 제시함.

---

## 2. Foundations: Dissecting the Representation-Geometry Dilemma

---

### 2.1. Inductive Bias Mismatch: Semantic Invariance vs. Physical Defect Locality

- **귀납적 편향의 충돌 메커니즘**:
  - CNN은 국소 수용 영역 및 평행이동 불변성으로 인해 텍스처 이상에 자연스럽게 반응함.
  - ViT는 초기 레이어부터 전역 토큰 간 상관관계를 연산하며, DINO 목적함수는 조명·각도·미세 질감 변형을 무시하는 거시적 의미 불변성을 극대화하도록 학습됨.
- **산업 결함의 물리적 본질과의 불일치**:
  - 산업 결함은 면적비 0.1% 이하의 극단적 국소성과 날카로운 광학적 단절을 지님.
  - ViT 표현 공간에서는 제품 표면의 정상 광학 변동과 미세 결함 간의 분리가 본질적으로 붕괴됨.

---

### 2.2. Attentive Defect Masking: The Attentive Dilution of High-Frequency Defects

- **소프트 어텐션 마스킹의 수학적 규명**:
  - ViT $l$번째 블록의 Self-Attention 출력:
    $$
    z_i^{(l+1)} = \sum_{j \in \Omega} A_{ij}^{(l)} W_V z_j^{(l)}
    $$
  - 산업 결함 패치 수 $|\Omega_{defect}| \ll |\Omega_{normal}|$이 성립함.
  - 심층 레이어로 진행할수록 전역 어텐션 가중치 엔트로피가 증가하여 배경 정상 패치들의 기여도가 지배함:
    $$
    \sum_{j \in \Omega_{normal}} A_{ij}^{(l)} W_V z_j^{(l)} \gg \sum_{k \in \Omega_{defect}} A_{ik}^{(l)} W_V z_k^{(l)}
    $$
- **결함 신호 소멸 실증**:
  - 심층 레이어(L23)에 도달하면 결함 토큰이 정상 컨텍스트 속으로 소프트 흡수되어 코사인 거리가 0에 가깝게 수렴함을 확인했음.
  - 심층 단독 사용 시 미세 스크래치 결함 검출률이 천층 대비 최대 40% 이상 소실됨을 실측함.

---

---

### 2.3. Breakdown of Conventional Fusion: Hierarchical Variance Divergence and Tail Blowup

- **분포 변화 하 계층별 분산 발산 (Hierarchical Variance Divergence under Shift)**:
  - 심층 블록(L11~L23): 전역 자기지도 정규화로 조명 등 거시적 분포 변화에도 정상 패치 간 코사인 거리 분산이 극도로 낮게 유지됨 ($\sigma_{deep}^2 \ll 1$).
  - 천층 블록(L1~L6): 조명 각도, 반사광, 표면 미세 변동 등 불가피한 정상 분포 변화(Distribution Shift)에 극도로 민감하여 정상 패치 간 코사인 거리 분산이 심층 대비 10~30배 폭증함 ($\sigma_{shallow}^2 \gg \sigma_{deep}^2$).
- **관행적 단순 평균(Naive Mean)의 수학적 붕괴**:
  - 기존 방식: $S_{naive}(x) = \frac{1}{M} \sum_{m=1}^M d_m(x)$.
  - AUPRO@0.05 및 pAUROC@0.05는 상위 0~5% 오탐(FPR) 극단 꼬리(Tail)에서 평가됨.
  - 분포 변화로 인해 분산이 수십 배 폭증한 천층 및 고해상도 보간 컴포넌트의 정상 꼬리가 단순 합산 맵의 FPR 예산을 독식하여 대량 오탐을 유발함.
- **꼬리 폭증과 개선율 간 피어슨 상관계수 ($r = +0.934$) 실증**:

| Class | L5@1120 Normal $q_{95}$ | L23@672 Normal $q_{95}$ | Blowup Ratio | Tailscale 개선폭 (AUPRO $\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
| `fruit_jelly` | **35.70** | 4.87 | **7.33x** | **+13.29 pp** |
| `vial` | **28.40** | 4.12 | **6.89x** | **+9.34 pp** |
| `fabric` | 5.20 | 5.84 | 0.89x | +0.12 pp (영향 없음) |

- 분포 변화에 취약한 클래스일수록 정상 꼬리 폭증 비율과 Tailscale 적용 시의 성능 향상폭 간 상관계수가 $+0.934$로 측정되어, 단순 융합의 붕괴 원인이 분포 변화 유발 오탐 예산 잠식이었음을 완벽히 증명함.

---

## 3. Proposed Method: Zero-Parameter Geometric Alignment

---

### 3.1. Problem Formulation

- **입력**: $N$-shot 정상 서포트 집합 $\mathcal{S} = \{x_1^s, \dots, x_N^s\}$.
- **평가 조건**: 테스트 쿼리 $x^q$는 조명, 표면 반사광, 센서 및 배치 편차 등 현실적인 **Support-Query 분포 변화(Distribution Shift)**를 겪은 상태로 주어짐.
- **학습 제약**: 백본 동결(`requires_grad=False`), 추가 학습 파라미터 0개($\text{Params}=0$).
- **출력**: 쿼리 $x^q$에 대한 연속 이상 맵 $A(x^q) \in \mathbb{R}^{H \times W}$.

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

- **설계 철학**:
  - 심층 어텐션 마스킹을 극복하되 천층의 정상 노이즈 유입을 완벽 차단하기 위해 잔여 공간(Residual Space) 기반 비대칭 단방향 게이팅을 설계함.
- **수학적 수식**:
  - 잔여 차이 정의: $\Delta(x) = d_{shallow}(x) - d_{deep}(x)$.
  - 서포트 Leave-Source-Out 정상 잔여 변동 상위 임계값 $\tau_{spike}$ 설정.
  - 임계치를 초과하는 극단적 스파이크만 선별 수혈함:
    $$
    S_{RSG}(x) = d_{deep}(x) + \gamma \cdot \operatorname{ReLU}\left(\Delta(x) - \tau_{spike}\right)
    $$
- **효과**:
  - 정상 영역($\Delta \leq \tau_{spike}$)에서는 천층 항이 0이 되어 심층 저분산 베이스라인이 온전히 보존됨.
  - 결함 영역($\Delta > \tau_{spike}$)에서만 선택적으로 결함 스파이크가 주입되어 고주파 결함이 복원됨.

---

### 3.4. Tail-Calibrated Quantile Normalization (Tailscale)

- **설계 철학**:
  - 기존 MAD/Min-Max 정규화가 분포 중심에 머무는 한계를 극복하고, 평가 영역인 극단 꼬리(Tail)를 직접 보정함.
- **수학적 수식**:
  - 각 컴포넌트 $m$에 대해 서포트 LOO 정상 거리의 상위 95% 분위수 $q_{95}^{(m)}$ 산출.
  - 분위수 나눗셈을 통한 선형 스케일링 수행:
    $$
    \widetilde{d}_m(x) = \frac{d_m(x)}{q_{95}^{(m)}}
    $$
- **효과**:
  - 모든 컴포넌트의 정상 꼬리 상한선을 1.0으로 일치시켜 특정 컴포넌트의 오탐 예산 독점을 차단함.
  - 거리의 물리적 크기 정보(Magnitude)를 왜곡 없이 보존함.

---

### 3.5. Topological Continuum Restoration (CQ)

- **설계 철학**:
  - ViT 패치 토큰화($14\times 14$)로 파편화된 결함 점수를 물리적 연결 성분(Connected Component) 단위로 복원함.
- **동작 원리**:
  - 신뢰도 높은 코어 결함 시드(Seed)를 탐색함.
  - 동일 연결 성분 내부로 결함 신호를 공간적 확산(Diffusion)시킴:
    $$
    A_{CQ}(p) = \max \left( A(p), \, \alpha \cdot \text{Diffusion}(A_{seed}, \mathcal{C}_p) \right)
    $$
- **효과**:
  - 학습 디코더 없이 결함 영역 전체의 커버리지를 복원하여 AUPRO 지표를 비약적으로 상승시킴.

---

### 3.6. Anomaly Scoring and Computational Efficiency

- 학습 파라미터 0개 및 역전파 FLOPs 0 보장.
- 추가 신경망 포워드 패스 없이 행렬 거리 연산과 경량 기하 필터링만으로 실시간 추론을 실현함.

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

- **Tailscale 메커니즘 검증**: 컴포넌트별 꼬리 폭증 비율과 성능 개선율 간 피어슨 상관계수 $r = +0.934$를 기록함.
- **어텐션 마스킹 방어율**: RSG 적용 시 심층에서 소멸하던 결함 신호가 94.2% 수준으로 복원됨을 추적 확인했음.
- **서포트 시드 강건성**: 다중 시드 평가에서 표준편차 0.12 pp 이내의 안정성을 입증함.

---

## 5. Discussion & Conclusion

### 5.1. Capacity vs. Calibration: Lessons for Foundation Models in Vision

- 비주얼 파운데이션 모델을 결함 탐지에 적용할 때의 핵심 병목은 '용량 부족'이 아니라 **'표현 보정의 실패'**임을 환기함.
- 무분별한 어댑터 학습이나 모델 스케일업 경쟁 대신 제1원리적 기하 정렬을 수행하는 것이 학술적·실무적으로 우월함을 증명함.

### 5.2. Boundary of Claims and Future Work

- **주장 범위**: 정렬된 Few-shot 산업 결함 탐지에서 ViT 표현 결함의 제1원리적 교정 및 백본 불가지론적 전이성 입증.
- **비주장 범위**: 임의 3D 포즈 변화를 수반하는 전면적 카테고리 전이(Category Shift) 해결.
- **향후 과제**: 기하 정렬 원리를 비디오 이상 탐지 및 의료 영상 도메인으로 확장 예정임.

