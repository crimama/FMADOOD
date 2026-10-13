# Dissecting Visual Foundation Models for Industrial Anomaly Detection: The Failure of Representation Calibration and Its First-Principles Resolution

**Hun Im\*, Author Two, Author Three, Pilsung Kang\***

---

## Index

- Abstract
- 1. Introduction
  - 1.1. Problem & Motivation: The Illusion of Foundation Backbones
  - 1.2. Key Insights: Dissection and Zero-Parameter Calibration
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

퓨샷 산업 이상 탐지(Few-shot Industrial Anomaly Detection)에서 비주얼 파운데이션 모델(ViT)의 활용이 주류로 자리잡았으나, 기존 연구들은 모델을 무작정 거대화하거나 복잡한 학습 어댑터 및 디코더를 추가하는 외부적 확장에 치중해왔다. 그러나 본 논문에서는 진정한 병목이 모델 용량이 아니라, 거시적 의미 불변성과 전역 어텐션 등 ViT 고유의 내부 메커니즘이 산업 결함의 물리적 국소성과 정면 충돌하는 '귀납적 편향의 불일치'에 있음을 지적한다. 이를 해결하기 위해, 우리는 추가 학습 파라미터 없이 백본의 표현 결함을 제1원리적으로 교정하는 **학습 파라미터 0개의 기하 정렬 프레임워크(Zero-Parameter Geometric Alignment Framework)**를 제안한다. 구체적으로, (1) 심층 어텐션에 의해 뭉개진 미세 결함을 잔여 공간에서 선별 주입하는 비대칭 잔여 스파이크 게이팅(RSG), (2) 계층 및 스케일 간 불균일한 정상 꼬리 분산을 Leave-Source-Out 분위수로 정규화하여 오탐 예산 잠식을 방지하는 Tailscale, (3) 이산적 패치 응답을 결함의 위상학적 연결 성분 내부로 확산시켜 영역 연속성을 복원하는 CQ를 도입한다. MVTec AD 2(16-shot) 벤치마크에서 제안 프레임워크는 기존 메모리 뱅크 및 학습 기반 베이스라인들을 모두 능가하며 새로운 SOTA(AUPRO 59.25%, pAUROC 77.74%)를 달성했다. 나아가 추가 재학습이나 구조 변경 없이 백본만을 DINOv2에서 DINOv3로 교체했을 때 pAUROC 82.36%(+5.25%p), F1-oracle 45.54%(+9.67%p)로 비약적 성능 향상을 보여 진정한 백본 불가지론적 플러그앤플레이(Backbone-Agnostic Plug-and-Play)를 실증함으로써, 파라미터 추가 없이도 파운데이션 모델의 진화를 온전히 흡수할 수 있는 새로운 이상 탐지 패러다임을 제시한다.

---

## 1. Introduction

### 1.1. Problem & Motivation: The Illusion of Foundation Backbones

- **파운데이션 모델 기반 산업 이상 탐지의 부상**:
  - 정상 데이터의 희소성과 결함 형태의 비정형성으로 인해 소수 참조 샘플 기반의 비지도 퓨샷 설정이 필수적임.
  - DINOv2, DINOv3, CLIP 등 대규모 자기지도 학습 ViT의 패치 표현을 메모리 뱅크에 저장하고 최근접 이웃(NN) 거리를 측정하는 패치 매칭 패러다임(PatchCore, SuperAD 등)이 사실상의 표준으로 정착됨.
  - 거대 모델의 강력한 표현력이 미세 결함 검출을 자연스럽게 해결할 것이라는 기대(The Illusion of Foundation Backbones)가 팽배함.
- **학계의 맹점: 세 갈래의 외부 확장 패러다임과 한계**:
  - MVTec AD 2와 같은 고해상도 실제 산업 환경에서 순수 사전학습 특징은 정상 영역의 조명·반사광 등 무해한 변동에 과민 반응하여 막대한 오탐(False Positive)을 발생시킴.
  - 학계는 이를 해결하기 위해 주로 다음 세 가지 외부적 확장에 의존해옴:
    1. **백본 스케일링 (Brute-force Scaling)**:
       - 모델 용량 증설로 문제를 돌파하고자 백본을 11억 개 파라미터의 ViT-Giant로 대형화하고 다중 계층 특징을 단순 합산함 (SuperAD, PatchCore, AnomalyDINO 등).
       - *한계*: 백본 내부의 어텐션 마스킹과 분산 발산을 해결하지 못한 채 무거운 연산 비용을 수반하며, 비보정 융합으로 정상 꼬리가 폭증하여 오탐 예산을 잠식함.
    2. **파라미터 효율적 미세조정 (PEFT & Learnable Adapters/Prompts)**:
       - 특징을 이상 탐지에 적응시키고자 Residual Adapter, LoRA, 학습 가능한 프롬프트 벡터를 모델 내부에 삽입하여 파인튜닝함 (PromptAD, AnomalyCLIP, AdaCLIP, AA-CLIP, AdaptCLIP, RegAD 등).
       - *한계*: 어댑터 파라미터가 특정 백본 체크포인트 좌표계에 종속되어, 백본 진화(DINOv2 $\to$ DINOv3) 시 구조 재설계 및 재학습이 불가피하여 파운데이션 모델의 전이성을 훼손함.
    3. **보조 재구성 디코더 (Auxiliary Reconstruction Decoders)**:
       - 백본 상단에 별도의 트랜스포머 디코더, 잠재 확산 모델, 또는 신경망 디노이저를 결합하여 재구성 오차로 결함을 검출함 (Dinomaly, UniAD, DecAD, DiAD, DVT 등).
       - *한계*: 백본 자체의 표현론적 결함은 방치한 채 외부 네트워크의 무거운 재구성 과정에 의존하여 학습 비용과 최적화 불안정을 초래함.
- **귀납적 편향 불일치: 모델 용량 대 표현 보정의 딜레마**:
  - 기존 세 흐름 모두 *"ViT 구조 자체가 산업 결함의 물리적 본질과 정면 충돌하는 표현 결함을 해결했는가?"*라는 근본 질문을 외면함.
  - 산업 결함의 본질은 수 픽셀 단위의 극단적 국소성(Extreme Locality) 및 고주파 경계 단절(High-Frequency Edge Discontinuity)에 있음.
  - 반면 ViT 사전학습은 거시적 의미론적 불변성(Semantic Invariance)을 극대화하고, 전역 Self-Attention은 소수 결함 신호를 다수 정상 배경으로 소프트 평균화(Soft-averaging)함.
  - 따라서 진정한 병목은 모델 '용량 부족'이나 '외부 모듈 부재'가 아니라, ViT 내부 표현이 결함의 물리 법칙과 어긋나 있는 **'표현 보정의 실패(Failure of Representation Calibration)'**에 있음.
  - 외부 파라미터를 단 1개도 추가하지 않고(Params=0), 백본 내부 메커니즘을 제1원리적으로 교정하는 기하 정렬 체계가 요구됨.

---

### 1.2. Key Insights: Dissection and Zero-Parameter Calibration

- **ViT 백본 표현 결함의 제1원리적 해부**:
  - 단순 블랙박스 취급에서 탈피하여 백본 내부 메커니즘을 정밀 해부함.
  - 심층 어텐션 결함 흡수, 기존 관행적 단순 융합의 꼬리 폭증, 이산 패치 격자화의 세 가지 구조적 결함을 수학적·실험적으로 규명함.
- **0-파라미터 기하 정렬 스택 (Zero-Parameter Geometric Alignment)**:
  - **RSG**: 심층 앵커의 안정성을 기저로 삼고, 잔여 공간에서 정상 변동을 초과하는 고주파 스파이크만 비파괴적으로 수혈함.
  - **Tailscale**: Leave-Source-Out 정상 $q_{95}$ 분위수로 컴포넌트를 정규화하여 FPR 예산 독점 현상을 수학적으로 완전 차단함.
  - **CQ**: 이산 패치 거리 응답을 위상학적 연결 성분 내부로 확산시켜 물리적 결함 연속체를 복원함.
- **백본 진화 무학습 흡수 (Backbone-Agnostic Plug-and-Play)**:
  - 어댑터 재학습이 필요한 기존 연구들과 달리, 파라미터 0개의 기하 보정 원리로 DINOv2 $\to$ DINOv3 백본 스왑 성능 폭등을 온전히 실증함.

---

### 1.3. Contributions

- **C1. ViT 백본 표현론의 심층 해부 (Theoretical & Empirical Dissection)**:
  - 산업 결함 탐지에서 ViT가 겪는 3대 구조적 결함(어텐션 결함 흡수, 분산 발산 꼬리 폭증, 이산 격자화 단절)을 수학적 메커니즘과 통계적 지표(피어슨 $r=+0.934$)로 규명함.
- **C2. 0-파라미터 제1원리적 기하 보정 프레임워크 제안 (Ours Framework)**:
  - 추가 학습 파라미터나 디코더 오버헤드 없이 백본의 구조적 결함을 정밀 타격하는 3단계 필터 스택(RSG + Tailscale + CQ)을 구축함.
- **C3. 진정한 백본 불가지론적 일반화 실증 (Backbone-Agnostic SOTA)**:
  - MVTec AD 2 벤치마크에서 SuperAD 등 기존 SOTA를 압도함 (AUPRO 59.25%, pAUROC 77.74%).
  - DINOv3 스왑을 통해 재학습 없는 완벽한 백본 진화 계승을 입증하여, 어댑터 튜닝 일변도 학계 트렌드에 대한 근본적 대안을 정립함.

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
    $$z_i^{(l+1)} = \sum_{j \in \Omega} A_{ij}^{(l)} W_V z_j^{(l)}$$
  - 산업 결함 패치 수 $|\Omega_{defect}| \ll |\Omega_{normal}|$이 성립함.
  - 심층 레이어로 진행할수록 전역 어텐션 가중치 엔트로피가 증가하여 배경 정상 패치들의 기여도가 지배함:
    $$\sum_{j \in \Omega_{normal}} A_{ij}^{(l)} W_V z_j^{(l)} \gg \sum_{k \in \Omega_{defect}} A_{ik}^{(l)} W_V z_k^{(l)}$$
- **결함 신호 소멸 실증**:
  - 심층 레이어(L23)에 도달하면 결함 토큰이 정상 컨텍스트 속으로 소프트 흡수되어 코사인 거리가 0에 가깝게 수렴함을 확인했음.
  - 심층 단독 사용 시 미세 스크래치 결함 검출률이 천층 대비 최대 40% 이상 소실됨을 실측함.

---

### 2.3. Breakdown of Conventional Fusion: Hierarchical Variance Divergence and Tail Blowup

- **계층별 분산 발산 (Hierarchical Variance Divergence)**:
  - 심층 블록(L11~L23): 의미론적 정규화로 정상 패치 간 코사인 거리 분산이 극도로 낮음 ($\sigma_{deep}^2 \ll 1$).
  - 천층 블록(L1~L6): 표면 반사광·조명 변동에 민감하여 정상 패치 간 코사인 거리 분산이 심층 대비 10~30배 폭증함 ($\sigma_{shallow}^2 \gg \sigma_{deep}^2$).
- **관행적 단순 평균(Naive Mean)의 수학적 붕괴**:
  - 기존 방식: $S_{naive}(x) = \frac{1}{M} \sum_{m=1}^M d_m(x)$.
  - AUPRO@0.05 및 pAUROC@0.05는 상위 0~5% 오탐(FPR) 극단 꼬리(Tail)에서 평가됨.
  - 분산이 수십 배 큰 천층 및 고해상도 보간 컴포넌트의 정상 꼬리가 단순 합산 맵의 FPR 예산을 독식하여 대량 오탐을 유발함.
- **꼬리 폭증과 개선율 간 피어슨 상관계수 ($r = +0.934$) 실증**:

| Class | L5@1120 Normal $q_{95}$ | L23@672 Normal $q_{95}$ | Blowup Ratio | Tailscale 개선폭 (AUPRO $\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
| `fruit_jelly` | **35.70** | 4.87 | **7.33x** | **+13.29 pp** |
| `vial` | **28.40** | 4.12 | **6.89x** | **+9.34 pp** |
| `fabric` | 5.20 | 5.84 | 0.89x | +0.12 pp (영향 없음) |

- 정상 꼬리 폭증 비율과 Tailscale 적용 시의 성능 향상폭 간 상관계수가 $+0.934$로 측정되어, 단순 융합의 붕괴 원인이 오탐 예산 잠식이었음을 완벽히 증명함.

---

## 3. Proposed Method: Zero-Parameter Geometric Alignment

---

### 3.1. Problem Formulation

- **입력**: $N$-shot 정상 서포트 집합 $\mathcal{S} = \{x_1^s, \dots, x_N^s\}$.
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

- **설계 철학**:
  - 심층 어텐션 마스킹을 극복하되 천층의 정상 노이즈 유입을 완벽 차단하기 위해 잔여 공간(Residual Space) 기반 비대칭 단방향 게이팅을 설계함.
- **수학적 수식**:
  - 잔여 차이 정의: $\Delta(x) = d_{shallow}(x) - d_{deep}(x)$.
  - 서포트 Leave-Source-Out 정상 잔여 변동 상위 임계값 $\tau_{spike}$ 설정.
  - 임계치를 초과하는 극단적 스파이크만 선별 수혈함:
    $$S_{RSG}(x) = d_{deep}(x) + \gamma \cdot \operatorname{ReLU}\left(\Delta(x) - \tau_{spike}\right)$$
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
    $$\widetilde{d}_m(x) = \frac{d_m(x)}{q_{95}^{(m)}}$$
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
    $$A_{CQ}(p) = \max \left( A(p), \, \alpha \cdot \text{Diffusion}(A_{seed}, \mathcal{C}_p) \right)$$
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

| Method | Backbone | Trainable Params | AUPRO@0.05 (%) | pAUROC@0.05 (%) | F1-oracle (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| SuperAD | DINOv2-L | 0 | 53.49 | 76.19 | 39.42 |
| PatchCore | DINOv2-L | 0 | 51.20 | 74.80 | 36.10 |
| Ours (Baseline MS1) | DINOv2-L | 0 | 56.75 | 76.43 | 41.20 |
| Ours (+ Tailscale, TC1) | DINOv2-L | 0 | 57.26 | **77.74** | 42.15 |
| Ours (+ CQ, TB1+CQ) | DINOv2-L | 0 | 58.67 | 77.35 | 43.30 |
| **Ours (Full, RSG3)** | DINOv2-L | **0** | **59.25** | 77.11 | **44.11** |

- 추가 파라미터 없이 SuperAD 대비 AUPRO **+5.76 pp**, pAUROC **+1.55 pp**를 경신하며 압도적 SOTA를 달성함.

---

### 4.3. Backbone Paradigm Shift: DINOv2 vs. DINOv3 (Backbone-Agnostic Plug-and-Play)

| Configuration | Backbone | pAUROC@0.05 (%) | F1-oracle (%) | `vial` Segmentation (%) |
| :--- | :--- | :---: | :---: | :---: |
| Ours (Full Framework) | DINOv2-L | 77.11 | 35.87 | 76.50 |
| **Ours (Full Framework)** | **DINOv3-L** | **82.36** | **45.54** | **84.07** |
| **Performance Jump ($\Delta$)** | - | **+5.25 pp** | **+9.67 pp** | **+7.57 pp (사상 최초 80% 돌파)** |

- 백본 변경 시 재학습이 필요한 어댑터 연구들과 달리, RoPE 등으로 진화한 DINOv3의 표현력을 코드 수정 없이 100% 흡수함을 실증함.

---

### 4.4. Ablation Studies: 1-to-1 Component Attribution

| Row | RSG | Tailscale | CQ | AUPRO@0.05 | pAUROC@0.05 | 대상 결함 해결 여부 |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | - | - | - | 53.49 | 76.19 | 베이스라인 (SuperAD 수준) |
| 2 | - | ✓ | - | 57.26 | 77.74 | **결함 2 해결**: 꼬리 폭증 제거로 pAUROC 대폭 상승 |
| 3 | ✓ | ✓ | - | 58.12 | 77.20 | **결함 1 해결**: 어텐션 마스킹 타격으로 미세 결함 복원 |
| 4 | ✓ | ✓ | ✓ | **59.25** | **77.11** | **결함 3 해결**: 연속체 복원으로 AUPRO 극대화 |

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
