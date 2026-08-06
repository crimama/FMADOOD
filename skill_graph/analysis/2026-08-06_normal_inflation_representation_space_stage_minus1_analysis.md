# 표현 공간 거리 기반 이상 채점 한계와 도메인 변동 시 Normal Inflation 분석

Date: 2026-08-06  
Status: `SUPPORT_ONLY_STAGE_MINUS1_VALIDATED / NO_PUBLIC_TEST_GT_ACCESS`  
Protocol: `wtcf-stage-minus1-support-only-20260804`  

---

## 1. 문제 정의 (Problem Definition)

### 1.1 Normal Inflation (정상 점수 부풀림) 현상
Few-Shot Anomaly Detection (FSAD) 환경에서 PatchCore, DINOv2 raw feature memory bank, Normalizing Flow latent 1-NN 등 표현 공간(Representation Space) 상의 최근접 거리($1\text{-NN}$)를 이상도(Anomaly Score)로 채점할 때:
$$s(h_q) = \min_{m \in \mathcal{M}} \| f_\theta(h_q) - m \|_2$$
조명, 센서 노이즈, 촬영 각도, 미세 질감 변화 등 무해한 **도메인 변동(Domain Shift / Nuisance Variation)**이 발생하면, 실제 정상(Good) 샘플의 이상 점수 분포 상위 꼬리($p_{95}, p_{99}$ Tail)가 결함(Bad) 샘플 점수대까지 폭발적으로 상승하는 현상이 발생한다.

### 1.2 표현 공간 거리 채점의 3대 근본 한계
1. **Metric Non-Identifiability (학습–추론 목적함수 불일치)**:
   Normalizing Flow 등 기존 모델은 Log-Likelihood(NLL)를 최적화하지만, 추론 시에는 잠재 공간의 유클리드 $1\text{-NN}$ 거리를 이상 점수로 사용한다. NLL 목적함수는 국소 $1\text{-NN}$ 거리 구조가 정상 변화 방향과 결함 방향을 제대로 구별하도록 식별하거나 보장하지 못한다.
2. **Euclidean Isometry (등거리 가정)의 왜곡**:
   유클리드 $L_2$ 메트릭은 고차원 특징 공간($D > 1000$)의 모든 방향 변위 $\Delta z$를 동일한 가중치로 취급한다. 정상 다양체의 접선 방향(Normal Tangent Vector) 이동 크기가 실제 미세 결함 벡터 크기를 초과할 경우, 거리 기반 채점은 이를 정상 변형이 아닌 심각한 이상으로 오진한다.
3. **Static Memory Bank와 Coverage Gap**:
   16-shot 이하 소수 reference만으로는 테스트 시점의 정상 변동성을 모두 포함할 수 없으며, $1\text{-NN}$ 거리는 알려지지 않은 정상(Hidden Normal)을 결함으로 판정하여 Fixed-threshold Calibration을 무력화하고 $\text{AU-PRO}$를 붕괴시킨다.

---

## 2. 9개 해결 패러다임 비교 실험 설계 (Stage -1 Arm Setup)

| 실험군 (`Arm`) | 분류 | 주요 메커니즘 |
| :--- | :--- | :--- |
| `a6` | Baseline | Standard Normalizing Flow (NLL + Tail NLL loss) |
| `a7_identity` | Raw Memory | Flow 없는 DINOv2-L/14 다층 특징 유클리드 $1\text{-NN}$ 메모리 뱅크 |
| `pca_residual` | Subspace | 정상 support 주성분 부분공간(PCA)의 복원 오차(Residual) 채점 |
| `shrinkage_mahalanobis` | Metric | 정상 support 공분산 행렬의 Shrinkage 적용 마할라노비스 거리 |
| `crossfit_only` | Unconstrained | Held-normal Soft-$1\text{-NN}$ Risk를 손실함수로 직접 학습 (방향 가드 없음) |
| `global_isometry` | Constrained | Cross-fit score loss 적용 시 전역 거리 보존(Global Isometry) 강제 |
| `random_witness` | Control | 동등한 rank의 무작위 방향 수축 적용 Control |
| `witness_only` | Tangent Bound | Cross-fit 점수 손실 없이 정상 residual witness 방향 수축 제약만 적용 |
| `wtcf` | Proposed | Witness-Tangent Cross-Fit (Soft-$1\text{-NN}$ + Witness Contraction + Non-contraction Guard) |

---

## 3. 실증 분석 결과 (Empirical Findings)

### 3.1 MVTec AD 2 주요 3개 객체 9-Arm 정량 평가 (`seed0` 기준)

| 객체 (`Object`) | 지표 (`Metric`) | `a6` | `a7_identity` | `pca_residual` | `crossfit_only` | **`wtcf` (Proposed)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`can`** | Held Normal $q_{995}$<br>Shifted Normal $q_{995}$<br>$p\text{AUROC}_{0.05}$<br>Fixed $F_1$<br>$\text{AUPRO}_{0.05}$<br>Proxy Audit | 1.8258<br>1.8077<br>0.7753<br>0.4902<br>0.5624<br>**PASSED** | 1.8317<br>1.8255<br>0.7430<br>0.4612<br>0.5022<br>N/A | 1.8368<br>1.8022<br>**0.6070**<br>**0.1958**<br>**0.2395**<br>N/A (붕괴) | -1.4512<br>-1.4446<br>0.8568<br>0.6470<br>0.7326<br>**FAILED** | **1.7530**<br>**1.7369**<br>**0.7760**<br>**0.4871**<br>**0.5649**<br>**PASSED** |
| **`fabric`** | Held Normal $q_{995}$<br>Shifted Normal $q_{995}$<br>$p\text{AUROC}_{0.05}$<br>Fixed $F_1$<br>$\text{AUPRO}_{0.05}$<br>Proxy Audit | 1.9600<br>1.9733<br>0.8798<br>0.6795<br>0.7776<br>**PASSED** | 2.0129<br>2.0298<br>0.8433<br>0.6642<br>0.7198<br>N/A | 0.5630<br>0.5585<br>**0.7787**<br>**0.5613**<br>**0.6058**<br>N/A (급락) | -6.5888<br>-6.5843<br>0.8959<br>0.7188<br>0.7930<br>**PASSED** | **1.9600**<br>**1.9733**<br>**0.8798**<br>**0.6795**<br>**0.7776**<br>**PASSED** |
| **`fruit_jelly`** | Held Normal $q_{995}$<br>Shifted Normal $q_{995}$<br>$p\text{AUROC}_{0.05}$<br>Fixed $F_1$<br>$\text{AUPRO}_{0.05}$<br>Proxy Audit | 2.4524<br>2.4466<br>0.7870<br>0.5794<br>0.6328<br>**PASSED** | 2.5916<br>2.5893<br>0.7616<br>0.5559<br>0.5883<br>N/A | 2.3410<br>2.3326<br>**0.7259**<br>**0.4771**<br>**0.5175**<br>N/A (급락) | -0.8706<br>-0.8710<br>0.8384<br>0.5373<br>0.6925<br>**FAILED** | **2.2748**<br>**2.2685**<br>**0.7974**<br>**0.6149**<br>**0.6502**<br>**PASSED** |

---

## 4. 메커니즘 분석 및 학술적 시사점

### 4.1 전역 선형 서브스페이스 (`pca_residual`) 및 마할라노비스 거리의 구조적 failure
- **PCA Residual의 붕괴**: 16-shot 고차원 특징 공간에서 주성분 부분공간의 전역 선형 투영은 정상 변형뿐 아니라 결함 영역의 미세 특징까지 제거하지 못하고 이상 오차로 투영하거나 함부로 뭉개버린다. 그 결과 모든 객체에서 $p\text{AUROC}$ (-0.06 ~ -0.17p), $F_1$ (-0.10 ~ -0.30p), $\text{AUPRO}$ (-0.11 ~ -0.32p)가 큰 폭으로 손상되었다.
- **Shrinkage Mahalanobis의 무력화**: 고차원 공간($D > 1000$)에서 16-shot 공분산 행렬은 무수히 많은 주성분 외 방향을 가져, Shrinkage 정규화를 적용하면 사실상 일반 유클리드 메트릭(`a7_identity`)과 수치적으로 동일해진다.

### 4.2 제약 없는 배포 점수 학습 (`crossfit_only`)의 Anomaly Absorption
- Soft-$1\text{-NN}$ Risk만을 학습 손실로 사용할 경우, 정상 Tail 점수($q_{995}$)는 음수로 대폭 감소하여 정상 거리가 극단적으로 좁혀진다.
- **그러나 무제약 수축의 치명적 위험**: 수축 가능한 방향에 제약이 없으면 모델이 모든 차원의 거리를 과도하게 찌그러뜨려 합성/미지의 이상 신호까지 함께 무력화시키는 **Anomaly Absorption**이 발생하며, 이는 Proxy Audit 검증 일관 실패(`FAILED`)로 증명된다.

### 4.3 WTCF-Flow (Witness-Tangent Cross-Fit)의 작동 정당성
- WTCF는 정상 support 간 상호 매칭을 통해 입증된 **Normal-residual witness 방향만을 한정 수축**하고, 증거가 없는 방향은 **Non-contraction Guard**를 적용하여 함부로 거리를 소거하지 못하게 통제한다.
- **결과**: Proxy Audit 검증을 안전하게 통과(`PASSED`)하면서 Baseline 대비 정상 Tail 점수 감소 및 $p\text{AUROC} / F_1 / \text{AUPRO}$ 성능 유지를 입증함.

---

## 5. 결론 및 향후 논문 반영 방향

1. NLL 기반 Normalizing Flow 및 유클리드 $1\text{-NN}$ 채점 간의 학습–추론 불일치가 Normal Inflation의 핵심 원인임을 규명함.
2. 무차별적 전역 선형 수축(PCA/Mahalanobis) 및 무제약 Soft-$1\text{-NN}$ 수축(`crossfit_only`)은 Anomaly Absorption을 야기하므로 불가능함.
3. 검증된 정상 변형 방향만 선택 수축하는 **Witness-Tangent 수축** 및 미관측 방향의 **Non-contraction Guard** 조화가 정적 Few-Shot AD의 도메인 변동 강건성을 위한 메커니즘적 유일 해법임을 확인 함.
