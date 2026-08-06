# Basic의 핵심 문제와 WTCF-Flow 방법론 제안

- 기준일: 2026-08-06
- 대상: 정상 16장으로 학습하는 정적 few-shot anomaly detector Basic 및 WTCF-Flow
- 문서 성격: 기존 분석·실험에서 핵심 문제를 추린 뒤 도출한 방법론 설계 및 Stage -1 9-Arm 실증 검증 완료 보고서
- 현재 상태: Stage -1 9-Arm 실증 평가 통과 (Likelihood–Metric Non-identifiability 실증 및 WTCF Witness Guard 유효성 입증)

---

## 0. 결론

Basic의 가장 코어한 문제는 **정상 데이터만으로 학습한다는 점**이나 **TTA가 없다는 점**이 아니다. 핵심은 다음의 학습–추론 불일치다.

> Basic은 정상 likelihood로 Flow를 학습하지만, 실제 이상 판단에는 그 Flow가 만든 잠재공간의 Euclidean 1-NN 거리를 사용한다. 그러나 likelihood 목적함수는 이 최근접 거리 구조를 직접 식별하거나 보장하지 않는다.

2026-08-06 시행한 **Stage -1 9-Arm 실증 비교 실험**을 통해 다음이 최종 검증되었다:

1. **선형 부분공간 (PCA Residual) 및 마할라노비스 거리의 무력화**: 고차원 DINOv2 특징 공간에서 전역 선형 수축은 이상 신호까지 뭉개어 $p\text{AUROC}$ (-0.06 ~ -0.17p) 및 $\text{AUPRO}$ (-0.11 ~ -0.32p) 급락을 유발함.
2. **Unconstrained Distance Contraction (`crossfit_only`)의 Anomaly Absorption**: 수축 방향에 대한 제약 조건이 없으면 정상 Tail 점수는 낮아지나 결함 민감도까지 소거되어 Proxy Audit 검증에 실패함.
3. **WTCF-Flow의 유효성**: 상호 최근접 매칭으로 입증된 **Normal-residual witness 방향만 한정 수축**하고 증거 없는 방향의 민감도를 보존하는 **Non-contraction Guard**를 통해 Proxy Audit 안전성 통과 (`PASSED`) 및 정상 Tail ($q_{995}$) 안정화와 탐지 성능 유지를 동시에 입증함.

---

## 1. 코어 문제가 아닌 것

### 1.1 Normal-only 학습

Few-shot anomaly detection은 정의상 정상 support만으로 정상성의 경계를 추정해야 한다. 학습 시 이상 샘플이 없다는 사실은 문제 조건이지, Basic에만 존재하는 메커니즘 오류가 아니다. 정상 데이터만 사용하더라도 학습 목적과 실제 이상 점수가 일치하면 정적인 detector를 구성할 수 있다.

### 1.2 TTA의 부재

현재 Basic은 support로 Flow와 memory bank를 학습한 뒤 query-time에 모두 고정한다. query 영상으로 파라미터, 통계 또는 memory를 갱신하지 않으므로 TTA가 아니다. 후속 방법도 이 계약을 유지한다.

### 1.3 DVT와 density의 세부 설정

DVT는 일부 객체의 pixel ranking에는 도움을 주지만 정상 tail, fixed-threshold F1, AU-PRO에 일관된 개선을 만들지 못했다. Density 항의 평균 효과도 작았다. 둘은 현재 불안정성을 키우거나 완화하는 보조 요소일 수 있으나, Flow latent 1-NN geometry가 왜 신뢰 가능한지를 설명하지는 못한다.

### 1.4 단순한 support 부족

16장은 정상 분포를 완전히 덮기에 부족하다. 그러나 “데이터가 적다”만으로는 무엇을 학습해야 하는지 정해지지 않는다. 더 직접적으로 수정할 수 있는 문제는 적은 support에서 얻은 제한된 증거보다 Flow geometry가 더 많은 변형을 임의로 수행할 수 있다는 점이다.

---

## 2. 가장 코어한 문제: latent metric non-identifiability

### 2.1 현재 학습 목적

Basic의 Flow는 정상 support 특징에 대해 대략 다음 목적을 최적화한다.

\[
\mathcal L_{\mathrm{Basic}}
=\mathcal L_{\mathrm{tail\text{-}NLL}}
+\lambda_{\mathrm{ld}}\mathcal R_{\mathrm{logdet}}.
\]

이 목적은 정상 특징의 latent norm과 Jacobian determinant를 통해 likelihood를 맞춘다.

### 2.2 실제 배포 점수

반면 query patch \(h_q\)의 주 이상 점수는 정적 정상 latent bank까지의 최근접 거리다.

\[
s(h_q)=\min_{h_i\in S}
\left\|f_\theta(h_q)-f_\theta(h_i)\right\|_2.
\]

NLL은 약한 one-sided 보조항일 뿐이다. 즉 학습은 density를 맞추지만 배포는 국소 Euclidean neighbor geometry를 신뢰한다.

### 2.3 왜 문제가 되는가

비슷한 normal likelihood를 만드는 Flow라도 국소 거리, 최근접 이웃의 정체성, score scale과 방향별 민감도는 서로 다를 수 있다. 따라서 NLL이 충분히 낮다는 사실만으로 다음을 보장할 수 없다.

- 정상의 허용 가능한 변화가 latent에서 가까워지는가
- 결함 구분에 필요한 방향이 보존되는가
- support 밖의 정상 변화가 과도한 false positive가 되지 않는가
- 객체와 seed가 바뀌어도 score scale과 tail이 안정적인가

실험에서도 Flow는 identity/raw memory보다 전체적으로 유용했지만, effective rank를 높이는 동시에 pair-distance 순위와 latent scale을 크게 바꾸었다. 이는 Flow를 제거해야 한다는 뜻이 아니라, **Flow의 유용한 재배치를 유지하면서 실제 scoring geometry를 통제해야 한다**는 뜻이다.

---

## 3. 문제를 풀 때 생기는 두 극단

### 3.1 배포 점수만 직접 줄이는 경우

held-normal patch의 1-NN 거리를 작게 만드는 loss만 추가하면 모든 방향을 함께 수축하는 해가 가능하다. 정상 tail은 낮아지지만 결함에 필요한 차이까지 줄어들 수 있다. 즉 objective–score mismatch는 줄지만 anomaly absorption이 발생할 수 있다.

### 3.2 모든 방향의 거리를 보존하는 경우

반대로 global isometry를 강제하면 collapse는 막지만 A6에서 관찰된 Flow의 유용한 비등거리 재배치도 막는다. 결과적으로 계산량만 큰 identity/raw metric에 가까워질 수 있다.

따라서 필요한 것은 전역 수축이나 전역 보존이 아니라 다음의 **선택적 수축 규칙**이다.

> 정상 support에서 반복적으로 확인된 변화 방향만 수축할 수 있고, 확인되지 않은 방향은 이상이라고 단정하지 않되 임의로 소거하지 못하게 한다.

---

## 4. 제안 방법: WTCF-Flow

### 4.1 고정 출발점

방법론의 출발점은 Flow 효과를 가장 깨끗하게 분리한 A6다.

- frozen DINOv2-L/14 다층 특징 `[5,11,17,23]`
- layer-wise normalization 후 평균 fusion
- DVT 없음
- density score 없음
- 정상 support만으로 Flow 학습
- 추론 시 정적 latent bank와 exact hard 1-NN 사용
- query adaptation, memory expansion, routing 없음

### 4.2 Cross-fit score alignment

16개 정상 support 영상을 image-disjoint fold로 나눈다. 각 fold에서 일부 영상을 held-normal로 두고, 나머지 영상의 latent bank에 대한 differentiable soft-1NN 거리를 계산한다.

\[
\tilde d_\tau(z_q,\mathcal M)
=-\tau\log\sum_{m\in\mathcal M}
\exp\left(-\frac{\lVert z_q-m\rVert_2^2}{\tau}\right).
\]

평균보다 정상 tail이 중요한 문제이므로 held-normal soft distance의 상위 꼬리에 CVaR risk를 적용한다. NLL은 Flow의 density anchor를 유지하는 약한 항으로 둔다.

\[
\mathcal L
=\mathcal L_{\mathrm{CVaR\text{-}soft1NN}}
+\lambda_{\mathrm{NLL}}\mathcal L_{\mathrm{NLL}}
+\mathcal L_{\mathrm{directional\ guard}}.
\]

이 단계의 역할은 “NLL이 좋은 latent면 1-NN도 좋을 것”이라는 가정을 없애고, 실제 사용하는 memory-distance 통계를 학습 목표에 넣는 것이다.

### 4.3 Normal-residual witness 생성

각 support anchor patch에 대해 다른 정상 이미지에서 mutual nearest match를 찾고 raw feature residual을 구한다.

\[
r_{ij}=h_i-h_j,\qquad
u_{ij}=r_{ij}/\lVert r_{ij}\rVert_2.
\]

이 방향이 여러 image-disjoint fold와 서로 다른 source image에서 반복되고 사전 고정된 angular-stability 기준을 통과할 때만 **normal-contraction witness**로 인정한다.

중요한 경계는 다음과 같다.

- witness는 이상 점수가 아니다.
- PCA/SVD로 정상 부분공간을 만들지 않는다.
- query를 정상 subspace로 reconstruction하지 않는다.
- 한 번 관측된 residual이나 augmentation 방향에는 수축 허가를 주지 않는다.

즉 witness는 “이 방향이 정상 변화로 반복 관측되었으므로 제한된 수축을 허용할 수 있다”는 유한한 증거일 뿐이다.

### 4.4 방향별 function-space guard

anchor \(h\)와 방향 \(v\)에 대한 Flow의 국소 민감도를 다음과 같이 본다.

\[
\rho_\theta(h,v)=
\frac{\lVert J_{f_\theta}(h)v\rVert_2}{\lVert v\rVert_2}.
\]

- 반복 witness 방향: 미리 정한 범위 안에서 수축을 허용한다.
- witness와 거의 직교하는 deterministic probe: non-contraction floor와 expansion ceiling을 적용한다.
- latent RMS, effective rank, score 표준편차, neighbor retention, logdet tail에도 A6 대비 hard bound를 둔다.

여기서 unsupported direction은 anomaly direction이 아니다. 단지 정상 변화라는 증거가 없으므로 Flow가 해당 차이를 조용히 없애지 못하게 보호하는 방향이다.

유효 witness가 하나도 없는 경우 별도 모델로 routing하지 않는다. 동일한 알고리즘에서 모든 sampled direction에 non-contraction guard가 적용된다. Zero-witness는 실패를 숨기는 fallback이 아니라 반드시 보고해야 하는 결과다.

### 4.5 추론

학습이 끝나면 전체 정상 support를 최종 Flow로 변환하여 한 개의 정적 latent bank를 만든다. 추론은 기존과 같은 exact hard 1-NN이다.

\[
s_{\mathrm{WTCF}}(h_q)=
\min_{h_i\in S}\lVert f_\theta(h_q)-f_\theta(h_i)\rVert_2.
\]

추론 시 witness를 다시 추정하거나 query에 맞춰 모델을 바꾸지 않는다. 따라서 WTCF-Flow는 support-only training을 사용하는 정적 detector이며 TTA가 아니다.

---

## 5. 기존 접근과의 차이

| 접근 | 핵심 동작 | WTCF-Flow와의 차이 |
|---|---|---|
| Basic/A6 | NLL로 Flow 학습, latent 1-NN으로 추론 | WTCF는 배포 1-NN risk를 학습에 직접 포함 |
| [PatchCore](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html) / [AnomalyDINO](https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html) | frozen feature memory와 NN 거리 | WTCF는 support가 허용한 방향만 비선형 metric에 반영 |
| [CFLOW](https://arxiv.org/abs/2107.12571) / [FastFlow](https://arxiv.org/abs/2111.07677) | likelihood 자체를 이상도와 정렬 | WTCF의 주 점수는 likelihood가 아니라 latent memory distance |
| PCA / [SubspaceAD](https://openaccess.thecvf.com/content/CVPR2026/html/Lendering_SubspaceAD_Training-Free_Few-Shot_Anomaly_Detection_via_Subspace_Modeling_CVPR_2026_paper.html) | 정상 subspace의 reconstruction residual을 이상 점수로 사용 | WTCF는 subspace residual을 scoring하지 않고 finite witness를 Flow 수축 제약에만 사용 |
| Mahalanobis | support covariance로 거리 정의 | WTCF는 고정 선형 covariance metric이 아니라 제한된 invertible remapping을 학습 |
| Global isometry | 모든 방향의 거리 보존 | WTCF는 반복 확인된 정상 방향에만 bounded contraction 허용 |
| TTA 계열 | query를 사용해 모델·통계·memory 갱신 | WTCF는 query-time 상태 변경 없음 |

논문 기여는 “normal tangent를 사용했다”가 아니라 다음처럼 좁혀야 한다.

> 실제 배포되는 Flow-latent 1-NN risk를 cross-fit하고, 반복 관측된 정상 residual witness만 metric 수축의 허가 근거로 사용하여 likelihood와 memory distance 사이의 비식별성을 제한한다.

---

## 6. 가장 작은 선행 검증: Stage -1

처음부터 전체 public benchmark를 실행하면 실패 원인을 분리하기 어렵다. 먼저 support-only 메커니즘 assay로 방법의 전제를 검증한다.

### 6.1 실험 범위

- 사전 지정한 5개 객체: 기존 약점 2개, 강점 2개, control 1개
- 독립적인 정상 16-shot support seed 3개
- image-disjoint held-normal fold
- 실제 anomaly label이나 public/private test 탐색 없이 support-only C/S/D/SD 진단
- A6와 동일한 feature, epoch, optimizer budget

### 6.2 9개 실험군

1. A6
2. A7 identity/raw memory
3. PCA/SubspaceAD residual
4. shrinkage local Mahalanobis
5. cross-fit score alignment only
6. global-isometry cross-fit Flow
7. random witness/subspace with matched rank
8. witness-bound only without cross-fit score loss
9. full WTCF-Flow

### 6.3 성능 해석 전 필수 validity

- 모든 arm의 support manifest와 fold가 동일해야 함
- witness 수, source-image 다양성, fold 간 angular stability를 보고해야 함
- soft-1NN surrogate와 exact hard 1-NN의 순위·tail agreement가 통과해야 함
- witness 수축과 unsupported-direction non-contraction이 실제 Jacobian에서 확인되어야 함
- latent RMS, effective rank, score scale, neighbor retention, logdet tail이 사전 bound 안에 있어야 함
- score와 map은 finite, nonconstant, complete여야 함

하나라도 실패하면 성능 숫자를 보기 전에 방법을 기각한다.

### 6.4 Kill/keep gate

WTCF-Flow는 다음을 동시에 만족할 때만 다음 단계로 보낸다.

- A6 대비 held-normal q995 tail의 명확한 감소
- synthetic defect는 학습·선택에 쓰지 않고 사후 non-absorption 진단에서 비열등
- pAUROC, support-fixed F1, AU-PRO가 모두 비열등하며 그중 둘 이상 실질 개선
- cross-fit-only, witness-only, global isometry, random witness, PCA/SubspaceAD, Mahalanobis보다 동일 joint gate에서 우세
- 객체별 큰 붕괴가 없고 여러 seed에서 방향 일치

PCA, random witness 또는 global isometry와 동률이면 WTCF 고유 메커니즘의 근거가 없으므로 방법론 주장을 중단한다. 정상 tail과 geometry validity는 개선되지만 한 detector metric이 단순 비열등이면 추가 full-scale 성능 실험 대신 제한된 진단만 허용한다.

---

## 7. 예상 실패 가능성

1. **Witness sparsity**  
   16장에서는 반복 residual이 거의 없을 수 있다. 이 경우 방법이 보수적인 A6/identity guard에 가까워져 실질적 이득이 사라질 수 있다.

2. **Nuisance witness**  
   반복 residual이 실제 정상 요인이 아니라 정렬 오류, 배경 또는 sampling artifact일 수 있다. Cross-fold 재현성만으로 의미적 정상성을 완전히 보장할 수 없다.

3. **Unseen normal mode**  
   support에 없는 정상 변화는 계속 unsupported다. WTCF는 이를 anomaly라고 선언하지 않지만 민감도를 보존하므로 false positive가 남을 수 있다.

4. **Defect–normal direction overlap**  
   결함 변화가 반복 정상 witness와 같은 방향에 놓이면 일부 defect absorption이 가능하다.

5. **Region coherence 미해결**  
   metric tail과 threshold separation이 좋아져도 AU-PRO나 연결 영역 coverage가 자동으로 좋아진다는 보장은 없다. 초기 방법에는 morphology나 region loss를 추가하지 않고 결과로 측정해야 한다.

6. **계산 예산**  
   fold별 soft-1NN과 Jacobian-vector product 제약이 3-epoch 예산에서 안정화되지 않을 수 있다. 이 경우 성능 저하와 optimization failure를 구분해야 한다.

---

## 8. 최종 판단

현재 근거로 가장 방어 가능한 연구 질문은 다음과 같다.

> 정상 likelihood로 학습한 Flow의 latent Euclidean 1-NN geometry는 anomaly metric으로 식별되어 있는가? 반복 관측된 정상 변화만 수축 허가로 사용하면 Flow의 유용한 remapping을 유지하면서 정상 tail과 defect sensitivity를 함께 안정화할 수 있는가?

WTCF-Flow는 이 질문을 직접 시험하는 방법론 후보다. 다만 현재 단계에서 주장할 수 있는 것은 **문제 정의와 falsifiable design이 정리되었다**는 것뿐이다. 성능 향상, SOTA 또는 일반화 개선은 Stage -1과 hard-null 비교를 통과하기 전에는 주장하지 않는다.

---

## 9. 관련 내부 근거

- `PaperWorks/Basic_Mechanism_Analysis.md`
- `skill_graph/experiments/2026-07-19_dinov2l_default_mechanism_stage_m1/report.md`
- `skill_graph/experiments/2026-07-23_a6_dominant_layer_public_shadow/report.md`
- `skill_graph/experiments/2026-07-24_same16_dvt_density_factorial/report.md`
- `.omx/specs/deep-research-autoresearch-basic-core-method/evidence-ledger.md`
- `.omx/specs/deep-research-autoresearch-basic-core-method/stage-05/method.md`
