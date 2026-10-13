# Distribution Shift 환경 Few-shot AD: 구조 탐색 프레임워크

기준일: 2026-08-07  
상태: 설계 프레임워크; 실증적 주장이나 후보 승격 없음

## 1. 핵심 질문

핵심은 현재의 Normalizing Flow baseline을 사후적으로 정당화하는 일이 아니다. Distribution shift 환경의 few-shot anomaly detection에서 frozen backbone 뒤 target-specific module이 반드시 수행해야 할 역할은 무엇이며, 어떤 목적함수·제약·변환 family·parameterization이 그 계약을 가장 잘 만족하는지를 묻는 일이다.

정상 support 집합을 `S`, frozen backbone의 patch feature를 `h(x,p)`, support-conditioned transform을 `T_S`라고 둔다.

\[
z(x,p)=T_S(h(x,p)),\qquad
s(x,p)=\min_{m\in\mathcal M_S}\|z(x,p)-m\|_2.
\]

바람직한 transform은 단순 density estimator가 아니다. 반복적으로 지지되는 normal variation에는 anomaly score가 덜 반응하게 하되, defect-induced일 수 있는 국소 증거는 보존해야 한다.

## 2. 식별 가능성의 경계와 inductive bias

Normal support만으로 관측되지 않은 query 차이가 unseen normal mode인지 defect인지를 완전히 판별할 수는 없다. 따라서 타당한 방법은 이 구분을 normal-only data에서 자동으로 알아낸다고 주장하는 대신, 사용하는 inductive bias를 명시해야 한다.

이 프레임워크는 다음의 **coherent-nuisance prior**를 사용한다.

\[
h(x,p)=h_{\mathrm{content}}(x,p)+g(x)+r(x,p)+\delta(x,p),
\]

여기서 `g(x)`는 image-wide 또는 저차원·공간적으로 coherent한 nuisance(예: exposure, white balance, illumination), `r(x,p)`는 normal local variation, `delta(x,p)`는 defect-induced residual을 뜻한다. Coherent한 차이는 normal nuisance일 가능성이 더 높은 것으로 취급할 뿐, normality의 증명으로 취급하지 않는다.

이 bias는 반드시 soft하게 적용해야 한다. Illumination은 spatially varying할 수 있고 defect는 넓은 영역 또는 global하게 나타날 수 있다. 그러므로 global nuisance branch가 local 또는 broad defect evidence를 설명해 제거하지 못하도록 막아야 한다.

### 2.1 기존 Stage -1에서 반영할 제한적 인사이트

`2026-08-06_normal_inflation_representation_space_stage_minus1_analysis`와
그 원천 WTCF sentinel은 새 방법의 성공 근거가 아니라, 구조 탐색에
반영할 **설계 제약**을 제공한다.

| 관찰된 인사이트 | 새 구조 탐색에서의 반영 |
|---|---|
| Raw/latent 1-NN은 적은 support가 덮지 못한 normal variation에 민감하다. | Held-normal tail을 normal-shift alignment의 필수 지표로 둔다. |
| Deployed 1-NN을 무제약으로 낮추면 feature geometry와 defect evidence가 함께 collapse할 수 있다. | 모든 family에 defect-retention 및 geometry validity gate를 둔다. |
| 전역 PCA/subspace 또는 고차원 covariance metric만으로는 few-shot local defect를 안정적으로 보존하지 못했다. | Global nuisance correction은 저차원·공유·soft한 branch로 제한하고, local residual scorer를 분리한다. |
| Normal residual witness는 후보 direction을 제안할 수 있지만, 기존 WTCF sentinel은 selective contraction validity를 통과하지 못했다. | Witness/correspondence는 Flow, Flow Matching 등에서 검증할 수 있는 evidence source일 뿐, 이미 검증된 해법으로 가정하지 않는다. |

따라서 기존 Stage -1은 “무제약 수축을 피하고, defect preservation과
geometry audit을 먼저 검증해야 한다”는 교훈으로만 사용한다. WTCF의
성능 또는 유일성을 근거로 후보 family를 우선순위화하지 않는다.

## 3. 요구 조건

| ID | 요구 조건 | 필요한 증거 |
|---|---|---|
| R1 | Target specificity | 동일 frozen-backbone raw-memory control 대비 개선. |
| R2 | Normal-shift alignment | 정의된 shift에서 held-normal high-tail score 감소. |
| R3 | Coherent nuisance separation | Global photometric 및 smooth-illumination shift에 대한 반응 감소. |
| R4 | Defect preservation | Alignment 뒤 local·broad defect contrast 비열등. |
| R5 | Patch locality | Image-level ranking뿐 아니라 region coverage와 AU-PRO 비열등. |
| R6 | Few-shot stability | Support seed와 image-disjoint support split에 걸친 효과 안정성. |
| R7 | Static protocol | Query-time parameter, memory, support-statistic update 없음. |
| R8 | Geometry auditability | Neighbor, scale, rank, local sensitivity 변화를 검사 가능. |
| R9 | Objective-score alignment | Training surrogate와 deployed hard 1-NN score의 일치. |

## 4. 공통 factorized interface

모든 구조 후보는 아래 interface 안에서 비교할 수 있어야 한다.

```text
support images -> frozen patch features -> support memory / normal correspondence evidence

query image -> frozen patch features
            -> robust global pooling -> nuisance code c(x)
            -> constrained global/coherent canonicalization
            -> canonicalized local patch features -> static support 1-NN -> anomaly map
```

Forward pass에서 `c(x)`를 계산하는 것은 persistent parameter, memory, support statistic을 갱신하지 않는 한 test-time adaptation이 아니라 static conditional inference다.

초기 core 비교는 동일한 static local 1-NN score를 고정하고 density fusion, RGB guidance, morphology를 끈다. 이 요소들은 transform family의 우위를 판단하는 근거가 아니라 이후 output-chain ablation으로 분리한다.

## 5. 설계 차원

| 설계 차원 | 질문 | 예시 |
|---|---|---|
| Learning target | 어떤 normal relation을 최적화하는가? | NLL, cross-fit 1-NN tail, correspondence consistency. |
| Safety constraint | 무엇이 흡수되면 안 되는가? | defect retention, non-contraction, identity-near bound. |
| Transform family | Feature geometry를 어떻게 바꾸는가? | affine adapter, invertible flow, ODE transport, denoiser. |
| Parameterization | Transform을 무엇으로 구현하는가? | MLP, CNN, Transformer/DiT. |
| Scoring | 무엇이 anomaly evidence가 되는가? | static 1-NN, density, local residual. |

DiT는 Normalizing Flow나 diffusion과 같은 층위의 method family가 아니라 parameterization option이다. 예를 들어 Flow Matching + DiT vector field와 diffusion + DiT denoiser는 모두 의미 있는 조합이다.

## 6. Family ladder: naive 결과로 family를 기각하지 않는다

각 family는 아래 단계를 거쳐 평가한다.

| 단계 | 목적 | 허용되는 결론 |
|---|---|---|
| N0: naive null | Default instance의 failure mode를 드러낸다. | 이 naive instance는 불충분하다. |
| N1: minimal task alignment | 명시된 requirement failure 하나를 해결한다. | Targeted modification이 그 failure에 효과가 있는지 판단한다. |
| N2: requirement-aware design | Objective, safety constraint, structure를 정렬한다. | Viable한 task-specific candidate다. |
| N3: bounded limit test | 사전 등록한 local design region을 검증한다. | 이 정의된 design region은 추가 우선순위의 근거가 부족하다. |

N0 implementation의 실패만으로 family를 기각하지 않는다. Task-aligned N1/N2 variant와 bounded N3 test까지 실패한 뒤에만 해당 family의 우선순위를 낮춘다.

| Family | N0 naive null | N1 minimal task alignment | N2 requirement-aware design |
|---|---|---|---|
| Identity memory | Raw frozen feature 1-NN. | Robust feature normalization. | Coherent nuisance 제거 뒤 local residual 1-NN. |
| Low-DOF adapter | Global 또는 position affine transform. | Small global-code FiLM/affine correction. | Robust pooled global branch + local non-contraction guard. |
| Normalizing Flow | Normal NLL 뒤 latent 1-NN. | NLL + cross-fit deployed-distance alignment. | Global-conditioned patch-local flow + selective contraction/geometry guard. |
| Flow Matching | Unconstrained normal transport field. | Identity-near conservative transport. | Correspondence-supported vector field + global/local transport 분리. |
| Diffusion / score model | Support-only denoising 또는 score matching. | Frozen prior + small support adapter. | Global appearance 처리 + local residual score + defect-safe constraint. |
| DiT parameterization | Full fine-tuning null. | Frozen DiT + PEFT. | Constrained flow 또는 score field의 restricted-attention 구현. |

## 7. 정당한 개선의 기준

Depth, width, epoch, attention span만 늘리는 것은 구조적 답이 아니라 tuning이다. 변경은 관측된 requirement failure를 해결하고 그에 대응하는 diagnostic이 있을 때만 정당화된다.

| Family | 정당한 task-specific 개선 |
|---|---|
| Normalizing Flow | NLL에 더해 image-disjoint held-normal 1-NN tail을 최적화하고, support-evidenced direction으로만 contraction을 제한. |
| Flow Matching | Normal correspondence가 지지하는 direction에서만 transport를 허용하고 unsupported direction을 제약. |
| Diffusion | Reconstruction quality 대신 frozen prior와 local score/residual evidence를 사용하고 defect smoothing을 제약. |
| DiT | Restricted local attention과 low-rank adaptation 하에서 constrained field를 parameterize하는 용도로만 사용. |
| Adapter | Shared·low-dimensional coherent-shift correction만 허용하고 arbitrary patchwise correction은 금지. |

## 8. Global branch 안전 계약

| 안전장치 | 목적 |
|---|---|
| Low-dimensional nuisance code | Global branch가 local defect를 암기하지 못하게 한다. |
| Robust global pooling | 작은 anomaly region의 오염을 제한한다. |
| Shared 또는 smooth transform | Global branch의 arbitrary patchwise repair를 막는다. |
| Identity-near regularization | 불필요한 canonicalization을 막는다. |
| Local non-contraction guard | Global correction 뒤에도 local defect residual 민감도를 유지한다. |
| Broad-defect stress test | Global-looking defect가 자동으로 정상화되지 않음을 검증한다. |

## 9. 공통 실험 및 의사결정 계약

모든 candidate transform은 frozen backbone, feature fusion, preprocessing, resolution, support manifest, static memory protocol, continuous patch-level output을 공유한다. Primary metric은 pAUROC@0.05, raw oracle F1, AU-PRO@0.05이며 단일 metric으로는 충분하지 않다.

필수 mechanism diagnostic은 다음과 같다.

1. held-normal `q_0.995` score tail;
2. local 및 broad defect retention;
3. global photometric 및 smooth-illumination shift response;
4. raw-to-latent distance rank와 nearest-neighbor identity retention;
5. latent RMS, effective rank, 그리고 해당 시 Jacobian/log-determinant behavior;
6. support-seed와 image-disjoint split stability;
7. training surrogate와 deployed hard 1-NN score의 agreement.

Candidate는 shifted-normal false positive를 줄이고, defect evidence를 보존하며, material한 pAUROC/F1/AU-PRO collapse 없이 few-shot·geometry audit을 함께 통과할 때만 다음 단계로 진행한다. 이 성질 없이 macro score만 양수인 것은 mechanism claim의 근거가 될 수 없다.

## 10. 현재 시사점

기존 Normalizing Flow는 연구 질문의 전제가 아니라 하나의 N1/N2 candidate로 취급해야 한다. 현재의 NLL-to-latent-1-NN objective mismatch는 여전히 설계 공백이다. Flow Matching, diffusion, DiT는 동일한 selective alignment와 defect preservation requirement에 대한 명시적 답을 갖는 형태로 구현될 때 열린 candidate surface로 남는다.
