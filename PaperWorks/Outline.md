# Extrinsic Robustness for Few-Shot Anomaly Detection under Distribution Shift

- 최초 작성: 2026-07-11
- 서사 재구성: 2026-07-14
- 문서 목적: foundation-model 기반 few-shot AD에서 distribution shift가 anomaly evidence를 어떻게 불안정하게 만드는지 정의하고, 이를 외재적으로 보완하는 세 구성요소와 검증 논리를 정리
- 핵심 구성요소:
  - DVT-lite
  - Static Flow-LatentBank with weak NLL correction
  - RGB Guide
- 작성 원칙:
  - 관찰된 결과, 연구 가설, 방법 설계, 향후 검증을 구분
  - 각 구성요소의 단독 novelty보다 하나의 shift-robustness framework 안에서의 역할을 강조
  - public benchmark 결과는 shadow evidence로만 사용
  - untouched/private 평가 전에는 최종 우월성·일반화 claim을 보류
  - full DVT와 DVT-lite를 구분
  - Flow projection, latent distance, density correction, RGB guidance의 기여를 분리

## 0. Central Thesis

### 0.1 출발점

- foundation model은 강력한 semantic representation을 제공하므로 소수 normal support만으로도 patch-memory 기반 anomaly detection을 가능하게 함
- 일반적인 scoring은 query patch와 normal memory의 nearest-neighbor distance에 기반:

  \[
  s(x_i^q)=\min_{m\in\mathcal M_S}d\bigl(\phi(x_i^q),m\bigr)
  \]

- 기본 직관:
  - normal support에서 멀수록 anomaly일 가능성이 높음

### 0.2 Distribution shift가 만드는 핵심 문제

- 기존 직관은 support와 query의 정상 feature가 충분히 비교 가능하다는 가정에 의존:

  \[
  \widehat P_S^N(\phi)\approx P_Q^N(\phi)
  \]

- distribution shift에서는 이 가정이 깨짐:

  \[
  \widehat P_S^N(\phi)\neq P_Q^N(\phi)
  \]

- 따라서 shifted normal과 true anomaly가 모두 높은 support distance를 만들 수 있음:

  \[
  \text{high score}
  \Leftarrow
  \{\text{shifted normal},\ \text{true anomaly}\}
  \]

- 핵심 명제:
  - **Strong representation does not guarantee reliable anomaly evidence under distribution shift.**
  - foundation model의 표현력이 강하다는 것과, 그 feature로 계산한 high score가 anomaly를 의미한다는 것은 별개의 문제

### 0.3 통합 관점: Shift-Induced Anomaly Evidence Degradation

- 관측 score를 개념적으로 다음과 같이 해석:

  \[
  s_{obs}(x)\approx s_{anom}(x)+b_{shift}(x)
  \]

  - \(s_{anom}\): 실제 defect가 만드는 anomaly evidence
  - \(b_{shift}\): benign support–query 차이가 anomaly score에 유입시킨 성분
- 목표:

  \[
  \min b_{shift}
  \quad\text{while preserving}\quad
  s_{anom}
  \]

- few-shot과 distribution shift가 표현 편향, 참조 편향 및 공간 증거 저하를 유발하는 것으로 봄:

  \[
  \{\text{Few-shot},\ \text{Distribution Shift}\}
  \rightarrow
  \{\text{Representation Bias},\ \text{Reference Bias},\ \text{Spatial Evidence Degradation}\}
  \]

### 0.4 제안 방법의 통합 구조

  \[
  \underbrace{\text{DVT-lite}}_{\text{debias descriptor}}
  \rightarrow
  \underbrace{\text{Static Flow-LatentBank}}_{\text{debias reference comparison}}
  \rightarrow
  \underbrace{\text{RGB Guide}}_{\text{restore spatial evidence}}
  \]

| 단계 | Shift가 만드는 문제 | 구성요소 | 역할 |
|---|---|---|---|
| Feature representation | 동일 정상 part가 layout에 따라 다르게 표현될 수 있음 | DVT-lite | 비교할 descriptor 보정 |
| Sparse-reference matching | Shifted normal이 제한된 memory에서 과도하게 멀어질 수 있음 | Static Flow-LatentBank | normal reference의 comparison geometry 보정 |
| Pixel localization | Coarse semantic evidence가 query의 fine structure를 충분히 보존하지 못함 | RGB Guide | query-native spatial detail 복원 |

### 0.5 한 문장 thesis

> **We introduce an extrinsic robustness stack that mitigates shift-induced bias in feature representation and sparse-reference matching, and then restores query-native spatial structure for anomaly localization.**

### 0.6 범위

- 대상:
  - 동일 industrial object 안의 support–query layout, appearance, acquisition 및 sparse-reference shift
- 주장하지 않는 범위:
  - arbitrary category/domain shift의 일반적 해결
  - support에 없는 모든 normal mode의 복원
  - geometric registration 또는 non-rigid correspondence 해결
  - RGB appearance shift에 대한 완전한 invariance

## 1. Abstract 흐름

1. Foundation models provide strong transferable features that enable few-shot anomaly detection through nearest-neighbor matching against a sparse normal memory.
2. Under support–query distribution shift, however, high feature deviation is no longer uniquely attributable to anomalies because benign shifted normal regions can also lie far from the support.
3. We interpret this high-score ambiguity through three problems induced by the few-shot setting and distribution shift: representation bias, sparse-reference bias, and spatial evidence degradation.
4. We introduce an extrinsic robustness stack on top of a frozen foundation model:
   - DVT-lite suppresses support-repeated position-conditioned descriptor components.
   - Static Flow-LatentBank constructs a support-conditioned latent geometry while retaining explicit normal references.
   - RGB Guide restores query-native edge and local structure to the coarse semantic anomaly map.
5. Flow likelihood is not used as the main detector; latent 1-NN distance is the main score and NLL is restricted to a weak one-sided correction.
6. Evaluation separates component attribution, resource regime, support-seed stability, worst-class harm, public shadow evidence, and untouched final claims.
7. 현재는 superiority보다 “is designed to”와 “we evaluate whether” 수준으로 작성.

## 2. Introduction

### 2.1 Foundation Models Enable Few-Shot AD

- 산업 AD에서는 defect의 형태와 크기를 미리 알기 어렵고 anomaly label은 부족함
- foundation model의 pretrained patch feature는 별도 defect supervision 없이 semantic comparison을 가능하게 함
- few-shot FM-based AD는 소수 normal support의 patch memory와 query patch를 NN으로 비교해 dense anomaly map을 생성
- 강점:
  - 적은 normal sample
  - 간단한 static memory
  - training-free 또는 lightweight adaptation

### 2.2 High-Score Ambiguity under Distribution Shift

- 기존 방식은 high support distance를 anomaly evidence로 해석
- 하지만 MVTec AD 2와 같은 shifted setting에서는 lighting, layout, material appearance 및 acquisition condition이 support와 query 사이에서 달라질 수 있음
- benign normal variation과 anomaly가 모두 support distance를 증가시킴
- 결과:

  \[
  s(x)\uparrow\not\Rightarrow y=\mathrm{anomaly}
  \]

- 논문의 중심 질문:
  - **How can anomaly evidence from a frozen foundation model remain reliable when shifted normal and true anomaly both produce high scores?**

### 2.3 Shift-Induced Evidence Degradation

- few-shot에서는 정상성의 근거가 소수 support와 그 패치 표현에 제한되어 정상 분포를 불완전하게 근사
- support에 충분히 관측되지 않은 조명, 배치, 외관 및 촬영 조건이 query에 나타나면 이 취약성이 증폭됨
- 즉, distribution shift가 shifted normal을 support coverage 밖으로 이동시키며 표현, 참조 및 공간 증거 문제를 심화한다고 봄

#### A. Descriptor bias

- global NN은 모든 support patch를 탐색하지만 동일 정상 pattern이 위치에 따라 같은 feature로 표현되는 것까지 보장하지 않음
- support와 query의 layout이 달라져 동일 normal part가 다른 grid 위치에 나타날 수 있음
- feature가 grid position에 민감하면 descriptor가 달라지고 support distance가 불필요하게 증가할 수 있음
- few-shot에서는 다양한 layout/context 변화를 상쇄할 support가 충분하지 않을 수 있으므로, shift가 클수록 정상 descriptor mismatch가 심해질 수 있음

#### B. Sparse-reference bias

- 정의: source 정상분포에서 선택된 유한 support를 target query의 정상성 기준으로 사용할 때 발생하는 coverage mismatch
- few-shot memory는 $P_S^N$ 전체가 아니라 선택된 몇 장의 empirical patch set이므로, 자주 관측된 정상 영역은 조밀하고 드물거나 미관측된 영역은 성김
- no-shift에서는 새로운 normal query도 support coverage 안에 놓일 가능성이 상대적으로 높음
- distribution shift로 $P_Q^N$의 probability mass가 support memory의 low-density/unobserved region으로 이동하면, 정상 query도 가까운 reference를 찾기 어려울 수 있음
- 따라서 NN distance와 density score가 defect뿐 아니라 부족한 reference coverage에도 반응하여 false positive를 만들 수 있음
- shift가 reference 자체를 바꾸는 것이 아니라, 고정된 few-shot reference와 target normal distribution 사이의 mismatch를 확대해 reference bias를 더 두드러지게 함

#### C. Spatial evidence degradation

- 정의: patch-level anomaly score가 현재 query의 정확한 boundary와 local structure를 충분히 특정하지 못하는 문제
- 한 token이 일정 영역의 여러 pixel을 집약하므로 anomaly response가 token 내부 어디에서 발생했는지에 관한 정보가 제한됨
- 단순 upsampling은 tokenization/downsampling에서 약화된 edge, thin defect 및 local contrast를 복원할 수 없음
- 이 한계는 no-shift에서도 존재하지만, layout/scale/viewpoint/illumination shift로 object boundary와 normal appearance transition의 위치가 달라지면 더 두드러질 수 있음
- shifted query에서는 한 token이 foreground/background 또는 서로 다른 local appearance를 함께 집약하여 정상 경계 응답이 번지거나 실제 defect response가 약화될 수 있음
- shift가 공간 정보를 직접 소실시키는 것이 아니라, fixed patch resolution이 변화된 query structure를 표현하지 못하는 경우를 늘려 기존 spatial loss의 영향을 심화
- query RGB는 shift 이후의 boundary와 local contrast를 높은 해상도로 직접 보존하므로 query-native refinement의 근거로 사용

### 2.4 Limitations of Existing Responses

- 기존 few-shot AD의 주된 초점:
  - pretrained/foundation representation의 adaptation 또는 multi-layer aggregation
  - normal/anomaly separability가 높은 feature space 구성
  - normal memory와의 matching 및 retrieval 성능 향상
- 이러한 접근은 강한 representation을 제공하지만 support와 query의 normal representation이 충분히 comparable하다는 가정에 의존하는 경우가 많음
- registration, augmentation, memory selection, position-aware modeling 등 shift의 일부 현상을 완화하는 연구는 존재
- 그러나 distribution shift가 normal descriptor, few-shot reference coverage 및 spatial localization에 유입시키는 bias를 명시적인 대상으로 함께 다루는 접근은 상대적으로 제한적
- 본 연구의 차별점은 새로운 strong representation 학습이 아니라 frozen representation에서 shift-induced bias를 분리해 완화하는 것
- 절대적 주장 회피: 기존 방법에 shift 대응이나 extrinsic module이 전혀 없다는 의미는 아님

### 2.5 Proposed Extrinsic Robustness Stack

- frozen foundation model을 재학습하지 않고 세 구성요소를 직렬 적용
- DVT-lite:
  - shift가 descriptor comparability에 유입시키는 position-conditioned component 완화
- Static Flow-LatentBank:
  - corrected support representation을 support statistics로 표준화
  - support-fitted Flow를 통해 normal support distribution이 표준화된 latent space 구성
  - raw feature geometry 대신 latent 1-NN distance로 query와 실제 transformed support instance를 비교
  - transformed support instance는 static bank로 보존하며 unseen mode를 생성하거나 coverage를 확장하지는 않음
- RGB Guide:
  - fixed patch resolution에서 손실된 boundary와 local contrast를 현재 query RGB에서 직접 가져옴
  - shifted query의 실제 공간 구조를 guide로 사용해 coarse anomaly evidence를 query-native structure에 맞게 정제
- 전체 해석:
  - DVT-lite: layout shift로 인한 position-conditioned representation bias 완화
  - Flow-LatentBank: support-normalized latent space를 통한 reference bias 완화
  - RGB Guide: shifted query RGB를 통한 spatial evidence restoration

### 2.6 Candidate Contributions

1. **Problem formulation**
   - distribution shift에서 shifted normal과 anomaly가 모두 high score를 만드는 현상을 shift-induced anomaly evidence degradation으로 정식화
2. **Extrinsic robustness stack**
   - support-fitted positional correction과 Static Flow-LatentBank를 결합해 frozen FM의 descriptor 및 sparse-reference matching을 보정
3. **Query-native spatial restoration**
   - 학습 없이 current query RGB를 사용해 coarse anomaly map의 fine spatial detail 복원
4. **Matched evaluation protocol**
   - DVT, Flow projection, latent distance, weak NLL, RGB Guide의 기여와 resource/claim boundary를 분리

## 3. Related Work

### 3.1 Foundation-Model Patch Memory for Few-Shot AD

- PatchCore/SuperAD 계열:
  - pretrained patch feature와 normal memory의 NN distance 사용
- 장점:
  - simple, strong, few-shot compatible
- 한계:
  - memory가 선택된 support normal mode에 치우침
  - shifted normal과 anomaly가 모두 high distance를 만들 수 있음
  - pretrained feature geometry를 anomaly metric으로 직접 사용
- 본 연구:
  - static memory 원칙은 유지
  - descriptor와 comparison geometry를 외재적으로 보정

### 3.2 Positional Artifact, Position-Aware AD, and Registration

- DVT:
  - ViT dense feature의 positional artifact를 분석하고 neural-field/student denoiser로 제거
- PNI:
  - aligned industrial setting에서 position을 유용한 normal prior로 적극 활용
- RegAD/FR-PatchCore:
  - support–query misalignment를 registration으로 해결
- 본 연구의 차이:
  - full DVT 또는 registration이 아님
  - support position first moment를 이용한 lightweight symmetric correction
  - aligned regime에서 position이 유용할 수 있음을 인정하고 shifted-layout regime에서만 nuisance hypothesis를 검증

### 3.3 Flow beyond Likelihood Scoring

- 전형적인 Flow AD의 NLL-only detection과 구분
- 관련 계열:
  - NF를 feature transformation target으로 사용하는 reconstruction 방식
  - NF-derived representation에서 KNN/distance를 사용하는 방식
  - memory와 NF feature를 결합하는 decoder 방식
- 본 연구의 차이:
  - frozen FM patch의 Flow output latent에 static support bank 구성
  - patchwise latent 1-NN이 main anomaly score
  - NLL은 weak one-sided auxiliary correction으로 제한

### 3.4 Query-Guided Spatial Refinement

- PNI:
  - RGB image와 coarse anomaly map을 learned refinement network에 입력
- CostFilter-AD:
  - query-guided learned filtering으로 matching cost noise를 억제하고 edge 보존
- 본 연구의 차이:
  - synthetic anomaly supervision이나 learned decoder를 사용하지 않음
  - fixed classical guided filter를 final continuous anomaly map에 직접 적용
  - lightweight, deterministic, training-free refinement

## 4. Problem Definition

### 4.1 Normal-Only Few-Shot Setting

- normal support:

  \[
  \mathcal S=\{x_j^n\}_{j=1}^{N}
  \]

- distributions:
  - \(\widehat P_S^N(h,u)\): support-derived empirical normal feature distribution
  - \(P_Q^N(h,u)\): benign query-normal distribution
  - \(P_Q^A(h,u)\): anomaly-induced local distribution
- few-shot shift:

  \[
  \widehat P_S^N(h,u)\neq P_Q^N(h,u)
  \]

- output:
  - dense continuous anomaly map \(A(x^q)\)
- supervision:
  - fitting과 statistics 추정에는 normal support만 사용
  - query label로 model, memory, hyperparameter를 갱신하지 않음

### 4.2 Objective

- support-calibrated threshold \(\tau_S\)에서 함께 평가:

  \[
  \operatorname{FPR}_{normal}
  =\Pr_{o\sim P_Q^N}[s_{\mathcal S}(o)>\tau_S]
  \]

  \[
  \operatorname{TPR}_{anom}
  =\Pr_{o\sim P_Q^A}[s_{\mathcal S}(o)>\tau_S]
  \]

- 성공:
  - shifted query-normal false positive 감소
  - anomaly ranking, AP, component recall, response retention 유지 또는 개선
  - support seed에 대한 ranking과 threshold 안정성 개선
  - query-native spatial structure를 반영한 localization 개선
- 실패:
  - 전체 score를 단순히 낮춰 FPR만 감소
  - true anomaly response를 함께 억제
  - 특정 object의 큰 harm를 mean gain으로 은폐

### 4.3 Research Questions

- RQ1: DVT-lite가 layout shift에서 normal descriptor의 불필요한 support-distance 증가를 완화하는가?
- RQ2: Static Flow-LatentBank가 sparse support를 raw FM geometry에서 비교하는 것보다 reliable anomaly ranking을 제공하는가?
- RQ3: weak NLL correction은 latent distance에 complementary evidence를 제공하며 no-harm를 만족하는가?
- RQ4: RGB Guide가 query-native structure를 이용해 coarse anomaly localization을 개선하며 worst-class harm를 통제하는가?
- RQ5: 전체 stack이 matched resource/evaluator와 support seed 변화에서 mean gain과 stability를 만족하는가?

## 5. Method

### 5.1 Overview

  \[
  H
  \xrightarrow{\text{DVT-lite}}
  \widetilde H
  \xrightarrow{\text{Flow-LatentBank}}
  d_z
  \xrightarrow{\text{weak NLL}}
  A_{coarse}
  \xrightarrow{\text{RGB Guide}}
  A_{rgb}
  \]

| 단계 | 입력 | 출력 | 역할 |
|---|---|---|---|
| Frozen backbone | support/query image | semantic patch feature \(H\) | broad anomaly representation |
| DVT-lite | \(H\) | corrected feature \(\widetilde H\) | descriptor bias 완화 |
| Static Flow-LatentBank | \(\widetilde H\) | latent NN distance \(d_z\) | reference comparison 보정 |
| Weak NLL | \(d_z\), support NLL statistics | \(A_{coarse}\) | one-sided density evidence 추가 |
| RGB Guide | \(A_{coarse}\), query RGB | \(A_{rgb}\) | spatial evidence 복원 |

- 모든 fitting 후 Flow, bank, statistics를 고정
- query/test stream으로 memory를 갱신하지 않음
- Basic default configuration:
  - frozen DINOv2 ViT-L/14, layers \([5,11,17,23]\)
  - shorter-edge resolution 672; `sheet_metal`만 448
  - exact SuperAD-selected 16 normal supports, seed 0
  - DVT-lite on
  - Flow latent projection과 static support bank on
  - LOO standardization off
  - latent 1-NN distance + one-sided weak NLL correction, density weight 0.25
  - fixed half-scale RGB guided-r8, \(\epsilon=0.01\)
  - fixed close--fill--erode binary morphology

### 5.2 Frozen Foundation Backbone

- default backbone: DINOv2 ViT-L/14
- input resolution: shorter edge 672; `sheet_metal`은 초광각 입력의 token 증가를 제한하기 위해 448
- layers: \([5,11,17,23]\)
- fusion: layer-normalized mean
- 역할:
  - strong semantic patch representation 제공
  - 자체적으로 shift-robust score를 보장한다고 가정하지 않음

### 5.3 DVT-lite: Support-Fitted Descriptor Debiasing

- support position field:

  \[
  \mu_{pos}(u)=\frac{1}{N}\sum_j H_j^s(u),
  \qquad
  \mu_{global}=\frac{1}{N|\Omega|}\sum_j\sum_v H_j^s(v)
  \]

  \[
  G_{pos}(u)=\mu_{pos}(u)-\mu_{global}
  \]

- correction:

  \[
  T_{\eta}(H)(u)=H(u)-\eta_{DVT}G_{pos}(u),
  \qquad \eta_{DVT}=1
  \]

- application:
  - \(G_{pos}\)는 normal support만으로 추정
  - support와 query에 동일 transform 적용
  - corrected support로 Flow와 latent bank 구성
- 역할:
  - layout 변화가 normal patch의 descriptor distance로 전달되는 정도 완화
- 한계:
  - position nuisance와 aligned object structure의 완전한 분리를 보장하지 않음
  - relative pose를 정렬하지 않음
  - full DVT reproduction이 아님

### 5.4 Static Flow-LatentBank: Support-Conditioned Reference Geometry

- preprocessing:
  - DVT-lite 이후 support statistics로 feature standardization
  - support-distance LOO standardization은 사용하지 않음
- Flow and bank:

  \[
  z=f_{\theta}(x),
  \qquad
  \mathcal M_z=\{f_{\theta}(x_{j,u}^s)\}_{j,u}
  \]

- main distance:

  \[
  d_z(i)=\min_{m\in\mathcal M_z}\|z_i^q-m\|_2
  \]

- fitting:
  - patch-wise affine-coupling MLP Flow
  - normal support patch만 사용
  - fitting 후 Flow와 latent bank 고정
- 역할:
  - sparse support instance를 explicit reference로 보존
  - raw FM geometry에 대한 normality comparison 의존 완화
- 한계:
  - unseen normal mode를 생성하지 않음
  - sparse coverage 자체를 해결하지 않음
  - Flow projection 단독 우월성은 matched ablation으로만 판단

### 5.5 Weak Density Correction and Coarse Score

- support-relative one-sided NLL:

  \[
  r_{NF}(i)=\operatorname{ReLU}
  \left(
  \frac{\ell_{NF}(x_i^q)-\mu_{NF}^{s}}{\sigma_{NF}^{s}}
  \right)
  \]

- final coarse score:

  \[
  s_{coarse}(i)=d_z(i)+0.25\,r_{NF}(i)
  \]

- 역할:
  - latent distance를 main anomaly evidence로 유지
  - support보다 NLL이 높은 query patch에만 약한 penalty 추가
- 한계:
  - NLL-only를 main localization score로 사용하지 않음
  - weak density의 기여는 distance-only arm과 분리해 검증

### 5.6 RGB Guide: Query-Native Spatial Restoration

- input:
  - upsampled continuous coarse map \(A_{coarse}\)
  - current query RGB \(I_{RGB}^q\)
- refinement:

  \[
  A_{rgb}=\mathcal G(A_{coarse},I_{RGB}^q)
  \]

- fixed setting:
  - half-scale guided filter
  - radius 8
  - \(\epsilon=0.01\)
  - thresholding과 morphology 이전 적용
- 역할:
  - semantic anomaly evidence는 유지
  - query RGB의 edge, contour, local contrast로 fine spatial detail 복원
- adaptation boundary:
  - ground truth와 anomaly label 미사용
  - backbone, Flow, bank, statistics 미갱신
- 한계:
  - RGB edge와 anomaly boundary의 일치를 보장하지 않음
  - texture edge에 의해 score가 왜곡될 수 있음

### 5.7 Method Claim Boundary

- 단독 novelty로 주장하지 않음:
  - positional artifact suppression
  - normalizing flow
  - memory bank와 NN scoring
  - RGB guided filtering
- candidate novelty:
  - few-shot support–query shift에서 descriptor–reference–localization을 잇는 extrinsic robustness stack
  - support-fitted symmetric DVT-lite
  - Flow output patch space의 static support latent bank와 NN scoring
  - training-free query-RGB refinement와의 결합

## 6. Experimental Design

### 6.1 Main Setting

- primary benchmark: MVTec AD 2 few-shot
- primary baseline: SuperAD
- default method configuration: DINOv2-L Basic, exact SuperAD-selected 16 supports, seed 0
- reporting ladder:
  - matched-backbone DINOv2-L Ours Basic: method gain
  - Ours-R with DINOv2-R: representation scaling
  - Ours++/Ours-H+ with DINOv3-H+: best-performance configuration
- method superiority와 backbone scaling을 분리해 해석

### 6.2 Data Roles and Resource Regimes

- P16-superad:
  - Ours Basic의 default support regime
  - exact SuperAD-selected 16 normals, seed 0
  - full-pool-selected support임을 표시
- P16-random:
  - support-selection robustness protocol
  - seeds \(0,1,2\)
  - exposed 16 normals 안에서 fold별 memory/calibration 분리
- M16-fullpool:
  - 기타 full-pool-selected 16-shot diagnostic
- Pfull:
  - all-normal access
- 서로 다른 support provenance를 동일 few-shot resource로 혼합하지 않음

### 6.3 Core Ablations

- DVT:
  - no-DVT
  - support-only correction
  - query-only correction
  - symmetric correction
- Flow/reference:
  - identity feature distance
  - Flow latent distance only
  - NF NLL only
  - identity distance + weak density
  - latent distance + weak density
- RGB structure:
  - no guide
  - fixed RGB guided-r8
  - RGB Guide와 binary morphology contribution 분리
- final stack:
  - identity baseline
  - DVT-lite + identity distance
  - Flow-LatentBank without DVT-lite
  - DVT-lite + Flow-LatentBank
  - DVT-lite + Flow-LatentBank + RGB Guide
  - default final stack은 LOO standardization을 제외
- 모든 matched arm은 support, backbone, resolution, map processing, evaluator를 동일하게 유지

### 6.4 Metrics

- ranking/localization:
  - pAUROC@0.05
  - AP
  - oracle max-F1
  - component recall
- shift robustness/calibration:
  - held-out normal p99/p99.9
  - fixed normal-threshold F1
  - threshold variance across support seeds
- no-harm:
  - worst-class delta
  - anomaly-response retention
  - object-seed paired bootstrap confidence interval
- continuous raw map을 primary로 보고하고 morphology는 동일 조건 diagnostic으로 분리

### 6.5 Stage Gates

- Gate 0 — protocol parity:
  - resource, backbone, evaluator, map, threshold, morphology 일치
- Gate 1 — descriptor debiasing:
  - current DINOv2-L Basic no-DVT 대비 DVT-lite mean gain
  - paired bootstrap lower bound와 worst-class harm 확인
- Gate 2 — reference geometry:
  - identity/Flow distance를 동일 density weight에서 비교
  - Flow projection과 weak density contribution 분리
- Gate 3 — spatial restoration:
  - no-guide/guided-r8 continuous-map 비교
  - localization gain, response retention, worst-class harm 보고
  - morphology contribution 분리
- Gate 4 — public shadow:
  - complete object-seed matrix
  - mean gain과 no-harm 동시 확인
- Gate 5 — untouched final:
  - configuration과 comparator 사전 고정
  - P16 seeds와 Pfull을 별도 보고

## 7. Current Evidence

### 7.1 DVT-lite Diagnostic

- 기존 protocol: matched DINOv3-L diagnostic
- 결과: 비워둠
- 사유:
  - Basic default의 DINOv2-L backbone 및 exact SuperAD-selected support contract와 불일치
  - default-matched no-DVT/DVT-lite 재실행 전에는 본문 근거로 사용하지 않음

### 7.2 Flow-LatentBank Diagnostic

- 기존 protocol: DINOv3-L 및 historical DINOv3-H+ diagnostics
- 결과: 비워둠
- 사유:
  - Basic default와 backbone, support provenance 또는 density contract가 불일치
  - identity/Flow/density를 제외한 모든 조건을 Basic default로 고정한 재실행 필요

### 7.3 RGB Guide Diagnostic

- 기존 protocol: frozen DINOv3-H+ coarse map + fixed half-scale guided-r8
- 결과: 비워둠
- 사유:
  - Basic default와 backbone, support 및 LOO 조건이 동시에 불일치
  - default coarse map에서 RGB Guide만 on/off한 결과로 교체해야 함

### 7.4 DINOv2-L Basic Component Ablation and Default

- controlled protocol:
  - MVTec AD 2 public 8 objects
  - DINOv2 ViT-L/14, layers \([5,11,17,23]\)
  - resolution 672; `sheet_metal` 448
  - exact SuperAD-selected 16 supports, seed 0
  - static memory, density weight 0.25, fixed close--fill--erode morphology
- evaluated arms (`pAUROC@0.05 / F1`, %):
  - **Default, Flow + DVT + RGB: `78.194 / 44.107`**
  - no Flow: `77.949 / 42.380`
  - no DVT: `76.965 / 43.053`
  - no RGB Guide: `77.335 / 41.654`
- evidence boundary:
  - 단일 support seed와 oracle best-threshold 기반 public-set 결과

### 7.5 SuperAD Main Comparison

- protocol:
  - MVTec AD 2 full `test_public`, all 8 objects
  - SuperAD: provided test-public result
  - Ours Basic: DINOv2-L/14, exact SuperAD-selected 16-shot, seed 0, no LOO
  - Ours F1은 fixed close--fill--erode 이후의 F1으로 비교
- mean (`pAUROC@0.05 / F1`, %):
  - SuperAD: `76.71 / 39.42`
  - Ours Basic: `78.19 / 44.11`
  - delta: `+1.48 / +4.69`
- interpretation:
  - Ours는 mean pAUROC와 6/8 객체의 pAUROC가 높음
  - F1은 mean과 3/8 객체에서 높음
  - F1 차이는 Ours의 fixed morphology 영향을 포함하며, 특히 Fabric의 큰 향상을 별도 해석
  - SuperAD 수치는 provided result이며 same-run reproduction이 아님을 명시

### 7.6 Pending Evidence

- DINOv2-L Basic no-LOO 조건의 no-DVT/DVT-lite matched comparison
- DINOv2-L Basic no-LOO 조건의 identity/Flow matched comparison
- weak NLL contribution과 no-harm
- P16-random seeds \(0,1,2\) stability
- DINOv2-L Basic no-LOO coarse map의 no-guide/guided-r8 comparison
- RGB Guide seed별 worst-class harm
- SuperAD matched-support/backbone comparison
- untouched final evaluation

## 8. Discussion and Claim Boundary

### 8.1 Expected Strengths

- strong representation과 reliable anomaly evidence를 명시적으로 구분
- 여러 component를 few-shot과 distribution shift가 유발하는 세 가지 증거 문제에 대한 통합 대응으로 구성
- frozen FM 바깥에서 작동하는 lightweight extrinsic stack
- normal support만으로 fitting하고 test-time memory contamination 방지
- mean gain과 worst-class harm를 함께 다루는 검증 계약

### 8.2 Main Limitations

- Basic default에서 DVT-lite의 독립 효과가 아직 미확정
- Flow projection의 독립적 causal gain이 작거나 미입증
- Static Flow-LatentBank는 missing normal mode를 복원하지 못함
- NN scoring의 sparse coverage 한계가 남음
- RGB edge와 anomaly boundary가 불일치할 수 있음
- Basic default에서 RGB Guide의 독립 효과와 class-level no-harm이 아직 미확정
- sub-token defect observability와 non-rigid alignment를 해결하지 않음
- fixed-threshold stability와 untouched superiority 미확보

### 8.3 Causal Attribution Rules

- descriptor debiasing claim:
  - matched no-DVT/DVT comparison만 지지
- Flow geometry claim:
  - 동일 density weight의 identity/Flow distance 비교만 지지
- density claim:
  - distance-only와 distance+density 비교만 지지
- RGB structure claim:
  - 동일 coarse map의 no-guide/guided comparison만 지지
  - morphology contribution 분리 필요
- shift robustness claim:
  - held-out shifted normal, support-seed stability, worst-class harm 필요
- final superiority claim:
  - untouched matched evaluation만 지지

### 8.4 현재 사용 가능한 Claim

- FM-based few-shot AD의 high-score ambiguity under support–query shift
- shift-induced evidence degradation의 descriptor–reference–localization 분해
- DVT-inspired support-fitted positional correction
- support-only Static Flow-LatentBank coarse anchor
- training-free query-native RGB guided refinement
- DINOv2-L Basic no-LOO component ablation
- DINOv3-L component diagnostic 및 H+ RGB diagnostic

### 8.5 현재 사용 불가능한 Claim

- 기존 방법에는 distribution-shift 대응 모듈이 전혀 없다는 주장
- full DVT reproduction
- Flow projection 단독 우월성
- NF likelihood가 main anomaly score라는 주장
- Flow-LatentBank가 unseen normal mode를 복원한다는 주장
- RGB Guide가 appearance-shift invariance 또는 all-class no-harm를 제공한다는 주장
- state-of-the-art, broad generalization, deployable fixed-F1 claim

## 9. Figure and Table Plan

### Figures

- Figure 1: FM few-shot matching과 distribution shift에서의 high-score ambiguity
- Figure 2: evidence degradation under few-shot distribution shift
  - few-shot + distribution shift → representation bias / reference bias / spatial evidence degradation
- Figure 3: 전체 extrinsic robustness stack
- Figure 4: DVT-lite support position field와 symmetric correction
- Figure 5: Static Flow-LatentBank와 weak NLL scoring
- Figure 6: coarse anomaly map과 query-native RGB refinement
- Figure 7: component gain, seed stability, worst-class harm

### Tables

- Table 1: SuperAD 대 Ours Basic의 객체별 main comparison
- Table 2: DINOv2-L Basic component ablation과 default 결정
- 나머지 결과 표는 default-matched 실험이 완료될 때까지 구성하지 않음

## 10. Internal Writing Checklist

- Introduction 전체에서 `strong representation ≠ reliable anomaly evidence` 메시지 유지
- 모든 component를 few-shot과 distribution shift가 유발하는 세 가지 증거 문제에 대응하도록 연결
- “기존 방법에 extrinsic module이 없다”는 절대 표현 금지
- DVT, PNI, RegAD, FR-PatchCore 차이 명시
- Flow NLL 계열과 non-NLL latent-distance 계열 구분
- PNI와 CostFilter-AD 대비 RGB Guide의 training-free 차이 명시
- current DINOv2-L Basic matched DVT 근거와 H+ 전이 근거를 구분
- identity/Flow/density arm의 조건 일치
- RGB Guide와 morphology contribution 분리
- 모든 main table에 support provenance와 comparability 표시
- fixed-threshold rule, runtime, memory footprint 고정
- 외부 공개본에서 internal path와 run identifier 제거

## 11. Internal Evidence Map

- DVT-lite design:
  - `skill_graph/analysis/2026-07-07_dvt_position_denoising_design.md`
- current method and claim audit:
  - `skill_graph/analysis/2026-07-09_flowtte_current_method_results_issues.md`
- DVT and Flow experiment reports:
  - `skill_graph/experiments/`
- RGB Guide diagnostic:
  - `skill_graph/experiments/2026-07-13_flowtte_ad2_hplus_guided_r8_morph/report.md`
