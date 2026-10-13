# Support-Calibrated Residual Fusion for Few-Shot Industrial Anomaly Detection

> 논문의 중심 질문: 소수 정상 reference와 변하는 촬영 조건에서, 서로 다른 표현의 최근접 거리 중 어느 반응을 이상 점수에 얼마나 반영할 것인가?

## Index

- Abstract
- 1. Introduction

  - 1.1. 소수 정상 reference로 시작하는 산업 검사
  - 1.2. 서로 다른 크기의 결함에 필요한 다층·다중 해상도 단서
  - 1.3. 정상 변동에 대한 component 반응과 fusion의 모호성
  - 1.4. 선행 접근과 남는 score 결합 문제
  - 1.5. 제안 관점과 기여
- 2. Method

  - 2.1. 문제 정의와 처리 흐름
  - 2.2. 정상 support bank와 최근접 거리
  - 2.3. Tailscale: component별 q95 정규화
  - 2.4. RSG3: shallow q99 초과분의 비대칭 결합
  - 2.5. 공간 투영과 점수 집계
  - 2.6. Query-side noise check
- 3. Experiments

  - 3.1. 비교 조건과 측정 지표
  - 3.2. 점수 구성 요소의 역할
  - 3.3. 정상 support 수와 획득 조건: 외부 방법 비교와 내부 구성 비교
  - 3.4. 노이즈 조건과 해상도 선택
  - 3.5. 정성 분석
- 4. Discussion & Conclusion

## Abstract

산업 이상 탐지에서는 새로운 제품을 검사할 때 소수의 정상 이미지로 reference를 구성해야 한다. 결함은 작은 질감·경계 변화부터 넓은 구조 변화까지 다양한 크기로 나타나므로, 동결된 시각 백본의 여러 layer와 입력 해상도에서 얻은 국소 단서와 문맥 단서를 함께 활용한다. 그러나 정상 외관의 미관측 변동이나 촬영 조건 변화도 각 표현의 최근접 거리를 키운다. 큰 거리의 원인이 모호한 데 더해, layer·해상도·bank마다 정상 거리의 크기와 상위 반응이 달라 raw score를 그대로 합산하면 특정 component의 정상 반응이 결과를 지배할 수 있다. 본 논문은 정상 support를 **최근접 매칭의 reference이자 component별 점수 결합 기준**으로 사용하는 **Support-Calibrated Residual Fusion**을 제안한다. Support의 leave-source-out 거리 q95로 각 component를 정상 상위 반응에 대한 상대적 단위로 바꾸고, deep score를 기본 신호로 유지하면서 정규화된 shallow score가 support q99를 넘은 양만 연속적으로 더한다. 입력 노이즈에 따라 query별 해상도를 선택하는 확장도 포함한다. 동일한 16-shot MVTec AD 2 개발 비교에서 residual 방식은 shallow 전체를 더하는 방식보다 AUPRO@0.05가 **5.13pp** 높았고, q95 scaling은 median absolute deviation(MAD) scaling보다 **1.66pp** 높았다.

## 1. Introduction

### 1.1. 소수 정상 reference로 시작하는 산업 검사

- 다품종 소량 생산과 신규 제품 도입에서는 제품마다 많은 정상 이미지를 수집한 뒤 모델을 다시 학습하기 어렵다. 따라서 초기에 확보한 소수의 정상 이미지로 reference를 만들고, 이후 들어오는 query를 검사하는 few-shot 설정이 실용적이다.
- 동일 제품의 정상 외관도 표면 질감, 반사, 배치와 촬영 조건에 따라 달라진다. 소수의 정상 reference는 이런 변동의 일부만 담는다. 검사 도중 조명, 노출, 센서 노이즈 등이 바뀌면 query의 정상 반응도 reference에서 관찰한 반응과 달라질 수 있다.
- 동결된 시각 백본의 patch feature를 정상 bank에 저장하고 최근접 이웃 거리를 측정하면 별도의 결함 학습 없이 anomaly map을 만들 수 있다. 이때 결함이 나타나는 공간 크기와 외관 형태가 다양하므로, 어느 표현에서 얻은 거리를 활용할지가 먼저 중요해진다.

### 1.2. 서로 다른 크기의 결함에 필요한 다층·다중 해상도 단서

- 결함은 작은 스크래치·질감 변화처럼 제한된 영역에 나타나기도 하고, 형태·배치의 변화처럼 넓은 문맥에서 드러나기도 한다. 한 표현의 공간 해상도와 수용 문맥만 사용하면 이런 단서들을 같은 방식으로 관찰하게 된다.
- 얕은 layer와 높은 입력 해상도는 국소 질감과 경계 차이에 반응할 수 있다. 깊은 layer는 더 넓은 문맥을 반영한다. 여러 layer와 해상도에서 계산한 거리는 서로 다른 크기의 이상을 관찰할 **후보 단서**를 제공한다. 어느 한 layer에서 약한 반응이 다른 layer·해상도에서는 뚜렷하게 나타날 수 있으므로, 여러 반응을 anomaly map 구성에 활용한다.
- 이처럼 표현을 늘리는 선택은 관찰 가능한 단서의 범위를 넓힌다. 다음 문제는 각 표현의 큰 거리가 무엇을 뜻하는지와, 그 거리를 최종 map에 얼마만큼 반영할지다.

### 1.3. 정상 변동에 대한 component 반응과 fusion의 모호성

- Query patch와 정상 bank 사이의 큰 거리는 결함을 반영할 수 있다. 동시에 소수 reference에 포함되지 않은 정상 외관, 표면 반사와 촬영 조건 차이도 거리를 키운다. 정상 reference의 제한은 기준을 구성하는 단계에서, 획득 조건 변화는 query를 관찰하는 단계에서 이 해석에 영향을 준다.
- 국소 변화에 민감한 표현은 작은 결함뿐 아니라 정상적인 질감·조명·센서 변화에도 반응할 수 있다. Layer·해상도·bank마다 정상 거리의 분포와 상위 반응 크기도 다르다. 따라서 같은 raw distance라도 component마다 정상 기준에서 다른 상대적 크기를 뜻한다. 이를 그대로 합산하면 큰 정상 반응을 가진 component가 최종 map에서 과도한 비중을 차지할 수 있다.
- 폭넓은 질문은 **큰 거리가 결함에서 왔는가, 미관측 정상 변동에서 왔는가**다. 제안 방법이 직접 조절하는 문제는 fusion에서 **각 반응이 자기 정상 기준에 비해 얼마나 큰가, 그리고 그 초과 반응을 얼마나 더할 것인가**다. 낮은 오탐 영역에서 큰 정상 반응의 크기가 중요하므로 component별 q95로 상대적 scale을 맞춘다. 그다음 넓은 문맥의 deep score를 기본 신호로 두고, 국소 반응인 shallow score에서 정상 q99를 초과한 양만 보완 신호로 더한다.
- 높은 입력 해상도는 국소 차이와 함께 센서 노이즈의 고주파 변화에도 반응할 수 있다. 이에 각 query의 노이즈를 support 이미지의 노이즈 수준과 비교하여 사용할 해상도를 선택한다. 이 점검의 측정 대상은 이미지 노이즈다.

### 1.4. 선행 접근과 남는 score 결합 문제

정상 이미지를 이용한 patch 매칭은 산업 이상 탐지의 중요한 출발점이다. [PatchCore (Roth et al., CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html)는 중간 계층의 국소 feature를 집계해 정상 patch memory를 만들고 최근접 거리를 측정하며, image score도 reference neighbor에 따라 재가중한다. 정상 이미지를 통계적 기준으로 사용하는 또 다른 경로인 [PaDiM (Defard et al., 2020 preprint)](https://arxiv.org/abs/2011.08785)은 다층 feature를 연결한 뒤 공간 위치별 정상 Gaussian 분포를 추정하고 Mahalanobis 거리로 점수를 낸다. 두 방법은 각각 정상 patch의 매칭과 feature 공간의 위치별 분포 모델링을 중심에 둔다.

Few-shot 설정에서는 동결된 시각 백본과 간단한 거리 계산의 장점도 활용된다. [AnomalyDINO (Damm et al., WACV 2025)](https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html)는 동결된 DINOv2 feature의 cosine 최근접 거리를 사용하고, query patch 거리의 상위 1%를 평균해 image score를 구성한다. 이 상위값 집계는 **query에서 이미 산출된 patch score를 image score로 모으는 단계**에 적용된다.

여러 layer·해상도에서 최근접 거리를 얻을 때에는 각 거리 map을 **자기 정상 상위 반응에 대한 상대적 크기로 읽고, 어느 국소 반응을 pixel map에 더할지** 정해야 한다. 제안 방법은 정상 support를 매칭 reference로 쓰는 동시에 그 support의 최근접 거리 분포에서 component별 q95 단위와 shallow q99 기준을 얻는다. 이를 deep base와 shallow 초과량의 비대칭 결합에 적용한다. 따라서 선행 방법과의 구체적 차이는 정상 통계의 사용 여부나 tail의 사용 여부 자체보다 **통계의 대상이 component별 정상 매칭 거리이고, 적용 위치가 pixel-level score fusion**이라는 데 있다.

### 1.5. 제안 관점과 기여

정상 support는 두 역할을 맡는다. 첫째, query feature와 비교하는 최근접 매칭의 reference다. 둘째, 자기 이미지를 제외한 정상 매칭 거리로 component별 q95 scale과 shallow q99 기준을 제공한다. 제안 방법의 구체적 차별점은 **정상 최근접 거리의 상대 단위와 deep base–shallow 초과량의 pixel-level 결합 규칙을 함께 정의하는 것**이다.

1. **Component별 정상 매칭 거리의 기준화**: 동일한 정상 집합으로 feature bank를 만들고, leave-source-out 최근접 거리에서 layer·해상도·bank별 q95 단위와 shallow q99 기준을 계산한다.
2. **상대 단위와 비대칭 score composition**: q95로 각 component를 자기 정상 상위 반응에 대한 상대적 단위로 바꾼다. Deep score는 기본 신호로 두고, shallow score는 support q99를 넘은 양만 0에서 연속적으로 더한다.
3. **query 노이즈에 따른 해상도 선택 확장**: 이미지 노이즈 수준을 support 기준과 비교하고, 높은 노이즈가 측정된 query에는 저해상도 경로를 적용한다.
4. **support 수와 획득 조건에 따른 평가**: 내부 구성 차이와 외부 방법 비교를 구분해 shot과 acquisition shift에 따른 anomaly map의 성능을 분석한다.

## 2. Method

### 2.1. 문제 정의와 처리 흐름

- 정상 support를 $\mathcal S_N=\{x_1^s,\ldots,x_N^s\}$, query를 $x^q$라 한다. 출력은 원본 이미지 공간의 anomaly map $A(x^q)\in\mathbb R^{H\times W}$이다.
- 같은 $\mathcal S_N$이 두 기준을 제공한다. Feature bank $B_m$은 query와 최근접 매칭하는 reference이고, support 내부의 leave-source-out 거리 분포는 component별 q95 scale과 shallow q99 가산 기준을 정한다. 두 기준은 query 평가 동안 고정된다.
- 백본은 동결한다. Support 단계에서 feature bank, component별 q95, shallow q99 gate와 support 노이즈 기준을 계산한다. Query별로 이미지 노이즈를 점검하여 해상도를 선택한 뒤, 고정 bank와의 거리, q95 정규화, 비대칭 fusion, 공간 집계를 순서대로 수행한다.
- Fitting에는 정상 support만 사용한다. Query와 정답 mask는 기준값 계산에 들어가지 않는다. 추가 학습 파라미터나 query를 통한 bank 갱신은 없다.

### 2.2. 정상 support bank와 최근접 거리

- 사용 구성은 frozen DINOv2-L의 layer $\{5,11,23\}$과 입력 해상도 $\mathcal R=\{448,672,896,1120\}$이다. 각 feature는 공간 token으로 만들고 L2 정규화한다.
- 정상 support feature를 해상도별 네 개의 고정 random bag bank에 저장한다. 하나의 **component** $m=(l,r,b)$는 layer $l$, 해상도 $r$, bank $b$의 조합이다.
- Query 위치 $p$의 feature $f_{l,r}(x^q,p)$와 해당 bank $B_m$ 사이의 최근접 거리로 raw score를 정의한다. 사용 구성은 $k=1$이다.

  $$
  d_m(x^q,p)=\min_{u\in B_m}\operatorname{dist}\bigl(f_{l,r}(x^q,p),u\bigr).
  $$

  거리는 L2 정규화된 token 사이의 cosine distance $\operatorname{dist}(u,v)=1-u^\top v$다. 제안 방법의 내부 ablation은 같은 backbone, bank, 거리 함수와 전처리를 사용한다.

### 2.3. Tailscale: component별 q95 정규화

서로 다른 layer·해상도·bank는 정상 support에 대해서도 다른 거리 크기를 보인다. Component $m$의 정상 거리 q95를 단위로 사용하면 같은 query distance가 **그 component의 정상 상위 반응에 비해 얼마나 큰지** 표현할 수 있다. q95는 각 score의 절대 판정선이 아니라 fusion에서 component 간 크기를 맞추는 기준이다.

- $N\geq2$일 때 각 정상 support 이미지 $x_i^s$의 token을 **자기 이미지를 제외한** 동일 component의 bank와 비교한다. 이렇게 얻은 leave-source-out 정상 거리 집합을 $D_m^{\mathrm{LSO}}$라 한다. 자기 token을 최근접 이웃으로 사용해 정상 거리가 인위적으로 작아지는 것을 막는다.
- Component별 scale과 query의 정규화 점수는 다음과 같다.

  $$
  c_m=Q_{0.95}\!\left(D_m^{\mathrm{LSO}}\right),\qquad
  z_m(x^q,p)=\frac{d_m(x^q,p)}{\max(c_m,\varepsilon)}.
  $$
- $\varepsilon=10^{-4}$는 분모가 0에 가까운 경우의 수치 안정화 값이다. 같은 방식으로 support의 leave-source-out 정상 거리도 정규화하여 다음 gate 기준에 사용한다.
- q95 나눗셈은 한 component의 모든 query 거리에 같은 양의 상수를 적용한다. 따라서 그 component 내부의 순위는 그대로이며, 정규화된 점수가 기준을 넘더라도 포화시키거나 이진화하지 않는다. 원래의 raw 수치 자체를 보존하는 것은 아니지만, 기준을 조금 넘은 반응과 크게 넘은 반응의 **상대적 강도**를 남겨 다른 component와의 합산 및 다음 shallow residual에 연속적으로 반영한다.
- **1-shot 정의**: support 이미지가 한 장이면 leave-source-out bank가 비므로, 같은 이미지에서 현재 token을 제외한 나머지 token을 이웃으로 쓰는 leave-token-out 거리로 $D_m$을 구성한다. 이때는 이미지 간 정상 변화 대신 이미지 내 token 차이로 기준을 추정한다.

### 2.4. RSG3: shallow q99 초과분의 비대칭 결합

Deep layer의 넓은 문맥 반응은 모든 위치의 기본 score를 이룬다. Shallow layer의 국소 민감도는 작은 결함을 보완할 수 있지만 정상 질감·조명 변화도 포착한다. 따라서 shallow score 전체를 기본 score에 누적하지 않고, 정상 support에서 관찰한 상위 반응을 넘어서는 부분만 더한다. 이 접근의 차이는 shallow와 deep를 함께 쓰는 사실이 아니라 **정상 기준으로 component의 단위를 맞춘 뒤 서로 다른 기여 규칙을 적용하는 점**이다.

- 해상도 $r$와 bank $b$를 고정하고, 같은 위치 $p$의 정규화 score를 $z_{5,r,b}(p)$, $z_{11,r,b}(p)$, $z_{23,r,b}(p)$라 쓴다. 정상 support에서 계산한 정규화된 shallow response의 q99를 $\tau_{5,r,b}=Q_{0.99}(Z_{5,r,b}^{\mathrm{normal}})$로 저장한다.
- Deep base와 shallow residual, fusion score는 다음과 같다.

  $$
  z_{d,r,b}(p)=\frac{z_{11,r,b}(p)+z_{23,r,b}(p)}{2},\qquad
  e_{5,r,b}(p)=\operatorname{ReLU}\!\left(z_{5,r,b}(p)-\tau_{5,r,b}\right),
  $$

  $$
  S_{r,b}(p)=z_{d,r,b}(p)+e_{5,r,b}(p).
  $$
- $z_5\leq\tau_5$이면 $S=z_d$이고, $z_5>\tau_5$이면 초과분만 deep base에 가산된다. 따라서 residual은 $z_5-z_d$가 아니다. q99 이후의 큰 값을 제거하는 연산도 아니다.
- 기준 이하에서 shallow 기여를 0으로 두면 정상 수준의 shallow 반응을 매 위치에서 반복 가산하지 않는다. 기준 통과 직후 shallow score 전체를 넣는 hard gate는 점수를 갑자기 뛰게 하지만, $\operatorname{ReLU}(z_5-\tau_5)$는 0에서 시작해 연속적으로 증가한다. 그러므로 기준을 조금 넘은 반응과 크게 넘은 반응을 다른 크기로 더할 수 있다. 이 기준은 기여량을 정하며 초과 여부로 결함 원인을 확정하지 않는다.
- $c_m$과 $\tau_{5,r,b}$는 support에서 한 번 정하고 query마다 고정한다. 이 규칙은 정상 기준을 이용해 shallow의 사용량을 조절하며, 결함 종류를 별도로 분류하거나 위치별 학습 가중치를 만들지 않는다.

### 2.5. 공간 투영과 점수 집계

- 각 $(r,b)$의 token score map $S_{r,b}$를 bilinear interpolation으로 원본 이미지 공간에 투영한다. 해상도 $r$에서 네 bank의 map을 평균하고, 선택된 해상도 집합 $\mathcal R(x^q)$의 map을 평균한다.

  $$
  A_0(x^q)=\frac{1}{|\mathcal R(x^q)|}\sum_{r\in\mathcal R(x^q)}
  \left[\frac{1}{4}\sum_{b=1}^{4}\operatorname{Up}_{H,W}\!\left(S_{r,b}\right)\right].
  $$
- Component-quantile 후처리(CQ)는 $A_0$의 99번째 백분위 이상을 high-score mask로 삼고, mask의 8-연결 성분 내부 값을 해당 성분의 최댓값으로 채운다. 최종 map을 $A$라 한다. 이 후처리는 제안 방법의 내부 ablation 전체에 동일하게 적용한다.
- Pixel 평가에는 $A$를 사용하고, image-level diagnostic score는 $\max_p A(p)$로 얻는다. 방법의 중심 연산은 q95 정규화와 q99 shallow residual이며, CQ는 공통 출력 규칙으로 기술한다.

### 2.6. Query-side noise check

높은 입력 해상도는 작은 외관 차이와 함께 센서 노이즈의 고주파 변화에도 반응할 수 있다. 노이즈 수준을 query 입력에서 직접 측정해 사용할 해상도를 정하고, 선택된 경로 안에서는 같은 support-calibrated fusion을 적용한다.

- Support 이미지와 query 원본의 grayscale 영상에서 3×3 Laplacian 차분을 이용한 Immerkær 노이즈 추정치 $\sigma(x)$를 구한다. Support 기준은 $\sigma_S=\operatorname{median}_{x\in\mathcal S_N}\sigma(x)$이다.
- $\sigma_S>0$이면 $\rho(x^q)=\sigma(x^q)/\sigma_S$다. $\sigma_S=0$인 경우에도 비율을 유한하게 정의한다($\sigma(x^q)=0$이면 $\rho=1$, 그 외에는 $\rho=\sigma(x^q)/10^{-12}$). $\rho(x^q)>9$이면 $\mathcal R(x^q)=\{448,672\}$, 그 외에는 $\mathcal R(x^q)=\{448,672,896,1120\}$로 둔다.
- 선택된 해상도에서 §2.3–2.5의 q95 정규화, shallow residual과 bank 집계를 동일하게 적용한다. 각 query는 독립적으로 처리하고 support bank와 기준값은 갱신하지 않는다.

## 3. Experiments

### 3.1. 비교 조건과 측정 지표

- 주 비교에는 MVTec AD 2의 정상 support와 regular 및 획득 조건별 query를 사용한다. 추가 데이터셋과 synthetic corruption은 서로 다른 외관·노이즈 조건에서 구성의 작동을 살피는 데 사용한다.
- 내부 component ablation에서는 정상 support 이미지, frozen backbone, feature layer, 해상도 후보, bank, 후처리 및 evaluator를 맞추고 score 구성 요소만 달리한다. 외부 방법 비교에는 각 방법의 backbone과 점수 후처리 차이가 포함되므로 support 예산, query 범위와 지표를 명시한다.
- Pixel-level 주 지표는 AUPRO@0.05이며, pAUROC@0.05와 F1-oracle은 서로 다른 관점의 보조 지표로 함께 제시한다. Image-level 결과는 map의 최댓값에서 계산한다. 획득 조건별 절대 성능과 같은 조건 안에서의 구성 차이를 구분하여 읽는다.

### 3.2. 점수 구성 요소의 역할

핵심 실험 질문은 **상대 scale 보정과 shallow residual이 각각 무엇을 바꾸는가**다. 같은 query와 support에서 다음 구성을 비교한다.

비교 scale인 median absolute deviation(MAD)은 정상 거리 표본 $D_m$의 중앙값 $a_m=\operatorname{median}(D_m)$에 대해 $\operatorname{median}_{d\in D_m}|d-a_m|$로 계산한다. MAD 구성은 query 거리에서 중앙값을 빼지 않고 $d_m/\max(1.4826\,\mathrm{MAD}_m,10^{-4})$로 나눈다. 따라서 아래 비교는 같은 거리 score의 **분모로 정상 q95와 중앙값 기반 MAD 중 무엇을 쓰는가**를 바꾼다.

| 구성            | Component scale | Shallow 기여 | 해석할 비교                |
| :-------------- | :-------------- | :----------- | :------------------------- |
| 제안 fusion     | 정상 q95        | q99 초과분   | 기준 구성                  |
| Ungated shallow | 정상 q95        | shallow 전체 | gate의 역할                |
| MAD + gated     | MAD             | q99 초과분   | scale 선택의 역할          |
| MAD + ungated   | MAD             | shallow 전체 | 두 요소를 함께 바꾼 경우   |
| Deep-only       | 정상 q95        | 없음         | shallow 보완 신호의 순기여 |

동일한 16-shot MVTec AD 2 개발 비교에서 제안 fusion의 AUPRO@0.05는 **59.15**였고, q95와 ungated shallow는 **54.02**, MAD와 gated는 **57.48**, q95와 deep-only는 **58.67**이었다. 이 표의 초점은 제안 방식의 절대 점수 하나보다 **같은 조건에서 score 구성만 바꾼 차이**에 있다. Deep-only와의 차이처럼 지표별 방향이 달라지는 부분은 AUPRO, pAUROC를 나란히 보여준다.

### 3.3. 정상 support 수와 획득 조건: 외부 방법 비교와 내부 구성 비교

- **주 외부 비교의 shot × shift 축**: MVTec AD 2에서 가로축을 1/2/4/8/16-shot, 세로축을 `shift_1`의 절대 AUPRO@0.05로 놓고 RSG3, SuperAD DINOv2, SuperAD DINOv3, PatchCore의 네 곡선을 그린다. 각 shot에서 방법 간 동일한 정상 support를 쓰고, 예산별 support 집합은 중첩되게 구성한다. 같은 shot과 같은 8개 클래스의 `regular` 결과를 짝지은 보조 패널에는 각 방법의 `shift_1 − regular`를 표시한다. 이 두 패널은 shift 상태의 절대 검출력과 regular에서의 변화량을 함께 보여준다.
- **추가 획득 조건**: `shift_2`와 `shift_3`에도 같은 shot별 네 방법 곡선을 그리고, 각각의 shift와 **같은 7개 클래스**로 다시 집계한 `regular` 곡선 및 `shift − matched regular` 곡선을 함께 제시한다. 동일 support를 사용한 기존 16-shot 외부 방법 비교는 이 shot별 그림의 기준점으로 표에 싣는다. 외부 방법 비교에는 backbone과 점수 후처리 차이가 포함되므로 내부 component ablation의 구성 요소 효과와 구분한다. 다른 외부 방법은 보조 표에 둔다.
- **내부 구성의 교차 배치**: 가로축을 shot 수, 패널을 `regular`와 `shift_1`로 하여 gated fusion, ungated shallow, deep-only의 AUPRO@0.05를 그린다. 같은 support·backbone·bank·CQ에서 `제안 fusion − ungated shallow`와 `제안 fusion − deep-only`를 표시하여 score 규칙의 효과를 읽는다. 각 예산에서 q95와 q99를 다시 계산하며, 1-shot은 §2.3의 leave-token-out 정의를 표기한다.
- **다른 정상 외관**: MVTec AD와 VisA의 shot별 gated·ungated 비교를 같은 방식으로 배치한다. MVTec AD 2의 acquisition shift 그림과 분리해 데이터셋별 외관 차이에서 residual 규칙의 효과를 보여준다.

### 3.4. 노이즈 조건과 해상도 선택

- Clean, 일반적인 광도 변화, 센서 노이즈 변형에서 $\rho$의 분포와 저해상도 경로 선택 빈도를 먼저 보여준다. 이 값은 noise check가 실제로 측정하는 입력 특성을 설명한다.
- 같은 query에서 full-resolution fusion, 항상 저해상도 fusion, query-side check를 비교한다. Clean 성능, 노이즈 조건별 AUPRO@0.05, 고해상도 경로 제거 시의 변화를 함께 제시하여 해상도 선택의 성능 및 비용을 읽는다.
- 노이즈 이외의 획득 변화에서도 check의 선택 분포를 표시한다. 그러면 noise check의 역할을 센서 노이즈 대응으로 명확하게 해석할 수 있다.

### 3.5. 정성 분석

- 정상적인 반사 또는 질감 변화, 작은 결함, 노이즈가 강한 query를 골라 raw layer distance, q95 정규화 score, deep base, shallow residual, 최종 map을 같은 공간 좌표에서 비교한다.
- Shallow 전체를 더했을 때 높아지는 정상 위치와 q99 residual 후에 남는 위치를 함께 보여준다. 반대로 shallow residual이 추가되어 결함 경계가 드러나는 사례와 deep-only가 더 적합한 사례도 제시한다.
- Noise check 사례에는 $\sigma(x^q)$, $\sigma_S$, $\rho$와 선택된 해상도를 표시한다. 이를 통해 map의 변화가 어떤 입력 통계와 경로 선택에서 나왔는지 확인한다.

## 4. Discussion & Conclusion

- 본 방법은 정상 support를 query의 매칭 reference로 사용하는 동시에, 그 support의 정상 최근접 거리 분포를 **component별 fusion 기준**으로 사용한다. q95는 layer·해상도·bank의 상대적 크기를 맞추고, q99 shallow residual은 deep base에 더할 국소 반응의 양을 정한다. 이는 다층 특징을 사용한다는 사실보다 score를 구성하는 단계와 규칙의 차이다.
- Query-side noise check는 이 fusion 규칙 위에서 사용할 해상도를 정한다. 측정 대상은 이미지 노이즈이며, query별 경로 선택을 통해 높은 해상도의 노이즈 반응을 다룬다.
- 추가 학습 없이 동결된 feature에서 정상 support 기준의 비대칭 결합을 수행한다. Support 수와 획득 조건별 비교는 정상 기준의 추정과 shallow 기여가 anomaly map에 미치는 영향을 드러낸다. 정상 reference가 극히 적을 때의 기준 추정과 센서 노이즈 외 획득 변화에 맞춘 query 점검 방식은 후속 연구 질문이다.
