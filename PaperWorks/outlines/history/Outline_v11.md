# Support-Calibrated Asymmetric Residual Fusion for Few-Shot Industrial Anomaly Detection

> 논문의 중심 질문: 소수 정상 reference에서 큰 거리의 원인이 모호할 때, 서로 다른 표현의 최근접 거리 중 어느 반응을 이상 점수에 얼마나 반영할 것인가?

## Index

- Abstract
- 1. Introduction

  - 1.1. 소수 정상 reference와 거리 해석의 모호성
  - 1.2. 다층·다해상도 표현의 상보성과 결합의 한계
  - 1.3. 다층·다해상도 거리 결합의 기술적 병목
  - 1.4. Support-Calibrated Asymmetric Residual Fusion과 기여
- 2. Method

  - 2.1. 문제 정의와 전체 추론 흐름
  - 2.2. 다층·다해상도 정상 support bank
  - 2.3. 정상 upper-tail 기반 component 보정
  - 2.4. Support-calibrated asymmetric residual fusion
  - 2.5. Query-conditional resolution routing
  - 2.6. 공간 투영과 최종 anomaly map
- 3. Experiments

  - 3.1. 비교 조건과 측정 지표
  - 3.2. 점수 구성 요소의 역할
  - 3.3. 기존 논문 방법 비교와 정상 support 수·획득 조건
  - 3.4. 노이즈 조건과 해상도 선택
  - 3.5. 정성 분석
  - 3.6. 추가 실험과 결과 확정 조건
- 4. Discussion & Conclusion

## Abstract

산업 이상 탐지에서는 새로운 제품을 소수의 정상 이미지로 검사해야 한다. 동결된 백본의 patch를 정상 reference와 매칭하는 방법은 재학습 비용을 줄이지만, 큰 최근접 거리는 결함뿐 아니라 support에 포함되지 않은 정상 변동이나 촬영 조건 변화에서도 발생한다. 여러 layer·해상도에서 거리를 얻으면 관측 범위는 넓어지지만, component마다 정상 거리의 크기가 다르고 shallow 반응을 항상 더하면 정상 수준의 반응도 anomaly map에 누적된다. 본 논문은 **Support-Calibrated Asymmetric Residual Fusion**을 제안한다. 정상 support의 leave-source-out 최근접 거리에서 component별 upper-tail 분위수로 상대적 단위를 정하고, deep score를 기본값으로 유지하면서 정규화된 shallow score가 정상 상위 분위수 기준을 넘는 초과량만 연속적으로 더한다. 동일한 support·백본·해상도·bank·후처리를 사용한 MVTec AD 2 16-shot 개발 비교에서 이 결합은 shallow 전체 가산보다 AUPRO@0.05가 **5.13pp**, MAD scale을 사용한 residual 결합보다 **1.66pp** 높았다. Deep-only 대비 차이는 **+0.47pp**로 작아 shallow의 독립적인 순이득은 확립되지 않았다. 동일 support와 DINOv2 ViT-L/14를 사용한 SuperAD와의 전체 파이프라인 비교에서는 AUPRO@0.05가 **4.31pp**, pAUROC@0.05가 **0.91pp** 높고 oracle F1은 **2.18pp** 낮았다. 강한 입력 노이즈에서 고해상도 경로를 제한하는 조건부 routing은 별도의 확장으로 평가한다.

## 1. Introduction

### 1.1. 소수 정상 reference와 거리 해석의 모호성

- **소수 정상 이미지에 기반한 검사**: 다품종 소량 생산과 빈번한 라인 전환에서는 제품마다 충분한 결함 데이터를 모으거나 모델을 다시 학습하기 어렵다. 동결된 시각 백본과 소수의 정상 이미지로 만든 reference bank를 사용하면, 별도 재학습 없이 query patch가 관측된 정상 patch에서 얼마나 떨어져 있는지 측정할 수 있다.
- **큰 거리의 원인을 구별하기 어려운 문제**: 소수의 정상 이미지는 부품 자세, 표면 반사, 조명, 센서 조건에 따른 정상 변화를 모두 담지 못한다. 따라서 결함이 없는 query patch도 reference bank에서 멀어질 수 있으며, 실제 결함 역시 큰 거리를 만들 수 있다. **큰 최근접 거리라는 관측만으로는 그 원인이 결함인지, reference에 포함되지 않은 정상 변화나 획득 조건 변화인지 구별할 수 없다.** 본 논문은 이처럼 큰 거리값이 이상 후보를 가리키지만 그 원인을 확정하지 못하는 문제를 **High-Score Ambiguity**라고 부른다.
- **다층 관측과 남는 문제**: 결함의 크기와 형태가 다양해 관측 범위를 넓히려는 다중 layer feature 활용은 선행 연구에서도 이루어져 왔다. 그러나 여러 표현에서 거리 반응을 더 얻는다고 해서 큰 점수를 곧바로 결함의 확실한 증거로 해석할 수 있는 것은 아니다. 따라서 관측 범위의 확장과 추가된 거리 반응의 해석은 함께 고려해야 한다.

### 1.2. 다층·다해상도 표현의 상보성과 결합의 한계

다층 feature와 다해상도 관측은 결함의 크기와 형태에 따라 서로 다른 단서를 제공할 수 있으며, 이러한 표현을 결합하는 접근은 기존 이상 탐지 연구에서도 사용되어 왔다. 그러나 다층·다해상도 결합만으로는 큰 최근접 거리의 발생 원인을 식별하기 어렵다. Few-shot 환경에서는 소수의 정상 reference가 정상 외관의 변동을 충분히 포함하지 못할 수 있고, support와 query의 획득 조건이 달라지면 정상 query patch에서도 큰 거리가 발생할 수 있다. 따라서 다층·다해상도 표현을 활용할 때에도 거리 반응의 점수 기여를 결정할 기준이 필요하다.

- **표현 깊이에 따른 상보적 관측**: ViT 표현 분석 연구([Raghu et al., NeurIPS 2021](https://arxiv.org/abs/2108.08810); [Amir et al., BMVC 2022](https://arxiv.org/abs/2112.05814))는 layer에 따라 전역·공간 정보와 semantic 정보의 구성이 달라질 수 있음을 보인다. 이러한 차이는 하나의 deep representation에서 약하게 드러나는 국소 변화를 shallow representation이 보완할 가능성을 제공한다. 이러한 보완 가능성이 모든 결함이나 획득 조건에서 성립하는 것은 아니므로, layer별 단서의 상보성은 결합 방식과 함께 평가해야 한다.
- **해상도에 따른 spatial sampling의 차이**: ViT의 patch size가 고정된 상태에서 입력 해상도를 높이면 더 조밀한 token grid가 형성되어 작은 결함과 얇은 경계에 대한 공간 표본화 밀도가 높아진다([Damm et al., WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html)). 낮은 입력 해상도만 사용할 경우 이러한 국소 변화를 충분히 표본화하지 못할 수 있다. 다해상도 관측은 동일한 query를 서로 다른 spatial sampling 수준에서 검사하여 결함 크기에 따른 검출 누락을 줄이기 위한 구성이다.
- **기존 방법의 상보적 정보 결합**: 학습 기반 방법은 다중 layer feature를 decoder나 판별기를 통해 결합한다([Deng &amp; Li, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Deng_Anomaly_Detection_via_Reverse_Distillation_from_One-Class_Embedding_CVPR_2022_paper.html); [Liu et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_SimpleNet_A_Simple_Network_for_Detecting_and_Localizing_Anomalies_CVPR_2023_paper.html)). 별도의 결합 함수를 학습하지 않는 대표적 사례로는 feature 연결과 거리 평균이 있다. PatchCore는 여러 intermediate-layer feature를 하나의 patch descriptor로 연결하고([Roth et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html)), SuperAD는 DINOv2의 여러 layer에서 계산한 최근접 거리 map을 평균한다([Zhang et al., 2025](https://arxiv.org/abs/2505.19750)).
- **고정된 거리 결합의 한계**: 특히 거리 map을 평균하거나 고정 비율로 가산하는 방식은 표현별 반응 차이를 반영하기 어렵다. 정상 최근접 거리의 규모가 큰 component는 동일 계수 평균에서도 최종 점수에 상대적으로 큰 영향을 미칠 수 있다. 또한 추가 표현의 반응을 항상 포함하면 결함 단서와 함께 정상 support에서 빈번히 나타나는 반응도 점수에 반영된다. 결과적으로 큰 거리의 원인이 불확실한 High-Score Ambiguity는 결합 점수에서도 해소되지 않는다. 따라서 상보적 표현을 활용하려면 **표현별 거리의 비교 기준과 추가 반응의 점수 기여량**을 정해야 한다.

### 1.3. 다층·다해상도 거리 결합의 기술적 병목

§1.2에서 제기한 거리의 비교 기준과 추가 반응의 기여량은 소수 정상 support와 획득 조건 변화가 있는 상황에서 더욱 중요해진다. 표현마다 정상 최근접 거리의 크기가 다르고, few-shot 환경에서는 그 비교 기준을 제한된 support에서 추정해야 한다. 또한 support와 다른 조건에서 촬영한 정상 query에서도 큰 거리가 발생할 수 있어, 추가 표현의 반응을 무조건 더하면 정상 변화가 anomaly score에 누적될 수 있다. 이에 따라 결합 단계에서는 정상 거리의 상대 단위를 정하는 문제와 추가 반응의 기여 범위를 정하는 문제를 구분한다.

1. **표현 간 upper-tail scale 불일치 (Scale Calibration Problem)**: 서로 다른 layer와 입력 해상도에서는 정상 support의 최근접 거리 크기와 상위 분위수 수준이 다르다. 이 차이를 고려하지 않으면 동일한 계수로 거리 map을 평균하더라도 정상 상위 반응이 큰 component가 최종 점수에 더 큰 영향을 미친다. Mean/std나 median/MAD는 중심 위치와 dispersion을 요약하지만, 저-FPR 평가에 중요한 정상 upper tail의 크기를 직접 정렬하지는 않는다. 따라서 첫 번째 과제는 **소수 정상 support에서 관찰된 상위 반응을 기준으로 component score의 상대 단위를 정하는 것**이다.
2. **정상 수준의 shallow 반응과 초과 evidence의 분리 (Contribution Control Problem)**: 단위를 맞춘 뒤에도 shallow score를 고정 가중치로 항상 더하면 support에서 흔히 관찰되는 정상 반응이 최종 점수에 누적된다. 획득 조건이 달라진 정상 query에서는 shallow 반응이 support 기준을 넘을 수도 있으므로, 큰 거리 자체를 결함의 확증으로 취급할 수 없다. 반대로 기준을 넘는 순간 shallow score 전체를 더하면 점수가 불연속적으로 증가한다. 따라서 두 번째 과제는 **support에서 흔한 반응의 상시 가산을 억제하면서 기준 초과 반응의 기여를 연속적으로 조절하는 것**이다.

**별도의 robustness extension—고주파 센서 노이즈에서의 고해상도 취약성**: 획득 조건 변화 중 강한 고주파 센서 노이즈는 고해상도 component의 반응을 과도하게 높일 수 있다. 이는 거리 점수의 결합과 별도로, query의 입력 조건에 따라 사용할 해상도를 정하는 문제다. 따라서 query noise가 support-relative 기준을 크게 넘는 경우에만 고해상도 경로를 제한하며, clean 입력에서는 전체 해상도 경로를 유지한다.

앞의 두 과제는 core 결합 방법으로, 센서 노이즈에 따른 해상도 선택은 별도의 확장으로 다룬다. Support에 포함되지 않은 정상 변화가 큰 거리를 만들 수 있다는 한계는 남으므로, 획득 조건별 효과는 §3.2에서 평가한다.

### 1.4. Support-Calibrated Asymmetric Residual Fusion (Ours)와 기여

본 논문은 frozen DINOv2 ViT-L/14를 기본 백본으로 사용하며, 추가 학습 없이 정상 support의 상호 매칭 통계로 다층·다해상도 거리를 결합하는 **Support-Calibrated Asymmetric Residual Fusion (Ours)**을 제안한다. 제안 방법은 거리의 발생 원인을 판별하지 않는다. 대신 component 간 상대 단위와 shallow evidence의 가산 범위를 support에서 정해, 모호한 반응이 최종 anomaly score에 기여하는 방식을 제한한다.

1. **정상 upper-tail 기반 component 단위 보정**: leave-source-out 정상 거리의 $\alpha$-분위수 $c_m$을 layer·해상도·bag별 단위 척도로 정의한다($z_m = d_m / \max(c_m, \varepsilon)$). 이 선형 재척도화는 각 component 내부 순위를 유지하면서 정상 상위 반응의 크기를 기준으로 component 간 score 단위를 맞춘다.
2. **Deep base와 shallow 초과량의 비대칭 잔차 결합**: deep score $z_d$를 기본값으로 유지하고, shallow score $z_5$는 정상 support의 $\beta$-분위수 기준 $\tau_5$를 초과한 양만 덧붙인다: $S = z_d + \operatorname{ReLU}(z_5 - \tau_5)$. Support 기준 이하의 shallow 반응은 가산하지 않고, 기준을 넘은 반응은 0에서 연속적으로 증가하는 초과량으로 반영한다.
3. **Query-side noise check를 통한 조건부 해상도 선택**: Immerkær 추정치로 얻은 query/support noise ratio가 임계값 $\kappa$를 초과할 때 896/1120 고해상도 component를 제외하고, 그렇지 않으면 전체 해상도 경로를 사용한다. 이 routing은 강한 입력 노이즈에서 고해상도 component의 영향을 제한하며, 조건별 효과와 발동 양상은 §3의 robustness 실험에서 평가한다.
4. **통제된 결합 비교와 전체 파이프라인 비교**: 동일 조건의 MVTec AD 2 16-shot 개발 비교에서 제안 결합의 AUPRO@0.05는 shallow 전체 가산보다 **+5.13pp**, MAD scale을 사용한 동일 residual 결합보다 **+1.66pp** 높았다. Deep-only 대비 이득은 **+0.47pp**에 그치고 pAUROC@0.05는 낮아, shallow 추가의 독립적인 순이득은 확립되지 않았다. 동일 백본·support를 사용한 SuperAD와의 전체 파이프라인 비교에서는 AUPRO@0.05 **+4.31pp**, pAUROC@0.05 **+0.91pp**, F1-oracle **−2.18pp**였다. 외부 비교에는 해상도·사용 layer·후처리 차이가 포함되며, MVTec AD 2에서 noise routing은 발동하지 않았다. Support seed 3개의 16-shot AUPRO@0.05는 평균 **59.63**, 표본분산 **0.63 pp²**였다. 결합 ablation은 §3.3, shot별 외부 방법 비교와 획득 조건별 결과는 §3.2에서 제시한다.
5. **해석 범위와 한계**: shallow 기준 초과량은 shallow 반응의 가산량을 정하지만, 그 반응의 원인을 실제 결함으로 확정하지 않는다. Support에 포함되지 않은 정상 변화도 같은 기준을 넘을 수 있다. MVTec AD Gaussian noise s3에서는 조건부 해상도 선택을 포함한 최종 방법의 AUPRO@0.05(69.01)도 SuperAD(71.10)와 PatchCore(74.03)보다 낮아 절대 성능 격차가 남는다.

## 2. Method

### 2.1. 문제 정의와 전체 추론 흐름

- **입력**: 정상 support 집합 $\mathcal S_N=\{x_1^s,\ldots,x_N^s\}$, query $x^q$
- **출력**: 원본 이미지 공간의 anomaly map $A(x^q)\in\mathbb R^{H\times W}$
- **백본**: frozen DINOv2 ViT-L/14의 patch representation
- **학습·갱신 없음**: 별도 파라미터 학습 없음, query-time bank 갱신 없음
- **Label 사용 범위**: fitting에는 정상 support만 사용. Query와 정답 mask는 어떤 기준값 계산에도 사용하지 않음
- **핵심 원리**
  - 큰 거리의 원인(결함, support에 없는 정상 변화, 획득 조건 변화)은 판별하지 않음
  - 대신 "처음 보는 정상 이미지가 support bank에 대해 보이는 거리 분포"를 support 안에서 추정하고, 이 분포를 기준으로 세 가지를 정함
    1. 서로 다른 component의 거리를 비교할 공통 단위 (§2.3)
    2. Shallow 반응이 최종 점수에 더해지기 시작하는 지점 (§2.4)
    3. Query마다 사용할 해상도 범위 (§2.5)

**Support fitting (support 집합당 1회)**

1. 다층·다해상도 feature bank 구성 (§2.2)
2. Support 이미지 간 최근접 거리로 component별 scale $c_{l,r,b}$와 shallow 가산 기준 $\tau_{5,r,b}$ 추정 (§2.3–2.4)
3. Support 영상의 기준 노이즈 수준 $\sigma_S$ 저장 (§2.5)

**Query inference (query마다)**

1. Support 대비 노이즈 비율 $\rho$로 사용할 해상도 집합 선택 (§2.5)
2. 선택된 각 layer·해상도·bag에서 최근접 거리 계산 (§2.2)
3. Scale로 component 단위를 맞춘 뒤 deep base에 shallow 기준 초과량 가산 (§2.3–2.4)
4. 모든 score map을 원본 공간에 투영·평균하고 CQ 적용 (§2.6)

**Support 산출물 요약.** 모두 support fitting에서 한 번 정하고 query 단계에서는 고정해 사용함.

| 산출물 | 정의 | 결정하는 것 | 대응 문제 | 절 |
| :-- | :-- | :-- | :-- | :-: |
| $B_{l,r,b}$ | 정상 patch feature bank | 최근접 거리의 기준 | — | §2.2 |
| $c_{l,r,b}$ | leave-source-out 정상 거리의 $\alpha$-분위수 | component 간 상대 단위 | Scale Calibration (§1.3) | §2.3 |
| $\tau_{5,r,b}$ | 정규화 shallow 정상 거리의 $\beta$-분위수 | shallow 가산 시작점 | Contribution Control (§1.3) | §2.4 |
| $\sigma_S$ | support 노이즈 추정치의 median | 해상도 routing 기준 | 고해상도 노이즈 취약성 (확장) | §2.5 |

**그림 1. RSG4의 support 보정과 query 추론 흐름.** 정상 support로 feature bank, component별 scale, shallow 가산 기준, 입력 노이즈 기준을 한 번 정한다. Query에서는 노이즈 비율에 따라 해상도를 선택하고, 선택된 component의 최근접 거리를 정규화한 뒤 deep base에 shallow 기준 초과량만 더한다. 각 score를 원본 크기로 투영·평균하고 CQ를 적용해 anomaly map을 얻는다. 점선 화살표는 query 단계에서 고정해 사용하는 support 산출물을 뜻한다.

### 2.2. 다층·다해상도 정상 support bank

**구성**

- **Layer**: $\mathcal L=\{5,11,23\}$ (5 = shallow, 11·23 = deep)
- **해상도**: 입력의 짧은 변 기준 $\mathcal R=\{448,672,896,1120\}$
- **전처리**: 종횡비를 유지해 bicubic(antialias)으로 resize한 뒤, 오른쪽과 아래쪽을 patch size 14의 배수로 자름
- **Feature**: 각 layer의 spatial token을 L2 정규화
- **Bank 표본화**
  - Bag마다 각 support 이미지에서 최대 256개 위치를 비복원 무작위 표본화
  - 표본 위치는 세 layer에 공통 적용
  - 해상도마다 서로 다른 고정 random bag 4개 구성 (bag별 고정 seed)
  - Bag bank $B_{l,r,b}$는 모든 support 이미지의 표본 token을 합친 집합 (16-shot 기준 layer당 최대 4,096개)
- **Component 정의**: $m=(l,r,b)$, 즉 layer $l$ × 해상도 $r$ × bag $b$의 조합. 기본 구성은 3 × 4 × 4 = 48개
- **Raw score**: query 위치 $p$의 feature $f_{l,r}(x^q,p)$와 bank $B_{l,r,b}$ 사이의 최근접 cosine distance

$$
d_{l,r,b}(x^q,p)=\min_{u\in B_{l,r,b}}
\left[1-f_{l,r}(x^q,p)^\top u\right].
$$

- **매칭 설정**
  - 기본값은 1-nearest neighbor
  - 같은 bag 안에서도 layer마다 자기 feature space에서 이웃을 독립적으로 선택
  - 이후 결합하는 것은 동일 위치의 layer별 거리이고, 공유된 이웃 identity가 아님

**작동 원리**

- **1-NN 거리의 의미**: support에 있는 정상 patch 중 하나와만 가까워도 낮은 점수를 받음. 정상 외관이 여러 mode로 나뉘어도 query가 그중 하나에 맞으면 정상으로 읽힘
- **Layer 축**: 같은 위치를 서로 다른 표현 깊이에서 측정함. Shallow layer는 국소 질감·경계 변화에, deep layer는 부품 구조·의미 수준 변화에 더 민감한 경향이 있음(§1.2)
- **해상도 축**: patch size가 14로 고정되어 있으므로 해상도가 높을수록 token 하나가 덮는 원본 영역이 좁아짐
  - 짧은 변 기준 token 수: 448 → 32, 672 → 48, 896 → 64, 1120 → 80
  - 따라서 1120 경로의 token 하나는 448 경로 token의 약 1/6 넓이를 표본화함. 작은 결함이 token 하나에서 차지하는 비중이 커짐
- **공통 표본 위치**: 세 layer의 bank가 support의 같은 위치를 담음. 따라서 동일 bag 안의 layer 간 거리 차이는 bank 내용의 차이가 아니라 표현의 차이에서 비롯됨
- **Bag 표본화**
  - 표본화로 bank 크기와 최근접 탐색 비용을 이미지당 256 token 수준으로 제한함
  - 한 bag에서 우연히 빠진 정상 패턴은 그 bag의 거리를 높일 수 있음. 서로 다른 4개 bag의 점수를 평균해(§2.6) 특정 표본 선택에 대한 의존을 줄임

### 2.3. 정상 upper-tail 기반 component 보정

**구성**

- **문제**: layer·해상도·bag마다 정상 최근접 거리의 범위가 다름. Raw score를 그대로 합치면 특정 component가 전체 map을 지배함
- **아이디어**: 각 component의 정상 upper tail을 상대 단위로 사용
- **정상 거리 표본 ($N\geq2$, leave-source-out)**
  - 해당 bag에 표본화된 support 이미지 $x_i^s$의 token을
  - 자기 이미지를 제외한 나머지 support source의 동일 component bank와 비교
  - 이렇게 얻은 거리 집합을 $D_{l,r,b}^{\mathrm{LSO}}$로 정의 (16-shot 기준 component당 최대 4,096개 거리)
- **Scale과 정규화 score** ($\alpha$: scale 분위수)

$$
c_{l,r,b}=Q_{\alpha}\!\left(D_{l,r,b}^{\mathrm{LSO}}\right),\qquad
z_{l,r,b}(x^q,p)=
\frac{d_{l,r,b}(x^q,p)}{\max(c_{l,r,b},\varepsilon)},
\quad \varepsilon=10^{-4}
$$

- **적용**: query는 support 전체로 만든 고정 bag bank와 매칭하고, 같은 bag에서 구한 $c_{l,r,b}$를 그대로 적용
- **1-shot 예외 (leave-token-out)**
  - 제외할 다른 support source가 없으므로, 같은 이미지 안에서 현재 token 자신만 제외해 $D_{l,r,b}$ 구성
  - 다중-shot의 이미지 간 정상 매칭과는 다른 근사. 같은 이미지 안의 인접 token은 서로 비슷하므로 이미지 간 거리보다 작게 추정될 수 있음
  - Query는 다중-shot과 마찬가지로 전체 1-shot bank와 매칭

**작동 원리**

- **Leave-source-out이 필요한 이유**
  - Bank token을 자기 자신이 포함된 bank와 비교하면 최근접 거리가 0이 되어 정상 거리의 규모를 과소추정함
  - Query는 bank에 들어 있지 않은 새 이미지임. 자기 이미지를 뺀 bank와의 거리가 "처음 보는 정상 이미지가 bank에 대해 보이는 거리"를 support 안에서 흉내 냄
- **Upper tail을 단위로 쓰는 이유**
  - 저-FPR 평가(AUPRO@0.05)에서 임계값 근처의 순위를 정하는 것은 정상 거리 분포의 중심이 아니라 상위 꼬리임
  - 중심과 산포(mean/std, median/MAD)가 같아도 꼬리 두께가 다르면, 같은 정규화 값에서 component별 오검출 비율이 달라짐
  - $\alpha$-분위수로 나누면 모든 component에서 "$z=1$ = 정상 LSO 거리의 $\alpha$-분위수"가 됨. 즉 support 기준으로 $z>1$인 정상 token의 비율이 모든 component에서 약 $1-\alpha$로 맞춰짐
- **보존되는 성질**
  - 양의 상수로 나누므로 component 내부의 score 순위와 초과량의 연속성이 유지됨
  - 거리 전체가 상수배로 커지거나 작아져도 $z$는 변하지 않음. Layer·해상도에 따른 거리 규모 차이가 제거됨
  - $c_{l,r,b}$는 anomaly 판정 임계값이 아니라 component 간 단위를 맞추는 척도임
- **Bag별 scale**: scale을 bag마다 따로 구함. 같은 bag의 bank와 scale이 짝을 이루므로, 표본화로 생긴 bag 간 거리 규모 차이도 함께 보정됨

### 2.4. Support-calibrated asymmetric residual fusion

**구성**

- **역할 분담**: deep이 기본 score, shallow는 추가 국소 evidence
- **Step 1. Deep base**: 해상도 $r$, bag $b$에서 layer 11·23의 정규화 score 평균

$$
z_{d,r,b}(p)=\frac{z_{11,r,b}(p)+z_{23,r,b}(p)}{2}.
$$

- **Step 2. Shallow 가산 시작점**: support에서 관찰된 정상 범위의 반응을 매 위치 더하지 않도록, 정규화된 leave-source-out shallow score의 $\beta$-분위수를 기준으로 둠

$$
\tau_{5,r,b}=Q_{\beta}\!\left(
\left\{d/\max(c_{5,r,b},\varepsilon):d\in D_{5,r,b}^{\mathrm{LSO}}\right\}
\right)
=\frac{Q_{\beta}\!\left(D_{5,r,b}^{\mathrm{LSO}}\right)}{\max(c_{5,r,b},\varepsilon)}
$$

- **Step 3. 초과량만 가산**

$$
e_{5,r,b}(p)=\left[z_{5,r,b}(p)-\tau_{5,r,b}\right]_+,
\qquad
S_{r,b}(p)=z_{d,r,b}(p)+e_{5,r,b}(p).
$$

**작동 원리**

- **기준값의 의미**
  - $\tau_5$는 shallow 정상 거리의 $\beta$-분위수와 $\alpha$-분위수의 비율임
  - $\beta>\alpha$이면 $\tau_5\geq1$이 됨. 즉 shallow score가 단위 척도($\alpha$-분위수)를 넘는 것만으로는 부족하고, 정상 분포의 더 바깥쪽 꼬리($\beta$-분위수)를 넘어야 가산이 시작됨
  - 두 분위수는 역할이 다름. $\alpha$는 component 간 단위, $\beta$는 shallow evidence의 가산 시작점을 정함
- **위치별 동작**
  - **정상 배경** ($z_5\leq\tau_5$): $S=z_d$. Support의 leave-source-out 기준으로 약 $\beta$ 비율의 정상 shallow 반응은 기여가 0이므로, 배경의 점수 수준이 deep base보다 높아지지 않음
  - **Shallow만 강하게 반응하는 위치** ($z_5>\tau_5$, $z_d$ 낮음): deep base가 놓친 국소 변화가 초과량만큼 점수를 올림
  - **두 표현이 모두 반응하는 위치**: deep base 위에 초과량이 더해져 점수가 더 높아짐
- **연속성과 단조성**
  - $S$는 $z_5$에 대해 연속이며 단조 비감소임. 기울기는 $\tau_5$ 이하에서 0, 이상에서 1
  - 기준을 넘는 순간 shallow score 전체를 더한다면 $\tau_5$ 지점에서 점수가 $\tau_5$만큼 뛰게 됨. Hinge는 이 불연속을 없앰
  - 초과량의 계수는 1로 고정함. 모든 score가 이미 $\alpha$-분위수 단위이므로, 초과량 1은 shallow 정상 $\alpha$-분위수 1배만큼의 추가 거리를 뜻함. 별도로 조정하는 가중치 없음
- **비대칭 구조**
  - Deep score는 항상 전부 포함함. 기본 판단 근거의 순위 정보를 그대로 보존하기 위함
  - Shallow score만 조건부로 더함. 정상 수준의 shallow 반응이 모든 위치에 누적되는 것을 막기 위함
- **해석 범위**
  - 큰 거리의 원인을 결함으로 판정하는 장치가 아님
  - 정상 token 중에서도 support 기준 약 $1-\beta$ 비율은 기준을 넘으며, support에 없는 정상 변화도 기준을 넘을 수 있음
  - 따라서 이 규칙은 기준을 넘은 shallow 반응이 최종 score에 기여하는 양을 제한하는 score composition임
  - 모든 $c_{l,r,b}$와 $\tau_{5,r,b}$는 support fitting에서 한 번 정하고 query마다 고정

### 2.5. Query-conditional resolution routing

**구성**

- **동기**: 고해상도 feature는 미세한 공간 변화를 보존하지만, 강한 고주파 입력 노이즈에도 민감함
- **원리**: resize 전 원본 영상의 노이즈 수준을 support 기준으로 상대화해 query별 해상도를 결정
- **Step 1. 노이즈 추정**: RGB를 grayscale로 변환한 뒤, 3×3 Immerkær 커널 $M=\left[\begin{smallmatrix}1&-2&1\\-2&4&-2\\1&-2&1\end{smallmatrix}\right]$로 $\sigma(x)$ 계산

$$
\sigma(x)=\sqrt{\frac{\pi}{2}}\,\frac{1}{6(W-2)(H-2)}\sum_{i,j}\left|(x*M)(i,j)\right|
$$

- **Step 2. Support 대비 비율**

$$
\sigma_S=\operatorname{median}_{x\in\mathcal S_N}\sigma(x),
\qquad
\rho(x^q)=\frac{\sigma(x^q)}{\sigma_S}
$$

  - 수치 예외: $\sigma_S=0$일 때 $\sigma(x^q)=0$이면 $\rho=1$, 그렇지 않으면 분모를 $10^{-12}$로 둠
- **Step 3. 해상도 선택** ($\kappa$: 노이즈 비율 임계값)

$$
\mathcal R(x^q)=
\begin{cases}
\{448,672\}, & \rho(x^q)>\kappa,\\
\{448,672,896,1120\}, & \text{otherwise}
\end{cases}
$$

- **적용 범위**
  - 선택된 모든 해상도에 같은 최근접 매칭, scale 보정, residual fusion 적용
  - Query마다 독립적으로 결정. Support bank와 calibration statistics는 변하지 않음
- **임계값 설정과 평가 분리**
  - $\kappa$는 구분된 clean·noise 입력 조건의 노이즈 비율 통계로 고정했고, 정답 mask나 검출 성능 지표는 사용하지 않음
  - 저해상도 전환 규칙은 MVTec AD Gaussian noise s3를 개발 조건으로 삼아 사전 등록
  - Shot·impulse·speckle noise와 다른 severity의 Gaussian noise는 규칙 결정에 쓰지 않은 held-out 조건으로 평가

**작동 원리**

- **노이즈 추정기가 영상 구조에 둔감한 이유**
  - $M$은 서로 다른 두 Laplacian 근사의 차이로, 밝기가 국소적으로 평면(선형)인 영역에서는 응답이 0임
  - 따라서 완만한 조명 변화나 부드러운 표면은 거의 반응하지 않고, 남는 응답은 주로 pixel 단위 고주파 성분에서 나옴
  - Resize 전 원본에서 계산하므로 해상도 선택 결과와 무관하게 query당 한 번만 계산함
- **Support 기준으로 상대화하는 이유**
  - 제품과 촬영 장비마다 정상 영상의 기본 노이즈 수준이 다름. 절대 임계값 대신 support median 대비 배수를 사용함
  - Median을 쓰므로 support 중 일부 이미지의 노이즈가 유난히 높거나 낮아도 기준이 크게 흔들리지 않음
- **고해상도 경로를 제외하는 이유**
  - 고해상도에서는 token 하나가 원본의 좁은 영역만 덮으므로 pixel 단위 노이즈가 feature에 덜 평균화된 채 반영됨
  - 저해상도로 resize할 때는 antialias 필터가 인접 pixel을 섞어 고주파 노이즈를 감쇠함
  - 따라서 노이즈가 support보다 크게 높은 query에서는 노이즈를 상대적으로 적게 보는 경로만 사용함
- **Clean query에서의 동작**: $\rho\leq\kappa$이면 전체 해상도 경로를 그대로 사용하므로 출력이 routing이 없는 구성과 같음
- **보정값 재사용**: 각 해상도의 bank, $c$, $\tau$는 해상도별로 이미 fit되어 있음. Routing은 평균할 해상도 집합만 바꾸고 새로운 fitting을 하지 않음

### 2.6. 공간 투영과 최종 anomaly map

**구성**

- **Step 1. 공간 투영**: 각 $(r,b)$의 token score $S_{r,b}$를 해당 token grid에 배열한 뒤, bilinear interpolation으로 원본 크기 $H\times W$에 투영
- **Step 2. 평균**: 네 bag을 평균하고, 이어서 query별로 선택된 해상도를 평균해 CQ 전 map $A_0$를 얻음

$$
A_0(x^q)=\frac{1}{|\mathcal R(x^q)|}
\sum_{r\in\mathcal R(x^q)}
\left[\frac{1}{4}\sum_{b=1}^{4}
\operatorname{Up}_{H,W}\!\left(S_{r,b}\right)\right].
$$

- **Step 3. Component quantile (CQ) 후처리**
  - $A_0$의 99번째 백분위 이상을 high-score mask로 지정
  - 팽창 없이 각 8-연결 성분 내부의 값을 그 성분의 최댓값($q=1.0$)으로 채움
  - Mask 밖의 값은 그대로 두어 최종 map $A$를 얻음
- **출력**
  - Pixel-level 평가: $A$
  - Image-level diagnostic score: $\max_p A(p)$
- **Label 사용**: CQ는 query map에만 적용하는 고정 출력 연산이며, support나 평가 label을 쓰지 않음

**작동 원리**

- **평균의 순서**: bilinear interpolation은 선형 연산이므로 bag 평균과 투영의 순서를 바꿔도 결과가 같음. 구현에서는 token grid에서 bag을 평균한 뒤 해상도마다 한 번 투영함
- **해상도 평균이 가능한 이유**: 모든 해상도의 score가 §2.3에서 각자의 정상 $\alpha$-분위수 단위로 맞춰져 있음. 따라서 추가 가중치 없이 단순 평균으로 결합함
- **CQ의 동작**
  - 임계값은 query map 자신의 99번째 백분위임. 절대 기준이 아니라 각 query의 상위 1% pixel이 대상임
  - 성분 내부의 값을 최댓값으로 채우므로 한 성분이 하나의 값을 갖게 됨. 임계값을 낮춰 갈 때 성분 전체가 한 번에 검출되고, 결함 영역 내부의 점수 요철로 인해 영역이 조각나는 일이 줄어듦
  - Mask 안의 값은 올라가거나 유지되고 mask 밖의 값은 변하지 않음. 따라서 mask 밖 pixel 사이의 순위, 그리고 mask 안과 밖의 순위 관계는 유지됨
  - 최댓값으로 채우므로 $\max_p A(p)=\max_p A_0(p)$임. Image-level score는 CQ의 영향을 받지 않음
- **효과의 검증**: CQ의 영역 단위 효과는 설계 근거이며, 같은 pre-CQ map에서 CQ를 켜고 끈 비교는 §3.6의 ablation으로 둠

## 3. Experiments

### 3.1. 비교 조건과 측정 지표

- **주 평가**: MVTec AD 2의 8개 클래스, 정상 support 16장, `test_public`의 동일 query·평가 계약을 사용한다. 이 split은 개발 평가에 이미 사용되었으므로 최종 미사용 평가로 부르지 않는다. Ours는 §2의 **RSG4** 전체 구성이다. 측정된 비교군은 RegAD, RD4AD, PatchCore, FastRecon, RN-FMLK, SuperAD이며, 추가 비교 후보인 AnomalyDINO, FiCo, EfficientAD, RD++, MSFlow, SimpleNet, DSR, RoBiS, SuperADD는 결과가 확보될 때까지 `—`로 둔다. 내부에서 시험한 별도 방법·변형은 외부 비교 표에 넣지 않는다.
- **비교 통제**: 가능한 경우 같은 support 이미지, query와 evaluator를 사용한다. 기존 방법은 원래의 scoring 설정을 유지하므로 layer·해상도·후처리까지 동일한 방법 단위 ablation으로 해석하지 않는다. 백본이 다른 결과는 별도 블록으로 표시한다. 방법 요소의 효과는 같은 백본·support·bank·후처리를 사용한 §3.3과 §3.4의 ablation에서만 논한다.
- **지표와 표기**: Pixel-level 주 지표는 클래스 macro AUPRO@0.05이고, pAUROC@0.05를 보조 지표로 둔다. F1-oracle은 query label로 임계값을 고르는 진단 지표이며 실제 운영 임계값 성능이 아니다. 모든 표의 수치는 백분율(%)이고 `—`는 해당 계약의 결과가 아직 없다는 뜻이다. 서로 다른 클래스 범위나 평가 조건의 수치는 같은 열에서 직접 순위화하지 않는다.

### 3.2. 기존 논문 방법 비교와 정상 support 수·획득 조건

**표 1. Main table: MVTec AD 2의 support shot별 기존 논문 방법 비교.** Full 8 classes, 동일 정상 support manifest·`test_public` query·streaming evaluator에서 측정한 AUPRO@0.05(%). 행은 support shot, 열은 방법이다. 괄호 안은 seed 0·1·2로 계산한 **표본분산 $s^2$ (pp²)**, 괄호 앞은 세 seed의 평균이다. 이 표기는 2/16-shot의 PatchCore·SuperAD·Ours에만 적용된다. 1/4/8-shot의 세 방법과 나머지 **측정된** 방법의 16-shot은 seed 0 단일 실행값이다. `—`는 해당 shot·평가 계약의 결과가 없음을 뜻하며, 논문에서 다른 정상 데이터 예산·백본·private split으로 보고한 수치를 대신하지 않는다. 현재 수치가 채워진 방법은 모두 DINOv2 ViT-L/14를 사용하지만 feature layer·해상도·후처리 설정은 방법마다 다르다. 추가 후보의 원 논문 설정은 각각 별도 확인하며, RoBiS·SuperADD의 발표된 full-normal 결과를 이 16-shot 표에 직접 대입하지 않는다.

| Shot | RegAD | RD4AD | RN-FMLK | FastRecon |    PatchCore |      SuperAD | AnomalyDINO | FiCo | EfficientAD | RD++ | MSFlow | SimpleNet | DSR | RoBiS | SuperADD |  **Ours (RSG4)** |
| ---: | ----: | ----: | ------: | --------: | -----------: | -----------: | ----------: | ---: | ----------: | ---: | -----: | --------: | --: | ----: | -------: | ---------------------: |
|    1 |    — |    — |      — |        — |        36.78 |        49.96 |          — |   — |          — |   — |     — |        — |  — |    — |       — |        **54.23** |
|    2 |    — |    — |      — |        — | 40.07 (5.90) | 50.56 (5.60) |          — |   — |          — |   — |     — |        — |  — |    — |       — | **55.62 (7.29)** |
|    4 |    — |    — |      — |        — |        44.41 |        53.68 |          — |   — |          — |   — |     — |        — |  — |    — |       — |        **58.36** |
|    8 |    — |    — |      — |        — |        45.73 |        54.73 |          — |   — |          — |   — |     — |        — |  — |    — |       — |        **58.40** |
|   16 | 15.93 | 39.17 |   44.39 |     47.05 | 49.16 (0.60) | 55.16 (0.21) |          — |   — |          — |   — |     — |        — |  — |    — |       — | **59.63 (0.63)** |

추가 후보의 역할은 [AnomalyDINO](https://arxiv.org/abs/2405.14529)(few-shot), [FiCo](https://arxiv.org/abs/2412.10115)(distribution shift), EfficientAD·RD++·MSFlow·SimpleNet·DSR(AD 2 데이터셋 논문에서 평가한 기존 방법), RoBiS·SuperADD(AD 2 중심 방법)로 구분한다. 이 분류는 후보 선정 기준이며 동일 조건의 성능 증거가 아니다. [AD 2 데이터셋 논문](https://arxiv.org/abs/2503.21622)의 full-normal/private 결과와 [RoBiS](https://arxiv.org/abs/2505.21152)·[SuperADD](https://arxiv.org/abs/2605.14808)의 원 논문 수치는 표 1의 빈칸에 전용하지 않는다.

2/16-shot의 세 방법은 동일 seed끼리 support·query를 맞췄다. RSG4−SuperAD의 seed별 AUPRO 차이 평균은 2-shot **+5.06pp**, 16-shot **+4.47pp**다. 나머지 네 방법의 16-shot 값은 seed 0 하나이므로, 이들과 세 seed 평균 사이의 차이를 paired 비교로 해석하지 않는다. Seed 0의 16-shot만 비교하면 Ours−SuperAD는 AUPRO **+4.31pp**(59.15 vs 54.84), pAUROC@0.05 **+0.91pp**(77.11 vs 76.19), F1-oracle **−2.18pp**(35.87 vs 38.04)이다. Shot별 pAUROC와 F1-oracle의 전체 수치는 아래 실행 기록에 보존한다. RSG4 noise guard는 MVTec AD 2의 각 실행에서 1,084개 query 중 0건 발동했다.

수치 출처: [기존 방법의 seed 0, 16-shot](../../../results/v7_evidence/E10_dsba3/comparison.md), [SuperAD·PatchCore의 seed 0 shot 비교](../../../results/paper_v10_stage1_20260928/comparison.md)와 [seed 1·2 반복](../../../results/paper_v10_stage2_20260928/comparison.md), [RSG4 seed 0](../../../results/rsg4_fewshot_seed0_20260926/comparison.md), [seed 1](../../../results/rsg4_fewshot_seed1_20260926/comparison.md), [seed 2의 2-shot](../../../results/rsg4_seed2_s02_20260926/comparison.md)·[16-shot](../../../results/rsg4_seed2_s16_20260926/comparison.md). 이 표는 전체 방법 비교이며 결합 규칙의 효과는 표 3에서 평가한다.

**표 1a. MVTec AD 2 공식 private 평가 지표 비교.** 각 셀은 `TESTpriv / TESTpriv,mix` 순서의 8개 클래스 macro 평균(%)이다. AU-PRO@0.05는 연속 anomaly map을 평가하고, ClassF1과 SegF1은 각각 이미지 판별과 결함 영역 분할의 임계값 적용 결과다. 이 표의 원 논문·공식 제출 결과는 방법별 정상 데이터 예산, 백본, 입력 크기와 후처리가 서로 다르며, 위의 동일 support·16-shot·`test_public` 표와 직접 비교하지 않는다.

| 방법                  |       AU-PRO@0.05 |           ClassF1 |             SegF1 |
| :-------------------- | ----------------: | ----------------: | ----------------: |
| PatchCore             |       28.8 / 26.0 |         8.0 / 9.9 |         3.7 / 3.4 |
| RD                    |       26.4 / 25.0 |       70.8 / 72.9 |       18.1 / 16.7 |
| RD++                  |       27.1 / 25.2 |       70.5 / 73.0 |       19.2 / 16.7 |
| EfficientAD-S         |       30.8 / 19.2 |       76.8 / 77.8 |        15.4 / 8.0 |
| MSFlow                |       24.3 / 11.9 |       51.1 / 72.0 |        21.8 / 9.0 |
| SimpleNet             |       21.1 / 13.7 |       73.3 / 77.1 |        17.7 / 8.7 |
| DSR                   |       20.3 / 17.4 |       78.0 / 78.0 |        10.9 / 9.5 |
| SuperAD               |     60.51 / 58.37 |     70.20 / 74.40 |     47.18 / 42.51 |
| RoBiS                 |     67.25 / 59.65 |     79.62 / 79.86 |     51.00 / 46.52 |
| SuperADD              |     61.36 / 59.07 |     68.03 / 73.25 |     57.42 / 54.35 |
| **Ours (RSG4)** | **— / —** | **— / —** | **— / —** |

PatchCore부터 DSR까지의 수치는 [MVTec AD 2 데이터셋 논문 Table 7·8·12](https://arxiv.org/abs/2503.21622)의 재평가 결과이며, 이 논문은 EfficientAD-S를 사용했다. SuperAD는 [논문 §3.1](https://arxiv.org/abs/2505.19750)과 그 AU-PRO에 대응하는 공식 제출(`VAND2025 SuperAD 4`), RoBiS는 [논문 Table 1·2](https://arxiv.org/abs/2505.21152), SuperADD는 [논문](https://arxiv.org/abs/2605.14808)에 연결된 공식 제출(`VAND2026_regular SuperADD`)을 따른다. 제출값은 [MVTec AD 2 공식 리더보드](https://benchmark.mvtec.com/)에서 확인했다. SuperAD 논문 Table 2의 SegF1 평균(47.8 / 43.2)은 §3.1 및 해당 공식 제출값(47.18 / 42.51)과 일치하지 않아, 이 표에는 동일 제출의 AU-PRO·ClassF1·SegF1 조합을 사용했다. Ours의 private 평가는 아직 없다.

**표 2. MVTec AD 2의 획득 조건별 AUPRO@0.05.** 16-shot, seed 0, 동일 support·query·evaluator. `regular`와 `shift_1`은 8개 클래스, `shift_2/3`은 각각 해당 조건이 존재하는 7개 클래스의 macro 값이다. `—`는 이 계약의 조건별 결과가 아직 없음을 뜻한다. Ours의 조건별 값은 [RSG4 전용 실행](../../../results/rsg4_fewshot_seed0_20260926/comparison.md)에 직접 저장된 stratum 지표를 집계했다.

| 방법                  |     Regular (8) |     Shift 1 (8) |     Shift 2 (7) |     Shift 3 (7) |
| :-------------------- | --------------: | --------------: | --------------: | --------------: |
| RegAD                 |           18.46 |           17.21 |           14.81 |           15.31 |
| RD4AD                 |           45.34 |           38.36 |           36.40 |           37.75 |
| PatchCore             |           49.95 |           48.94 |           47.91 |           47.03 |
| RN-FMLK               |           51.64 |           43.65 |           43.51 |           42.91 |
| FastRecon             |           53.75 |           49.42 |           49.08 |           47.69 |
| SuperAD (DINOv2)      |           56.82 |           55.84 |           54.02 |           54.66 |
| AnomalyDINO           |              — |              — |              — |              — |
| FiCo                  |              — |              — |              — |              — |
| EfficientAD           |              — |              — |              — |              — |
| RD++                  |              — |              — |              — |              — |
| MSFlow                |              — |              — |              — |              — |
| SimpleNet             |              — |              — |              — |              — |
| DSR                   |              — |              — |              — |              — |
| RoBiS                 |              — |              — |              — |              — |
| SuperADD              |              — |              — |              — |              — |
| **Ours (RSG4)** | **61.26** | **59.40** | **57.61** | **59.11** |

**그림 2. MVTec AD 2의 support shot별 획득 조건 AUPRO@0.05.** Seed 0, 동일 support·query·evaluator. 각 패널은 regular 또는 shift 조건의 클래스 macro AUPRO(%)를 shot에 따라 나타내며, 선은 기존 논문 방법 SuperAD·PatchCore와 Ours (RSG4)를 구분한다. Regular와 Shift 1은 8개 클래스, Shift 2·3은 해당 조건이 있는 7개 클래스이므로 서로 다른 범위의 패널 높이를 직접 빼서 변화량으로 해석하지 않는다.

![Support shot별 regular 및 shift AUPRO line plot](../../figures/fig_acquisition_shift_by_shot.png)

[벡터 PDF](../../figures/fig_acquisition_shift_by_shot.pdf) · [재현 스크립트](../../figures/plot_acquisition_shift_by_shot.py)

RSG4의 절대 AUPRO는 표시한 모든 shot·획득 조건에서 세 방법 중 가장 높다. 다만 동일 클래스의 regular를 기준으로 계산한 shift 감소량은 세 shift와 다섯 shot 모두에서 RSG4가 두 비교군보다 크다. 예를 들어 16-shot의 Shift 1/2/3 감소량은 RSG4 **−1.86/−4.59/−3.09pp**, SuperAD **−0.97/−2.68/−2.03pp**, PatchCore **−1.01/−1.34/−2.21pp**다. 따라서 그림은 shift 조건의 절대 검출 성능 우위를 보여주지만, 획득 조건 변화에 따른 성능 저하가 더 작다는 주장은 뒷받침하지 않는다. 수치 출처: [RSG4 shot별 전용 실행](../../../results/rsg4_fewshot_seed0_20260926/comparison.md)과 [SuperAD·PatchCore 실행](../../../results/paper_v10_stage1_20260928/comparison.md)의 클래스별 stratum 지표.

### 3.3. 점수 구성 요소의 역할

핵심 실험 질문은 **상대 scale 보정과 shallow residual이 각각 무엇을 바꾸는가**다. 같은 query와 support에서 다음 구성을 비교한다.

비교 scale인 median absolute deviation(MAD)은 정상 거리 표본 $D_m$의 중앙값 $a_m=\operatorname{median}(D_m)$에 대해 $\operatorname{median}_{d\in D_m}|d-a_m|$로 계산한다. MAD 구성은 query 거리에서 중앙값을 빼지 않고 $d_m/\max(1.4826\,\mathrm{MAD}_m,10^{-4})$로 나눈다. 따라서 아래 비교는 같은 거리 score의 **분모로 정상 $\alpha$-분위수와 중앙값 기반 MAD 중 무엇을 쓰는가**를 바꾼다.

**표 3. RSG4 score 결합 ablation.** MVTec AD 2, 16-shot, seed 0, full 8 classes, 동일 backbone·support·bank·CQ·evaluator. Ours 행은 clean query에서 guard가 발동하지 않아 RSG4와 점수가 같은 결합 구성이다.

| 결합 구성             | Component scale    | Shallow 기여         |   AUPRO@0.05 ↑ |  pAUROC@0.05 ↑ |    F1-oracle ↑ |
| :-------------------- | :----------------- | :------------------- | --------------: | --------------: | --------------: |
| MAD + ungated         | MAD                | shallow 전체         |           51.16 |           73.02 |           29.98 |
| $Q_\alpha$ + ungated   | 정상 $Q_\alpha$     | shallow 전체         |           54.02 |           74.04 |           32.90 |
| MAD + gated           | MAD                | $\tau_5$ 초과분       |           57.48 |           76.16 |           33.89 |
| $Q_\alpha$ + deep-only | 정상 $Q_\alpha$     | 없음                 |           58.67 | **77.35** | **37.11** |
| **Ours (RSG4)** | **정상 $Q_\alpha$** | **$\tau_5$ 초과분** | **59.15** |           77.11 |           35.87 |

수치 출처: [통제된 결합 비교](../../../results/comparisons/fusion_factorial_20260922/comparison.md). RSG4 전용 16-shot 실행에서 같은 값이 재현되었다.

동일한 16-shot MVTec AD 2 개발 비교에서 제안 fusion의 AUPRO@0.05는 **59.15**, $Q_\alpha$ scale과 ungated shallow는 **54.02**, MAD와 gated는 **57.48**, $Q_\alpha$ scale과 deep-only는 **58.67**이었다. $Q_\alpha$ scale을 고정한 **+5.13pp**는 shallow 전체 가산을 residual로 제한했을 때의 차이다. Gated 상태에서 $Q_\alpha$ scale과 MAD의 차이는 **+1.66pp**다. Deep-only 대비 **+0.47pp**는 등록된 개발 판정 기준 **0.86pp**보다 작고, pAUROC@0.05는 제안 fusion **77.11**보다 deep-only **77.35**가 높다. 따라서 이 비교는 shallow를 사용할 때 정상 수준의 반응을 제한하는 규칙의 역할을 뒷받침하며, shallow의 독립적인 순이득을 확립하지는 않는다.

이 표의 ungated 구성은 shallow의 **전체 score**를 deep base에 더하므로, 단순 3-layer 평균과는 다르다. 표 3만으로 기준 이하 제거와 기준 초과량의 연속 가산을 분리할 수 없다. 같은 $Q_\alpha$ scale에서 shallow의 고정 가중치와 hard gate를 바꾸는 ablation은 §3.6의 추가 실험으로 둔다.

같은 pre-CQ fusion map의 CQ 적용 전후도 별도 ablation으로 비교한다. 현재 표 3은 CQ를 공통으로 적용한 결합 규칙 비교다.

### 3.4. 노이즈 조건과 해상도 선택

**표 4. 기존 논문 방법과의 Gaussian noise 비교.** MVTec AD 15개 클래스, 정상 support 16장, seed 0, clean support와 동일 query corruption 계약. 주 지표는 macro AUPRO@0.05다. Ours의 clean 값은 guard가 발동하지 않아 동일 구성의 clean 결과와 일치한다. Gaussian noise s3는 routing 설계에 사용한 개발 조건이다.

| 방법                  |        Clean ↑ | Gaussian noise s3 ↑ |      변화량 (pp) |
| :-------------------- | --------------: | -------------------: | ---------------: |
| RegAD                 |           38.11 |                28.12 |           −9.99 |
| RD4AD                 |           76.42 |                40.41 |          −36.01 |
| RN-FMLK               |           74.53 |                46.56 |          −27.97 |
| FastRecon             | **85.74** |                57.70 |          −28.03 |
| SuperAD               |           84.49 |                71.10 |          −13.39 |
| PatchCore             |           83.97 |      **74.03** | **−9.94** |
| **Ours (RSG4)** |           85.36 |                69.01 |          −16.35 |

RSG4는 Gaussian s3에서 guard 비활성 구성보다 **+4.33pp** 높지만 SuperAD보다 **−2.10pp**, PatchCore보다 **−5.02pp** 낮다. 표 4는 서로 다른 방법의 절대 성능을 비교하고, 다음 표는 **RSG4 내부의 routing ablation**만 비교한다.

수치 출처: [기존 논문 방법의 동일 조건 비교](../../../results/comparisons/corruption_compare_20260922/comparison.md)와 [RSG4 routing 결과](../../../results/noise_remedy/remedy_summary.json).

**표 5. RSG4 noise guard ablation.** MVTec AD full 15, 16-shot, seed 0. `Guard off`는 같은 fusion·bank에서 네 해상도를 항상 사용하는 구성이다. 다른 노이즈 종류와 Gaussian s1/s5는 routing 선택에 사용하지 않은 held-out 조건이다.

| Query 조건                  | Guard off AUPRO | **Ours (RSG4)** AUPRO | 차이 (pp) |  Guard 발동 |
| :-------------------------- | --------------: | --------------------------: | --------: | ----------: |
| Gaussian s3 (개발)          |           64.67 |             **69.01** |     +4.33 | 1,725/1,725 |
| Shot noise s3 (held-out)    |           56.00 |             **65.10** |     +9.10 | 1,725/1,725 |
| Impulse noise s3 (held-out) |           54.02 |             **58.67** |     +4.65 | 1,725/1,725 |
| Speckle noise s3 (held-out) |           47.55 |             **60.56** |    +13.01 | 1,725/1,725 |
| Gaussian s1 (held-out)      |           75.06 |             **75.86** |     +0.80 | 1,725/1,725 |
| Gaussian s5 (held-out)      |           36.34 |             **48.49** |    +12.16 | 1,725/1,725 |

수치 출처: [고정된 noise remedy 집계](../../../results/noise_remedy/remedy_summary.json).

Guard의 동작은 $\rho$ 분포 및 조건별 발동률과 함께 보고한다. 기존 guardrail 집합의 clean·brightness/contrast s3·defocus blur s3·MVTec AD 2 public query에서는 0/7,984건이 발동했다. MVTec AD brightness s5에서는 pill 2/167장, toothbrush 1/42장의 오발동과 해당 클래스의 성능 변화(−0.54~+0.47pp)를 별도 진단으로 제시한다. 항상 저해상도를 쓰는 구성, guard off, CQ 제거의 비교는 표 4의 외부 방법 행으로 추가하지 않고 routing·후처리 ablation으로 보고한다.

### 3.5. 정성 분석

- 정상적인 반사·질감 변화, 작은 결함, 강한 입력 노이즈에서 RSG4의 raw layer distance, upper-tail 정규화 score, deep base, shallow residual, CQ 전후 map을 같은 좌표로 배치한다.
- 결합 ablation의 ungated shallow 및 deep-only map은 정상 반응의 누적과 shallow 추가의 득실을 보여주는 진단에만 사용한다. 사례 선택 기준과 실패 사례를 함께 기록한다.
- Noise routing 사례에는 $\sigma(x^q)$, $\sigma_S$, $\rho$, 선택된 해상도를 표기한다. 예시 map만으로 전체 조건의 개선을 주장하지 않고 표 4·5의 집계와 연결한다.

### 3.6. 실험 진행 상태와 결과 확정 조건

1. **획득 조건별 shot 축 확장 — 완료**: RSG4, SuperAD, PatchCore의 1/2/4/8/16-shot AUPRO를 main table인 표 1에, 획득 조건별 결과를 그림 2에 제시한다. 표 1의 2/16-shot은 세 seed의 평균과 표본분산이다. `shift_2/3` 변화량은 동일한 7개 클래스의 regular를 기준으로 집계했고, RSG4 전용 실행에 저장된 stratum 지표를 사용했다.
2. **외부 방법의 support seed 반복 — 완료**: SuperAD·PatchCore와 RSG4의 2/16-shot을 seed 0·1·2의 동일 support로 비교하고, 평균·표본분산을 표 1에 통합했다. 같은 seed에서의 방법 차이는 별도로 명시한다.
3. **결합 규칙의 식별 ablation**: $Q_\alpha$ scale을 고정하고 shallow 가중치, hard gate, $\tau_5$ 초과량을 분리한다. 표 3의 deep-only 대비 이득이 작은 점을 고려해 클래스·결함 크기별 paired 분석과 pAUROC/F1 trade-off를 함께 보고한다.
4. **출력·routing ablation**: 같은 pre-CQ map에 CQ를 켜고 끈 결과를 측정한다. Noise 조건에서는 항상 전체 해상도, 항상 저해상도, query별 routing을 같은 support와 query에서 비교하여 선택 규칙의 추가 이득과 clean 비용을 분리한다.
5. **데이터셋 및 평가 범위 확장**: MVTec AD·VisA에서 RSG4와 기존 논문 방법을 동일 few-shot 계약으로 비교한다. 현재 수치는 모두 개발 평가이므로, 논문의 최종 일반화 주장은 별도로 고정한 미사용 평가 또는 재현 가능한 외부 평가가 확보된 뒤 확정한다.
6. **정성 그림의 근거 확보**: §3.5의 동일 query에서 layer별 점수·CQ 전후 map·routing 정보를 저장하고, 성공 사례와 실패 사례의 선택 기준을 먼저 정한다. 현재의 수치 artifact만으로는 이 그림을 채울 수 없다.

## 4. Discussion & Conclusion

- 본 방법은 정상 support의 최근접 거리 분포에서 **component별 score 단위와 shallow 가산 시작점**을 정하고, deep base에 shallow 초과량을 더해 pixel score를 구성한다. 다층 정보를 얻는 단계보다 그 거리들을 정상 기준으로 읽고 비대칭적으로 결합하는 단계에 방법의 초점이 있다.
- 통제된 ablation에서는 component scale 보정과 shallow residual의 역할을 분리해 평가한다. Shallow 전체를 더하는 구성과 deep-only 구성을 함께 비교한 결과, 정상 수준의 shallow 반응을 제한하는 이득은 확인되지만 deep-only 대비 shallow 추가의 순이득은 확립되지 않았다. 세부 수치와 지표별 trade-off는 §3.3에서 보고한다.
- CQ는 fusion map의 공간 출력 구성이고, query-side noise check는 입력 노이즈에 따른 해상도 선택 확장이다. 내부 비교는 score-level fusion, 출력 후처리와 해상도 선택의 효과를 각각 드러내며, 외부 방법과의 비교는 전체 pipeline의 성능을 보여준다.
