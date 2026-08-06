# Basic 메커니즘 분석 및 실험 정리

- 기준일: 2026-07-27
- 대상: MVTec AD 2 16-shot DINOv2-L/14 Basic
- 문서 목적: Basic이 어떤 계산 경로로 이상 점수를 만들고, 각 구성요소가 실제로 어떤 역할을 하는지 지금까지의 분석 및 실험 근거를 중심으로 정리함
- 제외 범위: 새로운 방법론 설계, 후보 방법 개발, 하이퍼파라미터 탐색 과정, query-time adaptation, routing 및 후속 개선안

---

## 0. 핵심 결론

1. **Basic은 단순한 NF likelihood detector가 아님.**
   주 점수는 Flow가 만든 잠재공간에서 정적 normal support bank까지의 1-NN 거리이며, NF NLL은 작은 양의 보조항으로만 사용됨.

2. **Flow는 Basic의 가장 명확한 핵심 구조임.**
   동일한 다층 특징과 support를 사용한 원해상도 public-shadow 비교에서 Flow를 제거한 raw 1-NN보다 pAUROC, raw F1, AU-PRO가 모두 높았음. 다만 객체별 손실도 존재하므로 모든 객체에서 독립적으로 유리하다는 강한 주장은 불가함.

3. **DVT는 shift corrector라기보다 객체 의존적인 ranking transform에 가까움.**
   DVT는 평균 pAUROC와 일부 객체의 F1을 높였지만, AU-PRO를 낮추고 support-only shift 실험에서 shifted-normal tail과 fixed-threshold calibration을 악화시켰음. 따라서 “분포 변화를 제거한다”는 해석은 현재 근거와 맞지 않음.

4. **Density 항의 영향은 작음.**
   density weight 0.25는 pAUROC와 AU-PRO에 작고 양의 평균 효과를 보였으나, 사전 정의된 materiality 기준보다 작았고 F1에는 사실상 영향이 없었음. Basic의 주 성능을 설명하는 요소는 아님.

5. **다층 특징은 실제 public split에서 중요한 정보원이나, 고정 fusion의 보편적 synergy는 입증되지 않음.**
   원해상도 public-shadow에서는 `[5,11,17,23]` 다층 fusion이 고정 layer 17 단독보다 크게 우수했음. 반면 support-only 합성 실험에서는 객체별 최적 단일 layer envelope가 fusion을 앞섰음. 즉 한 개의 고정 layer로 대체할 수는 없지만, 모든 상황에서 단순 평균 fusion이 최적이라는 의미도 아님.

6. **RGB guide와 morphology는 core detector가 아니라 output formation임.**
   평균 segmentation 지표를 높이지만 `can`에서 ranking collapse가 확인되었고, morphology의 큰 F1 이득은 public best threshold를 사용한 oracle 진단임. 따라서 분포 변화 강건성이나 deployable 성능의 근거로 해석하면 안 됨.

7. **현재 Basic은 균일하게 강건한 detector가 아니라, 서로 다른 효과가 합쳐진 fragile macro compromise임.**
   Flow와 다층 특징이 기본적인 defect separation을 만들고, DVT가 일부 객체의 pixel ranking을 크게 끌어올리며, density가 작은 region-level offset을 더함. 이 효과가 객체와 지표마다 달라 평균 성능은 높아질 수 있지만, F1·AU-PRO·worst-object 성능이 동시에 개선되지는 않음.

---

## 1. 문서 범위와 증거 해석 원칙

### 1.1 여기서 말하는 Basic

본 문서의 Basic은 2026-07-24 Same16 실험에서 사용한 frozen raw detector를 의미함.

| 항목 | 고정 설정 |
|---|---|
| Dataset | MVTec AD 2, 8개 객체, `test_public/good,bad` |
| Support | 객체별 고정 정상 16장, seed 0 |
| Support manifest SHA-256 | `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3` |
| Backbone | frozen DINOv2 ViT-L/14 |
| Feature layers | `[5,11,17,23]` |
| Resolution | 기본 672, `sheet_metal` 448 |
| Support transform | identity only |
| Feature fusion | layer-wise L2 normalization 후 평균(`layer_norm_mean`) |
| DVT | support-fitted `position_mean`, alpha 1.0 |
| Flow | affine-coupling 2층, hidden multiplier 1, 3 epochs, LR `2e-4`, clamp 1.9 |
| Flow objective | tail weight 0.3, tail top ratio 0.05, log-det regularization `1e-3` |
| Memory | Flow latent로 변환한 static support bank |
| Main score | latent 1-NN distance |
| Density term | positive standardized NLL × 0.25 |
| LOO standardization | off |
| Context / conditioning | 없음 |
| Memory expansion | 없음(`expansion_budget=1.0`) |
| Core output | raw continuous anomaly map |

RGB guided refinement과 binary morphology는 별도의 output chain으로 다룸. 이는 최신 동일 조건 detector 비교에서 사용한 raw Basic과 과거 operational presentation setting을 혼동하지 않기 위함임.

### 1.2 증거 수준

Basic 관련 실험은 서로 다른 질문과 지표를 사용했으므로 다음 세 층을 구분해야 함.

| 증거 층 | 데이터와 지표 | 답할 수 있는 질문 | 주요 한계 |
|---|---|---|---|
| 원해상도 동일 조건 public-shadow | MVTec AD 2 전체 public split, pAUROC@0.05, raw oracle F1, AU-PRO@0.05 | 실제 Basic과 baseline/ablation의 상대적 detector 성능 | public split을 이미 관찰했으므로 최종 일반화·private 성능 주장 불가 |
| Support-only 합성 메커니즘 실험 | held-support C/S/D/SD, patch-grid pAUROC와 support-fit fixed F1 | shift, DVT, Flow, density, fusion이 내부 점수에 미치는 인과적 방향 | 실제 원해상도 benchmark 성능과 직접 비교 불가 |
| 과거 구성요소·출력 진단 | guided map, public best threshold, morphology 포함 | 구성요소 제거 시 평균 변화와 출력 단계의 경향 | evaluator·output contract가 최신 raw 비교와 다르고 oracle threshold 포함 |

따라서 이 문서에서는 서로 다른 층의 절대 수치를 한 표에서 순위화하지 않으며, 같은 방향의 결과가 반복될 때만 메커니즘 근거를 강화함.

### 1.3 평가 지표 해석

- **pAUROC@0.05**: pixel false-positive rate 0.05 구간의 standardized partial AUROC임. Pixel ordering에 민감함.
- **Raw F1**: 연속 anomaly map에서 test label을 사용해 찾은 best-threshold F1임. Detector의 threshold potential을 보여주지만 deployable threshold 성능은 아님.
- **AU-PRO@0.05**: 연결된 실제 결함 영역별 overlap을 false-positive rate 0.05까지 적분한 지표임. Pixel ranking뿐 아니라 region coverage와 공간적 일관성에 민감함.
- 세 지표가 서로 다른 방향으로 움직이면 단순한 “성능 향상/하락”보다 **ranking, threshold separation, region coherence가 서로 다르게 변했음**을 의미함.

---

## 2. Basic의 계산 메커니즘

### 2.1 다층 DINOv2 특징 추출과 fusion

정상 support와 query 영상에서 DINOv2 ViT-L/14의 네 intermediate layer `[5,11,17,23]` 패치 특징을 추출함. 각 layer의 패치 특징을 L2 정규화한 후 같은 위치의 특징을 평균함.

\[
h(x,p)=\frac{1}{4}\sum_{\ell\in\{5,11,17,23\}}
\frac{h_\ell(x,p)}{\|h_\ell(x,p)\|_2}
\]

이 과정은 얕은 층의 국소 정보와 깊은 층의 의미 정보를 하나의 고정 차원 descriptor로 결합함. 다만 layer 간 scale은 정규화되지만, 어떤 layer가 객체별 결함에 더 유효한지는 별도로 조절하지 않음.

### 2.2 Support-fitted position-mean DVT

Basic의 DVT는 학습 가능한 네트워크가 아니라 support 통계로 만든 위치별 bias field임. 정상 support의 각 grid 위치 평균에서 전체 위치 평균을 뺀 값을 artifact로 정의함.

\[
a(p)=\mathbb E_{x\in S}[h(x,p)]
-\mathbb E_{x\in S,p}[h(x,p)]
\]

support와 query 모두에 동일한 변환을 적용함.

\[
\tilde h(x,p)=h(x,p)-a(p)
\]

설계상 목적은 support에서 반복되는 위치 고정 성분을 줄이는 것임. 그러나 실제로 제거되는 성분이 positional artifact인지, 정렬된 객체의 유효한 정상 구조인지는 객체마다 다를 수 있음.

### 2.3 Support-conditioned Flow와 latent bank

DVT 이후의 support 특징을 채널별 평균과 표준편차로 표준화하고, affine-coupling normalizing flow를 정상 support만으로 학습함.

\[
z=f_\theta(\tilde h)
\]

Flow는 support 특징의 density를 학습하지만, Basic은 NLL 자체를 주 detector로 사용하지 않음. 학습 완료 후 모든 정상 support 패치를 latent로 변환하여 정적 메모리 \(\mathcal M_z\)를 구성함.

\[
\mathcal M_z=\{f_\theta(\tilde h(x,p))\mid x\in S\}
\]

query-time에는 Flow, DVT, support bank를 갱신하지 않음. 따라서 Basic은 이름과 달리 현재 frozen contract에서는 test-time memory expansion이나 query adaptation을 포함하지 않음.

### 2.4 Latent 1-NN 주 점수

query patch의 기본 이상 점수는 Flow latent에서 static support bank까지의 최근접 거리임.

\[
d(x,p)=\min_{m\in\mathcal M_z}
\|f_\theta(\tilde h(x,p))-m\|_2
\]

LOO standardization이 꺼져 있으므로 이 거리는 support leave-one-out 평균·표준편차로 재정규화되지 않음. 따라서 현재 Basic의 핵심은 **Flow로 재구성된 비교 geometry에서 실제 정상 support instance까지의 거리**임.

### 2.5 Density 보조항

Flow가 계산한 patch NLL을 support NLL 평균과 표준편차로 정규화한 뒤, 평균 이하 값은 0으로 절단함.

\[
r_{\mathrm{NLL}}(x,p)=
\max\left(
0,\frac{\operatorname{NLL}(x,p)-\mu_S}{\sigma_S}
\right)
\]

최종 patch score는 다음과 같음.

\[
s(x,p)=d(x,p)+0.25\,r_{\mathrm{NLL}}(x,p)
\]

따라서 density는 normal-like patch의 점수를 낮추는 양방향 calibration이 아니라, support 평균보다 NLL이 큰 patch에만 양의 penalty를 더하는 one-sided tail term임.

### 2.6 출력 형성

Core detector의 출력은 patch score를 원영상 크기로 복원한 연속 anomaly map임. 과거 operational output은 여기에 다음 단계를 추가했음.

1. query RGB를 guide로 사용하는 guided-r8 continuous-map refinement
2. thresholding
3. 17-pixel directional closing × 16 angles
4. hole filling
5. 3×3 erosion

이 단계는 feature 또는 normality 판단을 변경하지 않고 anomaly map의 경계와 binary shape를 바꿈. 따라서 detector mechanism과 presentation/segmentation mechanism을 분리해 해석해야 함.

---

## 3. 실험별 분석

### 3.1 2026-07-13 구성요소 제거 실험

### 질문

당시 Full Basic에서 Flow, LOO standardization, DVT, RGB guide를 하나씩 제거했을 때 평균 성능이 어떻게 변하는지 확인함.

### 관찰

| 제거 요소 | 당시 관찰된 주 변화 | 당시 해석 |
|---|---|---|
| Flow | pAUROC와 morphology F1이 모두 소폭 하락 | Flow가 identity support-bank보다 유효함 |
| LOO | pAUROC는 거의 유지되고 raw/morphology F1은 상승 | LOO가 unconditional gain이 아님 |
| DVT | pAUROC 하락 폭이 가장 큼 | DVT가 ranking에 크게 기여함 |
| RGB guide | raw F1과 morphology F1 하락 폭이 큼 | RGB guide가 공간 출력 형성에 기여함 |

이 결과에 따라 LOO standardization을 끈 `minus_loo`가 후속 Basic anchor가 되었음.

### 현재 관점에서의 의미

- 이 실험은 Basic의 주요 효과가 **DVT의 ranking 변화, Flow의 geometry 변화, RGB guide의 공간 정제**로 분리될 가능성을 처음 보여줌.
- 다만 guide와 public-best-threshold morphology를 포함한 과거 contract이므로, 최신 raw detector의 독립적 인과 효과로 그대로 사용할 수 없음.
- 이후 동일 조건 factorial과 support-only 실험이 DVT와 density 해석을 더 구체화했음.

### 3.2 2026-07-19 support-only C/S/D/SD 메커니즘 실험

### 질문

Basic이 distribution shift에서 실제로 무엇을 보정하는지 확인하기 위해 clean(C), shift(S), synthetic defect(D), shifted defect(SD)를 동일 support에서 생성하고 DVT, Flow, density, fusion을 요인별로 분리함.

### 주요 결과

1. **DVT**
   - SD patch pAUROC는 증가했으나 support-fit fixed F1은 크게 하락했음.
   - shifted-normal q.995 tail이 증가했고 nearest-support identity retention이 낮아졌음.
   - clean-to-shift cosine displacement도 DVT 이후 커졌음.
   - exposure와 white balance에서 악화가 두드러졌음.

2. **Flow**
   - DVT 없이 적용한 Flow는 SD pAUROC와 fixed F1을 모두 개선했고, 8개 객체에서 같은 방향을 보였음.
   - DVT와 함께 사용할 때는 ranking 개선은 남았지만 fixed F1 이득이 사라지고 shifted-normal tail이 증가했음.
   - Flow는 latent effective rank를 높였지만 raw-to-latent pair-distance rank와 latent scale을 변화시켰음.

3. **Density**
   - frozen weight에서 pAUROC, fixed F1, shifted-normal tail에 material한 기여가 없었음.

4. **Fusion**
   - fixed four-layer identity fusion은 객체별 best single-layer envelope보다 낮았음.
   - layer 17이 macro 기준 가장 강한 단일 layer였음.

### 메커니즘 해석

- DVT는 positional nuisance를 제거해 score calibration을 안정화한 것이 아니라, anomaly와 normal의 **상대 순위 일부를 재배열**한 것으로 해석하는 편이 타당함.
- Flow는 단순 NLL score가 아니라, support instance와 비교하는 geometry를 바꾸는 representation remapping으로 작동함.
- Flow와 DVT는 직렬로 연결되어 있으므로 각각의 단독 효과만으로 전체 Basic을 설명할 수 없음.
- 이 실험의 layer envelope는 객체별 최적 layer를 사후 선택한 상한선이므로, 고정 layer 17이 실제 public data에서 four-layer fusion을 대체할 수 있다는 뜻은 아님.

### 3.3 2026-07-23 Basic output-chain audit

### 질문

Core raw map(P0), RGB-guided continuous map(P1), public-best-threshold morphology(P2-oracle)를 분리하여 출력 단계의 역할을 확인함.

### 주요 결과

- P1은 macro pAUROC와 raw F1을 높였고 8개 중 7개 객체에서 양의 방향을 보였음.
- 그러나 `can`의 pAUROC가 object-collapse 기준을 넘게 하락했음.
- P2-oracle의 큰 F1 개선은 주로 `fabric`에 집중되었음.
- primary P2에 필요한 support-fit threshold는 과거 run에서 생성되지 않았음.

### 메커니즘 해석

RGB guide는 anomaly evidence 자체를 더 shift-invariant하게 만든다기보다, 일부 객체에서 coarse score를 query edge에 맞춰 재배치함. 이 과정은 경계가 anomaly와 정렬될 때 유리하지만, 정상 경계가 강하거나 score calibration이 약한 객체에서는 false-positive 구조를 강화할 수 있음.

Morphology는 threshold 이후의 binary shape만 바꾸므로 ranking을 개선할 수 없음. 따라서 P2-oracle은 presentation 효과이며 core detector 성능 또는 deployable threshold 성능으로 간주하면 안 됨.

### 3.4 2026-07-24 Same16 동일 조건 baseline 비교

### 질문

Basic, SuperAD-16, RN-FMLK를 동일한 DINOv2-L/14, 동일 16 support, 동일 원해상도 evaluator로 비교함.

### 결과

| Method | pAUROC@0.05 | Raw F1 | AU-PRO@0.05 |
|---|---:|---:|---:|
| Basic | 0.772635 | 0.355676 | 0.533408 |
| SuperAD-16 | 0.761857 | 0.368423 | 0.534924 |
| RN-FMLK | 0.745312 | 0.331950 | 0.425133 |

### 해석

- Basic은 세 방법 중 pAUROC가 가장 높음.
- SuperAD-16보다 raw F1과 AU-PRO는 낮음.
- 즉 Basic은 pixel ranking 일부는 개선하지만 threshold separation과 connected-region overlap을 동시에 개선하지 못함.
- RN-FMLK보다 macro 세 지표는 높지만 object-collapse와 방향 일관성 조건을 만족하지 못했음.
- 따라서 Basic은 retained descriptive detector이지만 SuperAD를 대체하는 claim-positive 방법은 아님.

### 3.5 2026-07-24 Flow 및 fusion 확인 실험

### 비교 arm

| Arm | 특징 | DVT | Flow | Density |
|---|---|---:|---:|---:|
| Basic | 4-layer fusion | on | on | 0.25 |
| A6 | 4-layer fusion | off | on | 0 |
| A7 | 4-layer fusion | off | identity | 0 |
| A6-L17 | layer 17 only | off | on | 0 |
| L17 | layer 17 only | off | identity | 0 |

### 결과

| Arm | pAUROC@0.05 | Raw F1 | AU-PRO@0.05 |
|---|---:|---:|---:|
| Basic | 0.772635 | 0.355676 | 0.533408 |
| A6 | 0.759284 | 0.350216 | 0.538267 |
| A7 | 0.718393 | 0.296003 | 0.499769 |
| A6-L17 | 0.722624 | 0.290316 | 0.455609 |
| L17 | 0.706543 | 0.276693 | 0.439253 |

### Flow 해석

A6는 A7보다 macro 기준 pAUROC `+0.040891`, raw F1 `+0.054212`, AU-PRO `+0.038498` 높았음. 이는 동일한 DVT-off, density-off, four-layer feature에서 Flow transform이 raw identity geometry보다 훨씬 유효함을 보여줌.

다만 Flow의 객체별 효과는 균일하지 않았음. pAUROC는 7/8, AU-PRO와 F1은 6/8 객체에서 양수였지만 `fruit_jelly` 등 일부 객체에서는 큰 손실이 있었고 bootstrap/no-collapse gate를 통과하지 못했음. 따라서 현재 근거는 다음 수준으로 제한됨.

> Flow가 Basic의 평균 성능을 구성하는 핵심 representation transform이라는 근거는 강하지만, 모든 객체에서 안정적인 독립 개선이라는 근거는 부족함.

### Fusion 해석

A6-L17은 A6보다 pAUROC `-0.036660`, raw F1 `-0.059900`, AU-PRO `-0.082659` 낮았음. 따라서 support-only 합성 실험에서 layer 17이 macro dominant였다는 사실만으로 실제 public detector를 layer 17 단독으로 축소할 수 없음.

두 결과는 모순이 아님.

- support-only 결과는 patch-grid 합성 defect에서 **객체별 best single-layer envelope**를 사용한 진단임.
- public 결과는 모든 객체에 **고정 layer 17 하나**를 적용한 원해상도 평가임.
- 결론적으로 일부 객체/결함에는 단일 layer가 강할 수 있으나, heterogeneous public population에서는 다층 특징이 필요한 객체가 많음.

### 3.6 2026-07-24 DVT × Density 2×2 factorial

### 질문

Basic과 A6의 차이가 DVT, density, 또는 두 요소의 interaction 중 무엇에서 발생하는지 분리함.

| Arm | DVT | Density |
|---|---:|---:|
| A6 | off | 0 |
| DVT-only | on | 0 |
| Density-only | off | 0.25 |
| Basic | on | 0.25 |

### 평균 효과

| 효과 | pAUROC@0.05 | Raw F1 | AU-PRO@0.05 |
|---|---:|---:|---:|
| DVT main effect | +0.012423 | +0.005457 | -0.008910 |
| Density main effect | +0.000928 | +0.000003 | +0.004050 |
| DVT×Density interaction | -0.000232 | -0.001381 | -0.001961 |

### DVT 해석

- Basic이 A6보다 높은 pAUROC와 F1을 보인 차이의 대부분은 density가 아니라 DVT에서 발생했음.
- 그러나 DVT의 pAUROC/F1 bootstrap interval은 0을 포함했고, AU-PRO는 평균적으로 하락했음.
- 객체별 방향 차이가 매우 큼.
  - `sheet_metal`: pAUROC와 F1이 크게 상승했지만 AU-PRO는 크게 하락함.
  - `fruit_jelly`, `wallplugs`: ranking/F1 개선이 확인됨.
  - `walnuts`: F1이 크게 하락함.
  - `can`, `vial`: AU-PRO가 하락함.
- 따라서 DVT는 모든 객체에 공통으로 적용되는 안정적인 bias correction이 아니라, 특정 객체에서 pixel ordering을 크게 바꾸는 heterogeneous transform임.

### Density 해석

- Density는 평균 pAUROC와 AU-PRO에 작고 양의 효과를 보임.
- F1 효과는 사실상 0임.
- DVT 유무와 관계없이 효과 크기가 작았고 interaction도 material하지 않았음.
- 따라서 density는 Basic의 성능을 결정하는 핵심 detector라기보다, NLL tail에 작은 positive offset을 주는 보조 calibration term으로 보는 것이 타당함.

---

## 4. 구성요소별 최종 메커니즘 판정

| 구성요소 | 관측된 역할 | 현재 판정 | 신뢰도 |
|---|---|---|---|
| DINOv2 다층 특징 | heterogeneous defect에 필요한 기본 표현 제공 | 고정 single layer보다 실데이터에서 유리하나 fusion synergy 자체는 미입증 | 중간 |
| Position-mean DVT | 일부 객체의 pixel ranking과 F1을 증가시킴 | `RANKING-ORIENTED / OBJECT-DEPENDENT`; shift corrector 아님 | 높음 |
| Normalizing Flow | support comparison geometry를 remap함 | Basic의 핵심 representation component | 높음 |
| Static latent bank | Flow latent에서 실제 support instance를 normal reference로 유지함 | NLL-only detector가 되지 않게 하는 anchor | 높음 |
| Latent 1-NN | 최종 score의 주 항 | Basic의 주 anomaly evidence | 높음 |
| Density penalty | high-NLL tail에 작은 양의 penalty 추가 | 작은 calibration 보조항; material detector contribution 없음 | 높음 |
| LOO standardization | support distance의 scale을 정규화 | mixed effect로 default에서 제거됨 | 중간 |
| RGB guide | query edge에 맞춰 continuous map을 재배치 | conditional presentation effect | 높음 |
| Morphology | threshold 이후 binary region을 정리 | oracle/output effect; core mechanism 아님 | 높음 |
| Memory expansion | 현재 Basic에서는 사용하지 않음 | Basic 메커니즘 범위 밖 | 높음 |

---

## 5. Basic 성능 패턴에 대한 통합 해석

### 5.1 왜 pAUROC는 높지만 F1과 AU-PRO는 따라오지 않는가

Basic의 주요 구성요소는 서로 다른 목표를 직접 최적화하지 않음.

- Flow는 support와 query의 상대적 거리 geometry를 바꿈.
- DVT는 위치별 feature field를 전체적으로 이동시킴.
- Density는 high-NLL patch에 작은 점수를 더함.
- RGB guide와 morphology는 공간적 경계와 binary shape를 바꿈.

이 중 DVT는 일부 객체에서 anomaly pixel을 normal pixel보다 위로 올려 pAUROC를 크게 높일 수 있음. 그러나 동시에 shifted normal score tail을 넓히거나 한 결함 영역 내부의 score coherence를 깨뜨리면 fixed-threshold F1과 AU-PRO는 낮아질 수 있음.

따라서 Basic의 `pAUROC 상승 + AU-PRO 하락` 패턴은 다음과 같이 해석할 수 있음.

> Pixel 단위 ordering은 좋아졌지만, 하나의 threshold로 분리되는 정도와 연결된 결함 영역 전체를 일관되게 덮는 능력은 개선되지 않았음.

이는 DVT factorial에서 pAUROC와 AU-PRO의 부호가 반대이고, support-only 실험에서 DVT가 ranking을 높이면서 shifted-normal tail과 fixed F1을 악화시킨 결과와 일치함.

### 5.2 왜 Basic은 객체별 편차가 큰가

Position-mean DVT는 모든 객체에 같은 식을 적용하지만, 제거하는 field의 의미는 객체 구조에 따라 다름.

- 위치 정렬이 강하고 정상 구조가 반복되는 객체에서는 위치 평균이 nuisance 또는 반복 bias와 가까울 수 있음.
- pose, 반사, texture 이동, foreground/background 경계 변화가 큰 객체에서는 위치 평균이 유효한 정상 구조를 포함할 수 있음.
- 이 경우 DVT는 정상 구조와 artifact를 구분하지 못하고 함께 제거함.

Flow도 support distribution에 맞춰 학습되므로, 16-shot support가 포괄하지 못한 normal mode나 객체별 feature topology에 따라 remapping 효과가 달라짐. 다층 fusion은 이러한 편차를 일부 완충하지만, 평균 방식이어서 객체별 최적 layer를 선택하지는 못함.

결과적으로 Basic의 macro 성능은 소수 객체의 큰 이득과 다른 객체의 손실이 상쇄된 값임. 평균만으로 “분포 변화에 강건함”을 주장하기 어려운 이유임.

### 5.3 Basic을 NF likelihood detector로 설명하면 안 되는 이유

Basic에서 NLL은 weight 0.25의 one-sided penalty이며, density factorial에서 영향이 매우 작았음. 반면 Flow를 유지하고 density를 제거한 A6는 identity Flow인 A7보다 모든 macro 지표에서 크게 높았음.

따라서 Flow의 주 역할은 다음과 같음.

1. support feature를 density model로 학습함
2. invertible transform을 통해 비교 geometry를 변경함
3. 변환된 실제 support instance를 latent bank에 보존함
4. query를 이 bank와 1-NN으로 비교함

즉 Basic은 **“NF로 density를 직접 탐지하는 방법”**보다 **“NF로 변환한 공간에서 support-reference distance를 계산하는 방법”**으로 설명하는 것이 정확함.

### 5.4 Basic은 test-time adaptation 방법인가

현재 frozen Basic은 test-time adaptation을 수행하지 않음.

- query로 Flow를 재학습하지 않음
- DVT field를 query로 갱신하지 않음
- latent memory를 확장하지 않음
- query condition routing을 사용하지 않음

따라서 현재 Basic의 결과를 online adaptation 효과로 해석하면 안 됨. Basic의 성능은 support-only fitting과 static inference에서 발생함.

---

## 6. 현재 근거로 가능한 주장과 불가능한 주장

### 6.1 가능한 주장

- Basic은 DINOv2 다층 patch feature를 support-fitted Flow latent로 변환하고, static support bank까지의 1-NN 거리를 주 점수로 사용함.
- Flow transform은 동일 four-layer feature의 raw 1-NN보다 평균적으로 더 유효한 비교 geometry를 제공함.
- DVT는 Basic의 pAUROC 상승에 크게 기여하지만, 그 효과는 객체와 지표에 따라 이질적임.
- Density는 주 detector가 아니라 작은 NLL-tail calibration 항임.
- RGB guidance와 morphology는 detector보다 output formation에 가까우며, 평균 이득과 객체별 손실이 공존함.
- Basic은 동일 조건 SuperAD보다 pAUROC는 높지만 raw F1과 AU-PRO는 낮음.

### 6.2 현재 근거로 불가능한 주장

- DVT가 distribution shift를 제거하거나 normal calibration을 안정화한다는 주장
- Basic의 모든 구성요소가 독립적으로 유의미한 성능 향상을 준다는 주장
- 다층 평균 fusion이 단일 layer보다 항상 우수하거나 layer synergy가 입증되었다는 주장
- Density/NLL이 Basic의 주 anomaly detector라는 주장
- RGB guide와 morphology가 모든 객체에 안전한 default라는 주장
- Public split 결과만으로 SuperAD 대비 우월성 또는 private/generalization 성능을 주장하는 것
- Raw oracle F1 또는 public-best-threshold morphology F1을 deployable threshold 성능으로 표현하는 것

---

## 7. Basic 메커니즘 이해를 위해 남은 확인 사항

새 방법 개발과 무관하게 Basic 자체를 더 정확히 이해하려면 다음 분석이 남아 있음.

1. **DVT가 제거하는 성분의 의미**
   - support position field의 크기·방향과 객체 정렬도, foreground boundary, 반사 및 texture 이동의 관계 확인 필요
   - DVT 전후 결함 내부 score coherence와 connected-component fragmentation 직접 측정 필요

2. **Flow 개선의 직접 원인**
   - raw와 latent 공간의 이웃 교체, local density, effective rank, class별 distance-tail 변화를 실제 public pixel/region 지표와 연결할 필요
   - Flow가 이득을 주는 객체와 손실을 주는 객체의 support coverage 차이 확인 필요

3. **다층 특징의 역할**
   - 고정 layer 17이 아닌 각 layer의 실제 public 기여와 상보성 확인 필요
   - 객체별 oracle layer 선택 없이 four-layer 평균이 어떤 결함 유형을 보존하는지 분석 필요

4. **Support 민감도**
   - 현재 주요 결론은 하나의 고정 16-shot manifest에 기반함
   - 동일 contract의 여러 support seed에서 DVT field, Flow geometry 및 세 지표의 안정성 확인 필요

5. **Threshold와 region behavior**
   - raw oracle F1과 별도로 support-calibrated fixed threshold 평가 필요
   - pAUROC 상승이 AU-PRO 하락으로 이어지는 객체에서 score tail과 region fragmentation을 직접 연결할 필요

6. **Untouched evaluation**
   - public split은 이미 분석과 의사결정에 사용되었으므로 최종 일반화 근거가 아님
   - 메커니즘 설명을 최종 claim으로 확장하려면 private 또는 설계에 사용하지 않은 evaluation이 필요함

---

## 8. 실험 흐름 요약

| 날짜 | 실험 | Basic 이해에 남긴 결론 |
|---|---|---|
| 2026-07-13 | Basic component ablation | DVT는 ranking, RGB guide는 output, Flow는 geometry에 기여하는 초기 분해를 제공함. LOO를 default에서 제거함 |
| 2026-07-19 | Support-only mechanism Stage M1 | DVT는 shift correction이 아니라 mixed/ranking effect, Flow는 DVT 없이 가장 안정적인 representation effect, density는 비물질적임 |
| 2026-07-23 | Basic output-chain audit | RGB guide/morphology는 조건부 presentation effect이며 unconditional default로 안전하지 않음 |
| 2026-07-24 | Same16 baseline closure | Basic은 SuperAD보다 pAUROC가 높지만 F1과 AU-PRO가 낮아 claim-positive 우월성은 없음 |
| 2026-07-24 | A6/A7/L17 confirmation | Flow와 four-layer feature가 Basic의 실제 public 성능에 중요한 구조임을 확인함 |
| 2026-07-24 | DVT×Density factorial | Basic-A6 차이의 대부분은 DVT의 이질적 ranking effect이며 density와 interaction은 작음을 확인함 |

---

## 9. 근거 문서

- Basic 구성요소 제거 실험:
  `skill_graph/experiments/2026-07-13_flowtte_basic_component_ablation/report.md`
- Basic 메커니즘 실험 설계:
  `skill_graph/analysis/2026-07-19_dinov2l_default_mechanism_experiment_plan.md`
- Support-only 메커니즘 결과:
  `skill_graph/experiments/2026-07-19_dinov2l_default_mechanism_stage_m1/report.md`
- Basic output-chain audit:
  `skill_graph/experiments/2026-07-23_basic_output_chain_custody_audit/report.md`
- Same16 baseline 비교:
  `skill_graph/experiments/2026-07-23_same16_baseline_metric_contract/report.md`
- Flow/fusion public-shadow 확인:
  `skill_graph/experiments/2026-07-23_a6_dominant_layer_public_shadow/report.md`
- DVT×Density factorial:
  `skill_graph/experiments/2026-07-24_same16_dvt_density_factorial/report.md`
- 최신 동일 조건 compact result:
  `results/remote_runs/dsba3/same16_baseline_metric_contract_20260723_v6/`
- DVT×Density compact result:
  `results/remote_runs/dsba3/same16_dvt_density_factorial_20260724_v1/`

---

## 10. 한 문장 정리

> Basic은 DINOv2 다층 특징을 support-fitted Flow latent로 재구성한 뒤 static normal bank와의 1-NN 거리로 이상을 판단하는 detector이며, 실제 성능은 Flow의 비교 geometry, DVT의 객체 의존적 ranking 변화, 작은 density calibration, 그리고 조건부 output refinement가 결합된 결과로 이해해야 함.
