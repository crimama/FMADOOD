# SCARF: Support-Calibrated Excess Fusion for Few-Shot Anomaly Localization

> **문서 역할**: 한국어 연구 논리와 영어 문단 초안을 함께 제공하는 CVPR 본문 약 8쪽의 작성 뼈대. §1–6은 논문 구성, §7–10은 저자용 근거·실험·검토 기록이다. 저자용 기록을 본문에 그대로 옮기지 않는다.
>
> **방법 명칭**: SCARF의 기존 풀네임은 *Support-Calibrated Asymmetric Residual Fusion*으로 유지한다. 제목에서는 실제 연산인 *excess fusion*과 주 평가 대상인 *localization*을 드러낸다. 기존 실험의 RSG3는 fusion core, RSG4는 여기에 noise routing을 더한 구현을 가리킨다. 원본 run ID는 바꾸지 않는다.
>
> **핵심 연구 질문**: 소수 정상 support를 기준으로 다층·다해상도 reference 관측을 어떻게 결합할 것인가? Component의 거리 단위, 각 해상도 안의 shallow 기여, 입력 noise에 따른 해상도 선택을 구분한다. 중심 방법론 질문은 support 기준 초과량이 deep-only 및 적절한 고정 shallow 가중치보다 언제 유용한가이다.
>
> **현재 근거가 지지하는 답**: 초과량 결합은 unrestricted shallow addition 대비 저-FPR localization 손실을 상당 부분 완화한다. 그러나 deep-only 대비 이득은 작고 조건 의존적이며, target FPR 및 deep-only 성능 보존을 보장하지 않는다. 다해상도 구성의 이득은 최고 단일 해상도와의 용량 통제 비교가, 조건부 해상도 선택은 clean/noise의 고정 4·2해상도 및 routing 비교가 각각 뒷받침한다. 이들이 하나의 안전성 이론이나 synergy를 입증하지는 않는다. 강한 고정 가중치 대비 필요성은 추가 실험으로 판단한다.

## Abstract — 현재 근거로 작성 가능한 영어 초안

Few-shot industrial anomaly localization combines reference distances from multiple frozen feature layers and input resolutions. This requires choosing component scales, layer contributions within each resolution, and which resolutions to retain under input noise. We introduce Support-Calibrated Asymmetric Residual Fusion (SCARF): normal-support quantiles scale each component, and only shallow excess above a support-derived threshold supplements the deep score. A support-relative noise statistic conditionally selects the resolution set; selected maps are equally averaged. In a matched 16-shot development factorial on MVTec AD 2, excess improves AUPRO at a 5% FPR integration limit by 0.47 points over deep-only and 5.13 over full shallow addition, while partial pixel AUROC is 0.24 points below deep-only. A separate resolution control supports four scales over the highest scale alone at a matched nominal bank budget. Routing mitigates the evaluated Gaussian-noise failure but never activates in the reported clean AD 2 benchmark. Across three support draws, the complete pipeline obtains 59.63 ± 0.80 AUPRO, compared with 55.16 ± 0.46 for the evaluated SuperAD configuration. Negative transfer and fixed-threshold results limit the claims; advantage over a development-selected global weight remains unresolved.

**한국어 논리 요약**: 여러 layer와 resolution에서 얻는 거리는 정상 reference와의 서로 다른 관측이다. 세 선택을 구분한다: (1) layer×resolution×bag별 q95는 거리 단위, (2) 해상도 내 q99 excess는 shallow 기여량, (3) support-relative noise는 사용할 해상도 집합을 정한다. 선택된 해상도는 동일 가중 평균한다. Multiscale은 관측 구성이고 routing은 조건부 관측 선택이며, 둘의 근거를 섞지 않는다. 핵심 검증은 excess의 필요성에 둔다. Support 분위수는 결함 확률이 아니고, 전체 시스템 이득은 residual 또는 routing 단독 효과가 아니다.

**초록 마감 조건**: 위 수치는 완료된 개발 비교의 값이다. 새 weighted/hard-gate 결과가 확정되면 핵심 비교 문장을 갱신한다. 미완료 실험의 개선량, 유의성, 일반화 성공을 미리 기입하지 않는다. ±는 세 support draw의 표본 표준편차이며 confidence interval이 아니다.

## 1. Introduction — 약 1쪽

### P1. 문제 설정: 정상 reference가 적을 때의 localization

새 제품을 소수 정상 이미지로 검사하는 상황을 정의한다. 결함 mask로 학습하지 않고 frozen encoder와 reference matching을 사용하는 접근의 실용성을 설명한다. 목적을 이상 이미지 분류 전반보다 **낮은 허용 FPR 구간에서의 결함 영역 localization**에 둔다. 이미지 단위 탐지는 별도 보조 결과로 보고한다.

*English topic sentence*: “Few-shot industrial inspection requires localizing defects from a small normal reference set without retraining the visual encoder for each product.”

### P2. 거리와 결함은 동일하지 않다

큰 최근접 거리는 결함과 연관될 수 있지만 reference에 충분히 표현되지 않은 정상 외관에서도 발생할 수 있다. 이를 동기 수준에서 설명하고, 모든 실패가 support coverage 부족 때문에 발생한다는 단일 원인론은 피한다. “Coverage void”를 확립된 잠재 변수나 인과 메커니즘으로 사용하지 않는다. 기존 void 분석에서 해당 설명이 지지되지 않았다는 이력이 있다.

*English bridge*: “A large reference distance is evidence of a mismatch, rather than a calibrated probability of a defect.”

### P3. 연구 공백: 다층·다해상도 관측을 어떻게 결합할 것인가

Layer는 표현 깊이를, 입력 resolution은 관측의 공간적 샘플링을 바꾼다. 추가 관측이 항상 유용하다고 가정하지 않는다. **Component의 거리 단위, 해상도 안의 layer 기여, noise 조건에서 사용할 resolution 집합은 별개 선택**이다. 여러 해상도를 구성할 필요성은 최고 단일 해상도·bank 용량 대조로, noise 조건부 선택의 필요성은 고정 4·2해상도와 routing 대조로 확인한다. Shallow fusion은 이 관측 구성 안에서 검증한다. 단순히 양의 점수가 더해지거나 threshold가 커지는 것만으로 순위 기반 지표가 나빠지지는 않는다. 핵심은 추가 신호가 **정상 위치와 결함 위치의 상대 순서**를 어떻게 바꾸는가이다. 아래 구체적인 관측을 사용한다.

- AD2 16-shot seed0에서 q95 full shallow addition은 54.02 AUPRO, q95 equal mean은 58.21, deep-only는 58.67이다. “단순 평균은 필연적으로 붕괴한다”는 주장은 이 표에 맞지 않는다.
- Residual은 59.15로 full addition보다 5.13pp 높지만, equal mean 대비 0.93pp, deep-only 대비 0.47pp다. 효과 크기는 비교군에 따라 크게 달라진다.
- 따라서 논문의 질문은 “양의 반응을 더하면 위험하다”가 아니라, **support 기준 초과량이 global weighting보다 유용한 선택 규칙을 제공하는가**이다.

근거: [factorial](../../results/comparisons/fusion_factorial_20260922/comparison.md), [equal-mean 16-shot](../../results/fig1_pilot_20260929/remote_full8/comparison.md).

### P4. 제안: support를 기준으로 단위·기여·해상도 선택을 연결한다

정상 support끼리의 self-match를 피한 거리에서 layer×resolution×bag별 q95 scale과 shallow q99 threshold를 구한다. Deep 두 layer의 평균을 기본 점수로 사용하고 shallow score의 threshold 초과량만 더한다. 이 연산은 support 수준의 작은 shallow 반응을 0으로 만들고 큰 반응도 threshold만큼 차감한다. Deep score가 최종 순위 또는 성능의 안전 하한이 된다고 주장하지 않는다.

원본 query의 noise estimate가 support median보다 충분히 클 때는 두 저해상도를, 나머지는 네 해상도를 사용한다. 해상도별 excess score를 bag 평균·resize한 뒤 선택된 해상도를 동일 가중 평균한다. 세 선택은 정상 support를 기준으로 연결되지만 독립적인 검증이 필요하며, 통일된 confidence 추정이나 보증으로 서술하지 않는다.

*English topic sentence*: “SCARF uses normal-support references to set component scales, regulate shallow contributions within each resolution, and conditionally select resolutions under input noise.”

### P5. 실증 범위와 논문의 주장

외부 방법 비교는 완성된 시스템의 유용성을, 동일 pipeline의 fusion 비교는 핵심 연산의 필요성을 평가한다. 기존 결과의 가장 강한 근거는 full addition의 손실 완화이다. Deep-only 대비 순이득, 고정 가중치 대비 차별성, 다른 dataset의 1-shot 실패를 함께 제시한다. “모든 shot·dataset·지표에서 SOTA”를 기여문에 넣지 않는다.

**Contributions 초안**

1. **Support-referenced multilayer and multiresolution scoring.** We formulate excess fusion within each resolution, with componentwise support scaling, conditional noise-based resolution selection, and equal aggregation of the selected maps. Excess fusion remains the central methodological question.
2. **Controlled evidence on distinct design choices.** We separate normalization, layer contribution, multiscale construction, and conditional resolution selection. Completed controls establish configuration-specific benefits; weighted and repeated fusion controls remain necessary.
3. **An evaluation of benefits and limits.** We assess component and whole-pipeline results with distinct scopes, reporting support variation, noise conditions, negative transfer, and the distinction between low-FPR ranking and fixed-threshold behavior.

Contribution 2의 weighted/hard-gate 세부 결과는 §8의 완료 상태에 맞춰 마감한다. Frozen DINO, nearest-neighbor memory, quantile, ReLU, bagging 자체를 새로운 발명으로 주장하지 않는다.

## 2. Related Work — 약 0.5쪽

**P1. Frozen-feature reference matching.** PatchCore, AnomalyDINO, SuperAD를 가장 가까운 비교 맥락으로 설명한다. SCARF가 공유하는 frozen encoder·normal reference·nearest-neighbor scoring과 다른 fusion rule을 분리한다. Backbone과 해상도가 다른 baseline의 열세를 fusion의 직접 증거로 해석하지 않는다.

**P2. Few-shot scoring and alternatives.** 재구성·registration·subspace 계열은 서로 다른 prior를 사용한다. FastRecon, RegAD, SubspaceAD 등은 전체 시스템 비교를 넓히는 역할이며, 논문의 중심 질문은 동일 feature/bank를 사용하는 deep-only·equal mean·weighted sum·hard gate와의 비교로 해결한다. 독립 재구현과 공식 구현을 명시한다.

**P3. Calibration and selective fusion.** 분위수로 거리 단위를 맞추는 연산, threshold exceedance와 shrinkage, 다층 feature 결합의 선행 사용을 구분한다. 관련성이 가까운 CFF(Sensors 2026)는 few-shot fine-tuning 중 adapter로 cross-stage representation을 정렬하고, SPARC(arXiv:2608.18585)는 검증된 정상 deployment 이미지로 spatial subspace를 보정한다. Auxiliary-conditioned multimodal detection(arXiv:2608.13973)은 학습된 residual feature update와 spatial modulation을 사용한다. 본 초안은 primary-source record/abstract를 확인해 이 세 연구를 인용한다. 차별화 대상은 **고정 bank의 scalar nearest-neighbor distance에 support quantile과 threshold 차감을 적용하는 연산**으로 한정한다. 해당 방법들을 재현하거나 우월성을 입증한 것은 아니다. 본 방법은 empirical scale calibration이지 conformal coverage, posterior calibration, target-FPR calibration이 아니다. 제한된 선행연구 확인으로 “최초”를 선언하지 않는다.

참고문헌은 [새 BibTeX](../cvpr2027_v16/references.bib)와 [원문·공식 템플릿 검증 기록](../cvpr2027_v16/TEMPLATE_PROVENANCE.md)을 사용한다. 기존 WACV BibTeX는 인용 권위가 아니며 잘못된 SuperAD 항목을 새 원고에 재사용하지 않는다. SuperAD의 확인된 arXiv ID는 2505.19750이다. [v15 비교 설정](history/Outline_v15.md)과 [baseline 비교 정리](../notes/Main_Table_Baseline_Comparison.md)의 실제 재현 설정도 원논문과 구분한다. 포괄적 신규성 인증이나 최초성 주장을 하지 않는다.

**가장 가까운 수학적 비교**: $z_5>0$에서 excess는 $w(z_5;\tau)z_5$, $w=\max(1-\tau/z_5,0)$인 점수 의존 가중치이며 $w(0;\tau)=0$으로 둔다. 핵심 차별화 질문은 이 support-derived adaptive weight와 threshold 차감이 고정 가중치·hard gate보다 유용한지다. 이 대수적 표현만으로 신규성을 입증하지 않는다.

## 3. Method — 약 2쪽

### 3.1. Multilayer and multiresolution reference observations

클래스별 정상 support 집합을 $\mathcal S=\{x_i\}_{i=1}^{N}$, query를 $x$, patch 위치를 $p$라 한다. Frozen DINOv2 ViT-L/14의 **0-based layer indices** $\mathcal L=\{5,11,23\}$, 입력 short-edge $\mathcal R=\{448,672,896,1120\}$를 사용한다. $m=(\ell,r,b)$는 layer·resolution·bag의 조합이다. 이후 식의 $\ell$ 외 인덱스는 설명을 위해 생략할 수 있지만 실제 scale과 threshold는 **각 resolution·bag마다 별도**다.

이미지를 RGB로 읽고, 종횡비를 유지하며 bicubic/antialias resize 후 ImageNet mean/std로 정규화한다. Patch size 14로 나누어떨어지도록 **아래쪽·오른쪽 가장자리**를 잘라낸다. 예측 token map은 cropped grid에서 원래 이미지 크기로 bilinear resize한다. 패딩이나 crop의 정확한 역변환이라고 표현하지 않는다. 이 경계 처리는 재현성 항목과 작은 경계 결함의 한계에 기록한다.

각 resolution에서 support별로 최대 $K=256$개 token을 비복원 무작위 표본화한다. 하나의 bag에서 layer 간 token 위치는 공유하지만 각 layer의 최근접 이웃은 독립적으로 선택한다. $B=4$개의 bag은 서로 겹칠 수 있으며 support 이미지를 배타적으로 분할하지 않는다. Feature는 token별 $L_2$ 정규화한다. Query distance는 다음 cosine 1-NN distance다.

$$
d_{\ell rb}(x,p)=\min_{v\in\mathcal B_{\ell rb}}\left(1-\langle f_{\ell r}(x,p),v\rangle\right).
$$

16-shot에서 최대 vector 수는 $16\times256\times3\times4\times4=196{,}608$이다. 중복 위치를 포함한 component bank vector 수이며 고유 이미지·patch 수가 아니다. Bagging은 유한 표본 bank에 대한 여러 관측을 평균하는 장치다. 비편향 추정이나 자동적인 bias 상쇄를 보장하지 않는다.

구현 근거: [feature extraction·bank·scoring](../../src/fmad/methods/reference_matching.py), [resize·crop](../../src/fmad/backbones/wrapper.py), [기본 RSG4 config](../../configs/methods/tail_bag_RSG4.yaml).

### 3.2. Support-referenced component scaling

**$N>1$: leave-source-out (LSO).** 각 표본 support token을 같은 이미지에서 나온 모든 token을 제외한 bank에 매칭한다. 이렇게 얻은 component별 거리 집합을 $D_{\ell rb}$라 한다. Calibration token도 cap 적용 후 표본이므로, 모든 native support pixel의 독립적인 정상 표본이라고 서술하지 않는다.

**$N=1$: leave-token-out (LTO).** 하나의 표본 support bank 안에서 자기 token의 diagonal match만 제외한다. 같은 이미지의 다른 token은 남는다. LSO와 LTO는 서로 다른 추정 조건이며, LTO는 새로운 정상 이미지의 변화까지 모사하는 보증을 제공하지 않는다. 무작위 subsampling과 LTO는 동시에 적용된다.

$$
c_{\ell rb}=\max\{Q_{0.95}(D_{\ell rb}),10^{-4}\},\qquad
z_{\ell rb}(x,p)=\frac{d_{\ell rb}(x,p)}{c_{\ell rb}},
$$

$$
\tau_{5rb}=\frac{Q_{0.99}(D_{5rb})}{c_{5rb}}.
$$

q95는 하나의 경험적 분위수를 공통 기준으로 삼는다. Component 하나 안에서 양의 상수로 나누므로 그 component의 순서는 유지되지만, component들을 합친 최종 순서는 바뀔 수 있다. 서로 다른 tail shape 전체가 정렬되거나 $z$가 확률/FPR 단위가 되는 것은 아니다. Floor가 작동한 component의 q95가 정확히 1이라고도 주장하지 않는다.

**학습·개발 사용의 구분.** 특정 config의 $c$와 $\tau$는 정상 support만으로 fit된다. 그러나 $0.95,0.99$, layer·resolution 선택, CQ 등의 설계는 기존 benchmark 개발 과정을 거쳤다. “Training-free”는 gradient로 trainable parameter를 갱신하지 않는다는 뜻이며, hyperparameter-free 또는 개발 label에 전혀 노출되지 않았다는 뜻이 아니다.

### 3.3. Excess layer fusion within each resolution

$$
z_{d,rb}=\frac{z_{11rb}+z_{23rb}}{2},\qquad
e_{5rb}=[z_{5rb}-\tau_{5rb}]_+,\qquad
s_{rb}=z_{d,rb}+e_{5rb}.
$$

Shallow가 support q99 기준 아래이면 기여가 0이고, 그 위에서는 기준 초과량을 연속적으로 더한다. Threshold crossing은 $d_{5rb}>Q_{0.99}(D_{5rb})$와 동치다. q95 divisor를 바꾸면 normalized deep와 residual의 크기는 바뀌지만 동일 raw q99 경계의 gate 활성 조건 자체는 바뀌지 않는다.

**비교군을 수식으로 명확히 정의한다.**

| Arm | Bag별 결합식 | 답하는 질문 |
|---|---|---|
| Deep-only | $z_d$ | Shallow가 필요한가? |
| Equal mean3 | $(z_5+z_{11}+z_{23})/3$ | 공정한 동일 layer 가중치로 충분한가? |
| Full shallow addition | $z_d+z_5$ | Shallow에 각 deep layer의 2배 가중치를 주면 어떻게 되는가? |
| Fixed weight | $z_d+\lambda z_5$ | Global weight 감소로 효과를 설명할 수 있는가? |
| Hard gate | $z_d+z_5\mathbf1[z_5>\tau_5]$ | Gate 선택만으로 충분한가? |
| SCARF | $z_d+[z_5-\tau_5]_+$ | 선택과 threshold 차감을 함께 하는 초과량 연산의 효과는? |

Equal mean3는 실수 연산에서 $(2/3)(z_d+0.5z_5)$다. 이 공통 양의 배율은 이상적인 순위 기반 지표에 영향을 주지 않는다. 다만 저장·평가 과정의 float16 양자화에서 tie가 달라질 수 있으므로 기존 mean3를 **정확히 동일한 수치 실행의 $\lambda=0.5$**로 단정하지 않는다. 새 protocol은 정확한 $\lambda=0.5$ 실행을 이미 채택했으며, 결정이 아니라 결과가 대기 중이다.

기존 [FusionFactorial](../../src/fmad/methods/fusion_factorial.py)은 mean3, all, off, hard, gated와 global gamma를 지원한다. 계획된 비교를 위해 새로운 fusion 구현을 발명할 필요는 없다.

**성립하는 제한적 성질과 성립하지 않는 보증.** 비음수 cosine distance와 동일 $z_d,z_5,\tau_5\ge0$를 전제로, CQ 이전에는

$$
z_d\le z_d+[z_5-\tau_5]_+\le z_d+z_5.
$$

이는 shallow **점수 기여량**의 범위다. 정상·결함 순위, AUPRO, recall 또는 FPR의 범위가 아니다. 예를 들어 deep score가 정상 0.8·결함 1.0이고 residual이 정상 0.5·결함 0이면 최종 순위가 역전된다. 따라서 “baseline preservation”, “lossless inclusion”, “safe expansion guarantee”는 사용하지 않는다. 이 부등식은 MAD와 q95처럼 $z_d$ 자체가 다른 arm 간 관계를 설명하지 않으며, 비선형 CQ 이후의 arm 간 pointwise 관계로 확장하지 않는다.

### 3.4. Noise-conditioned resolution selection

**관측 구성과 관측 선택을 구분한다.** 네 해상도는 후보 관측을 구성하고, routing은 query마다 사용할 집합을 선택한다. q95는 해상도 신뢰도 weight가 아니다. 원본 RGB 영상의 grayscale 고주파 noise estimate를 support median과 비교한다. Ratio가 $\kappa=9$를 넘는 query에 한해 $\mathcal R(x)=\{448,672\}$로 바꾸며, 나머지는 네 해상도를 사용한다. Support median이 0인 경우의 유한 ratio 처리도 구현에 따른다. 이는 일반 distribution-shift detector나 결함 판별기가 아니다.

$\kappa=9$는 support만으로 선택된 상수가 아니다. [noise attribution protocol](../../configs/benchmarks/noise_attribution/PROTOCOL_20260923.md)에 기록된 1,182개 non-noise 및 1,440개 noise 이미지의 unlabeled 통계로 설정했다. 해당 선택에서 model output/GT를 쓰지 않았다는 사실과 추가 개발 이미지에 접근했다는 사실을 함께 밝힌다. 다른 설계의 label 기반 개발 이력까지 “unlabeled”로 덮지 않는다.

주 AD2 few-shot 실행에서는 guard가 발동하지 않았으므로 그 조건의 RSG4 결과는 fusion core의 결과와 일치한다. 이를 모든 깨끗한 미래 입력에서의 보존 보증으로 확장하지 않는다. Routing, CQ, multiscale, bagging을 각각 별도 핵심 기여로 늘리지 않는다.

Noise ratio를 $\rho(x)$라 쓰면 다음과 같다. 0 median 처리와 estimator의 kernel은 TeX 보충자료에 명시한다.

$$
\mathcal R(x)=\begin{cases}
\{448,672\}, & \rho(x)>9,\\
\{448,672,896,1120\}, & \text{otherwise}.
\end{cases}
$$

Core fusion 비교에서는 routing을 끄고 네 해상도를 고정한다. Noise 통계와 cutoff는 선택에만 사용하며 남은 해상도에 learned/confidence weight를 주지 않는다. 고해상도가 항상 유해하다거나 noise가 모든 acquisition shift를 대표한다고 주장하지 않는다.

### 3.5. Aggregation and spatial postprocessing

각 resolution에서 layer fusion 후 bag 평균을 구하고, 원래 이미지 크기로 bilinear resize한다. 선택된 resolution들의 **동일 가중 평균**이 raw anomaly map이다. 세 선택이 최종 predictor에서 만나는 식은 다음과 같다.

$$
A_{\mathrm{raw}}(x)=\frac{1}{|\mathcal R(x)|}\sum_{r\in\mathcal R(x)}
\operatorname{Resize}_{H,W}\left[\frac1B\sum_{b=1}^{B}\left(\frac{z_{11rb}(x)+z_{23rb}(x)}{2}+[z_{5rb}(x)-\tau_{5rb}]_+\right)\right].
$$

**Component Quantile (CQ).** Query map 자체의 99th percentile 이상인 pixel로 mask를 만들고 8-connected component마다 내부 값을 그 component의 최댓값으로 채운다. Dilation은 사용하지 않으며 선택되지 않은 pixel의 값은 유지한다. 최종 map은 $A=\operatorname{CQ}(A_{\mathrm{raw}})$이고 image score는 $\max_p A(p)$다.

CQ의 per-query percentile은 support shallow q99와 다른 threshold다. CQ는 **예측기의 일부**이며 evaluator의 공통 연산이 아니다. 모든 내부 fusion 비교에서 CQ를 고정하고, residual만의 pre-CQ 효과가 필요하면 별도 no-CQ 분석으로 구분한다. Query에서 percentile을 계산하지만 backbone·bank·support scale을 업데이트하지 않는다.

근거: [CQ 구현](../../src/fmad/postprocess.py). Bag→layer 관계를 뒤집지 않도록 알고리즘에는 “각 bag 안에서 layer fusion → bag 평균 → resize → scale 평균 → CQ” 순서를 적는다.

### 3.6. Implementation and development choices

**재현성 문단**: normal support만 fitting/prediction에 제공한다. GT는 평가 단계에서 소비한다. Ordered support manifest, support/query/GT content hash, backbone weight hash, 코드 snapshot, resolution·crop, bank seed/cap/bags, normalization, CQ, precision/runtime, evaluator contract를 저장한다. Seed만 공개하는 것으로 충분하다고 쓰지 않는다. Base method는 cap256 FP32이며 FP16 실행 및 cap1024는 별도 구성으로 구분한다.

**Algorithm 1 초안**: (1) support별 multiscale frozen features 추출, (2) bag별 동일 layer 위치 표본화, (3) LSO/LTO 거리와 $c,\tau$ 저장, (4) query의 사용 resolution 선택, (5) component별 1-NN·q95 scaling·excess fusion, (6) bag/resolution 평균·CQ·max image score. 역전파, query bank update, target-label 입력 단계는 없다.

## 4. Experiments — 약 4쪽

### 4.1. Evaluation contract and development disclosure — 약 0.5쪽

**P1. Dataset와 scope.** AD2 public full8을 주요 개발 비교에 사용하고 MVTec AD full15·VisA full12로 적용 범위를 본다. Normal support count는 1/2/4/8/16이다. 동일 shot·seed·class에서는 support manifest를 고정한다. Partial class 결과를 full macro로 채우지 않는다.

**P2. Metrics.** AUPRO@0.05는 FPR $[0,.05]$의 PRO 적분을 .05로 나눈 값이다. pAUROC@0.05는 같은 구간의 **standardized partial pixel AUROC**이며 $a=.05$, $I=\int_0^a\mathrm{TPR}(f)df$에 대해 $0.5[1+(I-a^2/2)/(a-a^2/2)]$다. Native-image pixel을 class 안에서 모으되 PRO는 연결 GT region에 동일 가중치, pixel AUROC는 foreground pixel에 동일 가중치를 준다. 이상 이미지의 정상 영역도 FPR 분모에 포함한다. 공유 evaluator는 raw score를 float16 순서로 평가하며 이는 feature FP16 실행과 별개다. Class를 동일 가중 평균하고 draw별 macro의 mean±sample SD를 보고한다. 모든 표시값은 ×100이다. AUPRO 적분 한계는 운영 FPR 보증이 아니고, 지표 방향 차이만으로 미세 결함 회복을 주장하지 않는다. F1 oracle은 평가 label로 threshold를 고른 진단값이다. [Evaluator 구현](../../src/fmad/evaluation/streaming.py).

**P3. Two comparison levels.** 내부 fusion 비교는 feature, bank, support, calibration source, scales, CQ, evaluator를 고정한다. 외부 방법 비교는 동일 support/query/evaluator를 사용하되 각 방법의 backbone·layer·resolution 설정 차이를 표기한다. SuperAD와의 비교도 완전한 residual-only ablation이 아니다.

**P4. Development exposure.** AD2 public, E1, figure pilot, MVTec/VisA 진단은 이미 본 개발 데이터다. “No labels in fitting/prediction”와 “No labels in research development”를 구별한다. 새 global weight는 2-shot public label로 선택해 같은 query population의 16-shot에 고정 적용한다. 이는 shot-transfer 개발 비교이며 독립 test population 검증이 아니다. 새 support draw도 reference 선택 민감도의 반복일 뿐 untouched holdout이 아니다. Residual 상수에도 개발 이력이 있으며 private GT 비공개도 최초 단회 평가를 뜻하지 않는다.

### 4.2. Core comparison: does excess fusion improve on strong simple alternatives? — 약 1쪽

본문에서 가장 중요한 표다. Full addition만 상대하는 비교를 피하고 deep-only, equal mean, tuned global weight, hard gate를 같은 표에 둔다.

**Table 1a. 완료된 AD2 full8 16-shot seed0 fusion 비교 — 원본 suite 구분.** 모든 수치 ×100. A–E factorial은 class별 동일 실행 조건으로 대조된 비교다. Mean3와 신규 weighted/hard 행은 ordered support, content, backbone/bank hash, 전처리, evaluator, runtime 차이를 대조한 범위에서만 paired effect로 해석한다. Nominal config가 같다는 사실만으로 cross-suite 동일 실행을 주장하지 않는다. Cap256, frozen DINOv2-L/14, 네 해상도, 네 bag, CQ 조건이다. MAD/q95 교체는 deep component 단위도 바꾸며 MAD는 $1.4826\,\mathrm{MAD}$ divisor이고 median을 빼지 않는다.

| Calibration | Fusion | AUPRO@.05 | pAUROC@.05 | 해석 |
|---|---|---:|---:|---|
| MAD | Full shallow, A | 51.15843 | 73.01798 | Factorial control |
| q95 | Full shallow, B | 54.01569 | 74.03991 | Full addition의 상대 가중치 명시 |
| MAD | Excess, C | 57.48191 | 76.15659 | q95와의 비교에서 deep 단위도 변경 |
| q95 | Excess, D | 59.14576 | 77.10613 | SCARF core |
| q95 | Deep-only, E | 58.67094 | 77.35072 | 가장 중요한 단순 비교군 |
| None | Equal mean3 | 57.54827 | 76.25828 | 별도 figure-pilot 완료 arm |
| q95 | Equal mean3 | 58.21257 | 76.30651 | 이미 존재하는 강한 평균 비교군 |
| q95 | Global $\lambda^*$ (2-shot seed0에서 선택, 16-shot 고정) | PENDING | PENDING | 선택 protocol 고정, 결과 대기 |
| q95 | Hard gate | PENDING | PENDING | Threshold 차감의 필요성 검증 |

근거: [factorial](../../results/comparisons/fusion_factorial_20260922/comparison.md), [mean3](../../results/fig1_pilot_20260929/remote_full8/comparison.md), [pilot의 공통 설정](../../configs/benchmarks/fig1_pilot/PROTOCOL_20260929.md), [감사 및 factorial source mapping](../../results/remote_runs/dsba3/scarf_v16_fusion_20261003/REUSE_AUDIT.md). 완료된 감사에서 VOID/mean3를 이번 후보들과 동일 실행으로 재사용하는 것은 기각됐다. 별도 FF_A–E의 40개 manifest는 class 내 support/query/GT/backbone/bank/code/runtime/evaluator가 일치하며 VOID 감사 대상과 공통 run ID가 없다. 정확한 설계는 matched 2×2 normalization–residual factorial(A–D) 및 deep-only control(E)이다. 서로 다른 보고서의 마지막 $10^{-5}$pp 자리 차이를 새로운 효과로 해석하지 않는다.

**표 뒤 P1 — 무엇이 개선되는가.** D−B는 +5.13007 AUPRO, +3.06622 pAUROC, AUPRO 개선 6/8 class다. D−C는 +1.66385 AUPRO, 8/8 class 개선이다. D−equal mean은 +0.93319 AUPRO, +0.79962 pAUROC다. 앞의 두 비교는 같은 factorial의 full addition 억제와 scale 변경을 평가한다. Equal mean과의 차이는 별도 suite의 관측 격차이며 execution-matched fusion 효과로 해석하지 않는다.

**표 뒤 P2 — 무엇이 아직 약한가.** D−E는 +0.47482 AUPRO, −0.24459 pAUROC, 5/8 class 개선이다. Can 제외 +0.126pp, can+wallplugs 제외 −0.125pp는 *post-hoc sensitivity*로만 제시한다. 특정 class 이름을 곧바로 “미세 결함 전용 기여”의 증거로 사용하지 않는다. $(D-C)-(B-A)\approx-1.193$pp이므로 superadditive synergy를 주장하지 않는다.

**Table 1b. Shot 축의 강한 비교군 — 완료된 historical seed0 결과.** 셀은 AUPRO / pAUROC이다. 다섯 shot의 mean3 결과는 모두 완료되어 있으며 새 실험 누락으로 기재하지 않는다.

| Shot | Raw equal mean3 | q95 equal mean3 | q95 deep-only | q95 excess |
|---:|---:|---:|---:|---:|
| 1 | 52.09987 / 72.45589 | 52.38993 / 72.08337 | 54.55047 / 73.40395 | 54.22619 / 72.49745 |
| 2 | 55.16355 / 74.53498 | 54.92155 / 74.38633 | 58.66538 / 76.48480 | 58.70413 / 76.24473 |
| 4 | 56.67252 / 75.82566 | 56.58733 / 75.51907 | 58.12778 / 76.80224 | 58.36373 / 76.53931 |
| 8 | 56.95552 / 76.41271 | 57.36262 / 76.25240 | 57.65426 / 77.21815 | 58.40075 / 77.04259 |
| 16 | 57.54827 / 76.25828 | 58.21257 / 76.30651 | 58.67093 / 77.35073 | 59.14575 / 77.10613 |

근거: [mean3 1/2/4/8-shot](../../results/fig1_pilot_20260929/stage2_full8/comparison.md), [mean3 16-shot](../../results/fig1_pilot_20260929/remote_full8/comparison.md), [VOID deep/gated](../../results/imported/void_instrumentation_20260922/comparison.md). Figure-pilot은 VOID support manifests를 사용한다. 표 1a의 factorial 수치와 표 1b의 VOID 수치는 원본 suite의 값을 각각 보존했다. [재사용 감사](../../results/remote_runs/dsba3/scarf_v16_fusion_20261003/REUSE_AUDIT.md)는 VOID 2/16-shot off/all 대 gated/mean3의 runtime·bank 불일치를 확인했다. 1/4/8-shot의 exact execution pairing도 이 감사가 인증하지 않는다. 표 1b의 작은 차이는 clean fusion-only 추정량이 아니다.

**필수 해석**: 2-shot D−E는 **+0.03875pp AUPRO / −0.24007pp pAUROC**다. 1-shot에서는 AUPRO도 −0.32428pp다. 이 seed0의 다섯 shot 모두에서 pAUROC는 deep-only가 더 높다. 따라서 “support가 적을수록 residual의 추가 정보가 필요하다”는 결론은 현재 표에서 나오지 않는다. 본문에는 2/16-shot endpoint와 핵심 차이를 압축하고 전체 shot table은 보충으로 이동할 수 있다.

### 4.3. Resolution construction and conditional selection — 약 0.5쪽

**Q1. 왜 여러 해상도를 구성하는가?**

**Multiscale.** 네 해상도 59.14577 AUPRO, 1120-only cap256 56.95747, 1120-only cap1024 56.62937이다. 이 controlled development 비교에서는 총 bank 용량을 맞춘 단일 최고 해상도도 낮다. “다해상도의 보편적 본질적 이득” 대신 해당 config의 관측으로 보고한다. 근거: [E12](../../results/v7_evidence/E12_resolution_control/comparison.md).

**Q2. 왜 noise에 따라 선택하는가?** MVTec AD full15, 16 clean supports, seed0, 동일 clean/Gaussian-s3 population의 완료된 결과를 사용한다. 아래는 각 조건의 class-macro AUPRO@.05다. AD2 표와 다른 dataset이다.

| Resolution policy | Clean | Gaussian-s3 | 해석 |
|---|---:|---:|---|
| Fixed four: 448/672/896/1120 | 85.36138 | 64.67413 | Routing off, excess 유지 |
| Fixed low two: 448/672 | 84.60962 | 69.00893 | Noise 개선과 clean −0.75175pp |
| Noise-conditioned selection | 85.36138 | 69.00893 | 관측된 clean에서 미발동, Gaussian-s3 1,725/1,725 발동 |

고정 저해상도는 noise에서 +4.33479pp이나 clean에서 손실이 있다. Routing은 관측된 두 조건에서 해당 branch의 결과를 유지한다. 이는 이 개발 조건에서 조건부 선택의 실용적 근거이지 미래 clean 보존, 일반 shift 탐지, resolution confidence 추정의 보증은 아니다. Gaussian-s3 routed 값은 SuperAD 71.10·PatchCore 74.03보다 낮다. Routing은 noise 실패를 관측한 뒤 설계되었으며 cutoff 선택에는 추가 unlabeled 개발 이미지가 사용됐다.

근거: [attribution summary](../../results/noise_attribution/attribution_summary.json), [고정 저해상도 config](../../configs/benchmarks/noise_attribution/NA_lowres.yaml), [attribution protocol](../../configs/benchmarks/noise_attribution/PROTOCOL_20260923.md), [remedy summary](../../results/noise_remedy/remedy_summary.json), [corruption comparison](../../results/comparisons/corruption_compare_20260922/comparison.md). Fixed-low 비교는 이미 완료되어 missing experiment로 재등록하지 않는다. Stored corruption root를 유지해야 동일 비교다.

**Q3. Excess와 routing의 상호작용을 입증했는가?** 아니다. Core fusion은 routing off, 위 routing 비교는 excess 고정이다. 모든 fusion arm×routing의 factorial은 현재 근거가 아니며 synergy를 주장하지 않는다. 주 AD2 결과에서는 routing이 발동하지 않아 해당 이득을 routing으로 설명할 수 없다.

### 4.4. Whole-pipeline benchmark — 약 0.75쪽

**Table 2. AD2 public full8 16-shot, seed0–2 mean±sample SD.** 완성 pipeline의 맥락을 보여주는 표이며 residual 효과의 causal estimate가 아니다. 아래 값은 canonical CSV의 해당 seed별 macro를 재집계한 값이다.

| Method / evaluated configuration | AUPRO@.05 | pAUROC@.05 | Backbone·설정 구분 |
|---|---:|---:|---|
| SCARF / RSG4 cap256 FP32 | 59.62938 ± 0.79638 | 77.23270 ± 0.29345 | DINOv2-L/14, [5,11,23], four scales |
| SuperAD | 55.15555 ± 0.46035 | 76.53584 ± 0.60479 | DINOv2-L/14, [5,11,17,23], 672 |
| PatchCore | 49.15927 ± 0.77331 | 70.86701 ± 0.33249 | DINOv2-L/14 project setting, 672 |
| AnomalyDINO | 44.35635 ± 0.26730 | 75.09939 ± 0.17030 | DINOv2-L/14 project setting, [23] |
| SubspaceAD | 39.53507 ± 0.34457 | 71.80415 ± 0.23042 | DINOv2-with-registers-G/14 |
| RAD | 33.54179 ± 0.98069 | 70.02626 ± 0.77904 | DINOv3-B/16, independent implementation |
| FastRecon | 17.31487 ± 1.20342 | 60.64361 ± 0.96629 | ResNet-50, paper-config reimplementation |

근거: [canonical CSV](../../results/comparisons/psma_mr_vs_history_20260909/psma_mr_vs_history_20260909.csv), SCARF [seed0](../../results/rsg4_fewshot_seed0_20260926/comparison.md)·[seed1](../../results/rsg4_fewshot_seed1_20260926/comparison.md)·[seed2](../../results/rsg4_seed2_s16_20260926/comparison.md), SuperAD/PatchCore [seed0](../../results/paper_v10_stage1_20260928/comparison.md)·[seed1–2](../../results/paper_v10_stage2_20260928/comparison.md), AnomalyDINO [seed0](../../results/public_v13_20260929/anomalydino_seed0_s16/comparison.md)·[remaining](../../results/public_v13_20260929/anomalydino_remaining_full8/comparison.md), [SubspaceAD](../../results/public_v13_20260929/subspacead_paper_full8/comparison.md), [RAD](../../results/public_v13_20260929/rad_paper_full8/comparison.md), [FastRecon](../../results/public_v13_20260929/fastrecon_paper_full8/comparison.md).

**표 뒤 문단**: SCARF−SuperAD는 +4.47383 AUPRO / +0.69686 pAUROC다. 동일 support와 backbone을 사용하는 이 비교에서도 layer·resolution·bank·후처리 차이가 있으므로 이 차이를 residual의 개선량으로 쓰지 않는다. AnomalyDINO/PatchCore의 project backbone 설정과 독립 구현의 원논문 설정 검증은 보충에 둔다. 공식 checkpoint를 사용하는 RegAD 등 추가 baseline은 설정·scope가 확인된 행만 확장한다. 낮은 재현값만으로 해당 방법군 자체의 일반적 열세를 주장하지 않는다.

**Shot summary**: SCARF의 세 draw 평균은 2-shot 55.61652±2.69990, 16-shot 59.62938±0.79638 AUPRO다. Seed0 2-shot 58.70413과 세 draw 평균을 혼용하지 않는다. 본문 shot plot에는 반복이 있는 endpoint와 단일 seed만 있는 중간 shot을 서로 다른 표식으로 구분한다.

### 4.5. Support variation and transfer — 약 0.75쪽

**P1. 이미 완료된 반복 근거.** [E1_v2](../../results/v7_evidence/E1_v2/comparison.md)는 gated와 full-addition을 다섯 support draw에서 비교했다. 사전 정의된 **regular stratum**의 gating 이득은 2-shot이 16-shot보다 큰 방향을 5/5 draw에서 보였으며 평균 차이는 +3.78pp다. Whole-split gated AUPRO는 2-shot 56.07±2.39, 16-shot 59.48±0.60이다. 이는 D/B의 반복 근거이며, D/E나 D/weighted의 반복 근거로 바꾸어 쓰지 않는다. Shot 변경은 support 다양성, bank 총량, calibration 표본 수를 함께 바꾼다.

**P2. Transfer failure를 본문에 남긴다.** 아래 값은 동일 cap256 내부 arm 비교다. 특히 1-shot LTO의 한계를 연구 질문에 연결하되 원인으로 확정하지 않는다.

**Table 3. 다른 dataset의 완료된 fusion 비교, seed0 AUPRO@.05.**

| Dataset / scope | Shot | Full addition | Deep-only | Excess | Excess − deep (pp) |
|---|---:|---:|---:|---:|---:|
| MVTec AD full15 | 1 | 72.68286 | 79.35986 | 78.38775 | −0.97211 |
| MVTec AD full15 | 2 | 75.74859 | 81.93712 | 81.75016 | −0.18696 |
| MVTec AD full15 | 4 | 79.31851 | 83.54851 | 83.63268 | +0.08417 |
| VisA full12 | 1 | 61.78800 | 75.68858 | 72.58754 | −3.10104 |

근거: MVTec [1-shot](../../results/remote_runs/rsg4_diagnosis_20261001/mvtec_seed0_s01_full15/comparison.md)·[2-shot](../../results/remote_runs/rsg4_diagnosis_20261001/mvtec_seed0_s02_full15/comparison.md)·[4-shot](../../results/remote_runs/rsg4_diagnosis_20261001/mvtec_seed0_s04_full15/comparison.md), [VisA](../../results/remote_runs/rsg4_diagnosis_20261001/visa_seed0_allshots_full12/comparison.md). VisA suite 전체는 partial이지만 위 1-shot 네 arm은 각각 full12 완료다. Δ는 표시값 산술이며 반올림 전 원자료와 마지막 자리가 다를 수 있다.

**P3. Capacity를 혼동하지 않는다.** Cap1024 gated는 MVTec 1-shot 79.49687, VisA 1-shot 77.96597로 상승하지만 cap256 deep-only와의 비교는 residual-only 검증이 아니다. Capacity는 reference sampling과 calibration distribution을 함께 바꾼다. Capacity 설명이 다음 결정에 중요하면 deep/weighted/gated 모두를 같은 cap1024에서 비교한다.

Image AUROC·full-range pixel AUROC는 [MVTec E9](../../results/v7_evidence/E9_mvtec/comparison.md), [VisA E9](../../results/v7_evidence/E9_visa/comparison.md)를 보충에 정리한다. SuperAD가 여러 image-AUROC 조건에서 높으므로 localization 결과로 모든 탐지 목적의 우월성을 대신 주장하지 않는다.

### 4.6. Mechanism and operational boundary — 약 0.75쪽

**P1. Matched-FPR 순위 분석.** 동일 FPR에서 deep/weighted/excess가 회복하거나 잃은 GT region을 함께 센다. 새 수치가 없으면 이 분석은 PENDING이다. Total score 증가·감소를 검출 향상의 직접 증거로 쓰지 않는다. Defect-size별 이득을 주장하려면 native mask의 크기별 분석과 크기 정의가 필요하다.

**완료된 aggregate 근거와 strict-FPR 반례.** 기존 factorial에서 all-test FPR 0.1%의 PRO는 excess 19.733, deep-only 20.153, full addition 21.380이다. Shift1의 같은 지점은 각각 17.622, 20.700, 17.775다. 저장된 float16 ROC의 선형 보간값이며 source-calibrated 운영 threshold 결과는 아니다. AUPRO@.05 향상은 모든 낮은 FPR 지점의 향상을 뜻하지 않는다. CQ 이전 raw AUPRO도 full 53.485, deep 57.743, excess 58.321로 excess−deep 약 +.579pp가 존재하므로 CQ만의 효과도 아니다. 이 raw/CQ 및 aggregate PRO 근거는 이미 완료됐으며 추가 추론할 필요가 없다. 미완료는 개별 native GT region의 회복·손실과 크기별 분석이다. [원본 factorial §5](../../results/comparisons/fusion_factorial_20260922/comparison.md).

**P2. Gate activation의 정확한 의미.** 기존 16-shot full8 정상/결함 gate activation macro는 1.145%/11.222%다. 이는 결함과 연관된 선택성을 보여주지만 final FPR·recall·정상 원인 제거율이 아니다. Regular 정상 activation은 0.998%, shift1은 1.338%로 증가한다. v15의 1-shot “약 1.2%”는 해당 16-shot artifact에서 출처가 확인되지 않아 사용하지 않는다. 근거: [factorial gate analysis](../../results/comparisons/fusion_factorial_20260922/comparison.md).

집계는 bag별 binary gate 평균→token cell의 native 좌표 투영→resolution 평균→class 내 normal/defect pixel-weighted 평균→class 동일 평균이다. Image-mean과 pixel-weighted defect activation은 반대 방향으로 움직일 수 있으므로 aggregation을 바꾸어 유리한 결과만 제시하지 않는다.

**Table 4. Source-fixed threshold에서의 반례, 16-shot seed0 full8.** 명목 source FPR 1%, support와 분리된 정상 calibration 16장을 추가 사용했다. 값은 class macro %다.

| Arm | Regular FPR | Shift1 FPR | FPR increase (pp) | Shift1 PRO |
|---|---:|---:|---:|---:|
| Full addition | 1.08402 | 1.76215 | +0.67813 | 52.76962 |
| Deep-only | 1.12518 | 2.50976 | +1.38458 | 61.01662 |
| Excess | 1.11166 | 2.51585 | +1.40419 | 61.69716 |

Excess는 full addition보다 shift1 PRO가 +8.92754pp 높지만 FPR inflation은 +0.72606pp 나쁘다. 사전 정의된 복합 가설은 실패했고 정상 이미지만 사용한 확인에서도 같은 방향이다. 이 결과를 숨기지 않고 **low-FPR 곡선 성능과 배포 threshold 안정성은 다른 문제**라고 결론 낸다. 근거: [source-fixed analysis](../../results/comparisons/source_mechanism_20260922/comparison.md). 실패한 보증을 정당화하려고 같은 실험을 반복할 필요는 없다.

**P3. Acquisition 비교 범위.** 기존 native-stratum 결과에서 regular/shift1은 full8, shift2/3은 fruit_jelly를 제외한 full7이다. 모든 조건을 합친 all-test metric은 조건별 macro 평균의 평균이 아니다. 각 조건에서 높은 절대 성능과 가장 작은 degradation은 다른 주장이다. [factorial의 condition별 표](../../results/comparisons/fusion_factorial_20260922/comparison.md)를 근거로 제한적으로 서술한다.

### 4.7. Supporting results: main-text highlights, supplementary detail — 약 0.25쪽

**Private evaluation.** Base cap256 16-shot seed0의 공식 결과는 private 64.80 / mixed 62.49 AUPRO다. Cap1024+FP16 별도 구성은 65.22 / 62.47이다. Capacity·precision·execution이 함께 바뀌었고 mixed가 상승하지 않았다. Private F1에는 16-shot support 이외의 정상 train calibration 이미지가 사용되므로 “모든 결과가 16장만 사용”이라고 쓰지 않는다. Full-data leaderboard와 동일 조건 순위를 주장하지 않는다. 근거: [base submission](../../results/rsg4_private_submission_20260928/comparison.md), [cap1024 FP16 submission](../../results/remote_runs/dsba3/rsg4_private_cap1024_fp16_20261002/comparison.md).

**Efficiency.** Training-free가 inexpensive를 뜻하지 않는다. Cap256 FP16 execution suite는 seed0–2 full8 완료이며 seed0 FP32 대비 AUPRO −0.00058pp, pAUROC −0.00225pp다. Seed0 query time 5.476→1.498 s/image는 runner·feature device·chunk·reuse도 변경된 execution comparison의 descriptive 값이다. “FP16만으로 3.65× 가속”, 통계적 동등성, 다른 dataset의 정밀도 무손실을 주장하지 않는다. 근거: [seed0 precision/execution audit](../../results/remote_runs/dsba3/rsg4_cap256_fp16_public_20261003/seed0_full8/comparison.md), [seed1](../../results/remote_runs/dsba3/rsg4_cap256_fp16_public_20261003/seed1_full8/comparison.md), [seed2](../../results/remote_runs/dsba3/rsg4_cap256_fp16_public_20261003/seed2_full8/comparison.md). 이 검증은 이미 완료되어 중복 실행 슬롯으로 두지 않는다.

## 5. Discussion and Limitations — 약 0.35쪽

**P1. Method-specific limitation.** Support q95/q99는 유한하고 상관된 patch 표본의 경험적 기준이다. LSO는 더 작은 bank로 같은 support token을 평가하며 실제 query의 모집단은 다르다. Bank 크기에 대한 동일 token의 거리 단조성만으로 target query의 FPR를 제한할 수 없다. One-shot LTO는 이미지 간 정상 변동을 관측하지 못한다.

**P2. Performance boundary.** Residual은 full addition의 손실을 완화하지만 deep-only보다 항상 좋지 않다. AD2에서 pAUROC의 손해, MVTec/VisA low-shot 실패, fixed-source FPR inflation을 함께 인정한다. Global weight로 유사 성능이 나오면 방법의 독립적 필요성을 축소해야 한다. 비슷한 평균을 “더 안정적”이라고 바꾸어 주장하려면 그 안정성 지표와 반복 근거가 별도로 필요하다.

**P3. Resolution boundary.** q95는 layer뿐 아니라 resolution·bag별로 정의되지만 해상도의 신뢰도 추정은 아니다. Multiscale 비교는 단일 개발 draw의 구성 이득이고 routing은 제한된 noise 조건의 선택 근거다. 추가 개발 데이터로 정한 cutoff, AD2에서의 미발동, fusion×routing 상호작용 미검증을 함께 밝힌다.

**P4. Evaluation and computation.** Public benchmark가 개발에 사용되었고 공식 private 제출도 반복 이력을 가진다. 결과의 유효성과 독립 검증의 강도를 구분한다. 다해상도 feature extraction의 비용, cropped border의 표현 한계, CQ가 region score를 바꾸는 영향이 남는다. 산업 현장 배포 threshold 보증이나 실제 생산 안전성 평가는 제공하지 않는다.

## 6. Conclusion — 약 0.15쪽

*English draft*: “We studied how normal-support references guide the combination of multilayer and multiresolution observations. SCARF separates component scaling, shallow excess within each resolution, and noise-conditioned selection before equally averaging the selected maps. Controlled comparisons support recovery from unrestricted shallow addition and configuration-specific benefits of multiscale construction and resolution selection. Small deep-only gains, negative transfer, and fixed-threshold failures limit these conclusions. The pending weighted controls will determine whether excess provides value beyond simpler layer weighting.”

새 weighted 실험에서 확인된 범위에 맞춰 마지막 두 문장을 마감한다. 기대한 이득이 없으면 결론을 좁히고, 새로운 모듈을 이미 검증된 방법처럼 추가하지 않는다.

## 7. Claim–evidence–status matrix — 저자용

| Claim | 근거 | 상태 | 원고에서 허용하는 문장 / 남은 조건 |
|---|---|---|---|
| Full addition의 손실을 excess가 완화 | Factorial D−B +5.13007pp, E1 반복 | 조건부 지지 | 명시한 pipeline·dataset에서 관측 |
| q95가 MAD보다 적합 | D−C +1.66385pp, 8/8 | 조건부 지지 | Gated setting에서; 모든 분포의 tail alignment 아님 |
| 별도 suite의 equal mean 대비 관측 격차 | 완료된 1/2/4/8/16 seed0; 16shot +0.93319pp | 단일 draw; 동일 실행 비교 아님 | Mean3 완료 결과를 보존하되 bank/runtime 불일치로 paired fusion 효과로 해석하지 않음 |
| Deep-only에 추가 이득 | Matched 16shot +0.47482; historical 2shot +0.03875는 bank/runtime confound; pAUROC 음수 | 약함 / metric 의존 | Shallow 순이득의 보편성 미확립 |
| 다해상도 구성의 이득 | E12: four 59.14577 vs1120 cap1024 56.62937 | 조건부 지지 | 최대 해상도·nominal bank 용량 통제, 단일 개발 draw; 미세 결함 메커니즘 아님 |
| Noise 조건부 해상도 선택 | Fixed4/fixed2/routed clean·Gaussian-s3 완료 비교 (§4.3) | 조건부 지지 | 고정 저해상도의 clean 손실 회피를 관측; 미래 보존 보증 아님 |
| Routing이 AD2 이득의 원인 | 해당 population에서 미발동 | 지지되지 않음 | AD2 시스템 이득을 routing에 귀속하지 않음 |
| Excess×routing synergy | 교차 factorial 미완료 | 미확립 | 독립 효과 합산·synergy 주장 금지 |
| Global weighting보다 필요 | 새 weighted/hard-gate 비교 | PENDING | 결과 전 우월성 문장 금지 |
| Hard gate보다 초과량 차감이 유용 | Hard arm 실행 중; full8 결과 대기 | PENDING | Gate 효과와 threshold 차감 효과를 현재 분리하지 못함 |
| More support improves coverage로 모든 결과 설명 | VOID 가설 실패; shot당 bank/calibration도 변경 | 지지되지 않음 | 단일 인과 설명 삭제 |
| Baseline 성능 보존 | Pointwise score 부등식만 성립; 실제 counterexample | 잘못된 주장 | 점수 기여량 범위만 서술 |
| Target FPR 보존 | Source-fixed 복합 가설 실패 | 반증 존재 | 명시적 한계로 보고 |
| 16shot pipeline이 evaluated SuperAD보다 높음 | 3 draws +4.47383pp AUPRO | 지지 | Whole-pipeline 관측, residual 인과효과 아님 |
| 다양한 dataset의 low-shot에 항상 유용 | MVTec/VisA 1shot deep-only 열세 | 반증 존재 | Failure table을 본문에 유지 |
| 미세 결함을 선택적으로 회복 | 기존 그림은 downsample·size filter 사용 | 미확립 | Native GT size-stratified 분석 완료 시에만 사용 |
| Private benchmark에서 보편 SOTA | 설정·normal budget 다름; 다른 강한 공개 결과 존재 | 지지되지 않음 | 자사 공식 값과 조건만 명시 |
| Parameter update 없이 실행 | Frozen backbone, support fit, fixed inference | 지지 | Training-free; hyperparameter-free 아님 |
| FP16은 무손실·독립적인 가속 원인 | Cap256 AD2 combined execution comparison | 제한적 근거 | 해당 실행의 작은 metric 차이와 측정 시간만 보고 |

## 8. New experiment slots and decision rules — 저자용, 결과 미기입

### 8.1. Immediate controlled comparison

[실행 protocol](../../configs/benchmarks/scarf_v16_fusion_20261003/PROTOCOL.md)에 비교 조건을 기록한다. 이 outline은 protocol을 대체하거나 실행 상태를 추정하지 않는다.

| 항목 | 합의된 비교 방향 | 현재 문서 상태 |
|---|---|---|
| Population | AD2 public full8, 2/16-shot seed0, 기존 VOID support manifests | Seen-development |
| Fixed pipeline | Canonical FusionFactorial FP32, no noise guard, four scales, bags4, cap256, CQ | 기존 VOID 조건과 일치시킬 대상 |
| Historical evidence | VOID/mean3는 historical 참고, 별도 A–E factorial 근거 보존 | 감사에서 cross-suite runtime/bank 불일치 확인 |
| New matched controls | off/all/gated × 2/16shot × full8 = 48 | 현재 후보 큐 뒤 실행 등록; 결과 대기 |
| Newly run weight candidates | $\lambda\in\{0.125,0.25,0.5,2\}$, 2/16shot | RUNNING; full8 결과 대기 |
| Exact $\lambda=0.5$ | Mean3와 float16 scale/tie 차이를 확인하도록 실행 확정 | RUNNING; 결정 대기 아님 |
| Hard gate | $z_d+z_5\mathbf1[z_5>\tau]$ | 실행 큐 등록; full8 결과 대기 |
| Implementation | 기존 shallow mode와 gamma 사용 | 새로운 inference code 변경 예상 없음 |
| Hardware execution | dsba3 지정된 기존 container의 4 GPU | 실행 로그·결과는 담당자 보고에 따름 |

**확정된 Selection 원칙**: $\lambda\in\{0,.125,.25,.5,1,2\}$ 중 **2-shot seed0 full8 macro AUPRO@.05만으로** global $\lambda^*$를 선택한다. Unscaled metric 차이가 $10^{-10}$ 이내면 작은 weight를 선택한다. 16-shot 결과를 해석하기 전에 값을 잠그고, 16-shot sweep은 sensitivity·shot-transfer로만 보고한다. Class·shot별 최고값을 취합하지 않는다. 최종 primary 비교와 선택에는 이번 snapshot의 zero/unit weight와 residual을 사용한다. 초기 four weights+hard gate **80 class-runs**에 감사로 필요해진 off/all/gated **48 class-runs**를 추가해 **총 128 class-runs**를 실행한다. Selection 조건은 바꾸지 않는다. Seen-development, original FP32, cap256, four scales/bags, CQ 고정, noise routing off 조건이다.

**Decision A**: Residual이 강한 단순 arm보다 일관된 이득을 보이면, 그 차이를 paired support draw와 다른 dataset에서 검증한다. 초록은 확인된 metric·범위만 확장한다.

**Decision B**: 평균이 비슷하면 tuning transfer, class sensitivity, draw variation 중 실제로 측정한 장점이 있는지 본다. 측정 전 “stable”, “robust”, “tuning-free superiority”라는 대체 기여를 만들지 않는다.

**Decision C**: Full addition 대비 차이만 남으면 현재 residual의 CVPR 방법론적 근거는 불충분하다. 실패를 기록하고 범위를 제한한 방법 수정 여부를 별도 판단한다. 무관한 noise·backbone·postprocessing 모듈을 늘려 핵심 비교를 흐리지 않는다.

### 8.2. Follow-up slots: only missing evidence

| Slot | 비교 | 재사용할 근거 | 미완료 값 |
|---|---|---|---|
| N1 | Deep-only / frozen $\lambda^*$ / excess, paired support draws, 2/16shot | E1의 gated/all과 기존 manifest | Mean paired Δ, SD, class wins: PENDING |
| N2 | MVTec/VisA 1/16shot, fixed config와 $\lambda^*$ transfer | E9·diagnosis의 완료 cell | Weighted와 부족한 matched arm: PENDING |
| N3 | 필요 시 VisA 1shot cap256 vs1024, 세 finalist 모두 비교 | Cap256 negative transfer·cap1024 gated | Equal-capacity deep/weighted: PENDING |
| N4 | 개별 native GT region recovery/loss vs deep/weighted | Raw/CQ macro 및 aggregate matched-FPR PRO는 완료; 적합한 maps만 재사용 | 개별 region/size analysis: PENDING |

N1은 draw별 **paired difference**를 보고한다. Seed SD를 통계적 유의성으로 해석하지 않는다. 불확실성 분석 시 같은 query population을 공유한다는 점과 class/draw/image의 계층을 명시하며 pixel을 독립 표본 수로 부풀리지 않는다. N4의 맵 생성·보관은 별도 출력 설정과 scope가 필요하므로 이 문서가 자동 활성화하지 않는다.

**Resolution 후속 조건**: §4.3의 E12 및 fixed4/fixed2/routed clean·Gaussian-s3는 완료 근거다. Fusion×routing 상호작용은 synergy를 주장할 때 필요한 조건부 과제이며 현재 원고는 이를 주장하지 않는다. 새로운 혼합 noise population의 검증은 필요에 따라 별도 protocol로 정한다. 이 편집이 새 실행을 등록하지 않는다.

**반복하지 않을 완료 과제**: E12 resolution control, noise fixed4/fixed2/routing 비교, mean3 다섯 shot, E1 gated-vs-all 반복, source-fixed 실패 확인, cap256 AD2 FP16 seed0–2 검증. 관련 조건이 바뀌거나 provenance 공백이 발견될 때만 재실행한다.

## 9. Main / supplementary allocation and figure plan — 저자용

| Main text, 약 8쪽 | 내용 | Supplementary |
|---|---|---|
| Introduction 1.0 | 문제·강한 비교군·3개 기여 | 과거 연구 경로 제외 |
| Related work 0.5 | 가장 가까운 reference matching·fusion | Baseline 설정 감사 |
| Method 2.0 | 다층·다해상도 관측 → q95 → 해상도 내 excess → noise 기반 선택 → 평균/CQ | Crop·RNG·CQ·noise estimator, pseudocode 확장 |
| Experiments 4.0 | Core fusion → resolution 구성/선택 → pipeline → support/transfer/FPR | 전체 shot/class/stratum, private, noise 상세, capacity, cost |
| Discussion/conclusion 0.5 | 조건부 이득·실패·개발 노출 | 추가 sensitivity와 provenance |

**현재 TeX 그림 구성.** Fig. 1은 완료된 historical shot curve이며 cross-suite bank/runtime 차이를 caption에 유지한다. Fig. 2는 excess·full·weight·hard-gate 연산 도식이다. 방법 개요 Fig. 3은 support의 bank / q95 scale / shallow q99 / noise reference를 query의 resolution selection → component distance/scaling → 해상도 내 excess → bag 평균·resize·동일 scale 평균 → CQ로 연결한다. Noise cutoff의 추가 개발 데이터와 AD2 미발동을 본문에서 밝힌다. 미래 qualitative panel은 N4 완료 시에만 추가한다.

기존 [mechanism extraction](../figures/extract_rsg4_mechanism_maps.py)·[plotting](../figures/plot_rsg4_mechanism.py)은 매 3번째 정상/결함 이미지, short-edge256 cache, float16 저장, mask≥30 pixel 등의 선택을 포함한다. 해당 그림의 deep 68.3%, full 58.8%, gated 66.8%는 full-addition 손실의 회복을 보여주며 deep 성능 보존을 보여주지 않는다. 이를 그대로 사용할 때는 표본화·resize·mask filter를 caption에 명시하고 main native tiny-defect claim을 연결하지 않는다.

**Table budget.** Main Table1은 강한 fusion 비교, Table2는 whole-pipeline benchmark, Table3은 paired/transfer, Table4는 source-fixed boundary로 구성한다. Factorial A/C 전체와 모든 중간 shot은 보충으로 이동 가능하다. Source-fixed 실패와 low-shot negative transfer는 본문에 최소 한 문단과 수치로 남긴다. Auxiliary modules의 표가 핵심 표보다 더 많은 공간을 차지하지 않게 한다.

## 10. Review and feedback record — 저자용

### Initial author consistency pass

- v15의 baseline/FPR 보존 보증, 필연적 collapse, unified theory, 무파라미터·support-only 개발, broad SOTA 주장을 삭제하거나 조건부 관측으로 축소했다.
- 기존 mean3 완료 결과를 복원하고, full addition과 equal mean의 가중치 차이를 식으로 설명했다.
- AD2 2-shot D/E의 +0.03875pp 및 다섯 shot pAUROC 열세를 중심 비교에 반영했다.
- Core method의 수식과 LSO/LTO·crop·CQ·noise threshold 선택을 구현 및 원본 기록과 맞췄다.
- 완료 결과, post-hoc 해석, 새 실험 PENDING을 분리했다. 원래 v15/v15.1·코드·ledger의 수정은 이 문서 작성에 포함되지 않는다.

### Independent review and applied revisions

| 검토 질문 | 초기 상태 | 피드백 이후 조치 |
|---|---|---|
| 초록·기여가 strong comparator 결과보다 앞서 나가는가? | C1, m1 | Deep 비교를 초록 앞에 배치; weighted/repeated 근거의 미완료 유지 |
| 실행 protocol과 일치하는가? | M1 | §3.3·§8.1에 exact .5, 초기 80+감사 후 48=128 class-runs, 2-shot 선택·tie rule·16-shot 고정 반영 |
| Metric과 개발 노출이 분명한가? | M3, M5 | §4.1에 standardized pAUROC·float16·region/pixel weighting 및 동일 query shot-transfer 명시 |
| 기존 negative/보조 근거를 누락했는가? | M4, m2 | §4.6에 strict-FPR 반례·pre-CQ 근거·activation 집계 추가; N4 범위 축소 |
| 수식·method 설명이 실행 코드와 일치하는가? | 리뷰에서 확인 | 점수 부등식을 기여 보증으로 승격하지 않음 |
| 참고문헌과 신규성 표현이 정확한가? | M6 | 새 verified BibTeX 연결, score-dependent weight 해석 추가; 최초성 주장 없음 |
| Cross-suite 재사용과 핵심 과학적 근거가 완료됐는가? | M2, C1 | Audit 완료: historical VOID cross-runtime 불일치 확인 및 48개 동일환경 controls 등록. Weighted 결과·새 pairing 검증·paired deep 반복은 OPEN |
| 본문 분량과 수치 정밀도가 적절한가? | m1, m3 | TeX 주표 2자리, 작은 차이 .039pp, 세부 원자료는 보충·ledger 유지 |

독립 리뷰 원문과 수정 대응은 [Outline_v16_Review.md](Outline_v16_Review.md)에 보존한다. **초안의 문장·protocol 피드백을 반영했으나 CVPR 방법론적 근거의 검증 완료를 의미하지 않는다.** 새 실험 수치가 들어오면 Abstract→Contributions→Table1→Discussion→Conclusion 순서로 다시 검토한다.


**감사 후 추가 반영 (2026-10-03 03:31 KST).** 독립 감사에서 VOID와 mean3의 cross-environment 불일치가 확인되어 초록·표1·shot curve·§8의 reuse 조건을 수정하고 48개 matched controls를 추가 등록했다. 별도 FF_A–E factorial의 within-class matching과 historical VOID sweep을 구분한다. 이 조치는 실행 비교의 유효성을 확보하기 위한 것이며 residual의 추가 효용을 입증한 결과가 아니다.

**TeX 검토 피드백의 추가 반영 (2026-10-03 KST).** Introduction의 factorial과 historical mean3 비교를 분리하고 §7의 독립 인용 가능한 claim도 축소했다. CFF·SPARC·auxiliary modality 선행연구를 primary record에 근거해 §2에 추가했다. FastRecon의 rank-deficient coreset solver 확장을 공개하고 baseline별 구현 조건은 TeX 보충자료에 정리한다. Noise 비교는 MVTec AD full15/16shot/seed0/clean support 조건을 명시하고, noise kernel·정규화·zero-median 처리도 보충자료에 수식으로 제시한다. 시간 수치는 RTX A6000, batch1, support warm-up 1회, query별 CUDA synchronization, query I/O 포함·fitting/GT/evaluation 제외, class-macro 평균 조건으로 제한한다. 이 조치는 문장·재현성의 개선이며 새로운 성능 근거는 아니다.


**Resolution 서사 재정렬 (2026-10-03 KST).** 연구 질문·초록·Introduction·기여·Method §3.1–3.6·Experiments §4.3·한계·결론을 다층·다해상도 관측의 결합으로 통일했다. 핵심 novelty 검증은 excess에 유지한다. 원자료 재확인으로 fixed-low clean/noise 완료값을 복원했으며 새 실험으로 취급하지 않는다. Main AD2 routing 미발동, 동일 해상도 평균, 추가 cutoff 개발 데이터, 미검증 synergy를 명시했다. 이 편집에서 실험 코드·수치 원자료·ledger는 변경하지 않았다.
