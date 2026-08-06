# Distribution-Shift 기반 방법론 보완: SDIG v1.0 + NODyR Probe

Date: 2026-07-17
Status: DESIGN COMPLETE / STAGE A TESTPUBLIC SHADOW NOW COMPLETED

## 1. 결론

현재 방법론의 가장 큰 문제는 정상 reference의 선택이나 score calibration이 부족한 것이 아니다. **정상 촬영 변화와 결함이 같은 backbone residual 방향에 얽혀 있는데, 지금까지의 시도는 그 residual을 제한·정렬·밀도화·보정하는 방식으로만 수정했다.** 따라서 모델이 정상 변동을 더 정교하게 설명할수록 결함 대비까지 함께 제거하거나, 정상 manifold coverage를 줄여 정상 변동을 false novelty로 만드는 역효과가 발생했다.

이에 대한 보완은 두 층으로 설계한다.

1. **SDIG(Shift–Defect Identifiability Gate):** 새로운 scorer를 학습하기 전에 그 scorer가 보는 관계에서 distribution shift와 defect가 식별 가능한지를 반증 실험으로 검사한다.
2. **NODyR(Normal-Only Depth Dynamics Residual):** H+ residual을 재보정하지 않고, frozen DINO 내부에서 정상 patch가 layer 15에서 23으로 진행하는 *표현 전이 방향*의 위반을 독립 anomaly evidence로 시험한다.

NODyR는 아직 방법이 아니라 probe다. 동일 용량의 state-prediction/distillation null을 이기기 전에는 새 기여로 인정하지 않는다.

## 2. “어떤 시도가 왜 악화되었는가”

| 시도 | 기대 | 실제 결과 | 악화를 유발한 구조적 문제 |
|---|---|---|---|
| 위치 중심화 / z-score | 위치별 정상 bias 제거 | F1 `-0.097775 / -0.249284` | 제거한 축에 defect contrast도 포함되어 있었다 |
| hard-local / correspondence | 정상 대응을 정밀화 | pAUROC/F1 `-0.117298/-0.016150`; L0 FP/TP `4059.8` | 정상 coverage를 축소해 정상 appearance mismatch가 false novelty가 됐다 |
| registration | 기하 오차 제거 | 5/5 accepted, ranking 복구 실패 | 기하 정렬 가능성과 anomaly evidence 신뢰성은 별개였다 |
| density | 정상 multimodality 모델링 | D0-G0 `-0.026051/-0.011843` | likelihood가 defect보다 dominant normal/acquisition geometry를 더 강하게 추적했다 |
| top-M / support consensus | 복수 정상 mode 보존 | top-M 음수; consensus F1 `+0.000033`, shuffled-ID와 동일 | reference 선택이 causal하지 않았고 같은 bias의 반복이었다 |
| condition-aware correction | shift 제거 | regular/shift 모두 평균 음수, object-condition별 부호 반전 | shift는 하나의 전역 nuisance가 아니라 object x condition interaction이었다 |

이 결과들을 하나의 식으로 요약하면 다음과 같다.

`z = normal_structure + object_condition_acquisition + defect + noise`

현재 residual에서 acquisition 항과 defect 항이 겹치면, 같은 residual로 만든 위치 보정·density·reference selection은 두 항을 분리할 수 없다. 오히려 정상 변동을 제거하면서 defect도 지우거나, 정상 coverage를 좁히면서 FP를 키운다.

## 3. 폐기한 첫 보완안

초기에는 coarse feature로 fine feature를 예측하는 Cross-Scale Normal Compatibility를 고려했다. 그러나 다음 이유로 방법 후보에서 제외했다.

- 저장소에 이미 동일한 `Coarse-Conditioned Predictive Residual` 설계가 존재한다.
- 외부적으로 [Reverse Distillation](https://openaccess.thecvf.com/content/CVPR2022/html/Deng_Anomaly_Detection_via_Reverse_Distillation_From_One-Class_Embedding_CVPR_2022_paper.html)이 high-level bottleneck에서 low-level feature를 복원하는 거의 같은 구조를 제시한다.
- [EfficientAD](https://openaccess.thecvf.com/content/WACV2024/html/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.html)와 student-teacher 계열도 normal feature prediction residual을 이미 사용한다.
- fold agreement를 추가해도 기존 support-consensus 실패 및 [Uninformed Students](https://openaccess.thecvf.com/content_CVPR_2020/html/Bergmann_Uninformed_Students_Student-Teacher_Anomaly_Detection_With_Discriminative_Latent_Embeddings_CVPR_2020_paper.html)의 ensemble uncertainty와 겹친다.

따라서 coarse-to-fine prediction은 hard null로만 남기고, 상태 복원이 아닌 depth transition의 방향을 시험하는 쪽으로 이동했다.

## 4. NODyR probe

Frozen backbone의 patch feature를 `F15`, `F23`이라 둔다. 이 layer pair와 predictor capacity는 평가 전에 한 번만 고정하며 public metric으로 선택하지 않는다.

```text
x = LN(F15)
d = unit(LN(F23) - x)
d_hat = unit(Wx + b)
score = 1 - cosine(d_hat, d)
```

`W,b`는 동일 16개 정상 support의 patch로만 학습하는 dense isotropic ridge operator다. 좌표, condition label, 선택된 support ID, query 통계, 위치별 threshold, 별도 mask는 입력하지 않는다. 최종 map은 raw cosine incompatibility를 원영상으로 bilinear resize한 결과이며 추가 score shaping을 하지 않는다.

가설은 다음과 같다.

- 촬영 변화가 representation의 시작점을 이동시키더라도 정상적인 depth-processing 방향은 유지될 수 있다.
- 작은 결함은 layer를 따라 표현이 진행되는 방향을 국소적으로 굴절시킬 수 있다.

반대로 acquisition 변화가 transition law 자체를 바꾸거나 defect가 두 layer에 동일하게 전파되면 즉시 실패한다.

### 4.1 Evidence와 fixed decision의 분리

Raw NODyR가 아래 intrinsic-null envelope를 이긴 뒤에만, image-disjoint leave-one-reference-out normal map으로 empirical survival transform을 고정한다.

```text
p(u) = [1 + count(E_normal >= E(u))] / [N + 1]
A(u) = -log10 p(u)
```

fold별 query map은 ensemble하지 않는다. calibration 및 bootstrap 단위는 pixel이 아니라 image/source orbit이다. 이 변환은 monotone이므로 raw ranking 성능에 기여한 것으로 계산하지 않으며, support-only fixed F1과 public-best F1을 분리해서 보고한다.

H+와 NODyR가 각각 독립 gate를 통과하고 동일한 normal-tail scale로 보정된 경우에만 `p_final=min(1, 2*min(p_H,p_N))`의 Bonferroni OR를 별도 실험으로 허용한다. 이 결합은 실패한 raw NODyR를 구제할 수 없다.

## 5. 반드시 이겨야 하는 null

NODyR의 성능만 보는 것은 의미가 없다. 아래 null의 최대 성능을 이겨야 transition-direction 기여가 식별된다.

1. 동일 ridge가 `F23` 상태를 예측하고, 그 출력을 NODyR와 같은 transition-angle 공간에서 평가하는 null
2. unnormalized delta prediction
3. unit-delta의 Euclidean residual
4. image-pair 및 spatial pairing shuffle
5. 동일 support/layer/parameter budget의 coarse-to-fine state prediction / Reverse-Distillation-style null
6. channel permutation 및 orthogonal basis sensitivity

State predictor와 동률이면 NODyR는 새 메커니즘이 아니라 residual distillation의 재매개화로 판정한다.

## 6. SDIG 실험 순서

### Gate 0 — 독립 development normal-shift invariance

모든 object-condition cell에서 shifted-good upper-tail inflation이 derived-state null보다 작거나 같아야 한다. 독립 development population이 없으면 이 gate는 실행할 수 없으며, 공개 평가 결과로 대체하지 않는다.

### Gate A — causal diagnostic

- NODyR가 intrinsic-null envelope보다 macro pAUROC/F1 각각 `+0.005` 이상
- source-orbit paired bootstrap 95% lower bound가 두 metric 모두 `> 0`
- 모든 regular cell에서 H+ 대비 손실 `>= -0.01/-0.01`
- 모든 shifted cell에서 손실 `>= -0.02/-0.03`
- shifted macro gain은 두 metric 모두 `>= +0.005`
- 모든 cell의 FP/TP는 H+의 `1.25x` 이하
- true-pair가 shuffle보다 pAUROC `+0.010` 또는 F1 `+0.020` 이상의 사전 정의된 우위

통과해도 판정은 `PROMISING_DIAGNOSTIC`까지만 허용한다.

### Gate B — untouched confirmation

같은 reference-16을 사용하는 SuperAD-16, RN-FMLK/full-memory DINO kNN, H+, intrinsic-null envelope와 공식 private evaluation에서 비교한다. 공개 shadow 결과는 보고만 하며 어떤 선택·변경·중단에도 사용하지 않는다.

최소 승격 조건은 다음이다.

- SuperAD의 사전 고정 3개 seed 모두에 pAUROC/F1 `+0.005` 이상 및 paired lower bound `> 0`
- RN-FMLK와 intrinsic envelope에도 두 metric `+0.005` 이상
- 모든 regular object에서 두 metric 비음수
- shifted macro 두 metric `+0.005` 이상, 어떤 cell도 `-0.02/-0.03` 아래로 하락하지 않음
- intrinsic envelope를 두 metric 모두 이기는 cell이 전체의 2/3 이상이며, 각 object에 shifted 승리 cell이 최소 1개
- full-image map coverage 100%, zero/constant/always-positive map 없음

## 7. 현재 판정

`CONDITIONAL_KEEP_DESIGN_ONLY`가 최대 판정이다.

5단계 독립 평가 점수는 `0.59 -> 0.32 -> 0.65 -> 0.69 -> 0.79`였다. 순서대로 synthetic factorial projection, query-view persistence, CORA, raw NODyR, SDIG-NODyR 분리 설계다. Stage 5가 가장 완결된 설계이며 `conditional-keep` 판정을 받았다. 이는 방법 성능 점수가 아니라 설계의 식별 가능성·반증 가능성·배포 적합성에 대한 평가다. 종료 사유는 5단계 budget 소진이다.

- 방법 설계: 보완됨
- distribution-shift 중심 인과 질문: 명시됨
- 이전 실패 분기 재도입 방지: hard null로 명시됨
- 실험 결과: 없음
- NODyR 방법 유효성: 미검증
- 논문 novelty: residual-distillation envelope를 이긴 뒤에만 재검토

실행 전 남은 필수 산출물은 독립 development data/custody manifest, object-condition별 통계를 반환하는 sealed private evaluator, 정확한 Reverse-Distillation null manifest, support-16 선택 provenance, leave-one-image-out threshold 알고리즘, bootstrap replicate마다 intrinsic-null maximum을 재계산하는 규약이다. 이들이 없으면 `DESIGN_ONLY` 상태를 유지한다.

즉, 지금 바로 주장할 내용은 “NODyR가 더 좋다”가 아니다. **다음 실험은 새로운 점수의 평균 성능을 보는 실험이 아니라, 정상 shift와 defect를 표현 전이 관계에서 실제로 분리할 수 있는지 검증하는 식별 실험이어야 한다**는 것이 이번 보완의 핵심이다.

## 8. 외부 근거 경계

[MVTec AD2](https://arxiv.org/abs/2503.21622)는 미세 결함, 반사·조명 변화, 높은 정상 변동성을 함께 다루므로 macro 한 값보다 object-condition cell 분석이 필요하다. Cross-layer feature prediction 자체는 이미 Reverse Distillation, EfficientAD, contextual-affinity/reconstruction 계열과 가까우므로, 본 설계는 예측기 자체를 새롭다고 주장하지 않는다. 논문 수준 기여는 transition-direction score가 동일 geometry의 state-prediction null을 이기고, acquisition-shift cell에서 부호 일관성을 보일 때만 열린다.

## 9. Stage A empirical update (2026-07-18)

The all-eight TESTpublic shadow is complete. NODyR scored
`0.729019/0.326280` pAUROC/F1 versus `state_angle` at
`0.728246/0.325678`, a margin of only `+0.000773/+0.000602`; the object-wise
intrinsic envelope beat NODyR on both metrics in all eight objects. Shifted
normal score tails inflated strongly, and NODyR/state-angle image statistics
were almost identical. The empirical result therefore does not justify the
calibration/fusion continuation described conditionally above. Full evidence
and the public-shadow claim boundary are recorded in
`skill_graph/experiments/2026-07-17_sdig_nodyr_stage_a/report.md`.
