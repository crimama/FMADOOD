# RS-DVT Basic: 클래스 불균형 완화를 위한 신뢰도 축소형 DVT 설계

Date: 2026-07-25  
Status: `SUPPORT_PREFLIGHT_READY / NO_PERFORMANCE_CLAIM / NO_PUBLIC_PROMOTION`

## 1. 문제 정의

현재 Basic은 DINOv2-L/14의 `[5, 11, 17, 23]` 다층 특징, Flow, DVT
`position_mean`, latent 1-NN, density weight `0.25`를 사용한다. Same16에서
Basic은 SuperAD보다 pAUROC는 높지만 F1과 AU-PRO는 낮다. DVT의 평균 효과도
pAUROC/F1에는 양수이지만 AU-PRO에는 음수이며, 객체별 방향이 다르다.

따라서 다음 방법의 목표는 평균 점수만 높이는 것이 아니다.

1. 성능 하위 객체의 F1/AU-PRO를 올린다.
2. Basic의 Flow와 다층 특징 이득을 보존한다.
3. 객체 이름, TESTpublic 결과, query 상태를 사용하지 않는다.
4. 모든 객체에 동일한 알고리즘과 상수를 적용한다.

## 2. 제안: Reliability-Shrunk DVT

Basic의 DVT는 16개 support 특징의 위치 평균과 전체 평균의 차이를 하나의
고정 artifact field로 간주하고 전부 뺀다. 이 가정은 정렬이 안정적인 객체에는
유리하지만, 자세·조명·반사·패턴 이동이 큰 객체에서는 정상 변동까지 제거하거나
증폭할 수 있다.

RS-DVT는 artifact field의 **support 합의도**만큼만 제거한다.

### 2.1 입력과 고정 fold

- 입력은 현재 Basic과 동일한 fused, layer-normalized support feature map
  `F_i(p, c)`이다.
- 16개 support 경로를 SHA 순으로 정렬하고 4개 fold에 round-robin 배정한다.
- 각 support 이미지는 patch 수와 무관하게 동일한 가중치를 갖는다.
- fold, backbone, layer, 해상도, support identity는 추론 전에 고정한다.

### 2.2 강건 위치 편향

각 fold `k`에 대해:

`A_k(p,c) = mean_{i in k} F_i(p,c) - mean_{i in k,p} F_i(p,c)`

최종 위치 편향은 fold artifact의 좌표별 median이다.

`A(p,c) = median_k A_k(p,c)`

이는 한두 support 이미지의 반사나 위치 오차가 전체 DVT field를 지배하지
못하게 한다.

### 2.3 신뢰도 축소

위치별 signal과 fold 불일치를 채널 중앙값으로 계산한다.

`S(p) = median_c A(p,c)^2`

`U(p) = median_c median_k |A_k(p,c) - A(p,c)|^2`

`eps = max(1e-12, 1e-6 * median_p (S(p) + U(p)))`

`rho(p) = clip(S(p) / (S(p) + U(p) + eps), 0, 1)`

최종 변환은:

`F'_q(p,c) = F_q(p,c) - rho(p) * A(p,c)`

이다. `rho=1`이면 full-DVT에 가깝고, `rho=0`이면 no-DVT이다. 중요한 점은
`rho`가 객체 결과나 query anomaly score가 아니라 정상 support 합의로만
결정된다는 것이다.

### 2.4 나머지 Basic 경로 보존

RS-DVT 이후에는 기존 Basic을 그대로 사용한다.

- 동일한 Flow architecture와 학습 계약
- 동일한 multilevel fusion
- 동일한 hard latent 1-NN memory
- 동일한 `loo_standardization=off` (`minus_loo` anchor)
- 동일한 density weight `0.25`
- 동일한 raw continuous score map
- RGB guide, morphology, query adaptation, category routing 없음

따라서 검증 대상은 하나뿐이다: **full position-mean DVT보다 support-reliable
position field가 객체 간 하위 꼬리 성능을 개선하는가**.

## 3. 왜 더 일반적인가

RS-DVT는 `sheet_metal이면 DVT`, `walnuts이면 no-DVT` 같은 객체별 oracle
규칙을 만들지 않는다. 동일한 식이 정렬된 부품, 반복 texture, 반사 표면,
위치 변동이 있는 물체에서 각각 support 불확실성에 따라 자동으로 축소된다.

또한 새 score head나 후처리 계수를 추가하지 않고 정상 표현을 만드는 위치에서
한 번만 작동한다. 따라서 단순한 threshold, morphology, score fusion,
query-time prototype update와 구분된다.

## 4. 하드 널과 ablation

동일 실행에서 다음을 반드시 비교한다.

1. `Basic-exact`: 기존 ordinary-mean full-DVT 코드 경로.
2. `NoDVT+density`: DVT만 끄고 density `0.25`는 유지.
3. `Robust-only`: median-of-folds artifact, `rho=1`.
4. `Shrink-only`: ordinary-mean artifact `A_mean`을 쓰되, 동일 fold
   `A_k`의 `A_mean` 주변 MAD로 `U`를 계산하고 동일 식의 `rho`를 적용.
5. `RS-DVT`: robust artifact와 reliability shrinkage 모두 적용.
6. `Scalar-alpha`: support에서 얻은 `mean_p rho(p)`를 전체 위치에 적용한
   matched “less-DVT” 널.
7. `Rho-rank-shuffle`: `rho`의 값과 분포는 보존하되 SHA 고정 permutation으로
   위치 대응만 제거한 널. Artifact `A`는 원래 위치를 유지한다.
8. `Position-shuffle`: `rho`와 artifact를 함께 고정 permutation한 구조 널.
9. `SuperAD-16`과 `RN-FMLK-16`: 동일 support 및 evaluator 비교자.

RS-DVT가 Basic과 no-DVT 사이를 단순 보간한 결과라면 방법론 credit을 주지
않는다. `Robust-only`와 `Shrink-only`를 모두 이기거나, 둘 중 하나만 필요함을
명확히 보여야 한다.

## 5. 단계별 검증

### Stage -1: support-only falsification

- 4-fold held-support normal maps에서 q99.5 tail, hot-area 비율, NN identity
  instability를 Basic/no-DVT와 비교한다.
- DVT 원인 식별 gate로, 위치별 fold disagreement `U(p)`와 held-support에서
  full-DVT가 no-DVT보다 증가시킨 정상 tail `Delta_tail(p)`의 Spearman 상관을
  계산한다. 중앙 상관 `>=0.30`이고 8개 중 6개 이상에서 양수여야 한다.
- 16-support bootstrap과 leave-one-fold-out으로 `rho` 안정성을 검사한다.
  중앙 Spearman `>=0.80`, q95 `|Delta rho| <= 0.15`, low-reliability 상위
  20% 위치 IoU `>=0.60`을 모두 만족해야 한다. 실패하면 per-position
  shrinkage를 종료하고 `Robust-only`까지만 남긴다.
- 위 `0.30/6-of-8`, `0.80`, `0.15`, `0.60` 상수는 Stage -1 이전에
  사전등록하며 결과를 보고 재조정하지 않는다.
- photometric normal shift는 고정된 exposure, gamma, white-balance 축만
  사용하며 memory에 추가하지 않는다.
- 결함 보존은 공개 결과 선택에 사용하지 않는 sealed, held-family synthetic
  cue와 MVTec AD1 development diagnostic으로 검사한다.
- 정상 tail 감소가 synthetic cue 흡수로 설명되면 즉시 종료한다.

### Stage 0: independent-development diagnostic

- 상수나 식을 바꾸지 않은 채 MVTec AD1 또는 별도의 development benchmark에서
  객체별 pAUROC@0.05, raw-map oracle F1, support-calibrated fixed F1,
  AU-PRO@0.05를 계산한다.
- macro뿐 아니라 worst-2 객체 평균, 객체 간 MAD, 5/8 이상 승리 여부,
  최대 객체 하락을 보고한다.
- 이 저장소는 MVTec AD1을 과거 실험에서 이미 사용했으므로 AD1 결과는
  `development diagnostic`일 뿐 promotion proof가 아니다. `untouched`는
  RS-DVT 가설·식·상수·gate를 설계하는 동안 결과와 label을 한 번도 보지 않고
  접근 로그/manifest를 봉인한 benchmark 또는 partition만 뜻한다.

### Stage 1: MVTec AD2

- 이미 관찰한 TESTpublic은 kill-only diagnostic이다.
- 방법 promotion은 TESTprivate server 또는 설계에 사용되지 않은 benchmark의
  sealed 결과가 필요하다.
- support identity, code SHA, evaluator SHA, map coverage를 동일 실행에서
  검증한다.

## 6. 사전등록 성공 조건

Primary balance endpoints:

- `worst2(seg_F1_raw)`가 Basic 대비 `>= +0.010`
- `worst2(seg_AUPRO_0.05)`가 Basic 대비 `>= +0.010`

Co-gates:

- macro pAUROC/F1/AU-PRO 각각 하락이 `0.005` 이내
- 세 지표 중 두 지표에서 객체 승리 `>= 5/8`; 이는 통계적 유의성 검정이
  아니라 방향 일관성과 단일 객체 지배를 막는 consistency gate임
- 어떤 객체·지표도 Basic 대비 `-0.015` 이하 collapse 없음
- `seg_F1_raw`는 raw map의 oracle best F1임을 명시하고,
  support-calibrated fixed F1과 분리 보고
- 모든 객체의 prediction/GT/shape/nonconstant coverage 완전
- same-run SuperAD-16/RN-FMLK-16 및 hard null 완료
- object/source bootstrap은 효과 크기 불확실성의 descriptive evidence로만
  보고하며, 8개 객체만으로 얻은 하한을 단독 promotion 근거로 쓰지 않음

이 조건을 만족하지 못하면 “분산 감소”나 “더 일반적인 방법”을 주장하지 않는다.
단순히 best 객체 성능을 낮춰 표준편차만 줄인 경우도 실패다.

## 7. 예상 실패 모드

- support 16개로 `rho`가 불안정해져 artifact가 얼룩지는 경우
- 정상 위치 불확실성과 실제 결함 위치가 겹쳐 결함 evidence를 함께 줄이는 경우
- fold median이 texture의 유효한 고주파 위치 구조를 약화하는 경우
- macro 개선이 `sheet_metal` 한 객체에 집중되는 경우
- fixed F1 개선 없이 pAUROC만 오르는 기존 DVT 패턴이 반복되는 경우

각 실패는 위 hard null과 worst-object gate로 구분한다.

## 8. 실행 우선순위

1. RS-DVT estimator와 parity unit test 설계
2. `Basic-exact` original path 및 `rho=0` no-DVT exact parity 검증
3. Stage -1 support-only compact assay
4. 통과 시에만 independent-development diagnostic
5. method freeze 후 sealed AD2 확인

현재는 설계 단계이며 GPU 실행이나 성능 향상 주장을 승인하지 않는다.
