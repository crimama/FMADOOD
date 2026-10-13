# Stable-Key / Full-Value Retrieval Stage -1j 사전등록

## 1. 질문과 범위

앞선 고정 채널 readout은 support에서 shift에 둔감한 DINO 채널을 찾을 수 있음을 보였지만,
그 채널만 점수 계산에 사용하면 defect contrast와 clean geometry가 함께 소실되었다. 이번 실험은
채널을 제거하는 대신 역할을 분리한다.

- **key**: 정상 memory에서 비교 대상을 고르는 표현
- **value**: 선택된 대상과 anomaly distance를 계산하는 표현

핵심 가설은 `stable channel = 이웃 선택`, `full channel = 이상 증거 보존`으로 사용하면
shifted-normal tail을 낮추면서 defect contrast를 보존할 수 있다는 것이다. 이는 support-only
Stage -1 진단이며 새 방법, SOTA 또는 benchmark 성능 주장에 사용하지 않는다.

## 2. 고정 데이터·표현·금지사항

- 데이터: MVTec AD2 single-image의 `fruit_jelly`, `walnuts`, `vial`, `fabric`.
- support: 기존 checksum이 고정된 object별 normal 16장만 사용한다.
- fold: deterministic 4-fold, 각 fold `12 fit / 4 held`.
- backbone: 기존 DINO feature extractor와 동일한 checkpoint/layer/preprocessing.
- channel mask: 각 fold의 fit support와 고정된 seen transform만으로 산출한 channel sensitivity의
  lower 50%. held image/held compound transform은 mask 산출에 사용하지 않는다.
- k: 모든 arm에서 `k=5`로 고정하며 sweep하거나 object별 조정하지 않는다.
- DVT, Flow, query-time update, test/public/private image, GT mask, 파일명 condition token,
  EXIF/sidecar metadata는 사용하지 않는다.
- 합성 defect mask는 defect-retention 평가에만 사용하며 key, memory, channel selection,
  score calibration에는 사용하지 않는다.
- `benchmark_comparable=false`; SuperAD 및 RN-FMLK 비교는 이 단계에서 `N/A`이다.

## 3. Arm과 정확한 score

모든 arm은 동일한 clean fit-support patch memory와 동일한 row 순서를 공유한다. query patch
`q`, memory row `m_i`, key projection `K_a`, value projection `V_a`에 대해

```text
I_a(q) = stable_topk_i ||K_a(q) - K_a(m_i)||_2,  k=5
score_a(q) = mean_{i in I_a(q)} ||V_a(q) - V_a(m_i)||_2
```

동점은 memory row index 오름차순으로 결정하며 정확히 5개의 서로 다른 row를 반환한다.

| Arm | key | value | 역할 |
|---|---|---|---|
| A0 | full 1280-D | full 1280-D | 현재 full-feature 5-NN 기준선 |
| A1 | fold-local stable lower-50% | 같은 stable lower-50% | anomaly absorption 양성 대조군 |
| A2 | seed 고정 shuffled lower-50% | full 1280-D | 같은 차원의 hard null |
| A3 | fold-local stable lower-50% | full 1280-D | 역할 분리 후보 |

각 fold의 scale/normalization은 fit support만으로 고정하며 arm 사이에서 value 거리를 별도로
재보정하지 않는다. A2 mask는 A1/A3와 정확히 같은 channel 수를 갖는다.

## 4. 평가량

### 4.1 주 평가: shifted-normal q995 tail

held clean image와 사전 고정된 unseen compound transform에서 patch score를 계산한다.
object별 shifted-normal inflation은 `q995(shifted) - q995(clean)`이며,

```text
reduction(A3 vs Ax) = (inflation_Ax - inflation_A3) / max(|inflation_Ax|, eps)
```

으로 비교한다. 분모 부호 문제를 피하기 위해 `inflation_A0 <= 0`인 object는 주 relevance gate에
부적격으로 기록하며 임의의 대체 분모를 쓰지 않는다.

### 4.2 anomaly 보존

이전 실험과 동일한 mask-free score contrast 정의로 held clean에 합성한 local/broad defect의
contrast를 측정한다. A3 retention은 A0 contrast 대비 비율이다. 변하지 않은 patch를 평균에
포함해 smoothness를 부풀리는 방식은 사용하지 않는다.

### 4.3 retrieval 안정성(평가 전용)

held clean query patch와 그 unseen-transform 대응 patch가 고른 top-5 memory row의
Jaccard overlap을 측정한다. 대응 관계는 평가에만 사용하며 검색이나 score 계산에는 주입하지
않는다. object별 평균을 보고 A3-A0, A3-A2 차이를 계산한다.

### 4.4 유효성·custody

- fold disjointness, fit/held source identity, memory row identity 및 arm별 cardinality를 저장한다.
- 모든 입력/요약 JSON에 SHA-256 checksum과 access ledger를 기록한다.
- `test_accessed=false`, `gt_accessed=false`, `metadata_accessed=false`여야 한다.
- NaN/Inf, k!=5, 서로 다른 memory row ordering, mask cardinality 불일치는 즉시 invalid이다.

## 5. 사전 고정 gate

### G1. shifted-normal relevance

적격 object 중 A3가 아래를 동시에 만족하는 object가 최소 `3/4`이어야 한다.

- A0 대비 q995 inflation reduction `>= 0.10`
- A2 대비 q995 inflation reduction `>= 0.05`

### G2. defect evidence preservation

- A3/A0 local 및 broad contrast retention 각각 macro `>= 0.95`
- 어느 object·defect scale도 retention `< 0.90`이면 실패

### G3. role-separation mechanism

A3의 clean↔shift top-5 Jaccard가 A0 및 A2보다 각각 `>= 0.05` 높은 object가 최소 `3/4`이어야 한다.
A1은 예상되는 absorption을 확인하는 mechanism control일 뿐 G1의 비교 대상이 아니다.

## 6. 판정과 중단 규칙

- `PROMISING_ROLE_SEPARATED_RETRIEVAL`: validity와 G1/G2/G3 전부 통과.
- `RETRIEVAL_ONLY_SIGNAL / KILL_FOR_CLAIM`: G3와 G2는 통과하지만 G1은 실패.
- `ANOMALY_ABSORPTION_PERSISTS`: G2 실패.
- `HARD_NULL_ABSORBED`: A3가 A2를 G1 또는 G3에서 구분하지 못함.
- `NO_ROLE_SEPARATION`: validity는 통과하지만 위 조건에 해당하지 않음.
- `INVALID_ROLE_SEPARATION_DIAGNOSTIC`: custody/identity/numerical validity 실패.

어떤 실패에서도 k, stable fraction, transform, object별 threshold를 사후 튜닝하지 않는다.
성공하더라도 즉시 방법론 contribution으로 승격하지 않는다. 다음 단계는 동일한 A3를 고정한 채
기존 baseline scoring path 및 RN-FMLK/SuperAD 정합 benchmark에서 독립 검증하는 것이다.

## 7. 중복·해석 제한

이 실험은 k-NN 자체의 신규성이나 k=5 튜닝을 주장하지 않는다. `stable key`가 이웃 선택에는
유용하지만 anomaly value로는 손실을 낳는다는 인과적 역할 분리만 검증한다. 단순히 5-NN
smoothing 효과만 재현하거나 shuffled key와 구분되지 않으면 해당 구조는 폐기한다.
