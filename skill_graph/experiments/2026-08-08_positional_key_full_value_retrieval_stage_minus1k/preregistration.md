# Positional-Key / Full-Value Retrieval Stage -1k 사전등록

## 1. Negative evidence intake와 질문

Stage -1j는 full anomaly value를 유지하면 defect absorption은 피하지만, photometric sensitivity로
선택한 stable channel key는 full/shuffled key보다 정상 대응을 잘 고르지 못함을 보였다. 반면 별도
DVT causal control은 medium defect 이득이 정확한 support position correspondence에 의존하고,
position shuffle/global centering으로 설명되지 않음을 보였다.

이번 질문은 하나다. **동일 spatial position의 support patch 안에서만 full-feature top-5를 고르면,
full anomaly value를 보존하면서 shifted-normal retrieval이 안정화되는가?**

이는 k-NN 또는 support admission 신규성 주장이 아니다. 이미 독립적으로 확인된 positional
correspondence를 retrieval 역할에 옮길 수 있는지만 보는 reduced-object support-only Stage -1
진단이다. RN-FMLK/kNN duplicate와 position-prior overconstraint가 주요 failure basin이다.

## 2. 범위와 custody

- `target_dataset=MVTec AD2 single-image`
- dsba5 data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- objects: `fruit_jelly`, `walnuts`, `vial`, `fabric`
- object별 checksum 고정 normal support 16장, deterministic 4-fold `12 fit / 4 held`
- DINOv3-H+ `[7,15,23,31]` layer-normalized mean fusion
- fixed `k=5`; DVT, Flow, query update, learned parameter 없음
- fit/held photometric transform과 local/broad synthetic defect는 Stage -1j와 동일
- test/public/private, GT, filename condition, metadata 접근 금지
- synthetic defect mask는 retention 평가에만 사용
- `benchmark_comparable=false`; SuperAD/RN-FMLK와 method claim 비교는 `N/A`

## 3. Arm과 score

모든 memory row는 `(source_index, y, x)` identity를 가지며 value는 full 1280-D DINO feature다.
모든 arm은 query와 memory의 full-feature Euclidean distance로 최종 score를 계산한다.

| Arm | admissible memory rows | 목적 |
|---|---|---|
| P0_global_full | 모든 fit-support patch | unconstrained full-feature 5-NN baseline |
| P1_aligned_position | query와 같은 `(y,x)`의 fit-support patch | 정확한 position correspondence 후보 |
| P2_rolled_position | `(y+H//2,x+W//2) mod (H,W)`의 patch | equal-cardinality wrong-position hard null |
| P3_source_permuted_position | source별 seed 고정 coordinate permutation이 query `(y,x)`에 배정한 patch | arbitrary-position hard null |

각 query patch에서 admissible set 내부의 exact full-feature top-5 row를 고르고, 그 동일 row의 full
feature distance 평균을 score로 쓴다. 동점은 global memory row index 오름차순으로 결정한다.
P1/P2/P3는 각 query마다 정확히 `12` candidate rows를 가져야 하며 P0는 전체 fit memory를 쓴다.
Grid가 한 축이라도 2 미만이거나 arm 간 row identity/count audit가 깨지면 invalid이다.

## 4. 평가량

- shifted-normal tail: `q995(held_shift) - q995(held_clean)`
- A0 inflation이 양수인 row에 한해 relative reduction 계산
- local/broad defect contrast retention: P0 대비 비율
- clean↔shift retrieval stability: 대응 query patch의 exact top-5 global memory row Jaccard
- wrong-position separation: P1이 P2와 P3 모두보다 tail과 Jaccard에서 우수한지 평가

## 5. 고정 gate

### G1 shifted-normal relevance

최소 `3/4` object에서 P1이 동시에 만족해야 한다.

- P0 대비 q995 inflation reduction `>= .10`
- P2 및 P3 각각 대비 reduction `>= .05`

### G2 defect evidence preservation

- P1/P0 local 및 broad contrast retention 각각 object macro `>= .95`
- 어느 object·scale도 `< .90` 금지

### G3 positional retrieval mechanism

최소 `3/4` object에서 P1 clean↔shift top-5 Jaccard가 P0, P2, P3보다 각각 `>= .05` 높아야 한다.

### Validity

- support/fold/source disjointness, k=5, candidate cardinality, grid/coordinate hash 통과
- finite/nonconstant score, duplicate neighbor 없음
- `test_accessed=false`, `gt_accessed=false`, `metadata_accessed=false`

## 6. 판정

- `PROMISING_POSITIONAL_KEY_DIAGNOSTIC`: validity와 G1/G2/G3 모두 통과.
- `POSITIONAL_RETRIEVAL_ONLY / KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`: G2/G3 통과, G1 실패.
  허용되는 단일 후속은 동일 P1을 고정한 baseline scoring-path reduced-object check뿐이다.
- `POSITION_OVERCONSTRAINT`: G2 실패.
- `WRONG_POSITION_NULL_ABSORBED`: P1이 P2/P3와 G1 또는 G3에서 구분되지 않음.
- `NO_POSITIONAL_RETRIEVAL_GAIN`: validity는 통과하지만 나머지 gate 실패.
- `INVALID_POSITIONAL_KEY_DIAGNOSTIC`: custody/numerical/coordinate audit 실패.

실패 후 coordinate radius, k, roll offset, feature weight 또는 object별 정책을 튜닝하지 않는다.
성공해도 즉시 method claim으로 승격하지 않으며 all-8 TESTpublic/SuperAD 비교를 열기 전에 별도의
baseline-path 사전등록을 요구한다.
