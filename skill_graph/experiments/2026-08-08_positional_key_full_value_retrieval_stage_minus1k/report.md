# Positional-Key / Full-Value Retrieval Stage -1k

Date: 2026-08-08  
Verdict: `KILL_FOR_CLAIM / NO_CONTINUE_POSITIONAL_KEY`  
Mechanism verdict: `POSITION_OVERCONSTRAINT`

## Negative Evidence Intake

Stage -1j의 stable-channel key는 shuffled key와 구분되지 않았다. 이번 실험은 그 channel
sensitivity를 재튜닝하지 않고, 별도 DVT causal experiment에서 확인된 support positional
correspondence를 full-value retrieval에 옮겼다. 주요 위험은 RN-FMLK/kNN duplicate, 후보 제한에
의한 쉬운 안정성 proxy, wrong-position null absorption, local anomaly overconstraint였다.

## Motivation

Full value는 defect evidence를 보존하므로 그대로 두고, 정상 patch를 고르는 후보 집합에만
position inductive bias를 넣으면 distribution shift 아래 이웃 선택이 안정화되는지 검증했다.

## Implementable Design

- P0: 전체 memory full-feature 5-NN
- P1: 동일 `(y,x)`의 12 fit-support patch 안에서 full-feature 5-NN
- P2: 반 grid만큼 roll한 wrong-position 12-row hard null
- P3: source별 deterministic coordinate permutation 12-row hard null
- 모든 arm은 선택된 row의 full 1280-D distance 평균으로 score 계산
- 4 objects, fixed support-16, four-fold `12 fit / 4 held`, unseen compound shift

## Evaluation Alignment

이 단계는 support-only synthetic mechanism diagnostic이며 `benchmark_comparable=false`다.
TESTpublic segmentation claim이 아니므로 same-run SuperAD/RN-FMLK `seg_AUROC_0.05`와 `seg_F1`은
`N/A`이며, 해당 baseline을 대신하는 성능 주장을 하지 않는다.

## Code Modification / Creation

- `src/flow_tte/positional_retrieval.py`
- `scripts/run_positional_key_full_value_retrieval_stage_minus1k.py`
- `scripts/aggregate_positional_key_full_value_retrieval_stage_minus1k.py`
- `scripts/run_positional_key_full_value_retrieval_stage_minus1k_remote.sh`
- 세 개 focused test 파일

P0는 전체 candidate matrix를 만들지 않고 기존 exact chunked GPU 5-NN을 사용하며, P1--P3만
12-row exact CPU 후보 계산을 사용한다.

## Added Code Evaluation

- focused pytest: `10 passed`
- Ruff, Python compile, remote shell syntax: pass
- smoke: fixed `k=5`, finite score, global memory row identity와 candidate count audit 통과

## Remote Execution

- target: dsba5 fixed container `hun_fsad_tta_012`
- host GPU 3 → container CUDA 0
- data: `/home/hunim/Volume/DATA/mvtec_ad_2`
- artifact: `results/remote_runs/dsba5/positional_key_full_value_stage_m1k_20260808_v1`
- immutable snapshot/hash preflight 통과 후 실행; 회수 뒤 snapshot 삭제

## SuperAD Baseline and Unified Metrics

Reduced-object Stage -1이므로 SuperAD method comparison은 `comparable=false`, 모든 SuperAD unified
metric은 `N/A`다. Primary diagnostic은 P0 대비 shifted-normal q995 inflation reduction이고,
P2/P3 hard-null margin, local/broad retention, exact row Jaccard를 함께 판정했다.

## Results and Analysis

| Gate | 결과 | 판정 |
|---|---:|---|
| validity | all valid | pass |
| shifted-normal G1 | `1/4` objects | fail |
| defect preservation G2 | local macro `.8851`, fabric `.6422` | fail |
| positional mechanism G3 | `0/4` objects | fail |

P1의 object-median 결과:

| Object | tail reduction vs P0 | vs P2 / P3 | local / broad retention | Jaccard margin vs P0 | vs P2 / P3 |
|---|---:|---:|---:|---:|---:|
| fruit_jelly | +62.99% | +26.67% / +13.77% | 1.0523 / 1.3186 | +.3315 | -.0074 / -.0282 |
| walnuts | -144.67% | -5.36% / +.14% | .9231 / 1.1188 | +.1499 | +.0008 / -.0268 |
| vial | +28.89% | -5.90% / +20.00% | .9227 / 1.3675 | +.2821 | -.2256 / -.3044 |
| fabric | +50.22% | +2.55% / +25.13% | .6422 / 1.0528 | +.1891 | +.0044 / -.0125 |

P1은 P0 대비 tail을 object median `39.55%` 낮추고 Jaccard를 `+.2356` 높였다. 하지만 이
안정성은 correct-position-specific하지 않았다. P2/P3도 좌표 후보가 고정되어 Jaccard가 높았고,
P1의 margin은 각각 `-.0033/-.0275`였다. Tail도 wrong-position null 대비 선택적으로 우수하지
않았다.

더 중요한 실패는 local defect다. Broad retention은 `1.2144`였지만 local은 `.8851`, 특히
fabric `.6422`로 붕괴했다. 동일 위치만 허용하면 global shift에는 덜 흔들릴 수 있으나, 국소
구조 변화가 있는 query patch를 같은 위치의 정상 구조로 강제로 설명하면서 anomaly contrast가
항상 보존되는 것이 아니었다.

## Continuation Assessment

- Strict method claim: 지원하지 않는다.
- 추가 positional-key 진단: 지원하지 않는다.

Positive signal(P0 대비 tail/Jaccard)은 wrong-position null과 분리되지 않아 쉬운 후보 제한 효과로
설명되며, local defect safety도 실패했다. Radius, 위치 가중치, roll offset, k를 조정하는 후속은
사후 튜닝이므로 금지한다.

## Conclusion

`POSITION_OVERCONSTRAINT / KILL_FOR_CLAIM / NO_CONTINUE_POSITIONAL_KEY`다. DVT의 positional
correspondence 이득은 retrieval candidate restriction으로 그대로 이전되지 않는다. DVT는
feature residual prior로서는 medium localization을 돕지만, 동일 좌표 검색 제약은 wrong-position
null과 구분되지 않고 local anomaly를 손상시킨다.

남는 자산은 두 가지다. Full anomaly value 보존 원칙은 유지하며, DVT의 효과를 “위치 기반 이웃
선택”으로 설명해서는 안 된다는 failure boundary가 추가됐다.

## Post-Conclusion Storage Cleanup

- checksum 전부 통과
- 회수 후 aggregator 독립 재실행 결과 byte-identical
- `anomaly_maps/` 없음 확인
- remote immutable source snapshot 삭제
