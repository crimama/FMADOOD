# Stable-Key / Full-Value Retrieval Stage -1j 결과

## 결론

**Verdict: `HARD_NULL_ABSORBED / NO_CONTINUE_ROLE_SEPARATED_RETRIEVAL / DIAGNOSTIC_ONLY`**

Stable 채널을 anomaly value에서 제거하지 않고 이웃 선택 key로만 사용하면 이전의 defect
absorption은 피할 수 있었다. 그러나 그 key는 full key보다 shifted-normal tail이나 retrieval
일관성을 안정적으로 개선하지 않았고, 같은 수의 무작위 채널 key와도 구분되지 않았다.

## 실험 설정

- MVTec AD2 `fruit_jelly`, `walnuts`, `vial`, `fabric`
- object별 고정 normal support 16장, deterministic 4-fold `12 fit / 4 held`
- DINOv3-H+ fused feature, DVT/Flow/query update 없음
- fixed `k=5`; full/full(A0), stable/stable(A1), shuffled/full(A2), stable/full(A3)
- unseen compound photometric transform, synthetic local/broad defect
- test/public/private, GT, condition token, metadata 접근 없음
- dsba5 `hun_fsad_tta_012`, host GPU 3 → container CUDA 0
- 결과: `results/remote_runs/dsba5/stable_key_full_value_stage_m1j_20260808_v1`

## Gate 결과

| Gate | 요구 | 결과 | 판정 |
|---|---|---:|---|
| Validity | custody/numerical/fold 전부 valid | valid | pass |
| G1 shifted-normal | A0 대비 10%, A2 대비 5% 개선, 3/4 | `1/4` | fail |
| G2 defect preservation | macro ≥.95, object ≥.90 | local `1.0572`, broad `1.0155` | pass |
| G3 retrieval mechanism | A0/A2보다 Jaccard +.05, 3/4 | `0/4` | fail |

| Object | tail reduction vs A0 | vs A2 | local retention | broad retention | Jaccard margin vs A0 / A2 |
|---|---:|---:|---:|---:|---:|
| fruit_jelly | -22.74% | -11.41% | 1.0386 | 1.0451 | -.0025 / +.0064 |
| walnuts | +27.74% | +27.56% | 1.0147 | .9651 | +.0062 / +.0123 |
| vial | -17.05% | +11.82% | 1.0721 | 1.0331 | +.0050 / +.0058 |
| fabric | +7.56% | +14.18% | 1.1033 | 1.0185 | -.0062 / -.0043 |

전체 A3 tail reduction median은 A0 대비 `-4.74%`로 악화했고, A2 대비 `+13.00%`였지만
object 일관성 gate를 통과하지 못했다. Jaccard margin median은 A0 대비 `+.00125`, A2 대비
`+.00611`로 사전등록된 `+.05`와 거리가 컸다.

## 직관적 해석

1. **Value 역할 분리는 맞았다.** Stable value만 쓰는 A1은 local/broad defect contrast가
   `0.4901/0.5945`로 떨어졌다. Full value를 유지한 A3는 `1.0572/1.0155`를 보존했다.
2. **Stable key는 좋은 검색 주소가 아니었다.** Shift에 덜 움직이는 채널이 동일 정상 구조
   patch를 더 안정적으로 찾지는 않았다. Full key 및 shuffled key 대비 차이가 매우 작았다.
3. **낮은 photometric sensitivity는 nuisance/structure 분리 신호가 아니다.** 채널 제거보다
   안전하게 사용해도 defect는 보존하지만, 정상 대응 선택의 충분한 inductive bias가 아니다.

## 유효성 및 재현 확인

- focused tests `9 passed`; Ruff, Python compile, shell syntax 통과
- remote preflight에서 runner/aggregator/retrieval/readout/support hash 검증
- 결과 checksum 전부 통과
- 회수 후 aggregator 독립 재실행, `summary.json` byte-identical 확인
- anomaly map 미생성, 원격 immutable source snapshot 삭제
- `benchmark_comparable=false`; SuperAD/RN-FMLK 비교 `N/A`
- 실행 후 live `progress.jsonl`이 margin 계산 전에 쓰이던 순서를 수정했다. 판정 입력인 object
  `summary.json`은 계산 후 값을 사용하므로 원격 verdict와 수치에는 영향이 없었다.

## 중단 및 다음 방향

- stable fraction, k, transform, object별 gate를 사후 튜닝하지 않는다.
- trainable key projector나 attention retrieval로 확장하지 않는다.
- 유지할 인사이트는 **anomaly value는 full representation에 남겨야 한다**는 점이다.
- 다음 orthogonal 진단은 DVT에서 확인된 **support positional correspondence**를 검색 key에
  넣되 full value를 유지하고, position-shuffled hard null과 비교하는 방향이다.
