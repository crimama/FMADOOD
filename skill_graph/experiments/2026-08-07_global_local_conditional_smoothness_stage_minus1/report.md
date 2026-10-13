# Mask-free Conditional-Smoothness Stage -1a1 결과

## Verdict

`INCONCLUSIVE_OR_NOT_SUPPORTED / NO_NEW_MODULE / NO_PUBLIC_ACCESS`

비변화 patch가 평균 smoothness를 부풀리는 경로는 제거했지만, 사전등록한 전역
factorisation predicate는 `0/4` object에서만 성립했다. 따라서 이 결과는 Flow, Flow
Matching, Diffusion, DiT 혹은 global/local interface의 새 구현을 허가하지 않는다.

## Scope와 custody

- MVTec AD2의 고정 normal-support 16장, `fruit_jelly`, `walnuts`, `vial`, `fabric`만 사용했다.
- DINOv3 ViT-H/16+ DVT와 frozen Baseline1 Flow latent, 정적 raw-Euclidean patch 1-NN을
  사용했다. 새 module이나 parameter update는 없었다.
- `train/good` support 이외의 query adaptation, `test_public`, `test_private`,
  ground-truth 접근은 모두 없었다. 두 lane의 access ledger가 동일하다.
- local GPU 0과 dsba5 host GPU 3/container CUDA 0에서 동일 runner
  (`0ce4717e…aa7370dc`)와 support manifest (`9f223f7c…be81d18a5`)를 사용했다.

## 사전등록 측정량

각 clean/changed displacement \(\Delta_p\)에 대해
\(a_p=\|\Delta_p\|_2\), \(\tau=median(a)+MAD(a)\),
\(A=\{p:a_p\ge\tau\}\)를 사용했다. mask 없이 active-neighbor cosine에 active-pair
fraction을 곱한 conditional-smoothness와 effective energy coverage를 계산했다. 합성
defect mask는 contrast-retention safety proxy에만 사용했으며 이 측정량에는 입력하지 않았다.

## dsba5 결과 (median)

| Object | Global coherence | Defect coherence | Global conditional | Defect conditional | Global coverage | Defect coverage | Local / broad retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fruit_jelly | .2747 | .0553 | .0975 | .2434 | .8125 | .4777 | .9027 / .8385 |
| walnuts | .2121 | .0689 | .0620 | .1961 | .9073 | .4976 | .9557 / .9838 |
| vial | .1618 | .1005 | .1426 | .2296 | .7747 | .5470 | .9289 / .8182 |
| fabric | .1397 | .0512 | .0877 | .1471 | .9381 | .5334 | .9782 / .9715 |

- global coherence와 energy coverage는 각각 `4/4`에서 defect aggregate보다 컸다.
- 그러나 global conditional-smoothness는 모든 object에서 defect aggregate보다 작았다
  (`0/4`; 사전등록 gate는 `>=3/4`). active-pair fraction도 모두 양수였으므로 이는
  invalid-row나 빈 active set 때문이 아니라, 활성된 defect displacement의 내부 연결성이
  현재 global transform의 활성 영역보다 더 높다는 관측이다.
- local/broad defect retention은 모든 object에서 양수였다. 따라서 safety failure가 아닌
  factorisation signal failure로 분류한다.

## 재현성 및 artifact 검증

- local compact artifact의 SHA-256 manifest는 전 항목 통과했다.
- dsba5 v2 compact artifact의 SHA-256 manifest도 전 항목 통과했다. 원격 `run_manifest.json`과
  preflight가 같은 runner/support hash를 기록한다.
- 두 lane의 ledger와 입력 hash가 동일했고, 11개 집계 metric 전부 사전 허용오차
  (absolute `1e-3` 또는 relative `1%`) 내였다. 최대 절대차는 vial의
  `geometry_nn_retention`에서 `.0009765625`였다.
- dsba5 v2 결과 회수와 checksum 검증 후 immutable source snapshot
  `/workspace/isolated_runs/global_local_coherence_conditional_20260807_v2`을 제거했다.

## 해석과 stop rule

이 측정은 처음 우려했던 평균 오염을 피하면서, global shift가 더 넓게 퍼진다는 coverage
signal을 확인했다. 하지만 coverage만으로는 global nuisance와 defect를 가르는 선택 기준이
될 수 없다. active displacement의 조건부 smoothness는 반대로 defect 쪽이 더 높았으므로,
현재 frozen representation과 transform family에서 coherent-global prior를 module 설계 근거로
승격하지 않는다.

다음 단계가 가능하려면 이 statistic의 threshold/aggregation을 다시 조정하는 것이 아니라,
독립적인 evidence source로 global/local premise를 다시 세워야 한다. 그 전까지는 Baseline1을
유지하고 기존 global canonicalization 및 새 Flow/Flow-Matching/Diffusion/DiT branch는 닫는다.

## Artifacts

- [Preregistration](preregistration.md)
- Runner: `scripts/run_global_local_conditional_smoothness_stage_minus1.py`
- Parity/gate adjudicator: `scripts/aggregate_global_local_conditional_smoothness_stage_minus1.py`
- Remote controller: `scripts/run_global_local_conditional_smoothness_stage_minus1_remote.sh`
- Compact verdict: `results/stage_minus1/global_local_conditional_smoothness/aggregate_v2/verdict.json`
