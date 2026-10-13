# Spatial-consensus Conditioner Stage -1f

Date: 2026-08-07  
Verdict: `KILL_FOR_CLAIM / NO_CONTINUE_SPATIAL_CONSENSUS`

## Negative Evidence Intake

Stage -1e의 naive global moment alignment는 `0/4`였고 defect-induced conditioner drift가
`.218--.807`이었다. 본 실험은 그 transport의 threshold나 scale을 재조정하지 않고, “global
photometric shift는 spatial region 간 합의되고 local defect는 minority region에 머문다”는
inductive bias로 condition estimator 자체를 분리 검증했다.

## Motivation

더 강한 Flow/Flow Matching/Diffusion/DiT를 붙이기 전에, query-only condition이 실제 shift와
정렬되면서 defect에는 불변인지 확인해야 한다. 그렇지 않으면 expressive transport가 defect를
정상분포로 흡수할 수 있다.

## Implementable Design

- C0: Stage -1e의 flattened-memory/query channel median negative control.
- C1: support position template를 뺀 residual의 전체-position median.
- C2: feature grid를 고정 `4x4` block으로 나누고, 20-step Weiszfeld consensus에 가장 가까운
  8 block의 coordinate median.
- objects: fixed-16 `fruit_jelly/walnuts/vial/fabric`, 4-fold 12/4.
- clean, source-disjoint `U`, `U+D_local`, `U+D_broad`; full feature grid, no sampling.
- frozen DINOv3-H+/16 feature와 raw 1-NN sentinel만 사용했다.

## Evaluation Alignment

Paired clean feature와 synthetic mask는 oracle alignment 및 contrast 평가에만 사용했다. C0/C1/C2
conditioner는 held clean, paired transform 또는 mask를 입력받지 않았다. 이는 support-only
condition-identifiability diagnostic이므로 SuperAD/RN-FMLK와 segmentation benchmark 비교는
`N/A`, `comparable=false`다.

## Code Modification / Creation

- runner: `scripts/run_spatial_consensus_conditioner_stage_minus1f.py`
- gate/parity aggregator: `scripts/aggregate_spatial_consensus_conditioner_stage_minus1f.py`
- dsba5 controller: `scripts/run_spatial_consensus_conditioner_stage_minus1f_remote.sh`
- focused tests: `tests/test_spatial_consensus_conditioner_stage_minus1f.py`

## Added Code Evaluation

`py_compile`, controller `bash -n`, container smoke, focused pytest `4 passed`를 확인했다. Smoke에서
16 blocks, retained 8, Weiszfeld 20/`1e-6`, full-grid/no-sampling, exact C0/C1 definition과 translation
geometry sanity를 확인했다.

## Remote Execution

local GPU 0과 dsba5 host GPU 3/container cuda:0에서 동일 source/support hash로 병렬 실행했다.
두 lane 모두 exit-0였고 test public/private/ground truth access는 전부 `false`였다.

## SuperAD Baseline and Unified Metrics

이 단계는 anomaly segmentation method claim이 아니므로 `superad_seg_AUROC_0.05=N/A`,
`superad_seg_F1=N/A`, `RN-FMLK=N/A`, `comparable=false`다. primary metric은 C2 oracle alignment와
defect contamination이며 downstream q995/contrast는 safety sentinel이다.

## Results and Analysis

validity와 local/dsba5 parity는 통과했지만 identifiability와 downstream pass는 모두 `0/4`였다.

| Object | C0 oracle cos | C2 oracle cos | C2 oracle rel. error | C2 local / broad drift | C2-vs-C0 local / broad drift reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| fruit_jelly | .9710 | .9545 | .5360 | .6499 / 1.0245 | .1256 / .3422 |
| walnuts | .9703 | .8982 | .5102 | 1.5536 / 3.7475 | -.0213 / .0086 |
| vial | .9214 | .8872 | .5230 | .8340 / 1.0548 | .1401 / .0788 |
| fabric | .9564 | .8675 | .6341 | 1.5393 / 3.9493 | -.1471 / -.0959 |

C0는 photometric oracle **방향** cosine만 보면 이미 `.921--.971`이었다. 그러나 Stage -1e에서
확인했듯 방향 정렬은 defect invariance나 anomaly-score 개선의 충분조건이 아니다. C2는 oracle
cosine을 높이지 못했고 relative error도 `.51--.63`이었다. 더 결정적으로 local/broad drift가
`.65--3.95`였으며 C0 대비 25% 감소를 동시에 만족한 object가 없었다. `walnuts/fabric`에서는
오히려 contamination이 커졌다.

Downstream tail absolute reduction은 `-.002329/.000420/-.000166/.000588`로 작고 부호가
혼재했다. 반면 contrast retention은 대체로 보존됐다. 즉 실패 원인은 translation geometry나
contrast collapse가 아니라 condition estimator가 defect와 shift를 분리하지 못한 데 있다.

## Continuation Assessment

엄격한 method claim: 지원하지 않는다.  
동일 spatial-consensus family의 다음 진단: 지원하지 않는다.

Block 크기, retained fraction, Weiszfeld iteration을 튜닝하는 것은 같은 실패 family의 retread다.
남는 별도 연구 질문은 architecture가 아니라 representation measurement다. 이미지 공간에서
local인 defect가 DINO fused feature의 몇 개 layer/position까지 퍼지는지 receptive-field spillover를
측정해야 한다. 이는 새 conditioner 구현과 분리된 measurement-only 실험이어야 한다.

## Conclusion

`KILL_FOR_CLAIM / NO_CONTINUE_SPATIAL_CONSENSUS`. “global shift는 spatially coherent하고 defect는
local하다”는 이미지 수준 prior만으로는 fused feature에서 defect-invariant condition을 만들 수
없었다. Flow Matching/Diffusion/DiT 확장은 계속 차단한다.

## Post-Conclusion Storage Cleanup

두 lane 모두 compact JSON/TSV/log만 생성했고 `anomaly_maps/`는 생성하지 않았다. live Python
process가 없는 것을 확인한 뒤 remote immutable source snapshot만 제거하고 compact remote result는
보존한다.

Artifacts:

- local: `results/stage_minus1/spatial_consensus_conditioner/local_v1/mechanism/summary.json`
- dsba5: `results/remote_runs/dsba5/spatial_consensus_stage_m1f_20260807_v1`
- parity: `passed=true`, mismatches `0`

