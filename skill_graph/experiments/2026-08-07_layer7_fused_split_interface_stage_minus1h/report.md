# Layer-7 / Fused Split Interface Stage -1h

Date: 2026-08-07  
Verdict: `KILL_FOR_CLAIM / NO_CONTINUE_SPLIT_INTERFACE`

## Negative Evidence Intake

Stage -1f의 fused spatial consensus는 `0/4`였고, Stage -1g는 layer 7에서 local-defect
localization만 `4/4`, conditioner feasibility는 `0/4`임을 보였다. 본 실험은 그 제한적 신호를
사용해 layer 7을 mask-free patch gate로만 쓰고 condition vector와 anomaly score는 fused feature에
남기는 단일 split interface를 검증했다. Layer, retained fraction, transport를 sweep하지 않았다.

## Motivation

초기 layer가 local defect를 공간적으로 보존한다면 defect 영향을 받은 patch를 condition aggregate에서
제외할 수 있어야 한다. 이 효과가 실제 defect contamination 감소와 shifted-normal tail 개선으로
이어지는지를 직접 검증했다.

## Implementable Design

- `C0_fused_all`: 모든 fused residual patch의 channel median condition.
- `C7_gate_fused`: layer-7 residual의 geometric median과 cosine distance가 가까운
  `round(.75P)` patch를 고정 선택하고, 같은 patch의 fused residual median을 condition으로 사용.
- fused clean support memory에 condition location vector만 더한 뒤 fused raw 1-NN score 계산.
- fixed-16 support, 4-fold 12/4 source-disjoint, objects
  `fruit_jelly/walnuts/vial/fabric`.
- clean, unseen photometric `U`, `U+D_local`, `U+D_broad`; paired clean과 masks는 평가에만 사용.

## Evaluation Alignment

Primary gate는 C7의 C0 대비 local/broad conditioner-drift 25% 감소와 동시에 oracle alignment,
normal-tail reduction, defect contrast, translation geometry를 보존하는지다. Support-only interface
diagnostic이므로 `SuperAD/RN-FMLK=N/A`, segmentation metrics `N/A`, `comparable=false`다.

## Code Modification / Creation

- runner: `scripts/run_layer7_fused_split_interface_stage_minus1h.py`
- gate/parity aggregator: `scripts/aggregate_layer7_fused_split_interface_stage_minus1h.py`
- dsba5 controller: `scripts/run_layer7_fused_split_interface_stage_minus1h_remote.sh`
- focused tests: `tests/test_layer7_fused_split_interface_stage_minus1h.py`

## Added Code Evaluation

Focused pytest `7 passed`, `py_compile`, controller `bash -n`, container smoke를 통과했다. Runner SHA256은
`36f78eeac055b2202c943bf1fa81c2ea6836cdc81143b2344fd05d79090a146a`, 최종 aggregator SHA256은
`10b55aa5c704719f6aac65f9453669e5b26e587d510570a79de78aa21a048218`이다.

초기 집계에서 `vial` patch 수가 4의 배수가 아니어서 runner의 `round(.75P)`와 aggregator의 exact
`.75` validity 검사가 충돌했다. 과학 metric과 gate를 변경하지 않고, actual count/fraction consistency를
검사하도록 사전등록과 aggregator를 보정했다. 저장된 raw summaries를 재집계한 뒤 validity `4/4`를
확인했다.

## Remote Execution

Local GPU 0과 dsba5 host GPU 3/container cuda:0에서 동일 runner/support hash로 실행했다. 두 lane은
exit 0이며, 수정 aggregator로 양 raw summary를 재집계했다. Local/dsba5 metric parity는 mismatch 0,
절대 `1e-3` 또는 상대 `1%` 기준을 통과했다.

## SuperAD Baseline and Unified Metrics

Method/segmentation 성능 실험이 아니므로 `superad_seg_AUROC_0.05=N/A`, `superad_seg_F1=N/A`,
`RN-FMLK=N/A`, `comparable=false`다. Strict interface pass와 weak continuation pass가 각각 primary
판정이다.

## Results and Analysis

| Object | C0→C7 local drift | local 감소율 | C0→C7 broad drift | broad 감소율 | C7 tail reduction | C7−C0 tail margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fruit_jelly | .592→.548 | 6.8% | 1.115→1.035 | 7.2% | -204.3% | -26.9%p |
| walnuts | 1.309→1.359 | -0.2% | 3.415→3.353 | .4% | 12.1% | -3.4%p |
| vial | .677→.637 | 5.4% | .856→.830 | 3.1% | -143.0% | -54.1%p |
| fabric | 1.320→1.022 | 21.8% | 3.475→2.895 | 13.6% | -11.4% | +34.9%p |

Strict interface pass와 weak continuation pass는 모두 `0/4`였다. C7은 `fruit_jelly/vial/fabric`에서
local drift를 일부 낮췄지만 preregistered 25%에 못 미쳤고 broad drift 감소는 더 작았다.
`walnuts` local drift는 오히려 증가했다. Oracle cosine은 `.893--.966`으로 방향은 대체로 맞았지만
relative error가 `fruit_jelly/vial/fabric = .558/.684/.750`으로 컸다.

Defect contrast는 `fruit_jelly/walnuts/vial`에서 보존됐지만 `fabric` broad retention이 `.912`로
`.95` gate를 실패했다. 더 결정적으로 shifted-normal tail은 `walnuts`의 작은 `12.1%` 개선 외에는
악화됐고 C0보다 일관되게 낫지 않았다. 즉 layer-7 locality는 어떤 patch가 local하게 변했는지는
보여주지만, support-template residual에서 domain shift와 object/position variation을 분리하는
reliable gate로 전환되지 않았다.

## Continuation Assessment

Strict method claim은 지원하지 않는다. Weak continuation도 `0/4`이므로 retained fraction, distance,
geometric-median iteration, layer 조합을 튜닝하지 않는다. 이 결과로 Flow/Flow Matching/Diffusion/DiT
또는 public benchmark를 여는 것도 허용하지 않는다. Layer-separated spatial gating branch는 여기서
종료한다.

## Conclusion

`KILL_FOR_CLAIM / NO_CONTINUE_SPLIT_INTERFACE`. Layer 7의 공간적 defect localization은 재현 가능한
representation insight지만, 그것만으로 defect-invariant target condition을 식별하지 못했다. 현재
병목은 layer mixing보다 **normal domain shift를 정상 support만으로 식별할 관측 정보의 부족**에 더
가깝다.

## Post-Conclusion Storage Cleanup

두 lane 모두 compact JSON/TSV/log만 생성했고 `anomaly_maps/`는 생성하지 않았다. dsba5 immutable
source snapshot은 제거했고 compact result는 보존했다.

Artifacts:

- local: `results/stage_minus1/layer7_fused_split_interface/local_v1/mechanism/summary.json`
- dsba5: `results/remote_runs/dsba5/layer7_fused_split_stage_m1h_20260807_v1/mechanism/summary.json`
- parity: `passed=true`, mismatches `0`
