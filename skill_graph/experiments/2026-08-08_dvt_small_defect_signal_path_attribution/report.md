# DVT Small-Defect Signal-Path Attribution

Date: 2026-08-08  
Verdict: `LOSS_AT_DVT_SUBTRACTION`  
Scope: diagnostic-only, local only, SuperAD N/A, comparable=false

## 1. Negative Evidence Intake

이 실험은 "small local defect에서 변하지 않은 patch가 평균 smoothness를 부풀리는가"라는 질문을 검증하기 위한 신호 경로 진단이다.
핵심 가설은 DVT가 spatial signal을 제거하거나 재배치하는 시점에서 small defect coverage가 먼저 무너지고, Flow는 그 손실을 새로 만들지 않거나 되돌리지도 못한다는 것이다.

닫힌 branch를 다시 튜닝하는 실험은 아니다.
- support-admission이나 threshold sweep만으로 성능을 올리는 경로가 아니다.
- query metadata, GT, anomaly map, test-private, oracles는 사용하지 않는다.
- 이전에 이미 실패한 support / score reweighting 계열이 아니라, DVT 전후와 Flow 전후의 causal stage boundary를 분리해 보는 진단이다.

## 2. Motivation

기존 결과에서 medium 쪽 이득은 있어도 small defect coverage가 약해지는 패턴이 반복되어 왔다.
따라서 "DVT가 정확히 어디서 손실을 만들었는지"를 확인하지 않으면, Flow 개선이나 후속 모듈 탐색을 잘못된 위치에 붙일 위험이 크다.

이번 진단의 목적은 다음 두 질문을 분리하는 것이다.
- small defect loss는 DVT 이전 feature에서 이미 생기는가?
- Flow는 그 loss를 만들거나 복구하는가?

## 3. Implementable Design

- 대상: `target_dataset=MVTec AD2 single-image`
- object: all 8 objects
- support: fixed 16 train/good
- arms: `raw`, `position_current`
- ordered stages:
  1. `pre_dvt_feature`
  2. `post_dvt_feature`
  3. `flow_latent`
  4. `deployed_final`

설계 의도는 간단하다.
- DVT 전과 후를 같은 support custody 위에서 비교한다.
- Flow latent와 deployed final은 DVT 이후의 영향을 따로 분리해 본다.
- GT는 모든 stage map이 계산된 뒤에만 연다.

## 4. Evaluation Alignment

이 진단은 SuperAD 비교나 paper-level claim이 아니다.

- `SuperAD`: N/A
- `comparable`: false
- metric granularity: local diagnostic metric
- primary diagnostic:
  - `stage_macro_small_coverage_top1`
- secondary diagnostics:
  - `stage_macro_medium_pixel_ap`
  - `stage_macro_medium_coverage_top5`

해석 기준은 다음과 같이 사전 고정했다.
- small coverage loss가 `post_dvt_feature`에서 처음 나타나면 DVT-subtraction 쪽에 원인을 둔다.
- Flow latent와 deployed final이 거의 같으면 Flow는 loss의 원인도, 복구도 아니라고 본다.
- medium 쪽 이득이 DVT에서 먼저 생기고 Flow가 약하게만 만지면, medium benefit은 DVT 기여로 본다.

## 5. Code Modification / Creation

이번 진단을 위해 다음 실행 경로를 사용했다.
- runner: `scripts/run_dvt_small_defect_signal_path_attribution.py`
- aggregator: `scripts/aggregate_dvt_small_defect_signal_path_attribution.py`

런타임 수정은 다음 한 가지였다.
- 초기 시도에서 `evaluation.z`를 patch shape로 복원하려다 `RuntimeError: shape '[1, 42, 91]' is invalid for input of size 4892160`가 발생했다.
- 이후 patch-level score 복원 경로로 고쳐서 stage map lineage를 유지했고, 그 뒤 성공적으로 완료했다.

실패 시도 로그는 보존했다.
- `results/dvt_small_defect_signal_path_attribution_20260808/local/failed_attempt_1/controller.log`

## 6. Added Code Evaluation

로컬 검증은 다음으로 확인했다.
- runner / aggregator tests
- config sanity
- result schema sanity
- GT timing / custody gate

추가로 `anomaly_maps/`는 생성되지 않았고, 결과 루트에도 남아 있지 않다.

## 7. Remote Execution

이 실험은 로컬 diagnostic run이다.
- result root: `results/dvt_small_defect_signal_path_attribution_20260808/local`
- summary: `results/dvt_small_defect_signal_path_attribution_20260808/local/summary.json`
- success log: `results/dvt_small_defect_signal_path_attribution_20260808/local/controller.log`
- failure log: `results/dvt_small_defect_signal_path_attribution_20260808/local/failed_attempt_1/controller.log`

## 8. SuperAD Baseline and Unified Metrics

SuperAD baseline는 사용하지 않았다.
- `superad_metric`: N/A
- `method_metric`: N/A
- `delta_vs_superad`: N/A
- `comparable`: false

이 진단은 segmentation benchmark가 아니라 causal stage audit이다.

## 9. Results and Analysis

### Validity

`summary.json` 기준 validity는 모두 통과했다.
- `custody`: pass
- `pre_dvt_identity`: pass
- `final_reference_identity`: pass
- `tie_area`: pass
- `object_count_all8`: true

### Key stage metrics

`stage_macro_small_coverage_top1`
- `pre_dvt_feature`: `0.0`
- `post_dvt_feature`: `-0.011667226077776167`
- `flow_latent`: `-0.011532434176609057`
- `deployed_final`: `-0.011532434176609057`

`stage_macro_medium_pixel_ap`
- `pre_dvt_feature`: `0.0`
- `post_dvt_feature`: `0.07411705333351039`
- `flow_latent`: `0.07570566113324136`
- `deployed_final`: `0.07570566113324136`

`stage_macro_medium_coverage_top5`
- `pre_dvt_feature`: `0.0`
- `post_dvt_feature`: `0.07463964867298567`
- `flow_latent`: `0.07320727021568682`
- `deployed_final`: `0.07320727021568682`

### Intuitive causal reading

small defect 쪽에서는 손실이 `post_dvt_feature`에서 이미 나타난다.
- `pre_dvt_feature`는 0.0
- `post_dvt_feature`에서 즉시 음수로 전환
- `flow_latent`와 `deployed_final`은 같은 수준에서 유지

즉, small coverage 손실은 DVT subtraction boundary에서 시작하고, Flow는 그것을 만들지도 고치지도 않는다.

medium 쪽에서는 DVT가 먼저 이득을 만든다.
- `post_dvt_feature`에서 pixel AP와 coverage가 이미 양의 값으로 올라간다.
- `flow_latent`는 medium AP를 소폭 더 올리지만, coverage는 거의 같은 수준이다.

따라서 medium benefit은 DVT 기여가 크고, Flow는 2차 조정 정도로 보인다.

## 10. Conclusion

이 진단은 method claim이 아니라 failure-boundary map을 제공한다.

결론:
- small defect coverage loss는 DVT 이후 즉시 시작된다.
- Flow는 그 손실의 원인이 아니고, 복구도 하지 못한다.
- medium benefit은 DVT에서 이미 생기며, Flow는 이를 조금만 다듬는다.

Verdict: `LOSS_AT_DVT_SUBTRACTION`

남는 자산:
- DVT가 small/medium을 동시에 다루지 못하는 경계의 위치
- Flow가 후행 보정만 하고 있다는 점
- 다음 구조 탐색에서 DVT 이전 / 이후를 별도 분리해야 한다는 근거

## 11. Cleanup

- `anomaly_maps/`: 생성되지 않음
- 실패 시도 로그: 보존 완료
- result summary: 보존 완료

