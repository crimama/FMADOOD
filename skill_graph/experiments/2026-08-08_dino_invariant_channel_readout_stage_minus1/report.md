# DINO Invariant Channel Readout Stage -1

Date: 2026-08-08  
Verdict: `ANOMALY_ABSORPTION`  
Scope: diagnostic-only, remote dsba5 execution, SuperAD N/A, `comparable=false`

## 1. Negative Evidence Intake

이 실험은 DINO backbone 이후에 별도 장치를 두어 representation 자체를 shift invariant 하게 읽을 수 있는지 확인하려는 진단이다.

닫힌 branch를 다시 돌린 것은 아니다.
- query metadata, GT, test-private, anomaly maps는 사용하지 않았다.
- support admission을 학습하거나 파라미터를 업데이트하는 방식도 아니다.
- 목표는 frozen DINO feature에서 shift-sensitive channel을 덜 읽는 fixed readout이 실제로 계산 가능하고, 그 결과가 local defect contrast를 보존하는지 확인하는 것이었다.

이번 보고서의 scientific result는 v2만 기준으로 삼는다.
- v1은 naive float64 `QxMxC` evaluator가 병목으로 정지되었고, 그 결과는 과학적 결론으로 쓰지 않는다.
- v2는 exact GPU chunked evaluator와 equivalence tests를 통과한 뒤 완료되었고, 그 결과만 해석한다.
- extraction batching은 v2 snapshot에 포함됐지만, fake-backbone order/shape/fallback tests와 serial-compatible
  grouping을 통과한 운영성 최적화다. 실험 semantics, folds, masks, metrics는 바꾸지 않는다.

## 2. Motivation

사용자가 확인하려던 질문은 다음이다.
- backbone 이후에 target-specific module이 왜 필요한가?
- 그 module은 어떤 inductive bias를 가져야 하는가?
- DINO representation 자체를 shift invariant 하게 읽는 방법이 가능한가?

이 실험은 학습 가능한 module을 바로 얹기 전에, fixed readout만으로도 shift sensitivity를 줄일 수 있는지 보는 최소 진단이다.

## 3. Implementable Design

- 대상: `target_dataset=MVTec AD2 single-image`
- 객체: `fruit_jelly`, `walnuts`, `vial`, `fabric`
- support: fold-local fixed 16 normal `train/good`
- fold scheme: 4 folds, each `12 fit / 4 held`
- backbone: frozen DINOv3 ViT-H/16+, layers `[7, 15, 23, 31]`
- feature fusion: layer-wise token L2 normalization 후 mean fusion
- arms:
  - `I0_identity`
  - `I1_invariant_mask`
  - `I2_score_shuffled_mask`

핵심 구조는 다음과 같다.
- fit support의 photometric shift sensitivity를 channel-wise로 측정한다.
- sensitivity가 낮은 하위 50% channels만 선택한다.
- 동일 cardinality의 score-shuffled mask를 hard null로 둔다.
- clean/held-shift/held-defect pair에서 q995 inflation, retention, geometry retention을 본다.

이 설계는 support-only diagnostic이다.
- Flow, DVT, query adaptation, score fusion, learned adapter, parameter update는 모두 off다.
- test_public, test_private, GT, condition label, filename/split metadata 접근은 모두 금지다.

## 4. Evaluation Alignment

이 실험은 method claim이 아니라 support-only diagnostic이다.

- `SuperAD`: `N/A`
- `benchmark_comparable`: `false`
- `superad_metric`: `N/A`
- `method_metric`: `N/A`
- `delta_vs_superad`: `N/A`

평가 축은 사전등록된 진단 축에 맞췄다.
- primary: displacement ratio, q995 relative reduction vs `I0`, q995 reduction margin vs `I2`
- secondary: local / broad defect retention, clean pair geometry Spearman, nearest-neighbor identity retention

strict gate는 존재하지만, 이 실험은 benchmark claim이 아니라 mechanism diagnostic으로 해석한다.

## 5. Code Modification / Creation

이번 진단을 위해 다음 파일을 사용했다.
- runner: `scripts/run_dino_invariant_channel_readout_stage_minus1.py`
- remote controller: `scripts/run_dino_invariant_channel_readout_stage_minus1_remote.sh`
- aggregator: `scripts/aggregate_dino_invariant_channel_readout_stage_minus1.py`
- readout utility: `src/flow_tte/invariant_readout.py`

v1에서 확인된 실패는 분명했다.
- naive float64 `QxMxC` exact evaluator가 fold/object loop에서 진행 정체를 일으켰다.

v2에서는 이를 다음처럼 고쳤다.
- exact GPU chunked evaluation으로 교체
- equivalence tests 추가
- immutable script snapshot을 기준으로 실행

실행 시점의 핵심 해시 증거:
- runner sha256: `5eb53f7b776c84974effae2a45d0d79466934d9cc5352ed3f9b2865c17e7c4a1`
- readout sha256: `32473a09293685373dafabeb55afda7e4978228e1c1c7604c3d7faf5396ba664`
- aggregate sha256: `6613b7a84aa81d8abcaefa04d077527eac8fb3f7321d8be6dc4af2ca78423f5a`

## 6. Added Code Evaluation

preflight는 통과했다.
- support manifest SHA 일치: `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`
- smoke check: `true_cardinality=2`, `shuffled_cardinality=2`, `scale_equal=true`, `finite=true`
- access boundary 유지: test_public/test_private/GT 미열람

추가로 확인된 메타 증거:
- `status=passed`
- `host_gpu_id=3`
- `container_cuda_slot=0`
- remote result root는 완료 상태였다

## 7. Remote Execution

실행 정보:
- host: dsba5
- fixed container: `hun_fsad_tta_012`
- host GPU: `3`
- in-container CUDA slot: `0`
- remote result root:
  `results/remote_runs/dsba5/dino_invariant_channel_readout_stage_m1_20260808_v2`

실행 산출물:
- `controller.log`
- `mechanism/progress.jsonl`
- `mechanism/aggregate.json`
- `mechanism/summary.json`
- `mechanism/object_arm_metrics.tsv`
- `mechanism/access_ledger.json`
- `preflight.txt`
- `preflight_smoke.log`
- `preflight_dinov3.log`

## 8. SuperAD Baseline and Unified Metrics

SuperAD baseline는 사용하지 않았다.

- `superad_metric`: `N/A`
- `method_metric`: `N/A`
- `delta_vs_superad`: `N/A`
- `comparable`: `false`

이 실험은 비교 benchmark가 아니라 mechanism-only diagnostic이다.

## 9. Results and Analysis

### 9.1 Invariance signal

fixed lower-50% channel readout은 shift 자체는 분명히 줄였다.

- `I1` displacement ratio object median: `0.6078159848726381`
- `I2` displacement ratio object median: `1.0051765088837898`

즉, invariant mask는 shuffled null보다 더 강하게 shift를 눌렀다. Invariance specificity는 4/4 objects에서 성립했고, shuffled 대비 선택성도 통과했다.

### 9.2 Relevance / preservation failure

그러나 shift를 줄인 대가로 anomaly/geometry 정보도 함께 깎였다.

- raw `1-NN` relevance: `1/4` objects만 통과
- `local_retention_object_mean`: `0.49408040169096346`
- `broad_retention_object_mean`: `0.5948539241492397`
- gate: 둘 다 `>=0.95`
- `geometry_nn_retention_object_median`: `0.671875`
- `geometry_pair_spearman_object_median`: `0.803851634131681`
- gate: 둘 다 `>=0.90`

즉, fixed lower-50% channel rejection은 shift-sensitive channels를 찾는 데는 성공했지만, 그 channels가 defect/structure 정보와 상당히 겹쳐 있었다. representation을 안정화시키는 대신 anomaly signal까지 흡수해 버린 셈이다.

### 9.3 q995 behavior

`q995` 쪽은 부분적으로 좋아졌다.

- `q995_relative_reduction_vs_i0_object_median`: `0.17036357861236612`
- `q995_reduction_margin_vs_i2_object_median`: `-0.01311359741758416`

즉, identity 대비는 줄었지만 shuffled null을 이기지는 못했다. 따라서 이 신호는 "구조적 개선"보다는 "shift 억제에 따른 부분적 수축"에 가깝다.

### 9.4 Interpretation

핵심 해석은 다음과 같다.

1. fixed readout 자체는 shift invariant 방향의 signal을 실제로 추출할 수 있다.
2. 그러나 이 signal은 defect/structure 정보와 분리되어 있지 않다.
3. 그래서 raw 1-NN relevance, retention, geometry가 함께 무너졌다.

따라서 이 후보는 "좋은 shift-invariant representation extractor"라기보다, "shift-sensitive channels를 덜 읽으면 움직임은 줄지만 anomaly 정보도 같이 잃는 구조"로 보는 것이 맞다.

이 결과는 trainable diagonal gate로 바로 넘어가라는 근거가 아니다.
오히려 현재 mask가 defect/structure 정보까지 먹고 있다는 경계 증거이므로, 같은 방향의 파라미터화는 이 후보 위에서 진행하지 않는다.

### 9.5 v2 result boundary

v2는 과학적 결과로 해석 가능한 완결된 run이다.
- exact GPU chunked evaluator
- equivalence tests
- immutable script hash
- support-only / no test / no GT
- `benchmark_comparable=false`
- `SuperAD=N/A`
- `anomaly_maps` 없음

extraction batching은 실행 속도 최적화이며, 여기 적는 scientific claim은 v2의 fixed arms, masks,
folds, and metrics에만 걸린다.

## 10. Conclusion

이 실험은 strict claim을 지지하지 않는다.

Verdict: `ANOMALY_ABSORPTION`

남는 자산은 분명하다.
- DINO 뒤 fixed readout으로 shift-sensitive channels를 약화시킬 수 있다는 점
- 그러나 그 채널이 defect/structure 정보와 겹치면 anomaly까지 같이 흡수된다는 점
- 다음 설계에서 필요한 것은 단순 channel rejection이 아니라 shift factor와 defect factor의 분리다

현재 후보에 대해서는 trainable diagonal gate로 진행하지 않는다.
다음 단계가 필요하다면, "shift를 줄이는 구조"가 아니라 "shift와 defect를 분리하는 구조"를 새로 설계해야 한다.

## 11. Cleanup

- `anomaly_maps/`: 생성되지 않음
- dense map tree: 없음
- compact evidence retained:
  - `mechanism/summary.json`
  - `mechanism/aggregate.json`
  - `mechanism/object_arm_metrics.tsv`
  - `mechanism/access_ledger.json`
  - `controller.log`
  - `preflight.txt`
  - `preflight_smoke.log`
