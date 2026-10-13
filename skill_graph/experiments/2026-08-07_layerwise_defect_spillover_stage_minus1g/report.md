# Layerwise Defect Spillover Stage -1g

Date: 2026-08-07  
Verdict: `KILL_FOR_CLAIM / CONTINUE_LAYER_INTERFACE_DIAGNOSTIC`

## Negative Evidence Intake

Stage -1f의 spatial-consensus conditioner는 defect contamination을 줄이지 못해 `0/4`였다. 본
실험은 그 실패가 fused DINO feature의 layer mixing 때문인지, individual layer 자체의 non-local
표현 때문인지 분리하기 위해 수행한 measurement-only audit이다. 새 conditioner, layer selector,
score fusion, Flow 또는 transport는 구현하지 않았다.

## Motivation

이미지 공간에서는 조명 변화가 global하고 defect가 local하다는 inductive bias가 자연스럽다.
그러나 이 구분이 frozen representation에서도 유지되지 않으면 query-derived condition이 defect를
정상 shift로 흡수할 수 있다. 따라서 architecture capacity를 늘리기 전에 layer별 defect spillover와
condition contamination을 동시에 측정했다.

## Implementable Design

- frozen DINOv3 ViT-H/16+의 고정 layers `[7,15,23,31]`와 현재 `layer_norm_mean` fused map.
- fixed-16 normal support의 모든 source, objects `fruit_jelly/walnuts/vial/fabric`.
- clean, `U_unseen`, `U+D_local`, `U+D_broad`; defect displacement는 `H(U+D)-H(U)`.
- patch energy entropy, one-patch 이웃 밖 far spill, outside/core density, global-condition 대비
  defect-condition contamination, global alignment cosine을 고정 측정했다.
- synthetic mask는 측정 영역 정의에만 사용했고 model input이나 layer 선택에는 사용하지 않았다.

## Evaluation Alignment

동일 layer가 localization gate와 conditioner-feasibility gate를 각각 `>=3/4` objects에서
통과하는지를 사전등록했다. 이 단계는 support-only representation diagnostic이므로
`SuperAD/RN-FMLK=N/A`, segmentation metrics `N/A`, `comparable=false`다. public/private test,
ground truth, query adaptation은 접근하지 않았다.

## Code Modification / Creation

- runner: `scripts/run_layerwise_defect_spillover_stage_minus1g.py`
- gate/parity aggregator: `scripts/aggregate_layerwise_defect_spillover_stage_minus1g.py`
- dsba5 controller: `scripts/run_layerwise_defect_spillover_stage_minus1g_remote.sh`
- focused tests: `tests/test_layerwise_defect_spillover_stage_minus1g.py`

## Added Code Evaluation

Focused pytest `4 passed`, `py_compile`, controller `bash -n`, container smoke를 통과했다. Runner SHA256
`94c28e7e1c4eb3b60211b459ad1088239c1101de41b1ed217ebee24b0a520981`, support manifest SHA256
`9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`를 양 lane에서 고정했다.

## Remote Execution

local GPU 0과 dsba5 host GPU 3/container cuda:0에서 동일 코드를 실행했다. 두 lane은 exit 0,
validity pass였고 객체·layer별 metric parity는 절대 `1e-3` 또는 상대 `1%` 기준으로 mismatch 0이었다.

## SuperAD Baseline and Unified Metrics

성능 비교 단계가 아니므로 `superad_seg_AUROC_0.05=N/A`, `superad_seg_F1=N/A`,
`RN-FMLK=N/A`, `comparable=false`다. Primary evidence는 preregistered localization 및
conditioner-feasibility gate다.

## Results and Analysis

| Representation | Localization pass | Conditioner-feasibility pass | 결론 |
| --- | ---: | ---: | --- |
| layer 7 | 4/4 | 0/4 | 공간 localization은 유지되나 condition으로는 불안전 |
| layer 15 | 0/4 | 0/4 | 중단 |
| layer 23 | 0/4 | 0/4 | 중단 |
| layer 31 | 0/4 | 0/4 | 중단 |
| fused | control | control | layer 7보다 local spillover가 큼 |

Layer 7의 local far-spill은 `fruit_jelly/walnuts/vial/fabric =
.088/.082/.060/.067`, entropy는 `.650/.642/.643/.642`, outside/core density는
`.006/.005/.003/.003`이었다. 즉 이미지의 local defect가 초기 layer에서는 실제로 국소적인
energy 변화로 남고, fused representation으로 갈수록 이 신호가 퍼진다는 가설은 `4/4`에서
지지됐다.

그러나 layer 7의 local condition contamination은 `.214/.688/.281/.472`로 고정 `.15` gate를
모두 넘었다. 공간적으로 좁게 남는다는 사실은 spatial mean 등의 condition estimator가 defect에
불변이라는 뜻이 아니다. 특히 `walnuts/fabric`은 global shift condition norm이 작아 상대적인
defect contamination이 컸다. 따라서 layer 7을 바로 conditioner 또는 anomaly layer로 선택하는
결론은 허용되지 않는다.

## Continuation Assessment

엄격한 method/layer-selection claim은 기각한다. 다만 layer 7 localization이 4/4이고 fused control보다
명확히 국소적이므로, 사전등록대로 **conditioner 입력은 layer 7, 기존 anomaly scorer 입력은 fused**로
분리한 단일 support-only interface preflight만 허용한다. 이 후속은 layer 7 condition의 defect
contamination이 실제 aggregation interface에서 감소하는지를 확인해야 하며, layer sweep, learned
fusion, Flow Matching/Diffusion/DiT 또는 public benchmark로 확장해서는 안 된다.

## Conclusion

`KILL_FOR_CLAIM / CONTINUE_LAYER_INTERFACE_DIAGNOSTIC`. 초기 layer는 defect localization을
보존하지만 defect-invariant condition requirement는 만족하지 못했다. 핵심 병목은 표현의 공간적
locality 자체가 아니라, 국소 defect 성분이 condition aggregate에 들어오는 것을 막는 interface다.

## Post-Conclusion Storage Cleanup

양 lane 모두 compact JSON/TSV/log만 생성했고 `anomaly_maps/`는 생성하지 않았다. dsba5 immutable
source snapshot은 실행 후 제거했으며 compact result는 보존했다.

Artifacts:

- local: `results/stage_minus1/layerwise_defect_spillover/local_v1/mechanism/summary.json`
- dsba5: `results/remote_runs/dsba5/layerwise_spillover_stage_m1g_20260807_v1/mechanism/summary.json`
- parity: `passed=true`, mismatches `0`
