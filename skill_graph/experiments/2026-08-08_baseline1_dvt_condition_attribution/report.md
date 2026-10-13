# Baseline1 DVT 조건별 기여 분해 결과

Date: 2026-08-08
Protocol: `baseline1-dvt-condition-attribution-20260808`
Verdict: `DEFECT_EVIDENCE_GAIN` (reduced-object diagnostic)

## 1. 질문과 결론

DVT의 큰 성능 기여는 **distribution shift가 있는 정상 이미지의 오탐을 주로 제거해서가
아니라, 이미 존재하는 결함 영역의 pixel-level evidence를 더 강하게 만드는 효과**로
설명하는 편이 현재 결과에 더 잘 맞는다.

핵심 근거는 다음과 같다.

- defect-pixel TPR은 regular `+9.62%p`, photometric `+9.88%p`, geometric
  `+9.58%p`로 세 condition에서 거의 동일하게 증가했다.
- shifted-normal FPR은 photometric `-0.030%p`, geometric `-0.058%p`로 감소했지만
  절대 효과가 작고, 엄격한 정상-shift 억제 패턴은 `1/4` 객체에서만 재현됐다.
- shifted GT component recall은 macro `-1.12%p`였다. 즉 더 많은 결함 instance를 새로
  찾았다기보다, 이미 검출 가능한 결함 내부의 pixel coverage/ranking을 강화한 것으로
  해석하는 것이 안전하다.
- defect TPR 개선은 `fruit_jelly`, `vial`, `fabric`의 `3/4`에서 양수였고,
  `walnuts`는 `-0.60%p`였다. 특히 `fabric`의 `+32.74%p`가 macro 효과를 크게 이끌었다.

따라서 현재 Baseline1 논리는 `DVT = shift correction module`이 아니라
`DVT = object-aligned spatial prior that strengthens defect localization`으로 두는 편이
증거에 맞다.

## 2. 정보 사용 경계

- condition token은 anomaly map 생성 이후 평가 record를 그룹화할 때만 사용했다.
- backbone, DVT, Flow, memory, 1-NN score, threshold 선택에는 condition/file metadata를
  넣지 않았다.
- condition마다 threshold를 재최적화하지 않았다. 각 object/arm 전체 TESTpublic에서
  얻은 하나의 pooled oracle threshold를 세 condition에 동일 적용했다.
- DVT off/on은 동일 support-16, DINOv3 layer, Flow와 scorer 설정을 사용했다.

이는 metadata 기반 추론 실험이 아니라, 이미 생성된 score의 사후 귀인 분석이다.

## 3. 실행 구성

- objects: `fruit_jelly, walnuts, vial, fabric`
- arms: `fused_flow` vs `dvt_flow`
- DVT: position mean, alpha `1.0`
- Flow: 3 epochs, affine coupling, raw latent 1-NN, density weight `0`
- conditions: `regular`, `photometric(over/underexposed)`, `geometric(shift_*)`
- local: `hun_fsad_tta_local_stage1a0`, RTX 4090
- remote: dsba5 `hun_fsad_tta_012`, host GPU3/container CUDA0
- anomaly-map TIFF: `0`; in-memory float16 histogram evaluation
- local/dsba5 parity: 128개 핵심 수치의 worst absolute delta
  `2.1802e-6` (`<1e-5`, pass)

## 4. Macro condition contrast

모든 값은 `DVT on - DVT off`이다.

| Condition | good FPR | good p99.9 / threshold | defect-pixel TPR | component recall |
| --- | ---: | ---: | ---: | ---: |
| regular | -0.000189 | -0.00606 | +0.09616 | -0.01432 |
| photometric | -0.000302 | -0.00852 | +0.09878 | -0.00801 |
| geometric | -0.000575 | -0.01875 | +0.09577 | -0.01444 |

정상 score tail은 shifted condition에서 regular보다 더 줄어드는 경향이 있다. 그러나
그 크기는 작고 객체별 반복성이 부족하다. 반면 defect-pixel TPR 증가는 condition 종류와
무관하게 약 `+9.6~9.9%p`로 유지된다. 이 대칭성이 DVT를 shift-specific corrector보다
condition-agnostic localization prior로 보는 핵심 이유다.

## 5. 객체별 shifted-condition 평균

photometric과 geometric의 단순 평균이다.

| Object | Δ good FPR | Δ tail ratio | Δ defect TPR | Δ component recall | 해석 |
| --- | ---: | ---: | ---: | ---: | --- |
| fruit_jelly | -0.000371 | -0.00654 | +0.02233 | -0.00439 | 정상 tail과 defect TPR 동시 개선 |
| walnuts | -0.000267 | +0.00360 | -0.00605 | +0.00758 | pixel-level DVT 이득 미재현 |
| vial | +0.000298 | -0.03159 | +0.04536 | +0.01961 | defect 개선, FPR은 소폭 악화 |
| fabric | -0.001413 | -0.02001 | +0.32745 | -0.06771 | 큰 pixel coverage 개선, component recall 저하 |

엄격한 normal-shift FP suppression은 `fruit_jelly`만 통과했다. Defect-pixel TPR 양수는
`fruit_jelly`, `vial`, `fabric`의 `3/4`였다.

## 6. 직관적 해석

DVT는 support normal patch의 **같은 위치 평균**을 빼는 구조다. 이 연산은 조명이나
카메라 shift를 명시적으로 추정하지 않는다. 대신 객체가 정렬되어 있을 때 각 위치에서
반복되는 정상 구조를 제거하고, 그 위치의 정상 패턴과 다른 잔차를 두드러지게 만든다.

이번 결과는 바로 이 설명과 맞는다.

1. shift 종류가 바뀌어도 defect TPR 증가량이 거의 같다.
2. 정상 shifted image의 오탐 감소는 존재하지만 작고 객체 의존적이다.
3. component recall이 늘지 않으면서 defect pixel TPR이 늘었다. DVT가 새로운 defect를
   발견하는 것보다 이미 잡힌 defect의 내부 픽셀을 더 넓고 강하게 점수화한다는 뜻이다.

따라서 Baseline1의 강점은 현재 증거상
`DINOv3 representation → DVT object-position prior → small Flow interaction` 순서로
설명하는 것이 가장 자연스럽다.

## 7. Gate와 주장 한계

- `normal_shift_support_object_count = 1/4`
- `defect_gain_support_object_count = 3/4`
- verdict: `DEFECT_EVIDENCE_GAIN`

다만 이 결과는 reduced-object의 threshold-conditioned 사후 진단이다. DVT의 완전한
인과 메커니즘, 전체 8-object 일반화 또는 Flow/DiT/Diffusion 구조 선택을 단독으로
확정하지 않는다. 특히 macro defect TPR은 `fabric` 영향이 크므로 다음 검증에서는
all-8 object와 threshold-free defect ranking/coverage 지표로 확인해야 한다.

## 8. Artifacts

- preregistration: `skill_graph/experiments/2026-08-08_baseline1_dvt_condition_attribution/preregistration.md`
- local summary: `results/baseline1_dvt_condition_attribution_20260808/local/summary.json`
- local object JSON: `results/baseline1_dvt_condition_attribution_20260808/local/{fused_flow,dvt_flow}/objects/*.json`
- remote compact result:
  `results/remote_runs/dsba5/baseline1_dvt_condition_attribution_20260808_dsba5_v2_combined/`
- parity: `results/baseline1_dvt_condition_attribution_20260808/parity.json`
