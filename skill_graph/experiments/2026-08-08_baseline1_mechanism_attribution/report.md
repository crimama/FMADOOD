# Baseline1 Mechanism Attribution 결과

Date: 2026-08-08  
Verdict: `KILL_FOR_FLOW_MAIN_MECHANISM / RETAIN_BACKBONE-REPRESENTATION_EXPLANATION`

## 1. Negative Evidence Intake

새 module을 제안한 실험이 아니라 유지 중인 Baseline1의 성능 원인을 분해한 진단이다.
따라서 passive feature-surgery closed branch를 재튜닝하지 않았고, metadata·query adaptation·GT
조건을 사용하지 않았다. 기존 DINOv2 실험에서 관측된 큰 Flow 효과가 DINOv3-H+에서도
재현되는지가 핵심 미해결 질문이었다.

## 2. Motivation

Baseline1이 강한 이유를 Flow로 전제하면 이후 Flow Matching/Diffusion/DiT 탐색의 출발점이
잘못될 수 있다. DINOv3-H+ representation, layer fusion, Flow의 독립 효과와 상호작용을
동일 support/scorer 조건에서 분리했다.

## 3. Implementable Design

MVTec AD2 single-image의 `fruit_jelly, walnuts, vial, fabric` 네 객체와 고정 SuperAD-selected
support-16을 사용했다. DINOv3-H+ `[7]`, `[31]`, normalized mean `[7,15,23,31]` 각각에서
identity/raw와 3-epoch affine-coupling Flow를 비교했다. DVT와 density는 꺼 Flow contrast를
깨끗하게 만들었고, scorer는 raw latent 1-NN으로 고정했다.

세부 계약과 gate는 [사전등록](preregistration.md)에 있다.

## 4. Evaluation Alignment

- 동일 dataset, 객체, support path, resolution, backbone, TESTpublic population을 six arms가 공유했다.
- 모든 arm은 support count 16, finite map, prediction/GT count 일치를 통과했다.
- reduced-object 기전 진단이므로 SuperAD 우월성이나 방법론 KEEP 판정에는 사용할 수 없다.
- 기존 K1 artifact의 같은 네 객체 값은 전체 Baseline1 경로의 context로만 사용한다.

## 5. Code Modification / Creation

- `scripts/run_baseline1_mechanism_attribution_arm.sh`
- `scripts/aggregate_baseline1_mechanism_attribution.py`
- `skill_graph/experiments/2026-08-08_baseline1_mechanism_attribution/preregistration.md`
- 이 보고서

## 6. Added Code Evaluation

- runner와 aggregator `py_compile`/`bash -n` 통과.
- aggregator가 6 arms × 4 objects, identical support basenames, support count 16,
  finite/shape coverage, zero TIFF output을 검증했다.
- Phase-3 scorer의 검증된 float16 weighted-histogram evaluator를 사용하고
  `FLOWTTE_PHASE3_VARIANTS=raw_1nn`으로 불필요한 8개 scorer를 제외했다.

## 7. Remote Execution

로컬 RTX 4090과 dsba5 고정 컨테이너/GPU 3을 병렬 사용했다. dsba5 lane은 CPU histogram
집계가 느려 로컬에서 동일 late/fused arms를 중복 완료했고, 최종 factorial은 완결된 로컬
artifact를 사용했다. dsba5 중복 프로세스는 종료했다. gated DINOv3 cache는 dsba5의 검증된
cache를 로컬 workspace에 복제해 환경을 정렬했다.

Compact result: `results/baseline1_mechanism_attribution_20260808/local/summary.json`.

## 8. Results and Analysis

### Macro 결과

| representation | raw pAUROC / F1 / AU-PRO | Flow pAUROC / F1 / AU-PRO | Flow - raw |
| --- | --- | --- | --- |
| layer 7 | `.601980 / .160280 / .417424` | `.602888 / .160297 / .420298` | `+.000908 / +.000017 / +.002875` |
| layer 31 | `.860433 / .641497 / .432586` | `.859160 / .637743 / .431477` | `-.001273 / -.003754 / -.001110` |
| fused | `.808300 / .505523 / .562370` | `.810399 / .507866 / .558749` | `+.002099 / +.002343 / -.003621` |

핵심 결과는 세 가지다.

1. **Flow가 주된 성능 엔진이라는 가설은 DINOv3-H+에서 지지되지 않았다.** fused pAUROC
   이득은 `+.002099`로 사전등록 `+.005`에 미달했고, AU-PRO는 감소했다. layer 31에서는
   세 지표가 모두 하락했다.
2. **Flow×fusion synergy도 지지되지 않았다.** pAUROC interaction은 `+.002281`, positive
   object는 gate `3/4`를 충족하지 못했다.
3. **강한 표현이 핵심이라는 설명은 남는다.** raw fused는 raw single-layer 평균보다
   pAUROC `+.077093`, F1 `+.104634`, AU-PRO `+.137365`였다. 그러나 layer 31 raw 단독은
   fused raw보다 pAUROC `+.052133`, F1 `+.135974` 높았다. 즉 “fusion이 무조건 최적”이
   아니라, early layer 7의 약한 anomaly ranking과 late layer 31의 강한 ranking 사이에서
   fused representation이 AU-PRO/localization을 보완하는 절충이다.

### fused Flow의 객체별 변화

| object | Δ pAUROC | Δ F1 | Δ AU-PRO |
| --- | ---: | ---: | ---: |
| fruit_jelly | +.006179 | +.010592 | +.006565 |
| walnuts | -.005355 | -.020759 | -.030244 |
| vial | +.003865 | +.003527 | +.001957 |
| fabric | +.003706 | +.016015 | +.007237 |

Flow는 3/4 객체에서 작은 pAUROC 이득을 보였지만 `walnuts` 손실이 macro와 region metric을
상쇄했다. 이는 Flow가 보편적 defect separator라기보다 representation geometry를 약하게
재배치하며 객체에 따라 이득/손실을 만드는 보조 변환이라는 해석과 맞는다.

같은 네 객체의 기존 Baseline1 K1 context는 pAUROC `.848651`, F1 `.584089`, AU-PRO
`.570928`이다. 이번 fused Flow core보다 `+.038252 / +.076223 / +.012179` 높다. 이 차이는
DVT를 포함한 full path에 남아 있으나 density/calibration까지 얽혀 있으므로 이 실험만으로
DVT 단독 기여라고 부르지는 않는다.

## 9. Conclusion and Cleanup

Strict Flow mechanism claim은 `KILL_FOR_MECHANISM_CLAIM`이다. 현재 가장 타당한 설명은:

> Baseline1의 강함은 DINOv3-H+ late/fused representation에서 먼저 나오며, Flow는 그 위에
> 작고 객체 의존적인 변화를 더한다. 전체 Baseline1과 core의 남은 차이는 DVT/scoring
> 경로에 있으므로 다음 귀인 대상은 Flow 구조 대체가 아니라 DVT와 representation의
> 상호작용이다.

다음 단일 진단은 동일 네 객체에서 `fused raw/Flow × DVT off/on` 2×2를 수행해 DVT main
effect와 DVT×Flow interaction을 분리하는 것이다. DVT main effect가 pAUROC `+.005`를
넘지 못하거나 F1/AU-PRO 손실이 반복되면 DVT를 Baseline1의 일반적 원인으로도 승격하지 않는다.

완료 후 local/remote result root의 regenerable `anomaly_maps/`를 모두 제거했고 compact JSON,
logs, preregistration만 보존했다.
