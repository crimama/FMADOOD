# Baseline1 DVT × Flow Factorial 결과

Date: 2026-08-08  
Verdict: `PROMISING_DIAGNOSTIC / DVT_MECHANISM_SUPPORTED`

## 1. Negative Evidence Intake

새 모듈이나 DVT hyperparameter tuning이 아니다. 직전 DINOv3-H+ factorial에서 Flow main effect가
작았기 때문에, 전체 Baseline1과 core 사이의 남은 차이를 기존 DVT on/off로만 분해했다.
metadata, query adaptation, support admission, scorer sweep은 사용하지 않았다.

## 2. Motivation

Baseline1의 강함이 pretrained representation만으로 설명되는지, DVT position prior가 실제
ranking을 크게 바꾸는지, 그리고 DVT가 Flow와 결합될 때 추가 상호작용이 있는지 확인한다.

## 3. Implementable Design

동일 네 객체(`fruit_jelly, walnuts, vial, fabric`), fixed support-16, DINOv3-H+ fused
`[7,15,23,31]`, raw latent 1-NN에서 `DVT off/on × identity/Flow` 2×2를 구성했다.
Density는 0이며 DVT는 position mean alpha 1.0이다. 상세 gate는 [사전등록](preregistration.md)에 있다.

## 4. Evaluation Alignment

네 arm은 dataset, object/test population, support basenames, backbone, resolution, fusion, scorer를
공유했다. 모든 객체에서 support count 16, finite output, prediction/GT count, zero TIFF output을
통과했다. Reduced-object 결과이므로 SuperAD 우월성이나 method KEEP 주장은 하지 않는다.

## 5. Code Modification / Creation

- 기존 arm runner에 optional DVT mode를 추가했다.
- `scripts/aggregate_baseline1_dvt_flow_factorial.py`를 추가했다.
- 이 실험의 preregistration/report와 compact summary를 생성했다.

## 6. Added Code Evaluation

- runner `bash -n`, aggregator `py_compile` 통과.
- aggregator 재실행 시 동일 gate/summary를 재현했다.
- 로컬과 dsba5의 `DVT+Flow`를 독립 실행했고 세 지표의 최대 절대 차이는
  `2.18e-6`으로 parity tolerance `1e-5`를 통과했다.

## 7. Remote Execution

로컬 RTX 4090에서 `DVT+raw`, dsba5 고정 컨테이너 host GPU 3에서 `DVT+Flow`를 병렬 실행했다.
dsba5의 오래된 Phase-3 경로가 불필요한 9 variants를 여는 것을 감지해 그 PID만 중단하고,
동기화된 `/workspace/fsad_tta`에서 `raw_1nn` 단일 variant로 재실행했다. `DVT+Flow`는 로컬에서도
중복 완료해 parity를 검증했다.

## 8. Results and Analysis

### 2×2 macro

| arm | pAUROC | F1 | AU-PRO |
| --- | ---: | ---: | ---: |
| DVT off + raw | .808300 | .505523 | .562370 |
| DVT off + Flow | .810399 | .507866 | .558749 |
| DVT on + raw | .839836 | .570710 | .569636 |
| DVT on + Flow | .848213 | .581588 | .570856 |

### 사전등록 contrasts

| contrast | Δ pAUROC | Δ F1 | Δ AU-PRO |
| --- | ---: | ---: | ---: |
| DVT effect, Flow 없음 | +.031536 | +.065187 | +.007266 |
| DVT effect, Flow 있음 | +.037814 | +.073722 | +.012107 |
| DVT main effect | **+.034675** | **+.069455** | **+.009686** |
| DVT×Flow interaction | **+.006278** | **+.008535** | +.004841 |

DVT main effect는 pAUROC/F1/AU-PRO를 동시에 개선했고 pAUROC positive object `3/4`, collapse 없음으로
gate를 통과했다. Interaction도 pAUROC `+.006278`, positive `3/4`로 통과했다.

### DVT main effect의 객체별 비대칭

| object | Δ pAUROC | Δ F1 | Δ AU-PRO |
| --- | ---: | ---: | ---: |
| fabric | +.122908 | +.253761 | +.043573 |
| fruit_jelly | +.006656 | +.005666 | -.002300 |
| vial | +.014545 | +.021718 | +.003723 |
| walnuts | -.005408 | -.003326 | -.006250 |

효과 크기는 `fabric`이 주도하지만 pAUROC 부호는 `fruit_jelly`, `vial`에서도 양수다. 반대로
`walnuts`에서는 일관되게 작게 악화한다. 따라서 DVT는 보편적인 shift correction이 아니라,
공간 위치가 정상 구조와 defect ranking에 유용한 객체에서 강하게 작동하는 position prior다.

중요하게도 `DVT+Flow` `.848213/.581588/.570856`은 기존 full Baseline1 K1의 동일 네 객체
`.848651/.584089/.570928`과 각각 `-.000438/-.002501/-.000073` 차이다. 즉 이번 조건에서는
전체 Baseline1과 DVT+Flow core의 gap이 거의 닫혔다. Density 및 나머지 scorer 구성은 이 네 객체의
강함을 설명하는 주된 요소가 아니다.

## 9. Conclusion and Cleanup

`PROMISING_DIAGNOSTIC / DVT_MECHANISM_SUPPORTED`다. 현재 Baseline1의 설명은 다음처럼 수정된다.

> 강한 DINOv3-H+ representation이 기본 anomaly evidence를 제공하고, DVT가 object-aligned
> position prior로 ranking을 크게 개선하며, Flow는 DVT가 만든 좌표계에서 작은 추가 이득을
> 제공한다. Flow 단독이 핵심은 아니지만 DVT와의 interaction은 양수다.

이 결과가 DVT의 distribution-shift correction을 증명하지는 않는다. 다음 단일 진단은 DVT 효과를
public condition bucket별(`regular`, photometric, geometric shift)로 분해해, 이득이 shifted normal
false-positive 감소인지 defect localization 증가인지 확인하는 것이다. condition token은 분석용
사후 stratification에만 사용하며 inference input으로 사용하지 않는다.

Local/remote 결과 root에서 regenerable `anomaly_maps/`가 남지 않았음을 확인했고 compact JSON/log만
보존했다.
