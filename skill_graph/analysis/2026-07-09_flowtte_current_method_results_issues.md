# FlowTTE Current Method, Results, and Key Issues

## 1. Current Method

현재 유지 중인 방법론은 **Flow-LatentBank no-TTE with DINOv3-H+ and DVT-style denoise**로 볼 수 있다.

```text
DINOv3-H+/16 patch feature
layers [7,15,23,31]
-> layer_norm_mean fusion
-> DVT-style position_mean denoise, alpha=1.0
-> Normalizing Flow latent projection
-> fixed 16-shot support latent memory bank
-> test patch to support latent nearest-neighbor distance
-> continuous anomaly score map
```

핵심 설정:

- Dataset: MVTec AD2 single-image, all 8 public objects
- Split: full `test_public/good,bad`
- Support: fixed 16-shot reference JSON
- Backbone: `dinov3_vith16plus`
- Feature layers: `[7,15,23,31]`
- TTE: disabled, fixed support memory only
- Denoise: DVT-lite `position_mean`, `alpha=1.0`
- Score: latent NN distance 중심, density term 보조

이 구조의 의도는 test-time memory expansion 없이, DINOv3-H+ patch feature를 NF latent space로 projection한 뒤 support normal latent bank와의 거리를 이용해 patch-level anomaly ranking을 만드는 것이다.

## 2. Current Best Result

현재 내부 기준으로 가장 좋은 FlowTTE 계열 reference는 H+ backbone-only DVT branch다.

| Method | Backbone / Setting | seg_AUROC_0.05 | seg_F1 |
|---|---|---:|---:|
| FlowTTE H+ reference | DINOv3-H+, DVT alpha 1.0, fixed 16-shot | 0.836739 | 0.527427 |
| reported SuperADD context | DINOv3-H+, SuperADD-style scoring/postprocess context | 0.839300 | 0.626113 |

해석:

- AUROC는 SuperADD와 거의 비슷하다.
- F1은 약 `0.098686` 낮다.
- 따라서 현재 핵심 gap은 **ranking 자체보다 score map을 binary-quality mask로 만드는 과정**에 있다.

## 3. Recent Structural Diagnostics

최근 실험은 NF 구조, foreground/background 분리, score-field calibration이 현재 bottleneck을 해결하는지 확인하기 위한 all-object diagnostic이었다.

| Variant | Status | seg_AUROC_0.05 | seg_F1 | Delta F1 vs H+ |
|---|---|---:|---:|---:|
| H+ reference | baseline | 0.836739 | 0.527427 | 0.000000 |
| conditional_cls | completed | 0.832374 | 0.512126 | -0.015301 |
| foreground_flow_mixture | completed | 0.834712 | 0.519250 | -0.008177 |
| local_contrast | completed | 0.806200 | 0.438345 | -0.089082 |
| conditional_xy | runtime-blocked | NA | NA | NA |
| conditional_cls_xy | runtime-blocked | NA | NA | NA |

결론:

- CLS를 NF condition으로 넣는 방식은 성능 향상으로 이어지지 않았다.
- foreground/background flow mixture가 가장 근접했지만 H+ reference보다 낮았다.
- local contrast는 score noise를 키우며 broad하게 성능을 악화시켰다.
- coordinate-conditioned NF는 현재 구현 경로에서 runtime-blocked이며 실용성이 낮다.

## 4. Main Problems

### 4.1 F1 Gap Is the Main Problem

현재 방법은 continuous score ranking은 강하지만, F1-friendly mask를 만드는 능력이 약하다.

```text
FlowTTE H+ AUROC: 0.836739
SuperADD AUROC:  0.839300

FlowTTE H+ F1:   0.527427
SuperADD F1:     0.626113
```

즉 문제는 단순히 anomaly score가 낮은 것이 아니라:

```text
score field fragmentation
threshold sensitivity
boundary imprecision
object/background confusion
small defect mask quality
```

쪽에 더 가깝다.

### 4.2 `can` Collapse Persists

여러 class-agnostic variant에서 `can` F1이 거의 0에 가깝게 유지된다.

예:

- `conditional_cls`: `can` F1 `0.000756`
- `foreground_flow_mixture`: `can` F1 `0.000500`
- `local_contrast`: `can` F1 `0.001495`

이 현상은 mean F1을 크게 낮추며, 단순 foreground/background split이나 local contrast로 해결되지 않았다.

### 4.3 NF Conditioning Is Not the Positive Component

이전 CLS soft-distance 계열에서는 global context가 약한 개선을 보였지만, CLS를 NF 내부 condition으로 직접 넣는 방식은 H+ reference보다 낮았다.

```text
conditional_cls: 0.832374 / 0.512126
H+ reference:    0.836739 / 0.527427
```

따라서 global context는 현재 evidence 기준으로:

```text
NF 내부 condition
```

보다는:

```text
retrieval routing
score calibration
support/memory selection
mask formation
```

쪽에 쓰는 것이 더 자연스럽다.

### 4.4 Foreground/Background Separation Alone Is Insufficient

Foreground/background flow mixture는 가장 근접한 구조 variant였지만, all-object mean에서는 baseline을 넘지 못했다.

```text
foreground_flow_mixture: 0.834712 / 0.519250
H+ reference:            0.836739 / 0.527427
```

일부 object에서는 개선이 있었지만, object/background/defect evidence가 선형적으로 분리되는 문제가 아니었다.

핵심은:

```text
background-like patch = suppress
foreground-like patch = anomaly score 유지
```

같은 단순 rule이 defect evidence까지 같이 억제하거나, 반대로 정상 object variation을 anomaly로 키울 수 있다는 점이다.

### 4.5 Local Score Contrast Is Harmful

`local_contrast`는 smooth nuisance를 제거하기보다 정상 score variation과 score noise를 함께 증폭했다.

```text
local_contrast: 0.806200 / 0.438345
```

이는 현재 score field가 단순 local peak enhancement에 적합하지 않다는 것을 의미한다. local peak는 anomaly evidence일 수도 있지만, patch feature artifact, boundary variation, texture variation일 수도 있다.

### 4.6 Coordinate Conditioning Is Not Ready

`xy`, `cls_xy`는 metrics 없이 장시간 실행되어 runtime-blocked로 처리했다.

이 방향의 문제는 두 가지다.

1. 현재 구현 경로가 실험적으로 비효율적이다.
2. position context가 anomaly-aligned signal인지 불명확하다.

특히 DVT-style denoise가 position-dependent artifact를 줄이려는 방향인데, NF에 explicit coordinate condition을 넣으면 position nuisance를 다시 모델링할 위험이 있다.

## 5. Current Interpretation

현재 FlowTTE는 다음처럼 보는 것이 가장 정확하다.

```text
strong continuous patch ranker
but weak binary mask generator
```

강점:

- DINOv3-H+ feature와 DVT-style denoise 덕분에 AUROC는 SuperADD에 근접한다.
- no-TTE fixed memory라 anomaly absorption/ranking collapse 위험은 낮다.
- NF latent projection은 raw NN보다 완전히 무의미하지는 않으며, current H+ reference가 여러 raw/null variant보다 강하다.

약점:

- score field가 fragmented하거나 object boundary/texture variation에 민감하다.
- threshold와 post-threshold morphology가 약하다.
- foreground/background prior를 단순히 곱하거나 분리하는 방식은 anomaly evidence도 같이 훼손한다.
- NF를 더 복잡하게 하는 방향은 현재까지 성능 개선의 주된 길이로 보이지 않는다.

## 6. Practical Direction

현재 증거 기준으로 우선순위는 다음과 같다.

1. **NF 구조 확장보다 score-field to mask formation 개선**
   - threshold calibration
   - connected component stability
   - morphology/post-threshold policy
   - score map fragmentation analysis

2. **Foreground/background를 hard suppression이 아니라 reliability prior로 사용**
   - background score를 무조건 낮추기보다 uncertainty/reliability로 다루는 쪽이 더 안전하다.

3. **Global context는 NF condition보다 retrieval/calibration 쪽에 사용**
   - CLS/register를 patch localization evidence로 쓰는 것이 아니라 image-level context key로 쓰는 방향이 더 일관적이다.

4. **Coordinate-conditioned NF는 보류**
   - runtime path를 고치기 전에는 실험 효율이 낮고, mechanism justification도 약하다.

## 7. Current Verdict

현재 방법론 상태:

```text
FlowTTE H+ DVT fixed-memory latent NN
= strong diagnostic baseline
= not yet SuperADD-level method
```

최근 구조 실험 verdict:

```text
KILL_FOR_CLAIM / NO_CONTINUE
```

단, 이 verdict는 FlowTTE 전체 폐기가 아니라, 최근 테스트한 다음 구조 개선안들에 대한 verdict다.

```text
conditional NF with CLS
coordinate-conditioned NF
foreground/background flow mixture
local contrast score-field calibration
```

현재 남은 핵심 연구 문제는:

> DINOv3-H+ + DVT + NF latent distance가 만드는 continuous score map을 어떻게 class-agnostic하게 안정적인 object-level binary anomaly mask로 바꿀 것인가?

