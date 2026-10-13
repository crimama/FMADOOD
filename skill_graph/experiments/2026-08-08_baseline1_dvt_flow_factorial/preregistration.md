# Baseline1 DVT × Flow Factorial 사전등록

Date: 2026-08-08
Protocol: `baseline1-dvt-flow-factorial-20260808`
Scope: reduced-object mechanism diagnostic

## 질문

이전 factorial에서 DINOv3-H+ fused representation 위 Flow의 pAUROC 기여는 `+.002099`에
그쳤다. 전체 Baseline1과 DVT-off core 사이의 남은 차이가 DVT main effect인지, 아니면
DVT가 Flow와 결합될 때만 발생하는 interaction인지 분리한다.

## 고정 조건

- MVTec AD2 single-image; `fruit_jelly, walnuts, vial, fabric`.
- fixed SuperAD-selected support-16, DINOv3-H+/16 `[7,15,23,31]`, resolution 672.
- normalized layer-mean fusion, raw latent 1-NN, density weight 0.
- no query adaptation/memory update/metadata/morphology.
- metrics: `seg_AUROC_0.05`, `seg_F1`, `seg_AUPRO_0.05`, coverage.

## 2×2 arms

| | Identity/raw | Flow |
| --- | --- | --- |
| DVT off | 기존 `fused_raw` | 기존 `fused_flow` |
| DVT on | 신규 `dvt_raw` | 신규 `dvt_flow` |

DVT는 position-mean, alpha `1.0`; Flow는 기존 3-epoch affine coupling 설정을 그대로 쓴다.

## Contrasts와 gate

- DVT effect without Flow: `dvt_raw - fused_raw`.
- DVT effect with Flow: `dvt_flow - fused_flow`.
- DVT×Flow interaction:
  `(dvt_flow-dvt_raw) - (fused_flow-fused_raw)`.
- DVT main effect는 두 DVT contrast의 평균으로 정의한다.

`DVT_MECHANISM_SUPPORTED`는 pAUROC main effect `>=+.005`, positive object `>=3/4`,
macro F1/AU-PRO 각각 `>-0.01`, 객체별 collapse가 pAUROC `-.02`, F1 `-.03`,
AU-PRO `-.02`보다 크지 않을 때만 통과한다. Interaction은 pAUROC `>=+.005`와
positive object `>=3/4`일 때만 지지한다. Reduced-object이므로 결과는 방법론 KEEP이나
SuperAD 우월성 주장을 허용하지 않는다.
