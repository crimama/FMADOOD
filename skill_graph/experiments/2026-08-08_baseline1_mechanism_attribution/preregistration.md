# Baseline1 Mechanism Attribution 사전등록

Date: 2026-08-08
Protocol: `baseline1-mechanism-attribution-20260808`
Scope: `PROMISING_DIAGNOSTIC` 이하의 reduced-object TESTpublic 기전 진단

## Motivation

Baseline1의 강한 결과를 새 shift module의 근거로 간주하기 전에, DINOv3-H+ 표현 자체,
Flow 변환, 다중-layer fusion 가운데 무엇이 실제 성능을 만드는지 분리한다. 이는 새 passive
feature-surgery 방법이 아니며 이미 유지 중인 Baseline1의 인과 귀인 실험이다.

## 고정 조건

- dataset: MVTec AD2 single-image
- data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- objects: `fruit_jelly, walnuts, vial, fabric`
- support: frozen SuperAD-selected 16, manifest SHA-256
  `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`
- backbone: frozen `dinov3_vith16plus`, resolution 672
- no query adaptation, no memory update, no metadata/filename condition, no morphology
- scorer: raw Euclidean latent 1-NN, density weight 0, DVT off
- metrics: `seg_AUROC_0.05`, `seg_F1`, `seg_AUPRO_0.05`, map coverage

## Arms

| representation | raw arm | Flow arm |
| --- | --- | --- |
| early | layer `[7]`, identity | layer `[7]`, affine-coupling Flow |
| late | layer `[31]`, identity | layer `[31]`, affine-coupling Flow |
| fused | layers `[7,15,23,31]`, normalized mean, identity | same + Flow |

DVT와 density를 끈 이유는 Flow와 representation의 상호작용을 다른 구성요소와 분리하기
위함이다. 전체 Baseline1 및 SuperAD 값은 historical context일 뿐 이 reduced-object 결과의
직접 gate comparator가 아니다.

## 사전등록 contrasts와 gate

1. 각 representation의 `Flow - raw`를 계산한다.
2. interaction은
   `I = (fused_flow - fused_raw) - mean(layer7_flow-layer7_raw, layer31_flow-layer31_raw)`로 정의한다.
3. Flow main effect가 pAUROC에서 `>= +0.005`, 4개 중 `>=3` positive이고, F1/AU-PRO
   macro가 각각 `>-0.01`이며 어떤 객체도 pAUROC `-0.02`, F1 `-0.03`, AU-PRO `-0.02`
   아래로 붕괴하지 않으면 `FLOW_MECHANISM_SUPPORTED`다.
4. interaction `I >= +0.005`이고 4개 중 `>=3` positive이면
   `FUSION_FLOW_SYNERGY_SUPPORTED`다.
5. raw fused가 raw 단일-layer 평균보다 `>=+0.005`이면 backbone/fusion 자체의 이득,
   그렇지 않고 Flow 뒤에만 이득이면 fusion은 Flow가 활용하는 입력으로 해석한다.

Reduced-object이므로 어떤 양성도 방법론 `KEEP`이나 SuperAD 우월성 주장을 허용하지 않는다.
유효한 출력은 `PROMISING_DIAGNOSTIC`, `INCONCLUSIVE_MECHANISM`, 또는
`KILL_FOR_MECHANISM_CLAIM`뿐이다.

## 실행 및 저장

로컬 RTX 4090과 dsba5 고정 컨테이너 `hun_fsad_tta_012`/host GPU 3을 병렬 사용한다.
각 arm은 동일 runner/config와 support manifest를 사용하고 결과 map은 평가 직후 삭제한다.
compact metrics, manifest, logs, aggregate만 보존한다.
