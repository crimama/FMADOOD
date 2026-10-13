# Layerwise Defect Spillover Stage -1g 사전등록

## Negative evidence intake

Stage -1f의 spatial-consensus conditioner는 `0/4`였다. 이미지에서 local인 synthetic defect가
fused DINO feature에서는 여러 block의 conditioner residual을 바꿨지만, 그 원인이 layer mixing인지
각 layer 자체의 non-local representation인지 아직 분리되지 않았다.

본 실험은 새 conditioner, feature selector, score fusion 또는 transport를 구현하지 않는다. Layer를
사후 선택해 benchmark에 쓰지도 않는다. fixed DINOv3 layers `[7,15,23,31]`과 현재
`layer_norm_mean` fused representation의 **receptive-field spillover measurement**만 수행한다.

## Scope and custody

- target dataset: MVTec AD2 single-image,
  `/home/hunim/Volume/DATA/mvtec_ad_2`.
- fixed 16 normal support, objects `fruit_jelly/walnuts/vial/fabric`, 모든 16 source를 독립 측정.
- support manifest SHA256:
  `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`.
- frozen DINOv3 ViT-H/16+, shorter edge 672, layers `[7,15,23,31]`.
- representations: 각 single layer를 token-wise L2 normalize한 map과 동일 normalized layer들의
  mean인 `fused`. 이는 `layer_norm_mean` contract와 일치한다.
- conditions: clean, fixed `U_unseen`, `U+D_local`, `U+D_broad`. Defect displacement는
  `H(U+D)-H(U)`, global displacement는 `H(U)-H(clean)`으로 분리한다.
- synthetic mask는 measurement에만 사용하며 model input/selection에는 사용하지 않는다.
- Flow, DVT, fitting, query update, public/private test, GT는 모두 off.
- SuperAD/RN-FMLK와 segmentation metrics는 `N/A`, `comparable=false`: method claim이 아닌
  support-only representation diagnostic이다.

## Fixed spatial measurements

Patch displacement energy는 `e_p = ||delta_p||_2^2`다. Pixel defect mask를 patch grid로 변환한
`M0`와 8-neighbor로 정확히 한 patch 확장한 `M1`을 사용한다. Dilation 크기는 sweep하지 않는다.

각 representation과 source에서 다음을 기록한다.

1. `global_energy_entropy`: global energy를 patch 확률로 normalize한 Shannon entropy를
   `log(P)`로 나눈 값. 1에 가까울수록 image-wide하다.
2. `local_far_spill_fraction`: local-defect energy 중 `not M1`에 있는 비율.
3. `local_energy_entropy`: local-defect energy의 normalized entropy.
4. `local_outside_core_density_ratio`: `not M1`의 평균 energy / `M0` 평균 energy.
5. broad defect에도 같은 세 spillover 지표를 기록한다.
6. `local_condition_contamination_ratio`:
   `||mean_p delta_local|| / ||mean_p delta_global||`; broad도 동일하다.
7. `local_global_alignment_cosine`: aggregate local-defect vector와 global-shift vector cosine;
   broad도 동일하다.

Zero/non-finite energy는 validity failure다. 작은 값의 비율 폭주를 막기 위해 contamination ratio
분모에는 only numerical `1e-12`를 사용하고 raw numerator/denominator norm도 함께 저장한다.

## Fused control and fixed gate

각 individual layer는 같은 object의 fused median과 비교한다. 어떤 layer도 결과를 본 뒤 조합하지
않는다.

### Layer-localization object pass

한 layer가 한 object에서 다음을 모두 만족해야 한다.

- global energy entropy `>=.90`;
- local far-spill fraction `<=.25`;
- local energy entropy `<=.75`;
- local outside/core density ratio `<=.20`;
- fused보다 local far-spill이 `>=.10`p 작음.

### Conditioner-feasibility object pass

위 localization pass에 더해:

- local contamination ratio `<=.15`, broad contamination ratio `<=.30`;
- local과 broad aggregate alignment cosine의 절대값 `<=.30`;
- fused보다 local contamination ratio가 `>=.10` 작음.

## Verdict

Validity는 exact layer/grid/source coverage, finite metrics, manifest/hash, zero test/GT access,
local/dsba5 parity(절대 `1e-3` 또는 상대 `1%`)를 요구한다.

- 동일 individual layer가 localization과 conditioner feasibility를 모두 `>=3/4` object에서
  통과하면 `PROMISING_DIAGNOSTIC / ALLOW_LAYER_SPECIFIC_CONDITIONER_PREFLIGHT`.
- 동일 layer가 localization만 `>=3/4` 통과하면
  `KILL_FOR_CLAIM / CONTINUE_LAYER_INTERFACE_DIAGNOSTIC`: anomaly scorer와 conditioner의 layer를
  분리하는 단일 support-only preflight만 허용한다.
- 어떤 layer도 localization을 `>=3/4` 통과하지 못하면
  `KILL_FOR_CLAIM / NO_CONTINUE_LAYERWISE_CONDITIONER`.

어떤 결과도 layer 선택, multi-layer fusion tuning, public benchmark, Flow Matching/Diffusion/DiT
진입을 직접 허용하지 않는다.
