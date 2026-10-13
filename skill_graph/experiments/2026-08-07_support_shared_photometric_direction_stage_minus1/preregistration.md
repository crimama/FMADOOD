# Support-shared Photometric Direction Stage -1b 사전등록

## Negative evidence intake

Stage -1a0/a1은 한 image 안의 global/local spatial pattern을 조사했고, a1은 inactive
patch confound를 제거했어도 global conditional-smoothness가 `0/4`였다. 따라서 이 실험은
smoothness threshold, coverage aggregation, Flow, FiLM, projection, atlas, 또는 test-time
adaptation을 재시도하지 않는다. 새 module도 학습하지 않는다.

## Motivation

이번 질문은 공간적 smoothness가 아니라 다음의 독립 evidence이다. **고정된 전역 photometric
변환에 대한 feature displacement direction이 서로 다른 normal support image에서 재현되는가?**
재현된다면 normal support가 nuisance candidate의 방향을 제공한다는 최소 근거가 된다. 재현되지
않으면 normal-only global canonicalization은 support-identifiability 근거가 부족하다.

이는 synthetic transform의 representation diagnostic이며, real unseen shift의 성능이나 method
claim은 아니다. SuperAD comparison/segmentation metric은 `N/A`; public evaluation은 허용되지 않는다.

## Fixed design

- MVTec AD2 single-image, 고정 16 normal-support, `fruit_jelly`, `walnuts`, `vial`, `fabric`.
- DINOv3 ViT-H/16+ layers 7/15/23/31 layer-normalized mean fusion의 clean/changed feature
  difference만 사용한다. 이 difference audit에는 DVT/Flow fitting, 1-NN scoring, query
  adaptation, density/RGB/morphology가 모두 off다.
- Image-disjoint 4 folds (12 source direction bank / 4 held images).
- `S_global`: Stage -1a0과 같은 shared brightness/contrast/gamma/RGB-gain transform.
- `S_smooth`: Stage -1a0과 같은 shared low-frequency illumination field.
- `S_source_random`: source relative path SHA로 고정한 4개 서로 다른 global photometric
  transform 중 하나. 모든 image가 같은 nuisance direction을 공유한다는 가설의 null이다.
- 모든 measurement는 clean/changed DINO displacement만 사용한다. defect mask, test/public/private,
  ground truth는 사용·열거·접근하지 않는다.

각 held source \(i\), condition \(c\)에 대해 patch-average displacement \(d_i^c\)를 L2-normalize
한다. source bank \(F\)의 normalized mean direction \(\bar d_F^c\)에 대한
\(T_i^c=cos(d_i^c,\bar d_F^c)\)를 **cross-image transfer**로 기록한다.

## Pre-registered gate

각 object에서 아래 median은 held source/fold 단위 median이다.

1. validity: finite direction and nonzero source-bank norm for every row.
2. `S_global` transfer > `S_source_random` transfer + `0.05`.
3. `S_smooth` transfer > `S_source_random` transfer + `0.05`.
4. global 및 smooth transfer 모두 > `0.20`.
5. local GPU 0/dsba5 GPU 3 runner hash, support manifest, access ledger가 동일하고 각 aggregate
   absolute `1e-3` 또는 relative `1%` 이내.

`>=3/4` object가 1--4를 통과하면 `PROMISING_DIAGNOSTIC / CONTINUE_DESIGN_ONLY`다. 이는
support-shared nuisance direction의 추가 design specification 하나만 허용하며 어떤 module,
Flow/Flow-Matching/Diffusion/DiT implementation 또는 public benchmark도 허용하지 않는다.
그 외는 `INCONCLUSIVE_OR_NOT_SUPPORTED / NO_NEW_MODULE`이다.

## Continuation contract

양성일 때의 단 하나의 다음 diagnostic은 held support로 만든 shared direction이 unknown normal
photometric perturbation에 transfer되는지 검증하는 것이다. 음성일 때는 normal support 기반
global-direction family를 hard-stop하며, spatial statistic을 다시 조정하지 않는다.
