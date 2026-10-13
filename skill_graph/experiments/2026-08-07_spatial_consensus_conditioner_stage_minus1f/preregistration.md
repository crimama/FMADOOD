# Spatial-consensus Conditioner Stage -1f 사전등록

## Negative evidence intake

Stage -1e는 query 전체 patch의 channel median/MAD를 condition으로 사용한 naive moment
alignment가 `0/4`였고, defect-induced conditioner drift가 `.218--.807`이었다. 이번 실험은 그
transport의 threshold·scale·rank를 조정하지 않는다. transport capacity를 늘리지도 않는다.

새 질문은 condition estimator 자체에 한정한다.

> global photometric shift는 여러 spatial region에서 합의되지만 defect-induced structural
> shift는 일부 region에 집중된다는 inductive bias를 사용하면, mask/label 없이 shift-aligned하고
> defect-invariant한 condition을 식별할 수 있는가?

후보가 실패하면 Flow/Flow Matching/Diffusion/DiT로 진행하지 않는다. 성공하더라도 결과는
condition-identifiability diagnostic이며 method/benchmark claim이 아니다.

## Fixed scope

- target: MVTec AD2 single-image, data root `/home/hunim/Volume/DATA/mvtec_ad_2`.
- fixed 16 normal support manifest SHA256
  `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`.
- objects: `fruit_jelly`, `walnuts`, `vial`, `fabric`; fixed 4 folds, 12 fit / 4 held.
- frozen DINOv3 ViT-H/16+ fused feature. Flow/DVT/training/query update는 사용하지 않는다.
- conditions: clean, Stage -1c의 source-disjoint `U_unseen`, `U+D_local`, `U+D_broad`.
- masks와 paired clean feature는 **평가에만** 사용한다. 모든 candidate conditioner는 query와
  clean fit support만 입력받는다.
- SuperAD/RN-FMLK/segmentation metric은 `N/A`: support-only assumption diagnostic이며 public
  benchmark claim이 아니다.

## Fixed estimators

각 fold에서 fit support의 position template
`T[p,c] = median_i H_clean[i,p,c]`를 만든다. Query residual은 `R[p,c]=H_q[p,c]-T[p,c]`다.

### C0 — Stage -1e naive negative control

position을 무시하고 `H_q`와 flattened fit memory의 channel median 차이를 사용한다. 이전 E2와
동일한 conditioner다.

### C1 — position-aligned global residual median

`g_C1 = median_p R[p]`. Support의 동일 patch position을 먼저 빼 object-layout variation을
줄이되, 모든 spatial region을 동일하게 사용한다.

### C2 — fixed spatial block consensus

feature grid를 좌표 순서대로 `4 x 4` blocks로 나눈다. 각 block에서
`b_j = median_{p in block j} R[p]`를 계산한다. 16개 block vector의 coordinate median을 초기값으로
Weiszfeld geometric median을 정확히 20회 계산한다(`eps=1e-6`). 이 초기 consensus와 L2 거리가
가장 작은 8개 block을 stable ordering으로 고정 선택하고, 그 8개 block vector의 coordinate
median을 `g_C2`로 사용한다. block 수, retained fraction, iteration을 sweep하지 않는다.

## Evaluation-only targets and metrics

같은 held source의 clean condition을 이용해 instance bias를 제거한 estimated shift를
`d_C(U)=g_C(U)-g_C(clean)`으로 정의한다. 평가용 oracle photometric displacement는
`d_oracle=median_p(H_U[p]-H_clean[p])`다.

각 source에 대해 다음을 기록하고 object median으로 판정한다.

1. **shift alignment**: cosine(`d_C(U)`, `d_oracle`) 및
   `||d_C(U)-d_oracle|| / ||d_oracle||`.
2. **defect contamination**: `U+D`에서
   `||g_C(U+D)-g_C(U)|| / ||d_C(U)||`와 cosine(`d_C(U+D)`, `d_C(U)`).
3. **consensus advantage**: C2 contamination drift가 C0보다 감소한 상대 비율과, C2 oracle
   cosine이 C0보다 얼마나 변했는지.
4. **downstream sentinel**: conditioner로 clean memory를 translation한 raw 1-NN score에서
   U q995의 **절대 inflation 차이** `E0 inflation - C inflation`. Stage -1e의 near-zero denominator
   문제 때문에 상대비를 primary로 사용하지 않는다.
5. **safety sentinel**: conditioner를 각 defect query 자체에서 추정했을 때 local/broad defect
   contrast retention. mask는 이 계산에만 사용한다.
6. translation geometry sanity: pair-distance Spearman/NN retention `>=.999`.

## Gate

Validity requires finite outputs, exact manifest/hash/fold coverage, no public/private/GT access,
translation geometry sanity, and local/dsba5 metric parity(절대 `1e-3` 또는 상대 `1%`).

C2 object identifiability pass requires all:

- oracle cosine `>=.90`, oracle relative error `<=.35`;
- local contamination cosine `>=.95`, drift `<=.15`;
- broad contamination cosine `>=.90`, drift `<=.30`;
- C2 drift가 C0보다 local과 broad 각각 `>=25%` 감소;
- C2 oracle cosine이 C0보다 `.02` 이상 악화되지 않음.

Downstream continuation safety additionally requires local/broad contrast retention `>=.90` and
U q995 absolute inflation reduction이 `>=0`이다.

- `>=3/4` object가 identifiability와 downstream safety를 모두 통과하면
  `PROMISING_DIAGNOSTIC / ALLOW_ONE_BOUNDED_CONDITIONAL_MODULE`.
- `>=3/4`가 identifiability만 통과하고 safety가 부족하면
  `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`으로 estimator와 transport interface의 단일 분리
  실험만 허용한다.
- identifiability pass가 `<=2/4`이거나 C2가 C0 contamination을 줄이지 못하면
  `KILL_FOR_CLAIM / NO_CONTINUE_SPATIAL_CONSENSUS`다.

어떤 결과도 public benchmark나 architecture family 비교를 허용하지 않는다. 성공 시 다음 한
단계는 C2 condition을 frozen Baseline1 Flow의 condition input으로 넣기 전, identity-near bounded
affine module 하나만 별도 사전등록하는 것이다.
