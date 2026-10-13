# Mask-free Conditional-Smoothness Stage -1a1 사전등록

## Negative evidence intake

이 실험은 이미 종료된 paired-loss, shift-conditioned FiLM, semantic-atlas, WTCF branch의
재튜닝이 아니다. 새 transform도 학습하지 않는다. Stage -1a0에서 global displacement
coherence는 `4/4` object에서 defect보다 컸지만, all-patch adjacent cosine은 local defect의
많은 비변화 patch 때문에 `fabric` 외 `3/4`에서 defect보다 낮지 않았다. 이 audit은 그
**측정 오염 경로**만 제거하는 one-step diagnostic이다.

## Motivation and claim boundary

Claim: frozen Baseline1 representation에서 global/smooth nuisance의 **활성 displacement
region**은 local/broad defect보다 더 넓고 coherent하게 연결되는가?

이는 method claim 또는 SuperAD 비교가 아니다. `SuperAD baseline = N/A`이며, full TESTpublic,
seg_AUROC_0.05, seg_F1, RN-FMLK 비교는 범위 밖이다. 성공해도 `PROMISING_DIAGNOSTIC`일 뿐
module 구현을 자동 승인하지 않는다.

## Fixed protocol

- target dataset: MVTec AD2 single-image;
  `data_root=/home/hunim/Volume/DATA/mvtec_ad_2`.
- objects: `fruit_jelly`, `walnuts`, `vial`, `fabric`; 고정 16-image normal support manifest.
- representation: DINOv3 ViT-H/16+ layers 7/15/23/31, layer-normalized mean fusion,
  position-mean DVT, frozen Baseline1-core Flow latent, static raw-Euclidean patch 1-NN.
- split: image-disjoint 4 folds (12 fit / 4 held).
- conditions: Stage -1a0과 동일한 deterministic `S_global`, `S_smooth`, `D_local`, `D_broad`,
  `SD_global_local`, `SD_global_broad`.
- custody: `train/good` support만 사용. query adaptation, test_public, test_private,
  ground_truth, density/RGB/morphology 모두 off.
- parallel reproducibility: local GPU 0과 dsba5 host GPU 3 (container CUDA slot 0)에서
  동일 snapshot/support/checkpoint를 실행한다.

## Mask-free conditional-smoothness measurement

clean/changed DVT feature displacement를 \(\Delta_p\), magnitude를
\(a_p=\|\Delta_p\|_2\)라 둔다. 각 image-condition에서 아래를 **pixel mask 없이** 계산한다.

1. activation threshold: \(\tau=median_p(a_p)+MAD_p(a_p)\).
2. active set: \(A=\{p:a_p\ge\tau\}\).
3. active-pair fraction: 전체 horizontal/vertical neighbor edge 중 양 끝이 `A`인 비율.
4. conditional cosine: 양 끝이 `A`인 edge에서만
   \(cos(\Delta_p,\Delta_q)\)의 평균. valid active edge가 없으면 invalid이며 object는 gate를
   통과할 수 없다.
5. conditional-smoothness: `conditional cosine × active-pair fraction`.
6. energy effective coverage:
   \(1/(N\sum_p(a_p/\sum_q a_q)^2)\). 값이 클수록 displacement energy가 넓게 분포한다.

`D_local/D_broad/SD_*`의 synthetic mask는 위 1--6에 입력하지 않는다. mask는 기존처럼
defect contrast retention safety proxy에만 사용한다.

## Pre-registered gate

각 object에서 global aggregate는 `S_global`, `S_smooth` median, defect aggregate는
`D_local`, `D_broad` median이다.

- measurement validity: 모든 condition의 held row에서 finite metric, active-pair fraction > 0,
  valid active edge 존재.
- factorisation signal: global coherence fraction > defect coherence fraction, global
  conditional-smoothness > defect conditional-smoothness, global energy effective coverage >
  defect coverage.
- safety: `SD_global_local`과 `SD_global_broad` defect contrast retention median이 각각 > 0.
- reproducibility: two lane input ledger/script hash 동일, aggregate difference는 absolute `1e-3`
  또는 relative `1%` 이내.

`>=3/4` object가 모든 조건을 만족하면
`PROMISING_DIAGNOSTIC / CONTINUE_DIAGNOSTIC`으로 기록한다. 이는 one further
measurement/design specification을 허용할 뿐 module/public evaluation을 허용하지 않는다.
그 외는 `INCONCLUSIVE_OR_NOT_SUPPORTED / NO_NEW_MODULE`이다.

## Continuation contract

양성 결과의 다음 한 단계는 global/local interface와 mask-free proxy를 명시한 **N1 design
document**이며, Flow/Flow Matching/Diffusion 중 하나의 architecture를 구현하는 단계가 아니다.
음성 결과는 coherent-nuisance prior의 현재 feature-space operationalization을 hard-stop하고,
다른 evidence source 없이는 global canonicalization family를 열지 않는다.

## Planned artifacts

- runner, local/dsba5 remote controller, parity aggregator;
- `run_manifest.json`, access ledger, per-object summary, progress, checksums;
- final `report.md`, `tasks/todo.md`, index/log lifecycle entries.
