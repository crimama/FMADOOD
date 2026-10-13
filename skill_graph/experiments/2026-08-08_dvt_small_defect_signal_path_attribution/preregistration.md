# DVT Small-Defect Signal-Path Attribution 사전등록

Date: 2026-08-08  
Protocol: `dvt-small-defect-signal-path-attribution-all8-20260808`  
Stage: mechanism attribution; local-only diagnostic

## 1. Negative evidence intake and motivation

직전 positional causal control은 DVT의 medium-defect 이득이 단순 global centering이나 임의
subtraction이 아니라 **올바른 spatial correspondence**에서 비롯됨을 보였다. 동시에 small defect의
top-1% coverage는 raw 대비 `-0.011532`로 악화되었다. 따라서 다음 질문은 DVT 강도, gating,
fusion, k 또는 object별 정책을 다시 튜닝하는 것이 아니라 다음 하나다.

> small-defect evidence 손실은 DVT subtraction, Flow transform, 또는 최종 deployed scoring 중
> 처음 어디에서 발생하는가?

이 실험은 feature surgery의 새 변형, residual bypass, score fusion, query adaptation, Flow
Matching/Diffusion/DiT 또는 성능개선 방법 제안이 아니다. 이미 닫힌 무조건적 DVT tuning branch를
되살리지 않으며, 다음 구조 탐색이 보존해야 할 signal을 식별하기 위한 단일 attribution이다.

## 2. Target, data, and custody contract

- `target_dataset=MVTec AD2 single-image`.
- local data root: `/home/hun/Volume/DATA/mvtec_ad_2`.
- objects: `can`, `fabric`, `fruit_jelly`, `rice`, `vial`, `wallplugs`, `walnuts`, `sheet_metal`.
- population: all-8, full `test_public/good,bad`; `sheet_metal`은 448, 나머지는 672.
- fixed support: object당 SuperAD-selected normal 16장. DINOv3-H+ layers는 `[7,15,23,31]`이다.
- common model: 기존 Baseline1 Flow config, 3 epochs, density weight `0`, raw latent 1-NN,
  deployed final scoring을 그대로 사용한다. Support, preprocessing, resolution, seed/config은
  DVT mode 외에 두 arm에서 동일해야 한다.
- arms: `raw` (DVT 없음)과 `position_current` (deployed positional artifact, alpha `1.0`)만
  all-8에서 **새로** 실행한다. 이 문서는 기존 결과를 새 arm의 대체물로 쓰지 않는다.
- GT는 모든 score map과 intermediate representation이 작성된 뒤, scale/defect-background
  evaluation에만 사용한다. condition/filename/metadata, `test_private`, query-driven parameter
  update는 금지한다.
- SuperAD/RN-FMLK/segmentation claim: `N/A`, `comparable=false`. 이는 TESTpublic GT를 사후
  사용하는 mechanism diagnostic이며 same-condition SuperAD method comparison이 아니다.

기존 final-score reference는
`results/dvt_alignment_defect_scale_attribution_20260808/local/fused_flow` 및
`results/dvt_alignment_defect_scale_attribution_20260808/local/dvt_flow`이다. 새 final output은
이 참조와 raw/current 각각 `<=1e-6` 이내로 일치해야 한다.

## 3. Frozen signal path and formulas

각 patch feature를 `F`, support feature memory를 `M`, current positional DVT artifact를
`a_p=mu_p-g`라고 둔다. Ordered stages는 다음과 같다.

1. `pre_dvt_feature`: subtraction 전 raw backbone fused feature. `F_pre = F`.
2. `post_dvt_feature`: raw arm은 `F_raw=F`, current arm은 `F_dvt,p=F_p-a_p`.
3. `flow_latent`: 동일한 arm-specific Flow를 적용한 initial-bank latent `z=phi(F_arm)`.
4. `deployed_final`: 기존 Baseline1 run이 출력하는 최종 anomaly score map.

각 intermediate feature stage의 score는 해당 arm의 clean-support memory에 대한 고정 raw 1-NN
patch distance다.

\[
s^{feature}_{r,p}=\min_{m\in M_r}\lVert r_p-m\rVert_2,
\qquad r\in\{F_{pre},F_{raw/dvt},z\}.
\]

`pre_dvt_feature`는 두 arm에서 bitwise-equivalent numerical path여야 한다. `deployed_final`은
새 scoring definition이 아니라 기존 runner output의 identity audit이다. 따라서 어떤 stage에도
k sweep, score normalization, threshold, learned calibration, map fusion 또는 test image fitting을
추가하지 않는다.

## 4. Evaluation labels and metrics

GT component area / image area 기준으로 defect component를 다음 두 bins로 고정한다.

- small: `<=1%`.
- medium: `1--5%`.

각 bad image와 eligible scale에서 defect/background score로 다음을 계산한다.

- **primary:** tie-inclusive top-1% defect-pixel coverage (`coverage_top1`);
- **secondary:** pixel AP, top-1% component hit, top-5% coverage, top-5% component hit,
  full pixel AUROC;
- custody/audit: eligible image/component count, selected-pixel/tie count, finite/nonconstant score,
  map/feature shape.

Image-level paired current-minus-raw delta를 object 평균한 뒤, object equal-weight macro를 낸다.
Small bin은 primary loss-location endpoint이며, medium AP/top-5 coverage는 DVT의 이미 알려진
benefit이 intermediate path에서 보존되는지 확인하는 secondary evidence다.

## 5. Loss-onset rule

Ordered candidate onset stages는 `post_dvt_feature`, `flow_latent`, `deployed_final`이다.
어느 stage `j`가 small-defect loss onset이려면 다음을 모두 만족해야 한다.

- all 8 objects가 small `coverage_top1`에 eligible;
- macro `current_minus_raw.coverage_top1 <= -0.01`;
- 적어도 3 objects에서 current-minus-raw `coverage_top1 < 0`;
- 바로 전 stage의 macro delta가 `>-0.005`.

`pre_dvt_feature`는 onset 후보가 아니다. 이 stage의 two-arm raw score/feature는 `<=1e-6`으로
일치해야 하며, 그렇지 않으면 causal order 자체가 성립하지 않는다. 여러 stage가 조건을 만족하면
가장 이른 stage만 원인 위치로 판정한다. `deployed_final`에서만 처음 조건이 생겨도 upstream
operator 하나에 귀속할 수 없으므로 `NO_SINGLE_STAGE_ATTRIBUTION`으로 처리한다.

## 6. Validity and replication gates

다음을 모두 통과해야 한다.

- raw/current support basenames, DINO layers, resolution, Flow config, seed, score policy가
  DVT mode 외에는 정확히 동일하다.
- `pre_dvt_feature`의 raw/current feature와 1-NN score delta가 `<=1e-6`이다.
- final raw/current summary와 per-image deployed final score가 기존 alignment-defect-scale
  reference 각각 `<=1e-6`으로 재현된다.
- all-8 coverage, feature/map shape, GT-pixel/component custody, finite/nonconstant values가
  완전하다.
- tie-inclusive top-1%/top-5% selected area가 nominal budget의 `2x`를 넘지 않는다.
- GT access는 all intermediate/final maps 저장 후에만 발생하며, test/private/metadata/query update
  flags가 모두 false다.

하나라도 실패하면 loss direction이나 magnitude를 해석하지 않고 `INVALID_SIGNAL_PATH_ATTRIBUTION`을
선언한다. 본 stage는 local-only이므로 dsba5 parity gate를 두지 않는다.

## 7. Frozen verdicts and continuation contract

- `LOSS_AT_DVT_SUBTRACTION`: validity 통과, earliest onset이 `post_dvt_feature`.
- `LOSS_AT_FLOW_TRANSFORM`: validity 통과, post-DVT는 onset이 아니고 earliest onset이
  `flow_latent`.
- `NO_SINGLE_STAGE_ATTRIBUTION`: validity 통과이나 onset이 없거나, final-only/onset-order가
  upstream operator 하나에 귀속되지 않는다.
- `INVALID_SIGNAL_PATH_ATTRIBUTION`: replication, custody, numerical, eligibility 또는 final
  identity audit 실패.

이 stage에서는 `KEEP` verdict가 없다. `LOSS_AT_DVT_SUBTRACTION`일 때만 다음 실험을
**한 개** 열 수 있다: support-repeatable positional component는 제거하되 small local innovation을
보존하는 fixed subspace/residual decomposition preflight. `LOSS_AT_FLOW_TRANSFORM`일 때만 Flow
interface를 대상으로 한 local-detail preservation preflight를 별도 사전등록할 수 있다. 그 외에는
DVT, Flow, score policy tuning을 재개하지 않는다.

## 8. Execution, outputs, and cleanup

- execution lane: local RTX 4090 only; dsba5는 이번 experiment에 사용하지 않는다.
- compact output root: `results/dvt_small_defect_signal_path_attribution_20260808/local/`.
- arm/stage/object summary, run manifest, access ledger, replication comparison, aggregate summary와
  log만 보존한다. dense `anomaly_maps/`는 생성하지 않으며, 생성되면 conclusion 뒤 제거하고 부재를
  확인한다.
- code evaluation은 stage order, exact 1-NN definition, pre-DVT identity, final identity, GT timing을
  focused test/dry-run으로 검증한 뒤에만 GPU execution을 시작한다.

