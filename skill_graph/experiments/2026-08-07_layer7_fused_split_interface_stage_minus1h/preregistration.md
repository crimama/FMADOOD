# Layer-7 / Fused Split Interface Stage -1h 사전등록

## Negative Evidence Intake

Stage -1f의 fused-feature spatial consensus는 conditioner contamination을 줄이지 못해 `0/4`였다.
Stage -1g는 그 실패를 layer별로 분해했고, layer 7에서만 local-defect localization이 `4/4`였지만
단순 spatial mean condition은 feasibility `0/4`였다. 따라서 본 실험은 layer 7을 anomaly feature로
선택하거나 layer를 sweep하지 않는다. **Layer 7은 defect-sensitive patch를 배제하는 mask-free gate로만
사용하고, condition vector와 anomaly score는 기존 fused representation에서 유지**한다.

이는 spatial-consensus retune이 아니다. 새 근거는 fused layer에서는 없었던 layer-7 localization
`4/4`이며, 검증 대상은 하나의 split interface다. retained fraction, layer, threshold, transport 종류를
sweep하지 않는다.

## Scope and custody

- target: MVTec AD2 single-image, `/home/hunim/Volume/DATA/mvtec_ad_2`.
- objects: `fruit_jelly/walnuts/vial/fabric`; fixed normal support 16, 4-fold 12/4 source-disjoint.
- support manifest SHA256:
  `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`.
- frozen DINOv3 ViT-H/16+, shorter edge 672, layer 7 and current layers `[7,15,23,31]`
  `layer_norm_mean` fused representation.
- conditions: clean, fixed unseen photometric `U`, `U+D_local`, `U+D_broad`.
- Flow, DVT, fitting, query update, public/private test, GT are off.
- SuperAD/RN-FMLK and segmentation metrics are `N/A`, `comparable=false`: support-only interface
  diagnostic이며 method/benchmark claim이 아니다.

## Fixed interface

각 fold의 12개 fit support에서 position-wise layer-7 template와 fused template을 channel median으로
계산한다. Held query에서 다음 두 conditioner를 비교한다.

1. `C0_fused_all`: 모든 patch의 `(query_fused - fused_template)` channel median. Stage -1e/1f의
   naive fused condition에 해당하는 negative control이다.
2. `C7_gate_fused`: `(query_layer7 - layer7_template)` patch vector들의 geometric median을 구하고,
   그 중심과 cosine distance가 작은 `round(0.75 * P)` patch를 유지한다(`P`는 grid patch 수,
   Python/NumPy nearest-even round). 동일 patch index에서
   `(query_fused - fused_template)`의 channel median을 condition vector로 계산한다.

목표 retained fraction 75%는 고정이며 local defect가 patch minority라는 hypothesis를 직접 반영한다.
`P`가 4의 배수가 아닐 때 기록되는 실제 fraction은 `round(0.75 * P) / P`다.
동점은 patch index 순으로 결정한다. Synthetic mask, paired clean image, condition label은 gate 입력에
사용하지 않는다. Geometric median은 Weiszfeld 20 steps, epsilon `1e-6`으로 고정한다.

Condition vector `g`로 fit fused memory를 `memory + g`만큼 이동하고 held fused query의 raw 1-NN
score를 계산한다. Scale, rotation, projection, Flow는 사용하지 않는다.

## Fixed measurements

각 object의 16 held-source median을 기록한다.

- `U_oracle_cosine`, `U_oracle_relative_error`: paired clean은 평가 전용 oracle
  `median(U_fused-clean_fused)`를 정의한다.
- `UD_local/broad_condition_relative_drift_vs_U`.
- C7의 C0 대비 defect-drift relative reduction.
- shifted-normal q995 inflation과 identity 대비 relative reduction.
- C7의 C0 대비 tail-reduction margin.
- local/broad defect contrast retention versus identity.
- transported-memory pairwise-distance Spearman과 nearest-neighbor retention.
- retained patch count/hash, source/fold coverage, access ledger.

## Frozen gates

### Object interface pass

한 object가 다음을 모두 만족해야 한다.

- C7 `U_oracle_cosine >= .90`, `U_oracle_relative_error <= .35`;
- C7 local/broad drift `<= .15/.30`;
- C7이 C0보다 local/broad drift를 각각 `>=25%` 감소;
- shifted-normal tail reduction versus identity `>=15%` and C0 margin `>=10` percentage points;
- local/broad contrast retention `>=.95`;
- pairwise Spearman와 NN retention `>=.95`.

동일 C7 interface가 `>=3/4` objects에서 pass하면
`PROMISING_DIAGNOSTIC / ALLOW_MINIMAL_TARGET_MODULE_PREFLIGHT`다. 이는 Flow/benchmark 허가가 아니다.

### Weak continuation gate

Strict pass가 `<3/4`여도 다음을 동시에 `>=3/4` objects에서 만족하면
`KILL_FOR_CLAIM / CONTINUE_CONDITION_AGGREGATOR_DIAGNOSTIC`을 허용한다.

- C7 local과 broad drift가 C0보다 모두 감소하고, 둘 중 작은 감소율도 `>=25%`;
- `U_oracle_cosine >=.85`;
- local/broad contrast retention `>=.95`;
- geometry gates 통과.

그 외는 `KILL_FOR_CLAIM / NO_CONTINUE_SPLIT_INTERFACE`다. 어떤 결과도 layer sweep, learned fusion,
Flow Matching/Diffusion/DiT 또는 public benchmark를 직접 허용하지 않는다.

## Validity and parity

Exact object/fold/source coverage, 12/4 source disjointness, finite metrics, retained fraction/count,
support/code hash, zero test/GT access가 필수다. Local과 dsba5 aggregate metric은 절대 `1e-3` 또는
상대 `1%` 이내여야 최종 verdict가 유효하다.
