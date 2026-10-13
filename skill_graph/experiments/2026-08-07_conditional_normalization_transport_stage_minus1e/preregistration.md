# 조건부 정규화·수송 Stage -1e 사전등록

## 질문과 negative-evidence 경계

Stage -1b에서 support 간 global photometric displacement의 재현성은 확인됐지만, Stage -1c의
고정 rank-1 제거와 Stage -1d의 rank-4 residual 제거는 anomaly-score tail을 줄이지 못했다.
따라서 본 실험은 닫힌 projection/feature-surgery branch를 다시 튜닝하지 않는다. 검증 질문은
다음 하나다.

> query에서 관측되는 global shift를 제거할 nuisance가 아니라 **정상분포를 query condition으로
> 이동시키는 조건 변수**로 사용하면, unseen shift의 normal tail을 줄이면서 local defect
> contrast를 보존할 수 있는가?

이는 support-only 구조 선별 assay다. 학습 가능한 Flow/Flow Matching/Diffusion/DiT, public/private
test, ground truth, hyperparameter sweep 및 benchmark 성능 주장을 열지 않는다.

## 고정 데이터와 feature

- MVTec AD2 single-image, 고정 16 normal support manifest
  (`SHA256=9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`).
- `fruit_jelly`, `walnuts`, `vial`, `fabric`; 고정 4-fold, fold마다 12 fit / 4 held.
- frozen DINOv3 ViT-H/16+ fused patch feature와 deterministic 256 patch sampling.
- fit에는 clean support만 사용한다. held image는 query 역할만 하며 persistent parameter나 memory를
  갱신하지 않는다.
- held normal condition은 Stage -1c와 동일한 source-disjoint `U_unseen`; safety condition은
  `U+D_local`, `U+D_broad`이다. defect mask는 contrast 평가에만 사용한다.
- scorer는 현재 retained baseline contract와 맞춘 raw Euclidean 1-NN 하나로 고정한다.

## 비교 arm

### E0 — identity

12개 clean fit feature를 그대로 memory로 사용한다.

### E1 — projection negative control

Stage -1c와 동일하게 fit support의 `S_global-clean` 평균 direction 하나를 구하고 memory와 query
양쪽에서 rank-1 성분을 한 번 제거한다. 알려진 실패 대조군이며 strength/rank를 바꾸지 않는다.

### E2 — robust conditional normalization (location only)

fit memory의 channel-wise median `m_s`와 query patch의 channel-wise median `m_q`를 구한다.
조건 `g_hat=m_q-m_s`로 두고 memory만 `M'=M+g_hat`으로 이동한다. query는 바꾸지 않는다.
이는 query별 정상분포 위치를 이동시키는 가장 작은 target-specific module이다.

### E3 — robust diagonal conditional transport (location + scale)

E2의 위치 이동에 channel-wise MAD scale을 추가한다.

`M'=(M-m_s)*clip((MAD_q+eps)/(MAD_s+eps), 0.8, 1.25)+m_q`.

median/MAD와 고정 clip만 사용하며 학습, iteration, covariance/rank/strength 선택은 없다. E3가 E2를
명확히 이길 때만 더 유연한 Flow/Flow-Matching 계열을 탐색할 근거로 사용한다.

실제 결함 query에서는 E2/E3 conditioner를 그 결함 query 자체로 다시 추정한다. clean `U`에서
얻은 conditioner를 재사용하지 않아 defect contamination을 숨기지 않는다.

## 정량 평가

각 held source에 대해 다음을 기록하고 object median으로 판정한다.

1. **shifted-normal tail reduction**: clean-query 대비 `U` query의 q995 1-NN inflation을 구하고,
   E0 대비 각 arm의 상대 감소율 `(infl_E0-infl_arm)/abs(infl_E0)`을 계산한다.
2. **defect contrast retention**: local/broad mask 내부 평균 score와 외부 평균 score 차이의
   `arm/E0` 비율. mask는 이 계산 외에는 접근하지 않는다.
3. **clean geometry**: query-conditioned memory와 원 memory 간 pair-distance Spearman 및 1-NN
   identity retention. translation-only E2는 구현 sanity상 두 값이 수치적으로 1이어야 한다.
4. **conditioner contamination**: clean `U`와 `U+D`에서 추정한 location vector cosine,
   location 상대 drift, E3 log-scale vector의 RMS drift.
5. **complexity value**: E3 tail reduction에서 E2 tail reduction을 뺀 고정 margin.

## 사전등록 gate

먼저 모든 산출값 finite, manifest/support hash 일치, test/public/private/GT 미접근, E2 geometry
`>=.999`를 validity gate로 둔다. 이후 object pass는 다음을 모두 만족해야 한다.

- E2 또는 E3의 shifted-normal tail 상대 감소 `>=.15`;
- 같은 arm이 E1 projection보다 tail 감소율 `>=.10`p 우수;
- local 및 broad defect contrast retention 각각 `>=.95`;
- pair-distance Spearman과 NN identity retention 각각 `>=.95`;
- local 및 broad location conditioner cosine `>=.95`, 상대 drift `<=.10`;
- E3를 선택하려면 추가로 E3-E2 tail reduction `>=.05`p이며 E3 log-scale RMS drift
  `<=.10`; 그렇지 않으면 더 단순한 E2를 우선한다.

`>=3/4` object가 E2 또는 E3로 pass하면 `PROMISING_CONDITIONAL_DISTRIBUTION_ALIGNMENT`이며,
이때도 다음 단계는 작은 learnable conditional module의 별도 사전등록일 뿐이다. `<=2/4`이면
`STOP_CONDITIONAL_ALIGNMENT_FAMILY`. E2는 통과하지만 E3의 추가 gate가 실패하면
`PREFER_LOCATION_ONLY_NORMALIZATION`으로 판정하고 Flow Matching/Diffusion/DiT 복잡도 확장을
허용하지 않는다.

로컬 스모크 후 local/dsba5의 동일 script·manifest·ledger와 metric parity(절대 `1e-3` 또는 상대
`1%`)를 확인한다. local에서 명백한 hard-stop이 나더라도 코드/입력 parity 확인용 최소 remote
lane은 완료하되 추가 sweep은 수행하지 않는다.
