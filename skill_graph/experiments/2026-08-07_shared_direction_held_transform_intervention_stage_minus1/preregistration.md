# Shared-direction Held-transform Intervention Stage -1c 사전등록

## Negative evidence intake

이 실험은 닫힌 global rank projection/FiLM/atlas branch의 parameter retune이 아니다. 학습
가능 parameter, rank sweep, projection strength sweep, test-time adaptation, score fusion을 쓰지
않는다. Stage -1b에서 `4/4`로 확인된 **support-shared photometric direction** 하나를 이용해,
unseen transform에서 방향 성분을 정확히 한 번 제거하는 fixed intervention probe다.

목적은 method 성능 주장이 아니라 다음 반증 가능 질문이다. “Stage -1b direction이 실제로
unseen global photometric displacement를 줄이면서 defect contrast를 보존하는가?” 답이 아니면
support-shared direction family를 hard-stop한다.

## Fixed protocol

- MVTec AD2 single-image, fixed 16 normal support, `fruit_jelly/walnuts/vial/fabric`, 4-fold
  (12 direction/memory fit, 4 held).
- DINOv3 ViT-H/16+ fused patch feature. Static raw-Euclidean patch 1-NN only; Flow fitting,
  DVT, density, query adaptation, test/public/private/ground truth는 off.
- fit direction \(u\): 12 support image의 Stage -1a0 `S_global` clean-to-changed patch-average
  displacement 평균을 L2-normalize.
- null direction \(u_r\): 동일 support의 source-random photometric displacement 평균을
  L2-normalize.
- unseen condition \(U\): `brightness=.91`, `contrast=.96`, `gamma=.93`,
  `rgb_gain=(1.04,.98,1.02)`. 이는 fit `S_global`과 다른 고정 intervention이다.
- defect safety conditions: `U+D_local`, `U+D_broad`; masks are used **only** for contrast
  retention, never to fit/select \(u\).
- fixed operation: every feature vector \(h\) is replaced once by
  \(P_u(h)=h-(h^Tu)u\). 모든 memory/query에 같은 P를 적용한다. strength/rank는 1이며 tune하지
  않는다.

## Metrics and gate

- normal effect: held U q995 relative inflation; `shared_reduction = raw - P_u`,
  `random_reduction = raw - P_{u_r}`.
- safety: each combined-condition patch defect contrast의 `P_u/raw` retention.
- geometry: clean fit-bank raw-to-projected pair-distance Spearman 및 1-NN retention.

Object pass requires all: shared U reduction `> .02`, shared reduction > random reduction `+.02`,
local/broad retention `>= .95`, pair Spearman and NN retention `>= .95`. `>=3/4` and two-lane
hash/ledger parity (absolute `1e-3` or relative `1%`) gives only
`PROMISING_DIAGNOSTIC / CONTINUE_DESIGN_ONLY`; otherwise
`KILL_FOR_CLAIM / NO_CONTINUE_SHARED_DIRECTION_INTERVENTION`. Neither outcome permits public
benchmark or an architecture implementation.
