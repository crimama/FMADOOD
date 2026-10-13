# 조건부 정규화·수송 Stage -1e 결과

Date: 2026-08-07  
Verdict: `STOP_CONDITIONAL_ALIGNMENT_FAMILY / DO_NOT_EXPAND_FLOW_MATCHING_DIFFUSION_DIT`

## 무엇을 검증했는가

global shift를 feature에서 빼는 대신, query의 robust channel 통계로 정상 support memory를 query
domain으로 옮겼다. E2는 median 위치만, E3는 median과 clipped MAD scale을 이동했다. Stage -1c의
rank-1 projection은 알려진 negative control E1으로 두었다. 네 arm 모두 frozen DINOv3 feature와
raw 1-NN을 사용했고 학습·parameter update·test/label 접근은 없었다.

## 사전등록 결과

validity와 local/dsba5 parity는 통과했지만 E2/E3 object pass는 모두 `0/4`였다.

| Object | E0 U inflation | E1 tail reduction | E2 tail reduction | E3 tail reduction | E2 local / broad contrast | E2 local / broad conditioner drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fruit_jelly | .000917 | -.0002 | -.4531 | -.7693 | .9399 / .9639 | .4233 / .7794 |
| walnuts | -.001051 | .0141 | -.3333 | -.2842 | .9677 / .8950 | .2175 / .5782 |
| vial | .005046 | .0758 | -.0820 | -1.8130 | .9494 / .9599 | .6797 / .8072 |
| fabric | .000761 | -.1039 | .0537 | .2257 | .9867 / .9140 | .2672 / .6449 |

- E2는 세 object에서 shifted-normal tail을 악화시켰고 `fabric`의 유일한 양의 효과도 `5.4%`로
  `.15` gate에 미달했다.
- E3는 `fabric`에서 `22.6%` 감소했지만 E2 대비 추가 margin은 음수(`-.1053`)였고 broad defect
  contrast가 `.8503`, NN retention이 `.9434`로 안전 gate를 함께 실패했다.
- E2 translation은 모든 object에서 pair Spearman/NN retention `1.0`이었다. 즉 실패는 구현이
  clean geometry를 깨뜨려서가 아니라, query 전체 patch median이 정상 domain condition을 충분히
  식별하지 못했기 때문이다.
- 결함 query로 conditioner를 다시 추정하면 E2 location drift가 `.218--.807`이었다. 사전등록
  상한 `.10`을 모든 object가 넘었다. 특히 broad defect는 정상적인 global shift와 conditioner
  수준에서 분리되지 않았다.

E0 inflation 자체가 일부 fold/object에서 0에 매우 가까워 개별 상대 감소율이 크게 흔들렸다.
이 문제를 결과 확인 후 다른 분모로 바꾸지 않았고, raw inflation과 object median을 함께
기록했다. 작은 E0 값에도 불구하고 결론은 tail gate 하나에 의존하지 않는다. defect contrast와
conditioner-contamination gate도 독립적으로 실패했다.

## 해석

이번 결과가 부정하는 것은 **query 전체 patch의 channel median/MAD를 global condition으로 보고
정상 memory 전체를 동일하게 옮기는 naive moment-alignment 구조**다. “target-specific module은
필요 없다”거나 모든 conditional Flow/Flow Matching/Diffusion/DiT가 불가능하다는 뜻은 아니다.

다만 더 복잡한 생성·수송 모델을 즉시 붙일 근거도 생기지 않았다. 복잡도보다 먼저 해결해야 할
병목은 transport family가 아니라 **condition identifiability**다. query aggregate에는 photometric
shift뿐 아니라 object content와 defect-induced structural shift가 함께 들어가므로, conditioner가
둘을 분리하지 못하면 더 강한 transport는 오히려 결함을 정상 memory 쪽으로 흡수할 수 있다.

따라서 현재 허용되는 다음 질문은 “어떤 architecture가 더 강한가?”가 아니라 “mask/label 없이
defect에 불변인 global condition을 support에서 식별할 수 있는가?”다. 이 식별 gate가 생기기
전에는 Flow Matching, Diffusion, DiT를 naive하게 비교하지 않는다. 이후 새로운 conditioner는
E2/E3를 negative control로 두고, 동일한 tail·contrast·geometry·contamination 동시 gate를 먼저
통과해야 한다.

## 실행 및 재현성

- local: GPU 0, `hun_fsad_tta_local_stage1a0`.
- dsba5: host GPU 3 / container cuda:0, `hun_fsad_tta_012` immutable snapshot.
- runner SHA256: `714b2243d007f32c5ac8147a71a0c5801c098ea9966c9ef1ee5142187a4dad50`.
- support manifest SHA256:
  `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`.
- local/dsba5 parity: 모든 비교 metric 허용오차 내, mismatch `0`.
- test public/private 및 ground truth 접근: 모두 `false`.
- final summary:
  `results/stage_minus1/conditional_normalization_transport/local_v1/mechanism/summary.json`.
- remote evidence:
  `results/remote_runs/dsba5/cond_norm_transport_stage_m1e_20260807_v1`.

