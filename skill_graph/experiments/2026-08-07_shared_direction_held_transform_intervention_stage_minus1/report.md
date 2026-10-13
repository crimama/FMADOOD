# Shared-direction Held-transform Intervention Stage -1c

Date: 2026-08-07  
Verdict: `KILL_FOR_CLAIM / NO_CONTINUE_SHARED_DIRECTION_INTERVENTION / NO_PUBLIC_ACCESS`

## Motivation and design

Stage -1b의 cross-image shared photometric direction을 fixed rank-1 projection으로 한 번만
제거했다. 학습, rank/strength sweep, test access는 없었다. 12 support images로 `S_global`
direction을 만들고, 다른 fixed unseen photometric transform의 held 4 images에 적용했다.
source-random direction은 null이다. 목표는 raw 1-NN U-tail inflation 감소와 combined synthetic
local/broad defect contrast 및 clean geometry 보존의 동시 확인이었다.

## Result

Defect contrast retention (`.970`--`1.012`)과 geometry (pair Spearman `.993` 이상, NN `.966`
이상)는 보존됐다. 그러나 핵심 normal-shift reduction은 모든 object에서 `.02` gate를 크게
밑돌았다.

| Object | Shared reduction | Random reduction | Shared−random margin |
| --- | ---: | ---: | ---: |
| fruit_jelly | .00026 | -.00006 | .00045 |
| walnuts | .00016 | .00006 | .00011 |
| vial | .00029 | .00040 | .00018 |
| fabric | -.00024 | .00074 | -.00083 |

즉 support-shared direction은 feature displacement descriptor 수준에서는 재현됐지만, fixed
projection으로 제거해도 deployed raw-1NN tail을 줄이지 못했다. `fabric`에서는 random null이
더 좋았다. Stage -1b evidence를 nuisance-removal interface나 module 설계 근거로 승격하지 않는다.

## Execution and custody

local GPU 0 compact artifact checksum은 전부 통과했고 test/public/private/ground-truth 접근은
없었다. dsba5 v2는 local hard-stop이 `4/4` 확정된 뒤 추가 GPU 사용을 중단했으므로 parity
claim은 하지 않는다. immutable remote source snapshot은 정리한다.

## Conclusion

shared-direction intervention family는 이 fixed raw-feature projection operationalization에서
hard-stop한다. 남는 자산은 “cross-image direction reproducibility는 anomaly-score reduction의
충분조건이 아니다”라는 failure boundary다. Flow/Flow-Matching/Diffusion/DiT module과 public
benchmark는 열리지 않는다.
