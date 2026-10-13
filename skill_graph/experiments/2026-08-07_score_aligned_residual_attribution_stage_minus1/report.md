# Score-aligned Residual Attribution + kNN Scorer Sentinel Stage -1d

Date: 2026-08-07  
Verdict: `KILL_FOR_CLAIM / NO_FOUR_OBJECT_CONTINUATION / NO_PUBLIC_ACCESS`

## Design

Fixed-16 support, 4 folds, deterministic 256 patches, sentinel objects `fruit_jelly/vial`에서
support `S_global/S_smooth` displacement의 image-mean을 제거한 rank-4 subspace가 held unseen
shift의 actual nearest-memory residual을 설명하는지 측정했다. 동일 memory/query로 1-NN,
5-NN mean, 15-NN mean, fixed-temperature soft-min-15를 함께 기록했다. rank/scorer tuning,
Flow/DVT/query adaptation/test/GT는 없었다.

## Result

| Object | Residual capture | Random | Margin | 1NN reduction | 5NN | 15NN | Soft-min | Local/Broad retention | NN geometry |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fruit_jelly | .0550 | .0024 | .0525 | -.1096 | .0207 | -.0649 | .1221 | .9480/.9888 | .9453 |
| vial | .0747 | .0027 | .0722 | -.0018 | .0234 | -.0186 | .0051 | .9551/.9920 | .9160 |

Rank-4 nuisance subspace는 random보다 residual을 더 설명했지만 사전등록 margin `.10`에 못
미쳤다. 더 결정적으로 primary 1-NN inflation은 감소하지 않았고, 5-NN mean은 두 object에서
약 `2%`의 작고 일관된 감소만 보였다. 15-NN은 모두 악화됐고 soft-min의 `fruit_jelly` 양성은
`vial`에서 재현되지 않았다. geometry NN retention도 두 object 모두 `.95` 아래였다.

일부 row는 raw inflation이 0에 가까워 relative-reduction 값이 불안정했지만, object median의
primary failure, insufficient capture margin, geometry failure가 동시에 있어 이 validity issue가
판정을 바꾸지 않는다. compact checksum과 support-only custody는 통과했다. sentinel gate가
두 object 모두 실패해 dsba5/four-object scaling은 실행하지 않았다.

## Conclusion

현재 support-estimable rank-4 residual subspace는 score-aligned nuisance를 충분히 포착하지
못한다. Scorer 측면에서는 1-NN만의 특수한 실패는 아니다. k=5 averaging은 tail variance를
조금 낮추지만 효과가 작고, k=15는 오히려 정상 manifold의 다른 mode까지 섞어 악화된다.
soft-min도 object-dependent하다. 따라서 k 선택 sweep을 다음 연구 branch로 열지 않는다.

향후 scorer를 바꾸려면 단순 k-NN 평균보다 local-density calibrated distance 또는 learned
normal conditional likelihood처럼 명시적인 density/uncertainty 가설이 필요하며, 동일한
defect-retention·geometry·hard-null gate를 다시 가져야 한다.
