# Baseline1 k=5 scorer default A/B 결과

날짜: 2026-08-07  
판정: `RETAIN_K1 / K5_NOT_DEFAULT / NOT_A_METHOD_CONTRIBUTION`

## 실행

고정 SuperAD-16 support와 Baseline1 DINOv3-H+·DVT·Flow를 공유하고 최종
거리 집계만 1-NN과 5-NN mean으로 바꿨다. MVTec AD2 TESTpublic 8개 객체를
모두 평가했다. dsba5 병렬 shard는 invalid v2/v3 자식 프로세스가 shell 종료 뒤
남아 GPU를 점유한 상태에서 후속 run이 겹쳐 OOM이 발생했으므로 폐기했다. 최종
유효 artifact는 동일 로컬 컨테이너의 두 순차 shard에서 exit 0으로 생성했다.

원본 해상도 map을 디스크에 남기지 않으면서 AU-PRO를 계산하기 위해 float16의
유한한 score domain을 이용한 exact weighted-histogram evaluator를 추가했다.
작은 tied-score fixture에서 기존 full-sort F1/AP/pAUROC/PRO와 각각 exact 또는
부동소수점 `6e-9` 이내 일치를 확인했다.

## Macro 결과

| arm | seg_AUROC_0.05 | seg_F1_raw | seg_AUPRO_0.05 |
| --- | ---: | ---: | ---: |
| K1 | 0.839561 | 0.530177 | 0.511573 |
| K5 | 0.835339 | 0.527891 | 0.508115 |
| K5 - K1 | -0.004222 | -0.002285 | -0.003457 |

K5는 F1과 AU-PRO의 macro no-harm tolerance는 통과했지만, pAUROC 허용치
`-0.002`보다 큰 `-0.004222` 하락으로 주 gate를 실패했다.

## 객체별 변화

| object | Δ pAUROC | Δ F1 | Δ AU-PRO |
| --- | ---: | ---: | ---: |
| can | -0.002202 | -0.000353 | -0.005120 |
| fabric | +0.005767 | +0.035040 | -0.029530 |
| fruit_jelly | -0.011735 | -0.023678 | +0.014070 |
| rice | -0.003754 | -0.003807 | -0.017373 |
| sheet_metal | -0.007631 | -0.023771 | -0.030263 |
| vial | -0.013964 | -0.011475 | -0.011377 |
| wallplugs | +0.000951 | +0.000647 | +0.038671 |
| walnuts | -0.001211 | +0.009115 | +0.013262 |

`sheet_metal` AU-PRO가 `-0.030263`으로 객체별 `-0.03` floor를 넘겨
collapse gate도 실패했다. 8개 중 6개 객체에서 pAUROC가 감소해 손실이 한
객체의 우연한 결과로 설명되지 않는다.

## 정상 tail과 해석

K5의 macro good-pixel p99.9는 K1보다 `48.4553` 낮아져 의도한 outlier 완화
효과는 관측됐다. 그러나 oracle threshold에서 정상-image pooled FPR은
`+0.000325` 증가했다. 즉 K5는 score scale의 극단값은 줄이지만 정상/결함
ranking을 일관되게 개선하지 않는다.

따라서 k=5를 기본값으로 바꾸지 않는다. k를 추가로 sweep하거나 객체별로
선택하지도 않는다. 논문 방법론 contribution은 계속 target-specific module의
requirement와 구조에 두고, scorer는 현재 Baseline1의 k=1을 유지한다.

## 재현성과 제한

- 최종 artifact: `results/baseline1_knn5_default_ab_20260807_v5/summary.json`.
- coverage: 8/8 객체, 양 arm 모두 finite이며 map/GT shape 및 count 완결.
- K5는 2026-07-12 scorer-suite 결과를 수치 오차 수준으로 재현했다.
- 현재 K1 absolute row는 과거 anchor parity를 8개 중 2개만 통과했다. 이는
  현재 `score_flow_memory` raw calibration 경로와 과거 frozen artifact 사이의
  code-evolution 차이를 별도 추적해야 함을 뜻한다. 이번 결론은 현재 운용
  scorer의 default 변경 여부에 한정한다. K5가 이미 두 독립 실행 모두에서
  pAUROC/F1 우위를 보이지 않았으므로 default 보류 판정은 바뀌지 않는다.
- 호스트와 컨테이너에 pytest가 없어 전체 pytest suite는 실행하지 못했다.
  py_compile, scorer subset smoke, histogram-vs-full-sort parity, 두 shard exit 0,
  8-object aggregator 검증을 수행했다.
- 종료 시 dsba5의 실험 자식/대기 프로세스를 확인해 종료하고, 재생성 가능한
  11GB snapshot과 invalid remote result roots를 삭제했다. 최종 compact 결과는
  로컬 v5 root에만 보존했다.

## Gate

| check | result |
| --- | --- |
| macro pAUROC no-harm | FAIL |
| macro F1 no-harm | PASS |
| macro AU-PRO no-harm | PASS |
| no object collapse | FAIL |
| normal-tail improvement | PASS |
| coverage complete | PASS |

최종 판정은 `RETAIN_K1`이다.
