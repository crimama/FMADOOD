# DVT Alignment × Defect-Scale Attribution 결과

Date: 2026-08-08
Protocol: `dvt-alignment-defect-scale-attribution-all8-20260808`
Verdict: `ALIGNMENT_SCALE_ATTRIBUTION_SUPPORTED` (mechanism diagnostic only)

## 1. Negative Evidence Intake

이번 실험은 폐기된 query registration, alignment loss, condition estimator, DVT parameter sweep
또는 score fusion을 다시 시도하지 않았다. 기존 `fused+Flow`와 `DVT+Flow` score map을 그대로
생성하고, DVT의 all-8 이득이 정상 support의 공간 정렬도와 defect component 크기에 따라 달라지는지
사후 귀인했다. GT와 filename/condition metadata는 scoring과 alignment grouping에 사용하지 않았다.

## 2. Motivation

이전 all-8 결과는 DVT가 AP, full AUROC, top-5% coverage를 개선하지만 top-1% coverage는
악화함을 보였다. 이번 질문은 이 비대칭을 다음 가설로 설명할 수 있는지였다.

> 위치별 정상 template이 안정적인 객체에서 DVT가 더 유효하며, 특히 너무 작지 않은 defect에서
> ranking과 moderate-area coverage 이득이 커진다.

## 3. Implementable Design

- MVTec AD2 single-image, all 8 objects, full TESTpublic
- fixed SuperAD-selected support-16, DINOv3-H+ `[7,15,23,31]`
- common Flow/raw latent 1-NN; DVT off vs position-mean DVT on
- support-only alignment: leave-one-out position mean이 global mean보다 held-out normal feature를
  얼마나 더 잘 설명하는지 측정한 median ratio
- alignment 상위/하위 각 4개 객체는 GT를 열기 전에 고정
- GT component scale: small `<=1%`, medium `1--5%`, large `>5%`
- primary endpoint: scale별 paired per-image pixel AP; secondary full AUROC, top-1/5% coverage,
  component hit
- 다른 scale의 defect pixel은 scale별 AP/AUROC에서 제외

## 4. Evaluation Alignment

이 실험은 Baseline1 내부 메커니즘 진단이며 새 method comparison이 아니다. 따라서
`superad_seg_AUROC_0.05=N/A`, `superad_seg_F1=N/A`, `RN-FMLK=N/A`, `comparable=false`다.
대신 동일 all-8 DVT off/on 결과의 replication을 먼저 요구했다. AP/full AUROC는 `7/8`,
top-5% coverage는 `6/8` 객체에서 양수였고 모든 replication gate가 다시 통과했다.

## 5. Code Modification / Creation

- `src/flow_tte_phase3_scorer_suite.py`: 8-connected component scale binning, 다른-scale pixel
  exclusion, compact per-image AP/AUROC/top-area coverage/component hit 추가
- `scripts/run_flow_tte_phase3_scorer_suite.py`: GT/test 이전 support-only leave-one-out alignment 추가
- `scripts/aggregate_dvt_alignment_defect_scale_attribution.py`: paired image→object→macro aggregation,
  custody, scale/alignment gates와 frozen verdict 구현
- `tests/test_flow_tte_phase3_scorer_suite.py`,
  `tests/test_aggregate_dvt_alignment_defect_scale_attribution.py`: boundary, exclusion, tie,
  underpowering, verdict 및 실제 JSON schema 회귀검증

## 6. Added Code Evaluation

- focused tests: `28 passed`
- Ruff: pass
- Python compile: pass
- exact support/alignment custody across arms: pass
- tie-inclusive selected area `<2x` nominal: pass, violation `0`
- metadata boundary: pass
- 실행 중 실제 evaluator/aggregator schema mismatch를 발견했으며, 결과 판정 전에 실제
  `per_bad_image_scale_rows` schema 회귀 테스트를 추가해 수정했다.

## 7. Remote Execution

- local: `hun_fsad_tta_local_stage1a0`, RTX 4090
- dsba5: `hun_fsad_tta_012`, host GPU3/container CUDA0
- data roots: local `/home/hun/Volume/DATA/mvtec_ad_2`, remote
  `/home/hunim/Volume/DATA/mvtec_ad_2`
- 양 환경 모두 8 objects × 2 arms complete
- aggregate continuous metrics 400개 worst absolute delta `6.3310e-6 < 1e-5`
- verdict, alignment group, boolean gate mismatch `0`
- anomaly-map TIFF 및 `anomaly_maps/` 보존 없음

## 8. SuperAD Baseline and Unified Metrics

| Scope | Baseline | Candidate | Primary metric | Comparable |
| --- | --- | --- | --- | --- |
| mechanism | fused+Flow, DVT off | DVT+Flow | paired scale pixel AP | yes |
| method claim | same-condition SuperAD-16 | N/A | seg AUROC/F1 | no |

전체 defect를 합친 replication delta는 AP `+0.010637`, full AUROC `+0.008167`, top-5%
coverage `+0.006349`다. 이는 이전 all-8 결과와 일치하며 이번 scale 귀인의 전제는 통과했다.

## 9. Results and Analysis

### Support-only alignment grouping

| Rank | Object | Alignment score | Group |
| ---: | --- | ---: | --- |
| 1 | vial | 0.78543 | aligned |
| 2 | fruit_jelly | 0.57903 | aligned |
| 3 | sheet_metal | 0.56142 | aligned |
| 4 | walnuts | 0.55391 | aligned |
| 5 | can | 0.52619 | misaligned |
| 6 | rice | 0.43021 | misaligned |
| 7 | fabric | 0.34417 | misaligned |
| 8 | wallplugs | 0.30501 | misaligned |

### Scale macro: DVT on - off

| Scale | Eligible objects | Δ AP | Δ full AUROC | Matched coverage | Component hit | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| small `<=1%` | 8 | +0.00070 | +0.00235 | top-1 `-0.01153` | top-1 `-0.00607` | fail |
| medium `1--5%` | 6 | +0.07571 | +0.04317 | top-5 `+0.07321` | top-5 `+0.01250` | pass |
| large `>5%` | 2 | +0.05975 | +0.02055 | top-5 `+0.01319` | top-5 `0` | ineligible |

Medium defect는 AP 양수 `5/6`, object collapse 없음, coverage와 component hit no-harm을 모두
통과했다. Small defect는 AP가 거의 0이고 top-1% coverage가 감소했으며 `fruit_jelly`
`-0.03996`, `sheet_metal` `-0.04807` AP로 object-collapse gate를 실패했다.

Medium bin에서 aligned group은 AP `+0.10908`, top-5% coverage `+0.10830`; misaligned group은
AP `+0.00896`, coverage `+0.00302`였다. 사전등록 alignment modulation gate는 통과했다.
다만 aligned group은 4개 모두 eligible인 반면 misaligned group은 `rice`, `wallplugs` 2개만
eligible하다. 따라서 차이의 방향은 유의미한 diagnostic evidence지만, 균형 잡힌 4-vs-4 인과
효과로 과해석하지 않는다.

## 10. Continuation Assessment

엄격한 method claim은 지원하지 않는다. 그러나 medium-scale에서 scale gate와 alignment gate가
동시에 통과했고, local/dsba5에서 그대로 재현되어 하나의 bounded causal diagnostic은 허용된다.

허용되는 다음 실험은 동일 support/scorer에서 다음 고정 negative controls만 비교하는 것이다.

1. raw/no centering
2. current positional artifact `position_mean - global_mean`
3. position-shuffled artifact
4. global-only centering control

Primary gate는 medium-scale AP/coverage에서 current positional artifact가 global-only와 shuffled
position을 동시에 이기는 것이다. Small-defect AP/top-1 coverage no-harm을 함께 요구한다. 실패하면
alignment-conditioned gating, score fusion, centering 튜닝과 구조 확장을 중단한다.

## 11. Conclusion

Verdict는 `ALIGNMENT_SCALE_ATTRIBUTION_SUPPORTED`다. DVT의 일반 이득은 모든 defect scale에
균일하지 않다. 가장 강한 신호는 **정상 support의 position-specific template 설명력이 높고,
defect가 이미지의 1--5%인 경우**다. 반대로 `<=1%` 작은 defect에서는 peak localization이
약화될 수 있다.

따라서 DVT를 단순 shift corrector나 universal peak sharpener로 설명하지 않는다. 현재 허용되는
더 구체적인 설명은 `object-position prior that improves medium-scale defect ranking/coverage when
support spatial structure is repeatable`이다.

## 12. Post-Conclusion Storage Cleanup

- local/remote object JSON, summaries, logs와 parity만 보존했다.
- 두 결과 root 아래 `anomaly_maps/`가 없음을 확인했다.
- 원격 두 arm status는 `complete`; 관련 GPU process는 종료됐다.

Artifacts:

- preregistration: `skill_graph/experiments/2026-08-08_dvt_alignment_defect_scale_attribution/preregistration.md`
- local summary: `results/dvt_alignment_defect_scale_attribution_20260808/local/summary.json`
- remote summary: `results/remote_runs/dsba5/dvt_alignment_defect_scale_attribution_20260808_dsba5/summary.json`
- parity: `results/dvt_alignment_defect_scale_attribution_20260808/parity.json`

