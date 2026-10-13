# Baseline1 DVT all-8 threshold-free ranking/coverage 사전등록

Date: 2026-08-08
Protocol: `baseline1-dvt-all8-rank-coverage-20260808`
Scope: all-object mechanism diagnostic; not a new method claim

## Negative Evidence Intake

앞선 four-object condition attribution은 DVT가 shifted-normal FPR보다 defect-pixel TPR을
주로 높인다는 결과를 냈다. 그러나 효과가 oracle threshold에 의존하고 `fabric`의 큰 향상에
의해 macro가 부풀었을 가능성이 남는다. 이번 실험은 새 scorer, threshold, postprocess 또는
metadata branch가 아니라, 동일한 DVT off/on anomaly map을 all-8에서 threshold-free ranking
관점으로 재평가하는 단일 진단이다. 따라서 기존 killed score-tuning branch의 재시도가 아니다.

## Motivation과 질문

다음 두 가설을 구분한다.

- H1: DVT가 객체 전반에서 defect pixel의 상대 순위를 올리고 fixed area budget 안의 coverage를
  증가시키는 object-aligned localization prior다.
- H0: 이전 TPR 증가는 `fabric` 또는 arm별 oracle threshold 선택이 만든 제한적 현상이다.

## 고정 실험 구성

- target: MVTec AD2 single-image.
- objects: `can, fabric, fruit_jelly, rice, vial, wallplugs, walnuts, sheet_metal`.
- split: full `test_public/good,bad`; train/test cap 없음.
- support: fixed SuperAD-selected 16 normal images/object, 동일 manifest/seed.
- representation: DINOv3-H+/16 `[7,15,23,31]`, normalized layer mean.
- resolution: 672, 단 `sheet_metal=448` dataset-default.
- common module: 3-epoch patch-local affine Flow, raw latent 1-NN, density weight 0.
- arms: `fused_flow` (DVT off)와 `dvt_flow` (position mean, alpha 1.0).
- no query adaptation, memory update, metadata-conditioned inference, morphology.

본 비교는 Baseline1-core 내부 mechanism contrast다. 같은-condition SuperAD-16 artifact
`results/remote_runs/dsba3/same16_baseline_metric_contract_20260723_v6/summary.json`은
baseline custody 문맥으로만 보존하며, DVT off/on threshold-free 지표가 SuperAD의 공식 출력
metric이 아니므로 직접 우월성 gate에 사용하지 않는다. `KEEP` 주장을 허용하지 않는다.

## 사전등록 threshold-free 지표

점수 threshold와 condition/file metadata를 사용하지 않고, bad image 내부의 score ranking과 GT만
사후 평가한다.

1. `bad_image_pixel_ap_mean`: bad image별 defect-vs-background pixel AP의 macro 평균.
2. `bad_image_pixel_auroc_mean`: bad image별 full pixel AUROC의 macro 평균.
3. `defect_pixel_coverage_at_top_1pct`, `...top_5pct`: 각 bad image에서 score 상위 고정
   면적 budget에 포함된 defect pixel 비율의 macro 평균.
4. `component_hit_rate_at_top_1pct`, `...top_5pct`: 같은 top-area mask가 닿은 GT connected
   component 비율.

Top-area boundary의 동점은 index 순서로 자르지 않는다. cutoff score와 같은 모든 pixel을 포함하는
conservative tie-inclusive policy를 사용하고 실제 selected fraction을 함께 기록한다. 이는 score
calibration threshold가 아니라 고정 rank/area budget의 coverage 진단이다.

## Gate

Primary endpoint는 `bad_image_pixel_ap_mean`이다.

- `GENERAL_LOCALIZATION_PRIOR_SUPPORTED`:
  - AP delta macro `>0` 및 positive object `>=6/8`;
  - AUROC delta macro `>0` 및 positive object `>=6/8`;
  - top-5% defect coverage delta macro `>0`;
  - top-5% component hit delta `>=-0.01`;
  - 어느 객체도 AP `<-0.03` 또는 top-5% coverage `<-0.05` collapse가 없음;
  - `fabric`을 제외한 7-object macro AP와 coverage delta도 모두 `>0`.
- 위 조건 일부만 통과하면 `OBJECT_DEPENDENT_DEFECT_RANKING`.
- AP/AUROC/coverage가 `<=4/8` 방향성이거나 non-fabric 효과가 사라지면
  `THRESHOLD_OR_FABRIC_DEPENDENT`.

Primary/secondary 결과와 함께 기존 `seg_AUROC_0.05`, oracle `seg_F1`, AU-PRO, map coverage를
custody check로 보고하지만, threshold-free mechanism verdict를 대신하지 않는다.

## 실행과 재현성

- local 4090과 dsba5 GPU3/container CUDA0에서 가능한 arm/object를 병렬 실행한다.
- local/dsba5 핵심 지표 worst absolute delta `<=1e-5`를 parity gate로 사용한다.
- anomaly map은 RAM에서만 평가하고 TIFF를 남기지 않는다.

## 주장 한계와 단일 후속 계약

이 실험은 DVT의 localization mechanism을 판별할 뿐 SuperAD/RN-FMLK 우월성이나 새 방법 기여를
주장하지 않는다. 지지될 경우에만 다음 한 단계로 DVT가 어느 spatial frequency/region에서 순위를
올리는지 mask-free rank-gain map diagnostic을 허용한다. 실패하면 DVT를 일반 shift/localization
module로 확장하지 않고 Baseline1의 object-dependent engineering component로 한정한다.
