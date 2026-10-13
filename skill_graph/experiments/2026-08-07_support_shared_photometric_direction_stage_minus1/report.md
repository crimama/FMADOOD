# Support-shared Photometric Direction Stage -1b

Date: 2026-08-07  
Verdict: `PROMISING_DIAGNOSTIC / CONTINUE_DESIGN_ONLY / NO_PUBLIC_ACCESS`

## Negative Evidence Intake

이 실험은 Stage -1a1의 conditional-smoothness threshold를 바꾸지 않았고, 이전에 닫힌
FiLM/projection/atlas/paired-loss/WTCF branch도 재시도하지 않았다. Flow, 1-NN score, query
adaptation 또는 학습 가능한 module 없이 support가 shared nuisance direction을 제공하는지만
측정했다.

## Motivation

Stage -1a1은 spatially coherent global prior를 지지하지 못했다. 이에 독립적으로, 동일한
photometric nuisance가 서로 다른 normal support image에서 재현 가능한 feature direction을
만드는지 조사했다. 이 질문은 global canonicalization의 충분조건이 아니라 그 방향을 support로
식별할 수 있는지에 대한 최소 필요조건이다.

## Implementable Design

고정 normal support 16장과 DINOv3 ViT-H/16+ fused feature의 clean-to-changed patch-average
displacement를 direction으로 정규화했다. 4-fold에서 held image의 direction과 나머지 12장의
동일-condition mean direction cosine을 cross-image transfer로 정의했다.

- shared conditions: `S_global`, `S_smooth`
- null: source path hash로 고정된 서로 다른 전역 photometric transform `S_source_random`
- target: MVTec AD2 single-image, `fruit_jelly/walnuts/vial/fabric`; support-only
- SuperAD/segmentation comparison: `N/A` (method claim이 아닌 Stage -1 diagnostic)

## Evaluation Alignment

defect mask, test public/private, ground truth를 사용하지 않았다. 따라서 이 결과는 정상
support가 synthetic photometric intervention의 shared direction을 보유한다는 evidence일 뿐,
unknown real shift의 분리나 defect 안전성, SuperAD 우월성을 뜻하지 않는다.

## Code Modification / Creation

- `scripts/run_support_shared_photometric_direction_stage_minus1.py`
- `scripts/aggregate_support_shared_photometric_direction_stage_minus1.py`
- `scripts/run_global_local_conditional_smoothness_stage_minus1_remote.sh`의 runner override
- [Preregistration](preregistration.md)

## Added Code Evaluation

Host `py_compile`, shell syntax, aligned local container smoke를 통과했다. local/dsba5 compact
checksum manifest가 모두 통과했고, runner hash, support-manifest hash, access ledger가 일치했다.
모든 aggregate parity 차이는 absolute `1e-3` 또는 relative `1%` 이내였으며 최대값은
`1.41e-6`이다.

## Remote Execution

- local: host GPU 0, `hun_fsad_tta_local_stage1a0`
- dsba5: host GPU 3 / container CUDA 0, `hun_fsad_tta_012`, immutable v2 snapshot
- source snapshot은 pullback/checksum 검증 후 제거했다.

## SuperAD Baseline and Unified Metrics

`BLOCKED_BY_SCOPE / N/A`: 이 experiment는 score map이나 segmentation output을 만들지 않으며
method candidate가 아니다. 따라서 SuperAD/RN-FMLK/`seg_AUROC_0.05`/`seg_F1` 비교를 주장하지
않는다.

## Results and Analysis

| Object | Global transfer | Smooth transfer | Random null | Global margin | Smooth margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| fruit_jelly | .9863 | .9915 | .7121 | .2752 | .2790 |
| walnuts | .9518 | .7780 | .1518 | .7805 | .6197 |
| vial | .9808 | .9935 | .9023 | .0829 | .0918 |
| fabric | .9610 | .9428 | .5092 | .4093 | .4341 |

모든 object에서 두 shared condition의 transfer는 `.20`을 넘고 source-random null보다 `.05`
이상 높았다 (`4/4`). 이는 normal support가 같은 global photometric intervention에 대해 image
content를 넘는 공유 방향을 가진다는 양성 diagnostic이다. 특히 `vial`은 margin이 작지만
사전등록 최소치보다 높아, category별 nuisance identifiability 강도가 다름도 드러낸다.

이것이 Stage -1a1과 모순되지는 않는다. shared direction은 image 간 평균 displacement level에서
재현될 수 있지만, 그 direction이 각 image 안에서 locally smooth한 field일 필요는 없다. 따라서
coverage/smoothness 하나만으로 global nuisance를 고르는 방식은 여전히 부정적이다.

## Continuation Assessment

엄격한 method claim은 지지하지 않는다. 다만 다음 한 번의 bounded diagnostic은 정당화된다:
12-shot support에서 만든 shared direction을 **사전등록된 unseen photometric transform**에
transfer했을 때, held normal support의 변위를 줄이면서 synthetic local/broad defect contrast를
보존하는가. 이 단계도 support-only이며 Flow/Flow-Matching/Diffusion/DiT module 구현과 public
benchmark를 열지 않는다. 해당 transfer/retention gate가 실패하면 normal-support shared-direction
family를 hard-stop한다.

## Conclusion

`PROMISING_DIAGNOSTIC / CONTINUE_DESIGN_ONLY`. Stage -1b는 normal support 안에 repeatable
photometric direction evidence가 있음을 보였다. 그러나 이 evidence는 nuisance/defect selector나
성능 향상 증거가 아니므로 Baseline1을 유지하며 새 module branch는 차단한다.

## Post-Conclusion Storage Cleanup

Dense anomaly map을 생성하지 않았다. compact local/remote result, checksum, manifest, ledger,
summary만 보존했고 dsba5 immutable source snapshot을 제거했다.
