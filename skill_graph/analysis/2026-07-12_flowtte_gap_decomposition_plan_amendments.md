# 2026-07-12 갭 분해·스코어 등가화 실험 계획 수정안

원 계획: 사용자 제공 "FlowTTE 구조 분석 및 성능 개선 실험 계획안" (Phase 0-8).
본 문서는 실행 전 검토에서 확정한 수정 사항만 기록한다. 원 계획의 게이트,
KEEP 조건, 중단 목록은 그대로 유지한다.

## A1. Phase 0 전제 수정 (필수)

원 계획은 "저장된 anomaly map을 사용"한다고 가정하나, 정리 프로토콜에 따라
모든 완료 런의 `anomaly_maps/`는 0 TIFF 상태다. 따라서 Phase 0은 Phase-3
tuned anchor의 **맵 보존 1회 재생성**으로 시작한다.

- 앵커 설정: 표준 H+ DVT 기반 + `LAMBDA_LOGDET=2e-2`,
  `SUPPORT_BRIGHTNESS_RANGE=0.80,1.20`, `DENSITY_WEIGHT=0.25`
  (`scripts/run_flow_tte_hparam_phase3_remote.sh`의 `logdet020_br080120`
  변형과 동일). 기대 재현치: `0.837426 / 0.530635`.
- 생성된 맵은 Phase 0/1/2의 공용 입력으로 원격에 보존하고, Phase 2 분석
  종료 후 정리한다 (정리 증거 기록).
- 분석은 원격 컨테이너 CPU에서 수행하고 compact JSON/TSV만 로컬 pullback.

## A2. 조명 조건 층화 분석 추가 (필수)

2026-07-11 검토에서 확인된 사실: `test_public`은 이미지 인덱스당
`regular/overexposed/underexposed/shift_*` 6조건 균등 혼합이고, `train/good`
및 고정 16-shot support는 8객체 전부 100% regular다. 테스트 모집단의 5/6이
support가 본 적 없는 조명이다.

Phase 0에 다음을 추가한다 (파일명 기반, 라벨-프리):

- 조건별 good-픽셀 스코어 통계 (median, p99, p99.9)
- 조건 내 pooled oracle F1 vs 전조건 pooled oracle F1
- 조건별 스코어 오프셋 표 (드리프트의 방향과 크기)

이는 Δ_scale 판정의 메커니즘 확인이며 Phase 1 설계 선택(shrinkage λ,
분위수 q)의 근거가 된다.

## A3. fixed-threshold F1 정의

"normal-calibrated fixed-threshold F1"의 임계값은 support-only 통계
(support LOO 스코어 분포의 상위 분위수)에서 유도한다. `test_public/good`
기반 임계값은 transductive이므로 사용할 경우 diagnostic으로만 표기한다.

## A4. 평가기 주의사항

`src/post_eval.py`는 예측 맵을 float16으로 캐스트한 뒤 풀링한다. 표준화
스코어 저장 시 float32 TIFF 저장을 유지하면 문제없음을 확인했다. 새 분석
스크립트의 oracle F1 로직은 기존 평가기와 수치 일치를 단위 테스트로 검증한다.

## A5. can의 취급

can은 Type D (관측 상한 이전에 랭킹 자체가 붕괴: 672 pAUROC 0.56, native
G0 oracle F1 0.018)로 사전 분류한다. Phase 0-4에서 can 개선은 기대하지
않으며 no-harm 확인 대상으로만 취급한다. per-object floor 조건은 원 계획
유지.

## A6. 자원

dsba3 `hun_fsad_tta_012`, host GPU `0,1,2,3` 4-way 객체 샤딩 (2 objects/GPU).
다른 컨테이너 불간섭, 자격증명 비내장 원칙 유지.

## A7. 판정 주체 및 위임

코드 수정·실험 실행은 Codex에 위임하되, 게이트 판정과 KEEP/KILL 결정은
본 세션(오케스트레이터)이 compact 산출물을 직접 확인해 내린다. AD2 public은
shadow이므로 모든 oracle 스윕은 development 전용으로 기록한다.
