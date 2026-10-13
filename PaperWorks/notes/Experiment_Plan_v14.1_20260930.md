# v14.1 전체 실험 실행 계획

작성일: 2026-09-30. 대상 원고: `Outline_v14.1.md`.

## 1. 목적과 현재 실행 범위

구성 요소의 효과·메커니즘, 표준 벤치마크 일반화, corruption 강건성, 계산 비용을 검증한다. 이 문서의 전체 실험 목록은 계획이며, 각 작업의 실제 실행 상태는 구분한다.

- 사용자 현재 지시: **표 4의 RegAD MVTec AD 2 test_public 비교부터 실행**.
- 나머지 실험 묶음: 계획 상태. 표 4 작업 후 실행 순서를 적용한다.
- 기존 결과는 방법·config·shot·seed·support·split·evaluator가 일치할 때 재사용한다. 빈칸 수를 신규 실행 수로 간주하지 않는다.
- 내부 변형은 ablation에서만 사용한다. 외부 비교군의 기존 단순화 구현을 공식 방법으로 대입하지 않는다.

## 2. 서버와 운영 조건

2026-09-30 확인: dsba3 NVIDIA RTX A6000 48GB × 4, 기존 `hun_fsad_tta_012` 컨테이너 실행 중, 확인 시 네 GPU 모두 유휴.

| GPU | UUID |
| :-- | :-- |
| 0 | `GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f` |
| 1 | `GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae` |
| 2 | `GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639` |
| 3 | `GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62` |

- 실험 묶음은 순차 진행하고 각 묶음 내부는 네 GPU에 병렬 분배한다. GPU별 동시 실행은 하나다.
- 과거 클래스별 비용을 반영해 무거운 클래스를 분산한다. 동일 클래스·shot·seed의 paired ablation은 같은 GPU에 배정한다.
- 기존 컨테이너 재사용, 고정 코드 snapshot 사용, 실행 중인 checkout 갱신 금지. 컨테이너 교체·원격 artifact 삭제 없음.
- RSG4 코드와 기본 설정을 보존한다. 추가 방법·진단 계측은 격리한다. method-only 비교 외 변경 조건은 명시한다.
- 설정 순서 실행은 `scripts/experiments.py run`, 결과 등록은 기존 `collect`를 사용한다. 전용 queue/ledger를 새로 만들지 않는다.
- 정상 support만 fitting에 사용한다. GT는 evaluator·진단·시각화에만 사용한다.
- 원격 로그 유지, 로컬에는 metrics·config·manifest·최소 provenance를 회수한다. 지정된 시각화 query 외 anomaly maps와 weights는 저장하지 않는다.

## 3. 실험 행렬

| 묶음 | 조건 | 산출물 |
| :-- | :-- | :-- |
| 표 4 RegAD 보완 | AD2 test_public full8, 1/2/4/8/16-shot, seed 0–2; 기존 support manifests; 공식 registration model 고정 | AUPRO@0.05, pAUROC@0.05, I-AUROC; 평균±sample SD |
| 계산 비용 | Ours·Deep-only·1120 단일 해상도·SuperAD, 16-shot, 같은 GPU·precision·batch 1 | support 구축 시간, 평균/p95 query latency, 구축/추론 peak VRAM |
| 구성 요소 | A–I, AD2 full8, 2/16-shot, seed 0–2; Regular/Shift 1은 동일 실행의 분리 평가 | paired 성능 차이, gate·기여량·회복/손실 통계 |
| 표준 벤치마크 | MVTec AD full15·VisA full12, 표 5의 8개 방법, 1/2/4/8/16-shot, seed 0–2 | I-AUROC·P-AUROC 평균±SD |
| corruption 보강 | PatchCore·AnomalyDINO, clean·brightness·contrast·defocus·Gaussian s3, MVTec AD full15, 16-shot, seed 0–2 | AUPRO·pAUROC, clean 대비 변화 |
| α·β 민감도 | α={0.90,0.95,0.98}, β={0.95,0.98,0.99,0.995}, AD2 full8, 2/16-shot, seed 0–2 | 성능 heatmap, Regular/Shift 1 비교 |
| κ·routing | κ={5,7,9,11,15}, MVTec AD full15, 16-shot seed 0; clean, 기존 noise 및 deterministic clean/noisy mix | 성능·발동 비율·always-lowres 대조 |
| 1-shot LTO | LTO·공간 근접 토큰 제외 LTO·자기 매칭 허용 진단, AD2 full8 seed 0–2 | 분위수 변동, non-support normal query 초과 비율, 성능; LSO와 조건별 비교 |
| 백본 | DINOv2 ViT-B/14, Deep-only·전체 shallow 가산·Ours, AD2 full8 2/16-shot seed 0–2 | Appendix 비교표; layer 선택 규칙 기록 |
| 정성·메커니즘 | 지정 normal/defect/shift/noise query; 동일 색상 범위; CQ 전후·PRO–FPR | 그림 패널, 동일 evaluation FPR에서 회복/손실 집계 |

최대 신규 실행 전 계산: 표 5 3,240개 클래스 평가, α·β 576개, 구성 요소 432개. 실제 신규 수는 재사용 감사 후 확정한다. 학습형 비교군의 source 학습·checkpoint 조건이 불충분하면 별도 준비 시간을 기록하고 임의 구현으로 대체하지 않는다.

Fixed-weight는 정상 support LSO에서 residual 평균 가산량과 맞춰 결정하며 query 전체에 고정한다. 민감도는 개발 분석이고 기본값을 test 성능으로 다시 선택하지 않는다. κ의 development/held-out 조건을 구분한다. 동일 noisy root의 100% firing만으로 query별 적응성을 입증하지 않는다.

## 4. 순서와 시간 추정

표 4 RegAD 작업을 우선 삽입한 뒤 아래 계획을 적용한다.

| 순서 | 작업 | 예상 GPU-hours | 네 GPU 사용 시 경과 시간 |
| :-- | :-- | --: | --: |
| 우선 | 표 4 공식 RegAD bridge·설정·pilot 및 full8 | pilot 후 산정 | 준비와 pilot 후 갱신 |
| 0 | 결과 재사용 감사·설정/계측 준비 | — | 4–8시간 |
| 1 | 계산 비용 | 2–4 | 1–2시간 |
| 2 | A–I 대조·메커니즘 | 40–70 | 12–22시간 |
| 3 | 표 5 완성 | 70–140 | 20–42시간 |
| 4 | corruption 비교군 | 4–10 | 2–4시간 |
| 5 | α·β sweep | 80–130 | 22–38시간 |
| 6 | κ·routing | 8–18 | 3–6시간 |
| 7 | LTO 분석 | 12–24 | 4–8시간 |
| 8 | DINOv2-B | 8–16 | 3–5시간 |
| 9 | 추가 map 추출·집계·원고 반영 | 4–8 | 4–8시간 |

원래 전체 계획의 계산은 약 230–420 GPU-hours, 준비·정리를 포함한 75–145시간이다. 실패 복구 여유를 포함해 **4–7일**, component 거리 재사용 검증·적용이 가능한 경우 **3–5일**을 계획한다. 표 4의 신규 official RegAD 준비·실행 시간은 이 추정에 별도 가산한다. 모든 값은 일정 추정이며 완료 시간 보장이 아니다.

근거: dsba3의 RSG4 AD2 full8 16-shot seed2 약 1.61 GPU-hours, E12 3-arm full8 약 4.14 GPU-hours. 각 `status.json`의 started/ended 기준으로 산정했다. 클래스 불균형·초기화·I/O를 포함하므로 4로 단순 나누지 않는다.

## 5. 시간 절감과 완료 기준

- α·β·hard gate·fixed-weight·CQ는 동일 bank의 component 거리를 재사용할 수 있는지 확인한다. 기존 출력과 일치하는 경로만 사용하며 이 최적화를 사전 완료로 가정하지 않는다.
- 정성 자료는 사전 지정 query에서 실험과 함께 수집한다. 사후 선정한 예시는 post-hoc 표시한다.
- 정상/결함 반응, shallow 순이득, source/target calibration은 서로 구분한다. 결과가 가설을 지지하지 않아도 그대로 보고한다.
- 방법/config/seed/support/scope별 결과를 보존하고 mean±SD와 paired 차이를 계산한다. ledger Markdown·CSV를 함께 갱신한다.
- 검증된 결과만 원고 표에 반영하고 미완료 또는 비비교 결과는 `—`로 유지한다.

## 6. 표 4 RegAD 실행 설계

- 최종 실행 상태와 지표는 §7 기록 참조.
- 기존 `src/fmad/methods/regad.py`는 learned registration 없이 frozen feature 대응을 수행하므로 외부 RegAD 결과로 재사용하지 않는다.
- 공식 source: `MediaBrain-SJTU/RegAD@5e2c1f8c18d302b0354471567846fee3ed2ff063`.
- 기존 official MVTec AD reproduction은 source fidelity 근거다. AD2 support/query 행렬의 결과를 대신하지 않는다.
- 실행 전 protocol에 고정 source checkpoint 및 shot 확장 규칙을 기록했다. Source MVTec test-label checkpoint selection 이력을 명시했으며 AD2 labels는 선택·학습에 사용하지 않았다.

## 7. 표 4 RegAD 실험 실행 결과, 2026-09-30

- 상태: **완료 120/120**, 네 dsba3 lane 모두 exit 0, 실패 0. 공식 RegAD test-path adapter와 released source checkpoint를 사용했다.
- shot별 full8 macro AUPRO@0.05 / pAUROC@0.05 / I-AUROC (seed 0–2 mean ± sample SD, %):

| Shot | AUPRO@0.05 | pAUROC@0.05 | I-AUROC |
|---:|---:|---:|---:|
| 1 | 12.73 ± 0.38 | 57.76 ± 0.30 | 57.73 ± 1.82 |
| 2 | 15.05 ± 0.30 | 58.76 ± 0.49 | 60.56 ± 3.94 |
| 4 | 17.40 ± 1.07 | 60.36 ± 0.48 | 63.79 ± 2.34 |
| 8 | 19.25 ± 0.64 | 62.02 ± 0.32 | 64.34 ± 1.05 |
| 16 | 20.28 ± 0.80 | 63.46 ± 0.76 | 65.64 ± 1.34 |

- 네 lane의 클래스별 지표·config·manifest·status는 `results/remote_runs/dsba3/regad_ad2_table4_20260930/full_matrix/combined/`에 있다. 수치는 canonical `results/comparisons.md`와 paired CSV에도 수집됐다.
- released 2-shot bottle checkpoint는 전체 AD2 shot/seed에서 고정했다. 이 checkpoint의 선택 절차는 source MVTec test AUROC를 사용했으며 AD2 라벨 선택은 없다. 비정사각 AD2 입력은 공식 STN의 224×224 기하에 맞춰 정사각 stretch했다. 평가 split은 노출된 AD2 `test_public`이므로 개발 증거다.
- `Outline_v14.1.md` 표 4 RegAD 세 지표 칸과 출처를 갱신했다. 원고에 표시된 값은 소수 둘째 자리로 반올림했고 분석용 원값은 run report/ledger에 보존했다.
