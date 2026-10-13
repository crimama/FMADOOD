# FSAD-TTA 실험 가이드라인

- 제정: 2026-08-22
- 성격: 포트폴리오 운영 규칙 + 정식 계약의 통합 문서
- 근거: 2026-07~08 실험 이력 전체 검토(nuisance 식별 계약 ~15연속 KILL, RN 변형 5일 6연속 종료,
  method claim 0개 vs 반복 검증된 진단 다수)에서 도출된 수정 사항
- 상위 문서와의 관계:
  - `FMAD-OOD/docs/RUNBOOK.md`, `FMAD-OOD/AGENTS.md`는 운영 규칙으로 그대로 유효.
    FMAD-OOD 저장소가 활성 세션 하에 있어 pointer 수정은 보류하며, 본 문서가 FSAD-TTA의
    로컬 표준이다.
  - `skill_graph/analysis/2026-08-07_fewshot_distribution_shift_integrated_experiment_ledger.md`
    의 R1--R9는 모듈 후보 요구조건으로 그대로 유지된다. 본 문서는 R1--R9를 대체하지 않고
    그 위의 포트폴리오 판단 규칙을 추가한다.
  - `skill_graph/analysis/2026-08-07_fewshot_distribution_shift_structure_search_framework.md`
    의 N0→N3 family ladder는 그대로 적용한다. 단, ladder 도입(2026-08-07) 이전처럼 naive
    시도 1회로 family를 닫는 것은 금지된다(N1/N2 최소 시도 후에만 family 우선순위 강등 가능).

## 0. 유지 기준 (변경 없음)

다음 기존 표준은 본 개정과 무관하게 그대로 적용한다. 약화 금지.

1. 사전등록 → frozen gate → fail-closed 무효화(v1/v2 무효 처리 관행 포함).
2. 접근 사다리: support-only → reduced TESTpublic → all-eight TESTpublic;
   TESTprivate은 공식 서버 제출 경로로만 접근. query update/TTA/condition-token oracle 금지(R7).
3. 승격되는 모든 신호에 hard-null / shuffled control / causal control 필수.
4. Custody: support manifest SHA, checkpoint SHA, code/config hash, access ledger 기록.
5. 원격 실행 시 local↔remote parity 검증 후 compact artifact 확정, dense map cleanup 증거 보존.
6. 신규 후보 승격 전 closed-branch ledger 및 본 문서 G-1 상태 확인.
7. 실험당 메커니즘 변경 1개(FMAD-OOD AGENTS.md).

## G-1. Family-level 중단 규칙 (3-strike rule)

동일 메커니즘 질문이 연속 3회 `KILL_FOR_CLAIM`(또는 `VALID_NO_SIGNAL`/`NO_CONTINUE`)이면,
해당 질문을 후보 구현 이전 design-audit 단계에서 기본 기각한다.

- 질문 동일성 판정: 개별 구현이 아니라 **공유 failure basin**(attribution 실험이 밝힌
  공통 인과 병목) 기준. 예: "support-only shift-direction identifiability 부재".
- Design kill(코드/GPU 없는 설계 기각)도 같은 basin에 매핑되면 strike에 산입한다.
- 닫힌 질문에서 새 후보를 열려면 사전등록에 `failure-basin divergence` 섹션 필수:
  이전 실패들과 무엇이 달라 공유 병목을 벗어나는지 정량 서술. 심사 불충분 시 구현 금지.
- 채택 시점 기준 닫힌 질문(예시, 완전 목록 아님):
  - support-only nuisance/shift 방향 식별 계열: CLOSED(ledger 2026-08-07).
  - RN 스코어 수정 계열(TQRN, RN tangent, RNC-PA, CF-GC-RN, PRC-RN): CLOSED(2026-08-13~16).
  - FACT-RN: 자체 규칙(held-object mechanism assay 선결)에 따라 design-stage 유지.

## G-2. Gate 산출 근거와 kill/iterate 이원화

1. 모든 수치 gate는 다음 중 하나로 정당화되어야 한다. 근거 없는 round number 금지.
   - (i) 동일 파이프라인의 shuffled/hard-null arm에서 측정한 null 분포의 상위 백분위, 또는
   - (ii) 파일럿 데이터로 사전 측정한 최소검출가능효과(MDE).
2. Kill vs iterate 이원화: 주 효과의 방향이 통과하고 안전 gate(defect retention, geometry)
   도 통과했는데 크기만 미달한 경우 `NEAR_MISS`로 기록하고, 사전등록에 iteration lane을
   선언했다면 bounded iteration 1회를 허용한다. lane 미선언 시 kill 유지.
   gate를 결과를 본 뒤 조정하는 것은 어떤 경우에도 금지(기존 원칙 유지).
3. 모든 report에는 실패한 gate의 distance-to-gate 표를 남긴다(재발 방지 및 G-2-1 재산정 근거).
4. 역사적 교정 사례(참고): paired-loss V3 shift reduction 9.77% vs gate >=10%;
   low-pass DVT medium 보존율 67~72% vs gate >=75%; WTCF witness JVP .9236~.9493 vs <=.85.

## G-3. 진단→기여 전환 의무

1. 모든 `CONTINUE_DIAGNOSTIC` 및 positive 판정 report에 "기여 전환 조건" 필드 필수:
   벤치마크-comparable claim까지 남은 절차 / 논문 섹션 후보 / 폐기 중 하나 지정.
2. Consolidation checkpoint(2주 주기): 살아있는 진단을 {벤치마크 실행, 논문 섹션, 폐기}로
   배정하고 그 결정을 `skill_graph/todo.md`에 기록한다.
3. 채택 시점 backlog:
   - SPCT B1 full-support TESTpublic 집계·판정 미완(todo 2026-08-12 잔여 항목).
   - DVT localization prior → 논문 ablation 섹션 후보.
   - private-mixed 최소열화 관측(-4.11%) → public shift-proxy 설계 필요.
   - SAM region-refinement Stage-1 진행 중(preregistration DRAFT).

## G-4. 정식 메트릭 계약 (기존 논문 메트릭 사용)

원칙: **새 메트릭을 발명하지 않는다.** 비교 대상 논문(SuperAD/SuperADD/RoBiS/VisionAD)과
기존 프로젝트 표가 일관되게 사용해 온 메트릭 suite를 그대로 정식 계약으로 사용한다.

1. MVTec AD2 정식 계약:
   - `seg_AUROC_0.05`(pixel pAUROC@FPR<=0.05), `Seg-F1`, `AU-PRO@0.05`.
   - 공식 서버 행은 서버 반환값 그대로 Private/Private-mixed `AU-PRO@0.05`, `Seg-F1`,
     `Class-F1`을 추가한다.
2. MVTec AD1 정식 계약:
   - `image_AUROC`, `image_AP`, `pixel_AUROC`, `pixel_AP`, `pixel_PRO`
     (+ 내부 진단용 `seg_AUROC_0.05`, `seg_F1`; 기존 관행 유지).
3. Threshold provenance 표기 필수(Main Table 기존 관행): `raw oracle` / `calibrated`
   을 상호 교환 불가능하게 명기하고, 논문 본문 claim은 calibrated 또는 서버 값으로만 한다.
4. 진단 전용 보조 지표(component recall, top-k coverage, tail statistic 등)는 lane 고유
   출력으로 허용하되, 정식 계약과 함께 항상 병행 출력되는 한에서만 사용하며 단독 claim
   근거로 쓰지 않는다.
5. 신규 실험은 day one부터 정식 계약 값을 함께 산출한다(lane 고유 metric과 병행).
   소급 비용을 피하기 위함이다.

## G-5. 합성 shift 프록시의 외적 타당성

support-only Stage -1의 kill 기준으로 쓰는 합성 C/S/D/SD 변환이 실제 `shift_*` 조건을
대표하는지 1회 교정 실험으로 검증한다.

- 내용: 합성 변환 배터리 점수와 실제 shift_* 조건 지표의 상관 측정.
- 처리: 상관이 낮은 축은 R2/R3 gate에서 제외하거나 가중 조정하고, 그 결과를 ledger에 명문화.
- 선례 근거: default_shift_sequence(합성 설계와 달리 density가 shifted-tail 악화),
  DVT condition attribution(TPR 이득이 조건 무관)에서 이미 프록시-실조건 불일치 관측됨.

## G-6. 객체 이탈(outlier) 취약성 처리

1. 축소 객체 집합 사전등록 시 알려진 outlier 객체(fabric 반복 사례: TTE -0.178,
   positional-key retention 0.64, DVT factorial collapse flag)의 실패는 별도 보고하고
   전체 기각 근거에서 분리 표기한다.
2. `>=3/4` 형태 gate에서 "positive count 감소(예: 3/4→2/4)"가 단일 객체 몰빵이면
   `OBJECT_DRIVEN_FAILURE`로 태깅하고, 일반화 주장 시 all-8 또는 사전 지정 객체에서
   재확인을 요구한다.
3. 일반화 claim의 scope 객체는 결과를 보기 전에 고정한다.

## G-7. Verdict taxonomy 보강

기존 어휘(KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC / PROMISING_DIAGNOSTIC /
ACCEPT_MEASUREMENT / BLOCKED_BASELINE / VALID_NO_SIGNAL)에 다음을 추가한다.

| Verdict | 정의 | 후속 조치 |
|---|---|---|
| `UNDERPOWERED_N<min>` | eligible 객체/표본이 gate 판정에 부족 | 재시험 허용 조건을 report에 명기 |
| `QUESTION_EXHAUSTED` | 개별 후보가 아니라 질문 자체가 닫힘(G-1 발동) | 새 후보는 failure-basin divergence 없이 금지 |
| `MECHANISM_REFUTED` | attribution이 메커니즘을 직접 반증 | 해당 메커니즘 계열 재검토 금지 |
| `NEAR_MISS` | 방향+안전 통과, 크기 미달(G-2) | iteration lane 선언 시 1회 bounded retry |

## G-8. Runway-aware 승격 체크

신규 후보 승격 전 사전등록에 다음 두 줄을 추가한다.

1. 예상 비용(GPU 시간, 경과일, 참여 서버).
2. "실패해도 논문 콘텐츠(메커니즘 인사이트/교정 데이터)가 남는가?" — 아니오면
   잔여 runway 대비 우선순위를 낮춘다.

## 개정 절차

- 본 문서 변경은 날짜와 근거(실험 verdict)를 함께 기록한다.
- G-1 닫힌 질문 목록은 consolidation checkpoint(G-3-2) 때 갱신한다.
