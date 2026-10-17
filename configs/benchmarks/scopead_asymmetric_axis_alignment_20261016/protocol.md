# 비대칭 부분 보정: 32개 개발 평가

2026-10-16. 사용자의 추가 보완 요청에 따른 후속 실험이다. 이전 48개 평가의 종료 조건은 보존한다.

## 가설과 변경 범위

계층 보정을 약하게 적용하면 AUPRO 이득을 일부 유지하면서 pAUROC 손실을 줄일 수 있는지 확인한다. 기존 정상 support 기반 계층 계수 a와 해상도 계수 c를 그대로 사용하고, 거리 d를 d/(a^alphaL c^alphaR)로 변환한다. 학습 가능한 파라미터는 없다.

| 새 설정 | alphaL | alphaR | 직접 비교할 기존 설정 |
|---|---:|---:|---|
| weak_layer_half_resolution | 0.25 | 0.5 | half-R (0,0.5), half-LR (0.5,0.5) |
| weak_layer_full_resolution | 0.25 | 1.0 | R (0,1), LR (1,1) |

두 설정 모두 N, Q95, half-LR과 비교한다. 같은 해상도 강도의 비교군을 함께 보고 계층 보정의 기여를 분리한다. 클래스별로 설정을 선택하지 않는다.

기존 16-shot 결과에서 half-LR minus half-R는 AUPRO +0.15699pp, pAUROC -0.06970pp이고, LR minus R는 +0.18297/-0.19949pp다. 이는 약한 계층 보정을 시험할 사후 가설의 근거이며 최적성이나 보정 필요성의 증거는 아니다.

## 고정 조건과 예산

- AD2 test_public: 이미 관찰한 개발 데이터. 평가 결과를 이용해 이번 후보를 선택했으므로 최종 일반화 검증이 아니다.
- 2/16-shot, seed0, 기존 VOID 정상 support manifest, 8클래스 전체, CQ off. 두 shot을 독립 반복으로 간주하지 않는다.
- 기존 partial-axis 설정에서 alphaL/alphaR만 변경한다. Frozen DINOv2 ViT-L/14 FP16, layers5/11/23, resolutions448/672/896/1120, bank, fusion, noise guard, 전처리, 평가기는 동일하다.
- GT/평가 label은 평가 및 사후 진단에만 사용한다. 정상 참조 계수는 정상 support로만 산출한다.
- 총 2설정 × 2shot × 8클래스 = 32개 평가. 추가 탐색이나 자동 후속 실행 없이 종료한다.
- 실제 데이터 실행은 dsba3의 기존 컨테이너와 기존 4개 GPU UUID에서만 한다. 로컬은 설정 검증 및 결과 분석만 수행한다.
- 기존 PatchCore 작업과 동시 실행할 경우 간섭 때문에 소요시간/비용 비교는 해석하지 않는다. 기존 프로세스와 활성 checkout을 변경하지 않고 별도 immutable snapshot 및 출력 경로를 쓴다.
- anomaly maps/weights 저장은 off. 원격 로그를 보존하고 정량 결과/config/manifest만 회수한다.

## 판정과 수집

AUPRO@0.05가 주 지표이며 pAUROC@0.05를 함께 보고한다. 각 설정의 N/Q95/half-LR 및 같은 해상도 강도 비교군 대비 차이, 8클래스별 손익, 중위 변화, 클래스 하나씩 제외한 평균 범위를 기록한다. 두 지표가 엇갈리면 절충으로 보고하며 안정적 개선으로 명명하지 않는다. 평균 개선만으로 척도 불일치의 인과 효과를 주장하지 않는다.

기존 비교군은 support/query/GT identity와 content, bank digest, backbone, guard, evaluator 조건 확인 후 재사용한다. 실패와 성능 저하를 구분하고 누락값을 0으로 채우지 않는다. 새 설정이 유망해도 독립 support 반복은 별도 후속 검증으로 남긴다.

실행은 scripts/run_experiments.sh --backend remote와 scripts/experiments.py run, 수집은 scripts/experiments.py collect를 사용한다. 완료 시 기존 누적 Markdown/CSV의 해당 행만 갱신한다.
