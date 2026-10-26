# Q95 × Rényi 통계 분석 추가 실험 — 2026-10-20

## 범위와 가설

- AD2 public full8 × 2/16-shot × seed 0/1/2. 이미 개발에 사용한 public의 사후 분석.
- 원래 mechanism 실행의 동일 token cache, support manifest, FP16 특징, cap256,
  4 bags, deep mean + .125 shallow, 실제 noise-guard route, CQ99를 유지한다.
- 백본 fitting, feature 추출, 최근접 탐색은 다시 수행하지 않는다.
- 48 class 실행 × 6 조건. 기존 dsba3 컨테이너/GPU 0–3을 그대로 사용한다.
  각 lane의 기존 실행이 성공한 뒤 해당 GPU에서 순차 진행한다.
- primary AUPRO@FPR≤5%, secondary pAUROC@FPR≤5%, 높을수록 좋다.
- Q95×집계 상호작용, 순위 개입, 크기 개입, 두 개입의 결합을 측정한다.
  순위/크기 hybrid는 분석용이며 새로운 최종 방법 선택 실험이 아니다.

## 개입

none/Q95 각각의 해상도별 점수맵을 만든다. none/Q95로 정한 순서에
none/Q95 점수를 gather하여 기존 Rényi 가중치를 적용하는 네 조건과
none/Q95 단순 평균 두 조건을 비교한다. 오름차순이며 동점은 입력 해상도
순서로 결정한다. 자기 순위/자기 점수 두 endpoint와 평균 두 endpoint가
원래 분석의 AUPRO와 3e-5 이내 일치하지 않으면 해당 실행을 중단한다.
이미지 endpoint 오차 허용치는 2e-5이다. 기존 캐시를 수정하지 않는다.

## 통계 단위

- 클래스별로 세 support seed의 짝지은 효과를 먼저 평균한다.
- 8개 클래스 bootstrap 10,000회, 95% percentile 구간, 클래스별 효과,
  시드별 평균, leave-one-class-out, 탐색적 양측 sign-flip p값을 기록한다.
- sign-flip은 귀무가설 하 클래스별 부호 교환가능성이 필요하다.
  p값은 다중비교 보정 전 탐색값이며 확정적 발견으로 해석하지 않는다.
- 8개 클래스에 대한 민감도 분석이다. 모든 실제 환경으로 일반화하지 않는다.
- 순위/크기 기여와 비가산성은 지표 수준의 개입 비교다. 인과적 매개효과나
  고유 기여도의 유일한 분해로 해석하지 않는다.

## 정상 꼬리와 순위

이미지마다 조건 간 동일한 native pixel 표본을 사용한다. 정상 최대 16384,
결함 최대 4096픽셀. 해상도별 Q95/Q99/Q99.9, 로그 분위수의 해상도 간
표준편차, 순위 변경률, 최상위 해상도 변경률, Rényi−평균의 점수 변화를 저장한다.
Q99.9는 표본 꼬리이므로 오차가 크며 정확한 전체 픽셀 분위수로 부르지 않는다.
꼬리 정렬의 주 요약은 네 해상도를 모두 사용한 정상 이미지에 한정한다.
라우팅 이미지는 별도 구분하며 전체 성능 평가에서는 제외하지 않는다.
순위가 균등해지는 것을 성공 기준으로 두지 않는다.

## 임계값의 이미지 교차검증

클래스별 정상/결함 이미지에 대해 각각 고정 hash로 최대 5개 fold를 배정한다.
각 fold 평가 시 나머지 fold의 정상 이미지만으로 FPR .5/1/5% 이하가 되는
가장 낮은 float16 threshold code를 정한다. 해당 fold의 모든 이미지를 평가한다.
정상/결함 이미지가 각각 2개 미만이면 유효한 교차검증 결과로 기록하지 않는다.
동점은 >= threshold로 포함하며, 검증 이미지의 실제 FPR도 함께 보고한다.

이후 정상/결함 이미지 층화 bootstrap 2,000회로 1% 목표 구간의
TPR/PRO/실현 FPR 짝지은 차이를 계산한다. 같은 이미지 추출을 모든 시드와
조건에 공유한다. 임계값은 원래 교차검증 값에 고정하므로 이 구간은
평가 이미지 변동에 대한 조건부 구간이다. 임계값 추정 또는 새로운 support
모집단의 불확실성을 포함한다고 주장하지 않는다. 픽셀 bootstrap은 하지 않는다.

GT는 예측/뱅크 fitting에 사용하지 않는다. 위 정상 이미지 선택과 모든 통계는
예측 완료 후의 분석에서만 GT를 사용한다. 배포 임계값 산출 실험이 아니다.

## 출력과 예산

기존 generic runner를 사용한다. 클래스당 timeout 21,600초.
비유한 점수, 원본 불일치, 잘못된 config/support identity이면 중단한다.
실행 오류와 성능 실패를 구분한다. 주기적 사용자 모니터링은 추가하지 않는다.
원격 /var/tmp/fsad-analysis/scopead_statistical_20261020에 이미지별 sparse
평가 histogram, 통계 JSON, 설정과 source provenance를 보존한다.
모두 성공하면 통계 요약을 생성하고 compact JSON/YAML/Markdown만
/workspace/results_local_compatible/scopead_statistical_20261020_compact로 내보낸다.
지도/가중치는 새로 저장하지 않는다. 결과 보고서는 RESULTS.md 명칭을 사용한다.
