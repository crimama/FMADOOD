# Rényi 추가 응답과 검출 변화 추적 — 분석 전 계획

## 질문과 범위

최종 방법/가중치/보정 분위수/후처리는 고정한다. 성능 개선 후보나 새 파라미터를
선택하지 않는다. 기존 AD2 public 캐시의 full8 × 2/16-shot × seeds 0/1/2,
48개 조합을 dsba3 기존 hun_fsad_tta_012, GPU 0–3에서 재분석한다.
기존 동일 support/config/query/GT hash를 검증하고 backbone 및 NN은 재실행하지 않는다.
이는 이미 개발에 사용한 public GT의 사후 메커니즘 분석이다.

## H1: 평균 응답과 추가 응답

해상도 점수를 오름차순 s_(i)라 하면 R=mu+E이며, 4개 해상도에서는
E=(9*g1+14*g2+13*g3)/48, gi=s_(i+1)-s_(i)이다.
이 항등식은 CQ 전에서만 적용한다. 다른 resolution route는 일반 n의
Rényi 가중치에서 동일한 gap 분해를 계산하고 별도 표시한다.
normal/defect 표본은 이미지당 최대 8192/4096개, 같은 픽셀을 none/Q95에 공유한다.
E/mu(단, mu=0은 missing), 다른 fold 정상 mu Q95로 나눈 E,
정상 mu 분위수 구간별 E/mu와 E를 저장한다. 원점수와 보정점수의 절대 E만
비교하여 유리한 결론을 내리지 않는다. 정상 기준 분위수는 다른 fold의
정상 이미지에서만 정한다. 동일 mu 구간은 각 보정 조건의 정상 기준
분위수이며 서로 동일한 raw score 구간이라는 뜻은 아니다.

예측: Q95가 정상의 불필요한 추가 응답을 줄이고 결함 응답을 보존한다면
정상·결함의 상대 추가 응답 대비가 개선되어야 한다. 클래스별 반대 결과도 보고한다.
정상 감소/결함 증가를 전제하지 않는다. 이 비교는 설명용이며 인과적 매개효과가 아니다.

## H2: 검출 변화의 위치와 CQ 영향

none/Q95 × mean/Renyi 네 조건의 CQ 전/후 map을 재평가한다. CQ 후 네
endpoint의 AUPRO/pAUROC와 기존 원본을 각각 3e-5 이내로 검증한다.
각 arm과 stage에서 정상 이미지 기반 최대 5-fold, 목표 FPR .5/1/5%를
사용한다. CQ 후 threshold/fold는 앞선 statistical 실행에서 고정한 값을
그대로 재사용하고 pixel TP/FP 및 영역 PRO mass까지 재현하는지 확인한다.
CQ 전은 같은 fold의 다른 정상 이미지 histogram만으로 threshold를 계산한다.

같은 이미지/GT에서 mean→Renyi 및 none→Q95 비교별 newly-positive,
newly-negative 픽셀을 정상/결함으로 나눠 센다. 모든 GT 8-connected 영역의
coverage 변화와 gain/loss를 연결한다. threshold는 arm마다 다르므로 실제 FPR도
함께 보고하며 동일 FPR 조건으로 부르지 않는다.

CQ 후 두 detector margin 차이는 E + (CQ_R-R) - (CQ_mu-mu)
- (threshold_R-threshold_mu)로 분해한다. 검출에는 원래 float16 code를 사용하므로
margin에는 float16 양자화 보정항도 포함한다. 이 등식의 수치 오차를 검사한다.
새 검출/새 오탐 위치에서 각 항의 평균을 기록한다. 항이 양수라는 것만으로
그 항의 인과적 기여도 또는 통계적 유의성을 주장하지 않는다.

## H3: 어떤 결함에 유리한가

GT 영역의 면적 비율, bounding box, 유형 디렉터리, 회색조 영역과 8픽셀
주변 정상 ring의 명암 차이/표준편차, 해상도별 영역 평균 응답 및 최대 해상도
응답 비중을 저장한다. ring은 다른 결함을 제외한다. 표준편차가 0이거나
정상 ring이 없으면 standardized contrast는 missing이다. 회색조 대비는
결함 가시성의 제한된 proxy이며 모든 texture/색 차이를 표현하지 못한다.

크기 구간은 기존 진단과 동일하게 <0.1%, 0.1–1%, >=1%로 고정한다.
주 보고값은 연속형 영역 coverage 변화이며 크기 구간은 설명용이다.
각 클래스와 이미지의 영역 수 차이로 결과가 지배되지 않도록 클래스별로
영역 coverage를 먼저 집계하고 시드 평균 후 macro한다. 전체 영역 pooled
값과 면적 가중 픽셀 값은 별도 명시한다. 영역이 없는 구간은 0으로 채우지 않는다.
대비/해상도 집중도와 coverage 변화는 연관 분석이며 causal/성능 선택 근거가 아니다.

## 통계, 검증, 출력

클래스 내 세 seed 평균 후 클래스 bootstrap 10,000회, paired image bootstrap
2,000회(동일 image resample을 모든 seed/조건에 공유, threshold/support 고정)를
사용한다. 8개 클래스의 이질성, fabric 포함/제외, seed별 결과를 보고한다.
픽셀/영역을 독립 재표집하지 않는다. 평가 표본의 상관과 label exposure를 명시한다.
정상/결함 상대 추가 응답은 기술 통계이며 비교 가능한 이미지/클래스 범위를 표시한다.

실행 전 검증: Python/Bash 구문, gap 항등식/순열·동점·동일값, 양자화 margin
재구성. 실제 실행은 source hash, endpoint metric, crossfit counts/mass 재현을
검사한다. 불일치/비유한 값/잘못된 support면 중단한다. 클래스당 timeout 21600초.
원격 결과는 /var/tmp/fsad-analysis/scopead_response_trace_20261020에 보존한다.
새 dense map/weights는 저장하지 않는다. compact JSON/config/manifests만 회수한다.
전체 완료 시 자동 suite summary 및 compact export를 생성한다. 주기적 사용자
모니터링은 추가하지 않는다. RESULTS.md에 실행 및 결과 요약을 남긴다.
