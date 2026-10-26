# Q95 보정 축 분리 및 기존 메커니즘 자료 재분석 — 2026-10-21

## 범위와 연구 질문

최종 방법 고정. 성능 개선 후보/분모 튜닝/모델 선택을 하지 않는다.
AD2 public full8 × 2/16-shot × seed0/1/2, 기존 48개 token cache를 재사용한다.
GT는 기존 예측 후 평가/진단에서만 사용하고 fitting에 넣지 않는다.
이미 개발에 사용한 public의 사후 분석이다. 새 backbone/NN pass 없음.

1. 기존 paired sample을 baseline raw mu 구간으로 함께 나눴을 때도 Q95의
   상대 추가 응답 억제가 남는가? 기존 condition별 mu 분위수는 서로 다른
   픽셀 집합을 만들 수 있으므로 동일 raw mu 구간에 같은 paired pixel을 넣는다.
   raw mu 구간은 기존 다른 normal fold에서 정한 10개 분위수 구간을 그대로 쓴다.
   mu=0은 relative statistic missing. E/mu, E/normal-mu-Q95, mu/reference를 함께
   보고하여 분모 변화와 추가 응답 변화를 구분한다. 빈 구간은 0으로 채우지 않는다.
   대응 픽셀 층화는 baseline mu를 통제하지만 Q95 후 mu를 같게 고정하는 것은 아니다.
2. fabric 및 나머지 클래스에서 실제 detector gain/loss, FP 변화,
   defect type/면적/응답 집중도와 coverage 변화를 연결한다. 기존 fixed normal
   crossfit threshold(.5/1/5%)를 재사용하고 실제 FPR를 보고한다. 인과적 매개효과나
   영역 독립 검정으로 해석하지 않는다. fabric 제외 결과도 포함한다.
3. 계층 효과, 해상도 효과, 두 축 상호작용 중 어느 것이 보정 효과와 연결되는가?

## 로그 분모의 통제된 분해

c[b,l,r]는 원본 support Q95, log c = g[b]+L[b,l]+R[b,r]+I[b,l,r].
각 bag에서 g는 전체 layer-resolution 평균, L/R는 해당 축 평균−g,
I는 나머지 residual이다. 각 축 효과의 주변 평균은 0이며 full c를 정확히 재구성한다.
분해는 정상 support 분모만 사용한다. query/GT에 맞춰 분모를 추정하지 않는다.

- none: 분모 1, 원래 none endpoint.
- global: exp(g), bag 공통 척도만. bag마다 다른 공통 배율도 결과를 바꿀 수 있어 별도 대조.
- layer: exp(g+L), 같은 공통 배율 유지, 계층 축만.
- resolution: exp(g+R), 같은 공통 배율 유지, 해상도 축만.
- separable: exp(g+L+R), 두 축 주효과, residual 제외.
- full: c, 원래 Q95 endpoint.

각각 mean/Renyi 두 집계와 기존 CQ99를 사용한다. 48 captures ×12 arms,
class당 timeout21600초. deep mean+.125 shallow, bags4, cap256, FP16,
backbone/layers/resolutions 및 실제 noise-guard route는 유지한다.
공통 데이터 조건 및 source config/provenance를 검증한다.
각 bag의 전체 배율을 유지하므로 layer/resolution 비교의 기준은 global이다.
none→global은 별도 효과이며 이를 무시해 계층/해상도 고유 효과로 부르지 않는다.
full→separable은 계층×해상도 분모 residual의 진단이다. bag 차원을 분리한 실험은 아니다.

## 평가와 통계

Primary AUPRO@FPR<=.05, secondary pAUROC@.05, normal/defect E/mu와 PRO operating
points. label-free 최종 예측 조건을 바꾸는 것이 아니라 cache 개입 진단이다.
fixed source endpoint four metrics must match within3e-5. full reconstructed token
scores must match cached Q95 within2e-5; log decomposition tolerance1e-10.
비유한 값/identity mismatch/endpoint mismatch 시 해당 cell 중단.

사전에 지정한 비교: layer−global, resolution−global,
separable−layer−resolution+global, full−separable,
각 분모 아래 Renyi−mean. 비교 방향은 추정하고 음수 결과도 보존한다.
각 클래스 내3seed 평균 후 클래스 bootstrap10000회 95% 구간,
seed별 효과, fabric 제외를 보고한다. 픽셀 독립 표본/유의성 검정 없음.
기존 자료 재분석도 이미지·seed·클래스 순으로 평균하며 빈 표본 수를 명시한다.
개별 class 효과의 인과적 귀속/유일한 분해/미사용 최종 평가로 주장하지 않는다.

## 실행 및 산출물

기존 dsba3 hun_fsad_tta_012, GPU0–3 UUID 명시. 기존 scripts/experiments.py run과
run_experiments.sh를 이용한다. active checkout을 바꾸지 않고 불변 snapshot을 stage한다.
코드 commit/push 및 컨테이너 교체는 하지 않는다.

기존 sample의 조건부 재분석은 CPU, 축 분리 replay는 지정 GPU에서 실행한다.
원격 /var/tmp/fsad-analysis/scopead_axis_analysis_20261021 아래
axis_analysis.json, conditional_analysis.json, config/manifest/hash 및 로그를 저장한다.
새 dense map/weights는 저장하지 않는다. 원본 caches와 samples는 수정하지 않는다.
각각 full48 완료 시 자동 요약 및 compact export를 생성한다.
RESULTS.md에 요약을 남기고 검증된 새 지표만 generic collector로 누적 등록한다.
