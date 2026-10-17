# 정상 참조 다중 분위수 단일 척도: 개발 비교

2026-10-16 사후 방법 탐색. 가설: 관측마다 정상 거리의 Q95 한 지점만 사용해 나누는 것보다, Q50/Q75/Q90/Q95를 이용해 양의 단일 배율을 추정하면 이상 탐지 점수의 결합이 안정될 수 있다. 이는 검증할 가설이며 성능 향상 주장이 아니다.

방법 경계: 관측은 해상도 × bag × layer다. 각 관측의 정상 support source-LOO 거리 분위수 q를 구한다. q/Q95의 로그 평균으로 클래스 내 모든 48개 관측의 공통 target을 만들며 target Q95는 1이다. 양의 단일 배율 s=(q·q)/(q·target)를 적용해 d/s를 결합한다. image_median 후보는 support 이미지별 q와 같은 target으로 s_i를 추정한 뒤 log(s_i)의 중앙값을 지수화한다. Query 점수의 비선형 변환이나 명시적 가중치 변경은 없다. 학습 파라미터는 없다.

비교군: 기존 none, Q95, two-axis half (0.5,0.5), 새 multi_quantile_scale, 새 image_median_quantile_scale. 기존 비교군은 같은 shot/class support·query·GT·bank·backbone·평가 조건이 확인된 결과만 재사용한다. 절대 점수와 클래스별 Δ를 모두 제시한다.

고정 조건: AD2 test_public seen development, 2/16-shot, seed0, VOID 정상 support manifest, 전체 8클래스, CQ off. DINOv2 ViT-L/14 FP16, layers5/11/23, resolutions448/672/896/1120, 4 bags, source당 256 tokens, cosine 1NN, no_mask_no_rotation, 기존 noise guard, linear0125 및 float16 streaming evaluator를 고정한다. Label과 GT는 평가와 사후 진단에만 쓴다.

주 지표 AUPRO@0.05, 보조 지표 pAUROC@0.05; 둘 다 높을수록 좋다. 새 평가 2후보 × 2shot × 8클래스 = 32 class cells에서 종료한다. 8클래스 macro, 클래스별 손익, 중위 Δ, leave-one-class-out 평균 범위, 정상 참조 척도와 fallback 수를 보고한다. 이미 본 공개 개발 집합의 seed 하나이므로 유망한 결과도 최종 성능 또는 반복 재현성으로 주장하지 않는다.

실제 데이터 실행은 기존 dsba3 컨테이너의 지정 GPU UUID에서 immutable snapshot으로 진행한다. 기존 활성 checkout을 바꾸지 않고 출력 경로를 분리한다. 원격 로그·설정·manifest·정량 지표를 보존하며 maps/weights는 저장하지 않는다. 수집은 기존 collector로 누적 Markdown과 CSV를 함께 갱신한다.
