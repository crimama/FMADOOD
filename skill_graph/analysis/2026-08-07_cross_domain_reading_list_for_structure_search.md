# 구조 탐색 프레임워크를 위한 비-AD 분야 Reading List

기준일: 2026-08-07
상태: 참고 문헌 목록; 실증적 주장이나 후보 승격 없음
대상 문서: `skill_graph/analysis/2026-08-07_fewshot_distribution_shift_structure_search_framework.md`

## 0. 목적과 선정 기준

이 목록은 구조 탐색 프레임워크의 **미해결 설계 공백**에 대응한다.
Anomaly detection 문헌은 의도적으로 제외했다. AD 논문은 동일한 벤치마크
관행과 동일한 실패 모드를 공유하므로, 프레임워크가 요구하는
identifiability·geometry·objective-alignment 질문에 대한 외부 근거가 되지
않는다.

선정 기준은 다음 세 가지다.

1. 프레임워크의 특정 requirement(R1-R9) 또는 설계 차원(§5)에 직접 대응할 것.
2. 우리가 아직 답을 갖지 못한 질문에 **이미 정식화된 답 또는 불가능성 결과**를 줄 것.
3. AD 외 분야일 것(생성모형, OOD/uncertainty, meta-learning, 표현학습, 통계).

각 항목은 "왜 이 논문인가"를 프레임워크 조항 기준으로 명시한다.

## 1. 우선 읽기 3편

| 논문 | 대응 조항 | 이유 |
|---|---|---|
| On Feature Collapse and Deep Kernel Learning for Single Forward Pass Uncertainty (van Amersfoort et al., 2021) — [arXiv:2102.11409](https://arxiv.org/abs/2102.11409) | R4, R8, §8 | 무제약 목적함수가 far-away 입력을 training feature로 붙여버리는 **feature collapse**를 정면으로 정식화하고 bi-Lipschitz 제약을 해법으로 제시한다. §2.1의 "무제약 수축 시 defect evidence 동반 collapse" 관찰과 문제설정이 동형이다. |
| Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness (Liu et al., NeurIPS 2020) — [arXiv:2006.10108](https://arxiv.org/abs/2006.10108) | R8, §8 | spectral normalization으로 hidden layer bi-Lipschitz를 강제해 distance awareness를 유지한다. local non-contraction guard를 구현 가능한 형태로 옮길 때의 실용 레퍼런스. |
| What Should Not Be Contrastive in Contrastive Learning (Xiao et al., ICLR 2021) — [arXiv:2008.05659](https://arxiv.org/abs/2008.05659) | §2, §8 | "무엇에 invariant하면 안 되는가"를 설계 변수로 승격시킨다. coherent-nuisance prior를 hard invariance가 아니라 분리된 embedding subspace로 두는 대안 구조를 제시. |

## 2. 식별 가능성의 경계 (§2)

- **Challenging Common Assumptions in the Unsupervised Learning of Disentangled
  Representations** (Locatello et al., ICML 2019) —
  [arXiv:1811.12359](https://arxiv.org/abs/1811.12359).
  inductive bias 없이 unsupervised로 factor를 분리할 수 없다는 impossibility
  결과. §2의 "normal-only 데이터가 nuisance/defect 구분을 자동으로 알아낸다고
  주장하지 않는다"는 입장을 외부 근거로 방어할 수 있다.
- **Variational Autoencoders and Nonlinear ICA: A Unifying Framework (iVAE)**
  (Khemakhem et al., AISTATS 2020) —
  [arXiv:1907.04809](https://arxiv.org/abs/1907.04809).
  auxiliary variable로 conditioning하면 식별 가능성이 회복된다는 결과.
  support set을 auxiliary variable로 보는 §4의 support-conditioned transform에
  이론적 라이선스를 준다.
- **Contrastive Learning Inverts the Data Generating Process**
  (Zimmermann et al., ICML 2021) —
  [arXiv:2102.08850](https://arxiv.org/abs/2102.08850).
  어떤 조건에서 학습된 표현이 latent factor를 복원하는가에 대한 구성적 조건.

## 3. Coherent nuisance 분리 (§2, §8)

- **Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization**
  (Huang & Belongie, ICCV 2017) —
  [arXiv:1703.06868](https://arxiv.org/abs/1703.06868).
  "channel-wise 통계 = style(nuisance), 나머지 = content"라는 가장 저차원적인
  coherent-nuisance 모델. Low-DOF adapter family의 사실상 상한선.
- **FiLM: Visual Reasoning with a General Conditioning Layer**
  (Perez et al., AAAI 2018) —
  [arXiv:1709.07871](https://arxiv.org/abs/1709.07871).
  §4의 nuisance code `c(x)` → conditioning 경로의 표준 형태.
- **Revisiting Batch Normalization for Practical Domain Adaptation (AdaBN)**
  (Li et al., 2016) — [arXiv:1603.04779](https://arxiv.org/abs/1603.04779).
  통계 교체만으로 domain shift를 흡수하는 극단적 baseline. Identity-memory
  family의 N1(robust feature normalization) 대조군 설계에 직접 사용 가능.
- **Learning Invariances in Neural Networks (Augerino)**
  (Benton et al., NeurIPS 2020) —
  [arXiv:2010.11882](https://arxiv.org/abs/2010.11882).
  invariance의 **정도**를 학습 파라미터로 두는 방식. §2의 "이 bias는 반드시
  soft하게 적용해야 한다"를 정량화하는 한 가지 구현.

## 4. Objective-score mismatch (R9, §10)

현재 프레임워크의 최대 설계 공백은 NLL surrogate와 deployed hard 1-NN score의
불일치다. 아래는 "likelihood는 탐지 지표가 아니다"를 이미 확립한 문헌이다.

- **Do Deep Generative Models Know What They Don't Know?**
  (Nalisnick et al., ICLR 2019) —
  [arXiv:1810.09136](https://arxiv.org/abs/1810.09136).
  Flow가 OOD 입력에 더 높은 likelihood를 부여하는 현상. objective mismatch를
  우리 세팅의 특수 문제가 아닌 구조적 문제로 프레이밍할 근거.
- **Detecting Out-of-Distribution Inputs to Deep Generative Models Using
  Typicality** (Nalisnick et al., 2019) —
  [arXiv:1906.02994](https://arxiv.org/abs/1906.02994).
  density 대신 typical set 소속을 보는 대안 score.
- **Input Complexity and Out-of-Distribution Detection with Likelihood-based
  Generative Models** (Serrà et al., ICLR 2020) —
  [arXiv:1909.11480](https://arxiv.org/abs/1909.11480).
  likelihood가 입력 복잡도에 지배된다는 진단과 ratio 형태의 교정. §9
  diagnostic 5(log-determinant behavior) 해석에 필요.
- **Likelihood Ratios for Out-of-Distribution Detection**
  (Ren et al., NeurIPS 2019) —
  [arXiv:1906.02845](https://arxiv.org/abs/1906.02845).
  likelihood를 background 성분과 semantic 성분으로 분해해 교정. 이 분해는
  §2의 `g(x)` + `delta(x,p)` 분해와 같은 형태이며, coherent-nuisance prior를
  score 수준에서 구현한 선례로 읽을 수 있다.
- **A Metric Learning Reality Check** (Musgrave et al., ECCV 2020) —
  [arXiv:2003.08505](https://arxiv.org/abs/2003.08505).
  "training objective 개선이 deployed metric 개선과 같지 않다"는 실증과 공정
  비교 프로토콜. §9의 공통 실험 계약을 강화하는 방법론 참고.
- **Understanding Contrastive Representation Learning through Alignment and
  Uniformity on the Hypersphere** (Wang & Isola, ICML 2020) —
  [arXiv:2005.10242](https://arxiv.org/abs/2005.10242).
  끌어당김(alignment)과 geometry 유지(uniformity)를 측정 가능한 두 항으로
  분해. R2(normal-shift alignment)와 R4(defect preservation)의 트레이드오프를
  그대로 두 축으로 계측할 수 있다.

## 5. 가역성과 geometry audit (R8)

- **Understanding and Mitigating Exploding Inverses in Invertible Neural
  Networks** (Behrmann et al., AISTATS 2021) —
  [arXiv:2006.09347](https://arxiv.org/abs/2006.09347).
  Flow가 수치적으로 non-invertible해지는 조건과 진단 절차. §9 diagnostic
  4-5의 계측 설계를 그대로 차용할 수 있다.
- **Spectral Normalization for Generative Adversarial Networks**
  (Miyato et al., ICLR 2018) —
  [arXiv:1802.05957](https://arxiv.org/abs/1802.05957).
  Lipschitz 상한을 저비용으로 거는 표준 기법.
- **Contractive Auto-Encoders: Explicit Invariance During Feature Extraction**
  (Rifai et al., ICML 2011).
  Jacobian norm penalty로 contraction을 의도적으로 부과하는 반대편 논문.
  우리 세팅은 같은 양에 **하한**을 걸어야 하므로 대칭 참고자료로 유용하다.

## 6. Support-conditioned static transform (R7)

R7은 query-time parameter/memory/support-statistic 갱신을 금지한다. 아래는
갱신 없이 support에 조건화하는 기존 정식화다.

- **Fast and Flexible Multi-Task Classification Using Conditional Neural
  Adaptive Processes (CNAPs)** (Requeima et al., NeurIPS 2019) —
  [arXiv:1906.07697](https://arxiv.org/abs/1906.07697).
  support set을 adaptation network에 통과시켜 FiLM 파라미터를 생성한다.
  gradient 없는 test-time 조건화이므로 §4의 "static conditional inference"
  정의를 이미 확립한 선례.
- **TaskNorm: Rethinking Batch Normalization for Meta-Learning**
  (Bronskill et al., ICML 2020) —
  [arXiv:2003.03284](https://arxiv.org/abs/2003.03284).
  support/query 통계를 어떻게 결합해야 leakage가 없는가. R7 위반 판정 기준
  설계에 직접 대응.
- **Tent: Fully Test-Time Adaptation by Entropy Minimization**
  (Wang et al., ICLR 2021) —
  [arXiv:2006.10726](https://arxiv.org/abs/2006.10726).
  우리가 배제하기로 한 쪽(파라미터 갱신)의 대표 사례. R7 경계를 논문에서
  방어할 때 필요한 대비군.

## 7. Transport family 설계 (§6 Flow Matching 행)

- **Flow Matching for Generative Modeling** (Lipman et al., ICLR 2023) —
  [arXiv:2210.02747](https://arxiv.org/abs/2210.02747).
- **Improving and Generalizing Flow-Based Generative Models with Minibatch
  Optimal Transport (OT-CFM)** (Tong et al., TMLR 2024) —
  [arXiv:2302.00482](https://arxiv.org/abs/2302.00482).
  최소 변위 transport를 명시적 목적으로 삽입하는 방법. §6의 N1
  "identity-near conservative transport"에 대한 가장 가까운 기존 답.
- **Flow Straight and Fast: Learning to Generate and Transfer Data with
  Rectified Flow** (Liu et al., ICLR 2023) —
  [arXiv:2209.03003](https://arxiv.org/abs/2209.03003).
  직선 경로 = 변위 최소화 관점.
- **Diffusion Schrödinger Bridge with Applications to Score-Based Generative
  Modeling** (De Bortoli et al., NeurIPS 2021) —
  [arXiv:2106.01357](https://arxiv.org/abs/2106.01357).
  reference measure에서 최소한으로 벗어나며 두 분포를 잇는 entropic OT 정식화.
  §8의 identity-near regularization의 원리적 버전.

## 8. Tail 지표의 통계적 신뢰성 (R2, R6)

- **A Gentle Introduction to Conformal Prediction and Distribution-Free
  Uncertainty Quantification** (Angelopoulos & Bates, 2021) —
  [arXiv:2107.07511](https://arxiv.org/abs/2107.07511).
  held-normal `q_0.995` 같은 empirical quantile을 few-shot에서 사용할 때의
  유한표본 보증과 exchangeability 가정. R6를 "분산이 크다" 이상으로 진술하려면
  필요.
- **Testing for Outliers with Conformal p-values**
  (Bates et al., Annals of Statistics 2023) —
  [arXiv:2104.08279](https://arxiv.org/abs/2104.08279).
  normal-only calibration set에서 outlier test를 구성하는 정식 프레임. §9의
  tail diagnostic을 p-value 기반으로 재정의할 수 있다.

## 9. 읽는 순서 제안

| 단계 | 섹션 | 산출 기대 |
|---|---|---|
| 실험 초기 (구조 확정 전) | §4 objective mismatch, §5 geometry | Normalizing Flow N2 설계의 제약 조건 확정 |
| Flow Matching 브랜치 개시 시 | §7 transport | identity-near transport의 구현 형태 결정 |
| Adapter/global branch 설계 시 | §3 nuisance, §6 static conditioning | R7 준수 조건과 low-DOF 상한 확정 |
| 결과 해석 및 작성 단계 | §2 식별 가능성, §8 tail 통계 | 프레임워크 §2 및 R2/R6 진술의 방어 |
