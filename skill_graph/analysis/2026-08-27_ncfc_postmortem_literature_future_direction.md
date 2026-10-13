# NCFC 실패 분석·문헌 조사·향후 방향

Date: 2026-08-27  
Scope: `NCFC-S1` 이후, 새 backbone LoRA 학습 전 문제 분해와 연구 방향 선택  
Status: `DESIGN CONVERGED / NO NEW GPU RUN`

## 결론

현재 바로 backbone에 LoRA를 붙이는 방향은 근거가 부족하다. 가장 먼저
해야 할 일은 **ECO-Audit v2**라는 별도 frozen/no-training 실험으로
held-base-1 실패를 다음 세 원인으로 분해하는 것이다.

1. 저활성 residual token에서 cosine 방향이 불안정한 측정 문제;
2. clipping·blending·경계 때문에 J0와 G2 입력 연산이 이미 비가환인 문제;
3. operator가 정상인데도 anomaly-active feature가 실제로 비가환인 문제.

장기 학습 방향은 **AGIAS(Activity-Gated Identified Action Sidecar)**로
정한다. 순서는 `ECO-Audit -> action 식별/오염 검증 -> shallow frozen-feature
sidecar -> held capacity gap -> protected late-block LoRA`다. 각 단계가
실패하면 뒤 단계를 열지 않는다. Direct single-path backbone LoRA는 닫는다.

## 현재 결과를 다시 해석하면

### 1. clean correction 1.0은 학습 성공이 아니다

NCFC M1은 held clean pair에서
`c*=R(x)-R(T(x))`를 직접 계산하고 같은 값을 clean target으로 사용한다.
따라서 clean correction cosine/norm `1.0/1.0`은 정의상 항등값이다. 의미
있는 신호는 anomaly input에서도 correction 방향이 `0.956--0.993`으로
유지되고 RN·residual norm·top-token support가 보존된다는 점이다. 이것도
deployable estimator가 아니라 paired-oracle compatibility 신호다.

### 2. correction alignment와 residual commutation은 다른 명제다

Residual은 `R(T(A(x)))-R(T(x))` 또는 `R(A(T(x)))-R(T(x))`로 계산되므로
양쪽에 동일 correction을 더하면 상쇄된다. 즉 current commutation 실패는
M1 correction을 더 정확히 예측한다고 자동으로 고쳐지지 않는다.

실제로 global anomaly correction은 높지만 strict residual commutation은
order별 `3/8`만 `>=0.90`이다. held base 1은 balanced `0.717--0.742`,
mask-inside `0.907--0.922`, 추정 outside `0.518--0.575`다. 다른 base의
inside는 `0.991--0.998`이다. outside 저활성 metric 문제가 큰 몫일 수
있지만, inside 열화와 높은 interaction norm 때문에 진짜 base-specific
상호작용도 남아 있다.

### 3. norm/top-k/RN 보존은 anomaly semantics가 아니다

Residual norm과 top-10 support가 유지돼도 token 방향이 회전하거나 실제
anomaly score가 손상될 수 있다. RN rank/neighbor/scale도 geometry 안전성이지
canonicalization utility가 아니다. 따라서 representation commutation과
detector ranking/localization을 별도 표·별도 gate로 유지해야 한다.

### 4. 기존 실패는 capacity보다 identifiability를 가리킨다

- TARCD S1--S3: loss conditioning을 고쳐도 mechanism `0/12`, static
  correction보다 약했다.
- SCOPE/CEB: 정상 q99는 좋아졌지만 protected stream 대비 anomaly ranking
  기여가 0이었다.
- Oracle family/PCGrad: routing·optimizer를 바꿔도 hard null과 anomaly
  protection이 충분하지 않았고 실제 paired objective conflict는 0이었다.
- CRFR: nuisance code shuffle 효과가 거의 0이고 causal factorization `0/12`.
- SAEC: learned relation은 shuffle보다 낫지만 no-commutation control보다
  나빴고 action identification `0/4`, synthetic contribution `0/16`.

따라서 rank·layer·LR·loss를 더 넓게 탐색하기 전에 “shift action을 anomaly와
독립적으로 식별할 관측 정보가 있는가”를 먼저 증명해야 한다.

## 문헌에서 채택할 것과 채택하지 않을 것

### 진단 metric

[Kornblith et al., CKA, ICML 2019](https://proceedings.mlr.press/v97/kornblith19a.html)는
representation geometry 비교에 유용하지만 exact frozen-coordinate
commutation보다 관대하다. CKA는 coordinate drift 설명용으로만 채택하고,
coordinate cosine/SRE failure를 뒤집는 gate로 쓰지 않는다.

[SVCCA, NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/file/dc6a7e655d7e5840e66733e9ee67cc69-Paper.pdf)와
[PWCCA, NeurIPS 2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/a7a3d70c6d17a73140918996d03c014f-Abstract.html)는
low-energy direction이 비교를 지배하는지 보는 보조 분석으로만 둔다.

### operator/equivariance

[DUET, ICML 2023](https://proceedings.mlr.press/v202/suau23a.html)는
brightness/contrast clipping이 group composition을 깨뜨릴 수 있음을
명시한다. 따라서 J0를 exact group action으로 가정하지 않고 pixel
commutator·clipping·boundary를 먼저 감사한다.

[AugSelf, NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/94130ea17023c4837f0dcdda95034b65-Abstract.html)와
[Homomorphism Autoencoder, ICML 2023](https://proceedings.mlr.press/v202/keurti23a.html)는
free adapter보다 명시적 augmentation/action prediction과 composition
constraint를 쓰는 근거다. 다만 clean paired action 학습이 anomalous query의
오염 문제를 해결하지는 않으므로 별도 contamination gate가 필요하다.

[Locatello et al., ICML 2019](https://proceedings.mlr.press/v97/locatello19a)와
[weakly supervised disentanglement, ICML 2020](https://proceedings.mlr.press/v119/locatello20a.html)는
개입/약한 supervision 없이 free factorization을 식별할 수 없다는 경계를
제공한다. CRFR의 free nuisance code를 다시 키우는 방향은 기각한다.

### anomaly detection과 adaptation

[PatchCore, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html)와
[AnomalyDINO, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html)는
frozen patch matching이 강한 few-shot 기준임을 보여준다. C0 frozen stream은
모든 learned 후보 옆에서 전체 결과를 그대로 보고하고 abstention fallback으로
유지한다.

[CutPaste, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Li_CutPaste_Self-Supervised_Learning_for_Anomaly_Detection_and_Localization_CVPR_2021_paper.html),
[NSA, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7519_ECCV_2022_paper.php),
[SimpleNet, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.html)는
normal-only synthetic supervision과 shallow feature head의 근거다. 이것은
backbone LoRA보다 먼저 시험할 저위험 학습 pilot이다. Synthetic success는
real-defect transfer 증거로 사용하지 않는다.

[AA-CLIP, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Ma_AA-CLIP_Enhancing_Zero-Shot_Anomaly_Detection_via_Anomaly-Aware_CLIP_CVPR_2025_paper.pdf)는
naive adapter가 pretrained generalization을 해칠 수 있고 residual/frozen
path가 중요하다는 설계 원리만 채택한다. task와 supervision이 다르므로
성능 수치를 이 설정에 이식하지 않는다.

[LoRA, ICLR 2022](https://openreview.net/forum?id=nZeVKeeFYf9)는
parameter efficiency만 보장한다. Low rank를 anomaly preservation,
identifiability, safety 또는 novelty로 해석하지 않는다.

## 즉시 다음 실험: ECO-Audit v2

동일한 4 held base × 2 sign × 2 order × 2 readout의 32 cell을 유지한다.

- support 16장을 이미지별 정확히 `1/16` 가중한 empirical Q10으로
  `tau_raw/tau_std`를 held access 전에 한 번 봉인한다.
- original mask-balanced token cosine을 그대로 보존한다.
- active/inactive와 interior/boundary/outside에 대해 global flattened cosine,
  SRE, target energy coverage, norm ratio, feature-error energy를 추가한다.
- pixel-level `T(A)`/`A(T)` commutator, pre-clip violation, saturation,
  spatial commutator energy를 측정한다.
- CKA/token geometry는 explanation-only로 둔다.
- N5는 activity/region/norm/energy를 보존하고 correspondence만 파괴한다.
- empty/small/zero/nonfinite stratum은 pass로 빼지 않고 invalid 처리한다.
- 32 cell 중 readout/order가 불일치하거나 여러 원인이 섞이면
  `UNRESOLVED`로 종료한다. 평균·median·majority rescue는 금지한다.

판정에 따른 행동:

| 결과 | 해석 | 허용되는 다음 행동 |
| --- | --- | --- |
| operator confounded | 입력 합성부터 비가환 | 한 번의 energy-matched no-clipping/linear-light control |
| low-activity component | original cosine 실패의 큰 부분이 저활성 방향 | prospective metric 수정, backbone adaptation 종료 |
| boundary spillover | patch/mask boundary가 지배 | 한 번의 boundary-safe construction control |
| coordinate drift | 좌표는 다르나 geometry 유사 | frozen scorer-sensitivity probe만 허용 |
| genuine active mismatch | 실제 표현 상호작용 후보 | Gate B action identifiability만 개방 |
| invalid/mixed/readout-order disagreement | 원인 분리 실패 | 학습 중단, 새 설명 사전등록 |

## 조건부 장기 방향: AGIAS

### Gate B — 첫 learned milestone

두 개 parameter만 가진 support-anchored/query-conditioned J0 decoder를 clean
paired support intervention으로 학습한다. Held clean action MAE/sign,
action-shuffle/base-ID/wrong-sign, anomaly insertion drift, generator/base/
area/boundary/spectrum leakage, out-of-orbit abstention, exact C0 fallback을
각각 따로 gate한다.

`<50% token contamination`은 사실이 아니라 검증할 가정이다. 사전 고정한
sub-majority/exact-majority/super-majority/boundary/spillover panel에서 severity가
커질수록 acceptance가 증가하지 않아야 하며, contract를 벗어나면 C0로
abstain해야 한다. 실패하면 AGIAS와 모든 LoRA를 닫는다.

### Gate C — backbone 이전 shallow sidecar

Oracle action을 먼저, inferred action을 다음으로 넣는 width-16 zero-start
frozen-feature sidecar를 시험한다. Frozen C0 bytes는 불변이다.

- representation table: active cosine/SRE, anomaly residual norm/top-energy,
  parallel/perpendicular preservation, RN, matched static/action-shuffle/
  no-commutation controls;
- detector table: 전체 accepted+fallback denominator에서 C0 대비 ranking/
  localization, held generator, coverage.

Score-only gain은 commutation repair가 아니며 LoRA를 열 수 없다. Gate C가
목표를 달성하면 LoRA 없이 종료한다.

### Gate D — capacity gap 뒤 단 한 번의 LoRA

Output-space control과 late-block Jacobian basis/coefficients를 training cells에만
fit하고, untouched held bases/generator에서 한 번 평가한다. Held residual은
target·projection·layer·Q/V·rank 선택에 사용하지 않는다. Action error,
operator, shortcut, scorer 문제가 아닌 late-block capacity gap이 남을 때만
zero-start rank-4 block-11 Q/V side-branch pilot을 한 번 실행한다. Rank 4/QV는
최적이라고 주장하지 않는다. 실패 후 sweep은 없다.

## 설계 평가와 남은 위험

Deep-research 평가 trace는 `0.78 -> 0.82 -> 0.80 -> 0.80`이며 Stage 2
ECO-Audit v2가 최고점 `0.82`였다. 두 단계 연속 개선이 없어 수렴 종료했다.

남은 핵심 위험은 다음이다.

- contamination stress/abstention의 exact finite contract;
- Gate-B permutation test의 lower-is-better 방향과 tie 규칙;
- Gate C/D representation 및 detector metric의 exact delta/aggregation;
- remote preflight command/path enforcement 구현 및 독립 review;
- synthetic orbit에서 real acquisition shift로의 전이.

따라서 현재의 확정 행동은 ECO-Audit v2 설계/구현/원격 frozen audit까지다.
AGIAS Gate B 이후는 ECO 결과가 genuine active mismatch로 판정될 때만 연다.

## 산출물

- NCFC result: `skill_graph/experiments/2026-08-27_ncfc_s1_stage_minus1/report.md`
- Deep-research ledger: `.omx/specs/deep-research-autoresearch-ncfc-postmortem-activity-aware-commutation/evidence-ledger.md`
- Best immediate contract: `.omx/specs/deep-research-autoresearch-ncfc-postmortem-activity-aware-commutation/stage-02/method.md`
- Conditional AGIAS design: `.omx/specs/deep-research-autoresearch-ncfc-postmortem-activity-aware-commutation/stage-04/method.md`
- Final synthesis: `.omx/specs/deep-research-autoresearch-ncfc-postmortem-activity-aware-commutation/current-method.md`
