# Few-shot AD under Distribution Shift: 통합 실험 Ledger

기준일: 2026-08-07  
범위: `Baseline1`의 target-specific representation/module 탐색과 이에 직접 연결되는
support-only Stage -1 evidence  
상태: **현재 새 module 후보는 열려 있지 않음**

이 문서는 experiment directory에 흩어진 개별 사전등록·결과·artifact를 대체하지 않는다.
한 곳에서 연구 질문, 설계 계약, 어떤 가설을 왜 시험했는지, 무엇이 살아 있고 무엇이
닫혔는지, 그리고 다음 실험의 진입 조건을 볼 수 있게 하는 통합 의사결정 기록이다.

## 1. 한 줄 결론

Few-shot normal support만으로 unseen normal shift와 defect를 완전히 식별할 수 없으므로, backbone
뒤 module은 단순 normal fitting이 아니라 **정의된 nuisance에는 덜 민감하면서 local/broad
defect evidence와 deployed 1-NN geometry를 보존하는 제한된 변환**이어야 한다. all-patch
smoothness의 비변화-patch 오염을 제거한 Stage -1a1에서도 conditional-smoothness gate가
`0/4`였으므로, Normalizing Flow의 재설계나 Flow Matching/Diffusion/DiT module 구현은 아직
허가되지 않는다. 다만 Stage -1b는 shared photometric intervention에 대한 cross-image direction
transfer를 `4/4`에서 확인했으므로, module이 아닌 한 번의 held-transform transfer/retention
diagnostic만 조건부로 허용했다. Stage -1c가 이를 `0/4`로 종료했으므로 그 direction을 fixed
projection이나 architecture interface로 쓰는 branch도 닫힌다.
Stage -1d의 score-aligned rank-4 residual audit도 sentinel `0/2`였으며, 5NN mean의 약 `2%`
효과 외에는 1NN/15NN/soft-min에서 안정적 개선이 없었다. 따라서 단순 k 변경도 새 branch로
승격하지 않는다.

## 2. 문제 설정과 설계 계약

정상 support를 `S`, frozen backbone patch feature를 `h(x,p)`, support-conditioned transform을
`T_S`라고 둔다.

\[
z(x,p)=T_S(h(x,p)), \qquad s(x,p)=\min_{m\in\mathcal M_S}\|z(x,p)-m\|_2.
\]

여기서 `T_S`는 density를 높이는 장치가 아니라 다음 상충 조건을 만족해야 한다.

| 요구 | 의미 | 불충족 시 위험 |
| --- | --- | --- |
| R1 target specificity | target support에 맞는 변화 | raw memory보다 나을 근거가 없음 |
| R2 normal-shift alignment | held normal의 high-tail score 감소 | normal inflation 지속 |
| R3 coherent nuisance separation | 조명·색조·smooth illumination을 제한적으로 처리 | global shift를 defect로 오인 |
| R4 defect preservation | local과 broad defect contrast 유지 | anomaly absorption |
| R5 locality | pixel/patch evidence와 AU-PRO 보존 | image score만 좋아지는 착시 |
| R6 few-shot stability | support split·seed에서 일관 | 16-shot 우연 적합 |
| R7 static inference | query-time update 없음 | TTA/data leakage 혼입 |
| R8 geometry auditability | neighbor/rank/scale/Jacobian 검사 | score collapse를 숨김 |
| R9 objective-score alignment | 학습 목적과 deployed hard 1-NN 일치 | NLL 개선이 탐지 개선으로 이어지지 않음 |

### 왜 coherent-nuisance prior인가

이미지 shift는 조명, exposure, white balance처럼 image-wide 또는 공간적으로 smooth한 경우가
많다. 반면 defect는 대체로 국소 residual이지만 broad/global defect도 가능하다. 따라서 이는
normality 판별 규칙이 아니라 아래처럼 **soft inductive bias**로만 허용될 수 있다.

\[
h(x,p)=h_{content}(x,p)+g(x)+r(x,p)+\delta(x,p),
\]

여기서 `g`는 coherent nuisance, `r`은 normal local variation, `delta`는 defect residual이다.
global branch가 `delta`까지 제거하지 못하게 low-dimensional code, robust pooling,
identity-near bound, local non-contraction, broad-defect stress test가 필요하다.

## 3. 비교 단위와 금지선

```text
normal support -> frozen backbone -> support-specific transform T_S -> static memory
query normal/defect -> frozen backbone -> T_S -> static patch 1-NN anomaly map
```

- 초기 비교에서는 density/RGB/morphology/query adaptation을 끄고 static local 1-NN만 사용한다.
- `DiT`는 Flow/Flow Matching/Diffusion과 동급 family가 아니라 vector field/denoiser의
  parameterization이다.
- naive N0의 실패는 family 전체의 기각이 아니다. requirement failure를 정확히 겨냥한 N1,
  safety와 objective를 함께 맞춘 N2, 사전등록한 bounded N3까지의 evidence가 필요하다.
- 다만 이미 kill된 **동일 구성**의 parameter sweep은 새 N1/N2 mechanism이 아니다.

## 4. 의사결정 흐름

```text
Baseline1의 normal inflation / objective mismatch
        |
        +-- paired loss: normal alignment을 직접 주면 defect가 흡수되는가?
        |       -> 일부 alignment, defect retention 실패 -> close
        |
        +-- shift-conditioned Flow: global task code로 해결되는가?
        |       -> inflation/identity/retention 실패 -> close
        |
        +-- semantic local metric: semantic nuisance subspace가 있는가?
        |       -> 안전하지만 shift effect 부족·불안정 -> close
        |
        +-- WTCF / latent geometry: support witness가 selective direction을 주는가?
        |       -> JVP validity 및 witness sufficiency 미충족 -> close
        |
        +-- global/local coherence audit: coherent-nuisance prior 자체가 측정되는가?
                -> coherence는 관측, preregistered smoothness gate 1/4 -> inconclusive
                -> 새 module 구현 차단
```

## 5. 실험별 통합 결과

### 5.1 Paired-loss Flow diagnostic — alignment만 주면 안전하지 않다

| 항목 | 내용 |
| --- | --- |
| 질문 | Normal pair alignment loss가 held shifted-normal 1-NN inflation을 줄이는가? |
| 설정 | DINOv3-H+/16 + DVT + patch-local MLP Flow, four-object fixed-16 support-only |
| 가장 강한 arm | V3 raw pair + NLL weight 0.1 |
| 관측 | shift reduction `9.77%`, 4/4 object 양성이나 defect retention macro `0.8707` (`fruit_jelly .7372`, `vial .6683`) |
| gate | shift reduction `>=10%`, macro defect retention `>=.97`, 모든 object `>=.90` |
| 판정 | `KILL_FOR_CLAIM / RETAIN_BASELINE1 / NO_PUBLIC_ACCESS` |
| 설계 교훈 | normal pair를 맞추는 목적만으로는 local defect direction도 함께 줄어든다. alignment는 selective safety constraint 없이 충분하지 않다. |

원문: [preregistration](../experiments/2026-08-04_flowtte_paired_loss_diagnostic/preregistration.md),
[report](../experiments/2026-08-04_flowtte_paired_loss_diagnostic/report.md).

### 5.2 Shift-conditioned patch-local Flow — global task code만으로는 해결되지 않는다

| 항목 | 내용 |
| --- | --- |
| 질문 | task register와 bounded residual FiLM이 shift를 설명하고 defect는 남길 수 있는가? |
| 관측 | correct-register conditioning이 held shifted-normal inflation을 macro `20.41%` **증가**시켰고, zero/shuffled control 대비 win `0/4` |
| 안전성 | identity 실패 `3/4`; `fruit_jelly/walnuts` defect contrast `0.5396/.6846`; post-training logdet cancellation도 tolerance 초과 |
| 판정 | `EARLY_VALIDITY_KILL / KILL_FOR_CLAIM / NO_PUBLIC_ACCESS` |
| 설계 교훈 | global code가 있다고 해서 nuisance와 defect를 식별하지는 않는다. arbitrary patchwise correction을 허용하는 조건부 flow는 안전하지 않다. |

원문: [preregistration](../experiments/2026-08-04_shift_conditioned_patch_local_flow/preregistration.md),
[report](../experiments/2026-08-04_shift_conditioned_patch_local_flow/report.md).

### 5.3 SemanticAtlasMetric — 안전한 local metric만으로 effect가 충분하지 않다

| 항목 | 내용 |
| --- | --- |
| 질문 | cluster-local nuisance subspace metric이 shift를 흡수하면서 geometry/defect를 보존하는가? |
| 관측 | defect retention `0.9992` macro, min clean top-10 neighbor retention `>=.9847`, pair Spearman `>=.9994` |
| 한계 | shifted-normal inflation reduction `0.6788%` (요구 `10%`), `walnuts` regression, fold-refit ARI `0.5274` |
| 판정 | `KILL_FOR_CLAIM / RETAIN_BASELINE1 / NO_CONTINUE / NO_PUBLIC_ACCESS` |
| 설계 교훈 | anti-absorption과 geometry 보존은 필요조건이지만 normal-shift effect가 작거나 unstable하면 module claim을 열 수 없다. |

원문: [preregistration](../experiments/2026-08-05_flowtte_semantic_atlas_metric/preregistration.md),
[report](../experiments/2026-08-05_flowtte_semantic_atlas_metric/report.md).

### 5.4 WTCF sentinel — support witness는 검증된 nuisance direction이 아니다

| 항목 | 내용 |
| --- | --- |
| 질문 | repeated normal-residual witness에서 발견한 direction만 selective contraction할 수 있는가? |
| 범위 | `can/seed0` sentinel만 실행; 나머지 14 case는 scheduling하지 않음 |
| 관측 | pair Spearman `.998262`, neighbor retention `.929688`, inverse error `9.54e-7`은 통과 |
| 실패 | witness JVP `.9236--.9493`; frozen target `<=.85` 미달 |
| 판정 | `EARLY_VALIDITY_KILL / NO_FULL_STAGE / NO_PUBLIC_ACCESS` |
| 설계 교훈 | witness/correspondence는 candidate direction의 evidence source일 뿐, selective normal contraction의 성공 근거가 아니다. |

원문: [preregistration](../experiments/2026-08-04_wtcf_stage_minus1/preregistration.md),
[report](../experiments/2026-08-04_wtcf_stage_minus1/report.md).

### 5.5 Flow latent geometry audit — geometry의 관측은 mechanism 승격이 아니다

| 항목 | 내용 |
| --- | --- |
| 질문 | A6 Flow가 proven normal-variation direction과 defect-like direction을 분리하는가? |
| 관측 | 5,832 candidate 중 6 direction 수용, 9 case 중 `8/9`가 required three witness 미달; principal direction은 `9/9` contraction, 다른 family는 expansion 경향 |
| 판정 | `INCONCLUSIVE_WITNESS_SENTINEL / H1_BLOCKED / EXPLORATORY_OBSERVATIONS_ONLY` |
| 설계 교훈 | 16-shot support만으로 test-time shift direction을 안정적으로 얻는다는 가정을 세울 수 없다. |

원문: [preregistration](../experiments/2026-08-06_flow_latent_geometry_stage_minus1/preregistration.md),
[report](../experiments/2026-08-06_flow_latent_geometry_stage_minus1/report.md).

### 5.6 Global/local coherence Stage -1a0 — prior의 부분 증거와 현재의 중단점

이 experiment는 새 module 성능을 측정하지 않고, coherent-nuisance prior가 frozen
Baseline1 representation에서 사전등록한 방식으로 관측되는지만 audit했다.

| 항목 | 내용 |
| --- | --- |
| 대상 | MVTec AD2 single-image, `fruit_jelly/walnuts/vial/fabric`, fixed-16 support |
| 조건 | `S_global`, `S_smooth`, `D_local`, `D_broad`, `SD_global_local`, `SD_global_broad` |
| 병렬 실행 | local GPU 0과 dsba5 GPU 3; 동일 checkpoint/manifest/runner hash |
| custody | public/private/ground-truth access 모두 false |
| reproducibility | input ledger 동일, 모든 aggregate metric local–dsba5 delta `<=6.3e-7` |
| 부분 관측 | global coherence가 모든 object에서 local/broad defect coherence보다 큼; SD defect contrast retention은 local `.903--.978`, broad `.818--.984` |
| gate 실패 | preregistered global-smoothness > defect-smoothness가 `fabric`만 충족 (`1/4`, 요구 `>=3/4`) |
| 판정 | `INCONCLUSIVE_OR_NOT_SUPPORTED / NO_NEW_MODULE / NO_PUBLIC_ACCESS` |

핵심 해석은 “coherent displacement” 관측만으로 module을 설계할 수 없다는 것이다.
현재 adjacent-patch cosine smoothness는 local rectangular defect의 **변하지 않은 다수 patch**에
의해 높아질 수 있으므로, post-hoc threshold 조정이 아니라 새로운 measurement definition이
필요하다.

원문: [preregistration](../experiments/2026-08-07_global_local_coherence_stage_minus1/preregistration.md),
[report](../experiments/2026-08-07_global_local_coherence_stage_minus1/report.md),
[parity verdict](../../results/stage_minus1/global_local_coherence/aggregate_v2/verdict.json).

## 6. 현재 evidence가 허용하는 것과 허용하지 않는 것

| 항목 | 상태 | 이유 |
| --- | --- | --- |
| Baseline1 유지 | 허용 | 대체 candidate가 shift effect·defect·geometry를 동시에 통과하지 못함 |
| 새 global/local module 구현 | 차단 | Stage -1a0 `1/4`, mask-free Stage -1a1 conditional gate `0/4`; Stage -1b는 module 근거가 아님 |
| 기존 paired loss/FiLM/atlas 설정 재튜닝 | 차단 | 동일 configuration family의 사전등록된 gate가 이미 실패 |
| Flow family 전체 기각 | 허용하지 않음 | N0/N1 실패가 requirement-aware N2/N3 전체의 불가능성을 의미하지 않음 |
| Flow Matching/Diffusion/DiT 도입 | 보류 | family 비교 전에 global/local premise와 measurement contract가 부족 |
| public benchmark evaluation | 차단 | Stage -1 assumption/mechanism gate가 성립하지 않음 |

## 7. 다음 실험의 유일한 허용 진입점

다음 단계는 새 architecture가 아니었고, 아래 Stage -1a1 measurement audit을 별도
사전등록해 완료했다.

1. **변화량 조건부 smoothness**: \(\|\Delta_p\|\)가 충분히 큰 patch만 쓰되, threshold는
   결과를 본 뒤 조정하지 않고 transform 생성 규칙과 함께 고정한다.
2. **mask-free inference proxy**: synthetic mask를 evaluation에만 쓰고, 실제 model branch가
   mask를 요구하지 않는 proxy를 정의한다.
3. **local/broad 분리 검증**: global photometric, smooth illumination, local defect, broad defect,
   그리고 shift+defect 조합을 모두 포함한다.
4. **안전성 동시 gate**: defect contrast retention, clean geometry, held-normal q995 tail을
   독립적으로 통과해야 한다.
5. **두 lane 재현성**: fixed support content ledger, offline checkpoint, local/dsba5 aggregate
   parity를 사전등록한다.

결과적으로 global coherence와 energy coverage는 `4/4`였지만, active-pair conditional
smoothness는 `0/4`였다. 즉 inactive patch inflation을 제거해도 global coherence prior는
현재 feature-space operationalization에서 충분한 selector가 아니다. 같은 threshold/평균의
재조정은 허용하지 않는다. 별도 Stage -1b에서 shared photometric transform의 cross-image transfer는
`4/4` 통과했지만, 이는 selector/architecture 근거가 아니라 held-transform transfer/defect-retention
하나를 더 시험할 수 있는 evidence다. 그 gate가 통과한 뒤에만 다음처럼 requirement-aware
candidate를 하나씩 열 수 있다.

```text
validated factorisation measurement
  -> one bounded global/local interface
  -> one family (Flow OR Flow Matching OR Diffusion)
  -> one parameterization (MLP/CNN/DiT)
  -> support-only safety/mechanism gate
  -> only then public benchmark gate
```

## 8. Artifact map

| 역할 | 위치 |
| --- | --- |
| 구조 탐색 framework | [2026-08-07_fewshot_distribution_shift_structure_search_framework.md](2026-08-07_fewshot_distribution_shift_structure_search_framework.md) |
| 이번 factorisation audit 설계 | [preregistration](../experiments/2026-08-07_global_local_coherence_stage_minus1/preregistration.md) |
| 이번 audit 결과 | [report](../experiments/2026-08-07_global_local_coherence_stage_minus1/report.md) |
| local/dsba5 parity gate | `results/stage_minus1/global_local_coherence/aggregate_v2/verdict.json` |
| retained reference | [Baseline1](../methods/baseline1.md) |

## 9. Baseline1 k=5 운용 기본값 A/B

Stage -1d에서 5-NN이 1-NN보다 score tail을 완화할 가능성이 관측됐지만 이를
방법론 기여나 k tuning으로 확장하지 않고, 고정 `k=5` 운용 후보만 8-object
TESTpublic에서 검증했다. K5-K1 macro 변화는 pAUROC `-0.004222`, F1
`-0.002285`, AU-PRO `-0.003457`이었다. good-pixel p99.9는 감소했으나 정상
pooled FPR은 증가했고, `sheet_metal` AU-PRO가 `-0.030263` 하락했다.

따라서 `RETAIN_K1`: k=5는 outlier 완화 진단으로는 의미가 있으나 기본 scorer로
채택하지 않는다. 추가 k sweep이나 객체별 k 선택도 진행하지 않는다. 상세 결과는
[report](../experiments/2026-08-07_baseline1_knn5_default_ab/report.md)에 있다.

## 10. Stage -1e: global condition은 제거가 아니라 수송 조건인가

Stage -1c/1d의 projection 실패 뒤, query global statistic을 제거하지 않고 정상 support
distribution을 query 쪽으로 이동하는 최소 구조를 검증했다. channel median 위치 이동(E2)과
median+MAD 대각 수송(E3)을 Stage -1c projection negative control과 비교했다.

validity와 local/dsba5 parity는 통과했지만 E2/E3는 모두 `0/4`였다. E2 tail reduction은
`fruit_jelly/walnuts/vial/fabric = -.453/-.333/-.082/+.054`였고, 결함 query가 만든 location
conditioner drift는 local/broad에서 `.218--.807`로 전부 `.10` gate를 넘었다. E3의 추가 scale은
일관된 이득을 만들지 못하고 broad defect contrast/geometry를 더 훼손했다.

따라서 naive global moment alignment는 중단한다. 이 결과는 target-specific module 전체를
기각하지 않지만, 다음 병목이 architecture capacity가 아니라 defect-invariant condition
identifiability임을 보여준다. 해당 식별 gate 전에는 Flow Matching/Diffusion/DiT로 확장하지
않는다. 상세 결과는
[report](../experiments/2026-08-07_conditional_normalization_transport_stage_minus1e/report.md)에 있다.

## 11. Stage -1f: spatial consensus로 defect-invariant condition을 만들 수 있는가

Stage -1e의 condition-identifiability 병목을 직접 검증하기 위해 full feature grid를 `4x4` block으로
나누고, 고정 geometric-median consensus 주변 8 block만 사용하는 C2를 C0 naive median과
비교했다. C0의 oracle-direction cosine은 `.921--.971`로 높았지만, C2는 이를 개선하지 못했고
oracle relative error도 `.51--.63`이었다. C2 local/broad contamination drift는 `.65--3.95`, C0
대비 감소율은 네 object 모두 동시 gate를 통과하지 못했다. local/dsba5 parity mismatch는 0이다.

따라서 spatial-consensus conditioner는 `0/4`, `NO_CONTINUE`다. 이미지 공간 local defect가 fused
DINO feature에서는 여러 region의 conditioner residual로 퍼질 수 있으므로, 다음 허용 질문은
block hyperparameter가 아니라 layer/position별 receptive-field spillover measurement다. 상세
결과는 [report](../experiments/2026-08-07_spatial_consensus_conditioner_stage_minus1f/report.md)에 있다.

## 12. Stage -1g: local defect가 어느 DINO layer에서 non-local해지는가

Stage -1f의 실패 원인을 conditioner 구조와 분리하기 위해 layers `[7,15,23,31]` 및 fused feature의
receptive-field spillover를 support-only로 측정했다. Layer 7은 local far-spill
`fruit_jelly/walnuts/vial/fabric = .088/.082/.060/.067`로 localization gate를 `4/4` 통과했지만,
local condition contamination은 `.214/.688/.281/.472`로 feasibility gate를 `0/4` 통과했다.
Layers 15/23/31은 localization도 `0/4`였다. Local/dsba5 parity mismatch는 0이다.

따라서 초기 layer가 이미지 공간의 defect locality를 보존한다는 제한적 표현 인사이트는 얻었지만,
이를 곧바로 condition 또는 anomaly layer로 선택할 수는 없다. 다음으로 허용되는 것은 conditioner
입력(layer 7)과 기존 fused anomaly scorer 입력을 분리하는 단일 support-only interface diagnostic뿐이다.
Layer sweep, learned fusion, Flow Matching/Diffusion/DiT 및 public benchmark는 계속 차단한다. 상세
결과는 [report](../experiments/2026-08-07_layerwise_defect_spillover_stage_minus1g/report.md)에 있다.

## 13. Stage -1h: layer-7 gate와 fused scorer를 분리하면 condition contamination이 줄어드는가

Stage -1g가 허용한 단일 split-interface를 검증했다. Layer-7 residual consensus에 가까운 고정 75%
patch만 선택하고, 그 index의 fused residual로 condition을 계산했다. C0 대비 C7 local/broad drift
감소율은 `fruit_jelly 6.8/7.2%`, `walnuts -0.2/.4%`, `vial 5.4/3.1%`,
`fabric 21.8/13.6%`로 25% gate에 미달했다. Tail reduction은 `walnuts`만 `12.1%`였고 나머지는
악화됐다. Strict와 weak continuation은 모두 `0/4`; local/dsba5 parity mismatch는 0이다.

따라서 layer 7의 defect locality는 유효한 표현 인사이트지만 mask-free target-condition selector로
전환되지 않았다. Layer-separated spatial gating은 종료하며 fraction/distance/layer를 튜닝하지 않는다.
Flow/Flow Matching/Diffusion/DiT와 public benchmark도 계속 차단한다. 상세 결과는
[report](../experiments/2026-08-07_layer7_fused_split_interface_stage_minus1h/report.md)에 있다.

## 14. Stage -1i: metadata 없이 frozen external visual prior가 독립 evidence가 되는가

이미지 metadata, filename, split signal은 답을 알고 쓰는 auxiliary에 가까워 폐기했다. 대신 pixel-only
frozen prior인 CLIP ViT-L/14@336 patch token을 DINOv3-H+ layer 7과 같은 fixed-16 support source에서
비교했다. 조건은 `clean/U/U+D_local/U+D_broad`, score는 고정 `k=5` mean NN이며 query adaptation,
test/public/private/GT 접근, descriptor/fusion tuning은 모두 금지했다.

결과는 strict `0/4`, continuation `0/4`다. CLIP/DINO scale-normalized ratio는
`fruit_jelly` shift `3.43598`, local/broad core `.67925/.24213`, local/broad selectivity
`.25023/.12048`; `walnuts` `.24476`, `1.75838/.66450`, `.49467/.20622`; `vial`
`13.47296`, `1.18431/.11233`, `.09466/.01286`; `fabric` `.52876`, `2.74793/1.34388`,
`1.25653/.96616`이다.

즉 CLIP은 `walnuts/fabric` 같은 일부 texture object에서 normal-shift sensitivity를 낮추고 defect
gain을 키우지만, cross-object 독립 evidence는 아니다. `fruit_jelly/vial`에서는 shift sensitivity가
오히려 크게 증가했고, selectivity gate가 전반적으로 무너졌다. 따라서 CLIP fusion, descriptor swap,
prompt prior, per-object prior 선택 follow-up은 금지한다. 상세 결과는
[report](../experiments/2026-08-08_frozen_clip_prior_separability_stage_minus1i/report.md)에 있다.

## 14.5 Stage -1j: DINO channel readout만으로 representation을 shift-invariant하게 만들 수 있는가

Backbone 뒤 module 이전에 representation 자체를 shift-invariant하게 읽을 수 있는지 support-only로
검증했다. DINOv3-H+ fused feature에서 normal-support photometric pair sensitivity가 낮은 lower-50%
channel만 읽는 `I1_invariant_mask`를 만들고, 같은 cardinality/scale의 `I2_score_shuffled_mask`를
hard null로 뒀다. Flow, DVT, query update, test/public/private/GT, filename/metadata access는 모두
껐다.

결과는 양면적이다. I1의 held photometric displacement ratio는 `0.6078`로 크게 줄었고 shuffled
null은 `1.0052`였다. 따라서 frozen DINO 안에 photometric-sensitive coordinate가 있다는 점은
확인됐다. 그러나 local/broad defect retention은 `0.4941/0.5949`로 무너졌고, clean geometry도
Spearman `0.8039`, NN retention `0.6719`로 gate를 실패했다. 반대로 shuffled null은
local/broad retention `0.9750/0.9967`로 defect evidence를 보존했다.

따라서 verdict는 `ANOMALY_ABSORPTION`이다. "shift에 덜 민감한 channel만 읽기"는 representation
invariance를 만들 수 있지만, 그 channel selection이 anomaly evidence까지 제거한다. 이 branch에서는
trainable diagonal gate, channel-rank sweep, Flow/DiT/Diffusion/Flow Matching continuation을 열지
않는다. 다음 설계가 필요하다면 먼저 global nuisance suppression과 local/broad defect retention을
동시에 강제하는 새 식별 구조를 사전등록해야 한다. 상세 결과는
[report](../experiments/2026-08-08_dino_invariant_channel_readout_stage_minus1/report.md)에 있다.

## 22. Shift-stable channel을 key로만 쓰면 defect를 보존하며 retrieval이 안정화되는가

Stable lower-50% channel은 exact top-5 memory row를 고르는 key로만 사용하고, anomaly distance는
선택된 동일 row의 full 1280-D value로 계산했다. Full/full A0, stable/stable A1,
shuffled-key/full-value A2와 비교했다.

A3의 local/broad defect retention은 `1.0572/1.0155`로 보존 gate를 통과했지만, A1은
`0.4901/0.5945`로 이전 absorption을 재현했다. 즉 defect 손실은 stable key가 아니라 anomaly
value에서 나머지 channel을 제거할 때 발생한다.

그러나 A3의 shifted-normal tail reduction은 A0 대비 object median `-4.74%`, relevance `1/4`였다.
Clean↔shift top-5 Jaccard margin도 A0/A2 대비 `+.00125/+.00611`, mechanism gate `0/4`였다.
Verdict는 `HARD_NULL_ABSORBED`다. Full anomaly value는 유지하되 sensitivity 기반 key,
channel fraction/k tuning, trainable key projector는 닫는다. 다음 orthogonal 후보는 support positional
correspondence를 key에 주입하고 full value를 보존하는 진단이다. 상세 결과는
[report](../experiments/2026-08-08_stable_key_full_value_retrieval_stage_minus1j/report.md)에 있다.

## 23. DVT의 positional correspondence를 retrieval key로 옮길 수 있는가

동일 `(y,x)` support patch 12개 안에서만 full-feature top-5를 고르는 P1을 global P0,
half-grid rolled P2, source-permuted P3와 비교했다. Full feature를 key와 anomaly value 모두에
유지해 Stage -1j의 channel absorption과 분리했다.

P1은 P0 대비 shifted-normal tail을 object median `39.55%` 줄였고 clean↔shift exact-row
Jaccard를 `+.2356` 높였다. 그러나 P2/P3도 고정된 잘못된 위치 후보로 인해 높은 안정성을
보였으며, P1 Jaccard margin은 `-.0033/-.0275`, mechanism gate는 `0/4`였다. 즉 안정성은
correct positional correspondence가 아니라 후보 집합 제한만으로 생길 수 있었다.

Local defect retention도 `.8851` macro, fabric `.6422`로 실패했다. Broad retention `1.2144`와
대조되어 hard same-position constraint가 작은 국소 구조 변화를 과도하게 제한함을 보인다.
Verdict는 `POSITION_OVERCONSTRAINT / NO_CONTINUE_POSITIONAL_KEY`다. DVT의 position-specific
medium-scale 이득은 retrieval candidate restriction으로 이전되지 않는다. Coordinate radius,
weight, offset, k 튜닝은 열지 않는다. 상세 결과는
[report](../experiments/2026-08-08_positional_key_full_value_retrieval_stage_minus1k/report.md)에 있다.
## 15. Baseline1 성능 원인 재귀인: representation → DVT → Flow

새 shift signal 탐색에서 Baseline1 자체로 돌아가 DINOv3-H+의 성능을 두 factorial로 분해했다.

첫째, `layer7/layer31/fused × raw/Flow`에서 fused Flow의 기여는 pAUROC `+.002099`, F1
`+.002343`, AU-PRO `-.003621`로 Flow main-mechanism gate를 실패했다. Raw fused는 raw
single-layer 평균보다 강했지만 layer31 raw가 fused보다 pAUROC/F1이 높아, fusion은 ranking과
localization의 절충으로 해석된다.

둘째, fused representation의 `DVT off/on × raw/Flow`에서 DVT main effect는 pAUROC
`+.034675`, F1 `+.069455`, AU-PRO `+.009686`이었고, DVT×Flow pAUROC interaction도
`+.006278`로 gate를 통과했다. `DVT+Flow`는 동일 네 객체의 기존 Baseline1 K1과 거의
일치했다.

따라서 현재 설명 순서는 **강한 DINOv3 representation → object-aligned DVT position prior →
작은 Flow interaction**이다. DVT를 distribution-shift correction으로 부르지는 않는다. 다음
허용 진단은 inference에서 condition label을 사용하지 않는 사후 condition stratification으로,
DVT 이득이 shifted-normal false positive 감소인지 defect localization 증가인지 분리하는 것이다.

원문: [representation × Flow report](../experiments/2026-08-08_baseline1_mechanism_attribution/report.md),
[DVT × Flow report](../experiments/2026-08-08_baseline1_dvt_flow_factorial/report.md).

## 16. DVT 이득은 shifted-normal 억제인가 defect evidence 강화인가

condition token을 추론 입력으로 사용하지 않고, anomaly map 생성 이후에만 `regular / photometric /
geometric`으로 나눠 `fused+Flow`와 `DVT+Flow`를 비교했다. condition별 threshold 재최적화 없이
각 object/arm의 pooled oracle threshold를 고정 적용했다.

Defect-pixel TPR 변화는 regular `+.09616`, photometric `+.09878`, geometric `+.09577`로
condition에 거의 무관했다. 반면 good FPR 감소는 `-.00019/-.00030/-.00058`로 작았고,
엄격한 shifted-normal suppression 패턴은 `1/4` 객체에 그쳤다. shifted defect TPR은 `3/4`
객체에서 증가했지만 component recall macro는 `-.01123`으로 감소했다.

따라서 DVT는 현재 증거상 shift estimator/corrector라기보다, object-aligned position prior로서 이미
검출 가능한 결함 내부의 pixel coverage/ranking을 강화한다. Baseline1의 현재 설명은
`DINOv3 representation → DVT spatial localization prior → small Flow interaction`이다. 다음 확인은
all-8 object의 threshold-free defect ranking/coverage이며, 이 결과로 metadata-conditioned inference나
Flow Matching/Diffusion/DiT 확장을 정당화하지 않는다. 상세 결과는
[report](../experiments/2026-08-08_baseline1_dvt_condition_attribution/report.md)에 있다.

## 17. DVT localization prior는 threshold와 fabric을 넘어 일반화하는가

앞선 4-object condition attribution의 threshold/fabric 의존성을 제거하기 위해 전체 8개 객체에서
bad-image-only threshold-free pixel AP, full AUROC, top-1/5% defect coverage와 component hit를
사전등록했다. DVT on-off macro는 AP `+.010637`, full AUROC `+.008167`, top-5% coverage
`+.006349`, top-5% component hit `+.020672`였다. AP/AUROC는 각각 `7/8`, top-5% coverage는
`6/8` 객체에서 양수였고, fabric 제외 macro도 AP `+.008755`, coverage `+.005540`이었다.

따라서 verdict는 `GENERAL_LOCALIZATION_PRIOR_SUPPORTED`다. DVT 효과는 oracle threshold나 fabric
하나에 국한되지 않고 defect ranking과 moderate-area coverage에 존재한다. 다만 top-1% coverage와
component hit는 `-.005413/-.005730`이므로 peak sharpening으로 일반화하지 않는다. 이 진단은
distribution-shift correction, 새 architecture 선택 또는 benchmark contribution을 직접 확정하지 않는다.
상세 결과는 [report](../experiments/2026-08-08_baseline1_dvt_all8_rank_coverage/report.md)에 있다.

## 18. DVT 이득은 support spatial alignment와 defect scale에 의해 조절되는가

정상 support feature만으로 leave-one-out position-template explainability를 계산해 상·하위 4개
객체를 GT 이전에 고정하고, GT component를 `<=1% / 1--5% / >5%`로 나눠 기존 DVT off/on의
paired threshold-free 지표를 집계했다.

Medium `1--5%` defect는 AP `+.075706`, full AUROC `+.043168`, top-5% coverage `+.073207`,
component hit `+.012500`으로 scale gate를 통과했다. Alignment 상위 그룹의 medium AP/coverage는
`+.109076/+.108303`, 하위 그룹은 `+.008965/+.003016`으로 modulation gate도 통과했다. 반면
small defect AP는 `+.000696`에 그쳤고 top-1% coverage `-.011532`; `fruit_jelly`와
`sheet_metal` AP collapse로 scale gate를 실패했다. Large는 2개 객체뿐이라 ineligible이다.

Verdict는 `ALIGNMENT_SCALE_ATTRIBUTION_SUPPORTED`다. 다만 medium misaligned group의 eligible
객체가 2개뿐이므로 균형 잡힌 group causal effect로 과해석하지 않는다. 현재 DVT 설명은
`repeatable support spatial structure에서 medium-scale defect ranking/coverage를 강화하는
object-position prior`다. 다음 허용 단계는 positional artifact를 global-only 및 position-shuffled
negative control과 직접 비교하는 단일 causal ablation이다. 상세 결과는
[report](../experiments/2026-08-08_dvt_alignment_defect_scale_attribution/report.md)에 있다.

## 19. DVT medium 이득은 올바른 positional correspondence의 인과 효과인가

current DVT를 두 hard null과 비교했다. `position_shuffled`는 current artifact의 값과 L2 norm을
그대로 보존하되 `(H//2,W//2)` roll로 좌표 대응만 파괴했고, `global_center`는 support global mean을
모든 위치에서 동일하게 뺐다. alpha와 Flow/scorer/support는 고정했다.

Medium 1--5%에서 current-minus-shuffled AP/full AUROC/top-5 coverage는
`+.090245/+.042567/+.087314`, current-minus-global은 `+.075706/+.043168/+.073207`이었다.
6개 eligible 객체에서 모든 positional-specificity gate가 통과했고, 두 control은 raw 대비 medium
AP와 coverage의 동시 개선을 재현하지 못했다. 따라서 medium 이득은 generic centering이나 동일
크기의 arbitrary feature subtraction이 아니라 정확한 position correspondence에 의존한다.

그러나 current-minus-raw small top-1 coverage는 `-.011532`였고 객체별 collapse guard도 실패했다.
Verdict는 `POSITION_SPECIFIC_MEDIUM_GAIN_WITH_SMALL_DEFECT_COST`; mechanism 귀인은 강화됐지만
unconditional DVT/alpha/gating/fusion tuning은 종료한다. 이 결과만으로 Flow Matching, Diffusion,
DiT를 열지 않는다. 상세 결과는
[report](../experiments/2026-08-08_dvt_positional_causal_controls/report.md)에 있다.

## 20. Small-defect 손실은 DVT와 Flow 중 어디서 생기는가

동일한 all-8 query에 대해 `pre-DVT feature -> post-DVT feature -> Flow latent 1-NN ->
deployed final`을 한 실행 안에서 순서대로 측정했다. Raw/current의 pre-DVT 차이는 모든 scale에서
정확히 `0.0`이었고, 기존 final 결과도 tolerance `1e-6` 안에서 재현됐다.

Small defect top-1% coverage는 post-DVT에서 처음 `-.011667`로 하락했고 4개 객체가 음수여서
사전등록 onset gate를 통과했다. Flow latent와 deployed final은 모두 `-.011532`로 같았다. 반대로
medium AP/top-5 coverage는 post-DVT에서 이미 `+.074117/+.074640`이었고 final은
`+.075706/+.073207`이었다.

Verdict는 `LOSS_AT_DVT_SUBTRACTION`이다. Small-defect 비용과 medium 이득 모두 DVT에서 먼저
생기며, Flow는 이 scale tradeoff를 만들거나 복구하지 않고 소폭 조정한다. 따라서 다음 구조 탐색은
Flow 교체보다 DVT 이전 representation 또는 defect-preserving positional prior를 우선해야 한다.
상세 결과는 [report](../experiments/2026-08-08_dvt_small_defect_signal_path_attribution/report.md)에 있다.

## 21. DINO에서 shift-invariant channel만 읽으면 안전한 representation이 되는가

Normal support의 paired photometric transform으로 DINOv3-H+ channel sensitivity를 fold-local하게
계산하고, 민감도가 낮은 하위 50% channel만 남기는 고정 readout을 identity 및 equal-cardinality
score-shuffled null과 비교했다. Test/GT/condition metadata, DVT, Flow, query adaptation은 사용하지 않았다.

선택 readout의 held compound-shift displacement ratio는 object median `0.607816`으로 shuffled null
`1.005177`보다 크게 낮았고 invariance는 `4/4` 객체에서 통과했다. 그러나 local/broad defect contrast
retention은 `0.494080/0.594854`, clean NN identity와 pair Spearman은 `0.671875/0.803852`였다.
Raw 1-NN relevance는 `1/4`만 통과했고, q995 reduction은 `0.170364`였지만 shuffled-null 대비 margin은
`-.013114`였다.

Verdict는 `ANOMALY_ABSORPTION`이다. DINO 안에 photometric shift-sensitive channel subset은 실제로
존재하지만, 그 축은 defect/structure 정보와 분리돼 있지 않다. 따라서 단순 channel rejection과 이를
초기화로 삼는 trainable diagonal gate는 열지 않는다. 다음 후보는 shift sensitivity 최소화 단독이 아니라
**defect-preserving constraint가 구조적으로 포함된 readout**이어야 한다. 상세 결과는
[report](../experiments/2026-08-08_dino_invariant_channel_readout_stage_minus1/report.md)에 있다.

## 24. DVT 기여도는 1-NN outlier에 의존하는가

동일한 4-object `DVT off/on × identity/Flow` factorial에서 K1과 K5를 같은 feature/Flow 평가로
동시에 산출했다. K5에서도 DVT main effect는 pAUROC `+.030991`, F1 `+.069351`로 K1의
`+.034675/+.069455`를 대부분 유지했고, DVT×Flow pAUROC interaction도 `+.006424`로 유지됐다.

다만 DVT-positive object count가 K1 `3/4`에서 K5 `2/4`로 줄어 엄격한 verdict는
`SCORER_DEPENDENT_DVT_ATTRIBUTION`이다. 실패 원인은 효과 소실이 아니라 `fruit_jelly`가
`+.006656 → -.000676`으로 경계선을 넘은 것이다. AU-PRO main effect도 `+.009686 → +.000111`로
감소했다. 따라서 DVT의 큰 pAUROC/F1 효과는 1-NN outlier artifact가 아니지만, 객체별 일반성과
localization 품질은 scorer-independent하지 않다. K1 default는 유지하며 k sweep이나 Mahalanobis로
확장하지 않는다. 다음 구조 탐색은 small-defect 비용과 AU-PRO 민감성을 함께 다루는 scale-selective
positional prior가 우선이다. 상세 결과는
[report](../experiments/2026-08-08_baseline1_dvt_representation_scorer_factorial/report.md)에 있다.

## 25. DVT의 spatial-frequency tradeoff를 분리할 수 있는가

Current positional artifact에 고정 Gaussian low-pass(`sigma=2`, `truncate=3`)를 적용한 DVT를
동일 energy의 half-grid rolled hard null과 비교했다. Low-pass DVT는 raw 대비 small top-1
coverage를 current `-.011532`에서 `+.001132`로 회복했지만, medium AP/top-5 coverage 이득은
current의 `67.48%/71.71%`만 보존해 strict 75% gate를 실패했다.

그러나 correct-position low-pass는 shuffled null보다 medium AP/top-5 coverage가
`+.056823/+.063081` 높았다. 즉 medium 이득에는 실제 low-frequency positional component가 있지만,
단순 low-pass 교체는 medium signal을 과도하게 손실한다. Verdict는
`KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`. 다음 허용 분기는 sigma/alpha sweep가 아니라,
low-frequency positional correction과 high-frequency defect residual을 구조적으로 분리하는 단일
사전등록 실험이다. 상세 결과는
[report](../experiments/2026-08-08_scale_selective_residual_dvt_diagnostic/report.md)에 있다.
