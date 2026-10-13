# Nuisance-Orbit Anomaly Persistence Design Audit

Date: 2026-08-08

Verdict: `DESIGN_KILL_CLOSED_BRANCH`

## 1. Motivation

최근 문헌 검토 이후 제안된 `nuisance-orbit anomaly persistence`는 query image에
제한된 photometric/geometric nuisance transform family를 적용하고, inverse-aligned score
maps에서 persistent anomaly evidence만 유지하는 방향이었다. 의도는 정상 조명/색조 shift는
view마다 흔들리고 defect-induced shift는 유지된다는 inductive bias를 이용하는 것이다.

하지만 FMAD negative-evidence intake 기준에서 이 후보는 새 실험으로 승격할 수 없다.
이미 폐기된 `Query Counterfactual Component Persistence`와 동일한 핵심 조작을 사용한다.

- 후보 조작: multiple reversible query views, inverse alignment, persistent component retention.
- 선행 폐기 근거:
  [stage-02/method.md](../../../.omx/specs/deep-research-autoresearch-shift-defect-identifiability/stage-02/method.md),
  [stage-02/evaluator.json](../../../.omx/specs/deep-research-autoresearch-shift-defect-identifiability/stage-02/evaluator.json).
- evaluator verdict: `reject`, score `0.32`.
- top blockers: multiple counterfactual query transforms, post-hoc score fusion/map shaping,
  persistent nuisance와 transform-sensitive defect의 causal ambiguity.

따라서 이 문서는 실행 보고서가 아니라 설계 단계 negative-evidence intake 및 closed-branch
기록이다.

## 2. Implementable Design

실행 가능한 최소 형태는 다음과 같았을 것이다.

- frozen Baseline1 score map을 identity 및 제한된 nuisance orbit에서 계산한다.
- 각 transformed query score map을 원 좌표계로 되돌린다.
- patch/component 단위로 median, quantile, persistence ratio를 집계한다.
- shifted-good tail 감소와 bad-pixel recall retention을 gate로 둔다.

그러나 이 설계는 구현 전 폐기한다. 이유는 method mechanism이 새 evidence source를 추가하지
않고, 같은 query의 score maps를 변환별로 재가공하는 post-hoc fusion이기 때문이다. 이는
FMAD protocol의 closed-branch retread 조건에 해당한다.

SuperAD baseline source, RN-FMLK hard-null, segmentation metric schema는 정의하지 않는다.
동일 조건 benchmark로 승격하지 않으므로 `BLOCKED_BASELINE`이 아니라
`DESIGN_KILL_CLOSED_BRANCH`이다.

## 3. Evaluation Alignment

이 설계는 motivating claim과 정렬되지 않는다.

- Claim: normal distribution shift와 defect-induced shift를 구분하는 target-specific evidence가
  필요하다.
- Proposed evidence: query를 여러 nuisance transform으로 본 뒤 score-map persistence를 선택한다.
- Alignment failure: persistence는 anomaly-aligned evidence가 아니라 transform family와 score
  fusion rule에 의존하는 detector-sensitivity statistic이다.

특히 이미 기록된 Stage 2 evaluator는 동일한 구조를 다음 이유로 거절했다.

```text
evaluation_rate=0.32
verdict=reject
deployability=0.12
identifiability=0.25
confound_resistance=0.18
```

`strongest_surviving_claim`도 "offline detector-sensitivity diagnostic"에 한정되어 있다. 이는
method path가 아니라 보조 분석으로만 남길 수 있다는 뜻이다.

## 4. Code Modification / Creation

코드 수정은 수행하지 않았다.

변경 파일은 문서화 파일로 제한했다.

- `skill_graph/experiments/2026-08-08_nuisance_orbit_anomaly_persistence_design_audit/report.md`
- `tasks/todo.md`
- `skill_graph/log.md`
- `skill_graph/index.md`

GPU 실행, remote container 조작, benchmark run, result pullback은 수행하지 않았다.

## 5. Added Code Evaluation

추가 코드가 없으므로 unit test, py_compile, lsp diagnostics는 적용 대상이 아니다. 대신 문서
검증 대상으로 다음을 확인한다.

- relative link target이 존재하는지
- 보고서가 protocol 9-stage skeleton을 보존하는지
- verdict string이 `DESIGN_KILL_CLOSED_BRANCH`로 일관되는지
- git diff가 지정된 문서 파일에만 국한되는지

## 6. Remote Experiment Execution

실행 금지.

이 branch는 closed-branch retread로 판정되었으므로 dsba/local GPU 실행을 시작하지 않는다.
MVTec AD2 public/test 접근, remote snapshot 생성, anomaly map 생성도 없다.

## 7. Evaluation Results and Analysis

새 metric result는 없다. 판정은 기존 negative evidence intake와 선행 실험 결과를 재사용한다.

### 7.1 Query-view persistence retread

`.omx/specs/deep-research-autoresearch-shift-defect-identifiability/stage-02`의
`Query Counterfactual Component Persistence`는 정확히 다음 메커니즘을 제안했다.

- multiple reversible query views 생성
- score map inverse alignment
- persistent components 보존
- identity, average, max, alignment shuffle, component-ID shuffle nulls

평가 결과는 `reject`이고 score는 `0.32`였다. 핵심 차단 사유는 query transform을 여러 번
사용하는 inference-time post-hoc shaping이며, transform-sensitive defect와 persistent nuisance를
구분할 causal 근거가 없다는 점이다.

이번 `nuisance-orbit anomaly persistence`는 aggregation statistic을 `min` 대신
median/quantile/persistence로 바꾸더라도 같은 failure basin에 남는다. aggregation rule 변경은
closed branch의 threshold/score-fusion retune이며 새 anomaly-aligned evidence source가 아니다.

### 7.2 Support transformed-bank union/patchwise-min precedent

`2026-07-18_default_shift_sequence` Stage 2는 transformed support diversity의 coverage 효과를
이미 측정했다.

- union과 patchwise-min은 nearest-neighbor minimization over concatenated bank 때문에 정확히 동일했다.
- pAUROC gain: `+0.001790`
- shifted-good tail reduction: `8.53%`
- F1 delta: `-0.000049`
- effect concentration: `can` 중심, 다른 object는 약 `+0.00133` 이내
- routed single bank verdict: `NO_CONTINUE`

이는 nuisance orbit이 작은 coverage benefit을 줄 수는 있어도, robust distribution-shift solution
또는 identifiable acquisition-state recovery가 아님을 보여준다.

### 7.3 Rotation-8 support augmentation precedent

`2026-07-10_flowtte_hplus_dvt_superad_rotation8`은 exact SuperAD rotation-8 support augmentation을
동일 support/source path와 evaluator 조건에서 테스트했다.

- identity: `0.836389/0.527370`
- rotation-8: `0.819251/0.506201`
- delta: `-0.017137` AUROC, `-0.021168` F1
- verdict: `KILL_FOR_CLAIM / NO_CONTINUE`

이는 transform family를 늘리는 것이 자동으로 nuisance robustness를 주지 않으며, 오히려 object별
score geometry와 DVT/flow calibration을 손상할 수 있음을 보여준다.

### 7.4 Literature interpretation boundary

문헌은 이 설계의 위험을 보조적으로 설명하지만 직접적인 폐기 증명은 아니다.

- `What Should Not Be Contrastive`는 task-relevant factor까지 invariant하게 만들면 defect evidence를
  제거할 수 있음을 경고한다.
- `Augerino`는 transformation family 및 invariance extent를 제한해야 함을 시사하지만, 잘못 지정된
  transform family가 anomaly preservation을 보장한다고 말하지 않는다.

따라서 문헌상 근거는 "오지정된 transformation family와 hard invariance가 위험하다"는 배경 설명이며,
이번 폐기의 직접 근거는 repo-local negative evidence와 동일 failure basin 판정이다.

## 8. Conclusion

Verdict: `DESIGN_KILL_CLOSED_BRANCH`

새 실험을 진행하지 않는다. `nuisance-orbit anomaly persistence`는 다음 이유로 method branch로
승격 금지다.

1. 이미 거절된 query-view component persistence와 동일한 query-transform/post-hoc score-shaping
   구조다.
2. support transformed-bank union/patchwise-min은 작은 pAUROC/tail 신호를 보였지만 F1과 object
   heterogeneity gate를 넘지 못했고 `NO_CONTINUE`였다.
3. rotation-8 support augmentation은 동일 조건에서 AUROC/F1을 모두 악화했다.
4. aggregation을 `min`에서 median/quantile/persistence로 바꾸는 것은 새 evidence source가 아니라
   score-fusion retune이다.

비중복 다음 방향은 실험이 아니라 `external auxiliary variable/evidence source feasibility audit`이다.
즉, support/query score 재가공이 아니라 nuisance와 defect 분해를 식별하게 해 줄 외부 변수, 센서,
metadata, paired acquisition, synthetic-control provenance, 또는 frozen external prior가 실제로
가용한지부터 감사해야 한다. 이 audit 전에는 Flow Matching, Diffusion, DiT, 추가 orbit sweep으로
넘어가지 않는다.

## 9. Post-Conclusion Storage Cleanup

이번 branch는 설계 단계에서 폐기되었고 실행 산출물이 없다.

- remote result root: 없음
- local pullback root: 없음
- anomaly_maps cleanup 대상: 없음
- cleanup verification: 새 dense artifact를 생성하지 않았음
