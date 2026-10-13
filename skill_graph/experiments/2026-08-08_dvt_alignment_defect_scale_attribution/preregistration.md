# DVT Alignment × Defect-Scale Attribution 사전등록

Date: 2026-08-08
Protocol: `dvt-alignment-defect-scale-attribution-all8-20260808`
Stage: mechanism diagnostic; no method promotion

## 1. Motivation and negative-evidence intake

All-8 threshold-free 검증에서 DVT는 bad-image pixel AP/full AUROC/top-5% defect coverage를
개선했지만, top-1% coverage는 감소했고 `fruit_jelly`, `walnuts`, `sheet_metal`에 서로 다른
실패가 남았다. 이번 질문은 DVT를 새로 튜닝하는 것이 아니라, 이 object-position prior의 효과가
정상 support의 공간 정렬도와 결함 component 크기에 의해 설명되는지 검증하는 것이다.

이는 폐기된 query registration, alignment loss, condition estimation 또는 score-fusion branch의
재시도가 아니다. Score map을 바꾸지 않고 기존 DVT off/on의 paired per-image 결과를 사후
귀인한다. 알려진 failure basin은 (1) defect가 alignment 통계를 오염하는 것, (2) image-level
평균이 작은 component 실패를 숨기는 것, (3) top-area float16 tie inflation이다.

## 2. Target and fixed method contract

- `target_dataset=MVTec AD2 single-image`
- local data root: `/home/hun/Volume/DATA/mvtec_ad_2`
- dsba5 data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- objects: `can, fabric, fruit_jelly, rice, vial, wallplugs, walnuts, sheet_metal`
- population: full `test_public/good,bad`
- support: fixed SuperAD-selected 16 normal images/object
- backbone: DINOv3-H+ layers `[7,15,23,31]`, resolution 672; `sheet_metal` 448
- common scorer: Flow 3 epochs, raw latent 1-NN, density weight `0`
- paired arms: `fused_flow` DVT off vs `dvt_flow` position-mean DVT, alpha `1.0`
- primary metric family: bad-image pixel AP; secondary full AUROC, top-area coverage/component hit
- existing same-condition baseline source: the two arms above from
  `results/baseline1_dvt_all8_rank_coverage_20260808/local/`; this run must reproduce them.
- SuperAD/RN-FMLK fields: `N/A`, `comparable=false`. This is a Baseline1 mechanism diagnostic,
  not a new method or Table-1 superiority claim.

## 3. Information boundary

1. Alignment is computed from the 16 normal support feature maps only, before test GT is opened.
2. Filename, condition suffix, split suffix, metadata and GT never enter feature extraction, Flow, DVT,
   anomaly scoring or alignment grouping.
3. Test GT is opened only after score maps exist, to form defect-scale strata and paired endpoints.
4. Image stem is used only as an arm-pairing identifier.
5. No per-object threshold, group boundary, scale boundary or metric is tuned after results.

## 4. Support-only spatial alignment score

For object (o), support feature maps are (X_i\in\mathbb{R}^{H\times W\times D}), (i=1,\dots,16).
For each leave-one-out fold, compute position mean (\mu_{-i,p}) and global mean (g_{-i}). Define

\[
r_i = 1 - \frac{\operatorname{mean}_p\|X_{i,p}-\mu_{-i,p}\|_2^2}
                    {\operatorname{mean}_p\|X_{i,p}-g_{-i}\|_2^2 + 10^{-12}}.
\]

The object alignment score is (A_o=\operatorname{median}_i r_i). Higher means a position-specific
normal template explains held-out normal support better than a position-agnostic template. This is
feature-based, support-only and directly tied to the DVT artifact `position_mean - global_mean`.

Before any GT/result aggregation, rank the eight objects by (A_o): top four are `support_aligned`,
bottom four `support_misaligned`; exact ties use alphabetical object order. Record all fold values,
group membership and support basenames.

## 5. Defect-scale strata and paired endpoints

Connected components use 8-connectivity on native GT. Component area fraction is component pixels
divided by full native image pixels:

- `small`: `area_fraction <= 0.01`
- `medium`: `0.01 < area_fraction <= 0.05`
- `large`: `area_fraction > 0.05`

The boundaries are fixed from the already used top-1%/5% area budgets. For each bad image and scale,
the scale's component pixels are positives, background pixels are negatives, and pixels belonging to
other defect scales are excluded from AP/AUROC. Images without a component in that scale are ineligible.

Per eligible image/arm/scale record:

- pixel AP and full AUROC;
- defect-pixel coverage at tie-inclusive top 1% and 5% of the full image score;
- component hit rate at top 1% and 5%;
- eligible positive/background pixels, component count, nominal and selected area counts, boundary ties.

Compute paired `DVT on - off` image deltas, then image mean within object, then equal-weight object macro.

## 6. Validity and replication gates

- identical support-16 basenames across arms/environments;
- full object/image/GT coverage, native shape match, finite scores;
- arm-invariant GT/component counts exact;
- no gated endpoint whose tie-inclusive selected area exceeds `2x` nominal area;
- local/dsba5 continuous metric parity absolute delta `<=1e-5` and identical verdict/gates;
- reproduce prior all-object gates: macro AP/AUROC positive with at least `6/8` positive objects,
  top-5% coverage macro positive, no object AP `<-0.03` or coverage `<-0.05`.

## 7. Attribution gates

A scale bin is eligible when at least four objects contain eligible images/components.

`SCALE_GAIN_b`:

- object-macro `delta_AP_b > 0`;
- positive eligible objects `>= ceil(0.75 * eligible_object_count_b)`;
- `small`: macro `delta_coverage_top1 > 0`; `medium/large`: macro `delta_coverage_top5 > 0`;
- matching component-hit delta `>= -0.01`;
- no eligible object-bin `delta_AP < -0.03`.

`ALIGNMENT_MODULATION_b`:

- aligned-minus-misaligned group macro `delta_AP_b >= 0.005`;
- aligned group macro `delta_AP_b > 0`;
- at least `3/4` aligned objects positive when all four are eligible, otherwise at least 75%;
- aligned-minus-misaligned difference for the scale-matched coverage endpoint is positive.

## 8. Frozen verdicts

- `ALIGNMENT_SCALE_ATTRIBUTION_SUPPORTED`: replication passes and at least one bin passes both gates.
- `SCALE_ONLY_ATTRIBUTION`: replication passes, at least one scale gate passes, no alignment gate passes.
- `ALIGNMENT_ONLY_ATTRIBUTION`: replication passes, pooled-all-scale alignment gate passes, no scale gate.
- `GENERAL_DVT_EFFECT_NO_ALIGNMENT_SCALE_ATTRIBUTION`: replication passes, neither attribution passes.
- `DVT_REPLICATION_FAILED`: validity holds but prior general effect does not reproduce.
- `INCONCLUSIVE_UNDERPOWERED`: validity fails from scale/group eligibility or tie-area gate.

No verdict is a method `KEEP`. A positive attribution permits one separately preregistered
position-vs-global/shuffled-position causal ablation. Otherwise centering ablation, gating, fusion and
architecture expansion remain blocked.

## 9. Execution and custody

- Implement compact per-image/scale diagnostics and support alignment audit with focused regression tests.
- Run local RTX 4090 and dsba5 fixed container/GPU3 in parallel after preflight.
- Preserve JSON/TSV/log/report only; remove all regenerable `anomaly_maps/` after conclusion.

