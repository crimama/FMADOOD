# DVT Positional Causal Controls 사전등록

Date: 2026-08-08
Protocol: `dvt-positional-causal-controls-all8-20260808`
Stage: bounded mechanism ablation; diagnostic only

## 1. Negative evidence intake and motivation

직전 attribution에서 DVT의 medium-scale AP/top-5% coverage가 `+.07571/+.07321`이었고,
support positional explainability가 높은 eligible group에서 효과가 컸다. 그러나 이는 상관적
귀인이며 position-specific artifact가 실제 원인인지, 일반 centering 또는 동일 크기의 임의 spatial
field도 같은 효과를 내는지는 분리하지 못했다.

이번 실험은 새 module, gating, score fusion, alpha tuning 또는 killed registration branch가 아니다.
현재 DVT operator의 원인을 두 고정 hard null과 비교하는 마지막 단일-step causal diagnostic이다.
알려진 위험은 passive feature surgery가 Flow에 흡수되는 것, shuffled field가 current를 따라잡는 것,
medium 이득이 small-defect 손상을 감추는 것이다.

## 2. Target and common contract

- `target_dataset=MVTec AD2 single-image`
- local/remote data roots: `/home/hun/Volume/DATA/mvtec_ad_2`,
  `/home/hunim/Volume/DATA/mvtec_ad_2`
- all 8 objects, full `test_public/good,bad`
- fixed SuperAD-selected support-16
- DINOv3-H+ layers `[7,15,23,31]`, 672; `sheet_metal` 448
- common Flow 3 epochs, raw latent 1-NN, density weight `0`
- identical evaluator and per-image component scale contract: small `<=1%`, medium `1--5%`, large `>5%`
- metadata/filename/condition signals forbidden; GT post-scoring evaluation only
- SuperAD/RN-FMLK: `N/A`, `comparable=false`; no method or Table-1 claim

## 3. Frozen arms

All transforms are fitted from the same 16 normal support feature maps and applied identically to support/query
before Flow fitting/scoring.

1. `raw`: no subtraction.
2. `position_current`: subtract (a_p=\mu_p-g), the deployed DVT artifact.
3. `position_shuffled`: subtract `roll(a, H//2, axis=0)` then `roll(..., W//2, axis=1)`.
   This preserves artifact values, norms and marginal distribution while destroying position correspondence.
4. `global_center`: subtract spatially constant support global feature mean (g).

The roll is deterministic, has no seed/search, and must move at least one axis for any valid feature grid.
Alpha is fixed to `1.0` in every subtractive arm. No alternative shuffle, magnitude matching or normalization is allowed.

Existing complete `raw` and `position_current` artifacts from
`results/dvt_alignment_defect_scale_attribution_20260808/` are reused after exact support/config/custody checks.
Only `position_shuffled` and `global_center` are newly executed.

## 4. Primary comparisons and aggregation

For each eligible bad image/scale, compare per-image AP, full AUROC, tie-inclusive coverage and component hit.
Aggregate paired image delta within object, then equal-weight object macro.

Primary causal contrasts for medium defects:

- `current_minus_shuffled`
- `current_minus_global`

Secondary contrasts:

- each transformed arm minus raw for medium and small defects;
- all-defect threshold-free AP/AUROC/top-5 coverage;
- standard `seg_AUROC_0.05`, `seg_F1`, `seg_AUPRO_0.05` custody metrics.

## 5. Validity gates

- all 8 objects and exact support basenames/config contract across four arms;
- raw/current replication equals the previous attribution verdict and aggregates within `1e-8` locally;
- native GT/prediction coverage, component/pixel custody and finite scores pass;
- no gated top-area endpoint exceeds `2x` nominal area due ties;
- local/dsba5 aggregate metric parity `<=1e-5`, identical boolean gates/verdict;
- position-shuffled artifact L2 equals current artifact L2 within `1e-6` relative tolerance;
- global-center artifact is spatially constant by construction audit.

Any failure yields `INVALID_CAUSAL_ABLATION`.

## 6. Positional specificity gate

Medium bin must remain eligible with at least four objects. `POSITION_SPECIFICITY` passes only when:

- macro `current_minus_shuffled.pixel_ap >= 0.01`;
- macro `current_minus_global.pixel_ap >= 0.01`;
- macro `current_minus_shuffled.coverage_top5 >= 0.01`;
- macro `current_minus_global.coverage_top5 >= 0.01`;
- current beats each control in AP on at least 75% of mutually eligible objects;
- current medium component-hit@5% is not worse than either control by more than `.01`;
- no mutually eligible object has current-minus-control AP `<-0.03`.

This gate attributes the medium benefit to correct spatial correspondence rather than arbitrary feature subtraction.

## 7. Small-defect safety gate

Against raw, current DVT must satisfy all of:

- small macro AP `>=0`;
- small top-1% coverage `>=0`;
- small component-hit@1% `>=-.01`;
- no object small AP `<-.03`;
- no object small top-1% coverage `<-.05`.

This gate is intentionally retained even though the prior diagnostic suggests failure; it prevents a medium-scale
mechanism result from being promoted as a generally safe operator.

## 8. Frozen verdicts

- `POSITION_SPECIFIC_SCALE_SAFE`: validity, specificity and small safety all pass.
- `POSITION_SPECIFIC_MEDIUM_GAIN_WITH_SMALL_DEFECT_COST`: validity/specificity pass, safety fails.
- `GENERIC_CENTERING_OR_TRANSFORM_EFFECT`: validity passes, specificity fails, and at least one control improves
  medium AP and coverage over raw.
- `NO_CAUSAL_POSITION_SUPPORT`: validity passes, specificity fails, and neither control/current provides a
  distinct positive mechanism pattern.
- `INVALID_CAUSAL_ABLATION`: any custody, replication, artifact or parity gate fails.

No verdict is `KEEP`. Only `POSITION_SPECIFIC_SCALE_SAFE` would permit a separately designed deployment candidate.
`...WITH_SMALL_DEFECT_COST` closes unconditional DVT/gating/fusion tuning and retains only the mechanism statement.

## 9. Execution and cleanup

- Terra implements two config-driven denoiser controls and a four-arm compact aggregator with regression tests.
- Run new arms in parallel on local RTX 4090 and dsba5 fixed container GPU3.
- Preserve compact JSON/log/summary/parity only; remove/verify absence of `anomaly_maps/` after conclusion.

