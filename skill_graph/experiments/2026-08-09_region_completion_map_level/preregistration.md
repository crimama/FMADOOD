# Region-Completion Map-Level Combination Preregistration

Date: 2026-08-09

Protocol ID: `region-completion-map-level-all8-20260809`

Tier: mechanism diagnostic with a conditional promotion gate; not yet a
SuperAD-comparable method claim

## Negative Evidence Intake

Two facts define this protocol.

First, the official private anchor moved the leading deficit. Private AU-PRO@0.05
is `0.5440` against RoBiS `0.6725`, a gap of `-0.1285` that is now larger than
the Seg-F1 gap against any row. Public `gt_component_recall_at_oracle` is
`0.484` macro (`0.534` excluding `can`): at the best possible threshold, more
than half of ground-truth defect components are never touched. There is no
region-completion mechanism anywhere in Baseline1-core, while RoBiS completes
regions with SAM refinement.

Second, the two-band protected-residual branch failed with
`ANOMALY_ABSORPTION / NO_CONTINUE`. Its failure was diagnosed precisely: the
`2C` concatenated representation was consumed by the Flow, which reweighted and
absorbed the small-defect signal. The failure was located in the
**feature-space** combination and in Flow retraining, not in the idea that
current-DVT and low-pass-DVT carry complementary evidence.

That complementarity is measured and null-controlled. All-eight, versus raw:

| Operator | pAUROC | F1 | AU-PRO | small top-1 | small AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| current DVT | +.042530 | +.062212 | +.005437 | -.011532 | +.000696 |
| low-pass DVT | +.034767 | +.051028 | **+.010513** | **+.001132** | +.009410 |

Low-pass repairs `sheet_metal` AU-PRO from `-.06559` to `-.00274` and
`fruit_jelly` from `-.00641` to `-.00073`, and beats an equal-energy
shuffled-position null by `+.056823/+.063081` on medium AP/top-5 coverage.

## Explicit Relationship to the Closed Two-Band Prohibition

The two-band continuation assessment forbids tuning band weights, sigma,
dimensional scaling, Flow capacity, score fusion, and object-specific policy.
That prohibition is a bar on post-hoc rescue of the two-band branch and is
respected: nothing below reopens the `2C` representation, the band split, the
sigma, or Flow.

This protocol is a different object. Two independently complete, dimension-
unchanged pipelines are run to completion and their **output maps** are
combined by a **parameter-free** rule with a spatial hard null. No weight, no
sigma, no scale, no per-object policy exists to tune, because the combination
rule has no free parameter.

The risk this creates is honestly stated: if the combination gains come merely
from taking a pointwise maximum of two correlated maps, the shuffled null will
absorb the effect. If the null absorbs it, or if the promotion gate fails, the
entire "combine current and low-pass DVT" family is closed permanently and no
weighted, gated, or per-object variant may be proposed afterwards.

## Dataset and Frozen Baseline Contract

- `target_dataset=MVTec AD2 single-image`
- local data root: `/home/hun/Volume/DATA/mvtec_ad_2`
- remote data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- objects: `can,fabric,fruit_jelly,rice,sheet_metal,vial,wallplugs,walnuts`
- population: full `test_public/good,bad`
- support: exact frozen DINOv3-H+ Baseline1-core 16 normal images, manifest
  `skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json`,
  SHA-256 `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`
  — the manifest used by the official private submission and by the
  `raw`/`position_current`/`position_lowpass` anchors
- backbone: frozen DINOv3-H+ `[7,15,23,31]`, `layer_norm_mean`, resolution 672
  except `sheet_metal=448`
- Flow/scorer: existing Baseline1 Phase-3 contract, 3 epochs, density weight 0,
  `raw_1nn`, `rgb-guide none`, `threshold-calibration none`,
  `binary-postprocess none`
- forbidden: query-time update, memory expansion, condition/filename/split
  metadata, `test_private`, `test_private_mixed`
- SuperAD/SuperADD fields are `N/A` and `comparable=false` in this stage.

## Fixed Combination Rule

Let `M_cur` be the deployed score map produced with `position_mean` DVT and
`M_lp` the deployed score map produced with the frozen `position_lowpass` DVT
(`sigma=2.0`, `truncate=3.0`, reflect padding — reused unchanged, not retuned).

Define per-image rank normalization `R(M)` mapping each pixel to its
within-image ascending rank divided by the pixel count, with ties receiving the
mean rank. `R` has no free parameter and is applied identically to every arm.

Combination arms operate only on `R(M_cur)` and `R(M_lp)`. No weights exist.

## Arms

| Arm | Definition | Purpose |
| --- | --- | --- |
| `D0_rank_current` | `R(M_cur)` | rank-normalization control: isolates any gain caused by `R` alone |
| `D1_rank_max` | `max(R(M_cur), R(M_lp))` | candidate: recover components missed by either branch |
| `D2_rank_mean` | `mean(R(M_cur), R(M_lp))` | candidate: consensus rather than union |
| `D3_rank_max_shuffled` | `max(R(M_cur), roll(R(M_lp), H//2, W//2))` | spatial hard null: identical marginals, destroyed correspondence |

Frozen external anchors, regenerated under identical custody: `raw`,
`position_current`, `position_lowpass`. No further arm, rule, or parameter may
be added.

## Primary Metrics

- Primary: macro `seg_AUPRO_0.05` and macro `gt_component_recall_at_oracle`.
- Co-primary no-harm: macro `seg_AUROC_0.05` and macro `seg_F1` (raw oracle).
- Secondary: per-object versions of all four, `pooled_pixel_ap`,
  `normal_image_pooled_fpr_at_oracle`, and small/medium scale-binned AP and
  coverage using the existing component-scale evaluator.
- Original-resolution float16 tie-inclusive evaluator throughout.

## Validity Gates

All must pass before any endpoint is read:

1. all eight objects complete in every arm; full map/GT coverage; finite,
   nonconstant maps; no retained `anomaly_maps/` after conclusion;
2. exact support manifest, preprocessing, resolution policy, Flow config,
   scorer, and evaluator custody identical across all arms, asserted from the
   emitted run config;
3. `R` is verified rank-correct: `D0_rank_current` reproduces
   `position_current` `seg_AUROC_0.05` and `seg_AUPRO_0.05` to within `1e-6`,
   since rank normalization is strictly monotone per image;
4. `D3` roll offsets are nonzero and `D3` marginal distribution per image is
   identical to `D1`'s `R(M_lp)` marginal within `1e-6`;
5. no query update, no metadata/condition input, no private access;
6. duplicated local/dsba5 arm parity: maximum absolute difference of shared
   continuous metrics `<= 1e-3`.

Gate 3 is a genuine falsifier: if rank normalization is implemented
incorrectly it will not reproduce the monotone-invariant metrics, and the
protocol reports `INVALID`.

## Hard-Null Gate

Both must pass, else `HARD_NULL_ABSORBED` and the family closes:

1. `D1 - D3` macro `seg_AUPRO_0.05` `>= +0.010`;
2. `D1 - D3` macro `gt_component_recall_at_oracle` `>= +0.020`.

## Strict Promotion Gate

Let `C` be the better of `D1` and `D2` chosen by macro `seg_AUPRO_0.05`. All
must pass:

1. `C - position_current` macro `seg_AUPRO_0.05` `>= +0.015`;
2. `C - position_current` macro `gt_component_recall_at_oracle` `>= +0.030`;
3. `C - position_current` macro `seg_AUROC_0.05` `>= -0.005`;
4. `C - position_current` macro `seg_F1` `>= -0.005`;
5. no object loses more than `0.05` `seg_F1` or `0.05` `seg_AUPRO_0.05`
   against `position_current`;
6. `C - D0` macro `seg_AUPRO_0.05` `>= +0.010`, proving the gain is not from
   rank normalization alone.

## Continuation Gate

If the strict gate fails but all of the following hold, the verdict is
`CONTINUE_DIAGNOSTIC` and exactly one preregistered follow-up is licensed:

1. hard-null gate passes;
2. `C - position_current` macro `seg_AUPRO_0.05` `>= +0.005`;
3. `C - position_current` macro `gt_component_recall_at_oracle` `>= +0.015`;
4. no catastrophic object loss as defined in strict gate item 5.

Otherwise the verdict is `KILL_FOR_CLAIM / NO_CONTINUE` and the family closes.

## Anti-Overfitting and Stop Conditions

- No weight, sigma, scale, gate, threshold, k, or per-object policy may be
  introduced or swept. The combination rule is parameter-free by construction.
- The choice between `D1` and `D2` is made by the single preregistered
  criterion above and by nothing else.
- A promoted result is still not a SuperAD-comparable claim. Promotion only
  licenses running the paired public evaluator under the Main Table contract as
  a separate, subsequently preregistered step.
- Rank normalization removes the absolute score scale used by the calibrated
  threshold path. If the strict gate passes, the report must state that a
  deployable threshold rule for the combined map does not yet exist and is out
  of scope here.

## Execution and Cleanup

- Terra implements `R`, the four arms, the aggregator with all gates, and
  focused regression tests. No new scoring mathematics beyond `R` and the three
  fixed combination expressions.
- Anchor and `D0`/`D1`/`D2` runs on the local RTX 4090 container
  `hun_fsad_tta_local_stage1a0`.
- `D3` hard null runs on dsba5 container `hun_fsad_tta_012`, host GPU 3,
  container CUDA slot 0, via an immutable snapshot.
- One arm is deliberately duplicated across local and dsba5 for parity.
- Preserve compact JSON/log/summary/parity artifacts only; verify absence of
  `anomaly_maps/` after the conclusion.

## Reporting

The report must state the verdict, all four gate blocks with their numbers, the
per-object table, and an explicit statement of what the verdict licenses. If
the hard null absorbs the effect or the continuation gate fails, the report
must record that the current-plus-low-pass combination family is closed.
