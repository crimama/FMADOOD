# Normal-Tail Structure and Predictability Preregistration

Date: 2026-08-09

Protocol ID: `normal-tail-structure-predictability-all8-20260809`

Tier: measurement-only mechanism diagnostic. **No intervention is licensed by
this protocol.** No method claim, no SuperAD comparison.

## Negative Evidence Intake

Eleven module-level branches have been closed. Every one of them died at an
intervention gate — anomaly absorption, hard-null absorption, or catastrophic
object loss — after the implementation cost had already been paid. The
recurring error was proposing a transform before establishing that the thing it
targets is (a) structured and (b) identifiable without test access.

Directly relevant closures:

- 2026-07-09 object-prior / score-field calibration: `KILL_FOR_CLAIM /
  NO_CONTINUE`, concluding that the next direction "should not be a stronger
  hard foreground prior". Foreground/background priors are therefore closed and
  are not proposed here.
- Baseline1 `k=5` scorer: reduced the good-pixel extreme tail
  (`good_pixel_p99_9` by `-48.46`) but cost macro pAUROC/F1/AU-PRO
  `-0.004222/-0.002285/-0.003457`. Suppressing the tail *inside the scorer*
  trades away ranking, so the scorer is not the instrument.
- Two-band protected residual and the DVT family: closed for band weights,
  sigma, scaling, Flow capacity, score fusion, and per-object policy.

## Evidence Motivating This Protocol

The `can`-collapse attribution rejected all three of its hypotheses:

| Hypothesis | Test | Result |
| --- | --- | --- |
| H-A support insufficiency | `can` S128 minus S16 full-range pixel AUROC | `-0.0034` (threshold `+0.10`) — rejected, and more normals *hurt* |
| H-B representation deficiency | `can` best single-layer probe AUROC | `0.8442` (threshold `<0.60`) — rejected |
| H-C pipeline signal loss | best-layer minus deployed | `+0.0130` (threshold `+0.10`) — rejected |

What remains is an operating-point phenomenon. `can` ranks defects acceptably
overall (deployed full-range pixel AUROC `0.8312`) yet collapses where it is
scored: pAUROC@0.05 `0.5734`, raw-oracle F1 `0.0010`,
`gt_component_recall_at_oracle` `0.135`. Its normal false-positive rate at the
oracle threshold is `0.00562`, second highest of eight. `vial` shows the same
signature from the other side: `gt_component_recall_at_oracle` `1.000` and
AU-PRO `0.8305` — the best of the eight — but pAUROC@0.05 `0.7536`, F1
`0.4403`, and the highest normal FPR `0.00585`.

Across all eight objects, the three worst normal FPRs (`vial` `0.00585`,
`can` `0.00562`, `fabric` `0.00177`) and the best two (`rice` and
`sheet_metal`, both `0.00001`) order the objects by exactly the deficit under
study. That association is currently descriptive, not causal.

The macro stake is large: `gt_component_recall_at_oracle` is `0.484` (`0.534`
excluding `can`), so more than half of ground-truth components are never
touched at the best available threshold, and private AU-PRO trails RoBiS by
`-0.1285`.

## Research Question

What generates the normal-side high-score tail that destroys low-FPR
performance, and can that tail be identified using normal data only, without
any test access?

This protocol answers only that. It does not build, tune, or evaluate any
suppression mechanism.

## Dataset and Frozen Baseline Contract

- `target_dataset=MVTec AD2 single-image`
- local data root: `/home/hun/Volume/DATA/mvtec_ad_2`
- remote data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- objects: `can,fabric,fruit_jelly,rice,sheet_metal,vial,wallplugs,walnuts`
- scoring system: deployed Baseline1-core, unchanged — DINOv3-H+ `[7,15,23,31]`,
  `layer_norm_mean`, resolution 672 except `sheet_metal=448`, DVT
  `position_mean`, Flow 3 epochs, density weight 0, `raw_1nn`, `rgb-guide none`,
  `threshold-calibration none`, `binary-postprocess none`
- support: frozen DINOv3-H+ Baseline1-core support-16,
  `skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json`,
  SHA-256 `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`
- thresholds: the already-frozen per-object public thresholds in
  `configs/baselines/baseline1_core_public_thresholds.json`, used unchanged.
  No threshold is selected, swept, or re-derived in this protocol.
- forbidden: query-time update, memory expansion, condition/filename metadata,
  `test_private`, `test_private_mixed`, and any new fitting on `test_public`
- SuperAD/SuperADD fields are `N/A`; `comparable=false`.

## Definitions

For object `o` with frozen threshold `t_o`:

- **Test-normal tail**: pixels in `test_public/good` images whose deployed score
  exceeds `t_o`. These are exactly the false positives that suppress
  pAUROC@0.05 and F1.
- **Fit-normal set**: the 112 images `S128 \ S16` from the already-built,
  custody-verified ladder
  `configs/experiments/can_collapse_attribution_support_ladder.json`. These are
  `train/good` images disjoint from the support-16 and from every test split.
- **Fit-normal tail**: the same rule applied to fit-normal images.
- **Position recurrence** `r(p)`: over a given image set, the fraction of images
  whose score at patch position `p` lies in that set's tail.

## Block T1 — Tail Structure

Descriptive, on test-normal images.

- **Image concentration** `C_img`: share of total tail mass contributed by the
  top decile of normal images, plus the Gini coefficient of per-image tail mass.
- **Position concentration** `C_pos`: share of total tail mass carried by the
  top `1%` of positions ranked by `r(p)`.
- **Hard null** `C_pos_null`: recompute `C_pos` after an independent
  within-image permutation of positions per image, which preserves every
  image's score multiset exactly and destroys only spatial correspondence.
  Fixed seed 0, one permutation draw per image.

## Block T2 — Predictability From Normals Only

- Compute `r_fit(p)` on the fit-normal set and `r_test(p)` on the test-normal
  set.
- **Predictability** `rho`: Spearman rank correlation between `r_fit` and
  `r_test` over all positions.
- **Hard null** `rho_null`: Spearman between `r_fit` and a within-image
  position-permuted `r_test`, same permutation family and seed as T1.
- **Image-level predictability**: Spearman between a fit-normal-derived
  per-image tail-mass statistic and the test-normal per-image tail mass, with
  the same null. Reported for completeness; it does not gate.

## Block T3 — Anti-Absorption Pre-Check

This is the decisive economy of the protocol. Before any suppression mechanism
is ever built, measure whether suppression could possibly be safe.

- Let `P1` be the top `1%` of positions by `r_fit(p)` — chosen from fit-normal
  only, never from test.
- **Defect overlap** `V`: the fraction of all ground-truth defect pixels in
  `test_public/bad` that fall at positions in `P1`.

If recurrent normal-tail positions are also where defects appear, suppression
must destroy defect evidence, and no mechanism can escape it. That is precisely
how the two-band, invariant-channel, and stable-key branches died.

## Primary Endpoints and Gates

Evaluated in this order. The protocol stops at the first failure.

**G1 structure.** On at least `6/8` objects, `max(C_img_excess, C_pos_excess)`
must be positive by a margin, where `C_pos_excess = C_pos - C_pos_null >= 0.10`
and `C_img_excess` requires top-decile share `>= 0.30` (a decile carrying three
times its proportional share). If G1 fails the verdict is
`DIFFUSE_TAIL / NO_CONTINUE`.

**G2 predictability.** On at least `6/8` objects, `rho >= 0.30` and
`rho - rho_null >= 0.20`. If G2 fails the verdict is
`TAIL_NOT_IDENTIFIABLE_WITHOUT_TEST / NO_CONTINUE`.

**G3 anti-absorption.** On at least `7/8` objects, `V <= 0.10`; and on no
object may `V >= 0.25`. If G3 fails the verdict is
`SUPPRESSION_WOULD_ABSORB_DEFECTS / NO_CONTINUE`.

If G1, G2, and G3 all pass, the verdict is
`TAIL_STRUCTURED_AND_IDENTIFIABLE / CONTINUE_TO_ONE_INTERVENTION`, which
licenses exactly one subsequently preregistered intervention experiment. It
does not license that intervention's design here, and it is not a method claim.

## Validity Gates

All must pass before any endpoint is read:

1. all eight objects complete; full map/GT coverage; finite, nonconstant score
   maps; no retained `anomaly_maps/` after conclusion;
2. exact support manifest, threshold file, preprocessing, resolution policy,
   Flow config, scorer, and evaluator custody, asserted from the emitted run
   config rather than the launch command;
3. fit-normal set is exactly `S128 \ S16` per object, size `112`, drawn only
   from `train/good`, with zero intersection with support-16, `test_public`,
   `validation`, or any private split;
4. the permutation null preserves each image's score multiset exactly
   (sorted-value equality within `1e-6`) and is not the identity permutation;
5. no query-time update, no metadata/condition input, no private access, and no
   new fitting on `test_public`;
6. duplicated local/dsba5 arm parity: maximum absolute difference of shared
   continuous statistics `<= 1e-3`.

## Anti-Overfitting and Stop Conditions

- No threshold, quantile, percentile, kernel, or suppression weight may be
  swept. `1%`, `top decile`, and the gate constants above are fixed here and
  may not be changed after seeing results.
- `P1` is defined from fit-normal only. Defining it from test-normal would make
  G3 circular and is forbidden.
- A `CONTINUE` verdict licenses one intervention experiment, not a family. If
  that intervention later fails its own preservation gate, the normal-tail
  direction closes permanently.
- Reporting must state that the normal-FPR-versus-performance association is
  descriptive; this protocol does not establish that the tail *causes* the
  low-FPR deficit.

## Execution and Cleanup

- Terra implements the tail statistics, the permutation null, the three blocks,
  the aggregator with all gates, and focused regression tests. No new scoring
  mathematics: score maps come from the unchanged deployed pipeline.
- Score-map generation and T1/T2 run on the local RTX 4090 container
  `hun_fsad_tta_local_stage1a0`.
- The permutation-null arm runs on dsba5 container `hun_fsad_tta_012`, host
  GPU 3, container CUDA slot 0, via an immutable snapshot.
- One arm is deliberately duplicated across local and dsba5 for parity.
- Score maps are retained only for the duration of the statistics computation
  and deleted afterwards; verify absence of `anomaly_maps/` at conclusion.

## Reporting

The report must state the verdict, all three gate blocks with per-object
numbers, the validity block, and an explicit statement of what the verdict does
and does not license. If any gate fails, the report must record which specific
successor designs are thereby closed.
