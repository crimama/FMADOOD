# Synthetic-Shift Memory-Bank Expansion Preregistration

Date: 2026-08-10

Protocol ID: `synthetic-shift-bank-expansion-all8-20260810`

Tier: mechanism diagnostic with a conditional promotion gate. Not yet a
SuperAD-comparable method claim.

## Negative Evidence Intake

This protocol sits between two closed results and must not be confused with
either.

**Stage -1b (`PROMISING_DIAGNOSTIC`)** established the premise. Under a shared
global photometric intervention, support features move in a direction that
transfers across images, beating a source-random null on `4/4` objects:

| object | shared transfer | random null | margin |
| --- | ---: | ---: | ---: |
| fruit_jelly | .9863 | .7121 | .2752 |
| walnuts | .9518 | .1518 | .7805 |
| vial | .9808 | .9023 | .0829 |
| fabric | .9610 | .5092 | .4093 |

**Stage -1c (`KILL_FOR_CLAIM / NO_CONTINUE_SHARED_DIRECTION_INTERVENTION`)**
closed the obvious use of that premise. Removing the direction with a fixed
rank-1 projection reduced the deployed raw-1NN normal tail by `~.0002` against
a `.02` gate, and on `fabric` the random null was better. The recorded failure
boundary is: *cross-image direction reproducibility is not a sufficient
condition for anomaly-score reduction.*

**This protocol does not reopen Stage -1c.** That branch removed a direction
from features by subtractive projection. This one leaves every feature
untouched and instead *adds coverage* to the static normal memory. The
operationalization, the failure modes, and the risk profile are different. If
this fails, it does not revive Stage -1c, and Stage -1c's closure is not
evidence against this design — the two are separate operationalizations of a
shared premise.

Also respected and not reopened: foreground/background priors
(2026-07-09, `KILL / NO_CONTINUE`), the `k` scorer default (`RETAIN_K1`), DVT
parameter sweeps, per-image rank normalization (withdrawn as unsound for a
pooled evaluator), and identify-then-suppress on normal data
(2026-08-10, `TAIL_NOT_IDENTIFIABLE_WITHOUT_TEST`).

## Motivating Evidence

Two of our own measurements make synthesis the only remaining source of shift
coverage.

- `can-collapse-attribution-all8-20260809` Block A1: expanding the support from
  16 to 128 **real** `train/good` images changed full-range pixel AUROC by a
  control median of `+0.0047`, and `can` by `-0.0034`. More normals from the
  same condition do not help.
- `normal-tail-structure-predictability-all8-20260810` G2: the test-normal tail
  does not transfer from `train/good` (`rho >= 0.30` on `1/8`). The nuisance is
  absent from the training normals by construction — MVTec AD 2 places
  unseen lighting conditions in the test splits by design.

Together: adding more of the same distribution is useless, and the missing
distribution cannot be learned from what we have. Synthesizing it is what
remains. The reference implementation in the literature (RoBiS) applies exposure
and noise augmentation and reports `+1.9%` F1-max from it, so the expected
effect is real but modest.

## Research Question

Does expanding the static normal memory bank with synthetically exposure-shifted
support features improve low-false-positive-rate performance beyond (a) the
unaugmented baseline, (b) an equal-energy wrong-direction null, and (c) real
same-condition normals at comparable bank size?

## Dataset and Frozen Baseline Contract

- `target_dataset=MVTec AD2 single-image`
- local data root: `/home/hun/Volume/DATA/mvtec_ad_2`
- remote data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- objects: `can,fabric,fruit_jelly,rice,sheet_metal,vial,wallplugs,walnuts`
- population: full `test_public/good,bad`
- support: frozen DINOv3-H+ Baseline1-core support-16,
  `skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json`,
  SHA-256 `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`
- backbone: frozen DINOv3-H+ `[7,15,23,31]`, `layer_norm_mean`, resolution 672
  except `sheet_metal=448`
- scorer: existing Phase-3 contract, DVT `position_mean`, density weight 0,
  `raw_1nn`, `rgb-guide none`, `threshold-calibration none`,
  `binary-postprocess none`
- forbidden: query-time update, condition/filename metadata, `test_private`,
  `test_private_mixed`, any fitting on `test_public`
- SuperAD/SuperADD fields `N/A`; `comparable=false`.

## Critical Attribution Constraint — Flow Is Held Fixed

The Flow is fitted on the **original 16 support features only** in every arm,
with identical seed and epochs. Augmented features are then passed through the
**already-fitted, frozen** Flow to populate the latent memory bank.

If the Flow were refitted on the expanded set, a change in results could come
from either the Flow or the bank, and the experiment would not be attributable.
This constraint is a hard validity gate, not a convenience.

## Fixed Augmentation

Exposure only, matching the published RoBiS range and applied deterministically
rather than sampled:

```text
I_lambda = clamp(I * 2^lambda,  0, 255)
lambda in { -0.2, -0.1333, -0.0667, 0, +0.0667, +0.1333, +0.2 }
```

Seven levels including identity, symmetric, spanning exactly `[-0.2, +0.2]`.
`16 x 7 = 112` bank source views per object. No sweep of the range, the level
count, the spacing, or the noise term is permitted. Gaussian noise is
deliberately excluded so that a positive result is attributable to exposure
shift alone.

## Pre-existing Brightness Jitter

The codebase already exposes `--support-brightness-range` (deployed value
`0.80,1.20`) and a per-image `rng.uniform(0.8, 1.2)` brightness factor in
`src/flow_tte/superadd_bank.py`. That mechanism replaces each image with one
random brightness rather than expanding coverage, and it is a different thing
from this protocol's augmentation.

It must be held **identical across all arms** and its effective value recorded
in the emitted config, so that it cancels in every contrast. A validity gate
asserts this. No arm may change it.

## Arms

| Arm | Bank contents | Bank size | Purpose |
| --- | --- | ---: | --- |
| `E0_baseline` | 16 original support views | 16 | frozen deployed anchor |
| `E1_exposure` | 16 x 7 exposure levels | 112 | candidate |
| `E2_noise_null` | 16 x 7 additive-noise views, energy-matched to `E1` | 112 | equal-energy wrong-direction hard null |
| `S128_real` | 128 real `train/good` normals | 128 | frozen anchor, reused from A1 |

`E2` uses zero-mean Gaussian pixel noise whose per-object standard deviation is
calibrated, before any test access, so that the mean L2 feature displacement
from the original 16 views matches `E1`'s within `5%`. This isolates "correct
shift direction" from "more bank diversity of the same magnitude".

`S128_real` is reused unchanged from
`results/can_collapse_attribution_20260809/local/A1/S128` subject to custody
verification. Its bank is `14%` larger than `E1`'s, so the comparison is
conservative against the candidate. That asymmetry is declared here and may not
be adjusted afterwards.

## Primary Metrics

- Primary: macro `seg_AUROC_0.05` (pAUROC@0.05) and macro
  `normal_image_pooled_fpr_at_oracle`.
- Co-primary preservation: macro `seg_F1` (raw oracle) and macro
  `seg_AUPRO_0.05`.
- Secondary: `gt_component_recall_at_oracle`, `pooled_pixel_auroc_float16`,
  `pooled_pixel_ap_float16`, `good_pixel_p99_9`, and per-object versions of all
  of the above.
- Original-resolution float16 tie-inclusive pooled evaluator throughout.

pAUROC@0.05 is primary because the established deficit is confined to the
low-false-positive region, and normal FPR is the mechanism endpoint the
augmentation is supposed to move.

## Validity Gates

All must pass before any endpoint is read:

1. all eight objects complete in every arm; full map/GT coverage; finite,
   nonconstant maps; no retained `anomaly_maps/` after conclusion;
2. exact support manifest custody; identical preprocessing, resolution policy,
   scorer, DVT mode, and evaluator across arms, asserted from the emitted run
   config rather than the launch command;
3. **Flow fixed**: every arm reports a Flow fitted on exactly the 16 original
   support views, with identical seed, epochs, and resulting parameter
   checksum. Any arm whose Flow checksum differs fails this gate;
4. bank sizes are exactly `16 / 112 / 112 / 128` per object, and `E1`'s bank
   contains the 16 identity views as a subset;
5. `E2` mean feature-displacement energy matches `E1`'s within `5%` per object,
   and `E2` is not the identity;
6. `--support-brightness-range` and `support_transforms` are byte-identical
   across all arms and recorded;
7. no query update, no metadata/condition input, no private access, no fitting
   on `test_public`;
8. duplicated local/dsba5 arm parity: maximum absolute difference of shared
   continuous metrics `<= 1e-3`.

## Hard-Null Gate

Both must pass, else `HARD_NULL_ABSORBED` and the family closes:

1. `E1 - E2` macro `seg_AUROC_0.05` `>= +0.005`;
2. `E1 - E2` macro `normal_image_pooled_fpr_at_oracle` `<= 0` (the correct
   direction reduces normal false positives at least as much as the null).

## Real-Normal Gate

`E1 - S128_real` macro `seg_AUROC_0.05` `>= +0.005`. If this fails, any gain is
attributable to bank size rather than to shift coverage, and the verdict is
`BANK_SIZE_EFFECT_ONLY / NO_CONTINUE`.

## Anti-Absorption Gate

Expanding the normal manifold can swallow defects. Block A1 already showed
support enlargement making `can` worse, which is the signature of this failure.

1. no object loses more than `0.03` macro-scale `seg_F1` versus `E0`;
2. no object loses more than `0.03` `seg_AUPRO_0.05` versus `E0`;
3. macro `pooled_pixel_ap_float16` `>= E0 - 0.005`.

Any violation yields `ANOMALY_ABSORPTION / NO_CONTINUE`.

## Strict Promotion Gate

All must pass:

1. `E1 - E0` macro `seg_AUROC_0.05` `>= +0.010`;
2. `E1 - E0` macro `normal_image_pooled_fpr_at_oracle` `< 0`;
3. `E1 - E0` macro `seg_F1` `>= -0.005` and macro `seg_AUPRO_0.05` `>= -0.005`;
4. hard-null, real-normal, and anti-absorption gates all pass.

The `+0.010` threshold is set against measured context: the A1 real-normal
ladder moved the control median `+0.0047`, the rejected `k=5` scorer moved
macro pAUROC `-0.0042`, and the DVT main effect — the only mechanism this
project has confirmed — was `+0.0347`. The bar is above ladder-level noise and
well below the one confirmed effect.

## Continuation Gate

If the strict gate fails but hard-null, real-normal, and anti-absorption all
pass and `E1 - E0` macro `seg_AUROC_0.05` `>= +0.005`, the verdict is
`CONTINUE_DIAGNOSTIC`, licensing exactly one preregistered follow-up. Otherwise
`KILL_FOR_CLAIM / NO_CONTINUE`.

## Anti-Overfitting and Stop Conditions

- No sweep of the exposure range, level count, spacing, noise sigma, bank size,
  `k`, DVT parameters, thresholds, or per-object policy. Every constant is
  fixed above.
- Refitting the Flow on the expanded bank is out of scope. It is a separate
  question and would destroy attribution here.
- A promoted result is not a SuperAD-comparable claim; promotion only licenses a
  separately preregistered paired-evaluator run under the Main Table contract.
- If the result is positive only on the three high-normal-FPR objects
  (`can`, `vial`, `fabric`), that must be reported as a subgroup observation and
  must not be converted into a per-object policy.

## Execution and Cleanup

- Terra implements the deterministic exposure ladder, the energy-matched noise
  null, the frozen-Flow bank population path, the aggregator with all gates,
  and focused regression tests.
- `E0` and `E1` run on the local RTX 4090 container
  `hun_fsad_tta_local_stage1a0`.
- `E2` hard null runs on dsba5 container `hun_fsad_tta_012`, host GPU 3,
  container CUDA slot 0, via an immutable snapshot.
- `S128_real` is reused from A1 after custody verification; it is rerun if
  custody fails.
- One arm is deliberately duplicated across local and dsba5 for parity.
- Preserve compact JSON/log/summary/parity artifacts only; verify absence of
  `anomaly_maps/` after the conclusion.

## Reporting

The report must state the verdict, every gate block with per-object numbers, the
Flow-fixed checksum evidence, and an explicit statement of what the verdict
licenses. It must state plainly whether the effect exceeds the real-normal
ladder, since that is the difference between shift coverage and bank size.
