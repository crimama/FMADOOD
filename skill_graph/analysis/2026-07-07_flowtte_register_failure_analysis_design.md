# FlowTTE Register Failure Analysis Design

Date: 2026-07-07
Status: designed
Verdict: CONTINUE_DIAGNOSTIC

## Negative Evidence Intake

The previous structural-register experiment did not support a method claim:

| Method | `seg_AUROC_0.05` | `seg_F1` | Delta AUROC vs no-context | Delta F1 vs no-context |
|---|---:|---:|---:|---:|
| no-context DINOv3 | 0.797743 | 0.437800 | 0.000000 | 0.000000 |
| CLS w10 diagnostic | 0.805427 | 0.447118 | +0.007684 | +0.009318 |
| register top-M=4 | 0.798846 | 0.434411 | +0.001103 | -0.003389 |
| register-conditioned NF | 0.796554 | 0.436152 | -0.001189 | -0.001648 |

This analysis is not a broad hyperparameter sweep. It is a failure-mode
decomposition after `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.

Likely failure basins:

- register context is weaker than CLS for selecting useful support images;
- top-M routing removes useful patch neighbors even when global context seems
  similar;
- conditional NF changes latent density/volume in a way that hurts some
  categories, especially `fabric`;
- mean metrics hide object-specific improvements on `can`, `fruit_jelly`,
  `sheet_metal`, or `wallplugs`.

## Motivation

The current methodological question is:

> Should DINOv3 register tokens remain in the FlowTTE direction, and if so,
> should they be used for feature analysis, memory routing, conditional NF, or
> not at all?

The analysis should explain the observed pattern before running more GPU-heavy
experiments:

- register top-M slightly improves AUROC but hurts F1;
- register-conditioned NF improves a few objects but loses in the mean;
- CLS context remains the strongest simple context diagnostic.

## Implementable Design

Target dataset: MVTec AD2 single-image.
Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`.
Objects: all eight public objects unless a reduced map audit is explicitly
triggered:
`can,fabric,fruit_jelly,rice,vial,wallplugs,walnuts,sheet_metal`.
Primary metrics for experiment linkage: `seg_AUROC_0.05`, `seg_F1`.

This is an analysis-first protocol with two phases.

### Phase A: Context and Retrieval Diagnostics

Run first. This phase should avoid new benchmark sweeps.

Inputs:

- existing result summaries for no-context, `CLS w10`, `register_topm4`, and
  `register_condnf`;
- DINOv3 support/test context features from the same 16-shot DINOv3 coreset
  setup;
- latent memory/query distances from sampled test patches if full patch logging
  is too large.

Diagnostics:

1. Context separability
   - Compare context sources: `CLS`, `register`, `CLS+register`.
   - Per object, compute support-to-test-good and support-to-test-bad context
     distances.
   - Report separation:
     `delta_context = mean(distance_bad) - mean(distance_good)`.
   - Positive signal: `register` or `CLS+register` gives larger positive
     separation than `CLS` on objects where register methods improved.

2. Support routing quality
   - For each test image, compute selected support groups for:
     no-context all-support retrieval, `CLS` top-M, and `register` top-M.
   - Report top-M membership overlap and whether `register_topm4` excludes the
     support image that provides the nearest latent patch under no-context
     retrieval.
   - Positive signal: register changes retrieval toward lower false positives
     on improved objects without increasing nearest-distance inflation on
     degraded objects.

3. Latent distance inflation
   - Sample query patches per image and compare nearest latent distance under
     no-context memory vs register top-M memory.
   - Report good/test-bad distributions separately.
   - Failure signal: register top-M increases distances similarly for good and
     bad patches, which can raise F1 threshold noise without improving anomaly
     ranking.

4. Conditional NF distortion
   - Compare unconditional NF and register-conditioned NF on the same support
     and sampled query patch features.
   - Report support latent covariance volume proxy, NLL mean/std, and
     good-vs-bad NLL separation by object.
   - Failure signal: conditional NF compresses support normals but also
     compresses bad patches or destabilizes density on `fabric`.

Outputs:

- `skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/context_metrics.tsv`
- `skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/retrieval_metrics.tsv`
- `skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/nf_distortion_metrics.tsv`
- a short report with per-object interpretation.

### Phase B: Reduced Map Morphology Audit

Run only if Phase A shows a concrete mechanism worth visually/map-level
checking. This phase intentionally preserves selected anomaly maps until the
audit is complete.

Reduced object set:

- `fabric`: largest positive for register top-M but largest negative for
  register-conditioned NF;
- `can`: positive conditional-NF object;
- `wallplugs`: conditional-NF positive but top-M negative;
- `vial`: register top-M negative/control object.

Variants:

- no-context DINOv3;
- `CLS w10` diagnostic;
- `register_topm4`;
- `register_condnf`.

Map diagnostics:

- false-positive area on good images;
- best-threshold predicted area ratio;
- connected-component count and largest-component ratio;
- GT-overlap concentration for bad images;
- anomaly-map rank correlation between no-context and register variants.

Outputs:

- retained selected maps only, not full dense `anomaly_maps/` trees after the
  audit;
- per-object morphology TSV;
- figure/contact sheet for a small fixed sample, if needed.

## Evaluation Alignment

This design is diagnostic, not a strict SuperAD-16 method claim.

Comparable diagnostic baseline:

- no-context DINOv3 Flow-LatentBank, same 16-shot DINOv3 coreset/no-TTE setup.

Context benchmark:

- `CLS w10` diagnostic, same implementation family.

SuperAD/SuperADD:

- recorded SuperAD-16 and reported SuperADD remain context only;
- they are not used as the gate for this analysis because the current runs use
  DINOv3 and non-paper-aligned support/preprocessing.

The analysis tests the motivating claim because it separates:

- global context quality from patch localization quality;
- context routing from NF latent projection;
- rank quality from threshold/F1 morphology.

## Keep / Kill Gates

Strict method claim gate:

- not evaluated here. This analysis cannot produce `KEEP`.

Continuation gate for register:

- continue register structurally only if at least one mechanism is supported:
  context separation, retrieval improvement, conditional-NF latent separation,
  or map-level false-positive reduction on a target object;
- and the same mechanism is not matched or beaten by CLS on most objects.

Hard stop for register:

- if register context separation is below CLS on at least 6 of 8 objects and
  retrieval changes correlate with F1 degradation, stop register-only routing;
- if conditional NF improves neither good/bad NLL separation nor map morphology
  on its positive objects, stop register-conditioned NF;
- if all observed improvements reduce to `fabric` only, treat as
  category-specific diagnostic, not a method direction.

Next experiment gate:

- if CLS explains most of the positive signal, run one CLS-structural follow-up
  only: `CLS top-M` or `CLS-conditioned NF`, not both unless the first one
  passes its gate;
- if register has a unique map-level benefit, test one hybrid setting:
  `CLS` for routing and `register` for analysis/conditioning, with no broad
  weight sweep.

## Execution Budget

Phase A:

- preferred first step;
- can use dsba3 GPU 0 for feature extraction if needed;
- no all-object benchmark rerun.

Phase B:

- only after Phase A;
- reduced 4-object rerun on dsba3 GPUs 0,1,2 if maps are required;
- map retention must be documented and cleaned after audit.

## Report Skeleton

The final analysis report should answer:

1. Is register context actually more anomaly-aligned than CLS?
2. Does register top-M routing remove useful nearest-neighbor evidence?
3. Does register-conditioned NF improve density separation or distort latent
   geometry?
4. Are F1 losses caused by localization failure or threshold/map-area
   morphology?
5. Should the next branch be register-only, CLS-only, or CLS/register hybrid?

Expected verdict options:

- `CONTINUE_DIAGNOSTIC`: a bounded follow-up is justified;
- `KILL_FOR_CLAIM / NO_CONTINUE_REGISTER`: register-only mechanisms have no
  independent signal;
- `PROMISING_DIAGNOSTIC`: one structural mechanism has a clear positive subset
  and a narrow next run.
