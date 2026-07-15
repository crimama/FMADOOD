# AD2 H+ Flow-LatentBank + DVT + Density + RGB Guide + Morphology

Date: 2026-07-13
Verdict: `PASS_MEAN_GATE / FAIL_CLASS_HARM_GATE / BLOCKED_BASELINE`

## Outcome

The requested all-eight MVTec AD2 run completed. The retained H+ upstream
setting was frozen and its anomaly maps were passed through the fixed
half-scale RGB guided-r8 refinement and the default binary morphology:

```text
17px directional close x 16 angles -> hole fill -> 3x3 erode
```

The candidate improves mean morphology F1 from `0.542316` to `0.553859`
(`+0.011544`) while mean AUROC changes from `0.836739` to `0.837528`
(`+0.000789`). This passes the preregistered mean positive/retention gate.
It does not pass the class no-harm gate because `can` loses `0.036604` AUROC.

## Frozen setting

- MVTec AD2, all eight objects, full `test_public/good,bad`.
- Fixed 16-shot support JSON, seed 0, identity support.
- Frozen `dinov3_vith16plus`, layers `[7,15,23,31]`.
- Dataset-native resolution: 672 except `sheet_metal=448`; no tiling,
  masking, rotation, or brightness augmentation.
- Flow 3 epochs, 2 coupling layers, hidden multiplier 1, LR `2e-4`.
- Static latent bank (`expansion_budget=1.0`), latent 1-NN distance 1.0.
- Existing LOO support standardization, DVT position mean alpha 1.0.
- Density weight 0.25; no context/register/flow conditioning.
- Guided-r8: half-scale radius 8, epsilon 0.01.
- Both morphology rows use each continuous map's raw best threshold.

## Mean comparison

| Configuration | seg AUROC@0.05 | seg F1 | Delta AUROC vs raw | Delta F1 vs raw |
|---|---:|---:|---:|---:|
| Existing H+ raw (no guide, no morphology) | 0.836739 | 0.527428 | - | - |
| Existing H+ + morphology | 0.836739 | 0.542316 | +0.000000 | +0.014888 |
| H+ + RGB guide (no morphology) | 0.837528 | 0.546785 | +0.000789 | +0.019357 |
| **Proposed: H+ + RGB guide + morphology** | **0.837528** | **0.553859** | **+0.000789** | **+0.026432** |
| Reported SuperADD context | 0.839300 | 0.626113 | +0.002561 | +0.098685 |

Against the matched `Existing H+ + morphology` control, the proposed row is
`+0.000789` AUROC and `+0.011544` F1. Against reported SuperADD context it is
`-0.001772` AUROC and `-0.072254` F1. The SuperADD row is contextual, not a
strict same-run baseline.

## Per-object matched comparison

| Object | Control AUROC | Proposed AUROC | Delta | Control morph F1 | Proposed morph F1 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| can | 0.560495 | 0.523892 | **-0.036604** | 0.000505 | 0.000365 | -0.000140 |
| fabric | 0.968227 | 0.975629 | +0.007403 | 0.800681 | 0.803399 | +0.002718 |
| fruit_jelly | 0.781873 | 0.792065 | +0.010192 | 0.493198 | 0.517986 | +0.024787 |
| rice | 0.947121 | 0.950802 | +0.003681 | 0.712211 | 0.706800 | -0.005411 |
| vial | 0.746292 | 0.757292 | +0.011001 | 0.440806 | 0.454983 | +0.014177 |
| wallplugs | 0.908028 | 0.910773 | +0.002745 | 0.626132 | 0.666206 | +0.040074 |
| walnuts | 0.890238 | 0.893137 | +0.002899 | 0.731349 | 0.740414 | +0.009065 |
| sheet_metal | 0.891639 | 0.896633 | +0.004994 | 0.533643 | 0.540721 | +0.007078 |
| **Mean** | **0.836739** | **0.837528** | **+0.000789** | **0.542316** | **0.553859** | **+0.011544** |

RGB guidance improves AUROC on 7/8 objects and morphology F1 on 6/8. The
largest F1 gain is `wallplugs` (`+0.040074`); the primary failure is the
`can` AUROC regression. `rice` has a small F1 regression despite better
AUROC.

## Morphology contribution

Morphology adds `+0.014888` mean F1 to the unguided maps and `+0.007075` to
the guided maps. Its benefit is not uniform: much of the gain comes from
`fabric`, while it slightly lowers guided F1 for `fruit_jelly`, `wallplugs`,
`walnuts`, and `sheet_metal`. Therefore the proposed mean improvement should
not be described as uniform class-level improvement.

## Gate assessment

- Validity gate: pass. All eight raw and eight guided metrics are finite;
  manifests preserve the frozen H+ setting and fixed support JSON.
- Reproduction gate: pass. The new control exactly reproduces the historical
  H+ means (`0.836739/0.527428` before morphology).
- Mean positive/retention gate: pass through F1 (`+0.011544 >= +0.005`) with
  non-negative AUROC delta.
- Class harm gate: fail. `can` AUROC delta is `-0.036604`, beyond the locked
  `-0.03` threshold.
- Strict external claim: blocked. Reported SuperADD remains contextual and
  the proposed F1 remains `0.072254` lower.

The fixed guided-r8 component is therefore useful as a diagnostic and raises
the AD2 mean, but should not yet be promoted as an unconditional all-class
default. A follow-up must address `can` with a preregistered, ground-truth-free
guide-strength rule rather than selecting a class exception from these test
labels.

## Subsequent operational-default decision

After reviewing this result, the user explicitly selected RGB guided-r8 plus
binary morphology as the AD2 operational default. The runner therefore uses
that pipeline by default as of 2026-07-13. This does not retroactively change
the preregistered verdict or erase the `can` risk: it distinguishes the
configuration used by default from the stronger all-class method claim, which
remains unsupported by this experiment. Unguided and non-morphological arms
remain available through explicit CLI ablation switches.

## Execution and audit

- Server: dsba3 GPUs 0,1,2,3.
- Raw inference was executed once per object because historical dense maps had
  been cleaned; the guided arm reused those exact maps. No independent second
  baseline training run was performed.
- dsba5 was preflighted as requested but lacked both the DINOv3-H+
  `transformers` runtime and model cache. Its temporary AD2 copy and failed
  preflight output were removed without changing dependencies.
- Dense `anomaly_maps/` directories after completion: zero locally and
  remotely.
- Final dsba3 GPU state: 1 MiB and 0% utilization on GPUs 0--3.
- Full local regression after implementation: 434 tests passed.

## Artifacts

- Aggregate:
  `results/remote_runs/dsba3/flowtte_ad2_hplus_guided_r8_morph_all8_20260713_v1/summary.json`
- Per-object TSV:
  `results/remote_runs/dsba3/flowtte_ad2_hplus_guided_r8_morph_all8_20260713_v1/per_object_metrics.tsv`
- Preregistration:
  `skill_graph/experiments/2026-07-13_flowtte_ad2_hplus_guided_r8_morph/preregistration.md`
