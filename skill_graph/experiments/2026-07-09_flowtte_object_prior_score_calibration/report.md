# FlowTTE Object Prior and Score Calibration Diagnostic

Date: 2026-07-09
Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`

## Negative Evidence Intake

This run follows the raw hard-null conclusion: do not replace the raw scorer;
instead test stronger class-agnostic object/foreground priors and score-field
calibration on top of the retained H+ DVT NF latent branch.

This is not a class-specific threshold or morphology sweep. It keeps the
backbone, support set, NF latent scorer, DVT denoise, and no-TTE policy fixed.

Known failure basin: post-hoc score suppression can look helpful on one object
while destroying true anomaly evidence or strong/control categories. The run is
therefore diagnostic only and cannot support a method claim without beating the
current H+ DVT NF reference and preserving no-harm behavior.

## Motivation

The current retained branch is:

```text
DINOv3-H+/16 layers [7,15,23,31]
-> DVT-style position-mean denoise alpha 1.0
-> NF latent projection
-> fixed 16-shot support latent bank
-> latent NN distance + weak density
```

Reference metrics on all-eight MVTec AD2 `test_public`:

- H+ DVT NF latent reference: `0.836739` AUROC / `0.527427` F1
- Reported SuperADD context: `0.839300` AUROC / `0.626113` F1
- Recorded SuperAD-16 context: `0.765802` AUROC / `0.385534` F1

The motivation is to test whether the remaining F1 gap can be addressed by
class-agnostic objectness and support-score reliability without changing the
main scorer.

## Implementable Design

Four all-object variants were implemented:

| Variant | Mechanism |
|---|---|
| `rgb_object_prior` | support RGB border-background contrast prior |
| `rgb_feature_product_prior` | product of RGB contrast prior and support feature-energy prior |
| `support_score_reliability` | multiplicative suppression at positions with high support LOO scores |
| `rgb_prior_plus_reliability` | RGB contrast prior plus support-score reliability |

All variants use the same fixed support set and the same parameters for every
object. No class-specific tuning is allowed.

Continuation gate:

- mean F1 should improve over `0.527427`, or
- a weaker continuation signal must improve F1 with bounded AUROC/no-harm
  losses on strong objects.

Hard-stop condition:

- mean F1 falls below the retained H+ DVT NF reference, especially with broad
  object collapse or no-harm failure.

## Evaluation Alignment

- Target dataset: MVTec AD2 single-image.
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`.
- Objects: `can`, `fabric`, `fruit_jelly`, `rice`, `vial`, `wallplugs`,
  `walnuts`, `sheet_metal`.
- Split: full `test_public/good,bad`.
- Candidate reference policy: same fixed 16-shot support JSON used by the H+
  DVT NF diagnostics.
- Metrics: all-object macro mean `seg_AUROC_0.05`, `seg_F1`.

The strict SuperAD/SuperADD claim gate is not open here because this is an
internal diagnostic against the retained FlowTTE branch.

## Code Modification / Creation

Modified:

- `scripts/flow_tte_score_field.py`
  - added RGB contrast foreground prior;
  - added RGB-feature product prior;
  - added support-score reliability calibration.
- `scripts/flow_tte_score_priors.py`
  - added reusable feature-energy, RGB-contrast, prior thresholding, and
    support-score reliability helpers.
- `scripts/flow_tte_mvtec_ad2_core.py`
  - connected selected support paths to score-field stats for RGB prior
    construction.
- `scripts/run_flow_tte_mvtec_ad2.py`
  - exposed new score-field CLI choices and support score quantile.
- `scripts/run_flow_tte_dvt_denoising_all8_remote.sh`
  - passed and logged the new quantile option.
- `scripts/run_flow_tte_object_prior_remote.sh`
  - all-eight runner for the four variants.
- `tests/test_flow_tte_score_field.py`
  - focused tests for RGB prior and reliability calibration.

## Added Code Evaluation

Local checks before remote execution:

- `ruff check` on changed Python files: passed.
- `pytest tests/test_flow_tte_score_field.py -q`: passed, 6 tests.
- `basedpyright`: passed.
- `compileall` and `bash -n`: passed.

Final local checks before commit:

- `uv run ruff check src/flow_tte scripts/*.py tests`: passed.
- `uv run --with pillow --with tifffile --with opencv-python-headless pytest tests -q`:
  passed, 62 tests with one local CUDA-driver warning.
- `uv run basedpyright`: passed, 0 errors.
- `python3 -m compileall -q src/flow_tte scripts tests`: passed.
- `bash -n scripts/run_flow_tte_object_prior_remote.sh scripts/run_flow_tte_dvt_denoising_all8_remote.sh`:
  passed.
- `git diff --check`: passed.

Remote preflight:

- container: `hun_fsad_tta_012`;
- code root: `/root/fsad_tta_run`;
- compile and bash syntax check passed;
- host GPUs `0,1,2` were idle before launch.

## Remote Execution

Remote command ran inside the container:

```text
FSAD_ROOT=/root/fsad_tta_run
RUN_GROUP_NAME=flowtte_object_prior_all8_20260709_v1
RESULTS_ROOT=/root/results_remote
DATA_ROOT=/home/hunim/Volume/DATA/mvtec_ad_2
bash scripts/run_flow_tte_object_prior_remote.sh
```

Remote result root:

```text
/root/results_remote/flowtte_object_prior_all8_20260709_v1
```

Local pullback:

```text
results/remote_runs/dsba3/flowtte_object_prior_all8_20260709_v1
results/remote_runs/dsba3/flowtte_object_prior_all8_20260709_v1.tgz
```

## SuperAD Baseline and Unified Metrics

Primary internal comparator:

| Comparator | AUROC | F1 | Comparable |
|---|---:|---:|---|
| H+ DVT NF latent reference | 0.836739 | 0.527427 | true |
| Reported SuperADD context | 0.839300 | 0.626113 | context only |
| Recorded SuperAD-16 context | 0.765802 | 0.385534 | context only |

## Results and Analysis

| Variant | AUROC | F1 | dAUROC vs H+ DVT NF | dF1 vs H+ DVT NF |
|---|---:|---:|---:|---:|
| `rgb_feature_product_prior` | 0.767204 | 0.346625 | -0.069535 | -0.180803 |
| `rgb_object_prior` | 0.778452 | 0.374071 | -0.058287 | -0.153356 |
| `rgb_prior_plus_reliability` | 0.780889 | 0.376619 | -0.055850 | -0.150809 |
| `support_score_reliability` | 0.828178 | 0.505659 | -0.008561 | -0.021769 |

Per-object metrics are in `per_object.tsv`; summary metrics are in
`summary.tsv`.

Interpretation:

- RGB contrast objectness is too aggressive. It improves or preserves some
  obvious foreground classes, but it suppresses true anomaly evidence in
  objects such as `walnuts`, `wallplugs`, and `rice`.
- RGB-feature product is even more restrictive and has the worst mean F1.
- Support-score reliability is less destructive and shows positive local
  signal on `fabric` and `wallplugs`, but it still loses mean F1 and harms
  `can`, `sheet_metal`, and `walnuts`.
- Combining RGB prior with reliability inherits the RGB prior failure.

The core failure is now clearer:

> A hard object/background prior built from support RGB or support score fields
> suppresses anomaly evidence along with nuisance background. The current
> score-field needs an evidence-preserving, soft calibration mechanism rather
> than binary or near-binary objectness suppression.

## Continuation Assessment

Strict method claim: no. Every variant loses to the retained H+ DVT NF branch.

Continuation for this exact family: no. RGB priors are broadly harmful, and
support-score reliability has only a weak local signal that does not justify
another fixed-parameter sweep.

Useful retained asset:

- RGB prior and support-score reliability code can remain as diagnostic
  controls.
- The negative evidence rules out hard class-agnostic objectness suppression as
  the next main direction.

## Conclusion

`KILL_FOR_CLAIM / NO_CONTINUE`

The next structural direction should not be a stronger hard foreground prior.
It should measure and model score-field uncertainty or component fragmentation
without suppressing the continuous anomaly evidence before thresholding.

## Post-Conclusion Storage Cleanup

Remote and local result roots were checked after completion. No `anomaly_maps/`
directories remain under:

```text
/root/results_remote/flowtte_object_prior_all8_20260709_v1
results/remote_runs/dsba3/flowtte_object_prior_all8_20260709_v1
```
