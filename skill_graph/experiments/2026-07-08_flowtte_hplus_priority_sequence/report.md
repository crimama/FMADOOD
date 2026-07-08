# FlowTTE H+ Priority Sequence Diagnostic

Date: 2026-07-08
Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`

## Negative Evidence Intake

This is not a new method claim. It is a prioritized diagnostic sequence after
the H+ backbone-only run nearly matched reported SuperADD AUROC but still
trailed F1.

Known failure basin: the current branch is still a passive feature/memory NN
family. Apparent gains can come from backbone, threshold, or binary
postprocessing rather than a new anomaly-aligned mechanism. The strict claim
gate therefore remains SuperAD/SuperADD-context comparison plus no F1 collapse.

## Motivation

The previous H+ backbone-only result was:

```text
FlowTTE DVT alpha=1.0
+ DINOv3-H+/16 layers [7,15,23,31]
+ fixed 16-shot support
+ NF latent NN distance + 0.25 density
= 0.836739 AUROC_0.05 / 0.527427 F1
```

This nearly closed reported SuperADD AUROC (`0.839300`) but left F1 below by
`0.098686`. The priority order was therefore:

1. Keep H+ fixed and test threshold/morphology.
2. Keep H+ fixed and remove learned NF projection with an identity
   feature-distance control.

## Implementable Design

Shared target:

- `target_dataset=MVTec AD2 single-image`
- `data_root=/home/hunim/Volume/DATA/mvtec_ad_2`
- objects: `can, fabric, fruit_jelly, rice, vial, wallplugs, walnuts, sheet_metal`
- split: full `test_public/good,bad`
- support: fixed 16-shot JSON from the DINOv3 no-context run
- backbone: `dinov3_vith16plus`
- feature layers: `[7,15,23,31]`
- DVT denoise: `position_mean`, `alpha=1.0`
- no TTE: `expansion_budget=1.0`
- primary metrics: `seg_AUROC_0.05`, `seg_F1`

Priority 1:

- Preserve H+ NF latent continuous maps.
- Evaluate raw threshold, close/fill, and close/fill/erode at the metric
  best threshold.
- Use oracle-grid rows only as non-claim diagnostics because this reduced
  re-run used `threshold_count=1` to avoid a large CPU-only sweep over 17GB
  of TIFF maps.

Priority 2:

- Rerun all eight objects with `flow_transform_mode=identity`.
- Set `density_weight=0.0`.
- Keep `score_mode=latent_distance`.
- This is a no-NF standardized feature-distance control, not an exact
  SuperADD layer-wise raw NN implementation.

## Evaluation Alignment

The H+ NF, H+ postprocess, and H+ identity control are directly comparable on
dataset, objects, support, backbone, feature layers, DVT alpha, no-TTE policy,
and evaluator.

The comparison to SuperADD remains reported-context only. It is not a strict
same-run SuperADD artifact, and SuperADD still differs in raw layer-wise
distance, threshold calibration, high-resolution execution, and morphology.

## Code Modification / Creation

Created:

- `scripts/flow_tte_postprocess_core.py`
- `scripts/flow_tte_postprocess_eval.py`
- `tests/test_flow_tte_postprocess_eval.py`
- `scripts/run_flow_tte_hplus_postprocess_remote.sh`

Modified:

- `scripts/run_flow_tte_dvt_denoising_all8_remote.sh`

The remote DVT launcher now accepts environment overrides for
`FLOW_TRANSFORM_MODE`, `DENSITY_WEIGHT`, and `SCORE_MODE`, and records those
fields in `remote_run_complete.txt`.

## Added Code Evaluation

Local checks:

```text
uv run ruff check scripts/flow_tte_postprocess_core.py scripts/flow_tte_postprocess_eval.py tests/test_flow_tte_postprocess_eval.py
uv run --with tifffile --with pillow --with opencv-python-headless pytest tests/test_flow_tte_postprocess_eval.py -q
python3 -m py_compile scripts/flow_tte_postprocess_core.py scripts/flow_tte_postprocess_eval.py tests/test_flow_tte_postprocess_eval.py
bash -n scripts/run_flow_tte_hplus_postprocess_remote.sh scripts/run_flow_tte_dvt_denoising_all8_remote.sh
```

Observed result: postprocess tests `2 passed`; shell syntax checks passed.

Remote sync check:

```text
remote_postprocess_sync_ok
SYNC_OK
```

## Remote Execution

Priority 1 remote root:

```text
/workspace/results_remote/flowtte_hplus_postprocess_all8_20260708_v1
```

Local pullback:

```text
results/remote_runs/dsba3/flowtte_hplus_postprocess_all8_20260708_v1
```

Priority 1 note: saved H+ anomaly maps contained `1084` TIFF files,
approximately `16992.7 MB`. The full threshold grid was too slow for a
diagnostic pass, so the final postprocess evaluation used `threshold_count=1`
and focused on morphology at the source metric threshold.

Priority 2 command:

```bash
RUN_NAME=flowtte_hplus_identity_feature_nn_all8_20260708_v1 \
BACKBONE_MODEL=dinov3_vith16plus \
FEATURE_LAYERS=7,15,23,31 \
DVT_ALPHA=1.0 \
FLOW_TRANSFORM_MODE=identity \
DENSITY_WEIGHT=0.0 \
SCORE_MODE=latent_distance \
CLEANUP_MAPS=1 \
FMAD_DINOV3_OFFLINE=1 \
bash scripts/run_flow_tte_dvt_denoising_all8_remote.sh
```

Priority 2 local pullback:

```text
results/remote_runs/dsba3/flowtte_hplus_identity_feature_nn_all8_20260708_v1
```

## SuperAD Baseline and Unified Metrics

| Method | AUROC_0.05 | F1 | Delta F1 vs H+ NF | Delta F1 vs SuperADD | Note |
|---|---:|---:|---:|---:|---|
| FlowTTE H+ NF latent DVT a1.0 | 0.836739 | 0.527427 | +0.000000 | -0.098686 | priority reference |
| H+ NF + close/fill | 0.836739 | 0.541344 | +0.013916 | -0.084769 | morphology at source threshold |
| H+ NF + close/fill/erode | 0.836739 | 0.542316 | +0.014888 | -0.083797 | best priority-1 variant |
| H+ identity feature NN | 0.832461 | 0.524804 | -0.002623 | -0.101309 | no-NF control |
| SuperAD-16 recorded context | 0.765802 | 0.385534 | -0.141893 | -0.240579 | context baseline |
| SuperADD reported context | 0.839300 | 0.626113 | +0.098686 | +0.000000 | reported context |

## Results and Analysis

Priority 1 result:

- Morphology improves mean F1 from `0.527427` to `0.542316`.
- It explains about `0.014888` of the `0.098686` F1 gap to reported SuperADD.
- The largest gain is `fabric`: `0.697427 -> 0.800681`.
- Smaller gains appear on `fruit_jelly`, `vial`, `rice`, `sheet_metal`.
- `can` does not matter for the SuperADD gap because reported SuperADD can F1
  is also `0.0`.

Priority 2 result:

- Identity feature-distance control is slightly worse on all8 mean:
  `0.832461 / 0.524804`.
- It improves `wallplugs` F1 strongly:
  `0.631539 -> 0.692127`.
- It hurts `fabric`, `fruit_jelly`, `vial`, and `sheet_metal`.
- Therefore the learned NF projection is not the primary mean-metric
  bottleneck under the current H+ DVT setup.

Object-level F1 view:

| Object | H+ NF | Post Best | Identity | SuperADD Reported |
|---|---:|---:|---:|---:|
| can | 0.000634 | 0.000505 | 0.001000 | 0.000000 |
| fabric | 0.697427 | 0.800681 | 0.664092 | 0.937400 |
| fruit_jelly | 0.481412 | 0.493198 | 0.458343 | 0.546800 |
| rice | 0.711554 | 0.712211 | 0.713083 | 0.733100 |
| vial | 0.434360 | 0.440806 | 0.424845 | 0.647700 |
| wallplugs | 0.631539 | 0.632716 | 0.692127 | 0.791600 |
| walnuts | 0.733291 | 0.733503 | 0.732422 | 0.756900 |
| sheet_metal | 0.529204 | 0.533643 | 0.512521 | 0.595400 |

Interpretation:

- Postprocessing is a real but insufficient contributor.
- NF removal is not a mean improvement, so the next branch should not be a
  broad "remove NF" rewrite.
- Remaining weak objects should be treated only as failure buckets for
  analysis, not as targets for class-specific hyperparameter tuning.
- The next useful branch should be class-agnostic: one shared mechanism applied
  to all eight objects, then checked for weak-object gains and strong-object
  no-harm.

## Continuation Assessment

Strict method claim now: no. The best current variant is still below reported
SuperADD F1 by `0.083797`, and reported SuperADD is not a same-run artifact.

Small continuation justified: yes, but not as hyperparameter tuning. The next
bounded diagnostic should test a class-agnostic structural mechanism:

```text
global foreground/background score suppression or calibration learned without labels
```

Hard-stop condition: if the same rule applied to all eight objects does not
improve the all-object mean while preserving strong-category no-harm, do not
continue with class-specific thresholds, class-specific morphology, or
object-specific score weights.

## Conclusion

Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.

Priority results:

```text
H+ NF latent reference:              0.836739 / 0.527427
+ close/fill/erode morphology:       0.836739 / 0.542316
H+ identity feature NN no-NF control: 0.832461 / 0.524804
Reported SuperADD context:           0.839300 / 0.626113
```

The best current FlowTTE variant after this sequence is still H+ NF latent
with morphology, not identity feature NN. The evidence says the remaining F1
gap is not mainly caused by NF latent projection. It is more likely tied to
binary/postprocessing plus class-agnostic foreground/background separation or
high-resolution localization behavior.

## Post-Conclusion Storage Cleanup

Cleanup evidence:

- Priority 1 remote `anomaly_maps/` directory count: `0`.
- Priority 1 local `anomaly_maps/` directory count: `0`.
- Priority 2 remote `anomaly_maps/` directory count: `0`.
- Priority 2 local `anomaly_maps/` directory count: `0`.
- After Priority 2 completion, dsba3 GPUs `0,1,2` each reported `1 MiB` used
  and `0%` utilization.
