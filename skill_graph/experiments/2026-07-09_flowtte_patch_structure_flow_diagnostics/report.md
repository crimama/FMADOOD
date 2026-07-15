# FlowTTE Patch-Structure Flow Diagnostics

## Question

The current retained method treats DINOv3 patch embeddings as independent flow
samples:

```text
DINOv3-H+ patch feature
-> DVT position_mean alpha 1.0
-> NF latent projection
-> fixed support latent NN distance
-> anomaly map
```

The structural hypothesis is that the bottleneck is not only foreground/background
separability, but the patch-independent flow itself: global context, position,
foreground/background modes, and local score-field structure are not modeled.

## Fixed Reference

- Dataset: MVTec AD2 `test_public/good,bad`, all 8 objects.
- Support: same fixed 16-shot JSON used by the H+ reference branch.
- Backbone: `dinov3_vith16plus`, layers `[7,15,23,31]`.
- Denoise: `position_mean`, `alpha=1.0`.
- Internal retained reference: `seg_AUROC_0.05=0.836739`, `seg_F1=0.527427`.
- Reported context baseline: SuperADD `0.839300/0.626113`.

## Implemented Variants

See `experiment_plan.tsv`.

Code changes prepared:

- `xy`, `cls_xy`, `register_xy`, `cls_register_xy` context sources.
- Tiled extraction can now broadcast full-image context over accumulated feature
  grids instead of rejecting context sources.
- `foreground_flow_mixture` normality mode: fit foreground/background flow
  components from support feature-energy split, then score with componentwise
  `min(score_fg, score_bg)` instead of hard background suppression.
- `local_contrast` score-field calibration: amplifies local peaks over a smooth
  score field using `score + alpha * (score - box_filter(score))`.
- Remote runner: `scripts/run_flow_tte_patch_structure_remote.sh`.

## Verification

Passed locally:

```text
python3 -m py_compile scripts/flow_tte_mvtec_ad2_core.py scripts/run_flow_tte_mvtec_ad2.py scripts/flow_tte_score_field.py
python3 scripts/run_flow_tte_mvtec_ad2.py --help
uv run ruff check scripts/flow_tte_mvtec_ad2_core.py scripts/run_flow_tte_mvtec_ad2.py scripts/flow_tte_score_field.py tests/test_flow_tte_score_field.py tests/test_flow_tte_layerwise.py
python3 -m pytest tests/test_flow_tte_score_field.py tests/test_flow_tte_layerwise.py -q
```

`uv run pytest ...` is not the active test environment for this repo because the
uv environment lacks `cv2`; system Python has `cv2 4.13.0` and the relevant tests
pass there.

## Remote Execution

Executed on dsba3 in fixed container `hun_fsad_tta_012`, using host GPUs
`0,1,2`.

Protocol alignment:

- `target_dataset=MVTec AD2 single-image`
- `data_root=/home/hunim/Volume/DATA/mvtec_ad_2`
- objects: `can`, `fabric`, `fruit_jelly`, `rice`, `vial`, `wallplugs`,
  `walnuts`, `sheet_metal`
- split: full `test_public/good,bad`
- support: fixed 16-shot JSON from the H+ reference branch
- metrics: `seg_AUROC_0.05`, `seg_F1`

Remote roots:

```text
/workspace/results_remote/flowtte_patch_structure_all8_conditional_cls_20260709_v1
/workspace/results_remote/flowtte_patch_structure_all8_conditional_xy_20260709_v1
/workspace/results_remote/flowtte_patch_structure_all8_conditional_cls_xy_20260709_v1
/workspace/results_remote/flowtte_patch_structure_all8_foreground_flow_mixture_20260709_v1
/workspace/results_remote/flowtte_patch_structure_all8_local_contrast_20260709_v1
```

Local pullbacks are under:

```text
results/remote_runs/dsba3/flowtte_patch_structure_all8*_20260709_v1
```

## Results

Reference:

| method | seg_AUROC_0.05 | seg_F1 |
|---|---:|---:|
| H+ DVT FlowTTE reference | 0.836739 | 0.527427 |
| reported SuperADD context | 0.839300 | 0.626113 |

Patch-structure diagnostics:

| variant | status | objects | seg_AUROC_0.05 | seg_F1 | delta F1 vs H+ |
|---|---|---:|---:|---:|---:|
| conditional_cls | ok | 8 | 0.832374 | 0.512126 | -0.015301 |
| conditional_xy | runtime-blocked | 0 | NA | NA | NA |
| conditional_cls_xy | runtime-blocked | 0 | NA | NA | NA |
| foreground_flow_mixture | ok | 8 | 0.834712 | 0.519250 | -0.008177 |
| local_contrast | ok | 8 | 0.806200 | 0.438345 | -0.089082 |

`conditional_xy` was manually stopped after about 30 minutes with no chunk
metrics. `conditional_cls_xy` reproduced the same runtime pattern and was
stopped after about 15 minutes so the non-coordinate structural variants could
finish. Both runs produced no usable all-object metrics.

Per-object highlights:

- `foreground_flow_mixture` improved `fabric` F1 over `conditional_cls`
  (`0.690061` vs `0.604600`) and modestly improved `vial`, but still failed the
  all-object mean because `can` stayed near zero F1 and `wallplugs/sheet_metal`
  dropped.
- `conditional_cls` did not reproduce the earlier CLS soft-distance gain at the
  H+ setting. Global context as a conditional NF input is weaker than the fixed
  H+ reference.
- `local_contrast` is broadly harmful. It preserves `rice/wallplugs` reasonably
  but damages `fabric`, `fruit_jelly`, `vial`, `sheet_metal`, and `walnuts`.
- Coordinate-conditioned NF is not currently a practical structural path:
  `xy` and `cls_xy` both entered a high-cost path without producing metrics in
  the diagnostic budget.

Full table:

```text
skill_graph/experiments/2026-07-09_flowtte_patch_structure_flow_diagnostics/results_summary.tsv
```

## Cleanup Evidence

Dense `anomaly_maps/` cleanup was performed after result pullback.

- Remote cleanup: `REMOTE_LEFT 0`
- Local cleanup: `LOCAL_MAPS_LEFT=0`
- Remote GPUs after completion: `0,1,2` each `1 MiB`, `0%`

Preserved artifacts: `metrics.json`, `run_manifest.json`, logs,
`variant_status.txt`, and this report.

## Current Conclusion

Verdict: `KILL_FOR_CLAIM / NO_CONTINUE` for the tested patch-structure variants.

The experiment does not support a strict method claim. None of the completed
class-agnostic structural variants improves the H+ DVT FlowTTE reference F1, and
all remain far below the reported SuperADD F1 context.

The useful evidence is negative but specific: the main remaining gap is not
solved by moving CLS into the NF condition, by a two-component foreground/
background flow mixture, or by local score-field contrast. Coordinate condition
should not be pursued in this implementation without first fixing its runtime
path and defining why position conditioning should be anomaly-aligned rather
than nuisance-aligned.
