# FlowTTE SuperADD-Aligned Settings Diagnostic

Date: 2026-07-08

Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`

## Negative Evidence Intake

This branch is not a new method claim. It tests a suspected comparison
confound: whether SuperADD's non-structural choices explain the gap to our
Flow-LatentBank no-TTE + DVT branch.

Known failure basin:

- passive feature/kNN-style scoring can be absorbed by stronger preprocessing;
- F1 can collapse if maps are fragmented or thresholds are poorly calibrated;
- prior best FlowTTE DVT alpha `1.0` remained below reported SuperADD:
  `seg_AUROC_0.05=0.825207`, `seg_F1=0.468348`.

## Motivation

The user hypothesis was that SuperADD and our branch are structurally close
enough that backbone, tiling, resize, brightness augmentation, thresholding, and
post-processing may explain a large part of the observed gap.

Claim boundary:

- Claim: aligning SuperADD-like non-structural settings should improve FlowTTE.
- Evidence: all-eight MVTec AD2 `seg_AUROC_0.05` and `seg_F1`.
- Boundary: this is diagnostic only unless it uses the same dataset/split,
  reference budget, evaluator, and comparable preprocessing.

## Implementable Design

Target:

- `target_dataset=MVTec AD2 single-image`
- `data_root=/home/hunim/Volume/DATA/mvtec_ad_2`
- objects: `can,fabric,fruit_jelly,rice,vial,wallplugs,walnuts,sheet_metal`
- split: full `test_public/good,bad`
- metrics: `seg_AUROC_0.05`, `seg_F1`

Preserved FlowTTE structure:

- Flow-LatentBank no-TTE, `expansion_budget=1.0`
- NF latent distance + `density_weight=0.25`
- DVT-style support-fitted `position_mean`, `alpha=1.0`
- no register/context branch

Intended SuperADD-aligned settings:

- backbone: `dinov3_vith16plus`
- layers: `[7,15,23,31]`
- backbone resolution: `640`
- tiled extraction: `patch_size=640`, `overlap=128`
- image resize factor: `0.625` (`640/1024`)
- support brightness range: `[0.8,1.2]`

Execution adjustment:

- DINOv3-H+/16 is gated on Hugging Face and was not cached remotely.
- Online prefetch returned `401 Unauthorized`.
- Fallback run used `dinov3_vitl16` with valid layers `[5,11,17,23]` while
  preserving the SuperADD-like tiling, resize, and brightness controls.

Out of current continuous-map metric scope:

- SuperADD held-out threshold fitting and binary morphology were not mixed into
  this evaluator. The repository metric reports best-threshold F1 from
  continuous maps.

## Evaluation Alignment

Strict SuperADD comparability: `false`.

Reasons:

- H+ backbone and `[7,15,23,31]` layers could not be used without HF access.
- SuperADD reported numbers are not a pulled same-run artifact.
- binary morphology/held-out threshold calibration is outside this continuous
  map run.

Reference context:

- recorded SuperAD-16: `seg_AUROC_0.05=0.765802`, `seg_F1=0.385534`
- reported SuperADD TESTpublic: `AUROC_0.05=0.839300`, `F1=0.626113`
- previous FlowTTE DVT alpha `1.0`: `seg_AUROC_0.05=0.825207`,
  `seg_F1=0.468348`

## Code Modification / Creation

Created:

- `scripts/flow_tte_superadd_preprocess.py`
- `scripts/run_flow_tte_superadd_aligned_remote.sh`

Updated:

- `scripts/dinov3_backbone.py`
- `scripts/run_flow_tte_mvtec_ad2.py`
- `scripts/flow_tte_mvtec_ad2_core.py`
- `scripts/run_flow_tte_dvt_structural_analysis.py`
- `tests/test_dinov3_backbone.py`
- `tests/test_mvtec_classic_adapter.py`

Implemented knobs:

- `--feature-layers`
- `--backbone-resolution`
- `--tile-patch-size`
- `--tile-overlap`
- `--image-resize-factor`
- `--support-brightness-range`

## Added Code Evaluation

Passed before remote execution:

```text
uv run ruff check scripts/flow_tte_superadd_preprocess.py scripts/run_flow_tte_mvtec_ad2.py scripts/flow_tte_mvtec_ad2_core.py scripts/run_flow_tte_dvt_structural_analysis.py tests/test_mvtec_classic_adapter.py
uv run basedpyright scripts/flow_tte_superadd_preprocess.py tests/test_mvtec_classic_adapter.py tests/test_dinov3_backbone.py
uv run --with pillow --with opencv-python-headless pytest tests/test_dinov3_backbone.py tests/test_mvtec_classic_adapter.py -q
python3 -m py_compile scripts/flow_tte_superadd_preprocess.py scripts/run_flow_tte_mvtec_ad2.py scripts/flow_tte_mvtec_ad2_core.py scripts/run_flow_tte_dvt_structural_analysis.py scripts/dinov3_backbone.py
bash -n scripts/run_flow_tte_superadd_aligned_remote.sh
python3 scripts/run_flow_tte_mvtec_ad2.py --help
```

Observed test result: `22 passed`.

Remote preflight:

- host: `DSBA-Server3`
- container: `hun_fsad_tta_012`
- host GPUs `0,1,2`: idle before launch
- dataset split counts verified for all eight objects
- `/workspace/fsad_tta` and `/home/hunim/Volume/DATA/mvtec_ad_2` present
- new CLI flags visible inside the container

## Remote Execution

Full H+ run status:

- `facebook/dinov3-vith16plus-pretrain-lvd1689m` was not cached.
- `huggingface_hub.snapshot_download` returned `401 Unauthorized`.
- Therefore full SuperADD backbone/layer alignment is blocked without an
  authenticated HF token with DINOv3-H+ access.

Executed fallback:

```bash
RUN_NAME=flowtte_superadd_preproc_vitl16_all8_20260708_v1 \
BACKBONE_MODEL=dinov3_vitl16 \
FEATURE_LAYERS=5,11,17,23 \
BACKBONE_RESOLUTION=640 \
TILE_PATCH_SIZE=640 \
TILE_OVERLAP=128 \
IMAGE_RESIZE_FACTOR=0.625 \
SUPPORT_BRIGHTNESS_RANGE=0.8,1.2 \
DVT_ALPHA=1.0 \
FMAD_DINOV3_OFFLINE=1 \
bash scripts/run_flow_tte_superadd_aligned_remote.sh
```

Remote root:

- `/home/hunim/Volume/FMAD-OOD-remote/results_remote/flowtte_superadd_preproc_vitl16_all8_20260708_v1`

Local pullback:

- `results/remote_runs/dsba3/flowtte_superadd_preproc_vitl16_all8_20260708_v1`

## SuperAD Baseline and Unified Metrics

Primary metric direction: higher is better.

| comparator | comparator AUROC_0.05 | comparator F1 | method AUROC_0.05 | method F1 | delta AUROC | delta F1 | comparable |
|---|---:|---:|---:|---:|---:|---:|---|
| recorded SuperAD-16 | 0.765802 | 0.385534 | 0.762906 | 0.384584 | -0.002896 | -0.000950 | context only |
| reported SuperADD | 0.839300 | 0.626113 | 0.762906 | 0.384584 | -0.076394 | -0.241529 | context only |
| previous FlowTTE DVT a1.0 | 0.825207 | 0.468348 | 0.762906 | 0.384584 | -0.062301 | -0.083764 | same family previous |

Artifact files:

- `results/remote_runs/dsba3/flowtte_superadd_preproc_vitl16_all8_20260708_v1/summary_superadd_preproc_fallback.json`
- `results/remote_runs/dsba3/flowtte_superadd_preproc_vitl16_all8_20260708_v1/per_object_metrics.tsv`
- `results/remote_runs/dsba3/flowtte_superadd_preproc_vitl16_all8_20260708_v1/comparison_rows.tsv`

## Results and Analysis

Per-object fallback results:

| object | AUROC_0.05 | F1 |
|---|---:|---:|
| can | 0.522885 | 0.000329 |
| fabric | 0.609569 | 0.188994 |
| fruit_jelly | 0.771754 | 0.394732 |
| rice | 0.934426 | 0.741581 |
| vial | 0.704579 | 0.357529 |
| wallplugs | 0.853386 | 0.371612 |
| walnuts | 0.834148 | 0.600383 |
| sheet_metal | 0.872501 | 0.421509 |
| mean | 0.762906 | 0.384584 |

Interpretation:

- The fallback does not recover the SuperADD gap.
- It is worse than the previous DVT alpha `1.0` run by `-0.062301` AUROC and
  `-0.083764` F1.
- The degradation is not uniform: `rice`, `walnuts`, and `sheet_metal` remain
  usable, but `can` collapses almost completely and `fabric` is weak.
- This suggests that simply adding SuperADD-like tiling/resize/brightness to
  the current FlowTTE latent pipeline is not a safe improvement. The method is
  sensitive to the changed feature-map distribution, especially for some object
  geometries.

## Continuation Assessment

Strict method claim now: no.

Small continuation justified: no for this exact branch.

Reason:

- The only executed branch is a partial fallback because H+ is gated.
- The fallback is worse than the previous best FlowTTE DVT run.
- The failure is not a narrow threshold issue; continuous AUROC also drops.
- Continuing this branch would become hyperparameter tuning around a negative
  preprocessing perturbation.

Bounded future option, if needed:

- Run the true H+ setting only after HF access is available.
- Treat that as a new strict preflight, not as continuation of the ViT-L
  fallback result.

## Conclusion

Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`.

Asset preserved:

- Configurable SuperADD-style preprocessing knobs remain useful for controls.
- The run demonstrates that tiling/resize/brightness alone, when paired with
  ViT-L FlowTTE latent scoring, does not explain the SuperADD gap.

## Post-Conclusion Storage Cleanup

Remote and local chunk runs were launched with `--cleanup-maps`.

Evidence:

- each chunk contains `cleanup_evidence.txt`
- local pullback contains no `anomaly_maps/` directory
