# FlowTTE NF MVTec AD1 VisionAD-Preprocess 1/2/4-Shot Sweep

## 1. Motivation

The user requested a 1/2/4-shot MVTec AD1 experiment while aligning the
available backbone and preprocessing with VisionAD
(`https://arxiv.org/pdf/2504.11895`) as much as practical inside the current
FlowTTE NF pipeline.

This is a diagnostic alignment run, not a VisionAD reproduction. The anomaly
scoring, NF projection, memory expansion, and evaluator remain FlowTTE-NF. The
aligned parts are the visual feature extraction and support preprocessing.

## 2. VisionAD Alignment

Sources checked:

- Paper: `https://arxiv.org/html/2504.11895v1`
- Official code: `https://github.com/Qiqigeww/VisionAD`
- Local clone inspected at `/tmp/VisionAD`

Alignment implemented:

- Backbone: DINOv2-register ViT-L/14 via `dinov2_vitl14_reg`.
- Input preprocessing: square resize to `448`, center crop to `392`,
  ImageNet normalization.
- Feature layers: `4..18`, matching `dade_mvtec_test.py`.
- Feature fusion: mean raw layer tokens followed by L2 normalization
  (`visionad_mean_l2`).
- Support selection: seeded random without replacement, seed `1`.
- Support augmentation: identity, rotations 90/180/270, vertical flip,
  horizontal flip.
- Image score aggregation: mean top 1% full-resolution anomaly-map pixels.

Known mismatch: VisionAD performs search-based matching, whereas this run keeps
the FlowTTE NF density/projection and memory-expansion scoring path. Therefore,
these numbers measure whether VisionAD-like feature preprocessing helps the
current FlowTTE NF method, not whether FlowTTE beats VisionAD.

## 3. Implementation

Added/changed before the run:

- `scripts/visionad_aligned_backbone.py`: DINOv2-register adapter with
  VisionAD-style resize/crop/normalize and `get_intermediate_layers`.
- `scripts/flow_tte_support.py`: support image selection, support transforms,
  and feature-fusion helpers.
- `scripts/run_flow_tte_mvtec_ad1.py`: CLI flags for backbone, preprocessing,
  feature layers, fusion, support selection, and support transforms.
- `scripts/flow_tte_mvtec_ad2_core.py`: support transform expansion and feature
  fusion routing.
- `tests/test_mvtec_classic_adapter.py`: helper coverage for seeded support
  selection, support transforms, and VisionAD-style fusion.

Remote preflight confirmed `dinov2_vitl14_reg` loads in the container and
extracts 15 feature layers with grid `(28, 28)` and token shape `(784, 1024)`.

## 4. Remote Execution

Remote setup:

- Server/container: dsba3 `hun_fsad_tta`
- Docker GPU request: host GPU `3` only
- In-container CUDA slot: `0`
- Dataset: `/workspace/data/MVTecAD`
- Objects: all 15 classic MVTec AD1 classes
- Split: full `test/good` plus all defect-type test folders
- FlowTTE NF settings: 2 coupling layers, hidden multiplier 1, 3 epochs,
  density quantile 0.90, expansion budget 1.25.
- Dense anomaly maps were deleted after metrics were computed.

Run ids:

- 1-shot:
  `flow_tte_nf_mvtecad1_all15_visionadprep_shot1_20260706_v1`
- 2-shot:
  `flow_tte_nf_mvtecad1_all15_visionadprep_shot2_20260706_v1`
- 4-shot:
  `flow_tte_nf_mvtecad1_all15_visionadprep_shot4_20260706_v1`

Local pullbacks:

- `results/remote_runs/dsba3/flow_tte_nf_mvtecad1_all15_visionadprep_shot1_20260706_v1`
- `results/remote_runs/dsba3/flow_tte_nf_mvtecad1_all15_visionadprep_shot2_20260706_v1`
- `results/remote_runs/dsba3/flow_tte_nf_mvtecad1_all15_visionadprep_shot4_20260706_v1`

## 5. Mean Results

| shot | image_AUROC | image_AP | pixel_AUROC | pixel_AP | pixel_PRO | seg_AUROC_0.05 | seg_F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.914612 | 0.958344 | 0.908276 | 0.292286 | 0.814645 | 0.713017 | 0.369093 |
| 2 | 0.928304 | 0.960326 | 0.910621 | 0.306143 | 0.823791 | 0.723755 | 0.385652 |
| 4 | 0.937006 | 0.966165 | 0.912158 | 0.312937 | 0.829695 | 0.728259 | 0.392376 |

Trend: more shots monotonically improve the mean metrics, but the gains are
small. The VisionAD-aligned 4-shot run is substantially below the previous
FlowTTE NF 4-shot run that used the earlier DINOv2 ViT-L/14 four-layer
preprocessing.

## 6. Comparison to Previous FlowTTE NF 4-Shot

Previous run:
`flow_tte_nf_mvtecad1_all15_shot4_20260706_v1_pixelpro`

| metric | previous_4shot | visionad_4shot | delta |
| --- | ---: | ---: | ---: |
| image_AUROC | 0.969631 | 0.937006 | -0.032626 |
| image_AP | 0.983877 | 0.966165 | -0.017712 |
| pixel_AUROC | 0.964028 | 0.912158 | -0.051869 |
| pixel_AP | 0.576137 | 0.312937 | -0.263200 |
| pixel_PRO | 0.936470 | 0.829695 | -0.106775 |
| seg_AUROC_0.05 | 0.842635 | 0.728259 | -0.114376 |
| seg_F1 | 0.583916 | 0.392376 | -0.191540 |

Interpretation: aligning the feature preprocessing to VisionAD does not improve
the current FlowTTE NF implementation. The largest degradation is in pixel AP
and F1, indicating weaker localization precision rather than only weaker image
ranking.

## 7. 4-Shot Per-Object Results

| object | image_AUROC | image_AP | pixel_AUROC | pixel_AP | pixel_PRO | seg_AUROC_0.05 | seg_F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bottle | 0.965079 | 0.990457 | 0.917678 | 0.425244 | 0.802360 | 0.671110 | 0.468800 |
| cable | 0.799850 | 0.879292 | 0.629307 | 0.110318 | 0.564367 | 0.577330 | 0.186048 |
| capsule | 0.954128 | 0.990810 | 0.966304 | 0.272654 | 0.915491 | 0.787779 | 0.354837 |
| carpet | 0.998796 | 0.999627 | 0.973829 | 0.505046 | 0.926109 | 0.875040 | 0.568913 |
| grid | 0.997494 | 0.999123 | 0.972572 | 0.208371 | 0.906208 | 0.825498 | 0.302310 |
| hazelnut | 0.984286 | 0.992542 | 0.951352 | 0.289007 | 0.894727 | 0.715650 | 0.371596 |
| leather | 1.000000 | 1.000000 | 0.985772 | 0.286624 | 0.967490 | 0.874278 | 0.367107 |
| metal_nut | 0.952590 | 0.989869 | 0.804483 | 0.349087 | 0.710130 | 0.576008 | 0.370828 |
| pill | 0.950900 | 0.990769 | 0.919264 | 0.271327 | 0.875242 | 0.661188 | 0.380754 |
| screw | 0.589670 | 0.782375 | 0.967740 | 0.110553 | 0.873706 | 0.777938 | 0.219838 |
| tile | 0.979076 | 0.991969 | 0.932338 | 0.473062 | 0.782520 | 0.656953 | 0.578886 |
| toothbrush | 0.991667 | 0.996970 | 0.973051 | 0.337810 | 0.863003 | 0.792462 | 0.415766 |
| transistor | 0.909583 | 0.893817 | 0.805535 | 0.205871 | 0.679887 | 0.585531 | 0.267070 |
| wood | 0.998246 | 0.999462 | 0.922261 | 0.416265 | 0.823852 | 0.732941 | 0.513521 |
| zipper | 0.983718 | 0.995398 | 0.960885 | 0.432812 | 0.860337 | 0.814175 | 0.519362 |

Weak classes by pixel AP: `cable`, `screw`, `transistor`, `grid`, `pill`.
`screw` is notable because pixel AUROC/PRO stay high while image AUROC and
pixel AP are weak, suggesting ranking and calibration mismatch rather than a
complete spatial-response failure.

## 8. Verdict

Verdict: `BLOCKED_BASELINE / NEGATIVE_ALIGNMENT_DIAGNOSTIC`.

The sweep satisfies the requested execution: all 15 MVTec AD1 classes were run
for 1/2/4-shot with VisionAD-aligned backbone/preprocessing where feasible.
However, this alignment is not beneficial for the current FlowTTE NF method
under the tested settings. It should not replace the prior four-layer DINOv2
configuration without additional adaptation.

The SuperAD/VisionAD comparison gate remains blocked: there is still no
same-condition SuperAD or VisionAD artifact in the local records using the same
support samples, evaluator, and metric suite.

## 9. Cleanup and Verification

Verification completed:

- Remote Docker inspect confirmed container exposes only host GPU `3`.
- Remote backbone preflight loaded `dinov2_vitl14_reg` and extracted the
  expected 15 feature layers.
- Local checks passed after code changes:
  - `python3 -m py_compile ...`
  - `uv run --extra dev pytest tests/test_mvtec_classic_adapter.py`
  - `uv run --extra dev pytest`: 17 passed, 1 local CUDA-driver warning
  - `uv run --extra dev ruff check ...`
  - `uv run --extra dev basedpyright`
- Remote and local result roots have no `anomaly_maps/` directories after
  cleanup.
- Retained files per run: `metrics.json`, `metrics_seed=1.json`,
  `run_manifest.json`, `run.log`, `cleanup_evidence.txt`.
