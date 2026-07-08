# FlowTTE NF MVTec AD1 Full 15-Class 4-Shot Evaluation

## 1. Motivation

The user requested a full classic MVTec AD1 run for the NF-based FlowTTE
pipeline and explicitly added `pixel_PRO` to the evaluation metrics.

The experimental question remains diagnostic: can the normalizing-flow
projection path support test-time memory expansion without obvious anomaly-map
collapse across all 15 MVTec AD1 classes? This is not yet a method-superiority
claim because no same-condition SuperAD AD1 full-dataset baseline artifact was
available in the local FSAD-TTA/FMAD-OOD records.

## 2. Implementable Design

- Dataset: MVTec AD1 classic single-image.
- Data source: local `/home/hun/Volume/DATA/MVTecAD`, copied into container
  `/workspace/data/MVTecAD`.
- Objects: `bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`,
  `leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`,
  `transistor`, `wood`, `zipper`.
- Split: full `test/good` plus every non-`good` defect test folder.
- Few-shot support: first 4 `train/good` images per object, seed 0.
- Backbone: DINOv2 ViT-L/14, layer mean over `[5, 11, 17, 23]`.
- Method: FlowTTE NF, 2 coupling layers, hidden multiplier 1, 3 epochs,
  density quantile 0.90, expansion budget 1.25.
- Primary legacy metrics: `seg_AUROC_0.05`, `seg_F1`.
- Added metrics: `image_AUROC`, `image_AP`, `pixel_AUROC`, `pixel_AP`,
  `pixel_PRO`.
- `pixel_PRO` convention: AU-PRO integrated up to false-positive rate 0.30.
- Baseline source: `BLOCKED_BASELINE`.

Strict method-claim gate remains closed until a same-condition SuperAD full AD1
baseline exists.

## 3. Evaluation Alignment

The evaluator pairs AD1 classic `test/<defect>` images with
`ground_truth/<defect>/*_mask.png`, treats `test/good` as image-level normal,
and scores all non-`good` folders as anomaly.

Image-level scores use mean top 1% over the full-resolution anomaly map.
Pixel AUROC/AP are full-pixel metrics using the float16 histogram accumulator.
`pixel_PRO` uses the project `src.post_eval.compute_pro` implementation and is
reported as area-normalized AU-PRO over FPR `[0, 0.30]`.

## 4. Code Modification

Added before the run:

- `scripts/flow_tte_map_metrics.py`
- `scripts/flow_tte_mvtec_classic.py`
- `scripts/run_flow_tte_mvtec_ad1.py`
- `tests/test_mvtec_classic_adapter.py`

Modified before this full run:

- `scripts/flow_tte_map_metrics.py`: added `pixel_PRO` and
  `pixel_PRO_max_fpr` to `MapMetricSet`.
- `scripts/flow_tte_mvtec_classic.py`: aggregate `pixel_PRO` over objects.
- `scripts/flow_tte_mvtec_ad2_core.py`: count all non-`good` anomaly folders,
  not only AD2-style `bad`.

## 5. Added Code Evaluation

Pre-run checks passed after adding `pixel_PRO`:

- `python3 -m py_compile scripts/flow_tte_map_metrics.py scripts/flow_tte_mvtec_classic.py scripts/run_flow_tte_mvtec_ad1.py tests/test_mvtec_classic_adapter.py`
- `uv run --extra dev pytest tests/test_mvtec_classic_adapter.py`
- `uv run --extra dev ruff check scripts/flow_tte_map_metrics.py scripts/flow_tte_mvtec_classic.py tests/test_mvtec_classic_adapter.py`
- `uv run --extra dev basedpyright`

Final post-run checks:

- `python3 -m py_compile scripts/flow_tte_mvtec_ad2_core.py scripts/flow_tte_map_metrics.py scripts/flow_tte_mvtec_classic.py scripts/run_flow_tte_mvtec_ad1.py scripts/run_flow_tte_mvtec_ad2.py tests/test_mvtec_classic_adapter.py`
- `uv run --extra dev pytest`: 14 passed, 1 local CUDA-driver warning.
- `uv run --extra dev ruff check scripts/flow_tte_map_metrics.py scripts/flow_tte_mvtec_ad2_core.py scripts/flow_tte_mvtec_classic.py scripts/run_flow_tte_mvtec_ad1.py scripts/run_flow_tte_mvtec_ad2.py tests/test_mvtec_classic_adapter.py`
- `uv run --extra dev basedpyright`: 0 errors.

Script file sizes remain under the 250 pure-LOC rule:
`core=191`, `map_metrics=152`, `classic_adapter=171`, `ad1_runner=160`,
`ad2_runner=167`, `adapter_test=58`.

## 6. Remote Execution

Remote setup:

- Container: `hun_fsad_tta`
- Host GPU: `3`
- In-container CUDA slot: `0`
- Visible GPU in container: one RTX A6000
- Remote result:
  `/workspace/results_remote/flow_tte_nf_mvtecad1_all15_shot4_20260706_v1_pixelpro`
- Local pullback:
  `results/remote_runs/dsba3/flow_tte_nf_mvtecad1_all15_shot4_20260706_v1_pixelpro`

Preflight counts:

| object | train/good | test/good | test/defect | gt masks |
| --- | ---: | ---: | ---: | ---: |
| bottle | 209 | 20 | 63 | 63 |
| cable | 224 | 58 | 92 | 92 |
| capsule | 219 | 23 | 109 | 109 |
| carpet | 280 | 28 | 89 | 89 |
| grid | 264 | 21 | 57 | 57 |
| hazelnut | 391 | 40 | 70 | 70 |
| leather | 245 | 32 | 92 | 92 |
| metal_nut | 220 | 22 | 93 | 93 |
| pill | 267 | 26 | 141 | 141 |
| screw | 320 | 41 | 119 | 119 |
| tile | 230 | 33 | 84 | 84 |
| toothbrush | 60 | 12 | 30 | 30 |
| transistor | 213 | 60 | 40 | 40 |
| wood | 247 | 19 | 60 | 60 |
| zipper | 240 | 32 | 119 | 119 |

Command shape:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/workspace:/workspace/fsad_tta/src:/workspace/fsad_tta/scripts \
python3 /workspace/fsad_tta/scripts/run_flow_tte_mvtec_ad1.py \
  --data-root /workspace/data/MVTecAD \
  --output-root /workspace/results_remote/flow_tte_nf_mvtecad1_all15_shot4_20260706_v1_pixelpro \
  --project-root /workspace \
  --fsad-root /workspace/fsad_tta \
  --objects bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,screw,tile,toothbrush,transistor,wood,zipper \
  --shots 4 \
  --seed 0 \
  --device cuda \
  --flow-epochs 3 \
  --coupling-layers 2 \
  --hidden-multiplier 1 \
  --density-quantile 0.90 \
  --expansion-budget 1.25 \
  --top-percent 0.01 \
  --query-chunk-size 512 \
  --pro-integration-limit 0.05 \
  --cleanup-maps
```

## 7. Results and Analysis

Mean metrics:

| metric | value |
| --- | ---: |
| image_AUROC | 0.969631 |
| image_AP | 0.983877 |
| pixel_AUROC | 0.964028 |
| pixel_AP | 0.576137 |
| pixel_PRO | 0.936470 |
| seg_AUROC_0.05 | 0.842635 |
| seg_F1 | 0.583916 |

Per-object metrics:

| object | image_AUROC | image_AP | pixel_AUROC | pixel_AP | pixel_PRO | seg_AUROC_0.05 | seg_F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bottle | 1.000000 | 1.000000 | 0.982064 | 0.791725 | 0.961400 | 0.873337 | 0.747342 |
| cable | 0.895802 | 0.934940 | 0.897880 | 0.372466 | 0.843408 | 0.697051 | 0.397866 |
| capsule | 0.963702 | 0.992224 | 0.983121 | 0.526258 | 0.976609 | 0.879545 | 0.522181 |
| carpet | 0.999599 | 0.999875 | 0.990987 | 0.615905 | 0.977419 | 0.926939 | 0.664182 |
| grid | 1.000000 | 1.000000 | 0.995158 | 0.569042 | 0.973592 | 0.951842 | 0.566046 |
| hazelnut | 0.997857 | 0.998872 | 0.995340 | 0.801185 | 0.982495 | 0.955374 | 0.773050 |
| leather | 1.000000 | 1.000000 | 0.992781 | 0.500439 | 0.989057 | 0.926957 | 0.476980 |
| metal_nut | 0.993646 | 0.998538 | 0.910406 | 0.573365 | 0.905299 | 0.657138 | 0.553488 |
| pill | 0.971086 | 0.994860 | 0.930499 | 0.538581 | 0.969493 | 0.789411 | 0.560589 |
| screw | 0.807952 | 0.930304 | 0.985544 | 0.502750 | 0.945260 | 0.917217 | 0.527008 |
| tile | 1.000000 | 1.000000 | 0.972571 | 0.650201 | 0.930539 | 0.783737 | 0.712892 |
| toothbrush | 0.966667 | 0.988792 | 0.987261 | 0.495978 | 0.959798 | 0.877811 | 0.542138 |
| transistor | 0.951667 | 0.920838 | 0.897686 | 0.433025 | 0.728564 | 0.689468 | 0.441019 |
| wood | 0.996491 | 0.998916 | 0.963740 | 0.694070 | 0.964839 | 0.853400 | 0.677036 |
| zipper | 1.000000 | 1.000000 | 0.975374 | 0.577061 | 0.939281 | 0.860294 | 0.596915 |

Interpretation: the full AD1 run produces high image-level performance and
generally high pixel AUROC/PRO, but the pixel AP and F1 reveal weaker
localization precision on several classes, especially `cable`, `transistor`,
`toothbrush`, `leather`, and `screw`. The result is useful as a full-dataset
FlowTTE NF diagnostic, but it cannot establish superiority over SuperAD without
a same-condition comparator.

## 8. Conclusion

Verdict: `BLOCKED_BASELINE`.

The full MVTec AD1 FlowTTE NF 4-shot pipeline is operational with the requested
metrics, including `pixel_PRO`. The strongest immediate evidence is that the
NF-based path does not collapse to unusable maps across the full AD1 object set.
The weakest evidence remains comparative: no same-condition SuperAD AD1
full-dataset result was found locally, so direct comparison is still blocked.

Next constrained experiment: run or port a same-condition SuperAD AD1 baseline
for all 15 classes, with the same support selection, DINOv2 preprocessing, map
resolution, and metric suite.

## 9. Cleanup

Dense maps were removed after evaluation.

Evidence:

- Remote result root has no `anomaly_maps/` directory after cleanup.
- Local pullback has no `anomaly_maps/` directory.
- `cleanup_evidence.txt` contains `cleanup_anomaly_maps=true`.
- Retained files: `metrics.json`, `metrics_seed=0.json`, `run_manifest.json`,
  `cleanup_evidence.txt`.
