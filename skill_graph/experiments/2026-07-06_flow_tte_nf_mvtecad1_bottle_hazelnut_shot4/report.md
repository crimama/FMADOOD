# FlowTTE NF MVTec AD1 Bottle/Hazelnut 4-Shot Diagnostic

## 1. Motivation

The user requested a classic MVTec AD1 run for the NF-based FlowTTE pipeline.
The target mechanism remains the same as the AD2 diagnostic: use a normalizing
flow before test-time memory expansion so expanded-normal patches are not
allowed to dilute normal density and collapse anomaly ranking.

Negative-evidence intake: this is not a threshold, radius, descriptor-swap, or
score-fusion retune. It stays in the allowed normal-only adaptation with
anti-absorption family. The likely failure basin remains anomaly absorption
during memory expansion.

## 2. Implementable Design

- Dataset: MVTec AD1 classic single-image.
- Data source: local `/home/hun/Volume/DATA/MVTecAD`, copied into container
  `/workspace/data/MVTecAD` because remote `/home/hunim/Volume/DATA` did not
  contain AD1.
- Objects: `bottle`, `hazelnut`.
- Split: full `test/good` plus all defect-type test folders.
- Few-shot support: first 4 `train/good` images per object.
- Backbone: DINOv2 ViT-L/14, layer mean over `[5, 11, 17, 23]`.
- Method: FlowTTE NF, 2 coupling layers, hidden multiplier 1, 3 epochs,
  density quantile 0.90, expansion budget 1.25.
- Primary metrics: `seg_AUROC_0.05`, `seg_F1`.
- Additional requested metrics: `image_AUROC`, `pixel_AUROC`,
  `image_AP`, `pixel_AP`.
- Baseline source: `BLOCKED_BASELINE`; no same-condition SuperAD AD1
  `bottle,hazelnut` artifact was found in FSAD-TTA/FMAD-OOD records.

Strict method-claim gate is closed until a same-condition AD1 SuperAD baseline
exists.

## 3. Evaluation Alignment

The run tests whether the FlowTTE NF path can produce non-collapsed AD1
segmentation maps under few-shot memory expansion. It is not a SuperAD
comparison. Direct superiority claims are invalid because the same-condition
SuperAD AD1 baseline is missing.

Metric writer alignment: AD1 classic `test/<defect>` files are paired with
`ground_truth/<defect>/*_mask.png`, and predictions are evaluated through the
same `src.post_eval.eval_segmentation` fast pixel metric path used by AD2.

## 4. Code Modification

Added:

- `scripts/flow_tte_map_metrics.py`
- `scripts/flow_tte_mvtec_classic.py`
- `scripts/run_flow_tte_mvtec_ad1.py`
- `tests/test_mvtec_classic_adapter.py`

Modified:

- `scripts/flow_tte_mvtec_ad2_core.py`: count all non-`good` anomaly folders,
  not only an AD2-style `bad` folder.

## 5. Added Code Evaluation

Local checks:

- `python3 -m py_compile scripts/flow_tte_mvtec_ad2_core.py scripts/flow_tte_map_metrics.py scripts/flow_tte_mvtec_classic.py scripts/run_flow_tte_mvtec_ad1.py tests/test_mvtec_classic_adapter.py`
- `uv run --extra dev ruff check scripts/flow_tte_mvtec_ad2_core.py scripts/flow_tte_map_metrics.py scripts/flow_tte_mvtec_classic.py scripts/run_flow_tte_mvtec_ad1.py tests/test_mvtec_classic_adapter.py`
- `uv run --extra dev pytest tests/test_mvtec_classic_adapter.py`
- `uv run --extra dev basedpyright`

All passed. Script file sizes remain under the 250 pure-LOC rule:
`core=191`, `map_metrics=130`, `classic_adapter=167`, `ad1_runner=160`,
`adapter_test=44`.

## 6. Remote Execution

Remote setup:

- Container: `hun_fsad_tta`
- Host GPU: `3`
- In-container CUDA: `0`, one RTX A6000 visible
- Remote result:
  `/workspace/results_remote/flow_tte_nf_mvtecad1_bottle_hazelnut_shot4_20260706_v2_metrics4`
- Local pullback:
  `results/remote_runs/dsba3/flow_tte_nf_mvtecad1_bottle_hazelnut_shot4_20260706_v2_metrics4`

Preflight:

- `bottle`: train/good 209, test/good 20, test/bad 63, GT 63
- `hazelnut`: train/good 391, test/good 40, test/bad 70, GT 70

Command shape:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/workspace:/workspace/fsad_tta/src:/workspace/fsad_tta/scripts \
python3 /workspace/fsad_tta/scripts/run_flow_tte_mvtec_ad1.py \
  --project-root /workspace \
  --fsad-root /workspace/fsad_tta \
  --data-root /workspace/data/MVTecAD \
  --output-root /workspace/results_remote/flow_tte_nf_mvtecad1_bottle_hazelnut_shot4_20260706_v2_metrics4 \
  --objects bottle,hazelnut \
  --shots 4 \
  --flow-epochs 3 \
  --coupling-layers 2 \
  --hidden-multiplier 1 \
  --cleanup-maps
```

## 7. Results and Analysis

FlowTTE NF metrics:

- `bottle`: `seg_AUROC_0.05=0.8732891491`, `seg_F1=0.7473030353`,
  `image_AUROC=1.0000000000`, `pixel_AUROC=0.9820547297`,
  `image_AP=1.0000000000`, `pixel_AP=0.7916529588`
- `hazelnut`: `seg_AUROC_0.05=0.9553682850`, `seg_F1=0.7730048912`,
  `image_AUROC=0.9978571429`, `pixel_AUROC=0.9953419543`,
  `image_AP=0.9988721805`, `pixel_AP=0.8011351600`
- Mean: `seg_AUROC_0.05=0.9143287170`, `seg_F1=0.7601539633`,
  `image_AUROC=0.9989285714`, `pixel_AUROC=0.9886983420`,
  `image_AP=0.9994360902`, `pixel_AP=0.7963940594`

Additional metric definitions:

- `image_AUROC`/`image_AP`: image label from good vs defect folder; image score
  is mean top 1% of the full-resolution anomaly map.
- `pixel_AUROC`/`pixel_AP`: full-FPR pixel metrics over all pixels, using a
  float16 histogram accumulator. These are not the same as partial
  `seg_AUROC_0.05`.

Interpretation: the AD1 reduced run does not show an obvious metric collapse on
these two objects. However, this is diagnostic-only evidence because the
same-condition SuperAD comparator is missing. The result supports only that the
AD1 runner/evaluator path is operational and that this FlowTTE configuration
produces nontrivial maps on `bottle,hazelnut`.

Strict method claim now: no.

One-step continuation justified: yes, but only as a baseline-completion step.
The next run must produce a same-condition AD1 SuperAD baseline before any
FlowTTE-vs-SuperAD claim.

## 8. Conclusion

Verdict: `BLOCKED_BASELINE`.

Asset retained: AD1 classic runner/evaluator, path-mapping test, and reduced
FlowTTE metrics for `bottle,hazelnut`.

Single next experiment: run or port the same-condition SuperAD AD1 baseline for
`bottle,hazelnut` using the exact same 4 support images, DINOv2 preprocessing,
and `seg_AUROC_0.05`/`seg_F1` evaluator. Hard stop for method claims if the
baseline cannot be produced or if FlowTTE loses both AUROC and F1.

## 9. Cleanup

Dense maps were removed after evaluation.

Evidence:

- Remote result root has no `anomaly_maps/` directory after cleanup.
- Local pullback has no `anomaly_maps/` directory.
- `cleanup_evidence.txt` contains `cleanup_anomaly_maps=true`.
