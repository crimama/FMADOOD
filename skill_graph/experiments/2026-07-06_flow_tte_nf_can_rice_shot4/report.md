# FlowTTE NF Can/Rice 4-Shot Diagnostic

## 1. Motivation

FlowTTE targets the user-defined failure mode: TTE memory-bank expansion can
increase normality volume while reducing density, causing ranking collapse. The
candidate mechanism is a normalizing-flow projection before memory scoring and
expansion, intended to compress normal/expanded-normal distance while preserving
anomaly separation.

Negative-evidence intake: this is not a score-threshold or kNN-radius retune.
It is in the allowed family `normal-only adaptation with anti-absorption`.
The likely failure basin remains anomaly absorption/collapse during expansion.

## 2. Implementable Design

- Dataset: MVTec AD2 single-image.
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`.
- Objects: `can`, `rice`.
- Split: full `test_public/good,bad`.
- Few-shot support: first 4 `train/good` images per object.
- Backbone: DINOv2 ViT-L/14, layer mean over `[5, 11, 17, 23]`.
- Method: FlowTTE NF, 2 coupling layers, hidden multiplier 1, 3 epochs,
  density quantile 0.90, expansion budget 1.25.
- Primary metrics: `seg_AUROC_0.05`, `seg_F1`.
- Baseline source: recorded SuperAD-16 4-object metrics at
  `configs/baselines/recorded_superad16_mvtec_ad2_4object_metrics.json`.

Strict claim gate is not open because the run is 2-object, 4-shot, first-N
support, and does not share SuperAD-16 selected references. It is metric
diagnostic only.

## 3. Evaluation Alignment

The candidate and recorded baseline share dataset, split, metric evaluator, and
the `can,rice` object subset. They do not share reference budget or reference
selection policy. Therefore `comparable_for_metric_diagnostic=true` and
`strict_table1_claim_comparable=false`.

## 4. Code Modification

Added:

- `scripts/run_flow_tte_mvtec_ad2.py`
- `scripts/flow_tte_mvtec_ad2_core.py`
- `scripts/__init__.py`

The runner writes full-resolution TIFF maps, evaluates them with the existing
FMAD `eval_segmentation` path, then removes dense `anomaly_maps/`.

## 5. Added Code Evaluation

Local checks:

- `python3 -m py_compile scripts/run_flow_tte_mvtec_ad2.py scripts/flow_tte_mvtec_ad2_core.py scripts/__init__.py`
- `uv run --with ruff ruff check scripts/run_flow_tte_mvtec_ad2.py scripts/flow_tte_mvtec_ad2_core.py scripts/__init__.py`
- `uv run --extra dev basedpyright`

All passed. Script file sizes are below the 250 pure-LOC rule:
`run=167`, `core=186`.

## 6. Remote Execution

Remote setup:

- Container: `hun_fsad_tta`
- Host GPU: `3`
- In-container CUDA: `0`, one RTX A6000 visible
- Remote result: `/workspace/results_remote/flow_tte_nf_can_rice_shot4_20260706_v1`
- Local pullback: `results/remote_runs/dsba3/flow_tte_nf_can_rice_shot4_20260706_v1`

Preflight:

- `can`: train/good 412, test/good 72, test/bad 90, GT 90
- `rice`: train/good 313, test/good 42, test/bad 90, GT 90

Command shape:

```bash
PYTHONPATH=/workspace:/workspace/fsad_tta/src \
python3 /workspace/fsad_tta/scripts/run_flow_tte_mvtec_ad2.py \
  --project-root /workspace \
  --fsad-root /workspace/fsad_tta \
  --data-root /home/hunim/Volume/DATA/mvtec_ad_2 \
  --output-root /workspace/results_remote/flow_tte_nf_can_rice_shot4_20260706_v1 \
  --objects can,rice \
  --shots 4 \
  --flow-epochs 3 \
  --coupling-layers 2 \
  --hidden-multiplier 1 \
  --cleanup-maps
```

## 7. Results and Analysis

FlowTTE NF metrics:

- `can`: `seg_AUROC_0.05=0.6306599028`, `seg_F1=0.0034232527`
- `rice`: `seg_AUROC_0.05=0.9008846449`, `seg_F1=0.6277542086`
- Mean: `seg_AUROC_0.05=0.7657722739`, `seg_F1=0.3155887306`

Recorded SuperAD-16 context on the same two objects:

- Mean: `seg_AUROC_0.05=0.7576736900`, `seg_F1=0.3337097354`
- Delta FlowTTE vs SuperAD: `AUROC_0.05=+0.0080985839`,
  `F1=-0.0181210048`
- Win count: AUROC 1/2, F1 1/2

Interpretation: AUROC has a weak positive mean movement, mainly not enough for
a method claim, while F1 drops and `can` F1 is effectively collapsed. This is
consistent with partial ranking improvement without usable segmentation
coverage on at least one object.

## 8. Conclusion

Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.

No strict method claim is supported. A bounded continuation is justified because
mean AUROC did not collapse against recorded SuperAD context and `rice` remains
nontrivial, but the next run must isolate whether the issue is support choice or
TTE absorption.

Single next experiment: run `can,rice` with SuperAD-style DINO CLS coreset
support and compare `expand=true` against `expand=false` under the same 4-shot
budget. Hard stop if `can` F1 remains near zero or static no-expansion matches
the expanded branch.

## 9. Cleanup

Dense maps were removed after evaluation.

Evidence:

- Remote `find ... -name anomaly_maps` returned `0`.
- `cleanup_evidence.txt` contains `cleanup_anomaly_maps=true`.
