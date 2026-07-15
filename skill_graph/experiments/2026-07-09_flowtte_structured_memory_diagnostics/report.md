# FlowTTE Structured Memory Diagnostics

Date: 2026-07-09
Verdict: KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC

## Negative Evidence Intake

This branch tests whether flat support memory is a structural bottleneck in the
current FlowTTE score field. It is not a class-specific threshold, morphology,
or postprocessing retune. The failure basin is still close to memory-bank kNN,
so the result is diagnostic only unless it beats the current H+ DVT baseline
and preserves all-object no-harm.

## Motivation

Current FlowTTE uses a single flat support latent bank:

```text
DINOv3-H+ patch feature
-> DVT position_mean alpha=1.0
-> NF latent projection
-> flat support latent NN distance
```

The target question was whether score-field quality is limited by flat memory
retrieval. Two context sources were implemented:

- `feature_avg3_ch16`: local 3x3 feature-neighborhood context compressed to 16
  channel groups for soft context penalty.
- `image_feature_mean_ch16`: image-level mean feature compressed to 16 channel
  groups for support image top-M sub-bank routing.

## Implementable Design

Common settings:

- Dataset: MVTec AD2 single-image, all 8 public objects.
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`.
- Split: full `test_public/good,bad`.
- Container: `hun_fsad_tta_012`.
- Host GPUs: `0,1,2`.
- Backbone: `dinov3_vith16plus`.
- Layers: `[7,15,23,31]`.
- Support: fixed SuperAD-16 JSON reference.
- DVT: `position_mean`, `alpha=1.0`.
- No-TTE: `expansion_budget=1.0`.
- Scoring: latent distance + density weight `0.25`.

Execution fixes added before the all-8 run:

- Feature-derived memory contexts are now derived from already extracted feature
  maps instead of triggering a second DINOv3-H+ extraction pass.
- `TorchMemoryBank` context grouping is now lazy; grouping is only built for
  `top_m`, not for `soft_penalty`.
- `ScoreCalibration.fit` accepts `calibration_sample_size`; all-8 run used
  deterministic cap `4096` for leave-one-out calibration only. Full support
  memory remains intact.

## Evaluation Alignment

This is comparable to the current H+ DVT baseline for diagnostic purposes:
same dataset, fixed reference set, backbone, layers, DVT alpha, split, and
metrics. It is not a strict SuperADD claim because SuperADD still differs in
postprocessing, thresholding, and exact implementation details.

Primary metrics:

- `seg_AUROC_0.05`
- `seg_F1`

Baseline source:

`results/remote_runs/dsba3/flowtte_dvt_hplus_backbone_only_all8_20260708_v1`

## Code Modification / Creation

Changed files:

- `src/flow_tte/config.py`
- `src/flow_tte/scoring.py`
- `src/flow_tte/memory.py`
- `scripts/flow_tte_mvtec_ad2_core.py`
- `scripts/run_flow_tte_mvtec_ad2.py`
- `scripts/run_flow_tte_dvt_denoising_all8_remote.sh`
- `tests/test_flow_tte_context.py`
- `tests/test_flow_tte_layerwise.py`

New context sources:

- `feature_avg3`
- `feature_avg3_ch16`
- `feature_avg3_residual`
- `image_feature_mean`
- `image_feature_mean_ch16`

## Added Code Evaluation

Local checks passed:

```text
python3 -m py_compile src/flow_tte/config.py src/flow_tte/scoring.py src/flow_tte/memory.py scripts/flow_tte_mvtec_ad2_core.py scripts/run_flow_tte_mvtec_ad2.py
python3 -m pytest tests/test_flow_tte_layerwise.py tests/test_flow_tte_context.py tests/test_flow_tte.py -q
```

Result:

```text
25 passed
```

Manual QA gate:

- Remote `can` smoke with `feature_avg3_ch16 + soft_penalty + identity`
  completed after fixing lazy grouping and calibration cap.
- Remote `can` smoke with `image_feature_mean_ch16 + top_m=4 + flow`
  completed and produced evaluator metrics.

## Remote Execution

All-8 executed command family:

```text
RUN_NAME=flowtte_structmem_image_mean_ch16_topm4_cal4096_all8_20260709_v1
BACKBONE_MODEL=dinov3_vith16plus
FEATURE_LAYERS=7,15,23,31
MEMORY_CONTEXT_SOURCE=image_feature_mean_ch16
CONTEXT_MODE=top_m
CONTEXT_TOP_M=4
CALIBRATION_SAMPLE_SIZE=4096
FLOW_TRANSFORM_MODE=flow
bash scripts/run_flow_tte_dvt_denoising_all8_remote.sh
```

Remote result:

`/workspace/results_remote/flowtte_structmem_image_mean_ch16_topm4_cal4096_all8_20260709_v1`

Local pullback:

`results/remote_runs/dsba3/flowtte_structmem_image_mean_ch16_topm4_cal4096_all8_20260709_v1`

Cleanup evidence:

- Remote `anomaly_maps` directories after completion: `0`.
- Local pullback `anomaly_maps` directories after pull: `0`.

## SuperAD Baseline and Unified Metrics

Reported SuperADD context:

- `seg_AUROC_0.05=0.839300`
- `seg_F1=0.626113`

Current H+ DVT baseline:

- `seg_AUROC_0.05=0.836739`
- `seg_F1=0.527427`

Structured memory result:

- `seg_AUROC_0.05=0.832426`
- `seg_F1=0.524593`

Unified comparison:

| Comparator | AUROC_0.05 | F1 | Delta AUROC | Delta F1 |
|---|---:|---:|---:|---:|
| H+ DVT baseline | 0.836739 | 0.527427 | - | - |
| image mean top-M structured memory | 0.832426 | 0.524593 | -0.004313 | -0.002835 |
| reported SuperADD | 0.839300 | 0.626113 | -0.006874 | -0.101520 |

## Results and Analysis

Per-object result against H+ DVT baseline:

| Object | Base AUROC | Base F1 | Method AUROC | Method F1 | Delta AUROC | Delta F1 |
|---|---:|---:|---:|---:|---:|---:|
| can | 0.560495 | 0.000634 | 0.552458 | 0.000523 | -0.008037 | -0.000111 |
| fabric | 0.968227 | 0.697427 | 0.971664 | 0.723476 | +0.003437 | +0.026049 |
| fruit_jelly | 0.781873 | 0.481412 | 0.771338 | 0.459572 | -0.010535 | -0.021840 |
| rice | 0.947121 | 0.711554 | 0.941859 | 0.708799 | -0.005262 | -0.002755 |
| vial | 0.746292 | 0.434360 | 0.737323 | 0.400144 | -0.008969 | -0.034216 |
| wallplugs | 0.908028 | 0.631539 | 0.908862 | 0.637696 | +0.000835 | +0.006157 |
| walnuts | 0.890238 | 0.733291 | 0.885238 | 0.732766 | -0.004999 | -0.000525 |
| sheet_metal | 0.891639 | 0.529204 | 0.890666 | 0.533766 | -0.000972 | +0.004562 |
| mean | 0.836739 | 0.527427 | 0.832426 | 0.524593 | -0.004313 | -0.002835 |

Interpretation:

- Image-level support sub-bank routing is not a mean-metric improvement.
- Positive F1 movement appears on `fabric`, `wallplugs`, and `sheet_metal`,
  but `vial` and `fruit_jelly` losses dominate.
- Full-dimensional patch neighborhood context was computationally invalid for
  all-8 because context grouping and calibration became the bottleneck.
- After lazy grouping and calibration cap, the code path is usable, but the
  mechanism does not solve the structural score-field issue.

## Continuation Assessment

Strict method claim:

No. The method loses to the current H+ DVT baseline and remains far below
SuperADD F1.

Diagnostic continuation:

Yes, but not this exact branch. The useful asset is the execution fix:
feature-derived context sources are now feasible, and memory grouping no longer
punishes soft-penalty contexts. The next structural test should avoid simple
image-level routing and instead target score-field calibration or foreground
separation with no all-object harm gate.

## Conclusion

Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`

Do not continue `image_feature_mean_ch16 top-M` as a method candidate. Keep the
lazy grouping and calibration cap infrastructure for future structured-memory
diagnostics.

Next constrained experiment:

Test support-stat score calibration or foreground/background score suppression
using the now-fixed execution path. Hard-stop if mean AUROC/F1 does not beat
the H+ DVT baseline or if `vial`/`fruit_jelly` losses repeat.

## Post-Conclusion Storage Cleanup

Remote cleanup:

```text
find /workspace/results_remote/flowtte_structmem_image_mean_ch16_topm4_cal4096_all8_20260709_v1 -type d -name anomaly_maps | wc -l
0
```

Local cleanup:

```text
find results/remote_runs/dsba3/flowtte_structmem_image_mean_ch16_topm4_cal4096_all8_20260709_v1 -type d -name anomaly_maps | wc -l
0
```
