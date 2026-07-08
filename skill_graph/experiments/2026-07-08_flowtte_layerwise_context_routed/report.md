# Layer-wise Context-Routed Flow-LatentBank

Date: 2026-07-08
Verdict: KILL_FOR_CLAIM / NO_CONTINUE

## Negative Evidence Intake

This branch addresses a structural weakness rather than tuning a threshold or a
single object. Prior diagnostics showed that mean-fused DINO layers, a single
global memory bank, and weak score-field calibration remain bottlenecks after
moving to DINOv3-H+.

This first implementation intentionally avoids a broad sweep. It tests the
smallest structural change:

```text
per-layer feature -> per-layer Flow/NF latent bank -> per-layer distance map
-> score-level fusion
```

with optional image-context routed memory retrieval. It does not add TTE,
category-specific parameters, or new support selection.

## Motivation

SuperADD's strong setting uses layer-wise raw feature NN distance and fuses
scores after computing layer-level maps. Current FlowTTE instead normalizes and
mean-fuses multiple layers before a single NF latent projection. That can blur
the shallow/mid/deep layer roles and can make NF learn a single mixed feature
space rather than layer-specific normality.

The motivating question is whether score-level layer fusion improves spatial
ranking and F1 without relying on morphology or threshold calibration.

## Implementable Design

First-run candidate:

- Backbone: `dinov3_vith16plus`
- Layers: `[7,15,23,31]`
- Support: fixed 16-shot JSON already used by the H+ reference branch
- Denoising: DVT-style support position mean, `alpha=1.0`
- Normality mode: `layer_wise`
- Flow: one FlowTTE pipeline per layer
- Retrieval: `context_mode=top_m`, `context_source=cls`, `context_top_m=4`
- Fusion: arithmetic mean of per-layer patch score maps
- TTE: disabled with `expansion_budget=1.0`
- Score-field calibration: disabled in the first run

This is deliberately `CLS`-routed, not register-routed, because prior
all-object diagnostics showed CLS context was the stronger image-level
retrieval signal.

Control run added after the first result:

- same layer-wise score-level fusion
- `context_source=none`
- `context_mode=none`

This isolates whether the score-level layer fusion itself helps, apart from
CLS top-M routing.

## Evaluation Alignment

Target dataset: MVTec AD2 single-image.
Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`.
Objects: `can,fabric,fruit_jelly,rice,vial,wallplugs,walnuts,sheet_metal`.
Split: full `test_public/good,bad`.
Primary metrics: `seg_AUROC_0.05`, `seg_F1`.
Secondary diagnostics: high-score fragmentation metrics.

The direct internal comparator is the H+ DVT fused-layer FlowTTE reference
(`0.836739/0.527427`). Reported SuperADD is context only unless a
same-condition artifact is present.

## Code Modification / Creation

Implemented:

- `normality_mode=fused|layer_wise` in `RunConfig` and CLI
- layer-wise DINO feature extraction without mean fusion
- one FlowTTE pipeline and memory bank per layer
- image-context routed memory contexts for per-layer top-M retrieval
- score-level mean fusion of layer patch-score maps
- remote runner `scripts/run_flow_tte_layerwise_context_remote.sh`
- regression test `tests/test_flow_tte_layerwise.py`

## Added Code Evaluation

Local checks before remote execution:

- `ruff check` on changed Python files: passed
- focused pytest for adapter, layer-wise, context, score-field tests: passed
- `basedpyright`: passed
- `py_compile` and `bash -n`: passed

## Remote Execution

Remote container: `hun_fsad_tta_012`.
Host GPUs: `0,1,2`; in-container CUDA slots: `0,1,2`.
Remote result roots:

- `/workspace/results_remote/flowtte_layerwise_ctx_cls_topm4_all8_20260708_v1`
- `/workspace/results_remote/flowtte_layerwise_noctx_all8_20260708_v1`

Local pullbacks:

- `results/remote_runs/dsba3/flowtte_layerwise_ctx_cls_topm4_all8_20260708_v1`
- `results/remote_runs/dsba3/flowtte_layerwise_noctx_all8_20260708_v1`

Cleanup evidence: remote and local `anomaly_maps/` directory count is `0` for
both runs.

## SuperAD Baseline and Unified Metrics

Strict same-condition SuperAD rerun is not part of this branch. Reference
contexts:

- recorded SuperAD-16: `0.765802/0.385534`
- reported SuperADD: `0.839300/0.626113`
- directly comparable internal H+ FlowTTE reference: `0.836739/0.527427`

## Results and Analysis

Mean all-eight results:

| Method | AUROC_0.05 | F1 | dAUROC vs H+ fused | dF1 vs H+ fused | dAUROC vs SuperADD | dF1 vs SuperADD |
|---|---:|---:|---:|---:|---:|---:|
| H+ DVT fused FlowTTE baseline | 0.836739 | 0.527427 | 0.000000 | 0.000000 | -0.002561 | -0.098686 |
| Layer-wise no-context score fusion | 0.828210 | 0.499110 | -0.008529 | -0.028318 | -0.011090 | -0.127003 |
| Layer-wise CLS topM4 routed | 0.829923 | 0.508863 | -0.006816 | -0.018564 | -0.009377 | -0.117250 |

Detailed artifacts:

- `layerwise_summary.tsv`
- `layerwise_per_object_delta.tsv`

Per-object pattern:

- `fruit_jelly` improves in both layer-wise variants.
- `fabric` and `wallplugs` lose most of the mean F1.
- CLS top-M routing partially recovers the no-context layer-wise drop, but it
  still loses to the fused baseline.

Interpretation:

- Replacing feature-level layer mean fusion with score-level layer fusion is
  not a positive component in the current FlowTTE-H+ setup.
- The negative result is not explained only by CLS routing: the no-context
  control is worse than the routed variant and still below fused baseline.
- The fused normalized feature seems to be a useful stabilizer for NF latent
  ranking. Per-layer Flow banks may over-amplify layer-specific noise or
  fragment object/background evidence before score fusion.

## Continuation Assessment

Strict method claim: no. Both layer-wise variants lose to the directly
comparable H+ fused FlowTTE baseline and move farther from reported SuperADD.

Continuation: no for the tested layer-wise Flow-per-layer family. A future
layer-aware direction should avoid separate NF banks per layer and instead
consider raw SuperADD-like layer-wise NN as a hard null, or learn only a
lightweight layer weighting on top of the fused baseline.

## Conclusion

`KILL_FOR_CLAIM / NO_CONTINUE`.

Asset retained: config-driven `normality_mode=layer_wise` is useful as an
ablation/control path, but it should not be treated as the next main method
component.
