# FlowTTE DVT H+ Backbone-Only Ablation

Date: 2026-07-08
Verdict: KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC

## Negative Evidence Intake

This is not a new method claim. It is a controlled ablation of the current best
FlowTTE branch: Flow-LatentBank no-TTE with DVT-style position-mean denoising
at `alpha=1.0`. The only intended structural change is the DINOv3 backbone
from `dinov3_vitl16` layers `[5,11,17,23]` to SuperADD's
`dinov3_vith16plus` layers `[7,15,23,31]`.

Likely failure basin: passive feature/backbone upgrade can improve mean
metrics without explaining the remaining SuperADD F1 gap. This run therefore
tests how much of the gap is backbone quality rather than postprocessing,
threshold calibration, high-resolution tiling, or SuperADD's raw-feature NN
scoring design.

## Motivation

The previous best current method was:

```text
Flow-LatentBank no-TTE
+ DVT-style support/query position denoise alpha=1.0
+ DINOv3-L/16 layers [5,11,17,23]
+ fixed 16-shot support JSON
+ latent NN distance + 0.25 density
```

It reached mean `seg_AUROC_0.05=0.825207` and `seg_F1=0.468348`, still below
reported SuperADD (`0.839300` / `0.626113`). Since SuperADD uses
DINOv3-H+/16, this experiment isolates the backbone-only contribution before
adding other SuperADD settings.

## Implementable Design

- Dataset: MVTec AD2 single-image.
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`.
- Objects: all eight public objects:
  `can, fabric, fruit_jelly, rice, vial, wallplugs, walnuts, sheet_metal`.
- Split: full `test_public/good,bad`.
- Reference policy: same fixed support JSON used by the prior DINOv3-L DVT
  alpha `1.0` run:
  `/workspace/fsad_tta/skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json`.
- Candidate: FlowTTE DVT alpha `1.0` with `dinov3_vith16plus`,
  feature layers `[7,15,23,31]`.
- No TTE: `expansion_budget=1.0`.
- No register/CLS context: `context-source=none`, `context-mode=none`.
- NF setting: flow latent projection, `flow_epochs=3`, `density_weight=0.25`.
- Not included: SuperADD high-resolution tiled preprocessing, support
  brightness augmentation, layer-wise raw-feature NN averaging, threshold
  calibration, or morphology.
- Primary metrics: `seg_AUROC_0.05`, `seg_F1`.

Strict claim gate: cannot claim parity with SuperADD unless both AUROC and F1
match or exceed the reported context, and the remaining setting differences
are acknowledged. This run is diagnostic because only the backbone is aligned.

Continuation gate: continue only if H+ improves over the prior DINOv3-L best
and closes a meaningful part of the SuperADD gap without catastrophic broad
object harm.

## Evaluation Alignment

The H+ candidate and previous DINOv3-L best are directly comparable on dataset,
object set, support paths, no-TTE policy, DVT alpha, score mode, and evaluator.
The comparison to reported SuperADD is contextual rather than strict
same-condition because SuperADD still differs in preprocessing, scoring,
thresholding, and binary postprocessing.

## Code Modification / Creation

Modified:

- `scripts/run_flow_tte_dvt_denoising_all8_remote.sh`

The runner now accepts `FEATURE_LAYERS` as an environment variable and records
it in `remote_run_complete.txt`. This allows the same launcher to run
`dinov3_vitl16` with `[5,11,17,23]` or `dinov3_vith16plus` with
`[7,15,23,31]`.

Result artifacts:

- `results/remote_runs/dsba3/flowtte_dvt_hplus_backbone_only_all8_20260708_v1/summary_hplus_backbone_only.json`
- `results/remote_runs/dsba3/flowtte_dvt_hplus_backbone_only_all8_20260708_v1/per_object_metrics.tsv`
- `results/remote_runs/dsba3/flowtte_dvt_hplus_backbone_only_all8_20260708_v1/comparison_rows.tsv`

## Added Code Evaluation

- `bash -n scripts/run_flow_tte_dvt_denoising_all8_remote.sh` passed.
- `python3 -m py_compile scripts/run_flow_tte_mvtec_ad2.py scripts/flow_tte_mvtec_ad2_core.py scripts/dinov3_backbone.py` passed.
- DINOv3-H+/16 offline load succeeded inside `hun_fsad_tta_012` from the
  copied Hugging Face model cache:
  `facebook/dinov3-vith16plus-pretrain-lvd1689m`.

Remote H+ access note: the remote container did not have a Hugging Face token.
The model snapshot was downloaded locally with the existing local credential
and only the cached model files were copied to dsba3. No token was copied to
the remote container.

## Remote Execution

Executed on dsba3 in container `hun_fsad_tta_012` with host GPUs `0,1,2`:

```bash
cd /workspace/fsad_tta
RUN_NAME=flowtte_dvt_hplus_backbone_only_all8_20260708_v1 \
BACKBONE_MODEL=dinov3_vith16plus \
FEATURE_LAYERS=7,15,23,31 \
DVT_ALPHA=1.0 \
FMAD_DINOV3_OFFLINE=1 \
bash scripts/run_flow_tte_dvt_denoising_all8_remote.sh
```

Local pullback:

```text
results/remote_runs/dsba3/flowtte_dvt_hplus_backbone_only_all8_20260708_v1
```

## SuperAD Baseline and Unified Metrics

| Method | mean seg_AUROC_0.05 | mean seg_F1 | Note |
|---|---:|---:|---|
| FlowTTE DVT a1.0 DINOv3-L/16 | 0.825207 | 0.468348 | previous best current method |
| FlowTTE DVT a1.0 DINOv3-H+/16 backbone-only | 0.836739 | 0.527427 | this run |
| SuperAD-16 recorded context | 0.765802 | 0.385534 | context baseline |
| SuperADD reported context | 0.839300 | 0.626113 | reported context |

Unified deltas:

| Comparator | delta AUROC_0.05 | delta F1 |
|---|---:|---:|
| vs previous DINOv3-L best | +0.011532 | +0.059079 |
| vs recorded SuperAD-16 context | +0.070937 | +0.141893 |
| vs reported SuperADD context | -0.002561 | -0.098686 |

## Results and Analysis

Object-level metrics:

| Object | H+ AUROC_0.05 | H+ F1 | delta AUROC vs L | delta F1 vs L |
|---|---:|---:|---:|---:|
| can | 0.560495 | 0.000634 | -0.089451 | -0.001945 |
| fabric | 0.968227 | 0.697427 | +0.069797 | +0.176069 |
| fruit_jelly | 0.781873 | 0.481412 | -0.017147 | -0.036295 |
| rice | 0.947121 | 0.711554 | -0.002631 | +0.018595 |
| vial | 0.746292 | 0.434360 | +0.032257 | +0.044340 |
| wallplugs | 0.908028 | 0.631539 | +0.056606 | +0.199311 |
| walnuts | 0.890238 | 0.733291 | +0.039925 | +0.060847 |
| sheet_metal | 0.891639 | 0.529204 | +0.002900 | +0.011712 |

H+ improves over the previous best on 5/8 objects by AUROC and 6/8 objects by
F1. The improvement is large on `fabric`, `wallplugs`, `walnuts`, and `vial`,
which shows that SuperADD's larger backbone explains a real part of the
performance gap.

The remaining pattern is also clear:

- AUROC is now nearly tied with reported SuperADD: `0.836739` vs `0.839300`.
- F1 still trails SuperADD by `0.098686`.
- `can` remains essentially F1-collapsed, and `fruit_jelly` is worse than the
  DINOv3-L best.

This supports the user's hypothesis that some SuperADD advantage is setting or
preprocessing driven rather than purely structural. Backbone alignment closes
most of the AUROC gap, but the F1 gap remains too large to attribute only to
backbone size. The next likely contributors are operating-point calibration,
binary morphology, and high-resolution localization.

## Continuation Assessment

Does this support a strict method claim now? No. It is still below reported
SuperADD in F1 and is not a fully same-condition SuperADD comparison.

Does this justify a small next diagnostic? Yes. The H+ backbone gives a
meaningful positive gradient and nearly matches SuperADD AUROC. The next
single-step diagnostic should keep this H+ backbone fixed and add only one
SuperADD non-structural component at a time, starting with threshold/morphology
on saved continuous maps or a cleanup-safe rerun that emits compact binary
diagnostics.

Hard-stop condition: if H+ plus identical threshold/morphology still leaves the
F1 gap near `0.10` or keeps `can` at near-zero F1, the remaining gap is not
just postprocessing and the Flow latent projection/scoring should be audited
against SuperADD raw layer-wise NN.

## Conclusion

Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.

Backbone-only SuperADD alignment is a strong positive diagnostic:

```text
0.825207 / 0.468348  ->  0.836739 / 0.527427
```

It moves FlowTTE-DVT close to SuperADD on AUROC but not on F1. The main
remaining gap is therefore likely in thresholding, morphology, high-resolution
localization, or the raw layer-wise NN scoring path, not just the H+ backbone.

## Post-Conclusion Storage Cleanup

The run used `--cleanup-maps`. Verification after local pullback:

- Local `anomaly_maps/` directories under the result root: `0`.
- Remote `anomaly_maps/` directories under the result root: `0`.
- Remote GPU status after completion: GPUs `0,1,2` each reported `1 MiB` used
  and `0%` utilization.
- `remote_run_complete.txt` is present locally and remotely.
