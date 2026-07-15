# FlowTTE H+ DVT with SuperAD Rotation-8 Support Augmentation

Date: 2026-07-10

Status: `COMPLETED`

Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`

## Motivation

The strongest current FlowTTE configuration uses DINOv3 H+/16 layers
`[7, 15, 23, 31]`, fused layer-normalized features, support-position DVT, and
the patch-wise MLP normalizing flow. It reaches mean `seg_AUROC_0.05=0.836739`
and `seg_F1=0.527427` on all eight MVTec AD2 public objects. The remaining
reported SuperADD-context gap is small in AUROC but large in F1, while the
current FlowTTE support path uses only the identity view.

This diagnostic tests one bounded claim: whether the exact eight support
rotations used by the repository's prior SuperAD path improve the existing H+
DVT MLP pipeline without changing its backbone, feature fusion, DVT, flow,
scoring, support image paths, or evaluator.

- Claim: support rotation coverage improves anomaly score ranking or mask
  quality in the existing H+ DVT MLP pipeline.
- Evidence: all-eight-object deltas against the existing identity-only H+ DVT
  run under the same FlowTTE evaluator.
- Boundary: this is an internal augmentation ablation, not a same-condition
  SuperADD method comparison.
- Positioning: historical SuperADD numbers are context only because its raw
  maps and same-run evaluator artifact are unavailable.

This is not a repeat of the failed layer-wise-flow or Transformer-flow branches.
The scorer remains the strongest fused patch-wise MLP flow; only support image
augmentation changes.

## Implementable Design

### Fixed controls

- Dataset: MVTec AD2 single-image, full `test_public/good,bad`.
- Objects: `can`, `fabric`, `fruit_jelly`, `rice`, `vial`, `wallplugs`,
  `walnuts`, `sheet_metal`.
- Reference selection pool: full `train/good`.
- Candidate reference policy: the exact fixed 16 paths used by the H+ baseline.
- Backbone: `dinov3_vith16plus`, layers `[7,15,23,31]`.
- Feature fusion: `layer_norm_mean`.
- DVT: pooled support-position mean, alpha `1.0`.
- Normality model: fused patch-wise MLP flow.
- Flow and scoring hyperparameters: unchanged from
  `run_flow_tte_dvt_denoising_all8_remote.sh`, except both experimental arms
  use a matched `calibration_sample_size=4096` computational cap.
- Brightness augmentation: disabled with range `[1.0,1.0]`.
- Binary category-specific postprocessing: out of scope.

### Single changed variable

Each of the fixed 16 support images is expanded to the exact prior SuperAD
rotation set:

```text
0, 45, 90, 135, 180, 225, 270, 315 degrees
```

The implementation matches `src/utils.py::rotate_image`: OpenCV
`getRotationMatrix2D`, fixed input canvas, bilinear interpolation, and
`BORDER_DEFAULT`. This yields 128 support views per object. DVT remains the
existing pooled fit over all collected support views; transform-specific DVT is
explicitly excluded because it would introduce a second method change.

Runners:

```text
scripts/run_flow_tte_hplus_rotation8_ablation_remote.sh
  -> rotation-8 `can` runtime smoke
  -> identity control with calibration cap 4096
  -> rotation-8 candidate with calibration cap 4096
```

Run name:

```text
flowtte_hplus_dvt_identity_cal4096_all8_20260710_v1
flowtte_hplus_dvt_superad_rotation8_all8_20260710_v1
```

The matched identity rerun is required because full leave-one-out calibration
would grow quadratically from roughly 33,600--61,152 patches per object to
268,800--489,216. Comparing a capped candidate directly to the uncapped
historical baseline would confound augmentation with calibration sampling.

## Pre-registered Gates

Primary metrics are decimal macro means over all eight objects:
`seg_AUROC_0.05` and best-threshold pixel `seg_F1`.

Strict keep gate against the matched identity-only H+ DVT control:

- `delta_seg_AUROC_0.05 >= +0.003` and `delta_seg_F1 >= 0`, or
- `delta_seg_F1 >= +0.010` and `delta_seg_AUROC_0.05 >= -0.002`;
- no more than two of the seven non-`can` objects may regress by over `0.02`
  F1;
- `rice` acts as a strong no-harm control and must not regress by over `0.01`
  AUROC or `0.02` F1.

Continuation-only gate:

- at least three non-`can` objects improve by `>=0.01` F1 while mean AUROC
  stays within `-0.003`; or
- the result exposes a coherent rotation-sensitive object group that justifies
  exactly one transform-aware DVT follow-up.

Kill gate:

- both mean metrics fail to improve, or
- either mean AUROC regresses by more than `0.005` or mean F1 regresses by more
  than `0.010`, or
- gains are concentrated in at most two objects while strong controls collapse.

## Evaluation Alignment

The candidate is internally comparable to the identity-only H+ DVT baseline:
same dataset, split, eight objects, exact 16 source paths, backbone, feature
layers, DVT mode, flow, scorer, and evaluator. The effective support-view budget
is intentionally different (`16` versus `128`) because augmentation is the
tested variable.

It is not strictly comparable to reported SuperADD context. The required
unified output fields will therefore be reported as:

```text
superadd_seg_AUROC_0.05=0.839300
superadd_seg_F1=0.626113
method_seg_AUROC_0.05=<candidate>
method_seg_F1=<candidate>
delta_vs_superadd=<context-only delta>
comparable=false
```

The decisive comparator for this diagnostic is the same-evaluator matched
identity-control run produced by the ablation controller. The uncapped historical
H+ DVT run remains a secondary drift check at:

```text
results/remote_runs/dsba3/flowtte_dvt_hplus_backbone_only_all8_20260708_v1
```

## Runtime-Safe Implementation

The 128-view candidate exposes up to 489,216 support patch rows for `can`.
The previous flat-input path interpreted this as one image and performed one
unbounded autograd forward, so the implementation now microbatches flat support
rows while preserving the previous full-batch optimization semantics:

1. a no-grad chunked pass identifies the global tail top-k;
2. the exact global mean/tail NLL weights are formed;
3. chunk losses accumulate gradients without stepping the optimizer;
4. one global gradient clip and one AdamW step are applied per epoch.

Evaluation forwards are chunked by the same configured row bound. A regression
test compares a 37-row full-batch update with row batch size 8 and verifies the
epoch loss and every flow parameter after the update. This avoids making an
eight-times-larger optimizer batch or extra optimizer steps into a hidden
method variable.

The remote runners also record a content hash over the method bundle, validate
all manifest fields and selected support paths, terminate sibling GPU workers
on failure, and remove transient anomaly maps.

Changed or created implementation surfaces:

```text
scripts/flow_tte_support.py
src/flow_tte/trainer.py
scripts/run_flow_tte_mvtec_ad2.py
scripts/run_flow_tte_dvt_denoising_all8_remote.sh
scripts/run_flow_tte_hplus_rotation8_remote.sh
scripts/run_flow_tte_hplus_rotation8_ablation_remote.sh
tests/test_mvtec_classic_adapter.py
tests/test_flow_tte.py
```

## Local Implementation Evidence

- Exact eight-angle output is byte-identical to the original OpenCV expression
  for all `0,45,...,315` degree angles on a rectangular RGB test image.
- The six legacy support transforms remain unchanged.
- Focused support/DVT tests: `25 passed`.
- Broader FlowTTE regression suite after the trainer change: `48 passed`.
- Exact-gradient microbatch regression: passed with maximum forward rows `8`,
  loss tolerance `1e-5`, and parameter tolerance `1e-6`.
- Ruff on the changed support, trainer, and test files: passed.
- Focused BasedPyright on the trainer and its regression test: `0 errors`.
- Shell syntax for the base, rotation-8 wrapper, and ablation controller:
  passed.
- Python byte compilation and `git diff --check`: passed.
- Final repository unit suite: `85 passed, 5 warnings`; all warnings are the
  existing PyTorch nested-tensor notice from Transformer-flow tests.
- Manual library driver: `8/8` exact rotations were byte-identical at the
  expected rectangular shape, and an unknown rotation name was rejected with
  `RuntimeError`.
- CLI help exposed support transforms, DVT mode, normality mode, calibration
  cap, and cleanup controls.
- Repository-wide BasedPyright still reports 12 unrelated pre-existing errors
  in `conv2d_flow.py`, `memory.py`, and `transformer_flow.py`.

## Remote Execution and Provenance

Remote execution used dsba3 container `hun_fsad_tta_012` on host GPUs
`0,1,2`. Phase 3 of the preceding hyperparameter run completed before this
controller started the measured runs.

An initial smoke preflight at `02:15:31 UTC` found an incorrect provenance
path in the base runner. It produced no GPU body or result artifact; the path
was corrected, syntax-checked, redeployed, and the actual smoke began at
`02:16:41 UTC`.

Completed runs:

```text
flowtte_hplus_dvt_superad_rotation8_can_smoke_20260710_v1
flowtte_hplus_dvt_identity_cal4096_all8_20260710_v1
flowtte_hplus_dvt_superad_rotation8_all8_20260710_v1
```

Controller completion:

```text
[validated] identity objects=8 views=16
[validated] rotation8 objects=8 views=128
[complete] H+ DVT rotation-8 ablation 20260710_v1
```

Provenance recorded by the executed v1 runs:

```text
support manifest sha256:
9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5

executed v1 method-bundle sha256 (original nine-file subset):
21ea144524a3a09066b315c7d7c3ac433329661f73c2aca748016ffed44cc23b
```

A post-run review found that the v1 method hash was sufficient to show that
the identity and rotation arms used the same selected overlay files, but it did
not cover the full Python dependency closure. The runners now compute a
deterministic 48-file v2 hash over the FlowTTE overlay plus the actual
`PROJECT_ROOT` dataset/evaluator dependencies, including the direct
`src/post_eval.py -> src/utils.py` edge. The audited current hashes are:

```text
remote v2 closure: fffbc2c920325c9c5864f48963c4b36df7e1689dffbc64e58583b49acc8ff18e
local v2 closure:  31857fe1c376ec5b128e6e038e5d00ad64014fac332e242f806d3d58a1e4b027
```

These v2 hashes intentionally differ. The difference was isolated to the
base-project evaluator files `fmad/evaluation/metrics.py` and
`src/post_eval.py`: the remote evaluator used for both arms contains its
float16-histogram metric path, while the local checkout retains the slower
reference evaluator; `fmad/evaluation/metrics.py` also differs by an
annotation-only future import. This prevents claiming bit-for-bit local
reproduction of the absolute metrics from the current checkout, but it does
not confound the paired augmentation delta because both arms ran sequentially
against the same remote evaluator and their manifests, support paths, and v1
hashes match. The completed v1 markers were left immutable. Driving the updated
controller against the same suffix now exits before GPU work with
`method bundle mismatch`, proving that the broader v2 guard will not silently
reuse those older subset-hash artifacts.

For all three GPU chunks, the identity and candidate manifests have no
unexpected field difference after excluding the tested transform list and its
derived view count plus result-only fields. Every object uses the same ordered
16 source paths in both arms. The candidate changes only:

```text
identity:  identity, 16 effective support views
candidate: exact SuperAD rotations, 128 effective support views
```

The worst-case `can` smoke completed without OOM at approximately 24.5 GiB
peak observed GPU memory on a 49.1 GiB GPU. Its metrics
`0.545911/0.000514` are runtime-safety evidence only and are not an additional
statistical trial.

## Matched All-Eight Results

| Object | Identity AUROC | Rotation-8 AUROC | Delta AUROC | Identity F1 | Rotation-8 F1 | Delta F1 |
|---|---:|---:|---:|---:|---:|---:|
| can | 0.560031 | 0.545911 | -0.014121 | 0.000605 | 0.000514 | -0.000091 |
| fabric | 0.968286 | 0.956600 | -0.011687 | 0.697746 | 0.655124 | -0.042623 |
| fruit_jelly | 0.780557 | 0.780108 | -0.000449 | 0.479194 | 0.490443 | +0.011249 |
| rice | 0.946903 | 0.950235 | +0.003333 | 0.711481 | 0.709798 | -0.001683 |
| vial | 0.745669 | 0.732443 | -0.013227 | 0.433757 | 0.419883 | -0.013874 |
| wallplugs | 0.908154 | 0.908578 | +0.000424 | 0.633132 | 0.634051 | +0.000918 |
| walnuts | 0.889911 | 0.899557 | +0.009646 | 0.733970 | 0.730333 | -0.003638 |
| sheet_metal | 0.891598 | 0.780578 | -0.111020 | 0.529072 | 0.409467 | -0.119605 |
| **macro mean** | **0.836389** | **0.819251** | **-0.017137** | **0.527370** | **0.506201** | **-0.021168** |

## Gate Decision

The candidate fails every positive gate and satisfies the kill gate directly:

- both macro metrics regress;
- AUROC regression `-0.017137` is worse than the `-0.005` kill bound;
- F1 regression `-0.021168` is worse than the `-0.010` kill bound;
- only `fruit_jelly` reaches the continuation threshold of `+0.01` F1,
  instead of at least three non-`can` objects;
- `fabric` and `sheet_metal` each regress by more than `0.02` F1;
- `rice` satisfies the no-harm control, but that isolated pass cannot offset
  the broad mean failure.

Therefore exact SuperAD rotation-8 augmentation is rejected for the retained
H+ DVT MLP method. The result does not justify the pre-registered
transform-aware DVT follow-up: there is no coherent improving
rotation-sensitive object group, and the aggregate loss is too large.

## Structural Interpretation

This is not evidence that support augmentation is universally harmful. It is
evidence that this augmentation is structurally mismatched to the current
pooled-DVT plus flat-memory pipeline.

1. Pooled `position_mean` DVT assumes that a grid coordinate has comparable
   semantic meaning across support views. A 45-degree fixed-canvas rotation
   moves object content across coordinates, crops corners, and introduces
   reflected-border pixels. Pooling all rotations therefore averages
   non-corresponding regions before scoring.
2. The flow and nearest-neighbor memory treat all 128 views as normal samples.
   They are strongly correlated orbit copies, not independent observations,
   and they make interpolation, border, and rotated directional texture part
   of the normal support manifold.
3. The effective memory grows exactly 8x, but useful diversity does not grow
   proportionally. The fitted support NLL distribution becomes narrower,
   consistent with distributional flattening: its standard deviation falls
   from `273.42` to `166.81` on `sheet_metal`, while F1 falls by `0.119605`;
   on `can`, `fruit_jelly`, and `vial`, NLL spread also falls by roughly
   48--56 percent without a corresponding ranking gain.
4. Directional structure is particularly vulnerable. The largest failures are
   `sheet_metal` and `fabric`, consistent with rotations normalizing away
   orientation-sensitive texture and boundary evidence. The exact causal
   contribution of pooled DVT versus the enlarged correlated memory is not
   isolated here, so this point remains a mechanism-consistent inference.

The practical resolution is to keep identity-only support for the retained
method. If augmentation is reconsidered later, it must be designed as a new
method branch with transform-aligned coordinates or transform-specific
statistics plus balanced orbit aggregation, not inserted into the current
pooled DVT memory as additional independent normal rows.

## Historical and SuperADD Context

The matched identity control differs from the uncapped historical H+ run by
only `-0.000350` AUROC and `-0.000058` F1, supporting the stability of the
paired comparison:

| Run | AUROC | F1 | Comparable |
|---|---:|---:|:---:|
| historical H+ DVT | 0.836739 | 0.527427 | drift reference only |
| matched identity | 0.836389 | 0.527370 | yes, decisive control |
| rotation-8 candidate | 0.819251 | 0.506201 | yes, paired candidate |
| reported SuperADD | 0.839300 | 0.626113 | no, context only |

Required context fields:

```text
superadd_seg_AUROC_0.05=0.839300
superadd_seg_F1=0.626113
method_seg_AUROC_0.05=0.819251
method_seg_F1=0.506201
delta_vs_superadd_seg_AUROC_0.05=-0.020049
delta_vs_superadd_seg_F1=-0.119912
comparable=false
```

## Debugging Audit

Three runtime/method hypotheses were tested:

- **H1 exact-transform mismatch — confirmed and fixed.** The original support
  adapter did not implement the prior SuperAD OpenCV transforms. All eight
  exact angles are now byte-identical to the original expression.
- **H2 rotation-8 support OOM — confirmed as a design risk and fixed.** The old
  flat-row path attempted one giant autograd batch. Full-batch-equivalent row
  microbatching passed parameter-level regression and the worst-case remote
  smoke completed without OOM.
- **H3 invalid paired delta from configuration or support drift — not
  observed within the executed pair, with a reproducibility caveat.** Manifest
  comparison found no unexpected setting difference; ordered source paths,
  support hashes, and executed v1 method hashes match across arms. The
  post-run closure audit exposed a remote/local evaluator difference that was
  shared by both arms rather than an arm-specific drift. The new 48-file v2
  guard covers that dependency and rejects stale v1 reuse before GPU work.

## Cleanup

- All controller and child processes exited; dsba3 GPUs `0,1,2` returned to
  `0%` utilization and `1 MiB` reported memory.
- No smoke/identity/candidate child log contains OOM, traceback,
  killed-process, or error markers.
- Each smoke/identity/candidate chunk records
  `cleanup_anomaly_maps=true`.
- Local and remote retained `anomaly_maps/` directory count is `0`.
- The final stale-reuse QA launched no experiment worker; GPUs `0,1,2` remained
  at `0%` utilization and `1 MiB` after the expected validation failure.
- Pulled result roots:

```text
results/remote_runs/dsba3/flowtte_hplus_dvt_identity_cal4096_all8_20260710_v1
results/remote_runs/dsba3/flowtte_hplus_dvt_superad_rotation8_all8_20260710_v1
results/remote_runs/dsba3/flowtte_hplus_dvt_superad_rotation8_can_smoke_20260710_v1
```

## Conclusion

Adding the exact prior SuperAD rotation set to the existing H+ DVT MLP path is
a decisive negative result. It reduces both ranking and mask quality under a
matched all-eight comparison, with severe orientation-sensitive regressions.
Retain identity-only support and do not spend another full run on plain
rotation augmentation or transform-aware DVT for this branch.
