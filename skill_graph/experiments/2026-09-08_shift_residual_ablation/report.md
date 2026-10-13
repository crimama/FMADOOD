# Shift-Residual Full-Feature Ablation

Date: 2026-09-08  
Protocol: SRA-AD2-ALL8-SAME16-S3-v1  
Status: IMPLEMENTING  
Verdict: PENDING_IMPLEMENTATION_AND_EXECUTION

## Frozen matrix

The canonical design is design.md. The four cumulative arms are fixed:

- A0: normalized full-vector support residual control.
- A1: residual branch using the bounded normal-shift residual.
- A2: explicit shift-tangent coordinates.
- A3: synthetic gate auxiliary loss on the residual branch.

A3 has no guaranteed anti-absorption property. This is offline train-only
shift modeling; there is no online test-time adaptation claim.

All arms use the same frozen DINOv2 ViT-L/14, exact HAS16 references,
independent parameters with paired initialization, three seeds, and 576
updates. P is shared across layers within one arm only. The e=0 T0 view slot
reuses the clean/anomaly feature view.

## Population and custody

Target is MVTec AD2 single-image, all eight objects, full test_public
good/bad. Binding enumeration calls MVTecAD2Dataset.get_test_images and
requires exactly the keys {good,bad}; recursive PNG enumeration and every
ground_truth tree are excluded. Verified public query counts are
can=162, fabric=156, fruit_jelly=80, rice=132, sheet_metal=114, vial=140,
wallplugs=150, walnuts=150, total=1084.

References are the exact same 16 HAS16 paths per object. R_train is indices
0..11; R_hold is indices 12..15 and remains closed until all heads seal.
The full manifest SHA-256 is runtime-recorded. Per-reference feature grids may
be ragged; every layer stores its own H_l,W_l metadata, support patches carry
image/grid identity, and no global H/W assumption is permitted.

## Cache and optimization contract

There are 240 logical view slots per object/seed:
12 clean + 48 shifted-clean + 36 anomaly + 144 shifted-anomaly. T0 reuses
clean/anomaly views, so 192 distinct feature views are materialized. Quartets
are shared by all arms; no DINO forward occurs inside a head update. Use a
bounded FP16 patch-feature cache, 576 AdamW steps per arm, paired order, and
no dense native maps.

## Baselines and metrics

Independent comparators are same-manifest SuperAD-16 and RN16: last-layer
layer-23 DINO kNN using the same HAS16 references, not a full-memory baseline.
Primary metrics are seg_AUROC_0.05, oracleF1, and AUPRO.05. Emit
per-object/per-seed rows first, then paired-seed object macro mean plus
standard deviation, then the eight-object macro mean; never pool seeds or
pixels. Test GT is metric-only and no metric chooses an arm.

## Implementation and runtime status

Pending new source files are:

- src/flow_tte/shift_residual_cache.py
- src/flow_tte/shift_residual_ablation.py
- src/flow_tte/shift_residual_runner.py
- tests/test_shift_residual_ablation.py

Source changes are not yet present; code checks have not run. Runtime mapping
is pending auditor verification for dsba1 host GPUs 0,2,4, with one
independent object worker per GPU and UUID-verified container slots. No
training, metric execution, or remote launch has started.

The earlier implementable_design.md and preregistration.md in this directory
are retained as noncanonical expanded references; design.md is authoritative.
Existing C1, baseline, dataset, and metric source/results remain untouched.

## Latest execution binding

The current run is SRA-AD2-ALL8-SAME16-S1-v1: scientific seed 20260906 only,
four cumulative arms, and 32 heads across all eight objects/1084 public
queries. Seeds 20260907/20260908 are optional follow-up only. Existing
SuperAD-16 and RN-FMLK artifacts are preferred and are not automatically
rerun. Comparability audit mismatches are recorded as comparable=false and do
not block candidate-arm execution. Single-seed SD is null, not zero. GPU
hosts remain dsba1 [0,2,4].

## Integration-check status (authoritative, 2026-09-08)

Status: INTEGRATION_CHECK; Verdict: PENDING_IMPLEMENTATION_AND_EXECUTION.
The initial run remains single seed 20260906 with A0-A3 (32 heads), all eight
objects, and 1084 TESTpublic queries. Existing SuperAD-16 and RN-FMLK results
are reused through baseline_reuse_manifest.json; baseline rerun is not
automatic. Historic baselines are FP32 and candidate streaming metrics are
FP16, therefore candidate-vs-historic comparison must be marked
comparable=false. historical_pair_valid is a separate field and can be true
only for a historical pair whose own manifest/split/preprocessing/metric/
precision metadata agree; it does not make candidate-vs-historic comparable.
Do not mark RUNNING until READY/launch evidence is recorded. This document
does not claim that training or evaluation has started.

## Pre-launch deployment correction (authoritative, 2026-09-08)

Current status: DEPLOYING. Critic READY and local preflight evidence are
available; remote approval is pending a launch receipt/PID, so this report
does not claim RUNNING or started training/evaluation.

The canonical evaluator may enumerate all queries before scheduling, then
processes each query as prediction -> GT metric update -> immediate feature/map
discard. It is not a global prediction barrier with a later metric pass.
Training and selection remain GT-free.

## Runtime update: run_v2 (authoritative, 2026-09-08)

Status: RUNNING — feature extraction only; no metrics or finished heads are
available, so this is not a success claim. Launch receipt:
local results/remote_runs/dsba1/sra_ablation_20260908/run_v2/launch_receipt.json,
SHA-256
d6a24e01806e80bf76df6550ded4b8bde230e81b162407047af1252982838fc0.
The remote output root is /workspace/results_remote/sra_ablation_20260908/run_v2
and the immutable source snapshot is
/workspace/source_snapshots/sra_ablation_20260908_v2_final,
SHA-256
4e26bf2ae6536ed970257d0322317e82bf867b4c61e94f128c1f29000d9bb8db.
Controller PID 2196 and workers 2199/2200/2201 are alive; Python workers
2206/2207/2208 serve can/fabric/fruit_jelly on host GPUs 0/2/4 (about 2 GiB
each). Local tests, remote tiny calls, and strict preflight passed. v1 failed
before execution because scripts/run_cider_cf_stage_minus1.py was absent from
packaging; its logs are retained. v2 contains the canonical closure and source
manifest. Baseline reuse and the FP32-historic versus FP16-streaming
comparability caveat remain unchanged; dense map accumulation is false.
