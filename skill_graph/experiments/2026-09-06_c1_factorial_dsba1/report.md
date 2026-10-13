# C1 Factorial: Synthetic and Shift Augmentation on dsba1

Date: 2026-09-06  
Protocol: `C1-DIAG-AT-P16-can-vial-s3-v1`  
Status: `SMOKE_PASS / MAIN_RUN_NOT_STARTED`  
Verdict: `PENDING_EXECUTION`

No scientific training or evaluation has run. Environment setup and focused
test receipts exist; scientific result fields remain `N/A` until the compact
run receipts, hashes, and metrics exist.

## Negative Evidence Intake

C1 is the PSAR same-head factorial diagnostic for synthetic anomaly (`A`) and
shift-assumed augmentation/consistency (`T`). It is not a new backbone
architecture. Prior PSAR operational-invalid evidence is not a method
refutation. Historical DVT/private settings, including the density-weight
`0.25` versus `0` mismatch, are not same-condition baselines.

## Motivation

The primary decision claim is utility on the public can/vial identity
condition. The held `G2 × cos_1_1` branch is a secondary source-held
mechanism/transfer diagnostic; it does not enter promotion or kill decisions,
baseline superiority claims, or real-shift robustness claims. The primary
metric is `seg_AUROC_0.05`, with secondary oracle `seg_F1` and `AU-PRO_0.05`,
and no all-eight-object claim.

## Implementable Design

Frozen DINOv2 ViT-L/14 uses layers `5,11,17,23` at short edge `672`. The PSAR
same head must assert exactly `241` parameters. Arms are `C00`, `C10_A`,
`C01_T`, and `C11_AT`; seeds are `20260906/07/08`; 24 fits use exactly 576
updates each. New same-condition `SuperAD-P16` and `RN-P16` dsba1 reruns are
required. The exact split, augmentation order, objective, optimizer,
storage, and custody contract are recorded in
[preregistration.md](preregistration.md).

## Evaluation Alignment

The strict split uses P8 train indices `0..11` with deterministic exclusions
`d=(i+1)%12`, `k+=(i+2)%12`, and `k-=(i-2)%12`. Held bases `12..15` remain
closed until all 24 heads are sealed. Training uses `G0/G1/G4`; held anomaly
generator is `G2_structural_scratch`. Held T is only `cos_1_1` with the
predeclared field limit. Public evaluation uses P16. Old P12 fit and old
private metrics are excluded.

The exact Same16 manifest requires full runtime hash verification. Baseline
comparability is `N/A` until the new dsba1 SuperAD-P16 and RN-P16 artifacts
match dataset, split, support, preprocessing, feature layers, and evaluator.

Held T coordinates are fixed to native G2 synthesis, canonical PIL short-edge
672 resize, ToTensor, float32 resized-H/W `cos_1_1` exponential multiplicative
field (`CV.15`, field/max), Normalize, and patch-14 crop. Spatial
multiplication is followed by no clip, round, uint8 conversion, or PIL
re-entry. The native mask is not warped; it is applied after head-map native
`INTER_LINEAR` restoration.

Post-fit population is now bound explicitly. The public can/vial identity
condition alone supplies all `PROMISING`, `COMPONENT_SIGNAL_ONLY`,
`VALID_NO_UTILITY`, and B0/B1 comparison metrics. The held source diagnostic
uses only the four paired C1 arms on `G2_structural_scratch x cos_1_1`: 2
objects x 3 seeds x 4 held bases = 24 synthetic cells and 96 maps. Held B0/B1
maps (48 maps) are outside the population.

Held mask/synthesis seeds use
`psar_d1.deterministic_seed(PROTOCOL_ID, object, scientific_seed, base, donor,
'T_H_COS_1_1_CV015', 'G2_structural_scratch', role)` with the existing `0x1f`
eight-field encoding; `role` is `mask` or `synthesize`, and the held shift is
deterministic. The fixed order is global 24-head seal, calibration, held
source diagnostic, public global GT-free inventory, then exact-hash-matched GT
pass 2.

Calibration uses bases `12..15 x 4 train-T` on P15 with exact native float32
q95. Per object, 16 compact RN grids are shared by 12 heads; one head's
16-native calibration data is RAM-temporary only. Cache sizing uses
`Craw=20016 × H × W`, cap `ceil(1.15*Craw)`, and actual NPZ plus
manifest-byte checks.

The canonical parity audit is ordered after global 24-head seal, calibration,
and held completion, and before the global public inventory. B0/B1 must be
shape-exact, finite, and canonical-nonconstant, with
`max_abs(FP32_native-FP32_canonical) <= 1e-5` and `rtol=0`. Use the
lexical-first `train/good` image excluded from each object's Same16 set (one
per object, two total), recording path and file SHA. P16 self-query is
forbidden. This source is implementation-audit-only: it cannot enter training,
calibration, thresholding, model selection, or performance metrics. Missing
source aborts with no public/held fallback. Record shape, max/mean absolute
error, and pass status only; do not store maps.

## Code Modification / Creation

Required implementation scope: a C1 core, one CLI runner, and focused tests.
The dsba1 implementation branch now contains the C1 core, streaming path,
runner, and focused tests. Source custody is recorded in the environment
receipt; no benchmark or scientific result has been produced.

## Added Code Evaluation

The latest five-suite core/eval/runner/stream check set reports `40 passed`,
11 warnings, and 12.81 seconds after small fixes,
including the actual RGB `torch.equal` identity-path check. The remote command
was:

```text
PYTHONPATH=/workspace pytest -q tests/test_c1_factorial.py tests/test_streaming_segmentation.py tests/test_c1_runner.py
```

The environment receipt records the source hashes for
`src/flow_tte/c1_factorial.py`, `tests/test_c1_factorial.py`, and
`tests/test_c1_runner.py`, as well as the CUDA and FAISS checks. Learned-cache,
evaluation, and full-pipeline readiness remain `NOT_READY`; scientific
metrics are `N/A`. An RN normalized-proxy cache issue was found and is being
repaired. It must not be interpreted as a method failure. Runtime core,
evaluation, runner, and stream checks are passing. The earlier cache-fixture
issue and UUID-property preflight failure are historical repair entries, not
the current status.

## Environment and Verification Receipt

The authoritative setup receipt is
`results/remote_runs/dsba1/setup_c1_20260906/environment_manifest.json`.
The current receipt SHA-256 is
`ebfea5f8f23732bdc24fcfab0925c1c6d22e3ccc349627af8133a39437911b5a`.
It records:

- container: `hun_fsad_tta_c1_dsba1`;
- source image ID: `sha256:8dbd129dc64fbe1bbce27212902b27e2ecd3fea01794565c80f5ef415369b2f1`;
- host GPU `6` only, exposed as CUDA `0`;
- GPU UUID: `GPU-59115c6a-8ccb-5474-72e3-67cdf2d363fd`;
- dataset root `/home/hunim/Volume/DATA/mvtec_ad_2`, mounted read-only,
  with 8,711 files and all eight object populations present;
- DINOv2 ViT-L/14 checkpoint SHA-256
  `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428`;
- exact Same16 support manifest SHA-256
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`;
- loaded model-state hash `59a755...`, distinct from the checkpoint and
  verified as matching the expected loaded state;
- runtime FAISS `1.14.1` through `faiss-gpu-cu11 1.14.1.post1`, with matching
  host-module SHA and a passed normalized CPU/GPU neighbor toy check.

The receipt's source hashes are:

```text
src/flow_tte/c1_factorial.py       db9958b3a1777859c4d80fa3d9c2cceadfdbf046af09eeba8e40ff0459343207
tests/test_c1_factorial.py         224f40c66b17cbe7ad0a88e2bb800e1f739081d3893c466da3c44e4933d5583c
tests/test_c1_runner.py            a334cd1ad4e986057af6a524a2aa128a4a5d297fc5670eb809bdc4080f99a51e
FAISS module                      44fa8428148b159d87c643726a63d2c4ca49f914585ee83d6f25dd3dc3e56aa7
```

The dsba3-derived image and read-only dataset were validated for dsba1 use.
The old P12 fit remains discarded. The setup receipt also records that an
unneeded 5.4 GiB v1 task copy, `.claude` content, and a 938 MiB partial cache
were removed from the new run workspace; source, datasets, and model caches
were preserved. No old experiment result or unrelated cache was removed.

## Remote Execution

Status: `PREFLIGHT_PASS / SMOKE_PASS / MAIN_RUN_NOT_STARTED`. Planned host is
dsba1 GPU `6` only, exposed as CUDA `0` with UUID equality. Credentials are
sourced from `configs/remote/dsba1.env`. The validated dsba3 Docker environment
and dataset are reused through the dedicated dsba1 container. Existing dsba3
container/image/data/cache/results remain untouched. Busy GPU 6 defers the run;
there is no fallback. All cells are sequential.

The setup and diagnostic smoke have completed. The smoke receipt is
`/workspace/results_remote/c1_smoke_dsba1_20260906_retry1` and records
`can/20260906`, one cell, all four arms, one update, the 241-parameter head,
DINO bf16, RN fp32 on CUDA 0, host GPU 6 UUID
`GPU-59115c6a-8ccb-5474-72e3-67cdf2d363fd`, a 304,649-byte cache, zero dense
maps, no held/public GT access, `scientific=false`, and zero residual PIDs.
The planned next order is 24 full fits, checkpoint seal, pass-1 map/hash/shape
inventory without GT followed by dense map release, then pass-2 map
regeneration, exact-hash matching, and GT stream metrics. The smoke is invalid
for science. The 24-fit training run, calibration, held diagnostic, public
inventory, and evaluation remain `NOT_STARTED`.

Learned cache construction, evaluation, and full-pipeline readiness are not
complete. The report must not advance to `READY`, add metrics, or assign a
scientific verdict until the pipeline receipt is complete.

## SuperAD Baseline and Unified Metrics

Baseline source: new same-condition dsba1 `SuperAD-P16` rerun, pending. A new
same-condition `RN-P16` rerun is also pending. Historical or truncated-hash
artifacts cannot satisfy the baseline gate.

The output schema will report `superad_<metric>`, `method_<metric>`,
`delta_vs_superad`, `comparable`, counts, finite/nonconstant status, seed
metadata, and custody hashes. Primary metric is `seg_AUROC_0.05`; secondary
metrics are oracle `seg_F1` and `AU-PRO_0.05`, with matched float16-order
histogram aggregation. This histogram representation is not FP32-exact.
Values: `N/A`.

## Results and Analysis

No maps, checkpoints, metrics, or result hashes exist for this protocol yet.
The proposed descriptive quantities are:

```text
DeltaA = .5[(C10_A-C00)+(C11_AT-C01_T)]
DeltaT = .5[(C01_T-C00)+(C11_AT-C10_A)]
I_AT   = C11_AT-C10_A-C01_T+C00
```

They will not be treated as causal proof. The q95 activity threshold uses the
four post-seal held-normal bases x four train-T conditions and is activity/FPR
only; oracle F1 remains independent.

## Continuation Assessment

`N/A` until the registered metrics and controls are available. The prospective
`PROMISING` gate requires C11-C00 macro pAUROC `>=.005`, per-object delta
`>=-.010`, macro F1/AU-PRO deltas `>=-.020`, DeltaA or DeltaT `>=.003`, and
C11 improvement over both B0/B1 by `>=.005` without no-harm failure. A
component-only outcome is `COMPONENT_SIGNAL_ONLY`; valid no utility is
`VALID_NO_UTILITY`. A future `LATcons` removal is limited to one controlled
follow-up with the same inputs; C2/C3 are excluded.

## Conclusion

`PENDING_EXECUTION`. No `KEEP`, `CONDITIONAL KEEP`, `PROMISING_DIAGNOSTIC`,
or failure verdict can be assigned before implementation checks, custody,
same-condition baselines, and actual metrics complete.

## Post-Conclusion Storage Cleanup

`N/A`. After a recorded conclusion, retain compact manifests, hashes, configs,
logs, summaries, and selected evidence; remove only this run's regenerable
dense output/cache trees, and record verification. Existing dsba3 or unrelated
data/results/cache must not be removed.

The earlier UUID-property preflight stop is retained as a repair-history event
only. The current receipt records that the UUID compatibility fix passed
preflight and the smoke completed; the earlier stop is not the current status.
