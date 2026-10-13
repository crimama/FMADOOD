# Source/Part-Conditioned Local Transport Full-Shot Stage -1 Log

## 2026-08-11 — preregistered

- Frozen four objects, Probe16 paired outer evaluation, `support16` and
  `full_train_good` budgets,
  audited full counts, 24x24 DINO feature contract, graph/retrieval/tangent
  cardinalities, all arms, synthetic split, and queue allocation.
- Status remains `PRE-REGISTERED / NOT RUN` until code and manifest contracts
  are verified and immutable remote shards complete.

## 2026-08-11 — pre-launch implementation and feasibility amendment

- Sealed the full `train/good` manifest at SHA-256
  `58f6b249d82ac713c9a0051675244db5216ebbe995d18ecb123058c9499a592d`;
  counts and path inventories match local, dsba3, and dsba5.
- Implemented frozen layer-5 retrieval/layer-11 scoring, cross-source same-part
  graph, retrieval-root tangent fitting, independent normalized R0, local R1,
  PCA32 margin R2, true source/part/tangent nulls, and O0/O1 oracle boundaries.
- Replaced full-width SVD and Python retrieval loops with chunked Gram
  retrieval and 8x8 row-Gram tangent eigenspaces.
- Froze calibration and margin query caps at 64 and 32 SHA-selected fit
  sources. All eligible supports still participate in graph, retrieval, and
  tangent fitting.
- Focused core/runner/aggregate/ops suite passes 30 tests before independent
  final verification and GPU preflight.

## 2026-08-11 — CUDA transport execution amendment

- The first immutable remote attempt,
  `spct_stage_minus1_transport_fullshot_v1`, was stopped before producing any
  accepted shard. Both support16 jobs remained CPU-bound for more than 32
  minutes while GPU utilization stayed near zero; extrapolated full-shot
  retrieval time was operationally infeasible. Its remote directories are
  retained with `superseded_cpu_bound.txt`, and none of their partial state is
  eligible for aggregation.
- This is an execution-only amendment: graph, retrieval, tangent, scores,
  arms, manifests, folds, caps, and gates are unchanged. Graph construction,
  retrieval, sparse root eigenspaces, and transport scoring now remain on the
  selected CUDA device for the lifetime of each fold. R0, the isolated O1
  oracle, and the declared margin-head boundary remain on CPU.
- `score_batch_rows=262144` now bounds CUDA graph and retrieval chunks. The
  fold displacement hash is computed exactly once from the original CPU array
  and passed into every CUDA tangent-root fit, avoiding repeated multi-GiB
  device-to-host copies.
- CPU/CUDA verification retained identical graph, retrieval, and winning
  provenance. Maximum observed score and residual deviations were
  `3.82e-05` and `1.67e-06`. The complete focused suite passes 40 tests; Ruff,
  Python compilation, shell syntax, and scoped diff checks pass. A real CUDA
  runner smoke verified CUDA-resident roots.
- The largest dsba5 fold is estimated at 9.404 GiB resident transport state;
  the enforced 20 GiB free-GPU preflight leaves sufficient margin on the
  24 GiB device.

## 2026-08-11 — completed CUDA execution and aggregation

- CUDA v2 completed `vial`, `walnuts`, and `fabric` at both budgets plus
  `can/support16`. `can/full` failed at transport entry because a repeated
  `torch.isfinite(displacement).all()` allocated a 1.79 GiB full boolean mask;
  the process held 23.08 GiB on the 24 GiB device.
- The immutable v3 `can/full` retry performed bounded CPU fold validation,
  passed `prevalidated_displacement=true`, and released CUDA cache before and
  after every fold. Peak resident transport memory was 12.85 GiB. This was an
  execution-only change; method, feature, graph, retrieval, tangent, arms,
  folds, caps, manifests, and score formulas were unchanged.
- The three completed dsba5 v2 shards were imported byte-for-byte into the v3
  retry run with `retry_import_provenance.txt`; only `can/full` was recomputed.
- All eight shard summaries passed runner validity. The first strict aggregate
  rejected the exact floating result `AUPRC=1.0000000000000002` in one O1 row
  per budget. The aggregator now tolerates and clips only `1e-12` boundary
  overshoot; meaningful overshoot remains invalid. Focused SPCT verification
  passes 45 tests plus Ruff and Python compilation.
- Final aggregate verdict: `VALID_DIAGNOSTIC`. Sub-verdicts are retrieval
  `SIGNAL`, tangent `SIGNAL`, full-shot `SIGNAL`, and synthetic margin
  `NO_SIGNAL`.
- Full-shot R1 gain: `+0.080122`, 95% CI `[+0.039551,+0.117081]`, four of four
  object wins. Source and part shuffle gaps are `+0.163114` and `+0.258039`.
  Synthetic `R2-R1` and `R2-N4` are exactly zero.
- Compact result:
  `results/remote_runs/merged/spct_stage_minus1_transport_fullshot_cuda_v3_can_retry/summary.json`.
