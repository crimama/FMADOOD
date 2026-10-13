# ATAC-v1 Stage -1 support-validity experiment

Date: 2026-08-24  
Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`

## Negative Evidence Intake

ATAC-v1 was a frozen, forward-only DINOv2 token-acceleration diagnostic, not a
new detector claim. The local can support gate had already shown a fatal RGB
domain excursion. The dsba3 execution was restricted to cross-machine custody
replication and could not tune epsilon, open public images, or rescue the
branch.

## Motivation

Determine whether the local fatal was an environment artifact or a stable
property of the fixed `x +/- 0.05 H2` path. The experiment tests input-path and
numerical-control validity before any anomaly-performance measurement.

## Implementable Design

- Target: MVTec AD2 single-image, `can/train/good` only.
- Exact SuperAD-selected support: 16 images, manifest SHA-256
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Candidate: two-level exact orthonormal-Haar antithetic central acceleration,
  DINOv2-L/14 layer 23, epsilon `0.05`.
- Fatal gates: RGB identity `<=1e-6`, RGB excursion fraction `<=0.005`,
  no-op and linear-control acceleration `<=1e-6`, locality Spearman `>=0.90`,
  locality scale ratio in `[0.5,2.0]`, support LOO ratio `<=2.0`.
- Stopping: any fatal support gate forbids mechanism/public/all-eight stages.

## Evaluation Alignment

This is a support-validity replication, not a SuperAD/RN performance
comparison. SuperAD and RN public metrics remain locked. The maximum valid
claim is reproducible failure-boundary evidence.

## Code Modification / Creation

- `src/flow_tte/atac.py`: exact odd-boundary two-level Haar and diagnostics.
- `scripts/run_atac_stage_minus1.py`: support-only gate and compact evidence.
- `scripts/aggregate_atac_stage_minus1.py`: fail-closed partial/fatal aggregate.
- `scripts/run_atac_stage_minus1_remote.sh`: immutable dsba3
  `preflight/start/status/pull` controller.
- Focused tests in `tests/test_atac*.py`,
  `tests/test_aggregate_atac_stage_minus1.py`, and
  `tests/test_run_atac_stage_minus1_remote.py`.

## Added Code Evaluation

- `36 passed` across core, runner, aggregator, and remote-wrapper tests.
- `bash -n` and Python compilation passed.
- Remote preflight verified archive, runner, aggregator, core, and support
  hashes; DINOv2 offline cache; synthetic support gate; fixed container; and
  idle GPU UUID.

## Remote Execution

- Host: dsba3, `hunim@147.47.39.144:2222`.
- Fixed container: `hun_fsad_tta_012`.
- Run: `atac_v1_stage_minus1_dsba3_gate_can_20260824_v1`.
- Host GPU 0 / container CUDA slot 0; UUID
  `GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f`.
- GPUs 0--3 were idle and user-authorized; only GPU0 was used because the
  preregistered gate had one object and forbade expansion after a fatal.
- Snapshot SHA-256:
  `a12841ac524a6710e571a52b56b0b0a4018a6b7baec3791069bca94e50663635`.
- Remote root:
  `/workspace/results_remote/atac_v1_stage_minus1_dsba3_gate_can_20260824_v1`.
- Local pull:
  `results/remote_runs/dsba3/atac_v1_stage_minus1_dsba3_gate_can_20260824_v1`.

## SuperAD Baseline and Unified Metrics

No public query or ground truth was accessed. Consequently
`superad_seg_AUROC_0.05`, `method_seg_AUROC_0.05`, `seg_F1`, and performance
deltas are intentionally absent and `protocol_complete=false`.

## Results and Analysis

The dsba3 gate serialized `FATAL_PRECHECK`; the partial aggregate serialized
`KILL / FATAL_SUPPORT_GATE / KILL_FOR_CLAIM`.

| Gate | dsba3 result | Decision |
| --- | ---: | --- |
| RGB excursion fraction | `0.0096239--0.0162178`, 16/16 fail | fatal (`<=0.005`) |
| RGB central identity max | `1.19209e-7` | pass |
| no-op acceleration max | `0` | pass |
| linear-control acceleration | `1.34870e-4--1.84678e-4` | fatal (`<=1e-6`) |
| epsilon-locality median Spearman | `0.9917549` | pass |
| epsilon-locality scale ratio | `0.9556157` | pass |
| support LOO scale ratio | `1.0399660` | pass |

Excursion replicated the local min/max exactly and the 16-value vector within
`3.41e-7`; locality and LOO differences were at most approximately `1.15e-4`.
The excursion is therefore a stable structural failure. The linear-control
failure emerged after removal of a stale `1e-3` zeroing rule: fp32 affine
roundoff is magnified by division by epsilon squared. That exposes a numerical
contract failure, but the frozen threshold cannot be relaxed post hoc.

Custody fields record `test_public_accessed=false`,
`test_private_accessed=false`, `ground_truth_accessed=false`, and the exact
support-manifest hash.

## Continuation Assessment

ATAC-v1 has no continuation. Lowering epsilon, clipping, restoring numerical
zeroing, or relaxing thresholds would be post-result tuning. A future method
must use a separately named and preregistered bounded-domain symmetric path and
must specify fp32/float64 control tolerances before observing its support gate.

## Conclusion

`KILL_FOR_CLAIM / NO_CONTINUE`. The dsba3 replication contributes
cross-machine reproducible failure-boundary evidence only. It supplies no
SuperAD/RN performance evidence and does not authorize public evaluation.

## Post-Conclusion Storage Cleanup

The remote result and local pull contain no `anomaly_maps/` directories. All
compact artifacts listed in `compact_artifact_sha256.txt` passed SHA-256
verification. After completion GPU0 returned to 1 MiB used and 0% utilization.
