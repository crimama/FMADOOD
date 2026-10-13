# ATAC-v1 dsba3 support-gate replication preregistration

Date: 2026-08-24
Verdict ceiling: `REPLICATION_DIAGNOSTIC` (never a method claim)

## Motivation and boundary

The frozen local ATAC-v1 run already ended at `FATAL_SUPPORT_GATE` because all
16 exact `can` support images exceeded the preregistered RGB excursion limit.
The dsba3 run is authorized only to test execution/custody reproducibility on
the established remote stack. It cannot rescue ATAC-v1, retune epsilon, or
authorize `test_public` access.

## Frozen remote custody

- Target: `dsba3`, `hunim@147.47.39.144:2222`.
- Fixed container: `hun_fsad_tta_012`; no new container.
- Host GPU: `0`, selected after observing 1 MiB used and 0% utilization on all
  four RTX A6000 GPUs. Container CUDA slot: `0`.
- Target dataset: MVTec AD2 single-image.
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`.
- Object/population: `can/train/good`, exact 16 support paths only.
- Forbidden access: `test_public`, `test_private`, and ground truth.
- Method: frozen ATAC-v1, two-level exact orthonormal Haar, epsilon `0.05`,
  DINOv2-L/14 layer 23 normalized-token central acceleration.
- Support manifest SHA-256:
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Same-condition SuperAD/RN plan: locked because this replication may not open
  public images. No baseline performance claim is permitted.
- Primary gates: RGB identity `<=1e-6`, RGB excursion fraction `<=0.005`,
  epsilon-locality Spearman `>=0.90`, locality scale ratio in `[0.5,2.0]`, and
  support LOO scale ratio `<=2.0`.

## Immutable run and stopping rule

- Run ID: `atac_v1_stage_minus1_dsba3_gate_can_20260824_v1`.
- Upload an immutable source snapshot and verify archive, runner, aggregator,
  core, and support-manifest hashes before execution.
- Run only `run_atac_stage_minus1.py --phase gate --objects can`.
- Aggregate with `--partial`; a support fatal must serialize as
  `KILL / FATAL_SUPPORT_GATE`, not as a process crash or complete protocol.
- If any support gate is fatal, stop. Do not run `--phase mechanism`, any
  all-eight expansion, or any public metric.
- Preserve compact JSON/log/hash evidence and verify that no `anomaly_maps/`
  directory remains remotely or after local pullback.

## Expected evidentiary value

Agreement with the local excursion result supports only cross-machine custody
and numerical reproducibility. Disagreement triggers an implementation or
environment audit; it does not permit tuning or a performance claim.
