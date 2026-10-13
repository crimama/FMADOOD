# SO-CFM Synthetic-Guard Stage -1 Index

- Protocol: `socfm-synthetic-guard-stage-minus1-20260811`
- Dataset boundary: MVTec AD2 single-image support-only train/good diagnostic.
- Support manifest: `skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/superad16_dinov2_reference_paths.json`
- Support SHA-256: `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`
- SuperAD/RN-FMLK: `N/A`; `benchmark_comparable=false`.
- Runner: `scripts/run_socfm_synthetic_guard_stage_minus1.py`
- Remote wrapper: `scripts/run_socfm_synthetic_guard_stage_minus1_remote.sh`
- Aggregator: `scripts/aggregate_socfm_synthetic_guard_stage_minus1.py`
- Canonical reduced objects: `can`, `fabric`, `vial`, `walnuts`.
- Final verdict: `KILL_FOR_CLAIM_CONTINUE_DIAGNOSTIC`.
- Allocation: local GPU0 (`can,fabric`), dsba3 host GPU3/process `cuda:0` (`vial`),
  dsba5 host GPU3/container device 0 (`walnuts`).
- Aggregate summary: `summary.json`.
- Human-readable result: `report.md`.
- Pulled result roots:
  - `results/socfm_synthetic_guard_stage_minus1_20260811/local`
  - `results/remote_runs/dsba3/socfm_synthetic_guard_stage_minus1_20260811_dsba3_v3`
  - `results/remote_runs/dsba5/socfm_synthetic_guard_stage_minus1_20260811_dsba5_v3`
