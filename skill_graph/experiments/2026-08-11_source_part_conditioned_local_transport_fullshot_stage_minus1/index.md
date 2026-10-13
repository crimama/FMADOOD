# Source/Part-Conditioned Local Transport, Full-Shot Stage -1

Status: `COMPLETE / VALID_DIAGNOSTIC / RETRIEVAL+TANGENT+FULLSHOT SIGNAL / SYNTHETIC NO_SIGNAL`

This support-only diagnostic combines the strong SOVM spatial-position signal
and separate same-source-oracle headroom through frozen cross-image part
retrieval and a local normal nuisance tangent. It compares the historical
16-shot support budget with full `train/good` support on the same held Probe16
pseudo-queries.

- [Analysis](analysis.md)
- [Preregistration](preregistration.md)
- [Report](report.md)
- [Execution log](log.md)
- [Tasks](tasks/todo.md)
- Full manifest: `full_train_good_paths.json`
- Final compact aggregate:
  `results/remote_runs/merged/spct_stage_minus1_transport_fullshot_cuda_v3_can_retry/summary.json`

This is not a SuperAD benchmark run: `benchmark_comparable=false`.
