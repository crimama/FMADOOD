# Project Lessons

## dsba3 Remote Execution Policy

- Use a single fixed Docker container for FMAD/FlowTTE work on dsba3:
  `hun_fsad_tta_012`.
- Do not create additional per-experiment containers unless the fixed container
  is unusable and the reason is recorded first.
- Source the local-only remote preset before remote work:
  `source configs/remote/dsba3.env`.
- `configs/remote/dsba3.env` is intentionally ignored by git because it contains
  local connection credentials.
- Treat host GPU IDs as execution resources. The current fixed container is the
  default for GPU `0,1,2` runs.

## Experiment Result Cleanup Policy

- After an experiment has recorded metrics, logs, configs, and a conclusion,
  delete regenerable dense output directories named `anomaly_maps/` from both:
  remote result roots and local pullback roots.
- Preserve compact evidence artifacts:
  `metrics.json`, `run_manifest.json`, summary JSON/TSV files, logs, reports,
  configs, comparison tables, and explicitly selected figure assets.
- Do not delete datasets, model/cache directories, Docker images, Docker volumes,
  or other users' containers/results as part of experiment cleanup.
- If `anomaly_maps/` must be kept for a figure audit or follow-up analysis,
  record the exact path, reason, and expected cleanup point in `tasks/todo.md`
  or the experiment report.
- Verify cleanup with:
  `find <result_root> -type d -name anomaly_maps -prune -print`.
