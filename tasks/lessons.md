# Project Lessons

## Support-Only Flow Refinement Lessons

- Do not call a flow method geometry-bounded when geometry is only audited
  after optimization. Parameter-space proximity does not bound latent ranks,
  accumulated log-determinants, component scale, or deployed score variance.
- For inequality constraints, avoid a signed `lambda*g` objective that rewards
  already-satisfied stochastic cells. Use a projected positive-part augmented
  term and audit its behavior with one satisfied and one violated cell.
- A cyclic latest-cell dual is not a worst-axis or worst-family constraint.
  Materialize grouped replay and optimize the explicit max/CVaR statistic used
  by the held gate.
- Treat a cleanly completed preregistered scientific failure as terminal for
  that candidate. Correct objective structure in a newly named, newly frozen
  assay; do not tune and rerun the failed method.
- `copy.deepcopy` of a fitted CUDA pipeline can fail when it owns a
  `torch.Generator`; provide an explicit offline clone that copies states and
  reconstructs runtime generators.
- Deterministic CUDA `torch.cdist` requires
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` (or the documented alternative) before the
  CUDA context starts. Put this in the remote controller and test its presence.

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
