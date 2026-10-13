# FSAD-TTA project contract

This file is the project-specific contract for the current FMAD-OOD codebase.
Higher-level workspace instructions still apply. Current user instructions and
explicit task constraints remain authoritative; a restriction that is live in
the current task must not be weakened by this file.

## Sources of truth

- `skill_graph/current.md` is the current research state and working assumptions.
- `skill_graph/todo.md` is the active checklist; it is not a substitute for the
  current state or for recorded evidence.
- `skill_graph/log.md` and dated experiment documents are historical evidence.
  Their results and conclusions are preserved, but their old next-action,
  approval, or prohibition language is not current permission or authority.
- `skill_graph/archive/` contains superseded documents for traceability only.
- Do not read or edit `docs/method.md` unless the current user explicitly asks
  for that document. Routine result collection is not such a request.

## Cumulative comparison ledger

- For result lookup, first read the latest summary (roughly the first 35 lines) of the [compact comparison index](results/comparisons.md). Read its full macro table only for an all-method comparison; consult the latest section of [current research state](skill_graph/current.md) when research assumptions are needed. Do not routinely load either file's entire history. The canonical Markdown is `results/comparisons.md`; its paired CSV remains `results/comparisons/psma_mr_vs_history_20260909/psma_mr_vs_history_20260909.csv`. Do not create duplicate cumulative tables.
- When collecting or registering experiment results, update this Markdown and its paired CSV together with verified metrics, provenance, baseline comparisons, and interpretation. Replace partial status for the same run with completed status without duplicating identities; preserve historical experiments. A read-only result lookup or summary does not update files.
- Inspect original artifacts only for new missing values or verification; do not repeat broad scans.
- Keep the Markdown to one latest summary, one row per recorded CSV macro, scope keys and detail links. Store class tables, long interpretations and provenance audits in `<run-root>/comparison.md` (or a combined suite report), not repeated inline history. The pre-compaction detail is preserved in `results/comparisons/details/history_through_20260914.md`; search titles/methods with `rg` and read only the relevant section. It is an immutable historical snapshot, not another live ledger.
- Keep a unique method/run/config/seed/scope identity per row; distinguish `full8` from `partial`, and missing values from true zero. This table is an evidence index, not execution authorization.
- Use `scripts/experiments.py run` for config sequencing through the existing CLI and `collect` for cumulative result updates. Keep these helpers method-agnostic; do not add a separate queue or ledger script for each new method.
- Routine collection updates only the affected rows in the canonical CSV,
  the latest summary/affected macro rows in `results/comparisons.md`, and the
  relevant results section of the run's existing `comparison.md`. Preserve
  unrelated rows, ordering, scope IDs and manual interpretation; do not rebuild
  or reprint entire historical tables. Unchanged collection must be a no-op.
- Do not routinely edit `skill_graph/current.md`, `todo.md`, `log.md`, method
  overviews or other Markdown files just to mirror collected metrics. Update
  those only when the user requests them or the task changes research decisions
  or execution plans. Keep collection output to changed rows and a brief audit.
- Compute row/section updates inside the existing collector; do not return
  full-file payloads through tool output for the model to rewrite. Atomic file
  replacement for crash safety is permitted; it is not permission to regenerate
  unchanged content. Existing unmarked/manual reports receive targeted edits,
  not automatic replacement or a duplicate report.

## Research and implementation

- Keep the common data, support selection, backbone, training/fitting, query,
  and evaluation pipeline fixed when comparing methods. A method-only change is
  the default; pipeline or preprocessing changes must be explicit ablations.
- If the method adapts a backbone (for example with LoRA), declare the trainable
  parameters and isolate adapted state from the shared baseline. Record changed
  training or feature conditions as part of the method/ablation, not as an
  unnoticed change to the common comparator.
- Compare methods with the same data conditions, support images or manifest,
  seed, protocol, and declared metrics. Record enough configuration and
  provenance to reproduce the comparison.
- Before coding a claim of novelty, perform a brief related-work or novelty
  sanity check when evidence is available. Exhaustive mechanism or literature
  analysis is not a prerequisite for implementation.
- Use a concise hypothesis, implement it, and run a controlled development
  comparison when that is the next useful decision. Detailed mechanism or
  diagnosis is not a prerequisite for implementation; perform it when it can
  change the next method or claim.
- Separate execution validity, performance evidence, and research decisions.
  A failed run is not by itself evidence that a method family is invalid.
- Keep support data normal and do not pass dataset objects, ground truth, or
  evaluation labels into method fitting or prediction. Do not assume that any
  holdout is untouched: document the actual split and label use for each run.
- Keep development tuning and paper-ready final evaluation separate. Label
  post-hoc analyses and hypotheses as post-hoc rather than presenting them as
  preregistered decisions.

## Remote experiment workflow

- Modify and validate code locally, then reflect the selected committed/pushed
  version using `scripts/run_remote.sh update-code` in the designated existing
  container. Do not push code automatically without task authorization.
- Reuse the same container. Start it if stopped; if absent, report the missing
  container. Do not create, delete, recreate, or replace it for an experiment
  or retry. Image builds and provisioning are separate maintenance tasks.
- Explicitly select the assigned GPU(s) already exposed in that container;
  prefer GPU UUIDs. Do not change GPU allocation or container exposure silently.
- Do not update a checkout while an experiment uses it. The remote helper
  serializes its `update-code` and `run`/`exec` operations; manual commands must
  respect the same boundary.
- Retain experiment logs remotely. Recover quantitative results and minimal
  config/manifest provenance locally with `pull`; maps, checkpoints, and full
  logs are not part of default pullback. Do not automatically clean up remote
  artifacts or stop the container after completion/failure.

## Outputs and history

- Save quantitative results, training or fitting logs, configuration, and
  reproducibility metadata by default. Anomaly maps and fitted weights are
  opt-in outputs and remain off by default unless the current task explicitly
  enables them.
- Historical records are evidence, not current execution instructions. A past
  experiment's stop condition or method ban applies only to the documented
  conditions unless a current user explicitly re-adopts it.
- For CVPR-oriented iteration, prioritize a credible method/performance
  comparison first, then add the ablations and analysis needed to support the
  final claims.
