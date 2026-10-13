# Skill Graph Index

> **Current-document boundary — 2026-09-08**
>
> This page is a navigation index, not an experiment protocol. The active
> research state and next decisions live in [`current.md`](current.md). The
> project contract and runtime instructions are [`../AGENTS.md`](../AGENTS.md)
> and [`../README.md`](../README.md).
>
> Dated reports, preregistrations, methods notes, and logs preserve historical
> evidence. Their experiment-specific `KILL`, `CLOSE`, `NO_*`, approval, or
> revision-budget language is scoped to the named experiment and its recorded
> conditions; it is not a blanket prohibition or permission for current work.
> The pre-transition index and TODO are preserved byte-for-byte in
> [`index_history_pre_2026-09-08.md`](index_history_pre_2026-09-08.md) and
> [`todo_history_pre_2026-09-08.md`](todo_history_pre_2026-09-08.md).

## Current authority

- [Current research state and queue](current.md): active candidates, comparison
  conditions, pending decisions, and the next bounded action.
- [Project operating contract](../AGENTS.md): repository boundaries,
  preservation rules, and execution constraints.
- [Controlled FMAD runtime guide](../README.md): installation, method boundary,
  exact-shot protocol, output policy, result schema, and checks.

## Current architecture and evidence

- [Common controlled-experiment pipeline refactor](analysis/2026-09-08_common_pipeline_refactor/report.md):
  the active runtime is under `src/fmad`; the runner owns support selection,
  evaluation, and result persistence. This is an engineering change, not a new
  benchmark result.
- [Results consolidation](analysis/2026-09-08_results_consolidation/README.md):
  compact quantitative preservation layer and source-audit scope.
- [Consolidated insights](analysis/2026-09-08_results_consolidation/insights.md):
  decision-bearing anchors, validity boundaries, and incomplete-record policy.
- [Quantitative ledger](analysis/2026-09-08_results_consolidation/quantitative_ledger.json)
  and [source index](analysis/2026-09-08_results_consolidation/source_index.tsv):
  machine-readable values and their preserved source identities.

The common data, training, evaluation, and persistence path is the comparison
constant. Method changes belong behind the method interface and must be
compared with the same seed, support-image policy, preprocessing, and metric
contract. Default outputs are quantitative results, training logs, and
reproduction metadata; anomaly maps and fitted weights remain opt-in.

## Historical evidence

- [Superseded FMAD skill instructions](archive/instructions_2026-09-08/README.md):
  recovery copies only, not active workflow instructions.
- [Pre-transition index, byte-preserved](index_history_pre_2026-09-08.md).
- [Pre-transition TODO, byte-preserved](todo_history_pre_2026-09-08.md).
- [`log.md`](log.md): the transition note above the unchanged historical log
  body, followed by dated experiment records.
- [`analysis/`](analysis/) and [`experiments/`](experiments/): dated research
  records. Use the canonical report and consolidated ledger for interpretation;
  do not infer a current launch instruction from an old record alone.

## Reading rule

When documents disagree, resolve them in this order: the project contract,
[`current.md`](current.md), the current runtime README, then the named
experiment's preregistration and report. Historical records remain available
for reproducibility and falsification, but they do not silently reactivate
retired source paths, deleted task files, or closed branches. Current user
instructions and higher-priority instructions remain controlling.
