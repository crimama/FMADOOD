# Results consolidation

Date: 2026-09-08

This directory is the compact preservation layer created before cleaning
`results/` and `results_remote/`. It preserves final-shaped quantitative JSON
and the decision-level interpretation already established by the experiment
reports. It is not a claim that every recorded run is valid.

## Files

- `quantitative_ledger.json`: content-deduplicated quantitative documents.
- `source_index.tsv`: original source paths, byte sizes, hashes, and aliases.
- `insights.md`: consolidated findings, validity boundaries, and known gaps.

The ledger selected JSON files whose names contain `summary`, `aggregate`,
`metrics`, `result`, `verdict`, `leaderboard`, or `comparison`. Exact duplicate
content was stored once. Sequences longer than 64 entries and nested mappings
larger than 256 entries were replaced by their count and canonical JSON
SHA-256; long numeric sequences also retain minimum, maximum, and mean. Scalar
metrics, verdicts, validity gates, thresholds, counts, hashes, and short
confidence intervals remain present.

## Coverage

- Candidate source files: 3,604
- Unique documents: 2,846
- Exact duplicate aliases removed: 758
- Original selected bytes: 1,608,427,871
- Compact ledger bytes: 35,955,999
- Parse failures: 1 empty or malformed checkpoint file

The parse failure is
`results/remote_runs/dsba3/psar_d1_dsba3_q95_reseal_20260829_v1/checkpoint/fabric_repeat/checkpoint_summary.json`.
It contains no recoverable quantitative result and is recorded only through
the source audit.

## Integrity

```text
830f5fb4db76d9b4290d3f13c4bab6c70a51e7874471df80f98bc8e751ce0cd2  quantitative_ledger.json
574e5b861d627ac4f16ea42aea14d3674e31885b163b641f1228934cad1b8ae2  source_index.tsv
```

The original-file SHA-256 stored per document is the identity used to collapse
host, merged, object, retry, and pullback duplicates.
