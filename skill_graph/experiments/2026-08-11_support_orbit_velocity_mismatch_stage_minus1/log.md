# SOVM Stage -1 Log

## 2026-08-11 — Design frozen

- Completed negative-evidence intake against SO-CFM correction, score-aligned
  residual projection, and query-orbit persistence.
- Froze the affine-path squared velocity mismatch, `t²` time weighting, minimum
  anchor/code aggregation, all eight signed nuisance codes, and oracle ceilings.
- Split execution into Stage M normal-only evidence and conditional Stage G
  score-aligned synthetic ranking.
- No code, dataset query, GPU process, or result artifact was created.

## 2026-08-11 — Stage M completed with oracle-only / no-continue verdict

- The merged aggregate is valid and final: `validity_ok=true`,
  `benchmark_comparable=false`, `strict=false`, and `continuation=false`.
- `M0_sovm_enum` reached macro pAUROC `0.674283`, with local/broad
  `0.741302/0.607263` and normal FPR `0.045711`.
- `B0/B1/B2` deltas were `-0.000816/-0.002893/-0.002655`; `N0/N1/N2/N4`
  stayed non-positive, while `N3_spatial_source_shuffle` was strongly positive
  at `+0.145896` with 4/4 wins and CI `[0.114077, 0.177550]`.
- Selected-code entropy was `0.599870` and the max selected-code fraction was
  `0.713889`, which supports a code-collapse reading rather than broad nuisance
  disentanglement.
- Engineering fixes used to stabilize the run were exact parity benchmarking on
  GPU-resident data, clean cache handling, remote tar/stdin/same-file transfer,
  deterministic blur no-op fallback, and the aggregate ratio contract. These are
  execution enablers, not method evidence.
- Supplementary parity runs are complete across all four objects. The largest
  reported `M0` absolute difference is `1.99407044654e-7`; the largest pAUROC
  difference is `3.70415149442e-6`; the largest gap difference is
  `3.15085349778e-6`.
- Single-object controller exit code `2` was expected missing-object aggregate
  postprocessing, not shard failure.
- Compact parity table:

  | Object | Primary M0 | Parity M0 | Max pAUROC diff | Max gap diff | Artifact |
  | --- | ---: | ---: | ---: | ---: | --- |
  | can | 0.6625654754825345 | 0.6625652760754899 | 2.53229115332e-6 | 1.01340554559e-6 | `results/remote_runs/dsba5/sovm_stage_minus1_dsba5_20260811_v9/shards/can/summary.json` |
  | fabric | 0.6442480186358903 | 0.644247890417822 | 3.70415149442e-6 | 3.15085349778e-6 | `results/remote_runs/dsba3/sovm_stage_minus1_dsba3_20260811_v2/shards/fabric/summary.json` |
  | vial | 0.6903429565503281 | 0.6903429565503281 | 1.29239040536e-6 | 6.46195202569e-7 | `results/remote_runs/dsba3/sovm_stage_minus1_dsba3_20260811_v1/shards/vial/summary.json` |
  | walnuts | 0.6999743177947801 | 0.6999743177947801 | 1.47511423276e-6 | 7.37557116381e-7 | `results/remote_runs/dsba5/sovm_stage_minus1_dsba5_20260811_v8/shards/walnuts/summary.json` |
- Report:
  `report.md`.
