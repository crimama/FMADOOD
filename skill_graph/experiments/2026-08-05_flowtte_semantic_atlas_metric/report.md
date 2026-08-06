# FlowTTE SemanticAtlasMetric Stage -1 Report

Status: `COMPLETE / KILL_FOR_CLAIM / RETAIN_BASELINE1 / NO_CONTINUE / NO_PUBLIC_ACCESS`

## Conclusion

The semantic-local metric preserved Baseline1's synthetic local-defect contrast
and clean 1-NN geometry, but it did not remove enough held photometric shift to
justify continuation.  The proposed `A1_semantic_atlas_orth` reduced
shifted-normal inflation by only `0.6788%` macro against the preregistered
`10%` requirement, and `walnuts` regressed by `0.1467%`.

The central contribution hypothesis is therefore not supported.  Semantic
routing was unstable (`0.5274` macro fold-refit ARI versus `0.70` required),
the learned atlas captured less held-shift energy than the closed global-rank-4
comparator, and clean-tangent orthogonalization did not provide the required
safety advantage over the unorthogonalized atlas.  All-eight support replication
and every public/test evaluation remain blocked.

## Executed contract

- Run: `flowtte_semantic_atlas_20260805_v1`
- Host/container: `dsba5` / `hun_fsad_tta_012`
- GPU: host GPU 3, exposed as container `cuda:0`
- Duration: `2026-08-04T17:01:35Z` to `17:33:40Z` (`32m05s`)
- Objects: `fruit_jelly`, `walnuts`, `vial`, `fabric`
- Support: exact fixed SuperAD-selected 16 normal images per object
- Evaluation: four source-disjoint folds; support-only clean, held photometric
  shifts, and sealed synthetic local defects
- Prohibited access: no query adaptation, real anomaly, `test_public`,
  `test_private`, or ground-truth enumeration/opening

Synthetic defect retention in this report is a mechanism safety proxy.  It is
not public anomaly-detection performance and must not be reported as pAUROC,
F1, or real-defect evidence.

## Arm-level result

| Arm | Shift reduction | Defect retention | Shift-energy capture | Interpretation |
|---|---:|---:|---:|---|
| A0 Baseline1-core | `0.0000%` | `1.0000` | `0.0000` | frozen reference |
| A1 semantic atlas + orthogonalization | `0.6788%` | `0.9992` | `0.1623` | safe but too weak |
| A2 semantic atlas, raw direction | `0.6647%` | `1.0000` | `0.1839` | orthogonalization adds no required benefit |
| A3 global rank-4 | `1.1358%` | `1.0002` | `0.2686` | stronger shift capture but clean geometry fails |
| A4 shuffled semantic routing | `0.4773%` | `0.9985` | `0.0989` | much of A1 survives incorrect routing |
| A5 random rank-1 median | `-0.0007%` | `0.9998` | `0.0007` | learned directions are non-random |

The random control confirms that A1 learned a non-random transformation.
However, A4 retaining `0.4773%` of shift reduction and A3 outperforming A1 on
shift-energy capture prevent attribution to stable semantic-local routing.

## Proposed-arm result by object

| Object | Shift reduction | Defect retention | Positive folds | Disabled patch mass | Shift-energy capture |
|---|---:|---:|---:|---:|---:|
| fruit_jelly | `1.2865%` | `1.0000` | `4/4` | `1.06%` | `0.3330` |
| walnuts | `-0.1467%` | `1.0021` | `1/4` | `25.52%` | `0.0117` |
| vial | `1.5556%` | `1.0002` | `4/4` | `2.04%` | `0.2327` |
| fabric | `0.0199%` | `0.9943` | `4/4` | `0.00%` | `0.0719` |

The failure is not anomaly absorption: macro defect retention is `0.9992`,
the worst object remains `0.9943`, minimum clean top-10-neighbor retention is
`0.9847`, and minimum pair-distance Spearman is `0.9994`.  The failure is that
the safe local operator barely changes the nuisance component, especially on
`walnuts` and `fabric`.

## Preregistered gate result

| Gate | Result | Evidence |
|---|---|---|
| Baseline validity/audit | PASS | finite, deterministic, nonconstant, source-disjoint |
| Anti-absorption | PASS | `0.9992` macro defect retention |
| Clean geometry | PASS | neighbor retention `>=0.9847`, Spearman `>=0.9994` |
| Shift correction | FAIL | `0.6788%` versus `10%`; `walnuts` negative |
| Atlas robustness | FAIL | ARI `0.5274 < 0.70`; disabled mass `25.52% > 20%` |
| Identifiability | FAIL | A1 energy `0.1623`, below A3 `0.2686` |
| Orthogonalization value | FAIL | A1 retention is not `+0.03` over A2 |
| Non-retread versus global rank-4 | FAIL | A1 supplies neither required shift nor safety margin |
| Causal-control exclusion | PASS | random q95 and shuffled control do not pass full gate |
| Full proposal gate | FAIL | verdict `KILL_FOR_CLAIM` |

Fold-refit cluster ARI by object was `0.4346` for fruit_jelly, `0.4247` for
walnuts, `0.7512` for vial, and `0.4991` for fabric.  Only vial reached the
atlas-stability target.

## Research interpretation

- The safe-projection idea repairs the paired-loss experiment's main failure:
  it does not erase synthetic local anomaly contrast.
- Safety alone is insufficient.  A rank-1 local contraction with beta `0.25`
  removes too little of the actual shift seen by latent 1-NN.
- Increasing beta, sweeping K, or adding more loss weight would revisit the
  same safety-versus-correction tradeoff without fixing chart instability.
- The next credible branch should first obtain stable patch roles or semantic
  correspondences from support images, then learn nuisance directions inside
  those stable charts.  It should not continue by tuning this atlas instance.
- Until such a new representation is preregistered, `Baseline1` remains the
  retained method and raw latent Euclidean 1-NN remains the reference scorer.

## Reproducibility and custody

- Frozen protocol: [preregistration.md](preregistration.md)
- Machine verdict:
  `results/remote_runs/dsba5/flowtte_semantic_atlas_20260805_v1/protocol_verdict.json`
- Access ledger:
  `results/remote_runs/dsba5/flowtte_semantic_atlas_20260805_v1/mechanism/access_ledger.json`
- Compact artifact manifest SHA-256:
  `0a9e5478ec00eff126cc987dec2f9ee107134aedde850e1da029046b267f343`
- All 22 payload entries passed local SHA-256 verification after excluding the
  legacy manifest's self-reference.
- A local independent reaggregation of the four object summaries produced a
  byte-identical normalized verdict SHA-256:
  `081877b6bdb869217f83c13f921e36a59a3f3525700eda80c39a60046a43f044`.
- Focused verification: `22 passed`; Ruff, Python byte-compilation, and shell
  syntax checks passed.

The controller was corrected after execution so future manifests exclude their
own output file, and its legacy pull path now ignores only that self-reference.
The completed artifact still records the old impossible self-checksum entry;
all 22 actual payload hashes match, and no experimental result was modified.
