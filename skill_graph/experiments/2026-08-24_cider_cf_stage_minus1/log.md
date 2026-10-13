# CIDER-CF Stage -1 log

- 2026-08-24: preregistered before results; public/test/GT remains closed.
- 2026-08-24: froze dsba3 GPU map can:0, fabric:1, vial:2, walnuts:3.
- 2026-08-24: run `cider_cf_stage_minus1_dsba3_all4_20260824_v1`
  failed uniformly before metric output because fixed-container NumPy lacks
  `np.trapezoid`; froze the equivalent `np.trapz` compatibility fix and moved
  to immutable run ID v2.
- 2026-08-24: v2 reached covariance validation but exposed a non-scaling exact
  eigendecomposition (1024x1024 for each of 256 parts). Replaced it before any
  result was emitted with the mathematically conservative Weyl/spectral-norm
  condition bound; v2 was terminated by exact run-path match and v3 frozen.
- 2026-08-24: v3 profiling showed the same cubic cost remained in each fold's
  1024-D candidate PCA. Before any summary existed, froze a deterministic
  rank-64 covariance range sketch followed by the unchanged family-consensus
  ranking, terminated v3 by exact run-path match, and assigned v4.
- 2026-08-24: v4 profiling isolated outside-mask null oversampling as the
  remaining cost. Before summaries, froze an equal deterministic 512-patch
  cap per family, terminated v4 by exact run-path match, and assigned v5.
- 2026-08-24: v5 completed all four GPU shards and aggregation. Custody passed,
  but generator probe, hard-null margin, and walnuts nuisance specificity
  failed fatally. Final verdict: `KILL_FOR_CLAIM`; TESTpublic remains closed.
