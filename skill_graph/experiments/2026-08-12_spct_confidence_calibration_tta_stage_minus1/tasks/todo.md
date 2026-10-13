# Tasks

- [x] Review SPCT evidence and negative-evidence boundaries.
- [x] Freeze arms, nulls, custody, metrics, and independent gates.
- [x] Implement core gate/calibration/TTA primitives.
- [x] Implement runner, strict aggregator, and remote wrapper.
- [x] Pass unit, integration, CUDA parity, lint, compile, and shell checks.
- [x] Diagnose and replace the zero-result v1 CPU-stalled execution without
      changing experiment semantics.
- [x] Submit all eight object/budget shards as immutable v2 to dsba3/dsba5
      GPU3 queues.
- [x] Execute all eight object/budget shards.
- [x] Aggregate, interpret, document, and remove remote source snapshots.
- Result: valid 8-shard aggregate, `VALID_NO_SIGNAL`; all four confidence /
  conformal / synthetic / TTA mechanisms failed their independent gates.
- Surviving diagnostic: full support improved B0/B1 pAUROC by
  `+0.064629/+0.080122`, 4/4 wins with positive bootstrap lower bounds.
- Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`; promote the absolute-best
  usable full-support B1 arm directly to an all-eight TESTpublic comparison
  against matched full-memory DINO kNN/RN-FMLK. Support64 interpolation is not
  required.
