# Local RSG4 component cost profile — 2026-10-01

This augments the completed dsba3 total-cost profile with an RTX 4090 local
breakdown. It describes execution cost, not accuracy. The exact RSG4
configuration and 16-shot support manifest are reused from `rsg4.yaml`.

- Dataset: MVTec AD 2 `test_public`, eight classes. For each class, use six
  normal and six anomalous queries spread deterministically across sorted file
  names. This is a latency sample, not the whole evaluation population.
- Hardware: local `GPU-0070b668-714f-bcbb-6c46-f0b68c9b7f6b` (RTX 4090),
  one process and one class at a time. Keep other GPU jobs off the card.
- Timing: synchronize CUDA at each measured component boundary. Exclude model
  construction/checkpoint loading and one backbone warmup from fit/query time.
  Fit is `method.fit`; query is `method.predict` at batch size one. Report fit
  seconds, query mean/p95 milliseconds, PyTorch peak allocated GiB.
- Fit attribution: image read, noise-reference estimation, image preprocessing,
  backbone forward, remaining feature work, support bank assembly, and
  leave-source-out calibration. Query attribution: image read, noise guard,
  preprocessing, backbone forward, remaining feature work, nearest-neighbor
  distance, layer fusion/score work, CQ, and residual map aggregation. Each
  measured interval is exclusive of nested measured calls. Any time not
  covered by an explicit interval is reported as residual, so percentages sum
  to 100%; the residual includes synchronization/instrumentation overhead.
- The components are wall-time attribution, not independently removable costs.
  GPU memory peaks overlap and are not additive; only total fit/query peaks are
  reported. The older dsba3 report remains the full-population four-arm
  comparison on different hardware.
- Save JSON per class and a combined Markdown/CSV summary under
  `results/local_runs/rsg4_component_cost_20261001/`. Retain the exploratory
  bad-only fabric pilot separately and exclude it from the balanced summary.
