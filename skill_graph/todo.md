# Skill Graph TODO

- FlowTTE object/foreground prior: hard RGB/support-score suppression is killed
  for the main branch. Do not continue fixed-parameter foreground-prior sweeps.
- Next diagnostic should model score-field uncertainty or component
  fragmentation while preserving continuous anomaly evidence, not pre-threshold
  suppressing it with a binary objectness mask.
- FlowTTE H+ DVT branch: keep NF latent reference as the retained main branch;
  do not pivot to raw layer-wise tiled NN or support-norm foreground split.
- FlowTTE DVT branch: test constrained denoising operators before any alpha-heavy tuning.
  Priority order:
  1. query-side-only DVT diagnostic on full AD2 metrics,
  2. low-rank/truncated artifact subtraction,
  3. background/low-variance-position artifact fitting,
  4. support-only structural selector after the operator is constrained.
