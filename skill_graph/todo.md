# Skill Graph TODO

- FlowTTE H+ DVT branch: keep NF latent reference as the retained main branch;
  do not pivot to raw layer-wise tiled NN or support-norm foreground split.
- Next structural diagnostic should use a stronger class-agnostic
  object/foreground prior than support feature norm, and should measure
  connected-component fragmentation before adding postprocessing.
- FlowTTE DVT branch: test constrained denoising operators before any alpha-heavy tuning.
  Priority order:
  1. query-side-only DVT diagnostic on full AD2 metrics,
  2. low-rank/truncated artifact subtraction,
  3. background/low-variance-position artifact fitting,
  4. support-only structural selector after the operator is constrained.
