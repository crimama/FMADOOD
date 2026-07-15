# DARC AD2 Raw-Ladder Performance Pilot

Date: 2026-07-11  
Status: v2 contract amended after an invalid pre-metric v1 execution; no label
metric has been computed

## Claim, Evidence, Boundary, Positioning

- Claim under test: detached local alignment and robust reconstruction can improve
  the actual `can` anomaly ranking/localization, not only synthetic thin-cue
  diagnostics.
- Evidence motivating the run: the 672 H+DVT MLP anchor has a severe `can` F1
  failure, while native-resolution analysis shows an improvement gradient as
  the token spacing shrinks.
- Boundary: this is a performance-oriented diagnostic, not frozen Gate 3 and not
  a SuperAD/SuperADD superiority claim. The terminal coarse-confidence fusion is
  excluded because its confidence field and coarse-evidence calibration have no
  implemented or frozen definition.
- Positioning: the pilot isolates the implemented correspondence ladder before
  adding an undefined fusion rule.

## Frozen Pilot

- Dataset: MVTec AD2 single-image at
  `/home/hunim/Volume/DATA/mvtec_ad_2`.
- Initial object/cell: `can`, P16-random seed `0`, fold `0`, full
  `test_public/good+bad` population after a fixed-v2 two-image execution smoke.
- Support access: the seed-0 P16 permutation is frozen before image decode; fold
  0 uses its 12 memory images and does not expose the four held-out paths to the
  scorer.
- Backbone/features: existing DINOv3 ViT-H+ layer 7 native 1024-crop micro
  features and layer 23 coarse geometry; exact current correspondence settings.
- Arms: raw `G0`, `L0`, `L1`, and `R1` cosine-residual maps. The existing common
  G0 fallback is retained for tokens with fewer than three valid local supports.
- Aggregation: the initial metric cell is one fold. A later promoted cell must
  average exactly four fold maps in float64 and cast once to float32.
- Post-processing: none for primary continuous metrics. Shared morphology is
  diagnostic only.
- Evaluator: the common native-grid evaluator over full public good+bad images.
- Primary reported metrics: pixel `pAUROC@0.05`, AP, oracle F1, and oracle
  component recall. Image-disjoint fixed raw F1 is reported as a transductive
  threshold diagnostic.
- The so-called fixed F1 uses cross-fit `test_public/good` images and is
  therefore a transductive normal-labeled diagnostic, not deployment evidence.

## Decision Contract

- `CONTINUE_DIAGNOSTIC`: on full `can`, `R1-G0 oracle F1 >= +0.02` with
  `R1-G0 pAUROC@0.05 >= -0.01`, or `L1-L0` improves both oracle F1 and AP.
- `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`: the ladder has a smaller positive
  gradient but misses the continuation threshold; inspect failure maps once.
- `KILL_FOR_CLAIM / NO_CONTINUE`: R1 does not improve G0 on oracle F1, AP, or
  component recall, or local-map coverage collapses.
- A positive one-fold result promotes exactly one next experiment: four-fold
  seed-0 `can`, followed by the frozen five-object shadow only if the four-fold
  result survives.

## Comparability

- Historical H+DVT MLP (`0.836739/0.527427` macro) and SuperAD-16
  (`0.765802/0.385534` macro) are context only because their support selections
  and method conditions differ.
- Same-condition coarse fusion remains unevaluated until its normal-only
  calibration and confidence definition are fixed without looking at AD2
  labels.

## Pre-Full Operational Amendment

- The pre-fix v1 smoke produced finite maps but is invalid for metrics because
  its RANSAC seed included the `good|bad` directory. It is retained only as an
  execution trace. A new fixed-code v2 smoke is required before the full cell.
- The full `can` population is scheduled as four deterministic, disjoint query
  shards using the index modulo four over the frozen good-then-bad path order.
- Every shard independently uses the identical seed-0 P16 identity, fold-0
  12-image memory, backbone, scorer, and raw-map rules. Sharding changes only
  scheduling; no map is averaged across processes and the common evaluator
  merges the four non-overlapping map roots back into the exact full population.
- The scorer query identity is `object/content-SHA256`, excluding the
  `good|bad` directory. Consequently RANSAC sampling is content-bound and
  population-neutral; source paths remain in the manifest only for map/evaluator
  provenance.
- Each query/fold also persists pre-stitch scorer-token fallback fraction,
  support-validity histograms, accepted-registration counts, and audit hashes;
  these are label-free coverage diagnostics.
