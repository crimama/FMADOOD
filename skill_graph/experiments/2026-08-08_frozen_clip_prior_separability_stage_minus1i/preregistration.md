# Stage -1i Frozen CLIP Prior Separability Preregistration

## Motivation

The metadata auxiliary-variable route is explicitly discarded. It is too image- and acquisition-specific, and can become equivalent to using an answer-bearing side channel. Stage -1i instead tests a narrower question: whether a frozen external visual prior from pixels alone separates normal photometric shift from local/broad synthetic defect evidence better than the current frozen DINOv3-H+ layer-7 reference.

This is a mechanism diagnostic, not a descriptor-swap method claim and not a SuperAD comparison.

## Negative-Evidence Intake

- Not a metadata, filename, split, or path-condition experiment.
- Not query adaptation, prompt tuning, support admission, threshold tuning, or post-hoc score fusion.
- Not a Normalizing Flow, Flow Matching, Diffusion, or DiT parameterization claim.
- Known failure basin: frozen descriptors may simply rescale kNN distances or become another passive feature-surgery retread. The gate therefore requires within-prior scale-normalized ratios against DINO, not raw CLIP score improvement.

## Design

- Dataset: MVTec AD2 single-image support-only diagnostic.
- Objects: `fruit_jelly`, `walnuts`, `vial`, `fabric`.
- Sources: exact fixed 16 `train/good` support images from `dinov3_noctx_support_paths.json`.
- Forbidden access: no `test_public`, no `test_private`, no `ground_truth`, no metadata, no filename condition, no split signal.
- Priors:
  - Candidate: frozen OpenAI CLIP ViT-L/14@336 from cached JIT `/home/hun/.cache/clip/ViT-L-14-336px.pt`; manually extracted final patch tokens, L2-normalized.
  - Reference: frozen DINOv3-H+ layer 7; L2-normalized.
- Conditions per support source:
  - `clean`
  - deterministic unseen photometric `U`
  - `U + D_local`
  - `U + D_broad`
- Memory/scoring:
  - Leave-one-source-out clean memory from the other 15 support sources.
  - Fixed `k=5` mean nearest-neighbor distance.
  - No sweep.

## Metrics

For each prior and held source, all values are normalized by that prior's clean `q995` score scale.

- `shift_inflation`: `q995(U) - q995(clean)`.
- `core_gain`: defect-core mean score increase, `(U+D) - U`.
- `outside_contamination`: outside-mask mean score increase, `(U+D) - U`.
- `selectivity`: `core_gain / max(abs(shift_inflation), eps)`.
- `geometry/noncollapse`: finite features, nonzero singular spectrum, effective rank >= 2, nonzero mean pairwise distance.

CLIP is compared to DINO only through scale-normalized ratios.

## Gates

Strict useful independent-prior gate:

- At least 3/4 objects satisfy all of:
  - CLIP shift inflation <= `0.80 * DINO` for both local and broad rows.
  - CLIP local and broad core gain >= `0.80 * DINO`.
  - CLIP selectivity >= `1.25 * DINO`.
  - no collapse and no invalid access.

Continuation gate:

- At least 2/4 objects satisfy defect retention plus selectivity >= `1.10 * DINO`; allows exactly one bounded follow-up.

Otherwise the verdict is `KILL_FROZEN_CLIP_PRIOR`.

## Output

- Object summaries under `results/remote_runs/dsba5/<RUN_NAME>/mechanism/<object>/summary.json`.
- Aggregate summary and object TSV under `mechanism/summary.json` and `mechanism/object_summary.tsv`.
- Access ledger must explicitly record `metadata_used=false`, `filename_condition_used=false`, and all test/GT access flags false.
