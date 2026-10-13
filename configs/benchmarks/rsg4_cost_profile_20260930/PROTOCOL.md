# RSG4 inference cost profile — 2026-09-30

Registered before the cost runs. This is a descriptive runtime and memory
profile for Table 13 of `PaperWorks/Outline_v14.1.md`; it is not a performance
comparison or a method-selection experiment.

## Question and measurement boundary

Measure support fitting time, per-query mean and p95 latency, and fit/query peak
VRAM for RSG4, a deep-only profile, a single-1120 profile, and SuperAD. The
expected cost ordering is that four-resolution RSG4 uses more compute than its
deep-only and single-resolution arms; timings and memory are descriptive, with
no pass/fail performance threshold.

- Support fitting is the complete `method.fit(...)` call. It includes support
  image reading, preprocessing, feature extraction, memory/index construction,
  and support-derived calibration. Support selection, model construction and
  checkpoint loading, and the configured backbone warmup are outside this
  timer.
- Query latency surrounds `method.predict_batch(...)` at batch size 1, with
  CUDA synchronization before and after every query. It includes query image
  reading/preprocessing, feature extraction, nearest-neighbour search, fusion,
  and map postprocessing. Evaluator metrics, ground-truth loading, and artifact
  writes are outside the query timer.
- Fit and query VRAM each report both PyTorch's reset peak allocated memory and
  the selected GPU's peak total used memory sampled by `nvidia-smi` every
  100 ms. The latter includes allocations made outside PyTorch's allocator.

## Controlled conditions

- Dataset/split: MVTec AD 2 full8 `test_public`, seen development evidence;
  labels and masks are evaluator-only and never enter fitting or prediction.
- Seed/support: seed 0 and the identical
  `configs/methods/VOID_support_16.json` 16-normal-image manifest per class for
  all four arms. All 32 class runs must report identical support identities
  and content hashes.
- Hardware: existing dsba3 container `hun_fsad_tta_012`, GPU 0 only,
  `GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f` (RTX A6000 48 GiB); all four
  methods run sequentially on that GPU. No checkout changes during execution.
- Common pipeline: frozen DINOv2 ViT-L/14, FP32 feature extraction, one
  warmup iteration, streaming evaluator, batch size 1, no anomaly maps or
  fitted-weight outputs. Model/checkpoint loading is not timed.
- The common feature extractor has `half_precision=False`; the evaluator's
  `mvtec_ad2_float16_streaming_v1` metric contract does not change inference
  feature precision.

## Method declarations and planned deviations

| Arm | Feature layers | Query input resolutions | Reference selection |
|---|---|---|---|
| RSG4 | `[5, 11, 23]` | `[448, 672, 896, 1120]` | 256 tokens/support/resolution, 4 bags; k=1; tail-scale residual-spike fusion; k=9 low-resolution noise guard |
| Deep-only | `[11, 23]` | `[448, 672, 896, 1120]` | 256 tokens/support/resolution, 4 bags; k=1; tail-scale mean of deep layers; shallow layer and noise guard disabled |
| 1120-only | `[5, 11, 23]` | `[1120]` | 256 tokens/support/resolution, 4 bags; k=1; tail-scale residual-spike fusion; noise guard disabled because there is no lower resolution to route to |
| SuperAD | `[5, 11, 17, 23]` | `[672]` | All selected feature tokens; k=1 normalized-L2 FAISS indices |

RSG4 and both RSG4-derived arms use the same input support manifest. The
deep-only row is an isolated implementation profile that extracts only the two
deep layers; it can therefore be faster than the existing `shallow_mode: off`
score diagnostic, which retains layer 5 for matched diagnostics. The
1120-only row keeps 256 tokens per support per branch and consequently has
one-quarter of RSG4's bank entries; it is not the E12 1024-token
capacity-matched arm. SuperAD's declared input/layer configuration and
all-token bank remain method-specific.

RSG4 has 4,096 reference vectors per layer/resolution/bag component bank and
48 component banks (196,608 vector slots per class). Deep-only has 32 such
component banks (131,072 slots/class), and 1120-only has 12 (49,152
slots/class). SuperAD's per-class bank size is recorded from its fit metadata
because its token count follows its 672-input grids.

The existing SuperAD method declaration uses two backbone warmup iterations.
This descriptive profile uses one warmup iteration for every arm; warmup is
excluded from fit and query timers.

The RSG4 config uses `reference_matching` with the canonical RSG4 settings;
this avoids `fusion_factorial`'s prediction-audit collection overhead while
preserving the gated residual score and noise-guard path. Fusion diagnostics
and acquisition-stratum diagnostics are disabled for all arms during timing.
All arms retain the same common evaluator; its metrics are not a selection
criterion.

## Budget and stop rule

Run four configuration stages sequentially with `scripts/experiments.py run`,
one worker and all eight classes per stage (32 class executions total). Retain
remote logs and pull only the metrics, config, manifest, status, and profile
report provenance. Complete only when all 32 class executions have finite
fit/query timings, valid GPU-memory samples, and matching support identities.
Do not retry a method for a low/slow result; a safe retry is allowed only for a
recorded instrumentation or infrastructure failure, using the same GPU and
fixed code snapshot.
