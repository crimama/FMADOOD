# Streaming segmentation evaluator

## Motivation and boundary

Native-resolution anomaly maps caused avoidable disk growth.  This change keeps
the existing SuperAD spatial path (low-resolution DINO grid followed by native
resolution interpolation) but changes the lifetime of the interpolated map:
one inference mini-batch only.

## Implementation

- `StreamingSegmentationAccumulator` consumes score maps and GT masks batch by
  batch and retains three fixed 65,536-bin histograms only.
- Pixel pAUROC@0.05, oracle F1, and AUPRO@0.05 are finalized once per object;
  partial batch metrics are cumulative previews and are never averaged.
- The common FMAD pipeline owns batching, GT comparison, and metric
  accumulation; model adapters own object preparation and prediction only.
- FMAD SuperAD/RN-FMLK defaults to streaming evaluation and writes compact
  `progress.json` and final `metrics.json` artifacts instead of `anomaly_maps/`.
- The versioned metric contract is `mvtec_ad2_float16_streaming_v1`; query
  order is frozen as `(basename, anomaly_type)` and persisted outputs identify
  the contract explicitly.
- `run.py` remains the single composition root. New models are registered and
  selected through configuration instead of adding another `run_*.py` script.
- Pixel scores are explicitly quantized to float16 order codes.  Streaming
  results therefore require a same-mode baseline and must not be mixed with
  historical float32 evaluator values without a parity audit.

## Verification

- `pytest -q tests/test_streaming_segmentation.py tests/test_fmad_streaming_contract.py
  tests/test_fmad_swappable_pipeline.py tests/test_run_config_precedence.py`
  returned `12 passed`.
- `compileall` passed for the FMAD package, accumulator, CLI, and new tests.
- `git diff --check` passed for modified tracked files.

## Verdict

`KEEP` as a storage/runtime infrastructure change for same-mode SuperAD and
RN-FMLK evaluation.  Scientific comparisons remain bounded by the declared
float16 quantization and require identical evaluator mode on every compared arm.
