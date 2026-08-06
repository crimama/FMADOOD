# CF-NTR Stage -1 attempt ledger

## v1 — invalid infrastructure attempt

- Run: `cfntr_stage_minus1_20260718_supportonly_v1`
- Snapshot SHA-256:
  `ba0177b0cf941f78723fd7c887bc57a45daf308fc4fafeb382c5810e476b6ac4`
- Result: `INVALID_RUN` under G5; no object completed and aggregation was not
  executed.
- Failure: `mask_to_patch_grid` supplied a binary `{0,1}` uint8 mask to
  OpenCV `INTER_AREA`. Fractional occupancy below 0.5 was rounded to zero,
  erasing thin scratch defects before pixel metrics.
- Scope: an exhaustive geometry-only replay found exactly 16/64 degenerate
  masks per object (128/512 overall), all eliminated by the repair.
- Repair: use float32 occupancy before applying the frozen `> 0` any-native-
  pixel-overlap threshold. This restores the preregistered estimator rather
  than changing its threshold or gate.
- Verification: 512/512 sealed defect masks retain both classes; focused and
  related suite `40 passed`; Ruff, compilation, shell syntax, and diff checks
  pass.

## v2 — invalid stimulus-validity attempt

- Run: `cfntr_stage_minus1_20260718_supportonly_v2`
- Snapshot SHA-256:
  `e2c94d90086335074a42358620f6b16d54577320e87aae9301bf5b460c033e24`
- Allocation: dsba3 GPUs 0–3, two frozen objects per GPU.
- Result: all eight objects and all four GPU groups completed, then G5
  invalidated aggregation. No formal G1–G4 conclusion is permitted.
- Failure: all values were finite and row/count parity was exact, but 912
  `fruit_jelly` and 608 `vial` maps were constant. The counts factor as
  `48 × 19` and `32 × 19`: every arm was constant in a small set of raw-zero
  defect cases.
- Diagnosis: 80 zero-energy rows came from no-op synthetic stimuli, mainly
  Gaussian blur sampled on uniform regions and a few CutPaste cases erased by
  the assigned acquisition/downsampling. Shift-only rows had no zero raw p95.
- Compact invalid artifacts:
  `results/remote_runs/dsba3/cfntr_stage_minus1_20260718_supportonly_v2.invalid`.

## v3 — preregistered stimulus-validity amendment

- Support paths, folds, axes, amplitudes, layers, rank, random-null count,
  bootstrap counts, G1–G5 thresholds, and maximum verdict remain unchanged.
- The generator deterministically retries only when the synthetic image is a
  no-op in clean or assigned-shift pixels. The runner retries with a sealed
  alternate seed only when the raw score map is spatially constant in any
  layer/acquisition.
- Retry selection cannot inspect physical/PCA/random corrected scores,
  AUROC/F1, anomaly retention, or any gate statistic.
- This is an implementation-corrected diagnostic, not an exact byte-for-byte
  rerun of v2; v1/v2 remain invalid and are not pooled.
- Run: `cfntr_stage_minus1_20260718_supportonly_v3`
- Snapshot SHA-256:
  `e368cc1cc2945c54f6dc39a3699c6d500a1eee3b4612f8cef8b02ec47f40d7f0`
- Result: valid all-eight run; G1/G4/G5 pass, G2/G3 fail.
- Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`.
- G5: all 136,192 maps finite and nonconstant; exact 3,072 shift rows,
  4,096 defect rows, and 512 retained masks.
- Retry audit: 10 pixel-location retries, zero feature-level retries.
- Cleanup audit: compact results retained, forbidden dense directories zero,
  and the isolated v3 source snapshot removed after completion.
- Report: `report.md`.
