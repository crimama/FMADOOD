# Order-aggregation rank diagnostic (calibration x order aggregation)

## Question
Does order aggregation need per-bank Q95 calibration? Hypothesis: without
calibration, the resolution with the largest raw score range takes the top rank
at almost every pixel, so the Rényi order weights act like a fixed resolution
weighting. With Q95 calibration, the top rank moves to the resolution where the
defect stands out, so rank carries defect evidence.

## Arms (AD2 test_public full8, seed 0, 2/16-shot)
- q95: final ScopeAD (Q95 + deep mean + 0.125 shallow + Rényi order + CQ99).
  Must reproduce the recorded final AUPRO (59.92 / 60.16) as a parity check.
- none: identical except statistical_normalization: none.
Four configs x eight classes = 32 class runs. No tuning, no selection.

## Diagnostic (evaluation-only)
Per-resolution native maps are captured just before resolution aggregation.
After the method returns its prediction, the GT mask is read for this
diagnostic only. Recorded per class:
- rank-position counts of each resolution on normal and defect pixels;
- top-1 share, normal top-1 dominance, total variation between defect and
  normal top-1 distributions, effective Rényi weight per resolution;
- top-1 share by defect region size (<0.1%, 0.1-1%, >=1% of image area);
- agreement between the top-1 resolution and a scale-free evidence-best
  resolution (region mean divided by the same image's normal-pixel Q95).
Images routed by the noise guard (fewer than four resolutions) are counted and
excluded from rank statistics. GT never enters fitting or prediction.

## Execution
scripts/order_rank_diagnostic.py through scripts/run_experiments.sh on dsba3
hun_fsad_tta_012, the four assigned GPU UUIDs, staged immutable release.
Primary metric remains seg_AUPRO_0.05; the diagnostic JSON is written next to
metrics.json as order_rank_diagnostic.json.

