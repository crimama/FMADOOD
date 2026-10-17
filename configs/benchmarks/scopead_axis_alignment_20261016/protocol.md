# Fixed axis factorial, 2026-10-16

Hypothesis: support-only layer/resolution relative scale correction may improve fused AUPRO@.05; no assumed benefit. Secondary pAUROC@.05 and per-class signs are mandatory. Seen AD2 public development, seed0, VOID fixed normal manifests 2/16-shot, frozen ViT-L/14 FP16, CQ off, original guard and effective 0.5*l11+0.5*l23+0.125*l5 fusion, bag/resolution mean, evaluator unchanged. No trainable parameters; no query/GT/labels enter fitting or scoring.

Equal weighting of all 48 source-LOO Q95 observations (floor1e-4). mu=mean(log u), a=exp(mean_rb(log u)-mu), c=exp(mean_bl(log u)-mu). N divisor1, L=a, R=c, LR=a*c; GM factors1. Full observation Q95 is a separate reference. Layer contrast=(L-N+LR-R)/2, resolution=(R-N+LR-L)/2, interaction=LR-L-R+N.

Budget: 48 new class cells, L/R/LR x shots2/16 x full8. Local essential tests and real-backbone small-query smoke precede sweep. Existing diagnostic none/Q95 controls reused only after scoring parity and support/query/GT/bank/backbone/evaluator/guard verification. Minimal matching controls only if reuse impossible. Stop after valid48 and analysis; no automatic crossfit. Maps and weights off; remote logs retained. Raw_above_support_q95 remains meaningful for axis arms; normalized>1 is not a tail exceedance, threshold is u/divisor. Sample low-FPR diagnostics do not replace full evaluator metrics.
