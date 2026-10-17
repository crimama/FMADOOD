# Explicit GT-optimized oracle scale diagnostic
User authorized direct oracle scale fitting to test remaining calibration headroom.
Run after ALL lanes of scopead_statistical_followup_20261016 complete; stop on
predecessor failure. Existing generic experiments.py runner, custom CLI entrypoint.
Fixed seen-public AD2 test_public, seed0, normal VOID support2/16, all8 classes.
16 runner cells. Each cell runs unchanged normal-only Q95 pipeline once,
then an isolated post-prediction analysis uses EVALUATION GT.
NOT final evaluation; NOT deployable normal-support-only calibration.
48 positive multipliers per class/shot, one per4 resolutions x4bags x3layers.
All query images share the SAME48 multipliers; no per-image/condition/GT-mask
adaptive scale, no offsets, no spatial calibration. Positive diagonal scales
leave common distance extraction/backbone/banks intact; fixed linear0125 formula.
This rescales Q95-normalized scores, equivalently divisor=q95/multiplier.
Optimization uses seeded samples: <=128 normal pixels per image with importance
weight equal to its normal pixel count/sample count; <=16 pixels per GT
8-connected defect region, region sample weights summing1. Minibatch features
are bilinearly sampled exactly at original-image pixel coordinates.
Objective sampled AUPRO@.05 with float16 score ties and same PRO integration.
Differential evolution,48 dimensions, log-multiplier bounds[-log8,log8],
popsize2 (96population),20generations, seed0, Q95 vector x0=zeros, polishoff.
At most2017 sampled objective evaluations; preserve baseline + top3 candidates.
Final selection uses FULL original-resolution pixel AUPRO@.05 on SAME public
population among Q95 ORIGINAL and up to3 oracle candidates (+reconstructedQ95
for parity audit). Secondary pAUROC@.05 may decrease, all values retained.
The best selected metric cannot fall below Q95 because its original result is
included. Record whether an actual optimized candidate beats it.
Finite bounded optimization is NOT a proven global optimum/theoretical upper
bound. A gain shows achievable headroom; no gain does not rule it out.
Native normalized observation maps remain RAM-only for the current class.
No dense map/cache/weights export. Save48 scalar multipliers, corresponding
divisor scales, objective settings, full candidate metrics, baseline metrics,
Q95 reconstruction error, source/config/manifest provenance and label-use flags.
Normal method fit/predict receive no dataset/GT/labels; GT access is isolated in
post-prediction analysis. The whole final oracle diagnostic uses evaluation GT
for optimization AND candidate selection; never group with standard comparisons.
FP16 backbone, 4resolutions, linear0125, CQoff, frozen support and metric contract.
Stop if routing changes observation scope or reconstruction map error>2e-5 or
Q95 AUPRO parity error>3e-5 (absolute fraction). Record these errors explicitly.
Per-cell timeout10800sec. Four existing dsba3 GPU UUIDs and same container.
Prioritize this diagnostic after already authorized follow-up; no recursive
search expansion. Retain all results, including Q95-selected/no-gain cases.
