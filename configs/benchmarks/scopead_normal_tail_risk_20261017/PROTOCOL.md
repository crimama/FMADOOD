# Normal-only spatial tail-risk12 calibration
2026-10-17 user authorized experiments following oracle axis decomposition.
Hypothesis: same-coordinate normal false-positive tails can estimate useful
relative weights of resolution/layer observations without anomaly GT.
This is a NEW development hypothesis; oracle gains do not establish it.
Fixed AD2 test_public seen-development,seed0,normal VOID2/16supports,all8classes;
FP16 DINOv2-L,layers5/11/23,4resolutions448/672/896/1120,4random bags cap256,
same1NN distances,linear0125,CQoff,existingguard and float16 streaming evaluator.
Five arms: unchanged Q95; score_free; score_fixed_layer; excess_free;
excess_fixed_layer. Budget5x2shotsx8classes=80cells,10800sec timeout percell.
Normal-only fit, unchanged banks/query/backbone. All support/query/GT/bank hashes
must match Q95 before paired metrics. Primary AUPRO@.05, secondary pAUROC@.05.
All outcomes retained, no adaptive extra searches or premature method claim.

Fitting changes only:
During the SAME support backbone extraction as baseline, gather normal feature
tokens at16x16 relative-image cell centers, using the nearest token at each
resolution (approximate spatial alignment, not bilinear score interpolation).
After each existing bag is built, query these probes against that bank while
masking ALL tokens of their source image. Q95 scales remain the original
source-LOO scale estimates from the existing random bank tokens.
No extra backbone extraction. Extra source-LOO nearest-neighbor computation and
CPU LP fitting are declared additional method fitting cost.
Normalized probe scores average4bags, yielding[Snormal,256points,4res,3layers].
Each image has same probe count, hence equal image weight. Tokens are dependent;
no independent-token sample-size/FPR coverage guarantee. Query/GT/labels never
enter fitting or prediction.

Risk signal: score=Q95-normalized bag-mean probe score; excess=max(score-1,0).
Note excess is AFTER bag mean, matching intended12-condition shared-bag scale.
Objective CVaR95 of the fused NORMAL signal (coefficients.125,.5,.5,mean over4res).
CVaR represented as min_t t+mean(max(Xw-t,0))/.05. Solve its exact finite-sample
linear program with HiGHS,positive multiplier bounds[1/8,8].
Free arms constrain sum(all12weights)=12; fixed_layer arms constrain each layer's
sum across4res=4, preserving original total layer coefficients, including0.125.
This is positive diagonal calibration/equivalent observation weighting; no
new query dependence. Standard CVaR is not claimed novel; statistical reference
Rockafellar and Uryasev, Optimization of Conditional Value-at-Risk,
Journal of Risk2000,doi:10.21314/JOR.2000.038.
LP timeout120sec, solver failure invalidates run. All-zero signal uses identity
and records fallback. Record achieved empirical risk, constraints, multipliers,
normal source/probe counts. No guarantee normal-tail minimization preserves
defect sensitivity: selection biases/easy-observation collapse are hypotheses
to inspect if performance drops. Source-LOO vs full-bank distribution mismatch
and limited support remain unresolved limits.
Backbone trainable params0;12normal-fitted calibration weights (11free effective
degrees or9fixed-layer effective degrees). No GT-tuned fitting hyperparameters.
All data are already-seen public development data; labels only common evaluation
and post-hoc analysis. Not a final/independent evaluation.

Existing dsba3 hun_fsad_tta_012 container,assigned4GPU UUIDs,immutable stage-only
release; active checkout and existing jobs retained. No dense maps/weights.
Capture metrics/config/manifests/logs. Existing experiment runner and collector;
register affected canonical MD+CSV rows after completion with verified provenance.
Stop after80cells. Prior oracle axis results are GT-using diagnosis, kept separate.
