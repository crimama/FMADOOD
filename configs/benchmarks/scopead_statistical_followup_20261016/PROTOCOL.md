# Statistical calibration follow-up
User requested automatic sequential experiments after the moment-normalization suite.
Start only after all four predecessor status files report complete. Stop instead
of starting if any predecessor fails. Existing generic runner implements this gate.
Fixed seen-development budget: AD2 test_public, seed0, VOID normal support2/16,
all8 classes, 5 arms x2 shots x8 classes=80 cells; timeout10800 sec per cell.
Q95 fresh control, eb_mean_scale, eb_rms_scale, weibull_bank_q95, covariance_q95.
Primary AUPRO@.05; secondary pAUROC@.05; higher better. Not final evaluation.
FP16 DINOv2-L, layers5/11/23, four448/672/896/1120 resolutions,
fourbags cap256, linear0125, CQoff, existing guard and float16 evaluation fixed.
No dense maps or weights; no labels/query data for calibration.
Hypotheses and boundaries:
1) EB mean/RMS: support-image moment variation estimates log-scale uncertainty
via delta method, using ddof1 between images and n=2/16 (not token count).
Within each layer, pool16 resolution/bag observations. Log prior mean=mean(logs);
prior variance=max(var(logscales,ddof1)-mean(measurement_variance),0).
Shrinkage=measurement_variance/(prior_variance+measurement_variance), zero when
both vanish. This is an approximate normal-means EB adaptation; shared images
and nonexchangeable resolutions can violate its assumptions. No optimality claim.
Prior pooling can erase real resolution differences; retain all class outcomes.
2) Weibull bank correction: normal source-LOO1NN cosine distances fit Weibull
MLE, loc0. Adjust existing Q95 by ((S-1)/S)^(1/shape) for S normal images,
reflecting the Poisson-neighbor minimum-distance scaling hypothesis. Same256
tokens/source means nominal bank ratio equals image ratio. This uses cosine
distance directly; Weibull shape is NOT called intrinsic dimension (cosine is
squared Euclidean for unit features). Correlated tokens and heterogeneous normal
patterns may invalidate Poisson scaling. Constant/failed fit uses Q95 and records
fallback. Same1e-4 denominator floor. No FPR guarantee.
3) Covariance_q95: Q95 marginal calibration plus changed fusion. Source-image
mean calibrated LOO scores produce n_images x16 observation matrix per layer.
Ledoit-Wolf covariance, nonnegative simplex minimum-variance weights; multiply
each observation score by16*w before existing bag/resolution averaging.
Within-layer weights sum1; final deep mean+0.125shallow retained. This IS a fusion
ablation, not calibration-only. No new token kNN. Extra cost is fitting small
covariance matrices. Image means need not represent pixel-level score covariance.
Constant scores use uniform weights; optimizer failure invalidates the run.
Citations: Smyth2004 doi:10.2202/1544-6115.1027; SchaeferStrimmer2005
doi:10.2202/1544-6115.1175; LedoitWolf2004 doi:10.1016/S0047-259X(03)00096-4;
LevinaBickel NIPS2004 Maximum Likelihood Estimation of Intrinsic Dimension.
These are proposed adaptations, not reproductions of their original tasks.
Remaining surveyed methods (conformal held-out calibration/local neighbor
scaling/mutual proximity) require separate data/feature comparisons; ECDF and
parametric tails already have prior results. No automatic expanded sweep.
Execution: four existing assigned dsba3 UUIDs, existing hun_fsad_tta_012, immutable
release, predecessor experiments and active checkout retained.
Collect compact results through existing collector, update canonical MD+CSV
together, and verify support/query/GT/bank hashes before paired claims.
Compare against prior mean/RMS as well as Q95 after verifying matching provenance.
