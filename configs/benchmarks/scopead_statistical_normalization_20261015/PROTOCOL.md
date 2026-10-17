# Statistical normal-reference normalization — fixed development sweep

## Authorization, hypothesis and common conditions

2026-10-15 user requested all normalization families from the preceding research
discussion. Test whether normal upper-tail width, estimation stability, or shape
improves observation fusion over Q95 division. No improvement is assumed.
Post-hoc search on seen public development data; not final evaluation.

- MVTec AD 2 test_public, all eight classes, seed0, 2/16 normal supports using
  existing VOID_support_2/16.json manifests. Only normal source-LOO scores fit
  transforms; no query data, masks, labels or dataset objects enter fitting.
- Frozen DINOv2 ViT-L/14, FP16 extraction, layers5/11/23, resolutions448/672/896/1120,
  4bags, 256tokens/source, cosine1NN, deep mean + .125 shallow, CQ off, k9 guard.
- Same pipeline/evaluator; primary AUPRO@.05, secondary pixel pAUROC@.05, higher
  is better. F1 oracle diagnostic only. No backbone training or bank changes.
- 13arms × 2shots × 8classes = 208 class evaluations. No adaptive expansion,
  outcome-dependent stopping or label-based selection of fitting parameters.
- Existing dsba3 hun_fsad_tta_012 container, four assigned GPU UUIDs. Stage-only
  immutable absolute release; preserve active checkout and existing jobs. No
  commit/push, provisioning or GPU expansion. Concurrent runtime is not an
  isolated speed benchmark. Timeout10800 seconds/class; retain failures.

## Arms: exact fixed definitions

Every observation/bag uses its existing normal source-LOO sample D. The same
capped number of tokens/source contributes here. Tokens/bags are dependent,
not independent normal images. Keep negative scores; divisor floor1e-4.

| Arm | Definition |
|---|---|
| none | Raw d; fresh control. |
| q95 | d/max(Q95,1e-4); fresh existing control. |
| mad | d/max(1.4826 MAD,1e-4); legacy torch.median, no centering. |
| tail | Existing empirical -log survival body and exponential tail above Q95. |
| upper_width | (d-Q80)/max(Q95-Q80,.05Q95,1e-4); guard collapsed upper intervals. |
| shrink_q95 | alpha=.5 geometric shrinkage of bag Q95 toward geometric mean across bags of SAME layer/resolution. No layer/resolution pooling or fitted alpha. |
| qn | (d-median(D))/max(Qn(D),1e-4). Qn=2.219144465985076 × kth pair difference, h=floor(n/2)+1, k=choose(h,2). Deterministic sorted quantile thinning to1024; asymptotic constant, no finite-n correction. |
| log_mad | log(max(d/unit,1e-6)), unit=max(Q50,1e-6); transformed median centering and 1.4826MAD division. |
| boxcox_mad | Same unit/floor and transformed median/MAD. Normal-profile-likelihood lambda in[-1,1], log at lambda0; failed fit uses log with recorded fallback. |
| gamma_survival | Gamma MLE, location0, reference floor1e-8; -log survival via SciPy float64 CPU logsf. Incomplete-gamma asymptotic only at underflow. Constant/failed fit uses Q95 and records fallback. |
| lognormal_survival | Lognormal location0 MLE, log-reference mean/std, sigma floor1e-4; stable -log survival via log_ndtr. Constant reference uses Q95 with recorded fallback. |
| gpd_survival | Existing empirical body through Q95, then -log(.05)+log1p(xi*(d-Q95)/beta)/xi, exponential limit xi0. Constrained MLE xi[0,.5], beta[1e-6,2] permits unbounded extrapolation. >=8 nondegenerate excesses required; insufficient/failed fit uses existing exponential tail and records fallback. Constrained GPD, not unrestricted EVT. |
| empirical_survival | -log((1+#(D>=d))/(n+1)), inclusive ties/add-one smoothing, saturation above max reference. Conformal-style rank mapping; NO exchangeability/FPR coverage guarantee. |

These are project adaptations, not whole published-method reproductions. Normal
references exclude one source while queries use full banks; the existing mismatch
is unchanged. In fixed linear fusion centering alone adds a constant; denominators
and query-conditional routing can change ranking. Do not infer performance from
visual normality/alignment or attribute all differences to centering.

## Evidence and collection

New statistical_reference_matching subclass uses unchanged distance, fusion and
audit hooks. Persist fitted parameters, fallback reasons/count, transformed Q50/
Q95, support/query/GT/code/config hashes, bank digests, metrics and bounded score
diagnostics. Masks only after prediction. Native-grid normal samples use mask-empty
queries before interpolation and bag averaging; image-capped descriptive samples,
not unbiased population quantiles. No dense maps or weights saved.

Existing normalized>1 counters are NOT common Q95/FPR thresholds for new transforms.
Use actual transformed Q95 metadata. Raw>support-Q95 counters remain comparable.
Verify support/query content+identity, manifest/GT and bank hashes before paired
deltas. All full8 arms/class results and fallbacks retained; no partial macro.

Pull compact metrics/config/manifests with run_remote.sh pull scoped to this run.
Use scripts/experiments.py collect for four completed shards, canonical comparison
Markdown and paired CSV. Reports stay in run root; no duplicate ledgers/manuscript
edits. execution.json/comparison.md record SHA, launches, progress and validation.

## Statistical motivations (transfer limits apply)

Kriegel et al., Interpreting and Unifying Outlier Scores, SDM2011.
Rousseeuw/Croux, Alternatives to the Median Absolute Deviation, JASA1993.
Ledoit/Wolf, A Well-Conditioned Estimator for Large-Dimensional Covariance
Matrices, J.Multivariate Analysis2004: shrinkage motivation only; covariance
optimality does not prove quantile-shrinkage optimality here.
Box/Cox, An Analysis of Transformations, JRSS B1964.
Siffer et al., Anomaly Detection in Streams with Extreme Value Theory, KDD2017.
Bates et al., Testing for Outliers with Conformal p-values, Ann.Stat.2023.

## Validation

47 new tests passed: finite/monotone vector+scalar transforms; constant-reference
fallbacks; empirical ties/saturation; Qn order statistic; Gamma/lognormal versus
SciPy and extreme Gamma underflow; all-arm fit/predict/native diagnostics; no fit
mutation after query; shrinkage identity; four controls versus parent predictions
and bank hashes; registry/config guards. Regression checks recorded at launch.
