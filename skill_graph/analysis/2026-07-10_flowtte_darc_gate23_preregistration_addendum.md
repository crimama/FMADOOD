# DARC Gate 2/3 Preregistration Addendum

Date sealed: 2026-07-10T09:30:34Z  
Status: frozen before the Gate 1 aggregate existed  
Gate 1 state at sealing: 6 of 45 object-seed completions; no aggregate decision

This addendum resolves only conditions that were not mathematically unique in
`2026-07-10_flowtte_darc_experiment_design.md`. It does not change the frozen
model, crops, layers, support budget, geometry, evaluator, or public/private
claim boundary. MVTec AD2 public remains shadow development data.

## Gate 2 population and pairing

- Objects: all 15 MVTec AD1 `train/good` objects.
- Seeds: `0,1,2`; exactly 16 P16-random source images per object-seed.
- Expected paired units: `15 × 3 × 16 = 720` source rows.
- Thin queries: `thin-w1-l32` and `thin-w2-l48`; broad control is excluded from
  the signal and retention statistics.
- L0, L1, and R1 use the same source, fold, query-ranked support identities,
  `K=min(5,n_valid)`, pixels, and fallback population.
- Missing rows, NaN/Inf, arm-specific row/pixel removal, or unequal support
  identities produce `INVALID_GATE_INPUT`; they never count as a failed or
  successful scientific gate.
- A token with fewer than three valid local supports is not dropped. Its
  predeclared rung-independent layer-7 G0 residual/evidence is used for the
  Gate 2 statistics, matching the deterministic method fallback.

## Gate 2 normal residual condition

The residual condition uses raw layer-7 cosine residuals before upper-tail
calibration, confidence scaling, coarse fusion, thresholding, or morphology.
Using separately calibrated tails would normalize away the mechanism being
tested.

For every object-seed group, concatenate the same 16 held-out-clean token
population and compute the higher empirical quantile:

```text
Q_r[o,s] = quantile(raw_cosine_residual_r, 0.999, method="higher")
Q_r      = mean over the fixed 45 object-seed groups of Q_r[o,s]
```

The condition passes iff `Q_L0 > 0` and `Q_L1 <= 0.80 * Q_L0`. Equality at
exactly 20% reduction passes. No confidence/validity-selected subset is used.

## Gate 2 paired signal condition

For each source, concatenate the two complete thin-query score maps and masks
and compute continuous pixel AP for L0 and L1. Component recall is computed on
the exact non-dilated 8-connected cue masks across the same two queries.

```text
dAP_i = AP_L1_i - AP_L0_i
dCR_i = component_recall_L1_i - component_recall_L0_i
```

Bootstrap is the Gate 1 source-cluster bootstrap:

- strata: the fixed 45 `(object, seed)` groups;
- unit: source image, not pixels or cue components;
- resample 16 sources with replacement inside every stratum;
- average each stratum, then equally average the 45 strata;
- shared resample-index tensor for AP and component recall;
- 10,000 replicates, NumPy `PCG64(20260710)`;
- 2.5% linear quantile, evaluated in unrounded float64.

The conservative slash interpretation is frozen: **both**
`LB_0.025(mean(dAP)) > 0` and `LB_0.025(mean(dCR)) > 0` are required. Equality
to zero fails. This prevents choosing whichever metric happens to win after the
experiment.

## Gate 2 R1 retention condition

For rung `r`, source `i`, and thin profile `t`, define the clean-subtracted
micro-evidence response on the exact cue mask `M_i,t`:

```text
d[r,i,t] = mean over x in M_i,t of (
    evidence_r(cue_i,t, x) - evidence_r(clean_i, x)
)
```

- Evidence is measured before coarse max-fusion, confidence scaling, binary
  thresholding, and morphology.
- Clean and cue for a rung share the exact same normal-only upper-tail
  reference. L1 and R1 each use their frozen rung reference.
- Negative responses are retained; masks are not dilated and rows are not
  filtered.
- Average profiles within source, sources within object-seed, then equally
  average the 45 groups to obtain `S_L1` and `S_R1`.

Retention passes iff `S_L1 > 0` and `S_R1 >= 0.90 * S_L1`. Equality at 90%
passes. A non-positive/non-finite denominator fails the scientific gate.

Gate 2 passes only when the residual, paired AP, paired component-recall, and
R1-retention conditions all pass. Broad-control `R1-L0` pAUROC is preserved as
a no-harm diagnostic but does not add a post hoc gate beyond the frozen design.

## Gate 3 arms, folds, and map aggregation

- Candidate: terminal `R1 + unchanged coarse` fused map. L1 or another best
  rung cannot be substituted after seeing results.
- Comparator: the unchanged 672 H+ `[7,15,23,31]` DVT MLP coarse FlowTTE anchor
  regenerated in the same run.
- Resource: genuine `P16-random`, seeds `0,1,2`. Candidate and comparator share
  the exact selected paths and four consecutive `12-memory/4-calibration`
  folds. P16-superad and Pfull remain separate tables.
- Every public test image is scored by all four folds. Fold raw continuous maps
  are accumulated in float64, divided by exactly four, and cast once to
  float32. No fold or seed is selected, weighted, or ensembled across seeds.
- The common evaluator consumes the per-seed averaged raw map over the full
  `test_public/good+bad` population. Primary gate metrics are all-test raw-map
  oracle F1 and `pAUROC@0.05`, without morphology.
- Bad-only metrics, AP, fixed-threshold F1, and shared morphology are diagnostic
  and cannot change the Gate 3 decision.

For class `c` and seed `s`:

```text
dF[c,s] = oracle_F1_candidate[c,s] - oracle_F1_anchor[c,s]
dA[c,s] = pAUROC_candidate[c,s]    - pAUROC_anchor[c,s]
dF[c]   = mean over seeds 0,1,2 of dF[c,s]
dA[c]   = mean over seeds 0,1,2 of dA[c,s]
```

Use the fixed class set `{can,fabric,vial,wallplugs,rice}`, fixed gap set
`{fabric,vial,wallplugs}`, and treat rice as a strong/no-harm class. Classes are
equally weighted after seed averaging.

Gate 3 passes iff all five conditions hold on unrounded float64 values:

```text
mean_c(dF[c]) >=  0.03
mean_c(dA[c]) >= -0.002
min_c(dF[c])  >= -0.02
dF[can]       >   0
count(c in {fabric,vial,wallplugs} where dF[c] > 0) >= 2
```

Exact equality passes only for the three non-strict numeric boundaries. Zero is
not a can/gap gain. The complete `5 classes × 3 seeds × 2 arms` matrix and equal
map populations are required before any pass decision.

## Interpretation and stop rules

- Gate 1 failure stops Gate 2 and Gate 3 exactly as originally frozen.
- Gate 2 failure stops the alignment/reconstruction family; AD2 cannot be used
  to choose another rung.
- Gate 3 failure stops DARC-v1; public AD2 cannot be used to retune radius,
  fusion, rung, class weights, threshold, or morphology.
- Alignment novelty credit is reported separately from Gate 3 performance. If
  an operational shadow null is run, R1 must beat the preregistered strongest
  of G0-high and L0 in macro F1 while losing no more than 0.002 pAUROC; otherwise
  a positive Gate 3 result is described only as resolution engineering.
- Gate 4 remains blocked without untouched/private data and is the only place
  for a final matched SuperAD/SuperADD superiority claim.

