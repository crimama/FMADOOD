# Consolidated experiment results and insights

Date: 2026-09-08

## Executive conclusion

The result corpus does not establish a new generally superior replacement for
the exact RN/SuperAD baseline. The strongest repeatable signal is that local
source/part structure and DVT-related features contain useful diagnostic
information, but most learned transport, correction, quotient, velocity, and
adaptation variants either fail validity/no-harm gates or do not improve the
deployed detector consistently.

The most defensible next path is therefore a conservative exact-RN baseline
with confidence-gated local source/part retrieval. Oracle arms, support-only
diagnostics, runtime fixtures, incomplete shards, and invalid aggregates must
not be promoted to benchmark claims.

## Decision-bearing quantitative results

### Baseline anchors

- Exact RN TESTpublic baseline: pAUROC@0.05 `0.749400`.
- Baseline1 private submission: segmentation F1 `0.4594`; delta versus
  SuperAD-16 is `-0.0124` on private and `+0.0154` on private-mixed. This is an
  official anchor, not a superiority result.
- KNN default comparison: retain K1. K5 passes the F1 no-harm condition but
  fails the pAUROC no-harm condition.

### Source/part conditioned local transport

- Full-shot B1 pAUROC: `0.769953`; support16: `0.689831`; gain `+0.080122`,
  bootstrap 95% CI `[+0.039551, +0.117081]`, with wins on all four evaluated
  objects.
- Source-shuffle gap: `+0.163114`; part-shuffle gap: `+0.258039`.
- False-positive rate increases by `+0.055402`, so continuation must remain
  confidence-gated rather than becoming an unconditional method claim.
- Confidence/conformal follow-up: C0 pAUROC `0.746056` and FPR `0.000439`, but
  its coverage gate fails. Synthetic and TTA mechanisms do not validate.

### DVT diagnostics

- All-8 rank/coverage diagnostic: bad-image AP `+0.01064`, full AUROC
  `+0.02067`; pAUROC `0.796952 -> 0.839483` (`+0.042530`) and F1
  `0.466814 -> 0.529027` (`+0.062212`). AP/AUROC direction is positive on 7/8
  objects, with replication parity passing.
- Alignment/scale attribution replicates AP `+0.010637` and full AUROC
  `+0.008167`.
- Flow interaction in the mechanism attribution is only `+0.002281` pAUROC;
  the Flow component is not supported as the main mechanism.
- Positional controls show a medium-defect benefit but fail small-defect
  safety. The small-defect loss begins immediately after DVT subtraction;
  Flow-latent and deployed-final behavior are nearly identical.
- Scale-selective residual candidate: pAUROC `0.83172`, AU-PRO `0.51741`, F1
  `0.51784`; diagnostic continuation only.
- Two-band protected residual: pAUROC `0.83618`, AU-PRO `0.43806`, F1
  `0.51810`, but catastrophic no-harm and small-coverage failures imply anomaly
  absorption and `NO_CONTINUE`.

### RN compression and quotient controls

- Representative DVT full score `0.732581` versus exact RN `0.749400`: DVT is
  harmful in this test. R128 is near-lossless at `0.748028`, but supports only
  an engineering compression option.
- Patch k-center P50 `0.749121`; equal-budget random N50 `0.749247`; neither
  shows a consistent selector advantage over exact RN.
- Tangent quotient/orbit M0 `0.652313` versus exact RN `0.749400`: the current
  quotient formulation is harmful and closed for performance claims.

### Support-only mechanism diagnostics

- SOVM M0 macro pAUROC `0.674283`. Its spatial/source shuffle gap is
  `+0.145896`, CI `[+0.114077, +0.177550]`, with 4/4 wins; the learned
  velocity/code/time formulation is nevertheless rejected (`ORACLE_ONLY /
  NO_CONTINUE`).
- WTCF preserves global geometry but fails directional contraction. Median
  witness JVP ratio is about `0.9369` against the required ceiling `0.85`;
  unsupported ratio `0.9828`, pair-distance Spearman `0.9983`, neighbor
  retention `0.9297`. The candidate is closed.
- SoCFM A3 retains broad/local defects `0.99642/0.99899`, with shifted-normal
  displacement coverage `0.53032`; A4 gives `0.99643/0.99810` and `0.58962`.
  This is diagnostic evidence only (`KILL_FOR_CLAIM_CONTINUE_DIAGNOSTIC`).
- Normal-tail structure is concentrated (`C_img` Gini `0.7522`, top-decile
  share `0.6523`, positional concentration `0.7355` versus null `0.5527`), but
  cannot be identified without test access and is closed.

### Attention and completion controls

- DINO attention-facet extraction is valid but killed for claim. The hidden arm
  reaches image AUROC `0.73991`, segmentation pAUROC `0.80920`, and F1
  `0.50021` across four manifests.
- Cross-image region completion fails the rice F1 no-harm gate; medium-defect AP
  drops `-0.196325`. No continuation.
- Region-completion map-level aggregation is invalid because D0-versus-position
  reproducibility fails for all eight objects, with AUROC discrepancies up to
  `0.4131`.
- Spatial-illumination and synthetic-shift-bank comparisons are invalid because
  frozen-flow/scorer/preprocessing contracts do not match across arms.

## Incomplete, invalid, or non-scientific records

- SPCT full-shot shards have a valid merged diagnostic summary, while some host
  trees still carry `aggregate_pending.txt`; the merged summary is canonical.
- Local SOVM is invalid; the merged distributed summary is canonical.
- ATAC is a fatal support-gate failure with only `can` observed and 7/8 objects
  absent. Metrics are null rather than zero.
- Semantic-atlas has an in-progress report stub and no closed final result.
- SAM refinement lacks a trustworthy aggregate claim.
- Shift-proxy retains setup/custody summaries but no completed method claim.
- Shift-residual ablation contains a runtime fixture whose method, RN-FMLK, and
  SuperAD values are identical; it is not a completed experiment.
- C1 factorial is `SMOKE_PASS / MAIN_RUN_NOT_STARTED`; scientific metrics are
  intentionally `N/A`.
- TCCR local output is controller-only and has no aggregate result.
- KNN attempts v1-v4, several DARC superseded/smoke variants, early ECO attempts,
  and smoke-only directories contain no admissible final quantitative result.
- The empty PSAR repeat checkpoint summary is malformed and carries no result.

## Canonical-record policy

For interpretation, use the dated `skill_graph/experiments/*/report.md` as the
primary record. When a machine-readable value is needed, use
`quantitative_ledger.json` and resolve its original aliases through
`source_index.tsv`. Prefer the newest completed root or merged aggregate over
host shards, object copies, retry trees, partial runs, and local invalid runs.

The following are deliberately excluded from final claims: per-image rows,
raw JSONL, support path lists, controller logs, PIDs, locks, preflight output,
launch scripts, retry receipts, invalid or partial aggregates, and oracle-only
arms. Their removal does not change the consolidated conclusions above.
