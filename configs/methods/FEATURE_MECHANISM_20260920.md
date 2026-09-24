# QR1 feature/reference mechanism audit — 2026-09-20

User authorized finishing QM, then alignment, preservation and matching audits.
This is a post-hoc hypothesis test on seen test_public, not final evaluation.
No new detector, hyperparameter tuning, training, or novelty claim is introduced.

## Locked boundary and budget

First finish the missing five QM evaluations with one safe resume; retain the
SIGSEGV record. The local retry completed B16/fabric then failed C16/can with
OOM caused by another process using 20.54GiB. Four C/D16 can/fabric evaluations
remain and move to dsba3 GPU2. The original partial shard and both failure
records remain intact; the generic collector requires disjoint completed
stage/object identities and full combined coverage for recovery collection.
Then 16 fitted-QR1 runs: full8, nested 1/16 support, seed0,
SD_support manifests, frozen DINOv2-L layers11/23,448/672,256 tokens/source,
original R1 MAD fusion. Use the existing run/collect CLI. A new evaluation-only
hook reuses the fitted bank and computes counterfactual scores without mutation.
No labels, condition identity, paired image, or extra normal data enter fitting
or translation estimation. No maps, vectors or weights persisted. Existing
dsba3 container; remote physical GPU2 only, UUID
GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639 verified at launch. User prohibited local
compute on this turn: no further local experiment/test execution. Use the
existing isolated Python3.11/Torch2.6 runtime to match the recovered local
classes' software stack; hardware changes RTX4090 to RTX A6000 are declared,
not assumed bitwise equivalent. All16 new audit runs use this same remote
runtime/device. No automatic seed, severity, method or dataset sweep.

Primary conditions regular and shift_1, all available paired scenes. Labels
are used only after feature/score computation for evaluation strata and masks.
No threshold calibration is needed: condition AUPRO@.05, pAUROC@.05 and
within-bad-image defect-minus-normal score margins are used. Full-public QR1
metrics are retained through the unchanged common evaluator for parity audit.

## 1. Alignment

On paired GOOD images only, compute mean normalized feature displacement
v=mean(q_shift-q_regular), and differential fitted translation d=delta_shift-
delta_regular. Report cosine(d,v), d norm, v norm, projection coefficient,
and the alignment of the actual unit-normalized bank displacement difference
with v. Differential rather than only absolute delta removes a static
source-reference offset. Report mean cosine of delta_shift versus v secondarily.
Zero-length vectors yield missing cosine, never a fabricated zero.

Also hold regular-query NN indices fixed ACROSS conditions and compare mean
per-token squared residual displacement before/after each query's QR1 bank
correction, with fitted/random/opposite translation controls. This measures
reference-relative residual stability, NOT changed backbone representations.
Pair features are retained in evaluator memory for one scene/resolution only.

## 2. Preservation

For GOOD images report normal score mean. For BAD images report defect and
within-image normal means and their margin separately, using condition-specific
GT at native resolution. Report baseline, fitted, random, opposite,
fixed-correspondence fitted, baseline5NN and fitted5NN scores. Higher margin
and AUPRO are favorable but are not equivalent. Lower normal score alone is
not defect preservation. Query features themselves are unchanged by QR1;
therefore do not claim raw feature separability was improved.

## 3. Matching dependence

The fixed-correspondence control scores the ORIGINAL nearest reference after
translation+unit normalization, without reselecting its identity. Compare
against actual QR1 which reselects NN after translation. The alternative
5NN readout uses mean top5 cosine distance with the SAME original support MAD
scales, bank and fitted delta. This isolates scoring, not a separately tuned
5NN detector; it cannot establish detector-agnostic transfer.

## Decision rule (descriptive, not statistical significance)

Class is the macro unit; images/layers/resolutions are not independent trials.
Report every class and both shots plus leave-largest-gain-out means.
Nuisance-consistency requires positive class-macro alignment and residual
displacement reduction in >=5/8 classes at BOTH shots; fitted reduction must
also exceed random/opposite in >=5/8 classes. Preservation requires positive
class-macro margin change, >=5/8 positive classes, and positive AUPRO change at
both shots. If performance improves but margins do not, reject a broad
mean-separation mechanism, not the observed performance. Fixed/5NN positive
AUPRO gains in >=5/8 at both shots support limited scoring robustness, not
backbone adaptation or causal nuisance identification. Stop after this suite;
decide the next method from evidence rather than adding explanation searches.
