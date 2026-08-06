# DINOv2-L/14 Basic Default Mechanism Audit

Date frozen: 2026-07-19  
Protocol state: `DESIGN_FROZEN / NOT_RUN`  
Maximum verdict: `MECHANISM_SUPPORTED_ON_PUBLIC_SHADOW`

## 1. Purpose and claim boundary

This experiment does not search for another candidate or tune the default. It
asks why the retained DINOv2-L/14 Basic default works, which block is robust to
distribution shift, and which block merely improves the final metric while
increasing shift sensitivity.

The audit is causal only for the frozen component interventions below. It
cannot establish a new method improvement because MVTec AD2 `test_public` has
already been inspected. `TESTprivate`, TTA, query transformation, query-time
optimization, query memory update, routing, and metric-selected retries are
forbidden. The static support bank is never expanded.

## 2. Frozen operational default

The reference is the post-ablation `minus_loo` Basic arm, not the later tuned
`layer_late` variant and not DINOv3-H+.

| Item | Frozen value |
|---|---|
| Backbone | DINOv2 ViT-L/14 |
| Layers | `[5,11,17,23]` (paper-aligned levels 6/12/18/24) |
| Resolution | 672; `sheet_metal=448` |
| Support | exact SuperAD-selected 16 normal images, seed 0 |
| Support manifest | `results/remote_runs/dsba3/flowtte_basic_component_ablation_fixed16_20260713_v1/support_paths_dsba3.json` |
| Manifest SHA-256 | `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3` |
| Support views | identity only; brightness `[1.0,1.0]` |
| Fusion / DVT | `layer_norm_mean`; `position_mean`, alpha 1.0 |
| Flow | 2 coupling layers, hidden multiplier 1, 3 epochs, LR `2e-4`, clamp 1.9 |
| Flow objective | tail weight 0.3, top ratio 0.05, log-det weight `1e-3` |
| Score | latent 1-NN distance 1.0 + density 0.25, q=0.9, top 1%, nonsquared |
| Memory / context | static, expansion budget 1.0; context none |
| LOO standardization | off |
| Output | continuous raw -> guided-r8 -> close/fill/erode |

Historical operating anchor: macro pAUROC@0.05 `0.78194`, raw F1 `0.38108`,
and guided+morph F1 `0.44107`. These values are parity anchors, not targets for
selection. The historical morphology score used the public best-threshold
diagnostic and is not a deployable threshold estimate.

The later `[11,15,19,23]`, log-det `0.02`, brightness `[0.8,1.2]` result is
explicitly out of scope. The prior DINOv3-H+ shift sequence is hypothesis
context only and cannot be used as direct evidence for this backbone.

Historical component evidence is used only to set hypotheses. Relative to the
then-full Basic arm, removing DVT cost `0.01332` pAUROC, removing Flow cost
`0.00348` pAUROC and `0.01089` morphology F1, while removing LOO changed
pAUROC/F1 by `-0.00104/+0.00638`. This establishes that DVT is the first
mechanism to explain, Flow is a smaller effect requiring geometry evidence,
and LOO must remain off; it does not prove that either DVT or Flow corrects
distribution shift.

## 3. Mechanism model to test

The default is decomposed into four serial mechanisms:

1. multi-level DINO representation supplies local defect evidence;
2. support-fitted DVT removes a position-locked nuisance direction;
3. Flow remaps the fused feature geometry before static nearest-neighbor
   comparison;
4. density and RGB/morphology reshape the score tail and output boundary.

The central alternative is that stages 2--4 raise the public metric without
correcting shift. In that case shifted normal scores inflate, nearest-support
identity becomes unstable, or shifted-defect contrast falls even though an
aggregate pAUROC/F1 increases.

## 4. Stage M0 — parity and access preflight

Before scientific metrics:

- reproduce the disabled-instrumentation output byte-for-byte on deterministic
  fixtures and within `1e-6` on one fixed object smoke;
- verify backbone, layers, resolution, support paths/hash, seed, score weights,
  LOO-off state, and static-memory size in every arm manifest;
- verify every query is evaluated once with no mutation of model, DVT field,
  Flow, memory, threshold, or support bank;
- record an access ledger proving zero `test_private` reads and zero query
  update/TTA calls;
- run same-reference SuperAD-16 and RN-FMLK parity preflights before any public
  conclusion. A missing same-condition baseline blocks Stage M2 interpretation,
  but not the support-only Stage M1 mechanism assay.

Any parity or access failure is infrastructure-invalid and produces no
scientific result.

**Pre-metric execution amendment (2026-07-19).** The deterministic CPU
fixture retains the registered `1e-6` score tolerance. On dsba3, chunked CUDA
`torch.cdist` and the independent full-matrix CUDA reference returned identical
top-2 neighbor indices but differed by up to `1.1444091796875e-5` because the
kernel reduction order changes with matrix shape. Real CUDA runs therefore use
an absolute score tolerance of `2e-5` while still requiring zero neighbor-index
mismatches. This infrastructure-only amendment was made after all four groups
stopped at M0 and before any M1 condition row or metric was emitted; no model,
stimulus, score, threshold, arm, or scientific gate changed.

The first full v3 attempt then exposed a larger but still index-preserving
float32 CUDA `cdist` discrepancy on `walnuts`: maximum absolute score error
`3.0517578125e-05` with zero neighbor-index mismatches. Before any `walnuts`
condition row was emitted, the CUDA-only absolute score tolerance was therefore
amended once more to `1e-4`. The CPU fixture remains `1e-6` and exact neighbor
indices remain mandatory. v3 is infrastructure-invalid and none of its partial
object outputs are reused in v4. No model, support, stimulus, score, threshold,
arm, metric, or scientific interpretation gate changed.

## 5. Stage M1 — all-eight support-only factorial assay

### 5.1 Population and folds

Use all eight objects and the exact 16 frozen supports. For each object, create
four deterministic source-disjoint folds: 12 sources fit Flow/DVT/memory and
the remaining four sources are evaluation images. Rotate until every source is
held out once. Thresholds use fit sources only.

Each held-out source produces paired cells from the same pixels:

- `C`: clean;
- `S`: shift only;
- `D`: local synthetic defect only;
- `SD`: the identical defect under the identical shift.

Shift axes are exposure, gamma, contrast, and reciprocal red/blue white
balance, each at two frozen nonzero magnitudes and both directions. Defect
families are copy-paste texture, low-frequency chromatic stain, local blur,
and smooth local geometric warp. Generator parameters and retry order are
hash-derived before feature extraction; downstream scores cannot select a
stimulus. The same accepted image and mask are shared by every arm.

All eight objects remain in every primary macro. Historical results define
interpretive strata only: `can/fabric/vial` are ranking-weak sentinels,
`rice/walnuts` are ranking-strong controls, and
`fruit_jelly/wallplugs/sheet_metal` are middle/postprocessing-sensitive
controls. `can` and `fabric` are explicit RGB-guidance harm sentinels. These
roles cannot be changed or used to exclude an object after the assay.

### 5.2 Causal arms

| ID | DVT on fit support | DVT on held-out query | Flow | Density | Purpose |
|---|:---:|:---:|:---:|:---:|---|
| A0 | yes | yes | yes | 0.25 | exact continuous default |
| A1 | no | no | yes, refit | 0.25 | total DVT necessity |
| A2 | yes | no | yes, refit | 0.25 | support-side DVT intervention |
| A3 | no | yes | yes, refit on raw support | 0.25 | query-side DVT intervention |
| A4 | yes | yes | yes | 0 | density contribution |
| A5 | yes | yes | identity | 0 | raw fused 1-NN / Flow intervention |
| A6 | no | no | yes, refit | 0 | no-DVT Flow cell |
| A7 | no | no | identity | 0 | no-DVT raw fused cell |
| L5/L11/L17/L23 | matched per layer | matched per layer | identity | 0 | fixed single-layer null envelope |
| S16 | method-native | method-native | SuperAD | method-native | same-support external anchor |
| R16 | none | none | RN-FMLK | none | mandatory hard null |

A2/A3 are mismatch diagnostics, not deployable candidates. They test whether
DVT credit arises from query nuisance removal, support geometry alteration, or
placing both sides in the same coordinate system. A4--A7 form a density-free
DVT×Flow 2×2 block, so a compensating interaction cannot be misreported as an
independent component effect. Every Flow arm receives its own deterministic
fit; identity and layer-null arms reuse cached frozen features. No arm may
inherit a fitted state from an incompatible representation.

For A0 maps only, evaluate the output chain without refitting:

- P0: raw continuous map;
- P1: guided-r8 continuous map;
- P2: P1 thresholded with the fit-clean q=0.995 threshold, then frozen
  close/fill/erode.

The legacy public best-threshold P2 is reproduced only as a parity diagnostic
and is labeled `oracle`. P2 with the fit-clean threshold is the only primary
postprocessing result.

This separates detector credit from presentation/segmentation credit.

### 5.3 Recorded mechanism variables

For every paired source, condition, layer, and arm, retain compact records for:

- shifted-normal q95/q99/q99.5 tail inflation relative to paired clean;
- defect-mask contrast: median inside-mask score minus outside-mask score;
- DINO feature patch-grid pAUROC@0.05 and fixed-clean-threshold F1 for D and
  SD; these are not original-image pixel metrics;
- shifted-defect retention `metric(SD)-metric(D)` and factorial interaction
  `(inside-outside)_SD-(inside-outside)_D`;
- clean-to-shift nearest-support identity retention, first/second-neighbor
  margin, support-source entropy, patch-position displacement, and memory-use
  Gini coefficient;
- raw-to-latent pair-distance Spearman correlation, effective rank, latent
  RMS, Flow log-det quantiles, and distance/density score correlation;
- per-layer DVT field energy, feature variance removed, and clean/shift cosine
  displacement before and after DVT;
- component count, boundary F1, positive area, and small-component survival for
  P0/P1/P2.

Primary summaries are object-macro paired deltas with 2,000 hierarchical
bootstraps over object, held-out source, shift axis/magnitude, and defect
family. Single-layer and simple-baseline envelopes are recomputed inside every
bootstrap replicate. Oracle best F1 is diagnostic only and cannot pass a gate.

### 5.4 Frozen interpretation rules

Material metric effects are `0.005` pAUROC or `0.010` fixed F1. Material tail
effects are 10% relative change. Object collapse is below `-0.02` pAUROC or
`-0.03` F1. Directional claims require a positive paired 95% bootstrap lower
bound and the same sign on at least six of eight objects.

**DVT classification**

- `SHIFT_CORRECTOR`: A0 versus A1 reduces shifted-normal tail inflation by at
  least 10%, preserves D/SD metrics within `-0.005/-0.010`, and has no object
  collapse.
- `QUERY_DRIVEN`: A3 recovers at least 70% of the A0-versus-A1 shift-tail and
  SD-retention effect, while A2 recovers at most 30%.
- `COMMON_COORDINATE_ALIGNMENT`: only symmetric A0 passes; both one-sided arms
  fail or destabilize neighbor identity.
- `RANKING_ONLY_OR_MIXED`: A0 improves pAUROC but fails the tail/retention rule.

**Flow classification**

- Flow credit under DVT is A4 versus A5; without DVT it is A6 versus A7. It is
  `ROBUST_REPRESENTATION_REMAP` only if D and SD improve materially,
  shifted-normal inflation does not worsen by 10%, and neighbor/geometry
  diagnostics do not collapse.
- If metrics improve while raw-to-latent distance rank falls sharply or the
  shifted-normal tail rises, report `DISCRIMINATIVE_BUT_SHIFT_SENSITIVE`, not
  normalization or correction.
- If A4 does not beat A5 materially, the learned Flow is not necessary under
  this assay even if its training loss converged.
- The paired interaction
  `(A4-A5)-(A6-A7)` is reported with its bootstrap interval. A material
  interaction means DVT and Flow are a coupled mechanism; neither marginal
  ablation receives independent causal credit.

**Density classification**

- Density credit is A0 versus A4. Positive anomaly separation with at least
  10% extra shifted-normal inflation is `SEPARATION_AMPLIFIER_WITH_SHIFT_RISK`.
- No material separation gain plus tail inflation is `HARMFUL_TAIL_EXPANSION`.
- Density is a shift correction only if it improves SD retention and reduces
  shifted-normal tails simultaneously.

**Fusion and postprocessing classification**

- Multi-level synergy requires A5 to exceed the simultaneous L5/L11/L17/L23
  envelope by `0.005/0.010` with positive lower bounds. Otherwise report the
  dominant layer or destructive fusion; never select a new layer setting.
- RGB guidance/morphology receives only output credit. It is retained as
  globally safe only if fixed F1 improves by at least 0.010, pAUROC harm is
  below 0.005, and no object crosses the collapse bound. Improvements that
  coexist with `can` or another object harm are conditional presentation
  effects, not core distribution-shift robustness.

## 6. Stage M2 — frozen public-shadow confirmation

Stage M2 is allowed only after M0 validity and a complete M1 report. M1 cannot
change arms, thresholds, layers, hyperparameters, or object population.

Run all eight `test_public/good,bad` objects with acquisition groups parsed
from the existing AD2 filename/metadata convention. Evaluate A0, A1, A4, A5,
A6, A7, S16, and R16 on the exact same query population and exact 16 references.
P0/P1/P2 must reuse A0 scores. Report regular and each shifted condition
separately, plus condition-macro and object-macro values.

The public confirmation asks only whether the M1 mechanism signs transfer:
DVT tail effect, Flow effect, density effect, fusion/null margin, and output
effect. A mechanism is `SUPPORTED_ON_PUBLIC_SHADOW` when its preregistered
direction holds on at least six objects and the paired macro confidence
interval excludes zero. Failure becomes `SYNTHETIC_ONLY`, `OBJECT_SPECIFIC`,
or `FALSIFIED`. No public result authorizes tuning or a superiority claim.

## 7. Execution layout and artifacts

When execution is requested, use dsba3 GPUs 0--3 and keep every object's arms
on the same GPU:

- GPU 0: `can`, `rice`;
- GPU 1: `fabric`, `walnuts`;
- GPU 2: `vial`, `sheet_metal`;
- GPU 3: `fruit_jelly`, `wallplugs`.

Use a new empty result root. Resume is arm- and object-isolated. Save manifests,
access ledgers, compact condition JSONL, `causal_contrasts.tsv`,
`neighbor_stability.tsv`, `geometry.tsv`, `per_object_metrics.tsv`, bootstrap
summaries, and a final report. Dense maps may be deleted only after coverage,
alignment, checksum, and compact reaggregation checks pass.

## 8. Stop rules

- Stop as infrastructure-invalid on parity, support-hash, finite-map, alignment,
  access-ledger, or full factorial-coverage failure.
- Do not replace failed arms, adjust effect thresholds, select easier shift
  axes/defects, or tune the default after seeing M1/M2 metrics.
- If SuperAD-16 or RN-FMLK is not same-population and exact-reference matched,
  mark it `not comparable` and withhold all external-baseline interpretation.
- Final reporting must distinguish evidence from inference and assign each
  component exactly one of the frozen mechanism labels above.

## 9. Expected decision value

This design can distinguish four materially different explanations for the
current default: genuine shift correction, shared-coordinate alignment,
shift-sensitive discriminative remapping, and downstream score/output
amplification. Therefore a metric decrease in an ablation is not automatically
credited as a useful mechanism; credit requires the predicted intermediate
variables—tail stability, neighbor stability, geometry, and defect retention—to
move in the same direction.
