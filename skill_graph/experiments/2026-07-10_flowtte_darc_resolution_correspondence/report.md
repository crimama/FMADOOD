# FlowTTE DARC Resolution and Correspondence Gates

Date: 2026-07-10
Verdict: KILL_FOR_CLAIM / CONTRACT_CONFLICT; operationally stop raw R1

## Negative Evidence Intake

This is not another 672-grid flow/context/support-fusion retune. Transformer,
Conv2D, SCM, layer-wise flow, register/CLS routing, structured flat memory,
augmentation, morphology, and broad flow hyperparameters all failed to recover
`can`. DARC changes the upstream native sampling and defines an explicit
support-query coordinate counterfactual while preserving the existing coarse
branch as a no-harm anchor.

## Motivation

The current method is numerically close to the reported SuperADD AUROC context,
but that number is not a matched baseline: SuperADD uses P-full access and a
different test population, threshold, morphology, and output grid. Within the
current method's own P16 all-test evaluation, the large F1 gap and almost-zero
`can` F1 remain the target failure. Independent analysis identifies sub-token
cue loss and position-free global matching as the most direct structural causes.

## Implementable Design

The frozen design is recorded in
`skill_graph/analysis/2026-07-10_flowtte_darc_experiment_design.md`. Execution is
ordered as protocol parity, paired resolution, `G0/L0/L1/R1`, reduced frozen
AD2 shadow, then all-eight/untouched gates only when prerequisites hold.

### Actual-data raw-ladder pilot

The user authorized an actual AD2 performance diagnostic before the mechanism
gate is complete. The pre-metric contract is frozen in
`skill_graph/analysis/2026-07-11_flowtte_darc_ad2_raw_ladder_pilot.md`.
It evaluates the already-implemented raw G0/L0/L1/R1 ladder on full public `can`
without inventing the currently undefined confidence/coarse-evidence fusion.
This is explicitly not Gate 3 and cannot support a baseline-superiority claim.
The raw ladder is parameter-free: this cell performs real feature extraction,
support scoring, map generation, and evaluation, but it does not optimize a
trainable flow/head. A learned replacement is a separate future method family.

## Evaluation Alignment

- Target: MVTec AD2 single-image, `/home/hunim/Volume/DATA/mvtec_ad_2`.
- Public AD2 labels: development/shadow only; not an untouched claim set.
- Full objects: can, fabric, fruit_jelly, rice, vial, wallplugs, walnuts,
  sheet_metal; full `test_public/good,bad` when all-eight is run.
- Resource protocols: P16-random, P16-SuperAD-selected, M16-fullpool, and Pfull
  are reported separately.
- Primary metrics: pixel `seg_AUROC_0.05`, `seg_F1`; protocol parity also reports
  AP, bad-only metrics, fixed normal-threshold F1, coverage, and bootstrap bounds.
- Same-condition SuperAD artifact/rerun: pending protocol audit.

## Code Modification / Creation

- Added a deterministic AD2 raw-ladder runtime with P16 selection frozen before
  decode, fold-specific support caching, G0/L0/L1/R1 native map stitching,
  float64 fold aggregation, and per-query coverage audit:
  `src/flow_tte/darc_ad2_pilot.py`,
  `src/flow_tte/darc_ad2_pilot_io.py`, and
  `src/flow_tte/darc_ad2_pilot_runtime.py`.
- Added `scripts/run_flow_tte_darc_ad2_pilot.py` with selected execution-code
  hashes, model/support/query inventories, deterministic operational sharding,
  and completion records. The provenance limitation is disclosed below.
- The initial v1 runtime used a label-bearing source path as the deterministic
  RANSAC identity. It was invalidated before evaluation. The corrected v2 uses
  `object/content-SHA256`; identical bytes under good/bad paths now produce the
  same scorer identity and seed.
- Added nine focused pilot tests, including population-neutral query identity,
  disjoint/exhaustive sharding, coverage persistence, native map layout, and
  float64 fold averaging. The ninth locks the post-run fresh-root guard.

## Added Code Evaluation

- `tests/test_darc_ad2_pilot.py`: `9 passed`.
- Pilot plus common evaluator/map I/O suite: `18 passed in 1.01s`.
- Focused Ruff, basedpyright, `py_compile`, CLI help, and invalid-fold CLI checks
  passed before the full run.
- A fixed-code two-image v2 smoke produced eight finite, nonconstant
  `1024x2232` float32 maps, two coverage rows, population-neutral scorer IDs,
  and matching code/design hashes before the full cell was launched.

## Remote Execution

Execution target: dsba3 `147.47.39.144:2222`, fixed project container
`hun_fsad_tta_012`, host GPUs `0,1,2,3`. The valid full pilot used eight
deterministic query shards, two processes per GPU. Shard sizes were
`21,21,20,20,20,20,20,20`.

### Execution-scope correction

SuperADD P-full is retained only as the already-published external reference;
it is not rerun as a new experiment. A local parity implementation had entered
a production cold start, but the user correctly rejected that rerun because it
does not test the DARC claim and the paper already reports the anchor. Both
attempts ended before any category completion or scientific metric was emitted:
the first was invalidated during a producer-provenance audit and the second was
cancelled for scope. Their scratch data was deleted. Subsequent dsba3 GPU time
was assigned only to DARC gate and pilot work. Gate 1 used GPUs 0,1,2; after
the user's later resource update, the actual-data pilot used GPUs 0,1,2,3.

## Published Baseline Context and Unified Metrics

SuperADD Table 1 reports TESTpub macro pixel `pAUROC@0.05=0.8393` and
`F1=0.6261`. This remains an external paper-context row with
`comparable=false`; it is not rerun and no `delta_vs_superadd` superiority claim
will be computed. The incompatibilities are independent and material:

- resource: SuperADD is P-full, whereas the DARC claim path is genuine
  P16-random;
- population/evaluator: SuperADD excludes good test images and reports the F1
  of its fixed binary output, whereas Gate 3 uses all-test raw continuous maps
  and oracle max-F1;
- post-processing/grid: SuperADD applies its train-derived threshold and
  morphology on an `H/4 x W/4` output, not the native common-evaluator grid.

The local SuperAD-16 historical row is a separate DINOv2,
full-pool-selected-support result and must not be mislabeled as the published
SuperADD result or as genuine P16-random.

## Results and Analysis

### Gate 1 context

The preceding AD1 synthetic resolution gate completed over 720 source/profile
rows and recorded `passed=true`. Relative to 16-pixel G0, 8-pixel G0 gained
`0.096791` AP (`+90.72%` relative), with paired lower bounds
`AP=0.094459` and component recall `=0.004167`; the broad-control pAUROC delta
was `+0.001697`. This supports retaining unrestricted high-resolution G0, not
the later hard-local ladder. Its held-out threshold stability was false and
`deployable_fixed_f1_allowed=false`. The compact source is
`results/remote_runs/dsba3/darc_v11_g1_ad1synthetic15_p16r_20260710_v1/gate_decision.json`.

### Valid actual-data cell

- Object/resource: `can`, P16-random seed 0, fold 0 (`12` memory, `4`
  held out), full `72 good + 90 bad` public population.
- Output: `162/162` coverage rows, `648/648` maps, and `8/8` valid completion
  records. Every arm contains the same 162 logical image IDs.
- Operational deviation: the frozen amendment specified four modulo shards,
  but the later four-GPU execution used eight shards (two per GPU). This was a
  scheduling deviation, not a method change: the eight manifests have identical
  support/fold/model/design hashes, their modulo assignments are disjoint and
  exhaustive, and their union exactly reproduces all 162 queries and four maps
  per query. It is disclosed rather than retroactively written into the frozen
  design.
- Evaluator: `darc-common-eval-v1`, native `1024x2232` grid, all-test raw
  continuous metrics primary. Fixed F1 uses cross-fit public good images and is
  retained only as a transductive diagnostic.
- Compact artifact:
  `results/remote_runs/dsba3/darc_ad2_raw_ladder_can_s0f0_full8_20260711_v2`.

| Arm | pAUROC@.05 | AP | Oracle F1 | Fixed raw F1 | Fixed morph F1 | Component recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G0 | 0.667650 | 0.001515 | 0.017797 | 0.015652 | 0.010403 | 0.125000 |
| L0 | 0.547831 | 0.000177 | 0.000488 | 0.000000 | 0.000000 | 0.052083 |
| L1 | 0.579249 | 0.000233 | 0.000827 | 0.000000 | 0.000000 | 0.031250 |
| R1 | 0.550352 | 0.000203 | 0.001647 | 0.001301 | 0.001026 | 0.041667 |

The decisive terminal delta is:

| Delta | pAUROC@.05 | AP | Oracle F1 | Fixed raw F1 | Component recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| R1 - G0 | -0.117298 | -0.001312 | -0.016150 | -0.014351 | -0.083333 |
| L0 - G0 | -0.119819 | -0.001339 | -0.017309 | -0.015652 | -0.072917 |
| L1 - L0 | +0.031418 | +0.000056 | +0.000339 | 0.000000 | -0.020833 |
| R1 - L1 | -0.028897 | -0.000030 | +0.000820 | +0.001301 | +0.010417 |

Bad-only metrics do not reverse the result: G0 is
`0.659705 pAUROC / 0.025236 oracle F1`, while R1 is
`0.542097 / 0.002054`.

### Coverage and runtime validity

- All 162 queries accepted all five registrations.
- Token-weighted common fallback was `2.370760%` overall,
  `2.304021%` on good, and `2.424151%` on bad. Per-image median/p95/max were
  `0.387234% / 12.990994% / 16.063097%`.
- L0 had five valid supports for every one of `11,943,936` scorer tokens.
  The shared L1/R1 support-count histogram for counts 0..5 was
  `[1,796, 12,610, 140,425, 484,586, 1,181,878, 10,122,641]`.
- All 648 TIFFs decoded as finite, nonconstant, single-page float32 arrays of
  shape `1024x2232`. There were no missing/duplicate map IDs or map-generation
  tracebacks. The canonical evaluator log does contain `Terminated`: after G0
  completed, its redundant watcher/L0 child was deliberately stopped.
  Independently run L0/L1/R1 compact outputs were checked byte-for-byte against
  their canonical copies before `evaluation_complete` was written.
- Eight manifests agreed on all 26 recorded code hashes, design, model weights,
  P16 selection, fold, seed, and source inventory. The invalid v1 roots were
  never evaluated or mixed with v2.

### Structural failure localization

The all-test population contains only `15,348` anomalous pixels among
`370,262,016` pixels (`0.004145%`) and 96 GT components. At each arm's oracle
threshold:

| Arm | TP | FP | Pixel recall | Precision | FP / TP |
| --- | ---: | ---: | ---: | ---: | ---: |
| G0 | 332 | 21,630 | 0.021631 | 0.015117 | 65.2 |
| L0 | 434 | 1,761,968 | 0.028277 | 0.000246 | 4,059.8 |
| L1 | 165 | 383,441 | 0.010751 | 0.000430 | 2,323.9 |
| R1 | 163 | 182,404 | 0.010620 | 0.000893 | 1,119.0 |

This isolates the failure:

1. The hard coordinate-local constraint is the primary regression. L0 gains
   only 102 true-positive pixels over G0 but introduces about 1.74 million
   additional false positives. It is not merely using the wrong threshold;
   oracle F1 and ranking metrics already include the best attainable threshold.
2. Detached registration has a real but very small corrective effect relative
   to the broken identity-local arm: L1 improves L0 pAUROC/AP/F1, but remains
   far below G0 and loses component recall. Registration therefore does not
   make the local score field valid.
3. R1 suppresses some of L1's broad responses but does not recover anomaly
   ranking; pAUROC and AP fall again. The per-support/component and geometric
   medians likely collapse multimodal normal appearances into an off-manifold
   prototype. This mechanism is an inference from the rung ablation, not a
   separately identified causal parameter.
4. Coverage collapse, fixed-threshold choice, and morphology are ruled out as
   primary explanations. Coverage is high, all five registrations are accepted,
   local oracle F1 is already near zero, and shared morphology never rescues it.
5. The evidence is consistent with the frozen global 4-DOF similarity plus
   `3x3` local-window assumption being too rigid for `can`'s shift,
   exposure/specular, and local appearance variation. More importantly, the
   registration acceptance rule is not a reliability measure: it accepts
   `5/5` even when the downstream local residual is catastrophically noisy.

### Decision-contract overlap

`L1-L0` is mathematically positive on oracle F1 and AP, so the weak OR clause
in the pilot's `CONTINUE_DIAGNOSTIC` bullet is true. The explicit terminal stop
is also true because R1 is worse than G0 on oracle F1, AP, and component
recall. The contract did not state precedence for this overlap. Therefore the
frozen outcome is `CONTRACT_CONFLICT`, not a unique `NO_CONTINUE` verdict. The
post-result operational adjudication is to stop raw R1 because it is the
endpoint proposed for four-fold promotion; that adjudication is not presented
as preregistered precedence.

## Continuation Assessment

Do not promote raw R1 to four folds, five objects, or Gate 3. Do not use the
public AD2 labels to retune the local radius, RANSAC acceptance, support count,
reconstruction statistic, threshold, or morphology.

The reusable positive evidence is narrower: native layer-7 G0 is the strongest
of the four raw arms and is much better than the local arms, although its
absolute `can` F1 remains only `0.0178`. If a new method family is pursued, it
should be preregistered and learned/calibrated on normal-only AD1/support data:

1. retain unrestricted high-resolution G0 candidates rather than hard spatial
   exclusion;
2. train a high-resolution normal-density/flow or score head instead of using
   a single raw cosine residual;
3. combine it with the unchanged coarse H+DVT MLP branch using support-LOO
   normal calibration and an explicitly frozen reliability gate;
4. require that any future correspondence gate first reduce the held-out-normal
   upper tail without losing synthetic thin-cue response before seeing AD2.

## Conclusion

Actual full-public `can` evaluation rejects DARC-v1's raw local/reconstruction
terminal. Frozen verdict: `KILL_FOR_CLAIM / CONTRACT_CONFLICT`. Post-result
operational decision: stop raw R1 and do not promote it.

The executed manifests recorded hashes for 26 selected code files but did not
capture a complete dependency/environment lock; the prior report wording
"complete transitive code hashes" was too strong. The DARC files are also
currently untracked in the dirty worktree. Consequently this run's provenance
is anchored by its manifests and retained content hashes, not a git commit or a
fully sealed environment. Post-run hardening adds the two omitted local imports,
runtime package versions, a fresh-root guard, and a coverage-file digest for
future cells; these improvements are not back-projected onto this result.

## Post-Conclusion Storage Cleanup

Completed after common evaluation and validity review. The valid cell had 648
TIFFs before cleanup; the valid root and all related invalid/smoke/superseded
raw-ladder roots now contain `0` TIFFs. The latency-only
`common_eval_parallel_fast/` duplicate was also removed after its nine compact
outputs matched the canonical copies byte-for-byte. The canonical evaluator
still has `12/12` JSON/TSV/manifest files, all eight shard completion and
coverage records remain, and `evaluation_complete.txt` still reads
`evaluation_complete`. The local compact pullback has 46 files (`1.1M`) and
no TIFFs.
