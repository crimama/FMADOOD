# Support-Orbit Velocity Mismatch Stage -1 Preregistration

Date: 2026-08-11
Protocol ID: `sovm-stage-minus1-20260811`
Status: `PRE-REGISTERED / NOT RUN`
Maximum verdict: `PROMISING_DIAGNOSTIC`

## 1. Negative Evidence Intake

The predecessor SO-CFM correction achieved macro coverage `0.520364`, macro
p95 tail reduction `-0.003347`, and beat the A1 orbit-bank tail on `0/4`
objects. Its synthetic guard had zero terminal hinge loss in all 16 fits. SOVM
therefore cannot transport support, reuse the old guard, tune SO-CFM capacity,
or claim that a more expressive field repairs correction.

This branch is admitted only because it changes the operator and evidence:
normal velocity is a reference against which a geometric query path is tested.
It is not admitted if its score is absorbed by raw/orbit distance, a constant
field, shuffled velocity, shuffled condition labels, or an oracle.

## 2. Motivation and Claim Boundary

### Claim

Under a finite support-derived nuisance basis, a patch with a normal shift
should admit at least one low-disagreement support-to-observation velocity
path, while a defect patch should retain high disagreement.

### Evidence

Paired leave-one-support-image-out held normals and disjoint held synthetic
anomalies, scored without modifying query or support features.

### Boundary

- support-only synthetic mechanism assay;
- known synthetic transformation provenance is used only to create paired
  assay data and oracle ceilings;
- the primary score receives neither the true condition nor the clean source
  of the pseudo-query;
- no real anomaly-detection metric or public/private query is opened;
- strict result fields remain
  `superad_seg_AUROC_0.05=N/A`, `superad_seg_F1=N/A`, `rn_fmlk=N/A`, and
  `benchmark_comparable=false`.

### Positioning

SOVM is a candidate non-endpoint evidence source. It is not a deployable
method claim until a later all-eight same-16 SuperAD/RN-FMLK comparison.

## 3. Frozen Scope and Custody

- `target_dataset=MVTec AD2 single-image`
- remote data root, if later executed:
  `/home/hunim/Volume/DATA/mvtec_ad_2`
- accessible population in this stage: fixed support-16 `train/good` only
- support manifest:
  `skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/superad16_dinov2_reference_paths.json`
- support manifest SHA-256:
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`
- objects: `can`, `fabric`, `vial`, `walnuts`
- outer folds: SHA-ordered four-fold leave-one-support-image-out, 12 field-fit
  images and 4 held pseudo-query images
- object roles: flow-weak `can/vial`, flow-strong `fabric/walnuts`, broad
  retention stress `vial`
- backbone: frozen DINOv2 ViT-L/14, feature extraction contract
  `[5,11,17,23]`, primary readout layer 11, row L2 normalization
- resolution: 672
- forbidden: `test_public`, `test_private`, `ground_truth`, filename condition
  tokens, query-conditioned fitting, test-time gradient, memory update,
  post-metric threshold/layer/time/code/weight selection

Every artifact must record all access flags explicitly. Any forbidden access is
`INVALID_ORACLE_ACCESS`, not a negative scientific result.

## 4. Normal Field and Nuisance Basis

The signed code set is frozen:

```text
C = exposure±, gamma±, contrast±, reciprocal-red/blue-white-balance±
|C| = 8
```

For every fit support image and code, actual transformed images are passed
through DINO at normalized path times `0, 0.5, 1`. These correspond to physical
amplitude scales `0, 1, 2`; the endpoint scale 2 is frozen from the predecessor's
held-shift assay. This tests held-**source** generalization at a support-
synthesized, known amplitude; it does not test held-amplitude extrapolation.
The normal-only conditional field is fitted to the two explicit midpoint
finite-difference targets

```text
t=0.25: (z(0.5)-z(0))/0.5
t=0.75: (z(1)-z(0.5))/0.5.
```

No synthetic anomaly enters Stage M fitting.

For each code `c`, define fit-only energy

```text
E_c = median ||z_c(1) - z_c(0)||²
```

over field-fit support patches. `E_c` is fixed before any held score is read and
normalizes signed-code energy during enumeration.

## 5. Primary SOVM Score

For spatially aligned fit-support anchor patch `z`, held pseudo-query patch
`q`, signed nuisance code `c`, and

```text
T = {0.25, 0.75},
```

define

```text
x_t(z,q) = (1-t) z + t q
g(z,q)   = q - z
r_t(z,q,c) = ||v_theta(x_t,c,t) - g(z,q)||² / (E_c + 1e-6)
M(z,q,c) = sum_{t in T} t² r_t / sum_{t in T} t²
S_SOVM(q,p) = min_{fit anchor i, code c in C} M(z_{i,p},q_p,c).
```

Intermediate `x_t` is not row-renormalized because the field is trained on
affine midpoints of normalized endpoints. `t=1` is omitted. Minimum aggregation
is fixed; no mean, median, temperature, or top-k sweep is allowed.

The selected anchor/code is diagnostic output only. It causes no parameter or
memory update.

## 6. Pseudo-Query Construction

For every held support image and every signed code:

- held normal: apply the frozen endpoint nuisance transform at physical scale 2;
- held local anomaly: apply thin scratch, then the same nuisance transform;
- held broad anomaly: apply Gaussian blur, then the same nuisance transform.

The held image is excluded from the primary anchor bank. Stage M held-family
masks are opened only after every arm's score map is frozen. Thin scratch and
broad blur never enter Stage M fitting or normalization. After Stage M
authorization, only CutPaste/chromatic-blotch fit masks may enter Stage G
ranking; `stage_g_eval` masks remain sealed until every Stage G arm map is
frozen.

Stage M held masks use a `stage_m_eval` SHA seed domain. Stage G training
families are CutPaste and chromatic blotch. Stage G is evaluated on a locked
`stage_g_eval` seed domain of thin scratch and broad blur that is disjoint from
Stage M and is not generated or opened unless Stage G is authorized. All
source regions, areas, and locations are SHA-derived. No held-family seed or
mask participates in fitting.

## 7. Stage M Arms and Hard Nulls

### Candidate and baselines

- `M0_sovm_enum`: primary all-anchor/all-code SOVM score.
- `B0_raw_endpoint`: same-position raw support endpoint distance.
- `B1_orbit_endpoint`: exact scale-2 transformed-support orbit-bank distance,
  enumerated over the same eight signed codes.
- `B2_flow_endpoint`: endpoint distance to support transported by the same
  Stage M normal field. This isolates score operator from field quality.

### Hard nulls

- `N0_constant_field`: replace the field with each code's fit-support mean
  velocity; preserve code energy and scoring cardinality.
- `N1_velocity_shuffle`: permute finite-difference targets across source and
  signed code before fitting; preserve target energy/cardinality.
- `N2_code_fit_shuffle`: keep every feature/velocity-target pair intact but
  permute its signed-code label across training samples with a fixed seed;
  preserve the code histogram, target energy, architecture, and all-code
  enumeration. An inference-only label permutation is not used because a
  minimum over the complete code set is permutation invariant.
- `N3_spatial_source_shuffle`: pair each query patch with a deterministic
  rolled support position/source while preserving patch count and feature
  energy.
- `N4_time_shuffle`: evaluate the real field with the fixed swap
  `0.25 <-> 0.75`.

### Oracle ceilings

- `O0_true_code`: primary anchors, but only the true synthetic code.
- `O1_same_source`: all codes, but the unavailable clean held image is the
  anchor.
- `O2_full_oracle`: true code and same held-image anchor.

Oracle rows are marked `FORBIDDEN_ORACLE_BOUNDARY`. They cannot determine a
threshold, arm, continuation, or positive verdict. They may only force a
negative `ORACLE_ONLY / NO_CONTINUE` diagnosis when a primary arm fails. A free
patchwise code is not executed because it is a non-identifiable absorption
upper bound.

## 8. Stage M Metrics

The primary metric is held-synthetic pixel/patch
`diagnostic_seg_pAUROC_0.05`, calculated separately for thin scratch and broad
blur and macro-averaged over the four objects. It is not the benchmark
`seg_AUROC_0.05` and must never populate the SuperAD comparison field.

Each arm also reports:

- `diagnostic_seg_AUPRC`;
- `held_normal_fpr_at_fit_q95`, with the q95 threshold fitted from fit-support
  inner leave-one-image-out normal pseudo-queries only;
- `held_defect_tpr_at_fit_q95` for local and broad families;
- `top5_defect_recall`;
- `defect_clean_pair_auc` on matched patch pairs;
- `positive_map_coverage` at the fit q95 threshold;
- `unique_positive_coverage_vs_orbit`: positive patches selected by SOVM but
  missed by B1, divided by all positive patches;
- `score_spearman_vs_raw` and `score_spearman_vs_orbit`;
- selected-code entropy and maximum selected-code fraction;
- gaps to O0/O1/O2;
- per-object, fold, held source, code, and anomaly-family counts.

Uncertainty uses 10,000 paired hierarchical bootstrap draws, resampling objects
then held images within objects. Bootstrap intervals cannot rescue a failed
object-level safety gate.

## 9. Stage M Validity Gates

All must pass before endpoint metrics are read.

1. exact support hash, object set, four folds, signed-code set, times, layer,
   resolution, and physical amplitudes;
2. held pseudo-query source excluded from the primary anchor set;
3. all eight codes enumerated in `M0_sovm_enum`; no true-code routing;
4. Stage M field normal-only; no old A3 guard or synthetic loss;
5. no support/query correction and no query/test/GT/filename access;
6. training and held synthetic families are disjoint;
7. every real/null arm has identical held source/code/family coverage;
8. N0--N4 energy and cardinality differ from their matched real quantity by at
   most 5%;
9. every score is finite and nonconstant;
10. oracle rows are excluded from every candidate threshold and every positive
    or continuation verdict; they may trigger only an `ORACLE_ONLY` negative;
11. no dense maps are persisted after compact metrics are written.

Failure is `INVALID` or `INVALID_ORACLE_ACCESS`.

## 10. Stage M Strict and Continuation Gates

### Strict Stage M pass

All unrounded values must pass:

1. M0 macro `diagnostic_seg_pAUROC_0.05 >= 0.70`; local and broad family macro
   each `>=0.65`; no object-family row `<0.60`;
2. M0-minus-B0 macro pAUROC `>=0.03`, M0-minus-B1 `>=0.02`, and
   M0-minus-B2 `>=0.02`, with positive paired-bootstrap lower bounds and wins
   on at least `3/4` objects for every comparison;
3. M0-minus-N0, M0-minus-N1, and M0-minus-N2 macro pAUROC each `>=0.05`, with
   wins on at least `3/4` objects; M0 also beats N3 and N4 on at least `3/4`;
4. `held_normal_fpr_at_fit_q95 <=0.10` macro and no object exceeds `0.15`;
5. local and broad `held_defect_tpr_at_fit_q95 >=0.30` macro and
   `top5_defect_recall >=0.30` for both families;
6. `unique_positive_coverage_vs_orbit >=0.10` macro;
7. selected-code entropy `>=log(3)` on at least `3/4` objects and no code owns
   more than 70% of selected patches macro;
8. all four objects are complete and no object loses more than `0.02` pAUROC
   versus B0, B1, or B2.

Verdict: `STAGE_M_PASS / AUTHORIZE_STAGE_G`.

### One-step Stage M continuation

If strict gates fail, Stage G is still authorized as the single bounded
continuation only when all of the following hold:

- validity passes;
- M0 macro pAUROC `>=0.62` and both held families exceed `0.58`;
- M0 beats B1 and B2 by at least `0.01` macro;
- M0 beats N0, N1, and N2 by at least `0.02` macro and beats N3 and N4 by at
  least `0.01` macro;
- at least `2/4` objects improve over both B1 and B2;
- held-normal FPR is `<=0.15` macro;
- the primary enumeration, not only O0/O1/O2, contains the signal.

Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC_STAGE_G`.

Hard-stop Stage M with `KILL_FOR_CLAIM / NO_CONTINUE` if macro pAUROC is below
`0.62`, N0/N1 absorb the score, both anomaly families are non-positive, selected
codes collapse on at least `3/4` objects, or only an oracle succeeds.

## 11. Conditional Stage G: Score-Aligned Synthetic Ranking

Stage G is forbidden unless Stage M strict or continuation authorization fires.
It initializes from the Stage M normal field and preserves the same normal FM
loss. The new term is computed on CutPaste/chromatic-blotch positive patches
and their matched clean patches using the primary enumerated SOVM score:

```text
L_rank = mean relu(0.5 + S_theta(clean) - S_theta(defect))
L_total = L_normal_FM + 1.0 * L_rank.
```

The exact minimum over the same anchor/code set is used during fitting; no
soft-min temperature or alternative margin is evaluated. Every Stage G
continued-training arm uses exactly 20 epochs, AdamW learning rate `1e-3`,
weight decay `1e-4`, batch size `4096`, the same SHA-derived minibatch seed,
and identical normal-FM batches.

Stage G arms:

- `G0_frozen_normal`: frozen Stage M field with no additional optimizer step;
- `G0_continued_normal_only`: Stage M field continued for the same 20 epochs
  using only normal FM loss;
- `G1_score_ranked`: normal FM plus the score-aligned ranking term;
- `GN1_mask_shuffle`: G1 with positive masks spatially rolled before selecting
  ranking patches;
- `GN2_pair_label_swap`: G1 with clean/defect ranking labels exchanged.

Before any Stage G optimizer step, record ranking-hinge activation rate and
gradient norm. `guard_activation_rate_prefit >=0.20` and a finite nonzero
ranking gradient are mandatory. Otherwise Stage G stops as
`INACTIVE_GUARD / NO_CONTINUE`.

Stage G uses the unopened `stage_g_eval` thin-scratch/Gaussian-blur seed split
and the same fit-only q95 calibration rule as Stage M. It passes only if all
hold:

1. G1-minus-G0_continued macro pAUROC `>=0.03`, G1-minus-GN1 `>=0.03`, and
   positive paired-bootstrap lower bounds; G0_frozen is reported but cannot
   substitute for the continued-training causal control;
2. G1 pAUROC `>=0.72`, each held family `>=0.67`, and no object-family row
   `<0.60`;
3. G1 beats B0, B1, and B2 macro pAUROC and wins on at least `3/4` objects for
   every comparison;
4. G1 held-normal FPR increases by at most `0.03` absolute versus
   G0_continued and remains `<=0.12` macro;
5. G1 normal FM loss increases by at most 10%, selected-code entropy remains
   `>=log(3)` on `3/4`, and no signed code owns more than 70% macro;
6. G1 local and broad `held_defect_tpr_at_fit_q95 >=0.35` and
`unique_positive_coverage_vs_orbit >=0.12` macro;
7. GN2 macro held-family pAUROC is at most
   `G0_continued_normal_only + 0.005` and cannot match G1 on either held family.

Passing verdict: `PROMISING_DIAGNOSTIC`.

Stage G yields `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC_STAGE_G_REPLICATION` only
if G1 beats G0_continued and GN1 by at least `0.015` macro, both held families remain
above `0.62`, held-normal FPR remains at most `0.15`, and failure is confined to
one object. The only licensed continuation is a third `stage_g_replication`
SHA seed domain with identical constants and held families.

Otherwise the verdict is `KILL_FOR_CLAIM / NO_CONTINUE`.

## 12. Result Schema

Each object artifact and the aggregate must contain:

```json
{
  "protocol_id": "sovm-stage-minus1-20260811",
  "target_dataset": "MVTec AD2 single-image",
  "support_manifest_sha256": "7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3",
  "benchmark_comparable": false,
  "superad_seg_AUROC_0.05": null,
  "superad_seg_F1": null,
  "rn_fmlk": null,
  "method_seg_AUROC_0.05": null,
  "delta_vs_superad": null,
  "query_or_test_access": false,
  "test_public_accessed": false,
  "test_private_accessed": false,
  "ground_truth_accessed": false,
  "uses_filename_condition_tokens": false,
  "query_condition_oracle_used_by_primary": false,
  "dense_maps_persisted": false
}
```

Diagnostic metrics, object/fold rows, arm/null fields, hashes, runtime, peak
memory, and verdict are stored separately from benchmark fields.

## 13. Implementation Boundary

The completed SO-CFM runner and aggregator are immutable evidence and must not
be edited for this protocol. New code, if execution is subsequently requested,
is limited to:

- `src/flow_tte/sovm.py` for the affine-path mismatch score and Stage G loss;
- `scripts/run_sovm_stage_minus1.py`;
- `scripts/aggregate_sovm_stage_minus1.py`;
- `scripts/run_sovm_stage_minus1_remote.sh`;
- focused `tests/test_sovm*.py` files.

Existing support loaders, nuisance transforms, DINO wrapper, normal SO-CFM fit
primitives, synthetic generators, and metric utilities are reused. No backbone
or completed protocol file is changed.

## 14. Evaluation Alignment and Future Baseline

This design directly tests whether normal-flow disagreement is anomaly-aligned
and not merely an endpoint-distance duplicate. It does not test real query
condition identification, because the primary score statically enumerates a
frozen finite basis. It does test whether enumeration collapses or absorbs
defects.

No same-condition SuperAD metric is required for this support-only diagnostic.
A future benchmark promotion, not authorized here, must validate or rerun the
same-condition baseline using
`configs/baselines/same16_superad_ad2.yaml` and
`configs/baselines/same16_rn_fmlk_ad2.yaml`, the exact same 16 references, all
eight objects, full TESTpublic, and the repository `seg_AUROC_0.05`/best-F1
evaluator.

## 15. Execution and Storage Boundary

This document designs but does not launch the experiment. GPU allocation must
be checked afresh immediately before a later run; no previous server mapping is
frozen here.

When executed, only compact JSON/TSV/log/report artifacts are retained. Dense
feature paths, transformed images, synthetic masks, and `anomaly_maps/` are
regenerable and must be deleted after the result and verdict are recorded.
