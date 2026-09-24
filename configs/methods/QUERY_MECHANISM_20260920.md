# Query-reference correction mechanism controls — 2026-09-20

Frozen before new evaluations. User authorized testing the proposed QR1
mechanism linking sparse normal reference coverage and acquisition mismatch.
This tests an existing estimator, not a novel method or proven nuisance removal.

## Hypotheses, comparator and boundary

QM_A: original QR0. QM_B: original QR1 fitted bounded translation. QM_C: same
query/layer/grid translation magnitude, but an isolated seeded random direction
independent of query content. QM_D: negate the fitted vector, preserving norm.
All use initial independent reference NN correspondences, ridge1,max_norm0.1,
no trimming or spatial crossfit. Direction changes occur after bounded fitting
and before reference unit normalization. Pre-normalization norm equality does
NOT imply equal final bank displacement or equal score changes. Random seed
20260920+layer_index*1000003+height*1009+width; fixed per layer/grid, isolated CPU
generator, no support/bank RNG consumption. No donor query is added: that would
introduce cross-query data and a different adaptation boundary.

64 evaluations:4arms ×1/16shot ×full8, support seed0. Refit all arms for matched
runtime/code provenance. Same nested SD_support manifests, frozen DINOv2-L,
layers11/23,448/672,256 tokens/source, one bank, original R1 MAD pipeline.
No new training parameters. No hyperparameter/severity sweep or automatic
multi-seed replication. The current budget diagnoses direction specificity at
one support seed, not support-sampling robustness or a monotonic shot law.

## Evaluation evidence

Reuse the verified real_acquisition evaluation hook: full public scenes, all
available conditions, condition GT, primary regular and shift_1; others secondary.
Extra16 fixed normal calibration images excluded from support16 set thresholds;
these are evaluation-only, not a strict one-image deployment claim. Retain LOO
threshold sensitivity, per-image normal/defect means, FPR/TPR and condition-wise
AUPRO/pAUROC. Seen-development only. No condition labels or GT in fitting or
prediction; no maps/weights or persistent fitted-query state.

Primary mechanism questions:
1. Does fitted-direction B beat magnitude-matched random C and reversed D in
   AUPRO at both shots under shift_1, as well as beat A? Report regular separately.
2. For each matched bad image, define score margin=defect_mean-normal_mean.
   Compare B−A, B−C, B−D margin change alongside normal/defect mean changes.
   Normal-score reduction alone is not separation improvement; mean margin is
   also not equivalent to AUPRO. Use within-image normal pixels for this margin;
   use normal-only images for false positives. Never merge these populations.
3. Across classes, does B's benefit accompany correction of larger baseline
   normal mismatch at1 versus16shot? Report classwise values and exploratory
   association, not a causal/monotonic few-shot claim. This is not proof that
   reference sparsity and real shift share exactly one latent cause.

Direction-specificity gate: B−A, B−C, B−D shift1 macro AUPRO positive at both
shots, and B beats each comparator in>=5/8 classes at each shot. If not, the
strong direction-specific claim is unsupported. Mechanism support additionally
requires positive B−A mean within-bad-image margin change in>=5/8 classes at
both shots; otherwise distinguish performance from a mean-separation account.
Report these gates separately. There is no predeclared minimum significance
effect and no p-value claim from one seed. Passing prompts a replication plan,
not an automatic larger experiment or novelty claim. Failures are informative.

Calibration sensitivity uses existing LOO bounds and is not a CI. Labels and
condition names only index post-scoring analysis. Full correction vectors and
their alignment with physical-lighting feature changes are NOT saved here;
direction controls test specificity, not exact latent nuisance identification.

Existing local/dsba3 containers and assigned GPUs; local can/fabric, dsba3 six
remaining classes. Generic run/collect helpers, selected local snapshot, no
push/container changes. Preserve source artifacts and historical ledger rows.
