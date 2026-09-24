# RSG3 calibration × residual-gate factorial — prospective protocol

Authorized 2026-09-22 by user: run the planned experiment. Development study,
not an untouched final evaluation and not a new method/novelty claim.

## Hypotheses and fixed arms

Test whether normal-tail scaling improves multi-layer score comparability,
whether a support-q99 residual gate improves shallow-feature use, and whether
shallow features contribute beyond deep features. Multi-resolution is held fixed
as a method component, not introduced as the core problem.

| Arm | Divisor | Fused score per bag/resolution |
|---|---|---|
| FF_A | positive LOO MAD | deep mean + shallow |
| FF_B | LOO normal q95 | deep mean + shallow |
| FF_C | positive LOO MAD | deep mean + relu(shallow − normalized support q99) |
| FF_D | LOO normal q95 | deep mean + relu(shallow − normalized support q99), exact RSG3 |
| FF_E | LOO normal q95 | deep mean only |

B−A and D−C isolate scaling; C−A and D−B isolate residual gating;
D−E tests shallow contribution. Interaction is (D−C)−(B−A).
Ungated means ADDITION, not three-layer mean. The gate is threshold subtraction
plus rectification, not simply a binary mask. Its raw shallow-distance boundary
is q99 for both divisors, so normalization should not change binary activation
except floating-point ties. Gate rates are counterfactual diagnostics for A/B/E.

## Matched conditions and budget

Five arms × eight classes = 40 completed class evaluations, normal16 seed0.
Fixed manifest: `configs/methods/SD_support_16.json`. Frozen DINOv2 ViT-L/14,
layers 5/11/23, resolutions 448/672/896/1120, four random 256-token/source bags,
1NN cosine, gamma1, CQ99 component-max with no dilation. Zero trainable parameters.
No extra normal calibration set, no target adaptation, no additional backbone
calls for diagnostics. Store bank-feature hashes to verify within-class pairing.
Each class stays on one device/runtime for all arms. Local RTX4090 handles
can/fabric; existing dsba3 container GPUs 0/1/2/3 handle sheet_metal,
rice/vial, fruit_jelly/wallplugs, walnuts respectively. Explicit existing UUIDs
are used by launch commands; no container or GPU allocation change.

Primary higher-is-better AUPRO@0.05; secondary pAUROC@0.05; same native float16
streaming evaluator. `test_public_seen_development`, no_mask_no_rotation.
Evaluation masks are consumed only AFTER prediction. Full-public metrics plus
raw/pre-CQ and CQ metrics for each actual acquisition condition (including
regular and shift_1), PRO at FPR .001/.005/.01/.05, and normal/defect gate rates.
Condition-specific ROC metrics do NOT establish fixed-source-threshold transfer.
Gate activation is averaged over bags/scales with nearest-cell native projection;
both pixel-weighted and per-image summaries plus compact per-query rows retained.
Maps/checkpoints OFF; configs, provenance, fitting logs and compact results ON.

## Validation and stop/next decision

Before execution: formula tests, matched-bank test, exact FF_D/reference map
parity, diagnostics/label-boundary tests, existing streaming/pipeline regression.
During execution: finite scores, complete coverage, input/GT/backbone provenance;
runtime failures get a recorded safe retry, never a scientific negative label.
FF_D must reproduce the retained baseline within the existing numerical contract;
if material deviation occurs, diagnose comparability before interpreting effects.

Stop after the 40-cell screen: no parameter sweep or automatic shot/seed expansion.
Report all contrasts, per-class signs and largest-class-excluded mean, raw/CQ
differences, and acquisition-stratified gate behavior. Positive full8 D−B and
D−E, with positive largest-gain-excluded means and no >1 pp macro pAUROC loss,
qualify the corresponding claim for a proposed seed replication, not proof.
Few-shot amplification and target nuisance suppression remain unconfirmed unless
separate shot-controlled/threshold-transfer evidence supports them. Do not
relabel this post-hoc-motivated study as preregistered research discovery.
