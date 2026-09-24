# K3 — budget-constrained Adam displacement (2026-09-12)

## Hypothesis and scope

The latest matched H1/K1/K2 comparison does not support replacing K1 with K2:
K2 improves only vial/sheet_metal versus K1 and reduces the full8 primary metric.
See the [canonical evidence](../../results/comparisons.md#student_boundary_20260912_combined_status).
The posthoc hypothesis is that some useful K1 learning can be retained while
restricting parameter changes that increase the existing normal preservation
surrogate. Gradient conflict is NOT established by the existing scalar losses.

K3 changes one component: the applied optimizer displacement. Keep K1's frozen
teacher candidate selection (.25/.5/1), source exclusion, normal-tail ranking,
paired shift/contrast losses, immediate preservation gain, frozen backbone,
adapter/metric and inference. Loss formulas stay fixed; their later values
may change because the learned parameters follow a different trajectory.

This is a bounded constrained-optimization ablation, not a novelty claim.
Existing F3 projects score-leaf anomaly gradients before VJP replay. K3 instead
uses a parameter-space preservation gradient and projects the actual AdamW
proposal after momentum/preconditioning. It neither implements score-space
PCGrad nor assumes projecting a raw gradient constrains an Adam update. No
K2 selector, new synthetic family, confidence gate or loss coefficient is added.

## Mechanism

For the current source-excluded normal support view, preserve H1's objective:

    P(theta) = keep_weight * MSE(s_theta, s_teacher)
               + beta * SmoothL1(standardize(s_theta), standardize(s_teacher))
    b = budget_base / (1 + susceptibility_scale * frozen_shift_susceptibility)
    L = L_rank + L_shift + L_contrast + clip(P/b, lambda_init, lambda_max) * P

The gain remains detached. The common loop computes and clips dL/dtheta as
before. Replay the cached clean features once to obtain g = dP/dtheta without
changing those accumulated gradients. Let d be AdamW's proposed theta change:

    c = max(b - P, 0)
    d_star = argmin_u 0.5 * ||u - d||^2, subject to g.T @ u <= c
           = d - max(g.T @ d - c, 0) / ||g||^2 * g, when g != 0
    theta_next = theta + d_star

With zero g, retain d and record that the first-order guard is uninformative.
Numerically use a normalized gradient and float64 reductions, then apply in
the original parameter dtype. When inactive, leave Adam's parameters bitwise
untouched. Keep Adam's proposed first/second moments and step count unchanged;
this is projected Adam, not an optimizer-state rollback or moment correction.
Log actual rounded displacement, so any first-order numerical residual is
visible. The existing logged losses are pre-update quantities.

Below budget the step may spend the remaining linearized slack. Above budget
it may not increase the surrogate to first order; it is NOT forced back below
budget. Orthogonal displacement is retained and the ideal projection cannot
increase step norm. This does not guarantee retention of useful discrimination.

## Predictions and failure conditions

- K1 synthesis/strength/family and support schedule remain identical, including
  RNG consumption. The preservation gradient replay must reproduce P.
- When g.T @ d > c, projection activates; applied g.T @ d_star <= c up to
  floating-point rounding. Record raw/applied linear change, slack, gradient
  norm, raw/applied step norm, correction norm and zero-gradient steps.
- The performance hypothesis is K3 improves full8 AUPRO over K1 while reducing
  the vial regression. Neither lower P/b nor activation alone is success.
- A zero gradient at initialization, high curvature, NN identity switches or
  changing support images can violate finite-step/all-support preservation.
- Preserving a biased frozen teacher can remove useful changes. A large
  correction fraction may simply suppress adaptation, repeating K2's failure
  pattern. Adam's retained moments may repeatedly propose rejected directions.
- The preserved score surrogate does not prove representation improvement,
  calibrated confidence, low query FPR or real-defect generalization. Class
  fits are independent; there is no cross-class shared-parameter tradeoff.

## Locked development comparison — launch authorized 2026-09-12

The user subsequently requested starting this experiment. The bounds below
are now the authorized launch plan; live execution status is recorded in
skill_graph/current.md and each queue's status.json, not inferred from this file.

- Minimal config ablation: K1 -> K3, 8 classes each; K3 switch off is an exact
  K1 control. Turning off boundary_synthetic as well reproduces H1. The latest
  H1/K2 results provide context, not replacements for matched current controls.
- Fixed seed0, normal16, 448+672, layer23, 64 steps; common CLI/evaluator and
  no_mask_no_rotation. Seen TESTpublic development, not a sealed final set.
- Reuse the support selection policy and verify exact support/query/GT content
  identities against the latest K1 manifest roots linked by the canonical
  ledger. Accept deltas only when the compared configurations and manifests
  match apart from the declared method change.
- Primary: full8 macro AUPRO@.05, K3 minus K1; <=0 is no primary improvement.
  Secondary: pixel pAUROC@.05; oracle F1 diagnostic only. Report all classes,
  especially vial recovery and whether K1's six class gains are retained.
  Positive macro but no vial recovery supports only a narrower method claim.
- Limit: 16 class runs, <=3600s/class, no sweep, automatic revision,
  extra seed or additional diagnostic evaluation. Stop on execution/finite/
  output-integrity errors.
- Keep each pair on its assigned existing device:
  local GPU0 can/fabric/fruit_jelly/rice/sheet_metal/walnuts; dsba3 physical GPU2
  vial/wallplugs, at most two workers/GPU. Reuse the existing containers and
  local-source snapshot synchronization; no Git push or provisioning.
- Queue roots: local /results/projected_update_20260912_local and remote
  /results/projected_update_20260912_dsba3_gpu2, K1 -> K3 in each queue.
- Use existing scripts/experiments.py run/collect and canonical Markdown/CSV;
  no new runner, collector, cumulative result table or invented performance row.

## Cost and implementation controls

No new trainable parameters (same 100384 for the full config) or inference
branch. One additional clean scoring/VJP per step, reusing current frozen CPU
features: zero extra backbone extraction. Parameter vectors are O(trainable
parameters), transient; no extra cross-step dense score/feature accumulation.
Measure fit time when run; do not claim equal compute to K1. Anomaly maps and
weights remain off. Fitting sees normal support only, never query data or GT.

Required checks: exact off parity to K1/H1; actual Adam-momentum counterexample;
halfspace minimality including zero/tiny gradients and budget slack; preservation
replay agreement; unchanged total .grad and Adam state; frozen backbone and
source/selection parity; same common config; scalar-only new diagnostics and
existing output-policy/runner regression tests. These CPU fixtures are code
verification, not benchmark results or evidence that the hypothesis succeeds.
