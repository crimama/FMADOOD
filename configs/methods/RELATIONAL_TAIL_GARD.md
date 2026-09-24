# GARD-RGTA (G1–G2)

GARD preserves the D3 residual adapter and adds a support-only adaptive
preservation budget. A frozen baseline teacher measures photometric shift
susceptibility

`nu = mean(|teacher_shift - teacher_clean|) / (mean(|teacher_clean|)+eps)`

and sets `b = b0/(1+kappa*nu)`. The observed F3 support-training scale was
`P≈4e-4…1.7e-3`, so G1/G2 lock `b0=0.0012` and `kappa=1`.
The preservation penalty is
`P = keep + beta*relational_loss`; a detached clipped dual variable increases
when `P > b`, yielding `L = anomaly + lambda*P`. Lambda uses the detached
budget-relative violation clipped to `[-1,1]`, EMA decay `0.9`, and update rate
`rho=0.01`; this avoids raw-score unit dependence and immediate saturation in
64 steps. Teacher inputs contain only
normal support views; query images and labels are never used for fitting.

G1 enables adaptive preservation. G2 additionally weights trusted synthetic
positive hinges by detached shift-view consistency and budget slack. The
weights affect only synthetic sampling loss and cannot bypass the preservation
constraint. All diagnostics (`nu`, `budget`, `P`, `lambda`, `violation`, and
synthetic weight) are retained in compact training history; maps and weights
remain disabled.

| Config | Adaptive budget | Budget-aware synthetic |
|---|---:|---:|
| `relational_tail_G1.yaml` | on | off |
| `relational_tail_G2.yaml` | on | on |

This is a development ablation on the existing seen-dev protocol. It is not a
claim of robustness or paper-level validation until matched results support it.

## H1 — one issue only: delayed/zero preservation feedback (2026-09-12)

Existing G1 vial logs show a zero preservation coefficient at steps 19–33,
including eight budget-violating steps. H1 addresses only that controller
behavior; it is an engineering correction, not a novelty claim or a proven
explanation of the performance regression.

`relational_tail_H1.yaml` equals G1 plus `immediate_preservation: true`.
Normal support selection, backbone, metric/adapter parameters, ranking,
photometric transforms, paired contrast and budget definition remain unchanged.
Trusted weak synthetic and budget-aware synthetic stay **off**, as in G1;
the original synthetic training of G1 is retained.

Replace the accumulated EMA-dual coefficient with current-step feedback:

`lambda_t = stop_gradient(clip(P_t / b_t, lambda_init, lambda_max))`

`L_t = L_anomaly,t + lambda_t * P_t`

For this config the floor is 0.1 and the ceiling is 1.0. The coefficient never
turns off and reaches 1.0 at the budget boundary, including after many low-error
steps. It is computed **before** the current optimizer update. The controller
gain is detached; the existing exact four-view gradient replay is unchanged.
`lambda_lr` and `violation_ema_decay` are retained in the config for a one-key
diff but unused in H1. The default switch is off, preserving G1/G2/F1–F3 paths.
Conflict projection is rejected with this mode because that path would bypass
the weighted preservation gradient. No new trainable parameters, caches,
forward passes, maps or checkpoints are introduced.

Verification in this change: coefficient/gradient checks below and above the
budget; recovery from old zero-lambda/negative-EMA state; config-only difference;
tiny-backbone integration and historical-method regression tests. These tests
are not benchmark runs and do not establish improved AUPRO.

Next development comparison is only G1 versus H1: seed0, normal16 with matching
support identities, all 8 existing seen-development classes, 448+672, 64 steps,
same evaluator. Primary decision is full8 AUPRO@0.05; vial and constraint
violations are diagnostics, not substitutes for the macro metric. Budget at
most one run per config/object (16 class runs), no sweep or extra components.
Reject a performance claim if the macro does not improve; a positive coefficient
does not guarantee P <= b, useful anomaly discrimination, or superiority to D3.
No benchmark has been launched for H1 in this change.
