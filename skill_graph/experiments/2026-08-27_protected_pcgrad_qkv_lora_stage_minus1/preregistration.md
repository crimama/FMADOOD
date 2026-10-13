# Protected Asymmetric-PCGrad QKV-LoRA Stage -1 preregistration

Date: 2026-08-27  
Protocol: `protected-pcgrad-qkv-lora-stage-minus1-20260827`  
Status: `PREREGISTERED / NO NEW-PROTOCOL RESULTS INSPECTED`  
Seed: exactly one, `20260826`

## Negative evidence and positive signal intake

TARCD-S3 showed that a shared DINOv2 QKV-LoRA adapter is trainable after the
vector-loss conditioning repair, but the candidate remained below the static
hard null (`-0.1439931`) and strict synthetic contribution was `0/12`.

The Oracle-Family-Expert upper bound removed cross-family parameter averaging.
Its global hard-null margin improved to `-0.0832106`, q99 reduction was
`+0.0102168`, positive hard-null cells appeared in `3/12`, and the first
strict synthetic contribution appeared in `1/12`.  It still failed anomaly
retention on `can/vial`.  J1 obtained the strongest correction direction while
erasing anomaly evidence in those objects; J2 approached the hard null but
collapsed correction norm.

This rules out inactive LoRA as the primary explanation and closes
family-routed/hypernetwork LoRA.  SCOPE Cycle 2 also closes score-only CEB,
threshold, and max-stream continuations.  The remaining falsifiable question
is whether nuisance and anomaly objectives conflict inside the same shared
QKV-LoRA update.

## Sole change

Keep TARCD-S3's single shared DINOv2 ViT-L/14 adapter and replace only the
candidate gradient-composition rule.

For the existing candidate loss components, define

`L_N = correction_clean + 0.1 * identity_clean`

and

`L_P = correction_anomaly + 0.5 * recomposition +
       0.1 * identity_anomaly + 0.1 * RN_guard`.

For all trainable LoRA tensors concatenated as one logical vector:

`g_N = grad(L_N)` and `g_P = grad(L_P)`.

If `dot(g_N,g_P) < 0`, replace only the nuisance gradient by

`g_N' = g_N - dot(g_N,g_P)/(||g_P||^2 + 1e-12) * g_P`.

Otherwise `g_N' = g_N`.  The optimizer receives `g_N' + g_P`, followed by the
unchanged global clip `1.0` and AdamW step.  The protected gradient is never
projected away.  This is a one-sided PCGrad/A-GEM-style protection rule.

Everything else is frozen from S3:

- one shared adapter per object; no family-specific parameter or router;
- DINOv2 ViT-L/14, fused QKV blocks 8--11, rank `4`, alpha `8`, zero-B,
  dropout `0`;
- AdamW `1e-4`, zero weight decay, bf16 forward, fp32 loss;
- exact 144-row schedule, loss weights, randomization namespace, support,
  shifts, generators, masks, memory, scoring, and evaluator;
- no early stopping or checkpoint selection;
- ordinary one-adapter inference with no projector, second stream, score
  fusion, threshold, CEB, or bypass.

Shift-family labels define the already frozen synthetic schedule and evaluation
strata only.  They never route parameters at training or inference.

## Arms

- `B0_FROZEN_TEACHER`: frozen backbone.
- `B1_IDENTITY_ZERO_INIT`: zero-init LoRA identity.
- `M0_ENDPOINT_KD`: matched endpoint-distillation control.
- `D0_DIRECT_CLEAN`: clean correction only.
- `D1_DIRECT_RECOMPOSITION`: clean plus synthetic correction/recomposition.
- `D2_SCALAR_SHARED`: S3 scalar-sum shared-LoRA baseline.
- `P2_PROTECTED_PCGRAD`: candidate; identical D2 losses with the sole
  protected gradient-composition change.
- `N1_SHUFFLED_PROTECTION_PCGRAD`: wrong-base/target correction with the same
  protected update rule.
- `N2_RANDOM_MATCHED_LORA`: norm-matched random adapter.
- `N3_STATIC_MEAN_ORACLE`: privileged shift-family static correction.
- `O_PAIRED_TEACHER`: privileged paired-clean upper bound.

Trained arms are M0, D0, D1, D2, P2, and N1.  B0/B1/N2/N3/O remain frozen,
analytic, or matched-state controls.  The candidate is P2.  Its hard-null
envelope is the maximum correction cosine among M0, D2, N1, N2, and N3, so the
immediate scalar-sum causal baseline cannot be bypassed by a separate label.

## Fixed schedule and trace

For every object and trained arm, execute exactly
`16 support bases x J0/J1/J2 x G0/G1/G4 = 144` updates in S3's frozen shuffled
order.  Record fixed traces at steps `1,36,72,108,144`; traces cannot select a
checkpoint.

For P2 and N1, record:

- every loss component and total scalar-equivalent loss;
- `||g_N||`, `||g_P||`, their dot and cosine before projection;
- conflict flag, projection coefficient, removed-gradient ratio;
- final gradient norm before/after clipping and `cos(g_final,g_P)`;
- normalized protected alignment after projection;
- the actual AdamW parameter delta and `-dot(g_P, delta_theta)` normalized by
  the two norms;
- cumulative conflict count/rate, adapter norm, and adapter hash.

All values must be finite.  On conflict steps, the raw post-projection
protected alignment must be at least `-1e-8` after norm normalization.  The
actual AdamW-update alignment is a separate fail-closed mechanism diagnostic,
because momentum/preconditioning can reintroduce conflict.

## Custody

- Dataset: `/home/hunim/Volume/DATA/mvtec_ad_2`.
- Objects: `can`, `fabric`, `vial`, `walnuts`.
- Exact SuperAD-16 support manifest SHA-256:
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Fit: exact 16 support `train/good` paths and deterministic J0/J1/J2 plus
  G0/G1/G4 descendants only.
- Evaluation: the same four support-disjoint `train/good` bases and unseen
  `G2_structural_scratch` descendants used by S3 and Oracle-Family.
- `TESTpublic`, private test, GT, query updates, query-memory updates, benchmark
  thresholds, and filename-condition routing remain sealed.
- No dense feature, score, gradient, or anomaly maps may be retained.

## Cell contribution definitions

`pcgrad_contribution_pass` requires all of:

1. `P2` held-G2 recomposition error `<=0.90 * D2`;
2. `P2` anomaly cosine `>= D2` and P2 anomaly norm remains in `[0.70,1.40]`;
3. `P2` correction cosine `>= D2 - 0.05`;
4. `P2` q99 reduction `>= D2 - 0.02`;
5. the object had at least eight conflict steps and at least 90% of its
   conflict steps had nonnegative normalized actual-update protection within
   `1e-8` tolerance.

The existing D1-versus-D0 synthetic-contribution and P2-versus-D1 RN
contribution diagnostics are retained separately.

## Preregistered gates

Aggregation is cell -> three-shift object median -> four-object median.

`PROMISING_PROTECTED_PCGRAD_SIGNAL` requires:

1. all 12 cells structurally valid, finite/nonconstant, exact custody/hash and
   zero-init/RN tolerances, exact 144 steps and fixed traces, nonzero trained
   adapters, and no forbidden access;
2. every object has at least eight P2 conflict steps, every raw projected
   alignment passes, and at least 99% of conflict steps have nonnegative
   normalized actual-update protection within `1e-8`;
3. global P2 hard-null correction margin `>=0`, with at least three of four
   object medians strictly `>0`;
4. global P2 q99 reduction `>=0`, with every object `>=-0.05`;
5. every object median anomaly cosine `>=0.85`, anomaly norm in `[0.70,1.40]`,
   and at least `10/12` cells pass the same anomaly guard;
6. `pcgrad_contribution_pass` in at least `6/12` cells;
7. global P2 correction cosine exceeds D2 by at least `0.025`, and P2 beats
   shuffled protected-PCGrad in correction cosine without worse held-G2
   recomposition.

`PROGRESSIVE_PROTECTED_PCGRAD_SIGNAL` is a non-claim continuation signal only.
It requires:

1. structural/custody validity and the raw projection algebra above;
2. every object has at least eight conflict steps and at least 90% of conflict
   steps have nonnegative actual-update protection;
3. global q99 reduction `>=0` and every object `>=-0.05`;
4. at least `9/12` anomaly-guard cells, no object median anomaly cosine below
   `0.80`, and all object anomaly norm medians in `[0.70,1.40]`;
5. `pcgrad_contribution_pass` in at least `3/12` cells;
6. global P2 hard-null margin `>=-0.094`, P2 improves the D2 hard-null margin
   by at least `0.025`, and at least `4/12` P2 cells have positive hard-null
   margin.

Otherwise the verdict is
`INSUFFICIENT_PROTECTED_PCGRAD_SIGNAL / END_SYNTHETIC_QKV_LORA_BRANCH`.
This closes gradient-combination-only shared QKV-LoRA.  No second seed, rank,
block, alpha, learning-rate, optimizer, loss-weight, threshold, generator,
mask, checkpoint, family-routing, or score-bypass sweep is licensed.

No public or benchmark branch is opened by this Stage -1 result.

## Remote execution and other-user safety

- Use only dsba3 GPUs `0,1,2,3` in the fixed project container
  `hun_fsad_tta_012`.
- Fixed map: `can/fabric/vial/walnuts -> 0/1/2/3`.
- Abort before snapshot creation and again before launch if any selected GPU
  is occupied; do not stop, reset, or inspect the owner of an occupied GPU.
- Never inspect, enter, stop, restart, rename, mount, or modify another user's
  container, process, file, volume, or job.
- Do not use dsba2.  Do not list other containers or system-wide user/PID
  details.  Process checks, if needed, must match this exact run ID inside the
  fixed project container.
- Use one immutable snapshot, offline DINO cache, focused remote tests, an
  actual CUDA QKV-LoRA model/gradient smoke, fail-closed aggregation, and
  compact JSON/text artifacts only.

The run stops after the support-only four-object aggregate.
