# FBPA — Family-Balanced Paired Adaptation

Date: 2026-09-12. Initially a design-only request; the user subsequently
authorized implementation and experiments. The original design rationale and
future-tense protocol below are preserved; this execution addendum is current.

## Authorized implementation / execution addendum

- Implemented family_balanced_paired.py and existing CLI registration, with
  configs family_balanced_paired_{L3C,M0,M1,M2}.yaml. Child episode_enabled
  remains separate from the K1 synthesis switch. Off delegates exactly to L3C.
- Local existing-container validation:173 targeted tests passed,4 skipped
  (one opt-in CUDA test and three unavailable-rsync checks). These are implementation checks, not benchmark
  results. Full tiny-backbone fits verify all128 episode provenance records
  are identical across M0/M1/M2, shared-shift hashes and bank owners included.
- Verified two losses at unchanged parameters, one combined clip and one Adam
  step per update; exact mean-loss gradients, shared-clean loss accounting,
  off parity including forced weak synthesis, frozen backbone and scalar-only
  persistent state. Main-view extraction counts are1024/1024/768 across M0/M1/M2
  at two resolutions; total extraction calls and fit CUDA peak are also logged.
- Authorized bounded queue: L3C -> M0 -> M1 -> M2, full8/seed0/normal16,
  64 updates,3600s/class,32 class runs,1 worker/GPU. Local GPU0 six classes and
  dsba3 GPU2 vial/wallplugs, existing containers only. No sweep, Git push,
  provisioning, power changes or dense map/checkpoint output.
- Remote validation, immutable snapshot and launch evidence are recorded in
  skill_graph/current.md and skill_graph/log.md after they occur. Do not alter
  source/configs once either queue starts. No performance is claimed here.

## 1. Evidence and the single target problem

- Canonical results: results/comparisons.md, operating_restore_transfer_20260912
  full8 block. L3C-K1 AUPRO +0.25697pp; L3-L3C -0.04096pp. L3C improves5/8
  classes, especially can/vial/walnuts, but loses on fabric/rice/wallplugs.
- tail_contrast.py:174-189: K1 uses one source/family per update, switching
  family after16 updates. Across64 updates it already balances both families.
- transfer_update.py:140-170: L3C averages two sources/opposite families at
  every update. It also doubles episode exposure and excludes two bank sources.
- Thus the evidence supports a useful training-configuration signal, not
  meta-learning benefit or a demonstrated cause. Same source count, optimizer
  steps, and code do NOT imply equal synthetic exposure or equal compute.

Target hypothesis H1: long family-homogeneous update blocks bias the trajectory
toward the currently active synthetic family. Per-update family balance may
reduce this sensitivity. This is a posthoc hypothesis, not observed forgetting.

H2, the proposed extension: balancing families on the SAME normal source and
photometric transform may make their joint update less dependent on incidental
source/shift differences. This does not guarantee gradient agreement or better
generalization; using only one source per update can increase source variance.

Competing explanation: the prior gain came from more episodes, changed bank
population, changed random draws/order, or seed-specific variation. The design
below holds the first three fixed within M0/M1/M2; update packing/order is the
intervention. It cannot retroactively identify why original L3C improved.

## 2. Fixed episode population, independent of execution order

Use the same16 normal supports and frozen features as the current comparison.
For each source s, repeat r in0..3, and family f in {CutPaste, Scratch}, define
one episode e=(support_content_id, r, f):16*4*2=128 episodes per class.
No query image, dataset object, GT, or evaluation label enters this construction.

Each episode contains x_s, T_sr(x_s), A_srf(x_s), T_sr(A_srf(x_s)), with the
existing K1 boundary-strength selection. The two families for the same (s,r)
share the same T_sr in ALL three variants, even if executed at different steps.
Sample synthetic generation and photometric transforms with separate stable
keyed RNGs. Key ranking RNG by episode too; do not use Python's process-random
hash() or depend on how many draws the preceding episode consumed. Hard-mined
indices may differ as theta changes; that is an optimization outcome, not an
input-manifest mismatch. Do not assert identical hard samples across trajectories.

Before packing, pair support identities (a,b) for each repeat, without labels.
For the initial design use fixed manifest-order pairs (0,1),(2,3),...,(14,15)
for all repeats. Random pairing or pair rotation is NOT another enabled factor.
Every episode from either source uses the same frozen B_ab=B\{a,b}. This
assignment belongs to the episode and never changes with its batch companion.
Even a same-source batch excludes the designated other source. Hence each
episode has the SAME14-source teacher/scoring bank in M0/M1/M2. Query scoring
uses the ordinary full support bank, as in K1/L3C; training exclusion is not
carried into inference. No target-memory adaptation is introduced.

This bank rule avoids a hidden M1/M2 change from14 to15 reference sources.
Fixed pairing/keyed RNG/shared-shift episodes differ from historical L3C;
therefore historical L3C is not the isolated control for either component.

Store episode identities, RNG keys, bank-owner identities and compact input/
mask hashes, NOT128 cached images, dense maps, graphs or feature tensors.
Generate on demand; retain frozen features only within the current update.

## 3. One method with two incremental components

| Variant | family_balance | source_pairing | Two episodes per update |
|---|---|---|---|
| M0 | false | false | different sources, same family;16-step family blocks |
| M1 | true | false | different sources, opposite families |
| M2 (FBPA) | true | true | same source and shift, opposite families |

All three:128 identical episode identities,64 real AdamW updates,8 episodes
per source,4 episodes per source/family, same initialization/loss/learning rate,
same per-episode synthetic mask/strength/transform/reference bank. No new seed.
M0 is an exposure-matched control, NOT the original one-source K1.

Exact packing for N=16, r=0..3 and each fixed pair (a,b):

    M0:
      for h in0..1:
        for f in [CutPaste, Scratch]:
          for r in [2*h, 2*h+1]:
            for (a,b): update(e_a,r,f, e_b,r,f)
    M1:
      for r in0..3:
        for (a,b):
          update(e_a,r,CutPaste, e_b,r,Scratch)
          update(e_a,r,Scratch, e_b,r,CutPaste)
    M2:
      for r in0..3:
        for (a,b):
          update(e_a,r,CutPaste, e_a,r,Scratch)
          update(e_b,r,CutPaste, e_b,r,Scratch)

An integer-only schedule audit verified all three contain the identical128
episode multiset and64 updates, with the same exclusion mapping. M0/M1/M2 have
0/64/64 mixed-family updates and0/0/64 same-source updates. This is a design
consistency check, NOT a model test, evaluation or performance result.

## 4. Objective and update mechanism

Keep the K1 loss for episode e:

    ell_e(theta) = rank_e + shift_e + contrast_e + stopgrad(lambda_e)*P_e
    P_e = clean_score_MSE + 0.25*relational_score_loss
    lambda_e = clamp(P_e / budget_e, 0.1, 1.0)
    budget_e = 0.0012 / (1 + susceptibility_e)
    L_batch = (ell_e1(theta) + ell_e2(theta)) / 2

For M2, both episodes share clean/shifted-clean scores, teacher and budget.
The exact same objective can be evaluated as:

    L_M2 = (rank_C + rank_S)/2 + (contrast_C + contrast_S)/2
           + shift_clean + stopgrad(lambda)*P_clean

Do NOT count common shift/preservation twice after averaging. Do NOT enforce
equal anomaly scores across families: their genuine proxy difficulty differs.
Do NOT add meta-learning, confidence weights, gradient projection, restoration,
new anomaly types, harder shifts, a new calibration rule, or a new inference head.

Accumulate both episode gradients at the SAME theta, clip the combined gradient
once at5, and call the existing AdamW once (lr=.001, weight_decay=0). Two Adam
steps are NOT equivalent. Sequential gradient accumulation/VJP replay is valid;
avoid holding two full model graphs. Shared frozen clean features in M2 can be
reused within the update; all dense episode state is released afterwards.

At any fixed theta, the full-deck averaged objective is identical across
M0/M1/M2, provided each episode uses its keyed randomness. Only grouping and
finite-step optimization order differ. Therefore this is NOT a new population
loss or proof of learned invariance. The claim is about the update estimator
and trajectory under a small support/step budget.

M1 averages source variation as well as family variation. M2 conditions on a
single source; it may remove a source/family mixing ambiguity but increase
source-to-source update noise. There is no general variance-reduction theorem
or guarantee that M2 must beat M1. Gradient-norm CV alone cannot settle this.

## 5. Code/config boundary for a future implementation

Planned method file: src/fmad/methods/family_balanced_paired.py, using existing
K1 loss/scorer and normal-only fitting/query pipeline. Use existing registry,
CLI and scripts/experiments.py; no separate queue or cumulative-ledger helper.

Proposed configuration keys (not implemented; not an executable config):

    method: family_balanced_paired
    method_config:
      episode_enabled: true
      family_balance: true
      source_pairing: true
      episode_repeats: 4
      episode_seed: 0
      reference_policy: fixed_pair_excluded
      steps: 64

source_pairing requires family_balance. The child switch must be
episode_enabled, NEVER overwrite BoundarySynthetic.enabled. With
episode_enabled=false, delegate exactly to unchanged L3C (meta off), including
its existing scheduling/RNG/bank policy, rather than approximate its behavior.
That off path provides the same-batch L3C performance anchor.
The enabled path is bounded initially to16 supports/two families/four repeats;
reject inconsistent shots/steps instead of silently truncating the deck.

Trainable parameters remain100384; backbone frozen; no changed inference cost
by construction. M0/M1 have8 distinct four-view episode inputs per update;
M2 has6 unique inputs because clean views are shared. This can reduce frozen
feature extraction, NOT guarantee25% total time savings: selector, NN scoring,
backward, and evaluation costs remain. Log actual time/extract calls/peak memory.

Pre-implementation novelty position: a controlled episode-construction and
packing hypothesis, not a claimed new general algorithm. Balanced batching,
paired augmentation and ordinary ERM are not claimed novel. A brief primary-
source sanity check is required before any novelty claim or implementation
framed as novel; an exhaustive literature/pilot loop is not a prerequisite.

## 6. Future final-metric ablation, not a pilot ladder

Proposed sequence: same-batch L3C off-path anchor -> M0 -> M1 -> M2.
Each variant runs the full8-class evaluation, single seed0, same normal16
support content, frozen DINOv2 ViT-L14 layer23,448+672, no_mask_no_rotation,
streaming mvtec_ad2_float16_streaming_v1. TESTpublic remains seen development.
Primary full8 macro AUPRO@.05; secondary pAUROC@.05; oracle F1 diagnostic.

Proposed finite budget, NOT a current execution instruction:32 class runs,
at most3600s/class, no sweep or additional seed. When separately authorized,
use the existing local container/GPU0 for six classes and existing dsba3/GPU2
for vial/wallplugs, with current settings verified then. No hardware changes,
provisioning, remote full-log/map/checkpoint pull, or automatic retry expansion.

Interpretation and stop conditions:

- M1>M0 on primary: supports balanced packing/scheduling under matched episode
  exposure and bank conditions, NOT uniquely gradient-conflict reduction.
- M2>M1 on primary: supports conditional same-source pairing in this setting.
  M2<=M1: do not add/promote source pairing; M1 remains the simpler candidate.
- M1<=M0: the balance hypothesis lacks primary support under these conditions.
  Do not explain success using lower training loss or gradient-norm variability.
- Candidate>same-batch L3C is required for a new best-method development claim.
  M0/M1/M2 versus L3C is a declared method-level comparison, not isolated bank,
  randomization or exposure attribution. Historical K1 remains a reference.
- Report all8 class effects, particularly the prior fabric/wallplugs declines;
  do not select a winning subset after seeing results. Tiny gains remain
  tentative; no numeric gain prediction, significance, or sealed-final claim.
- If execution/provenance fails, mark invalid rather than a method-family loss.
  Stop after the bounded comparison; do not automatically add new mechanisms.

Essential implementation checks later: all-off L3C parity including weak
selection; identical episode/input/bank manifests across M0/M1/M2; correct
loss averaging and clean gradient accounting; two gradients before one Adam
step; frozen backbone; finite streaming coverage; no cross-step dense state.
Use normal targeted tests, not an extra model-evaluation pilot program.

Scalar logs should identify both episodes, sources, families, exclusion pair,
loss components, P/b/lambda, combined gradient norm, updates/episodes consumed,
fit time and output policy. Optional gradient-direction diagnostics must not
change the optimizer or become prerequisites for reporting final metrics.

## 7. Failure cases and bounded decision

Shared synthetic shortcuts may be reinforced by balanced batches. Opposite
family gradients can cancel useful real-defect directions. One-source updates
can overfit support appearance; fixed pair exclusion can remove a useful normal
mode. Neither exact same inputs nor synthetic loss reduction establishes real-
anomaly generalization. The marginal historical gain may also be seed-specific.

Proceeding candidate: M1 balanced two-source updates, then M2 same-source
paired updates as the one structural extension. No other component is bundled.
This document and linked status notes are the deliverables for the current
design-only request. No result row or runnable experiment is created.
