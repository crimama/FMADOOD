# SAEC-LoRA-S1 Stage -1 preregistration

Date: 2026-08-27  
Protocol: `saec-lora-s1-shared-action-equivariant-commutation-20260827`  
Status: `PREREGISTERED / RESULT-FREE`  
Claim tier: support-only mechanism diagnostic; not benchmark comparable

## Negative evidence intake and sole change

The preceding QKV-LoRA, family-routing, PCGrad, projector, CRFR, and explicit
pixel-inverse branches are closed.  SAEC changes one mechanism: the trainable
backbone displacement must encode a relation shared by different normal bases
under the same assumed action.  It does not learn a correction decoder, route
an adapter from a family code, project away a nuisance subspace, fuse scores,
or use a paired-clean correction target.

This is a new failure-basin test because CRFR could ignore its optional code,
whereas SAEC is evaluated directly on the backbone displacement relation.  A
correct relation must beat a base-dependent shuffled action graph.  Synthetic
anomalies enter only through an action/anomaly commutation term, whose value is
tested against a matched no-commutation arm on held generator G2.

## Data and custody

- Dataset: MVTec AD2 single-image at
  `/home/hunim/Volume/DATA/mvtec_ad_2` on the fixed dsba3 project container.
- Objects: `can`, `fabric`, `vial`, `walnuts`.
- Per object: the immutable SuperAD-selected 16 `train/good` support paths for
  fitting and exact RN memory; exactly four support-disjoint `train/good`
  bases for evaluation.
- Fit generators: `G0_poisson_patch`, `G1_material_blotch`,
  `G4_frequency_local`; held generator: `G2_structural_scratch`.
- Actions: `J0_NEG=-0.1333 EV`, `J0_POS=+0.1333 EV`, deterministic
  Gaussian-RMS exposure control, and fixed spatial illumination `cos_1_1` at
  CV 0.15.
- TESTpublic, TESTprivate, test images, ground truth, query updates, query
  memory, and benchmark selection are forbidden.
- `T(A(x))` and `A(T(x))` use the same base, donor, mask, synthesis seed, and
  action specification, but are rendered independently and never assumed
  equal.
- Only compact JSON/log/hash evidence is retained.  Dense features, maps,
  images, and checkpoints are forbidden.

## Frozen architecture

- Frozen anchor `F0`: DINOv2 ViT-L/14 block-11 token readout, deterministically
  pooled to 16x16 and token-wise standardized/L2 normalized.
- Student `Ftheta`: one zero-initialized rank-4, alpha-8 LoRA on
  `model.blocks[11].mlp.fc2` only.  Prefix blocks 0--10 are detached; every
  non-LoRA parameter remains frozen.
- No learned projector or deployable action code is used.  The training-only
  four-way linear action classifier receives the mean and standard deviation
  of the token displacement and is discarded after the diagnostic.
- One object-specific adapter is trained per arm.  Rank, location, optimizer,
  step count, seed, and loss weights are not selectable after results.

## Schedule and arms

Each exhaustive arm schedule contains
`16 bases x 4 actions x 3 fit generators = 192` updates.  Base `i` is paired
with base `(i+1) mod 16`.  AdamW uses LR `1e-4`, weight decay `0`, bf16
forward, fp32 loss, gradient clip `1.0`, and seed `20260827`.

| Arm | Definition |
| --- | --- |
| `B0_FROZEN` | adapter disabled; frozen reference |
| `I0_ZERO_INIT` | zero-start adapter; must equal B0 within `1e-7` |
| `M1_SAEC_SHARED_ACTION` | correct shared-action graph + commutation + anchor |
| `N1_SHUFFLED_ACTION_GRAPH` | matched capacity/schedule, base-dependent cyclic action graph |
| `N2_NO_COMMUTATION` | correct shared-action graph and anchor, commutation weight zero |

For normalized tokens `z(v)`, define `d(v,a)=z(T_a(v))-z(v)`.

- `L_shared`: `1-cos(d(x_i,a),d(x_j,a))` on flattened token displacement,
  plus four-way action cross-entropy from the displacement mean/std.  N1 uses
  the frozen base-dependent cyclic graph
  `a'=(a+1+(i mod 3)) mod 4`: the second base is rendered with `a'`, and both
  paired displacements receive the correspondingly permuted label `a'`.
- `L_comm`: mask-balanced cosine plus smooth-L1 log-norm distance between
  `d(A_g(x),a)` and stop-gradient `d(x,a)` for both bases.
- `L_anchor`: cosine no-drift loss on unshifted clean and synthetic-anomaly
  endpoints against `F0`.
- M1/N1 loss: `1.0 L_shared + 0.5 L_comm + 0.1 L_anchor`.
- N2 loss: `1.0 L_shared + 0.1 L_anchor`.

No q99/correction objective, static mean, paired-clean correction target,
projector, router, expert, PCGrad, or score fusion is allowed.

## Held action-identifiability metrics

Primary descriptors are flattened/L2-normalized held-G2 backbone
displacements.  Per object record:

- mean cosine between different bases under the same action;
- q95 cosine between different actions and their margin;
- leave-one-base-out four-class action balanced accuracy;
- candidate relation error versus N1 and N2;
- adapter norm and displacement energy;
- generator-ID BA from anomaly-conditioned action displacement over
  G0/G1/G2/G4, grouped leave-one-base-out;
- base-ID BA grouped leave-one-generator-out;
- maximum absolute Spearman correlation of anomaly action-drift norm with mask
  area and boundary.

The sole primary diagnostic metric is higher-is-better held-G2 relational gain

`relation_gain = 1 - M1_relation_error / min(N1_relation_error,N2_relation_error)`,

where

`relation_error = (1-same_action_cosine) + max(0, wrong_action_q95 - same_action_cosine + 0.10)`.

Per object, relation error is first averaged over the four fixed actions, then
the displayed relation-gain equation is applied to the three arm means.  The
reported global primary metric is the unweighted macro mean of the four object
relation gains.  Action margin and action BA are secondary gates and cannot
replace a failed primary metric.

`ACTION_IDENTIFIED` requires at least three of four actions with same-action
cosine `>=0.70` and margin `>=0.10`, action BA `>=0.75`, M1 relation error
`<=0.90 * min(N1,N2)`, M1 action margin at least `+0.05` over B0 and N1,
nonzero adapter/displacement energy, generator-ID BA `<=0.35`, base-ID BA
`<=0.35`, and mask correlation `<=0.35`.

The diagnostic continuation floor is weaker but fixed: positive M1 margin over
B0 and N1, relation error `<=0.95 * min(N1,N2)`, action BA not more than 0.02
below B0, and no shortcut ceiling violation.

## Held anomaly and composition sentinels

For every held base/action, record separately:

- `r0 = Ftheta(A(x))-Ftheta(x)`;
- `rTA = Ftheta(T(A(x)))-Ftheta(T(x))`;
- `rAT = Ftheta(A(T(x)))-Ftheta(T(x))`;
- their frozen-anchor counterparts and the pixel difference between `T(A)`
  and `A(T)`.

The training tensor mapping is fixed.  For each base quadruplet ordered as
`[x, T(x), A(x), T(A(x))]`, `d_clean=z[1]-z[0]` and
`d_anomaly=z[3]-z[2]`.  `L_comm` compares these two `[P,D]` tensors using the
same generated `[P]` grid mask, with inside and outside patches each receiving
total mass 0.5.  The separately rendered `A(T(x))` is evaluation-only and
never enters optimization.  Cell metrics first aggregate the four independent
held bases within one object/action cell; object gates then aggregate exactly
four action cells without dropping clipped or failed rows.

`J0_NEG` is the sole reversible stratum.  It requires `cos(rTA,r0)>=0.90`,
norm ratio in `[0.80,1.25]`, both residuals at cosine `>=0.90` to the frozen
canonical residual, and RN rank/neighbor/scale `>=0.90`, `>=0.90`, and
`[0.80,1.25]`.

`J0_POS`, Gaussian, and spatial actions are no-erasure strata.  They require
candidate residual cosine `>=0.95` and norm ratio `[0.80,1.25]` against the
corresponding frozen shifted residual for both `T(A)` and `A(T)`, commutation
cosine degradation versus B0 no worse than `0.02`, anomaly norm reduction no
worse than 10%, and RN topology within the same bounds.  J0 saturation is
recorded and never used to exclude or reweight a row.

For all actions, shifted-normal exact-RN q99 reduction must be `>=-0.05`; all
maps and metrics must be finite/nonconstant.  M1 must improve held-G2
commutation cosine over N2 by at least `+0.02` in at least three of twelve
object/action cells to establish a synthetic contribution signal.

## Sequential gates

1. GPU0 `can` primary/repeat must pass custody, model-smoke, exact arm/step
   counts, I0 identity `<=1e-7`, scientific metric max difference `<=1e-7`,
   and identical combined LoRA/classifier state hash.  Expansion is based on
   validity only, never on scientific performance.
2. `PROMISING_SAEC_STAGE1`: valid `4/4`, action-identified `>=3/4`, anomaly
   protection `>=3/4`, joint pass `>=2/4`, synthetic contribution in `>=3/12`
   cells across at least two objects, no shortcut ceiling violation, and no
   object q99 harm below `-0.05`.
3. `ACTION_ONLY_SIGNAL`: action-identified `>=3/4` but anomaly protection
   `<3/4`; do not deploy the adapted stream or tune it.
4. `CONTINUE_ONE_DIAGNOSTIC`: the fixed continuation floor passes on at least
   two objects and at least one held-G2 anomaly cell improves over N2 without
   RN harm.  This authorizes one separately preregistered change only.
5. `NO_IDENTIFIABLE_ACTION`: action pass `<=1/4`, global M1 relation gain over
   N1/N2 `<0.05`, or shortcut ceilings fail on two or more objects.  Close this
   SAEC relation; no rank/LR/block/loss/seed/action-strength sweep.

## Baseline and metric alignment

This Stage -1 never opens TESTpublic and is therefore not comparable to
SuperAD.  Result fields are frozen as
`superad_seg_AUROC_0.05=N/A`, `method_seg_AUROC_0.05=N/A`,
`delta_vs_superad=N/A`, `comparable=false`.  A later benchmark is unauthorized
unless the mechanism gate passes, after which the exact SuperAD-16 reference
set, RN-FMLK, full TESTpublic, segmentation AUROC@0.05, F1, and coverage
contract must be preregistered.

## Cost, safety, and retained value

- Expected cost: one exact-repeat can sentinel plus three-object panel on
  dsba3 GPUs 0--3; approximately 4 object-runs x 3 trained arms x 192 updates.
- Use only host `hunim@147.47.39.144:2222`, fixed container
  `hun_fsad_tta_012`, and host GPUs 0--3 after per-GPU idle/UUID checks.
- Never enumerate, stop, alter, or clean other users' containers, processes,
  files, volumes, or jobs.
- Even on failure, the experiment leaves a direct measurement of whether
  action-conditioned backbone displacement is identifiable across bases and
  whether synthetic commutation changes held-generator anomaly geometry.

## Frozen source identity

Frozen after the Stage-2 conditional-keep review and before any result-bearing
remote launch. The exact manifest is `source_hashes.json`, SHA256
`627eda145278b028161e10423b503ac8d16b131427d434fad4d53988843b3bf5`.

- runner: `9028652a656d1a7e73ad6a6ee5c73bef8b7d8d67b46ca8cd2b8ad6e00f77f735`
- core: `95188b0435feed877a9c33505768bf787510368f1724fd4e20b728627b89c9e6`
- aggregator: `39317e5fe7c9eea1a87f40f428ecb18acf00f6e534d5f9fd9a417a5e29e7a546`
- remote controller: `59676ee038d1b747c5fe2476bca850b7b5bf6274017404f869b0ef71a9fdd25b`
- frozen S2 runner: `242c51b3953796fd8408bf0fc9af0a9f0d3123a364e9a49b9e8a94d6db156ea5`
- frozen S2 core: `77805354431ac9d56615a8babad976feae8b9e74fe30d975332a2642ef25fa43`
- frozen RN: `27d6bf919dd8731cf3199de4cab66176c7d7288ad465e296188856ea21855ded`
- frozen CIDER: `3ec73279cfe4b8c0ac16bf6032a51bbfbeb09d9301d5ee4bafd887f5acb5da2a`
- frozen shift bank: `38a546bd890b2f7fefd1753f872db3e6a46def5186c974895eb3270dbe125ff9`
- frozen spatial bank: `ddfa9dcb647392b692aba6c30a48da9bf9d10495e39bd0fe033d58be92667c8c`

### Pre-metric threshold clarification

Recorded after remote model-smoke and training step 1, but before the first
scientific checkpoint or held evaluation. For `J0_POS`, `J1`, and `J2`, the
two preregistered statements are conjunctive: corresponding-frozen residual
topology uses the general `[0.80,1.25]` scale bound, while “anomaly norm
reduction no worse than 10%” imposes the stricter residual norm lower bound
`0.90`. Therefore the executable gate is `[0.90,1.25]`; this clarification
cannot relax a future failure and does not alter source identity.

### v1 implementation incident and v2 non-scientific repair

Run `saec_lora_stage_minus1_dsba3_20260827T1506KST_v1` completed all three
`can` training arms through step 192, then failed before producing the first
held scientific metric. `CIDER.mask_grid()` returns the frozen 16x16 mask in
flat row-major form, while the preregistered mask-geometry shortcut probe
requires a two-dimensional mask. The runner passed the flat representation
directly to that probe. No held metric, sequential scientific gate, or panel
expansion was observed from v1.

The v2 repair restores only that known spatial shape with `reshape((16,16))`
at the shortcut-probe boundary and adds the same check to model-smoke. It does
not change pixels, mask membership, seeds, schedules, losses, arms, data,
metrics, thresholds, or continuation rules. Because executable source identity
changed, v2 uses a new run name and manifest SHA256
`30bf70d9b93bcbb2386c12c3ce91176a2a1f3087f53f1eba6709e252c42d1bbe`
with runner SHA256
`c535fb907e75b5bf9becb89a81585593a7aab586e5a3ef7d84fc3d65a5055c8c`;
v1 remains an invalid implementation incident rather than scientific evidence.
