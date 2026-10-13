# SAEC-LoRA-S1 Stage -1 report

Date: 2026-08-27  
Protocol: `saec-lora-s1-shared-action-equivariant-commutation-20260827`  
Status: `COMPLETED / SUPPORT-ONLY / EXACT-REPEAT`  
Verdict: `NO_IDENTIFIABLE_ACTION`

## Outcome

The backbone LoRA trained stably and produced a real, but bounded, relational
signal.  The correct shared-action graph had lower held-G2 relation error than
the shuffled graph on all four objects, by `4.24%--33.47%`.  The effective
adapter norm was nonzero on every object (`2.708--4.121`), and the exact `can`
repeat reproduced every scientific metric with maximum absolute difference
`0.0` and the same combined model-state SHA-256.

That signal was not attributable to the synthetic-anomaly commutation term.
The matched no-commutation arm N2 had slightly lower relation error than M1 on
every object.  The four-object macro M1 gain over the strongest N1/N2 null was
`-0.00336924`, compared with the frozen `+0.05` gate, and the M1-minus-N2
commutation-cosine difference was only `-0.0000999` to `+0.0004255`, compared
with `+0.02`.  Action identification, joint success, synthetic contribution,
and continuation therefore passed `0/4`, `0/16`, `0/16`, and `0/4`.

The experiment answers the Stage -1 question negatively: attaching a single
block-11 `mlp.fc2` rank-4 LoRA to DINOv2 can fit the support action task, but
the present shared-action plus synthetic-commutation objective does not force
an identifiable held action relation or protect anomaly geometry.  This is not
a model-performance result and is not comparable with SuperAD.

## Frozen experiment

- Backbone: offline DINOv2 ViT-L/14.
- Adaptation: one zero-start rank-4, alpha-8 LoRA on block 11 `mlp.fc2`;
  blocks 0--10 were detached.
- Candidate: `M1_SAEC_SHARED_ACTION` with cross-base shared-action,
  mask-balanced synthetic commutation, and frozen-anchor losses.
- Controls: frozen B0, zero-init I0, shuffled action graph N1, and matched
  no-commutation N2.
- Fit: exact support-16 `train/good`, generators G0/G1/G4, four actions, one
  fixed 192-update schedule per trained arm.
- Held: four support-disjoint `train/good` bases per object and held generator
  G2; `T(A)` and `A(T)` were rendered and measured separately.
- Objects: `can`, `fabric`, `vial`, `walnuts`.
- Locked exclusions: TESTpublic/private, GT, query images, query-memory update,
  rank/LR/block/loss/seed/action-strength sweep.

## Result table

| object | M1 relation error | N1 shuffled | N2 no-comm | M1 gain vs strongest null | action pass | anomaly pass | worst M1 q99 reduction | minimum RN neighbor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| can | `0.145785` | `0.163505` | `0.145451` | `-0.229%` | `0/4` | `1/4` | `-1.62450` | `0.75391` |
| fabric | `0.189397` | `0.284659` | `0.188406` | `-0.526%` | `0/4` | `4/4` | `-0.01236` | `0.94043` |
| vial | `0.048271` | `0.052611` | `0.048043` | `-0.476%` | `0/4` | `0/4` | `-1.06384` | `0.66602` |
| walnuts | `0.556429` | `0.581091` | `0.555781` | `-0.117%` | `0/4` | `4/4` | `+0.00080` | `0.93945` |

Aggregate facts:

- primary macro relation gain: `-0.0033692394`;
- action-identified objects: `0/4`;
- anomaly-protected objects: `2/4` (`fabric`, `walnuts`);
- joint pass cells/objects: `0/16`, `0/4`;
- synthetic-contribution cells/objects: `0/16`, `0/4`;
- continuation objects: `0/4`;
- q99 no-harm: fail, driven by severe J1/J2 harm on `can` and J0_POS/J1/J2
  harm on `vial`;
- shortcut-free: fail on `4/4` objects;
- structural custody and repeatability: pass.

## Training behavior and positive signal

| object | M1 loss step 1 -> 192 | M1 change | N2 loss step 1 -> 192 | N2 change | M1 adapter norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| can | `1.5767 -> 1.3741` | `-12.85%` | `1.5247 -> 1.3344` | `-12.48%` | `3.9385` |
| fabric | `1.7604 -> 1.4492` | `-17.68%` | `1.7364 -> 1.4241` | `-17.98%` | `4.1207` |
| vial | `1.4377 -> 1.6540` | `+15.04%` | `1.4076 -> 1.5676` | `+11.37%` | `3.2268` |
| walnuts | `2.3607 -> 1.7397` | `-26.31%` | `2.2153 -> 1.6816` | `-24.09%` | `2.7082` |

Three objects show useful optimization progress, gradient norms remained
finite, and every adapter moved away from zero.  More importantly, M1 beat the
shuffled N1 relation error on every object: `10.84%` on `can`, `33.47%` on
`fabric`, `8.25%` on `vial`, and `4.24%` on `walnuts`.  The backbone therefore
contains trainable cross-base action structure; the negative verdict is not a
zero-gradient or failed-LoRA result.

N2 improved by essentially the same amount and beat M1 on every object.  This
isolates the positive signal to shared-action fitting rather than to the
synthetic commutation term.  Increasing rank, duration, LR, or commutation
weight would not answer that causal failure and is prohibited by the frozen
hard-stop rule.

## Failure analysis

### 1. The commutation treatment was causally inactive

No cell reached the preregistered M1-over-N2 commutation gain of `+0.02`.
Observed differences were about two orders of magnitude smaller, with maximum
absolute difference `0.0004255`.  M1 and N2 also converged to similar adapter
norms and loss trajectories.  The anomaly term therefore did not create a
distinct held representation.

### 2. Base identity remained an easier shortcut

The leave-base-out base-ID balanced accuracy was `0.3750`, `0.4844`, `0.5312`,
and `0.5000` on `can/fabric/vial/walnuts`, all above the `0.35` ceiling.  The
generator-ID and mask-geometry probes stayed below their ceilings, so the
shortcut failure is specifically base identity, not generator or mask shape.
This explains why a low relation loss did not become an identifiable
base-independent action representation.

Action balanced accuracy itself reached the ceiling even for B0 and the hard
nulls, so it was non-diagnostic and could not assign causal credit to M1.

### 3. Anomaly/RN behavior was object- and action-dependent

`fabric` and `walnuts` preserved all four anomaly cells, but `can` and `vial`
did not.  Gaussian/spatial actions caused the largest normal-tail and neighbor
damage: `can/J1` q99 reduction was `-1.62450`, `can/J2` was `-0.83674`, and
`vial/J1` was `-1.06384`.  This rules out deploying the adapted stream even
though two objects looked safe.

## Remote execution, repair, and reproducibility

- Completed run:
  `saec_lora_stage_minus1_dsba3_20260827T1549KST_v2`.
- Fixed project container: `hun_fsad_tta_012` on dsba3 GPUs 0--3.
- GPU/object panel: GPU0 `can`, GPU1 `fabric`, GPU2 `vial`, GPU3 `walnuts`.
- Immutable snapshot SHA-256:
  `020c35b4b3a456b39e7761e5f3652d48b056019585395682fc351b8604c7911b`.
- Host/container GPU-map SHA-256:
  `a27a8597ba2d435fc05d059d2f9f28bdb323881bda68ad9bd9574c145492595d`.
- Source-manifest SHA-256:
  `30bf70d9b93bcbb2386c12c3ce91176a2a1f3087f53f1eba6709e252c42d1bbe`.
- Aggregate SHA-256:
  `f85b2d36d9d7d127c8aac0ee569234ad65385cd3d2fd35f5229b13ff526011c6`.
- Repeat: maximum metric difference `0.0`; model-state SHA-256
  `79a675dd7847dfba45db4332be492085d1c0b979aa018702ac3a47cab344cd8a`.
- Preflight: `18/18` focused tests, offline model load, actual CUDA LoRA
  gradient/identity smoke, flat-mask-to-2D geometry smoke, source/custody and
  GPU UUID checks all passed.
- Independent numerical reaggregation was byte-identical to the remote
  aggregate and produced the same aggregate SHA-256.
- All forbidden-access ledgers are empty.  `support_only=true`,
  `benchmark_comparable=false`, and SuperAD fields remain `N/A`.
- No `.npy`, `.npz`, `.pt`, `.pth`, `.pkl`, or `anomaly_maps/` artifact was
  retained.  GPU0--3 returned to `1 MiB / 0%` after completion.
- After the compact pull and audit, the exact v1 and v2 immutable code
  snapshots were removed; the compact remote result directory was preserved.

The preceding v1 run
`saec_lora_stage_minus1_dsba3_20260827T1506KST_v1` completed the three `can`
training arms, then failed before the first held metric because the mask
shortcut probe received CIDER's flat 256-cell mask instead of a `(16,16)`
geometry.  The v2 repair only reshaped that descriptor at the probe boundary
and added a model-smoke regression check.  v1 remains an implementation
incident with no scientific result.

## Stop and next boundary

Close this SAEC shared-action/commutation relation.  Do not sweep its rank,
block, LR, seed, duration, loss weight, threshold, or action strength.  The
surviving architectural insight is narrower: normal action structure is
trainable, but base identity and the adapted anomaly residual must not share
one unconstrained stream.

Any continuation must be a separately named architecture, not a retuned SAEC
arm.  The next exploration should test a genuinely separated path: keep the
frozen anomaly/RN evidence as an untouched branch, learn a sparse or gated
normal-action adapter in another branch, and require the gate itself to beat a
no-adapter/static-gate control under held bases.  That hypothesis must be
preregistered before another result-bearing GPU run; no successor experiment
was launched from this failed gate.
