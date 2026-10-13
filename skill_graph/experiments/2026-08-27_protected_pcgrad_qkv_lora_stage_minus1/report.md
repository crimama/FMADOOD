# Protected Asymmetric-PCGrad QKV-LoRA Stage -1 report

Date: 2026-08-27  
Protocol: `protected-pcgrad-qkv-lora-stage-minus1-20260827`  
Status: `COMPLETED / SUPPORT-ONLY FOUR-GPU PANEL`  
Verdict: `INSUFFICIENT_PROTECTED_PCGRAD_SIGNAL / END_SYNTHETIC_QKV_LORA_BRANCH`

## Outcome

Protected PCGrad was implemented correctly, but it did not become active on
the correctly paired synthetic-shift objective and did not improve the shared
QKV-LoRA candidate.  P2 had zero gradient-conflict steps for every object, so
its update rule collapsed to the D2 scalar-sum baseline.  Strict synthetic and
PCGrad-specific contributions were both `0/12`.

The global candidate correction cosine was `0.3664246`, correction norm ratio
was `0.3416666`, and candidate-minus-best-hard-null margin was `-0.1441973`.
This was worse than the Oracle-Family diagnostic (`-0.0832106`) and effectively
returned to the TARCD-S3 failure level.  Shifted-normal q99 reduction remained
positive (`+0.0187851`), but it did not translate into correction superiority.
Only `7/12` cells passed the anomaly guard.

The negative control supplied the useful positive mechanism signal.  Shuffling
the protected pairing created `88/94/52/85` conflict steps for
`can/fabric/vial/walnuts`, and every conflict step retained a nonnegative
post-PCGrad AdamW protection direction.  Thus the implementation and algebra
worked; the hypothesized conflict simply was not present under the real paired
training objective.

Per the preregistered stop rule, this closes gradient-combination-only shared
QKV-LoRA and the broader synthetic QKV-LoRA continuation.  No optimizer-state
projection, second seed, rank, block, alpha, learning-rate, loss-weight,
threshold, generator, mask, checkpoint, or routing sweep is licensed.

## Remote execution and custody

- Immutable run:
  `protected_pcgrad_qkv_lora_stage_minus1_dsba3_20260827Tpcgrad01Z_v2`.
- Host/container: dsba3, fixed project container `hun_fsad_tta_012`.
- GPU map: `can/fabric/vial/walnuts -> 0/1/2/3`.
- Remote preflight passed 15 focused tests, offline DINOv2 loading, and an
  actual CUDA fused-QKV LoRA model/gradient smoke.
- Support manifest SHA-256:
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Immutable snapshot SHA-256:
  `2c64df222479aa1e931b5068fbdcbadbc5fa413f65444d76c5cbf4f3498030c9`.
- Aggregate SHA-256:
  `9cce4ae313da3f9c75b5286247193b8c41720326ff3d1f5170fc763a0f221187`.
- All four object summaries, `12/12` cells, exact 11 arms, exact six trained
  arms, 144 updates, and traces at `1/36/72/108/144` were present.
- `support_only=true`, `benchmark_comparable=false`, and the public/private/
  test/GT/query-memory paths remained sealed.  No dense maps were retained.
- GPUs 0--3 returned idle after aggregation.  The compact result remains; only
  this run's own immutable source snapshot was removed.
- Only the fixed project container and this exact run were inspected or
  modified.  No other user's container, process, file, volume, or job was
  queried or touched.

The first `v1` attempt stopped before model execution because the remote
wrapper pre-created the runner's fail-closed output root.  The wrapper was
repaired and a new immutable `v2` run was launched; no partial `v1` scientific
result was used.

## Frozen design

- Frozen DINOv2 ViT-L/14 with rank-4, alpha-8 fused-QKV LoRA in blocks 8--11.
- One shared adapter per object, no family router or second score stream.
- `L_N = correction_clean + 0.1 identity_clean`.
- `L_P = correction_anomaly + 0.5 recomposition + 0.1 identity_anomaly +
  0.1 RN_guard`.
- On `dot(g_N,g_P)<0`, only `g_N` was projected away from `g_P`; otherwise the
  optimizer received the unchanged `g_N+g_P`.
- Global clip `1.0`, AdamW `1e-4`, one seed `20260826`, exact 144-row schedule.
- Held evaluation used support-disjoint normal bases and unseen
  `G2_structural_scratch`; no checkpoint selection or early stopping.

## Aggregate result

| measure | observed | progressive requirement | result |
|---|---:|---:|---|
| structural/custody | true | true | pass |
| P2 conflict steps | `0/0/0/0` | `>=8` per object | fail |
| global hard-null margin | `-0.1441973` | `>=-0.094` | fail |
| global q99 reduction | `+0.0187851` | `>=0` | pass |
| anomaly-guard cells | `7/12` | `>=9/12` | fail |
| synthetic contribution | `0/12` | diagnostic | fail |
| PCGrad contribution | `0/12` | `>=3/12` | fail |
| strict/progressive gate | false / false | true | fail |

Global medians were anomaly cosine `0.8599742` and anomaly norm ratio
`0.9838359`.  The apparently acceptable global anomaly median hides the same
object split seen earlier: `fabric/walnuts` retained anomalies, while
`can/vial` remained below the desired protection level.

### Object medians

| object | correction cosine | correction norm | hard-null margin | q99 reduction | anomaly cosine | anomaly guards |
|---|---:|---:|---:|---:|---:|---:|
| can | `0.2894313` | `0.3053514` | `-0.1476741` | `+0.0312476` | `0.7713786` | `0/3` |
| fabric | `0.5005563` | `0.3779818` | `-0.1407204` | `+0.0063226` | `0.9485699` | `3/3` |
| vial | `0.4434178` | `0.4338156` | `-0.2052550` | `+0.1222175` | `0.7669909` | `1/3` |
| walnuts | `0.2510227` | `0.2818776` | `-0.0883632` | `+0.0020594` | `0.9502899` | `3/3` |

## Mechanism analysis

1. The paired objective did not contain the assumed gradient conflict.  Every
   P2 object had zero conflicts, strong positive raw protected alignment, and
   therefore no PCGrad projection.
2. The shuffled control generated many conflicts and passed the actual-update
   protection check.  This separates a valid implementation from an inactive
   treatment and shows that correct pairing itself aligns the two losses.
3. P2 and D2 consequently learned nearly the same representation and achieved
   zero PCGrad-specific held contribution.  Projecting the post-AdamW delta
   would only address two transient trace violations, not the dominant held
   failure, because no sustained P2 gradient conflict or contribution existed.
4. Positive q99 reduction coexisted with a strongly negative hard-null margin.
   The adapter can suppress shifted-normal tails, but the learned correction
   is still weaker than a static support-derived correction and has too little
   magnitude on `can/walnuts`.
5. Oracle family separation had improved the margin, but shared protected
   optimization lost that gain.  Together, the sequence rules out inactive
   LoRA, family averaging alone, scalar loss conditioning, and gradient
   conflict as the primary explanations.

The remaining problem is representation identifiability: paired synthetic
shift/anomaly targets can make training losses mutually agreeable without
making the learned correction transfer beyond the synthetic generator and
beat support-derived hard nulls.

## Continuation boundary

Do not continue QKV-LoRA tuning or add an update-level optimizer projection.
Any next experiment must be a separately named representation architecture,
not another optimizer, loss-weight, adapter-location, routing, retrieval-key,
or score-bypass variant.  It must keep frozen anomaly evidence causally
separate from learned nuisance canonicalization, retain the same held G2 and
hard-null controls, and fail closed before any public/test benchmark branch.

No benchmark or public performance claim is licensed by this result.
